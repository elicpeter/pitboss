//! Per-session git worktrees for parallel grind sessions.
//!
//! When a prompt declares `parallel_safe: true` and `max_parallel > 1`, the
//! runner dispatches each session into its own worktree under
//! `<run-root>/worktrees/session-NNNN/` so concurrent sessions can stage,
//! commit, and diff against a clean tree without colliding on the main
//! workspace's index.
//!
//! Each worktree carries an ephemeral branch
//! `pitboss/grind/<run-id>-session-NNNN` cut from the run branch (note the
//! hyphen rather than a slash before `session-NNNN` — see
//! [`session_branch_name`] for why git's ref store forces this). The agent
//! commits land on the ephemeral branch; on session completion the runner
//! fast-forwards the run branch to the ephemeral tip and deletes the branch.
//! A non-fast-forward at merge time means the prompt violated its
//! `parallel_safe` claim — the session record gets `Error` status with a
//! verbatim contract-violation summary.
//!
//! Sequential sessions never go through this module; they run in the main
//! workspace exactly the way phase 07 wired them up.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, warn};

use crate::git::{CommitId, Git, ShellGit};

use super::prompt::PromptDoc;
use super::run_dir::{RunPaths, SessionStatus};

/// Filename used for the per-session scratchpad copy inside each worktree.
const PER_SESSION_SCRATCHPAD: &str = "scratchpad.md";

/// One live worktree owned by a single in-flight parallel session.
///
/// Construction creates the worktree directory and the ephemeral branch in
/// one shot. The handle then lives for the duration of the session: the
/// runner reads `path()` to set the agent's working directory, reads
/// `scratchpad_path()` to seed and consume the per-session scratchpad view,
/// and uses `worktree_git()` to commit inside the worktree without touching
/// the main workspace.
///
/// Cleanup is **not** automatic on drop — the runner explicitly chooses
/// between [`SessionWorktree::cleanup`] (drop the directory and the ephemeral
/// branch) and [`SessionWorktree::quarantine`] (move the directory under
/// `worktrees/failed/session-NNNN/` for forensics) so failed sessions stay
/// inspectable and successful ones don't balloon the run dir.
pub struct SessionWorktree {
    seq: u32,
    path: PathBuf,
    branch: String,
    failed_root: PathBuf,
    scratchpad_path: PathBuf,
    /// Cached scratchpad seed so the merge step at session end can decide
    /// whether the agent actually mutated the per-session view.
    seed_scratchpad: String,
    worktree_git: ShellGit,
}

impl SessionWorktree {
    /// Create the worktree, the ephemeral branch, and seed the per-session
    /// scratchpad view from `scratchpad_seed`.
    ///
    /// `repo_git` is the workspace-rooted git handle — the same one the
    /// runner uses for the main workspace — and only its trait surface is
    /// touched here, so any [`Git`] impl that implements `add_worktree`
    /// participates.
    pub async fn create(
        repo_git: &dyn Git,
        run_paths: &RunPaths,
        run_id: &str,
        run_branch: &str,
        seq: u32,
        scratchpad_seed: &str,
    ) -> Result<Self> {
        let path = run_paths.worktrees.join(format!("session-{seq:04}"));
        let branch = session_branch_name(run_id, seq);
        let failed_root = run_paths.worktrees.join("failed");

        // The parent worktrees/ dir is created by RunDir::create, but a
        // resumed run that landed before phase 11 may not have it on disk.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("worktree: ensuring parent dir {:?}", parent))?;
        }

        repo_git
            .add_worktree(&path, &branch, run_branch)
            .await
            .with_context(|| {
                format!(
                    "worktree: creating session-{seq:04} branch {branch:?} at {:?}",
                    path
                )
            })?;

        let scratchpad_path = path.join(PER_SESSION_SCRATCHPAD);
        std::fs::write(&scratchpad_path, scratchpad_seed).with_context(|| {
            format!(
                "worktree: seeding scratchpad at {:?} ({} bytes)",
                scratchpad_path,
                scratchpad_seed.len()
            )
        })?;

        let worktree_git = ShellGit::new(path.clone());

        Ok(Self {
            seq,
            path,
            branch,
            failed_root,
            scratchpad_path,
            seed_scratchpad: scratchpad_seed.to_string(),
            worktree_git,
        })
    }

    /// 1-based session sequence.
    pub fn seq(&self) -> u32 {
        self.seq
    }

    /// Filesystem path of the worktree.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ephemeral branch name (`pitboss/grind/<run-id>-session-NNNN`; see
    /// [`session_branch_name`] for the hyphen-vs-slash rationale).
    pub fn branch(&self) -> &str {
        &self.branch
    }

    /// Per-session scratchpad path the agent should be told about via
    /// `PITBOSS_SCRATCHPAD`.
    pub fn scratchpad_path(&self) -> &Path {
        &self.scratchpad_path
    }

    /// Snapshot of the run-level scratchpad at session start, retained so the
    /// merge step can detect whether the agent actually wrote to its view.
    pub fn scratchpad_seed(&self) -> &str {
        &self.seed_scratchpad
    }

    /// Git handle scoped to the worktree path. Used by the parallel-session
    /// task to stage/commit on the ephemeral branch without touching the
    /// main workspace's index.
    pub fn worktree_git(&self) -> &ShellGit {
        &self.worktree_git
    }

    /// Drop the worktree and the ephemeral branch.
    ///
    /// Both operations are best-effort: a missing worktree directory or a
    /// branch that was never created (e.g., creation failed mid-flight)
    /// resolves cleanly so the caller doesn't have to special-case error
    /// paths.
    pub async fn cleanup(&self, repo_git: &dyn Git) -> Result<()> {
        repo_git
            .remove_worktree(&self.path)
            .await
            .with_context(|| format!("worktree: removing {:?}", self.path))?;
        repo_git
            .delete_branch(&self.branch)
            .await
            .with_context(|| format!("worktree: deleting branch {:?}", self.branch))?;
        Ok(())
    }

    /// Move the worktree directory under `worktrees/failed/session-NNNN/` and
    /// drop the ephemeral branch + git's worktree bookkeeping.
    ///
    /// Used when the session resolved as a non-`Ok` outcome so the failure
    /// state is preserved for triage. The directory move happens before
    /// `git worktree remove` is invoked so the bookkeeping cleanup doesn't
    /// also delete the forensics copy.
    pub async fn quarantine(&self, repo_git: &dyn Git) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.failed_root)
            .with_context(|| format!("worktree: creating {:?}", self.failed_root))?;
        let dest = self.failed_root.join(format!("session-{:04}", self.seq));
        // Drop a stale forensics copy from a prior identical seq if one is
        // somehow present so the rename below is unambiguous.
        if dest.exists() {
            std::fs::remove_dir_all(&dest)
                .with_context(|| format!("worktree: clearing stale {:?}", dest))?;
        }
        if self.path.exists() {
            std::fs::rename(&self.path, &dest)
                .with_context(|| format!("worktree: quarantining {:?} to {:?}", self.path, dest))?;
        }
        // The directory is gone from the worktree's recorded path; tell git to
        // forget it. `remove_worktree` accepts an already-missing path.
        let _ = repo_git.remove_worktree(&self.path).await;
        repo_git
            .delete_branch(&self.branch)
            .await
            .with_context(|| format!("worktree: deleting branch {:?}", self.branch))?;
        Ok(dest)
    }

    /// Run the parallel-session merge dance: sync the worktree to the
    /// run-branch tip, commit the agent's changes, fast-forward the run
    /// branch, then stash any leftover edits.
    ///
    /// Held under `run_branch_lock` for the entire dance so a sibling parallel
    /// session cannot interleave between sync and ff-merge. Returns a
    /// [`MergeOutcome`] the caller folds into the session record. The starting
    /// `status`/`summary` are passed in so a non-`Ok`/`Error` agent outcome
    /// (e.g., `Aborted` from a Ctrl-C, `Timeout` from a per-prompt cap) can
    /// short-circuit straight to teardown without touching git state.
    #[allow(clippy::too_many_arguments)]
    pub async fn merge_into<G: Git + ?Sized>(
        &self,
        repo_git: &G,
        run_branch: &str,
        run_branch_lock: &TokioMutex<()>,
        prompt: &PromptDoc,
        run_id: &str,
        starting_status: SessionStatus,
        starting_summary: String,
    ) -> Result<MergeOutcome> {
        let g = self.worktree_git();
        let _guard = run_branch_lock.lock().await;

        let mut status = starting_status;
        let mut summary = starting_summary;
        let mut commit: Option<CommitId> = None;
        let mut sync_ok = true;

        // Step 1 — sync the worktree's session branch to the current
        // run-branch tip. When the session was created run_branch was at
        // commit A; another parallel session may have advanced it to A'
        // since. Replaying that fast-forward inside the worktree is what
        // makes the eventual run-branch ff-merge possible. If the FF
        // refuses (because the agent's uncommitted edits would be
        // overwritten by the incoming run-branch tip), the prompt
        // violated its `parallel_safe: true` claim — we mark the session
        // Error and skip the commit / merge entirely.
        if status == SessionStatus::Ok || status == SessionStatus::Error {
            if let Err(e) = g.merge_ff_only(run_branch).await {
                warn!(
                    run_id = %run_id,
                    seq = self.seq,
                    error = %format!("{e:#}"),
                    prompt = %prompt.meta.name,
                    "grind: parallel_safe contract violation (worktree sync)"
                );
                status = SessionStatus::Error;
                summary = parallel_safe_violation_summary(
                    &prompt.meta.name,
                    ParallelSafeViolationSite::WorktreeSync,
                );
                sync_ok = false;
            }
        }

        // Step 2 — commit on top of the synced HEAD. The per-session
        // scratchpad lives at the worktree root and must stay out of
        // the run-branch tree; the runner merges it back into the
        // run-level scratchpad outside this method.
        let pitboss_rel = Path::new(".pitboss");
        let scratchpad_rel = Path::new(PER_SESSION_SCRATCHPAD);
        let parallel_exclusions: [&Path; 2] = [pitboss_rel, scratchpad_rel];
        if sync_ok {
            commit = match status {
                SessionStatus::Ok | SessionStatus::Error => {
                    try_commit_session(g, self.seq, prompt, run_id, &parallel_exclusions).await?
                }
                _ => None,
            };
        }

        // Step 3 — fast-forward the run branch to the session tip. The
        // sync above guarantees this is a strict descendant unless
        // run_branch raced forward between sync and merge — but the
        // run_branch_lock prevents that.
        if sync_ok && commit.is_some() {
            if let Err(e) = repo_git.merge_ff_only(self.branch()).await {
                warn!(
                    run_id = %run_id,
                    seq = self.seq,
                    error = %format!("{e:#}"),
                    prompt = %prompt.meta.name,
                    "grind: parallel_safe contract violation (run-branch ff)"
                );
                status = SessionStatus::Error;
                summary = parallel_safe_violation_summary(
                    &prompt.meta.name,
                    ParallelSafeViolationSite::RunBranchMerge,
                );
                commit = None;
            }
        }

        // Step 4 — stash any leftover edits the agent left behind in
        // the worktree so the directory is clean before teardown.
        // Skipped when the sync failed (we never advanced HEAD, so the
        // leftover is exactly what the agent wrote — quarantine will
        // keep it). Same exclusions as the commit step so the
        // per-session scratchpad survives the stash for the merge.
        if sync_ok {
            let stash_label = format!("grind/{}/session-{:04}-leftover", run_id, self.seq);
            match g.stash_push(&stash_label, &parallel_exclusions).await {
                Ok(true) => {
                    warn!(
                        run_id = %run_id,
                        seq = self.seq,
                        stash = %stash_label,
                        "grind: leftover changes stashed (parallel)"
                    );
                    if status == SessionStatus::Ok {
                        status = SessionStatus::Dirty;
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    warn!(
                        run_id = %run_id,
                        seq = self.seq,
                        error = %format!("{e:#}"),
                        "grind: stash_push failed (parallel) — treating as merge conflict"
                    );
                    if status == SessionStatus::Ok || status == SessionStatus::Dirty {
                        status = SessionStatus::Error;
                        summary = merge_conflict_summary(&prompt.meta.name, &e);
                    }
                }
            }
        }

        Ok(MergeOutcome {
            status,
            summary,
            commit,
            sync_ok,
        })
    }
}

/// Result of the parallel-session merge dance.
///
/// Returned by [`SessionWorktree::merge_into`] so the runner folds a single
/// value back into the session record instead of mutating four locals across
/// branching git steps.
#[derive(Debug, Clone)]
pub struct MergeOutcome {
    /// Final session status after sync → commit → ff-merge → stash.
    pub status: SessionStatus,
    /// Final summary text. Carries the `parallel_safe` violation label or the
    /// merge-conflict diagnostic when the dance promoted the status to
    /// `Error`; otherwise the caller's starting summary is returned verbatim.
    pub summary: String,
    /// Commit produced by the session (if any). `None` when the agent made no
    /// code changes, when the dance promoted the status to a terminal failure
    /// before the commit step, or when the run-branch ff-merge refused.
    pub commit: Option<CommitId>,
    /// `true` when the worktree-sync step (step 1) succeeded — i.e., the
    /// run-branch tip was successfully replayed into the session worktree.
    /// `false` short-circuits steps 2-4 and tells the runner to quarantine
    /// the worktree without further git churn.
    pub sync_ok: bool,
}

/// Stage and commit any code changes a grind session produced. Returns the
/// new commit id, or `None` if there was nothing code-side to commit (e.g.,
/// the agent only edited `.pitboss/`).
///
/// `exclude` is the per-call exclusion set forwarded to
/// [`Git::stage_changes`]. Sequential sessions pass just `.pitboss/`;
/// parallel sessions also pass the per-session `scratchpad.md` so the
/// worktree-rooted scratchpad never lands in the run-branch tree (it lives
/// outside git's history; pitboss merges it back via [`merge_scratchpad_into_run`]).
pub(crate) async fn try_commit_session<G: Git + ?Sized>(
    git: &G,
    seq: u32,
    prompt: &PromptDoc,
    run_id: &str,
    exclude: &[&Path],
) -> Result<Option<CommitId>> {
    git.stage_changes(exclude)
        .await
        .with_context(|| format!("grind: staging session {seq} changes"))?;

    let has_staged = git
        .has_staged_changes()
        .await
        .with_context(|| format!("grind: checking staged changes for session {seq}"))?;
    if !has_staged {
        debug!(seq, prompt = %prompt.meta.name, "grind: no code changes to commit");
        return Ok(None);
    }

    let message = format!(
        "[pitboss/grind] {} session-{:04} ({})",
        prompt.meta.name, seq, run_id,
    );
    let id = git
        .commit(&message)
        .await
        .with_context(|| format!("grind: committing session {seq}"))?;
    Ok(Some(id))
}

/// Summary text recorded when a session ends with a working-tree state that
/// `git stash push` cannot capture (most commonly an unresolved merge / index
/// conflict the agent left behind). The next session's pre-flight reads the
/// status field on the prior record to know the tree it inherits had to be
/// rolled back. Kept short so it renders cleanly in `sessions.md`.
pub(crate) fn merge_conflict_summary(prompt_name: &str, err: &anyhow::Error) -> String {
    let one_line = format!("{err:#}")
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if one_line.is_empty() {
        format!("merge conflict at session end (prompt {prompt_name})")
    } else {
        format!("merge conflict at session end (prompt {prompt_name}): {one_line}")
    }
}

/// Sweep stale parallel-session worktrees left behind by an interrupted
/// run. Used by `pitboss grind --resume` to clean up before any new
/// session is dispatched.
///
/// Walks the immediate children of `<run_paths.worktrees>/` (the `failed/`
/// subdirectory is intentionally skipped — it holds quarantined forensics
/// copies that must survive a resume), parses the `session-NNNN` suffix
/// off each entry, and drops every entry whose seq is strictly greater
/// than `last_session_seq`. For each match we ask git to remove the
/// worktree (force-removal — the entry is by definition not in flight, so
/// any local edits would have been lost when the host process died), then
/// delete the matching ephemeral branch.
///
/// Errors during a single entry are logged and skipped so a partial sweep
/// still cleans up everything it can. Returns the number of entries that
/// were removed.
pub async fn sweep_stale_session_worktrees(
    repo_git: &dyn Git,
    run_paths: &RunPaths,
    run_id: &str,
    last_session_seq: u32,
) -> usize {
    let Ok(read_dir) = std::fs::read_dir(&run_paths.worktrees) else {
        // No worktrees/ dir → nothing to sweep. Sequential-only runs hit
        // this path; not an error.
        return 0;
    };
    let mut removed = 0usize;
    for entry in read_dir.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Skip the quarantine root entirely — those entries are forensic.
        if name == "failed" {
            continue;
        }
        let Some(seq_str) = name.strip_prefix("session-") else {
            continue;
        };
        let Ok(seq) = seq_str.parse::<u32>() else {
            continue;
        };
        if seq <= last_session_seq {
            continue;
        }
        let branch = session_branch_name(run_id, seq);
        if let Err(e) = repo_git.remove_worktree(&path).await {
            warn!(
                run_id = %run_id,
                seq,
                path = %path.display(),
                error = %format!("{e:#}"),
                "grind: resume sweep: remove_worktree failed"
            );
        }
        // Even if remove_worktree failed, drop the directory if it is
        // still on disk so a future resume doesn't trip over it. The
        // path may already be gone if `git worktree remove` succeeded
        // but the bookkeeping warned anyway.
        if path.exists() {
            if let Err(e) = std::fs::remove_dir_all(&path) {
                warn!(
                    run_id = %run_id,
                    seq,
                    path = %path.display(),
                    error = %format!("{e:#}"),
                    "grind: resume sweep: remove_dir_all failed"
                );
                continue;
            }
        }
        if let Err(e) = repo_git.delete_branch(&branch).await {
            warn!(
                run_id = %run_id,
                seq,
                branch = %branch,
                error = %format!("{e:#}"),
                "grind: resume sweep: delete_branch failed"
            );
        }
        removed += 1;
    }
    removed
}

/// Compose the per-session branch name. Stable so resume / forensics tooling
/// can pattern-match on it.
///
/// Note the `-session-NNNN` suffix instead of `/session-NNNN`: git's
/// filesystem-backed ref store refuses to create `pitboss/grind/<run-id>/x`
/// when `pitboss/grind/<run-id>` already exists as a branch ref (`<run-id>`
/// can't be both a file and a directory under `refs/heads/`). The hyphen
/// keeps the run-id prefix grep-able while side-stepping the collision.
pub fn session_branch_name(run_id: &str, seq: u32) -> String {
    format!("pitboss/grind/{run_id}-session-{seq:04}")
}

/// Which step of the merge dance produced a `parallel_safe` contract violation.
///
/// The runner enforces `parallel_safe: true` at two distinct sites:
///
/// - [`Self::WorktreeSync`]: bringing the current run-branch tip into the
///   session worktree (`merge_ff_only(&run_branch)`) refused, meaning the
///   agent's uncommitted edits would have been overwritten by another
///   parallel session's commit.
/// - [`Self::RunBranchMerge`]: fast-forwarding the run branch to the session
///   tip refused. The run-branch lock prevents this from racing in normal
///   operation, so it should only fire when an external writer advanced the
///   run branch out of band.
///
/// Both kinds get the same `SessionStatus::Error` outcome; the labeled summary
/// just lets users tell the two cases apart when triaging a failed run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelSafeViolationSite {
    /// `git merge --ff-only <run_branch>` inside the session worktree refused.
    WorktreeSync,
    /// `git merge --ff-only <session_branch>` against the run branch refused.
    RunBranchMerge,
}

impl ParallelSafeViolationSite {
    fn label(self) -> &'static str {
        match self {
            ParallelSafeViolationSite::WorktreeSync => "worktree sync",
            ParallelSafeViolationSite::RunBranchMerge => "run-branch merge",
        }
    }
}

/// Formatted error message used when one of the two `merge --ff-only` steps
/// in the parallel-session dance fails. The runner records a
/// `SessionStatus::Error` with this text as the session summary so
/// `pitboss grind` users can grep for the contract violation and tell the two
/// failure sites apart.
pub fn parallel_safe_violation_summary(
    prompt_name: &str,
    site: ParallelSafeViolationSite,
) -> String {
    format!(
        "parallel_safe contract violated by prompt {prompt_name} ({})",
        site.label()
    )
}

/// Merge the per-session scratchpad view back into the run-level scratchpad.
///
/// Three-way comparison:
///
/// - Session view == seed → no merge needed (the agent didn't touch it).
/// - Session view != seed AND run-level == seed → fast-merge: write the
///   session view as-is.
/// - Session view != seed AND run-level != seed → another session has
///   already merged a change. Append the session view under labeled
///   `<!-- pitboss:session-NNNN -->` markers rather than attempting a
///   3-way text merge that would silently drop content.
pub fn merge_scratchpad_into_run(
    run_scratchpad_path: &Path,
    session_view: &str,
    seed: &str,
    seq: u32,
) -> Result<()> {
    if session_view == seed {
        return Ok(());
    }
    let current_run = std::fs::read_to_string(run_scratchpad_path).unwrap_or_default();
    let new_content = if current_run == seed {
        session_view.to_string()
    } else {
        let mut buf = current_run;
        if !buf.is_empty() && !buf.ends_with('\n') {
            buf.push('\n');
        }
        buf.push_str(&format!("<!-- pitboss:session-{seq:04} -->\n"));
        if !session_view.is_empty() {
            buf.push_str(session_view);
            if !session_view.ends_with('\n') {
                buf.push('\n');
            }
        }
        buf.push_str(&format!("<!-- /pitboss:session-{seq:04} -->\n"));
        buf
    };
    std::fs::write(run_scratchpad_path, new_content)
        .with_context(|| format!("scratchpad merge: writing {:?}", run_scratchpad_path))
        .map_err(|e| anyhow!(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_branch_name_uses_canonical_pattern() {
        assert_eq!(
            session_branch_name("20260430T120000Z-aaaa", 7),
            "pitboss/grind/20260430T120000Z-aaaa-session-0007"
        );
    }

    #[tokio::test]
    async fn sweep_removes_only_seqs_above_last_session_seq() {
        // Build a fake run dir with five session-NNNN worktrees plus the
        // forensic `failed/` and an unrelated entry. Sweep with
        // last_session_seq = 2: entries 1 and 2 stay, 3/4/5 leave, and
        // forensic + unrelated are untouched.
        use crate::git::{MockGit, MockOp};
        let dir = tempfile::tempdir().unwrap();
        let run_root = dir.path().join("run-root");
        std::fs::create_dir_all(&run_root).unwrap();
        let paths = RunPaths {
            root: run_root.clone(),
            sessions_jsonl: run_root.join("sessions.jsonl"),
            sessions_md: run_root.join("sessions.md"),
            scratchpad: run_root.join("scratchpad.md"),
            transcripts: run_root.join("transcripts"),
            worktrees: run_root.join("worktrees"),
            state: run_root.join("state.json"),
        };
        std::fs::create_dir_all(&paths.worktrees).unwrap();
        for seq in 1..=5u32 {
            let p = paths.worktrees.join(format!("session-{seq:04}"));
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("marker"), format!("{seq}")).unwrap();
        }
        // Forensic copies live under failed/ and must survive the sweep.
        let failed = paths.worktrees.join("failed").join("session-0009");
        std::fs::create_dir_all(&failed).unwrap();
        std::fs::write(failed.join("marker"), b"forensic").unwrap();
        // Unrelated entries that don't match the session-NNNN pattern.
        std::fs::create_dir_all(paths.worktrees.join("scratch")).unwrap();
        std::fs::write(paths.worktrees.join("loose-file"), b"x").unwrap();

        let git = MockGit::new();
        let removed = sweep_stale_session_worktrees(&git, &paths, "rid", 2).await;
        assert_eq!(removed, 3, "expected to remove session-0003..0005");

        // 1, 2 still present.
        for seq in 1..=2u32 {
            assert!(
                paths.worktrees.join(format!("session-{seq:04}")).exists(),
                "session-{seq:04} should still exist"
            );
        }
        // 3, 4, 5 gone.
        for seq in 3..=5u32 {
            assert!(
                !paths.worktrees.join(format!("session-{seq:04}")).exists(),
                "session-{seq:04} should have been swept"
            );
        }
        // Forensics + non-pattern entries untouched.
        assert!(failed.exists(), "failed/ entry should survive sweep");
        assert!(paths.worktrees.join("scratch").exists());
        assert!(paths.worktrees.join("loose-file").exists());

        // Each removal hit both `remove_worktree` and `delete_branch`.
        let ops = git.ops();
        let removes: Vec<&PathBuf> = ops
            .iter()
            .filter_map(|op| match op {
                MockOp::RemoveWorktree(p) => Some(p),
                _ => None,
            })
            .collect();
        let deletes: Vec<&String> = ops
            .iter()
            .filter_map(|op| match op {
                MockOp::DeleteBranch(b) => Some(b),
                _ => None,
            })
            .collect();
        assert_eq!(removes.len(), 3);
        assert_eq!(deletes.len(), 3);
        assert!(deletes
            .iter()
            .any(|b| *b == "pitboss/grind/rid-session-0003"));
        assert!(deletes
            .iter()
            .any(|b| *b == "pitboss/grind/rid-session-0004"));
        assert!(deletes
            .iter()
            .any(|b| *b == "pitboss/grind/rid-session-0005"));
    }

    #[tokio::test]
    async fn sweep_no_op_when_worktrees_dir_missing() {
        use crate::git::MockGit;
        let dir = tempfile::tempdir().unwrap();
        let run_root = dir.path().join("run-root");
        std::fs::create_dir_all(&run_root).unwrap();
        let paths = RunPaths {
            root: run_root.clone(),
            sessions_jsonl: run_root.join("sessions.jsonl"),
            sessions_md: run_root.join("sessions.md"),
            scratchpad: run_root.join("scratchpad.md"),
            transcripts: run_root.join("transcripts"),
            worktrees: run_root.join("worktrees"),
            state: run_root.join("state.json"),
        };
        // No worktrees/ dir created. Sequential-only runs hit this path
        // and the sweep must be a silent no-op.
        let git = MockGit::new();
        let removed = sweep_stale_session_worktrees(&git, &paths, "rid", 0).await;
        assert_eq!(removed, 0);
        assert!(git.ops().is_empty());
    }

    #[test]
    fn merge_conflict_summary_names_prompt_and_includes_first_error_line() {
        let err = anyhow!("stash failed\nstderr: CONFLICT (content)");
        let s = merge_conflict_summary("fp-hunter", &err);
        assert!(s.contains("fp-hunter"), "summary missing prompt name: {s}");
        assert!(s.contains("merge conflict"), "summary missing label: {s}");
        assert!(
            s.contains("stash failed"),
            "summary missing first err line: {s}"
        );
        assert!(
            !s.contains("CONFLICT"),
            "summary leaked tail of multi-line err: {s}"
        );
    }

    #[test]
    fn parallel_safe_violation_summary_names_prompt_and_site() {
        assert_eq!(
            parallel_safe_violation_summary("fp-hunter", ParallelSafeViolationSite::WorktreeSync),
            "parallel_safe contract violated by prompt fp-hunter (worktree sync)"
        );
        assert_eq!(
            parallel_safe_violation_summary("fp-hunter", ParallelSafeViolationSite::RunBranchMerge),
            "parallel_safe contract violated by prompt fp-hunter (run-branch merge)"
        );
    }

    #[test]
    fn merge_scratchpad_noop_when_session_did_not_touch_view() {
        let dir = tempfile::tempdir().unwrap();
        let run_pad = dir.path().join("scratchpad.md");
        std::fs::write(&run_pad, "run state\n").unwrap();
        let seed = "seed\n";
        merge_scratchpad_into_run(&run_pad, seed, seed, 1).unwrap();
        // Run-level scratchpad must remain whatever the runner wrote — the
        // session merge must not touch it when nothing changed.
        assert_eq!(std::fs::read_to_string(&run_pad).unwrap(), "run state\n");
    }

    #[test]
    fn merge_scratchpad_fast_merges_when_run_level_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let run_pad = dir.path().join("scratchpad.md");
        std::fs::write(&run_pad, "seed\n").unwrap();
        let seed = "seed\n";
        let session_view = "seed\n- session 1 added a line\n";
        merge_scratchpad_into_run(&run_pad, session_view, seed, 1).unwrap();
        assert_eq!(std::fs::read_to_string(&run_pad).unwrap(), session_view);
    }

    #[test]
    fn merge_scratchpad_appends_labeled_view_on_concurrent_modification() {
        let dir = tempfile::tempdir().unwrap();
        let run_pad = dir.path().join("scratchpad.md");
        // Another session already merged its update.
        std::fs::write(&run_pad, "seed\n- session 1 added a line\n").unwrap();
        let seed = "seed\n";
        let session_view = "seed\n- session 2 added a different line\n";
        merge_scratchpad_into_run(&run_pad, session_view, seed, 2).unwrap();
        let after = std::fs::read_to_string(&run_pad).unwrap();
        assert!(after.contains("- session 1 added a line"));
        assert!(after.contains("<!-- pitboss:session-0002 -->"));
        assert!(after.contains("- session 2 added a different line"));
        assert!(after.contains("<!-- /pitboss:session-0002 -->"));
    }
}
