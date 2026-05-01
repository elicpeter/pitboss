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

use crate::git::{Git, ShellGit};

use super::run_dir::RunPaths;

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
            std::fs::rename(&self.path, &dest).with_context(|| {
                format!("worktree: quarantining {:?} to {:?}", self.path, dest)
            })?;
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

/// Formatted error message used when `git merge --ff-only <session_branch>`
/// fails. The runner records a `SessionStatus::Error` with this text as the
/// session summary so `pitboss grind` users can grep for the contract
/// violation.
pub fn parallel_safe_violation_summary(prompt_name: &str) -> String {
    format!("parallel_safe contract violated by prompt {prompt_name}")
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

    #[test]
    fn parallel_safe_violation_summary_names_prompt() {
        assert_eq!(
            parallel_safe_violation_summary("fp-hunter"),
            "parallel_safe contract violated by prompt fp-hunter"
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
