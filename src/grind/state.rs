//! Persisted resume state for a `pitboss grind` run.
//!
//! A run's `state.json` (written to
//! `.pitboss/grind/runs/<run-id>/state.json`) caches everything
//! `pitboss grind --resume` needs to pick up where the original loop left
//! off:
//!
//! - the scheduler position (rotation count + per-prompt run counts),
//! - the budget tracker's cumulative spend,
//! - the run's branch and rotation name,
//! - the prompt-name list at run start, used as a "did the prompt set change"
//!   fingerprint, and
//! - the run's lifecycle status ([`RunStatus`]).
//!
//! `sessions.jsonl` remains the source of truth for the per-session record
//! stream; this file is a small derived cache so the resume path doesn't have
//! to re-aggregate every JSONL line on startup. Writes are atomic via
//! [`crate::util::write_atomic`] and happen after every session.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::util::paths::grind_runs_dir;
use crate::util::write_atomic;

use super::budget::BudgetSnapshot;
use super::plan::GrindPlan;
use super::prompt::PromptDoc;
use super::run_dir::{RunPaths, SessionRecord, SessionStatus, STATE_FILENAME};
use super::scheduler::{Scheduler, SchedulerState};

/// Lifecycle status persisted with [`RunState`]. The resume entry-point picks
/// the most-recent run whose status is [`RunStatus::Active`] (still running
/// when its host process died) or [`RunStatus::Aborted`] (Ctrl-C drained the
/// loop).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    /// The run is mid-execution, or its host process died before writing a
    /// terminal status. Resumable.
    Active,
    /// The scheduler exhausted naturally (or all configured caps held). Not
    /// resumable.
    Completed,
    /// The user aborted the run (`Ctrl-C` drain or explicit abort). Resumable.
    Aborted,
    /// A budget tripped or the consecutive-failure escape valve fired. Not
    /// resumable.
    Failed,
}

impl RunStatus {
    /// Whether `--resume` is allowed to pick this run up.
    pub fn is_resumable(self) -> bool {
        matches!(self, RunStatus::Active | RunStatus::Aborted)
    }
}

/// Source-of-truth resume snapshot for a single grind run. Written atomically
/// after every session and on every terminal exit; a missing or malformed
/// file refuses resume rather than silently producing stale results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    /// Run id this snapshot belongs to. Mirrors the directory name; carried
    /// inside the file so a misplaced state.json still self-identifies.
    pub run_id: String,
    /// Branch the runner committed onto. Re-checked-out on resume.
    pub branch: String,
    /// Plan name in effect for the run. Compared against the resolved plan
    /// on resume; a mismatch refuses to continue.
    pub plan_name: String,
    /// Sorted list of prompt names the original plan referenced. Used as a
    /// fingerprint so a removed (or renamed) prompt rejects the resume.
    pub prompt_names: Vec<String>,
    /// Scheduler state captured after the last recorded session.
    pub scheduler_state: SchedulerState,
    /// Budget tracker snapshot captured after the last recorded session.
    pub budget_consumed: BudgetSnapshot,
    /// Sequence number of the last session written to `sessions.jsonl`. The
    /// resumed runner dispatches `last_session_seq + 1` next.
    pub last_session_seq: u32,
    /// Wall-clock time the run was originally started.
    pub started_at: DateTime<Utc>,
    /// Wall-clock time of the most recent state write.
    pub last_updated_at: DateTime<Utc>,
    /// Lifecycle status. See [`RunStatus`].
    pub status: RunStatus,
}

impl RunState {
    /// Atomically write this state to `<run-root>/state.json`.
    pub fn write(&self, paths: &RunPaths) -> Result<()> {
        let mut bytes =
            serde_json::to_vec_pretty(self).context("grind state: serializing RunState")?;
        bytes.push(b'\n');
        write_atomic(&paths.state, &bytes)?;
        Ok(())
    }

    /// Read the state at `<run-root>/state.json`.
    pub fn read(paths: &RunPaths) -> Result<Self> {
        Self::read_path(&paths.state)
    }

    /// Read a state file from an explicit path. Errors carry the path so the
    /// CLI can surface a precise diagnostic.
    pub fn read_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("grind state: reading {}", path.display()))?;
        let parsed: RunState = serde_json::from_str(&raw)
            .with_context(|| format!("grind state: parsing {}", path.display()))?;
        Ok(parsed)
    }
}

/// One entry from [`list_runs`]. The `state` is the parsed
/// `state.json` contents; the `path` is `<root>/state.json` so errors that
/// happen later (resume failed, validation rejected) can point users at the
/// exact file.
#[derive(Debug, Clone)]
pub struct RunListing {
    /// Run id (directory name under `.pitboss/grind/runs/`).
    pub run_id: String,
    /// Path to the `state.json` file.
    pub state_path: PathBuf,
    /// Parsed state.
    pub state: RunState,
}

/// Walk `<repo>/.pitboss/grind/runs/` for runs that have a parseable
/// `state.json`, regardless of their lifecycle status. Errors from individual
/// run dirs are dropped so a single corrupt run doesn't hide the rest. The
/// result is sorted by `last_updated_at` descending (most recent first).
pub fn list_runs(repo_root: &Path) -> Vec<RunListing> {
    let grind_root = grind_runs_dir(repo_root);
    let entries = match fs::read_dir(&grind_root) {
        Ok(it) => it,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<RunListing> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let state_path = path.join(STATE_FILENAME);
        let Ok(state) = RunState::read_path(&state_path) else {
            continue;
        };
        let run_id = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        out.push(RunListing {
            run_id,
            state_path,
            state,
        });
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.state.last_updated_at));
    out
}

/// Find the most-recent run under `<repo>/.pitboss/grind/runs/` whose
/// persisted status is [`RunStatus::Active`] or [`RunStatus::Aborted`].
/// Returns `None` when no resumable run exists. The default target for
/// `--resume` with no argument.
pub fn most_recent_resumable(repo_root: &Path) -> Option<RunListing> {
    list_runs(repo_root)
        .into_iter()
        .find(|r| r.state.status.is_resumable())
}

/// Errors a resume validation can produce. Each variant maps onto a clear
/// CLI diagnostic; none of them leave the run in a partially resumed state.
#[derive(Debug, thiserror::Error)]
pub enum ResumeError {
    /// No resumable run exists under `<repo>/.pitboss/grind/runs/`.
    #[error("no resumable grind run found under {dir}")]
    NoResumableRun {
        /// `.pitboss/grind/runs/` path searched.
        dir: PathBuf,
    },
    /// The named run does not exist on disk.
    #[error("grind run {run_id:?} not found at {dir}")]
    RunNotFound {
        /// Requested run id.
        run_id: String,
        /// The expected `<repo>/.pitboss/grind/runs/<run-id>/` path.
        dir: PathBuf,
    },
    /// The run's `state.json` is missing or malformed.
    #[error("grind run {run_id:?}: failed to read state: {source:#}")]
    StateUnreadable {
        /// Requested run id.
        run_id: String,
        /// Underlying cause.
        #[source]
        source: anyhow::Error,
    },
    /// The run already exited terminally and cannot be resumed.
    #[error(
        "grind run {run_id:?} is {status:?} and cannot be resumed; start a new run with `pitboss grind`"
    )]
    NotResumable {
        /// Requested run id.
        run_id: String,
        /// Persisted status that disqualified the run.
        status: RunStatus,
    },
    /// The plan name in `config.toml`/CLI no longer matches the run's
    /// recorded plan.
    #[error(
        "grind run {run_id:?}: plan name changed (was {original:?}, now {current:?}); start a new run with `pitboss grind`"
    )]
    PlanRenamed {
        /// Requested run id.
        run_id: String,
        /// Original plan name persisted at run start.
        original: String,
        /// Plan name resolved on the resume invocation.
        current: String,
    },
    /// The plan's prompt-name list differs from the original.
    #[error(
        "grind run {run_id:?}: prompt set changed (added: {added:?}, removed: {removed:?}); start a new run with `pitboss grind`"
    )]
    PromptSetChanged {
        /// Requested run id.
        run_id: String,
        /// Prompts present now that were not at run start.
        added: Vec<String>,
        /// Prompts present at run start but not now.
        removed: Vec<String>,
    },
    /// `state.json`'s `last_session_seq` disagrees with the actual tail of
    /// `sessions.jsonl`. This happens when the host process died between the
    /// JSONL append and the `state.json` write — the scheduler / budget
    /// snapshot is then one session behind the source-of-truth log, and a
    /// blind resume would re-dispatch the missing session under a colliding
    /// seq. Refusing here keeps the run repairable by hand instead of
    /// silently doubling a session record.
    #[error(
        "grind run {run_id:?}: state.json out of sync with sessions.jsonl (state says \
         last_session_seq={state_seq}, log tail has {jsonl_seq}); start a new run or repair \
         state.json by hand"
    )]
    StateOutOfSync {
        /// Requested run id.
        run_id: String,
        /// `last_session_seq` from the cached `state.json`.
        state_seq: u32,
        /// Highest seq actually present in `sessions.jsonl` (`0` when empty).
        jsonl_seq: u32,
    },
}

/// Compare the persisted `prompt_names` against the current plan's prompt
/// list. Returns the (added, removed) split when they differ.
pub fn diff_prompt_names(
    original: &[String],
    current: &[String],
) -> Option<(Vec<String>, Vec<String>)> {
    let mut a: Vec<String> = original.to_vec();
    a.sort();
    a.dedup();
    let mut b: Vec<String> = current.to_vec();
    b.sort();
    b.dedup();
    if a == b {
        return None;
    }
    let added: Vec<String> = b.iter().filter(|n| !a.contains(n)).cloned().collect();
    let removed: Vec<String> = a.iter().filter(|n| !b.contains(n)).cloned().collect();
    Some((added, removed))
}

/// Resolve a `--resume [<run-id>]` invocation to a run on disk. When
/// `requested` is `Some`, the named run must exist; when `None`, picks the
/// most-recent run whose status is `Active` or `Aborted`.
pub fn resolve_target(
    repo_root: &Path,
    requested: Option<&str>,
) -> Result<RunListing, ResumeError> {
    let grind_root = grind_runs_dir(repo_root);
    match requested {
        Some(id) => {
            let dir = grind_root.join(id);
            if !dir.is_dir() {
                return Err(ResumeError::RunNotFound {
                    run_id: id.to_string(),
                    dir,
                });
            }
            let state_path = dir.join(STATE_FILENAME);
            let state =
                RunState::read_path(&state_path).map_err(|e| ResumeError::StateUnreadable {
                    run_id: id.to_string(),
                    source: e,
                })?;
            Ok(RunListing {
                run_id: id.to_string(),
                state_path,
                state,
            })
        }
        None => {
            most_recent_resumable(repo_root).ok_or(ResumeError::NoResumableRun { dir: grind_root })
        }
    }
}

/// Cross-check a resume target against the current plan. Returns the
/// validated [`RunListing`] when everything lines up, or a [`ResumeError`]
/// describing the first mismatch.
pub fn validate_resume(
    listing: RunListing,
    current_plan_name: &str,
    current_prompt_names: &[String],
) -> Result<RunListing, ResumeError> {
    if !listing.state.status.is_resumable() {
        return Err(ResumeError::NotResumable {
            run_id: listing.run_id,
            status: listing.state.status,
        });
    }
    if listing.state.plan_name != current_plan_name {
        return Err(ResumeError::PlanRenamed {
            run_id: listing.run_id,
            original: listing.state.plan_name,
            current: current_plan_name.to_string(),
        });
    }
    if let Some((added, removed)) =
        diff_prompt_names(&listing.state.prompt_names, current_prompt_names)
    {
        return Err(ResumeError::PromptSetChanged {
            run_id: listing.run_id,
            added,
            removed,
        });
    }
    Ok(listing)
}

/// Outcome of [`reconstruct_state_from_log`]. Carries the scheduler /
/// budget snapshot the resumed runner should use plus the highest seq
/// observed in `sessions.jsonl`. `records_replayed` is `0` when the cached
/// `state.json` was already aligned with the log; `> 0` means the host
/// process died between a JSONL append and the matching `state.json` write
/// and we recovered by replaying the missing records through the scheduler.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconciledState {
    /// Scheduler state to seed [`Scheduler::with_state`] with.
    pub scheduler_state: SchedulerState,
    /// Budget snapshot to seed [`crate::grind::BudgetTracker::from_snapshot`] with.
    pub budget_consumed: BudgetSnapshot,
    /// Highest seq actually present in `sessions.jsonl`. Used as the new
    /// `last_session_seq` so the resumed runner dispatches `last + 1` next.
    pub last_session_seq: u32,
    /// Number of session records the function replayed past the cached
    /// `state.json`. `0` for the perfectly-aligned case.
    pub records_replayed: usize,
}

/// Reconcile the cached `state.json` snapshot with the source-of-truth
/// `sessions.jsonl` tail.
///
/// `sessions.jsonl` is appended to before `state.json` is written, so a host
/// process that dies between the two produces a JSONL log that is one or more
/// records ahead of `state.last_session_seq`. The historical behavior here was
/// to refuse the resume in that case; now we replay the missing records
/// through the scheduler so a single dropped `state.json` write doesn't strand
/// the run. The reverse — `state.last_session_seq` claims more records than
/// `sessions.jsonl` actually has — is genuinely broken (the cached scheduler
/// state is ahead of the log) and still refuses with
/// [`ResumeError::StateOutOfSync`].
///
/// Replay is conservative: each missing record is matched against the
/// scheduler's next pick, and a divergence (different prompt name or scheduler
/// exhaustion) refuses rather than silently producing a state that doesn't
/// match what the original loop dispatched. In practice the scheduler is
/// deterministic over `(plan, prompts, state)` so replay only diverges when
/// a user has changed prompt frontmatter (weight / every / max_runs) between
/// the original run and the resume.
pub fn reconstruct_state_from_log(
    state: &RunState,
    log_records: &[SessionRecord],
    plan: &GrindPlan,
    prompts: &BTreeMap<String, PromptDoc>,
) -> Result<ReconciledState, ResumeError> {
    let jsonl_seq = log_records.iter().map(|r| r.seq).max().unwrap_or(0);
    if jsonl_seq == state.last_session_seq {
        return Ok(ReconciledState {
            scheduler_state: state.scheduler_state.clone(),
            budget_consumed: state.budget_consumed,
            last_session_seq: state.last_session_seq,
            records_replayed: 0,
        });
    }
    if jsonl_seq < state.last_session_seq {
        return Err(ResumeError::StateOutOfSync {
            run_id: state.run_id.clone(),
            state_seq: state.last_session_seq,
            jsonl_seq,
        });
    }

    let mut missing: Vec<&SessionRecord> = log_records
        .iter()
        .filter(|r| r.seq > state.last_session_seq)
        .collect();
    missing.sort_by_key(|r| r.seq);

    let mut sched =
        Scheduler::with_state(plan.clone(), prompts.clone(), state.scheduler_state.clone());
    let mut budget = state.budget_consumed;

    for rec in &missing {
        let picked = sched.next();
        match picked {
            Some(p) if p.meta.name == rec.prompt => {
                sched.record_run(&p.meta.name);
            }
            _ => {
                return Err(ResumeError::StateOutOfSync {
                    run_id: state.run_id.clone(),
                    state_seq: state.last_session_seq,
                    jsonl_seq,
                });
            }
        }

        budget.iterations = budget.iterations.saturating_add(1);
        budget.tokens_input = budget.tokens_input.saturating_add(rec.tokens.input);
        budget.tokens_output = budget.tokens_output.saturating_add(rec.tokens.output);
        budget.cost_usd += rec.cost_usd;
        match rec.status {
            SessionStatus::Ok | SessionStatus::Dirty => {
                budget.consecutive_failures = 0;
            }
            SessionStatus::Error | SessionStatus::Timeout => {
                budget.consecutive_failures = budget.consecutive_failures.saturating_add(1);
            }
            SessionStatus::Aborted => {}
        }
    }

    Ok(ReconciledState {
        scheduler_state: sched.state().clone(),
        budget_consumed: budget,
        last_session_seq: jsonl_seq,
        records_replayed: missing.len(),
    })
}

/// Helper used by both initial-write and per-session-write paths. Builds a
/// fresh [`RunState`] from the parts the runner has on hand and stamps
/// `last_updated_at` to [`Utc::now`].
#[allow(clippy::too_many_arguments)]
pub fn build_state(
    run_id: String,
    branch: String,
    plan_name: String,
    prompt_names: Vec<String>,
    scheduler_state: SchedulerState,
    budget_consumed: BudgetSnapshot,
    last_session_seq: u32,
    started_at: DateTime<Utc>,
    status: RunStatus,
) -> RunState {
    RunState {
        run_id,
        branch,
        plan_name,
        prompt_names,
        scheduler_state,
        budget_consumed,
        last_session_seq,
        started_at,
        last_updated_at: Utc::now(),
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grind::run_dir::RunPaths;
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    fn fixture_state(run_id: &str, status: RunStatus) -> RunState {
        let mut runs = BTreeMap::new();
        runs.insert("alpha".to_string(), 2u32);
        runs.insert("bravo".to_string(), 1u32);
        RunState {
            run_id: run_id.to_string(),
            branch: format!("pitboss/grind/{run_id}"),
            plan_name: "default".into(),
            prompt_names: vec!["alpha".into(), "bravo".into()],
            scheduler_state: SchedulerState {
                rotation: 3,
                runs_per_prompt: runs,
            },
            budget_consumed: BudgetSnapshot {
                iterations: 3,
                tokens_input: 1500,
                tokens_output: 750,
                cost_usd: 0.045,
                consecutive_failures: 0,
            },
            last_session_seq: 3,
            started_at: "2026-04-30T17:00:00Z".parse().unwrap(),
            last_updated_at: "2026-04-30T17:30:00Z".parse().unwrap(),
            status,
        }
    }

    #[test]
    fn round_trips_through_disk() {
        let repo = tempdir().unwrap();
        let run_id = "20260430T180000Z-rt00";
        let paths = RunPaths::for_run(repo.path(), run_id);
        fs::create_dir_all(&paths.root).unwrap();
        let state = fixture_state(run_id, RunStatus::Active);
        state.write(&paths).unwrap();
        let back = RunState::read(&paths).unwrap();
        assert_eq!(back, state);
    }

    #[test]
    fn malformed_state_is_rejected_with_path_in_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("state.json");
        fs::write(&path, "{not json").unwrap();
        let err = RunState::read_path(&path).unwrap_err();
        assert!(
            err.to_string().contains("state.json"),
            "expected error to surface path, got: {err}"
        );
    }

    #[test]
    fn list_runs_returns_runs_sorted_by_last_updated_desc() {
        let repo = tempdir().unwrap();
        for (run_id, ts) in [
            ("rid-old", "2026-04-30T10:00:00Z"),
            ("rid-new", "2026-04-30T18:00:00Z"),
            ("rid-mid", "2026-04-30T15:00:00Z"),
        ] {
            let paths = RunPaths::for_run(repo.path(), run_id);
            fs::create_dir_all(&paths.root).unwrap();
            let mut s = fixture_state(run_id, RunStatus::Active);
            s.last_updated_at = ts.parse().unwrap();
            s.write(&paths).unwrap();
        }
        let listings = list_runs(repo.path());
        let ids: Vec<&str> = listings.iter().map(|l| l.run_id.as_str()).collect();
        assert_eq!(ids, vec!["rid-new", "rid-mid", "rid-old"]);
    }

    #[test]
    fn most_recent_resumable_skips_terminal_runs() {
        let repo = tempdir().unwrap();
        for (run_id, ts, status) in [
            (
                "rid-completed",
                "2026-04-30T18:00:00Z",
                RunStatus::Completed,
            ),
            ("rid-aborted", "2026-04-30T17:00:00Z", RunStatus::Aborted),
            ("rid-active", "2026-04-30T16:00:00Z", RunStatus::Active),
        ] {
            let paths = RunPaths::for_run(repo.path(), run_id);
            fs::create_dir_all(&paths.root).unwrap();
            let mut s = fixture_state(run_id, status);
            s.last_updated_at = ts.parse().unwrap();
            s.write(&paths).unwrap();
        }
        // rid-completed is the freshest but Completed is terminal; resume
        // skips it and lands on rid-aborted.
        let pick = most_recent_resumable(repo.path()).unwrap();
        assert_eq!(pick.run_id, "rid-aborted");
    }

    #[test]
    fn most_recent_resumable_returns_none_when_grind_dir_is_missing() {
        let repo = tempdir().unwrap();
        assert!(most_recent_resumable(repo.path()).is_none());
    }

    #[test]
    fn resolve_target_explicit_run_not_found() {
        let repo = tempdir().unwrap();
        let err = resolve_target(repo.path(), Some("ghost")).unwrap_err();
        assert!(matches!(err, ResumeError::RunNotFound { .. }));
    }

    #[test]
    fn resolve_target_default_no_resumable() {
        let repo = tempdir().unwrap();
        let err = resolve_target(repo.path(), None).unwrap_err();
        assert!(matches!(err, ResumeError::NoResumableRun { .. }));
    }

    #[test]
    fn validate_resume_rejects_terminal_status() {
        let repo = tempdir().unwrap();
        let run_id = "rid";
        let paths = RunPaths::for_run(repo.path(), run_id);
        fs::create_dir_all(&paths.root).unwrap();
        let s = fixture_state(run_id, RunStatus::Completed);
        s.write(&paths).unwrap();
        let listing = resolve_target(repo.path(), Some(run_id)).unwrap();
        let err =
            validate_resume(listing, "default", &["alpha".into(), "bravo".into()]).unwrap_err();
        assert!(matches!(err, ResumeError::NotResumable { .. }));
    }

    #[test]
    fn validate_resume_detects_removed_prompt() {
        let repo = tempdir().unwrap();
        let run_id = "rid";
        let paths = RunPaths::for_run(repo.path(), run_id);
        fs::create_dir_all(&paths.root).unwrap();
        let s = fixture_state(run_id, RunStatus::Active);
        s.write(&paths).unwrap();
        let listing = resolve_target(repo.path(), Some(run_id)).unwrap();
        let err = validate_resume(listing, "default", &["alpha".into()]).unwrap_err();
        match err {
            ResumeError::PromptSetChanged { removed, added, .. } => {
                assert_eq!(removed, vec!["bravo".to_string()]);
                assert!(added.is_empty());
            }
            other => panic!("expected PromptSetChanged, got {other:?}"),
        }
    }

    #[test]
    fn validate_resume_detects_added_prompt() {
        let repo = tempdir().unwrap();
        let run_id = "rid";
        let paths = RunPaths::for_run(repo.path(), run_id);
        fs::create_dir_all(&paths.root).unwrap();
        let s = fixture_state(run_id, RunStatus::Active);
        s.write(&paths).unwrap();
        let listing = resolve_target(repo.path(), Some(run_id)).unwrap();
        let err = validate_resume(
            listing,
            "default",
            &["alpha".into(), "bravo".into(), "charlie".into()],
        )
        .unwrap_err();
        match err {
            ResumeError::PromptSetChanged { added, removed, .. } => {
                assert_eq!(added, vec!["charlie".to_string()]);
                assert!(removed.is_empty());
            }
            other => panic!("expected PromptSetChanged, got {other:?}"),
        }
    }

    #[test]
    fn validate_resume_accepts_unchanged_prompt_set() {
        let repo = tempdir().unwrap();
        let run_id = "rid";
        let paths = RunPaths::for_run(repo.path(), run_id);
        fs::create_dir_all(&paths.root).unwrap();
        let s = fixture_state(run_id, RunStatus::Active);
        s.write(&paths).unwrap();
        let listing = resolve_target(repo.path(), Some(run_id)).unwrap();
        let ok = validate_resume(listing, "default", &["bravo".into(), "alpha".into()]).unwrap();
        assert_eq!(ok.run_id, run_id);
    }

    #[test]
    fn validate_resume_rejects_renamed_plan() {
        let repo = tempdir().unwrap();
        let run_id = "rid";
        let paths = RunPaths::for_run(repo.path(), run_id);
        fs::create_dir_all(&paths.root).unwrap();
        let s = fixture_state(run_id, RunStatus::Active);
        s.write(&paths).unwrap();
        let listing = resolve_target(repo.path(), Some(run_id)).unwrap();
        let err =
            validate_resume(listing, "fp-cleanup", &["alpha".into(), "bravo".into()]).unwrap_err();
        assert!(matches!(err, ResumeError::PlanRenamed { .. }));
    }

    #[test]
    fn run_status_is_resumable_truth_table() {
        assert!(RunStatus::Active.is_resumable());
        assert!(RunStatus::Aborted.is_resumable());
        assert!(!RunStatus::Completed.is_resumable());
        assert!(!RunStatus::Failed.is_resumable());
    }

    #[test]
    fn missing_state_json_is_skipped_in_listings() {
        let repo = tempdir().unwrap();
        let dir = repo.path().join(".pitboss/grind/runs/no-state");
        fs::create_dir_all(&dir).unwrap();
        assert!(list_runs(repo.path()).is_empty());
    }

    fn fixture_session_record(seq: u32) -> super::super::run_dir::SessionRecord {
        fixture_session_record_named(seq, "alpha", SessionStatus::Ok, 0, 0, 0.0)
    }

    fn fixture_session_record_named(
        seq: u32,
        prompt: &str,
        status: SessionStatus,
        input: u64,
        output: u64,
        cost: f64,
    ) -> SessionRecord {
        use crate::git::CommitId;
        use crate::state::TokenUsage;
        use std::collections::HashMap;
        use std::path::PathBuf;
        SessionRecord {
            seq,
            run_id: "rid".into(),
            prompt: prompt.into(),
            started_at: "2026-04-30T18:00:00Z".parse().unwrap(),
            ended_at: "2026-04-30T18:01:00Z".parse().unwrap(),
            status,
            summary: Some(format!("session {seq}")),
            commit: Some(CommitId::new(format!("abc{seq:040}"))),
            tokens: TokenUsage {
                input,
                output,
                by_role: HashMap::new(),
            },
            cost_usd: cost,
            transcript_path: PathBuf::from(format!("transcripts/session-{seq:04}.log")),
        }
    }

    fn fixture_prompt(name: &str) -> PromptDoc {
        use crate::grind::prompt::PromptMeta;
        PromptDoc {
            meta: PromptMeta {
                name: name.into(),
                description: format!("desc for {name}"),
                weight: 1,
                every: 1,
                max_runs: None,
                verify: false,
                parallel_safe: false,
                tags: vec![],
                max_session_seconds: None,
                max_session_cost_usd: None,
            },
            body: format!("body for {name}"),
            source_path: PathBuf::from(format!("/fixture/{name}.md")),
            source_kind: crate::grind::prompt::PromptSource::Project,
        }
    }

    fn fixture_plan_one_prompt(name: &str) -> (GrindPlan, BTreeMap<String, PromptDoc>) {
        use crate::grind::plan::default_plan_from_dir;
        let prompts = vec![fixture_prompt(name)];
        let plan = default_plan_from_dir(&prompts);
        let lookup: BTreeMap<String, PromptDoc> = prompts
            .into_iter()
            .map(|p| (p.meta.name.clone(), p))
            .collect();
        (plan, lookup)
    }

    fn fixture_state_aligned_with_log(records: &[SessionRecord]) -> RunState {
        let mut state = fixture_state("rid", RunStatus::Active);
        state.last_session_seq = records.iter().map(|r| r.seq).max().unwrap_or(0);
        // Align the cached scheduler_state with what the original loop would
        // have produced: every record bumps runs_per_prompt for its prompt
        // and rotation goes up by one (every-gating is absent in fixtures).
        let mut runs: BTreeMap<String, u32> = BTreeMap::new();
        for r in records {
            *runs.entry(r.prompt.clone()).or_default() += 1;
        }
        state.scheduler_state = SchedulerState {
            rotation: records.len() as u64,
            runs_per_prompt: runs,
        };
        state
    }

    #[test]
    fn reconstruct_returns_identity_when_state_matches_jsonl_tail() {
        let (plan, prompts) = fixture_plan_one_prompt("alpha");
        let records = vec![
            fixture_session_record(1),
            fixture_session_record(2),
            fixture_session_record(3),
        ];
        let state = fixture_state_aligned_with_log(&records);
        let recon = reconstruct_state_from_log(&state, &records, &plan, &prompts).unwrap();
        assert_eq!(recon.records_replayed, 0);
        assert_eq!(recon.last_session_seq, 3);
        assert_eq!(recon.scheduler_state, state.scheduler_state);
    }

    #[test]
    fn reconstruct_passes_on_empty_log_with_zero_state() {
        let (plan, prompts) = fixture_plan_one_prompt("alpha");
        let records: Vec<SessionRecord> = Vec::new();
        let mut state = fixture_state("rid", RunStatus::Active);
        state.last_session_seq = 0;
        state.scheduler_state = SchedulerState::default();
        let recon = reconstruct_state_from_log(&state, &records, &plan, &prompts).unwrap();
        assert_eq!(recon.records_replayed, 0);
        assert_eq!(recon.last_session_seq, 0);
    }

    #[test]
    fn reconstruct_replays_missing_records_when_jsonl_is_ahead() {
        // The "single dropped state.json write" recovery path: state.json
        // captured rotation=2 / runs[alpha]=2, the next session landed in the
        // JSONL but the host died before state.json picked it up. We replay
        // session 3 through the scheduler and recover.
        let (plan, prompts) = fixture_plan_one_prompt("alpha");
        let records = vec![
            fixture_session_record_named(1, "alpha", SessionStatus::Ok, 100, 50, 0.01),
            fixture_session_record_named(2, "alpha", SessionStatus::Ok, 100, 50, 0.01),
            fixture_session_record_named(3, "alpha", SessionStatus::Ok, 200, 100, 0.02),
        ];
        let state = fixture_state_aligned_with_log(&records[..2]);
        let original_budget = state.budget_consumed;
        let recon = reconstruct_state_from_log(&state, &records, &plan, &prompts).unwrap();
        assert_eq!(recon.records_replayed, 1);
        assert_eq!(recon.last_session_seq, 3);
        assert_eq!(recon.scheduler_state.rotation, 3);
        assert_eq!(recon.scheduler_state.runs_per_prompt.get("alpha"), Some(&3));
        assert_eq!(
            recon.budget_consumed.iterations,
            original_budget.iterations + 1
        );
        assert_eq!(
            recon.budget_consumed.tokens_input,
            original_budget.tokens_input + 200
        );
        assert_eq!(
            recon.budget_consumed.tokens_output,
            original_budget.tokens_output + 100
        );
        assert!((recon.budget_consumed.cost_usd - (original_budget.cost_usd + 0.02)).abs() < 1e-9);
    }

    #[test]
    fn reconstruct_resets_consecutive_failures_on_replayed_success() {
        // Replayed Ok records reset the consecutive-failure counter, just like
        // the live tracker would. This keeps the escape valve in sync after a
        // recovery.
        let (plan, prompts) = fixture_plan_one_prompt("alpha");
        let records = vec![
            fixture_session_record_named(1, "alpha", SessionStatus::Error, 0, 0, 0.0),
            fixture_session_record_named(2, "alpha", SessionStatus::Ok, 0, 0, 0.0),
        ];
        let mut state = fixture_state_aligned_with_log(&records[..1]);
        state.budget_consumed.consecutive_failures = 1;
        let recon = reconstruct_state_from_log(&state, &records, &plan, &prompts).unwrap();
        assert_eq!(recon.budget_consumed.consecutive_failures, 0);
    }

    #[test]
    fn reconstruct_rejects_when_state_claims_more_than_jsonl_has() {
        let (plan, prompts) = fixture_plan_one_prompt("alpha");
        let records = vec![fixture_session_record(1)];
        let mut state = fixture_state_aligned_with_log(&records);
        state.last_session_seq = 5;
        let err = reconstruct_state_from_log(&state, &records, &plan, &prompts).unwrap_err();
        assert!(matches!(err, ResumeError::StateOutOfSync { .. }));
    }

    #[test]
    fn reconstruct_rejects_when_scheduler_diverges_from_recorded_prompt() {
        // The recorded prompt name is unknown to the scheduler — replay can't
        // match the source-of-truth log so we refuse rather than silently
        // producing a state that doesn't reflect what the original dispatched.
        let (plan, prompts) = fixture_plan_one_prompt("alpha");
        let records = vec![
            fixture_session_record_named(1, "alpha", SessionStatus::Ok, 0, 0, 0.0),
            fixture_session_record_named(2, "ghost", SessionStatus::Ok, 0, 0, 0.0),
        ];
        let state = fixture_state_aligned_with_log(&records[..1]);
        let err = reconstruct_state_from_log(&state, &records, &plan, &prompts).unwrap_err();
        assert!(matches!(err, ResumeError::StateOutOfSync { .. }));
    }
}
