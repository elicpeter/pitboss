//! Persisted resume state for a `pitboss grind` run.
//!
//! A run's `state.json` (written to `.pitboss/grind/<run-id>/state.json`)
//! caches everything `pitboss grind --resume` needs to pick up where the
//! original loop left off:
//!
//! - the scheduler position (rotation count + per-prompt run counts),
//! - the budget tracker's cumulative spend,
//! - the run's branch and plan name,
//! - the prompt-name list at run start, used as a "did the prompt set change"
//!   fingerprint, and
//! - the run's lifecycle status ([`RunStatus`]).
//!
//! `sessions.jsonl` remains the source of truth for the per-session record
//! stream; this file is a small derived cache so the resume path doesn't have
//! to re-aggregate every JSONL line on startup. Writes are atomic via
//! [`crate::util::write_atomic`] and happen after every session.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::util::write_atomic;

use super::budget::BudgetSnapshot;
use super::run_dir::{RunPaths, STATE_FILENAME};
use super::scheduler::SchedulerState;

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
        let mut bytes = serde_json::to_vec_pretty(self)
            .context("grind state: serializing RunState")?;
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
    /// Run id (directory name under `.pitboss/grind/`).
    pub run_id: String,
    /// Path to the `state.json` file.
    pub state_path: PathBuf,
    /// Parsed state.
    pub state: RunState,
}

/// Walk `<repo>/.pitboss/grind/` for runs that have a parseable `state.json`,
/// regardless of their lifecycle status. Errors from individual run dirs are
/// dropped so a single corrupt run doesn't hide the rest. The result is sorted
/// by `last_updated_at` descending (most recent first).
pub fn list_runs(repo_root: &Path) -> Vec<RunListing> {
    let grind_root = repo_root.join(".pitboss").join("grind");
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
    out.sort_by(|a, b| b.state.last_updated_at.cmp(&a.state.last_updated_at));
    out
}

/// Find the most-recent run under `<repo>/.pitboss/grind/` whose persisted
/// status is [`RunStatus::Active`] or [`RunStatus::Aborted`]. Returns `None`
/// when no resumable run exists. The default target for `--resume` with no
/// argument.
pub fn most_recent_resumable(repo_root: &Path) -> Option<RunListing> {
    list_runs(repo_root)
        .into_iter()
        .find(|r| r.state.status.is_resumable())
}

/// Errors a resume validation can produce. Each variant maps onto a clear
/// CLI diagnostic; none of them leave the run in a partially resumed state.
#[derive(Debug, thiserror::Error)]
pub enum ResumeError {
    /// No resumable run exists under `<repo>/.pitboss/grind/`.
    #[error("no resumable grind run found under {dir}")]
    NoResumableRun {
        /// `.pitboss/grind/` path searched.
        dir: PathBuf,
    },
    /// The named run does not exist on disk.
    #[error("grind run {run_id:?} not found at {dir}")]
    RunNotFound {
        /// Requested run id.
        run_id: String,
        /// The expected `<repo>/.pitboss/grind/<run-id>/` path.
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
    /// The plan name in `pitboss.toml`/CLI no longer matches the run's
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
    let grind_root = repo_root.join(".pitboss").join("grind");
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
            let state = RunState::read_path(&state_path).map_err(|e| {
                ResumeError::StateUnreadable {
                    run_id: id.to_string(),
                    source: e,
                }
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
            ("rid-completed", "2026-04-30T18:00:00Z", RunStatus::Completed),
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
        let err = validate_resume(
            listing,
            "default",
            &["alpha".into(), "bravo".into()],
        )
        .unwrap_err();
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
        let ok = validate_resume(
            listing,
            "default",
            &["bravo".into(), "alpha".into()],
        )
        .unwrap();
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
        let err = validate_resume(
            listing,
            "fp-cleanup",
            &["alpha".into(), "bravo".into()],
        )
        .unwrap_err();
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
        let dir = repo.path().join(".pitboss/grind/no-state");
        fs::create_dir_all(&dir).unwrap();
        assert!(list_runs(repo.path()).is_empty());
    }
}
