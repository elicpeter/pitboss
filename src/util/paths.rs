//! Workspace-relative paths for everything pitboss writes under `.pitboss/`.
//!
//! The directory layout is:
//!
//! ```text
//! .pitboss/                # entirely gitignored
//! ├── config.toml          # shared config, both play and grind
//! ├── play/                # multi-phase runner artifacts
//! │   ├── plan.md
//! │   ├── deferred.md
//! │   ├── state.json
//! │   ├── logs/
//! │   └── snapshots/
//! └── grind/               # rotating prompt runner
//!     ├── prompts/         # project-local prompt files
//!     ├── rotations/       # rotation TOML files (selected with --rotation)
//!     └── runs/
//!         └── <run-id>/
//! ```
//!
//! Every callsite that needs one of these paths goes through a helper here so
//! the layout is single-sourced.

use std::path::{Path, PathBuf};

/// `.pitboss/` — the umbrella directory that holds every pitboss artifact.
pub fn pitboss_dir(workspace: impl AsRef<Path>) -> PathBuf {
    workspace.as_ref().join(".pitboss")
}

/// `.pitboss/play/` — multi-phase runner state.
pub fn play_dir(workspace: impl AsRef<Path>) -> PathBuf {
    pitboss_dir(workspace).join("play")
}

/// `.pitboss/grind/` — grind session state.
pub fn grind_dir(workspace: impl AsRef<Path>) -> PathBuf {
    pitboss_dir(workspace).join("grind")
}

/// `.pitboss/grind/prompts/` — project-local prompt files for `pitboss grind`.
pub fn grind_prompts_dir(workspace: impl AsRef<Path>) -> PathBuf {
    grind_dir(workspace).join("prompts")
}

/// `<home>/.pitboss/grind/prompts/` — global prompt files for `pitboss grind`.
pub fn home_grind_prompts_dir(home: impl AsRef<Path>) -> PathBuf {
    home.as_ref().join(".pitboss").join("grind").join("prompts")
}

/// `.pitboss/grind/rotations/` — rotation TOML files selected by `--rotation`.
pub fn grind_rotations_dir(workspace: impl AsRef<Path>) -> PathBuf {
    grind_dir(workspace).join("rotations")
}

/// `.pitboss/grind/runs/` — parent of every per-run directory.
pub fn grind_runs_dir(workspace: impl AsRef<Path>) -> PathBuf {
    grind_dir(workspace).join("runs")
}

/// `.pitboss/grind/runs/<run-id>/` — root of a specific grind run on disk.
pub fn grind_run_dir(workspace: impl AsRef<Path>, run_id: &str) -> PathBuf {
    grind_runs_dir(workspace).join(run_id)
}

/// `.pitboss/config.toml` — shared configuration for both play and grind.
pub fn config_path(workspace: impl AsRef<Path>) -> PathBuf {
    pitboss_dir(workspace).join("config.toml")
}

/// `.pitboss/play/plan.md` — phased plan that the runner walks through.
pub fn plan_path(workspace: impl AsRef<Path>) -> PathBuf {
    play_dir(workspace).join("plan.md")
}

/// `.pitboss/play/deferred.md` — agent-writable scratchpad for items the
/// runner couldn't land in the active phase.
pub fn deferred_path(workspace: impl AsRef<Path>) -> PathBuf {
    play_dir(workspace).join("deferred.md")
}

/// `.pitboss/play/state.json` — runner state checkpoint (phase progress,
/// attempts, branch, budgets).
pub fn state_path(workspace: impl AsRef<Path>) -> PathBuf {
    play_dir(workspace).join("state.json")
}

/// `.pitboss/play/logs/` — per-phase, per-attempt agent and test logs.
pub fn play_logs_dir(workspace: impl AsRef<Path>) -> PathBuf {
    play_dir(workspace).join("logs")
}

/// `.pitboss/play/snapshots/` — pre-agent snapshots of plan.md / deferred.md
/// used to restore on tampering or parse failure.
pub fn play_snapshots_dir(workspace: impl AsRef<Path>) -> PathBuf {
    play_dir(workspace).join("snapshots")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn layout_is_nested_under_pitboss_dir() {
        let ws = PathBuf::from("/tmp/ws");
        assert_eq!(pitboss_dir(&ws), ws.join(".pitboss"));
        assert_eq!(play_dir(&ws), ws.join(".pitboss/play"));
        assert_eq!(grind_dir(&ws), ws.join(".pitboss/grind"));
        assert_eq!(grind_prompts_dir(&ws), ws.join(".pitboss/grind/prompts"));
        assert_eq!(
            grind_rotations_dir(&ws),
            ws.join(".pitboss/grind/rotations")
        );
        assert_eq!(grind_runs_dir(&ws), ws.join(".pitboss/grind/runs"));
        assert_eq!(
            grind_run_dir(&ws, "20260501T000000Z"),
            ws.join(".pitboss/grind/runs/20260501T000000Z")
        );
        let home = PathBuf::from("/home/u");
        assert_eq!(
            home_grind_prompts_dir(&home),
            home.join(".pitboss/grind/prompts")
        );
        assert_eq!(config_path(&ws), ws.join(".pitboss/config.toml"));
        assert_eq!(plan_path(&ws), ws.join(".pitboss/play/plan.md"));
        assert_eq!(deferred_path(&ws), ws.join(".pitboss/play/deferred.md"));
        assert_eq!(state_path(&ws), ws.join(".pitboss/play/state.json"));
        assert_eq!(play_logs_dir(&ws), ws.join(".pitboss/play/logs"));
        assert_eq!(play_snapshots_dir(&ws), ws.join(".pitboss/play/snapshots"));
    }
}
