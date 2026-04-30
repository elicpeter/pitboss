//! `foreman resume` — pick up a halted run from `.foreman/state.json`.
//!
//! Resume reuses the runner driver from [`crate::cli::run::execute`] with
//! [`StartMode::Resume`], so the only behavioral delta from `foreman run` is
//! that a missing or empty `state.json` is an error: there is nothing to
//! resume. The per-run branch is checked out, [`crate::runner::Runner`] is
//! constructed against the loaded plan / deferred / state, and execution
//! continues from `plan.current_phase` — which the runner advanced to the next
//! phase the last time it persisted state.

use std::path::PathBuf;

use anyhow::Result;

use super::run::{execute, StartMode};

/// Top-level entry point for the `resume` subcommand. `tui` toggles the
/// `ratatui` dashboard the same way [`crate::cli::run::run`] does.
pub async fn run(workspace: PathBuf, tui: bool) -> Result<()> {
    execute(workspace, tui, StartMode::Resume).await
}
