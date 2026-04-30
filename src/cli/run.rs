//! `foreman run` — execute the plan against the configured agent.
//!
//! Loads the workspace's `foreman.toml`, `plan.md`, `deferred.md`, and
//! `state.json`; ensures a per-run branch exists; spawns a [`broadcast`]
//! subscriber that streams [`runner::Event`]s to stderr; then drives the
//! runner until the plan completes or a phase halts.
//!
//! On a fresh run (state file is `null` or missing) this command derives a new
//! `run_id` and per-run branch from the current UTC timestamp, captures the
//! current branch as `original_branch` for `foreman abort --checkout-original`,
//! and creates the branch in git. On a continuation (state present) the
//! existing branch is checked out instead. Phase 17's `foreman resume` reuses
//! [`execute`] with [`StartMode::Resume`] to require an existing state file.
//!
//! Aborted runs (`state.aborted == true`) are refused — the user must clear
//! `.foreman/state.json` to start a new run.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use tokio::task::JoinHandle;

use crate::agent::claude_code::ClaudeCodeAgent;
use crate::config;
use crate::deferred::{self, DeferredDoc};
use crate::git::{Git, ShellGit};
use crate::plan::{self, Plan};
use crate::runner::{self, RunSummary, Runner};
use crate::state;
use crate::tui;

/// Whether [`execute`] is allowed to start a fresh run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartMode {
    /// `foreman run`: a missing or `null` state file kicks off a fresh run.
    Fresh,
    /// `foreman resume`: a missing state file is an error.
    Resume,
}

/// Top-level entry point for the `run` subcommand.
///
/// `tui` toggles between the plain stderr logger (default) and the
/// `ratatui` dashboard.
pub async fn run(workspace: PathBuf, tui: bool) -> Result<()> {
    execute(workspace, tui, StartMode::Fresh).await
}

/// Shared runner driver used by both `foreman run` and `foreman resume`.
///
/// `mode` selects fresh-start vs. resume semantics. The function loads config,
/// the plan, and the deferred doc; reconciles state with `mode`; ensures the
/// per-run branch is checked out; spawns the configured event subscriber
/// (logger or TUI); then drives [`Runner::run`] to completion or halt.
pub async fn execute(workspace: PathBuf, tui: bool, mode: StartMode) -> Result<()> {
    let config = config::load(&workspace)
        .with_context(|| format!("run: loading config in {:?}", workspace))?;
    let plan = load_plan(&workspace)?;
    let deferred = load_deferred(&workspace)?;

    let existing_state = state::load(&workspace)
        .with_context(|| format!("run: loading state in {:?}", workspace))?;

    let git = ShellGit::new(workspace.clone());

    let (state, is_fresh_run) = match (existing_state, mode) {
        (Some(s), _) => {
            if s.aborted {
                bail!(
                    "state.json marks run {} as aborted; remove .foreman/state.json to start over",
                    s.run_id
                );
            }
            (s, false)
        }
        (None, StartMode::Fresh) => {
            let original_branch = git.current_branch().await.ok();
            let mut s = runner::fresh_run_state(&plan, &config, Utc::now());
            s.original_branch = original_branch;
            (s, true)
        }
        (None, StartMode::Resume) => {
            bail!(
                "no run to resume: .foreman/state.json is empty; use `foreman run` to start a fresh run"
            );
        }
    };

    if is_fresh_run {
        git.create_branch(&state.branch).await.with_context(|| {
            format!(
                "run: creating per-run branch {:?} (workspace must already be a git repo)",
                state.branch
            )
        })?;
    }
    git.checkout(&state.branch)
        .await
        .with_context(|| format!("run: checking out {:?}", state.branch))?;
    state::save(&workspace, Some(&state))
        .with_context(|| format!("run: persisting initial state in {:?}", workspace))?;

    let agent = ClaudeCodeAgent::new();
    let mut runner = Runner::new(workspace, config, plan, deferred, state, agent, git);

    let summary = if tui {
        tui::run(&mut runner).await?
    } else {
        let logger = spawn_logger(&runner);
        let result = runner.run().await;
        let _ = logger.await;
        Some(result?)
    };

    match summary {
        None => Ok(()),
        Some(RunSummary::Finished) => Ok(()),
        Some(RunSummary::Halted { phase_id, reason }) => {
            Err(anyhow!("run halted at phase {phase_id}: {reason}"))
        }
    }
}

fn load_plan(workspace: &Path) -> Result<Plan> {
    let path = workspace.join("plan.md");
    let text = fs::read_to_string(&path).with_context(|| format!("run: reading {:?}", path))?;
    plan::parse(&text).with_context(|| format!("run: parsing {:?}", path))
}

fn load_deferred(workspace: &Path) -> Result<DeferredDoc> {
    let path = workspace.join("deferred.md");
    match fs::read_to_string(&path) {
        Ok(text) => {
            if text.trim().is_empty() {
                Ok(DeferredDoc::empty())
            } else {
                deferred::parse(&text).with_context(|| format!("run: parsing {:?}", path))
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(DeferredDoc::empty()),
        Err(e) => Err(anyhow::Error::new(e).context(format!("run: reading {:?}", path))),
    }
}

fn spawn_logger<A, G>(runner: &Runner<A, G>) -> JoinHandle<()>
where
    A: crate::agent::Agent + 'static,
    G: Git + 'static,
{
    let rx = runner.subscribe();
    tokio::spawn(runner::log_events(rx))
}
