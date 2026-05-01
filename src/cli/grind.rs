//! `pitboss grind` — sequential rotating prompt runner.
//!
//! This is the user-facing front-end on top of [`crate::grind::GrindRunner`]:
//! it loads `pitboss.toml`, discovers prompts from the project / global /
//! `--prompts-dir` precedence chain, picks (or synthesizes) a plan, opens a
//! per-run directory under `.pitboss/grind/<run-id>/`, creates and checks out
//! the run branch (`pitboss/grind/<run-id>`), wires up Ctrl-C handling, and
//! drives the runner to completion.
//!
//! Phase 07 shipped the sequential MVP. Phase 08 adds the run-wide budgets
//! (`--max-iterations`, `--until`, `--max-cost`, `--max-tokens`) and the
//! documented [`crate::grind::ExitCode`] policy.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use clap::Args;
use tokio::signal;
use tracing::{info, warn};

use crate::agent::{self, Agent};
use crate::config::{self, Config};
use crate::git::{Git, ShellGit};
use crate::grind::{
    default_plan_from_dir, discover_prompts, generate_run_id, load_plan, resolve_budgets,
    resolve_target, run_branch_name, validate_resume, DiscoveryOptions, ExitCode, GrindPlan,
    GrindRunner, GrindShutdown, GrindStopReason, PlanBudgets, PromptDoc, ResumeError, RunDir,
    RunListing, SessionRecord, SessionStatus,
};
use crate::style::{self, col};

/// `pitboss grind [options]` argument surface.
#[derive(Debug, Args)]
pub struct GrindArgs {
    /// Plan name to load. Resolves to `.pitboss/plans/<plan>.toml`. Without
    /// this flag the runner falls back to `[grind] default_plan` from
    /// `pitboss.toml`, then to a synthesized default-rotation plan over every
    /// discovered prompt.
    #[arg(long)]
    pub plan: Option<String>,
    /// Override the prompt discovery directory. Suppresses both project
    /// (`./.pitboss/prompts/`) and global (`~/.pitboss/prompts/`) sources.
    #[arg(long = "prompts-dir")]
    pub prompts_dir: Option<PathBuf>,
    /// Resolve and print the planned rotation, then exit without dispatching
    /// any agents or creating a run directory. Phase 12 fleshes this out;
    /// Phase 07 prints a one-paragraph summary so users can sanity-check the
    /// resolved configuration before kicking off a long run.
    #[arg(long = "dry-run")]
    pub dry_run: bool,
    /// Stop after this many sessions have been dispatched. Overrides
    /// `[grind.budgets] max_iterations` from `pitboss.toml` and the plan's
    /// `PlanBudgets`.
    #[arg(long = "max-iterations", value_name = "N")]
    pub max_iterations: Option<u32>,
    /// RFC 3339 wall-clock cutoff. Once `Utc::now() >= until` the runner
    /// finishes any in-flight session and exits with code 3. Overrides
    /// `[grind.budgets] until` from `pitboss.toml` and the plan's
    /// `PlanBudgets`.
    #[arg(long = "until", value_name = "RFC3339", value_parser = parse_rfc3339)]
    pub until: Option<DateTime<Utc>>,
    /// Hard cap on cumulative agent cost in USD. Computed from
    /// `[budgets.pricing]`. Overrides the corresponding `[grind.budgets]` /
    /// plan field.
    #[arg(long = "max-cost", value_name = "USD")]
    pub max_cost: Option<f64>,
    /// Hard cap on cumulative tokens (input + output) across all roles.
    /// Overrides the corresponding `[grind.budgets]` / plan field.
    #[arg(long = "max-tokens", value_name = "N")]
    pub max_tokens: Option<u64>,
    /// Resume a previous grind run instead of starting a new one. Without an
    /// argument, picks the most-recent run whose persisted status is `Active`
    /// or `Aborted`. With an argument, resumes the named run id (the
    /// directory name under `.pitboss/grind/`). Refuses to resume when the
    /// plan or prompt set has changed in a way that would invalidate the
    /// scheduler.
    #[arg(long = "resume", value_name = "RUN_ID", num_args = 0..=1, default_missing_value = "")]
    pub resume: Option<String>,
}

fn parse_rfc3339(s: &str) -> std::result::Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| format!("not a valid RFC 3339 timestamp ({e}): {s:?}"))
}

/// Entry point invoked from `cli::dispatch`. Returns an [`ExitCode`] mapping
/// to the documented `pitboss grind` exit policy. Setup errors that happen
/// before any session is dispatched surface as
/// [`ExitCode::FailedToStart`] with a stderr message.
pub async fn run(workspace: PathBuf, args: GrindArgs) -> Result<ExitCode> {
    let config = match config::load(&workspace) {
        Ok(c) => c,
        Err(e) => {
            print_failed_to_start(&format!("loading config: {e:#}"));
            return Ok(ExitCode::FailedToStart);
        }
    };
    let prompts = match load_prompts(&workspace, &config, args.prompts_dir.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            print_failed_to_start(&format!("{e:#}"));
            return Ok(ExitCode::FailedToStart);
        }
    };
    let plan = match resolve_plan(&workspace, &config, args.plan.as_deref(), &prompts) {
        Ok(p) => p,
        Err(e) => {
            print_failed_to_start(&format!("{e:#}"));
            return Ok(ExitCode::FailedToStart);
        }
    };
    if let Err(e) = plan.validate_against(&prompts) {
        print_failed_to_start(&format!("validating plan {:?}: {e:#}", plan.name));
        return Ok(ExitCode::FailedToStart);
    }

    if args.dry_run {
        print_dry_run_summary(&workspace, &config, &plan, &prompts)?;
        return Ok(ExitCode::Success);
    }

    let agent = match agent::build_agent(&config) {
        Ok(a) => a,
        Err(e) => {
            print_failed_to_start(&format!("building agent: {e:#}"));
            return Ok(ExitCode::FailedToStart);
        }
    };

    if let Some(target) = args.resume.as_deref() {
        let requested = if target.is_empty() { None } else { Some(target) };
        return execute_resume(workspace, config, plan, prompts, agent, &args, requested).await;
    }

    execute(workspace, config, plan, prompts, agent, &args).await
}

async fn execute<A>(
    workspace: PathBuf,
    config: Config,
    plan: GrindPlan,
    prompts: Vec<PromptDoc>,
    agent: A,
    args: &GrindArgs,
) -> Result<ExitCode>
where
    A: Agent + 'static,
{
    let run_id = generate_run_id();
    let branch = run_branch_name(&run_id);

    let git = ShellGit::new(workspace.clone());
    if let Err(e) = git.create_branch(&branch).await {
        print_failed_to_start(&format!(
            "creating run branch {:?} (workspace must be a git repo): {e:#}",
            branch
        ));
        return Ok(ExitCode::FailedToStart);
    }
    if let Err(e) = git.checkout(&branch).await {
        print_failed_to_start(&format!("checking out {:?}: {e:#}", branch));
        return Ok(ExitCode::FailedToStart);
    }

    let run_dir = match RunDir::create(&workspace, &run_id) {
        Ok(d) => d,
        Err(e) => {
            print_failed_to_start(&format!("creating run directory for {run_id}: {e:#}"));
            return Ok(ExitCode::FailedToStart);
        }
    };

    // Layer the budget sources so a CLI flag wins over `[grind.budgets]`
    // which wins over the plan's `PlanBudgets`. Order docs in
    // `crate::grind::resolve_budgets`.
    let cli_budgets = PlanBudgets {
        max_iterations: args.max_iterations,
        until: args.until,
        max_tokens: args.max_tokens,
        max_cost_usd: args.max_cost,
    };
    let budgets = resolve_budgets(&config.grind.budgets, &plan.budgets, &cli_budgets);
    let consecutive_failure_limit = config.grind.consecutive_failure_limit;

    let prompts_map = into_lookup(prompts);
    let runner_git = ShellGit::new(workspace.clone());
    let mut runner = GrindRunner::new(
        workspace.clone(),
        config,
        run_id.clone(),
        branch.clone(),
        plan,
        prompts_map,
        run_dir,
        agent,
        runner_git,
        budgets,
        consecutive_failure_limit,
    );

    let shutdown = GrindShutdown::new();
    let signal_task = spawn_signal_handler(shutdown.clone());

    announce_start(&run_id, &branch);

    let result = runner.run(shutdown.clone()).await;

    signal_task.abort();
    let _ = signal_task.await;

    let outcome = result?;
    announce_finish(
        &outcome.run_id,
        &outcome.branch,
        &outcome.stop_reason,
        outcome.sessions.len(),
    );

    Ok(classify_outcome(&outcome.stop_reason, &outcome.sessions))
}

async fn execute_resume<A>(
    workspace: PathBuf,
    config: Config,
    plan: GrindPlan,
    prompts: Vec<PromptDoc>,
    agent: A,
    args: &GrindArgs,
    requested: Option<&str>,
) -> Result<ExitCode>
where
    A: Agent + 'static,
{
    let listing = match resolve_target(&workspace, requested) {
        Ok(l) => l,
        Err(e) => {
            print_failed_to_start(&render_resume_error(&e));
            return Ok(ExitCode::FailedToStart);
        }
    };

    let current_prompt_names: Vec<String> = plan.prompts.iter().map(|p| p.name.clone()).collect();
    let listing = match validate_resume(listing, &plan.name, &current_prompt_names) {
        Ok(l) => l,
        Err(e) => {
            print_failed_to_start(&render_resume_error(&e));
            return Ok(ExitCode::FailedToStart);
        }
    };

    // Spec: "After-resume sanity: re-checkout the run branch; if working tree
    // is dirty, halts with exit code 4." We check `is_clean` *before*
    // checkout so a dirty tree we'd otherwise carry into the resumed branch
    // is surfaced as the failure point.
    let git = ShellGit::new(workspace.clone());
    match git.is_clean().await {
        Ok(true) => {}
        Ok(false) => {
            print_failed_to_start(&format!(
                "resume {:?}: working tree is dirty; commit or stash changes before resuming",
                listing.run_id
            ));
            return Ok(ExitCode::FailedToStart);
        }
        Err(e) => {
            print_failed_to_start(&format!(
                "resume {:?}: checking working tree: {e:#}",
                listing.run_id
            ));
            return Ok(ExitCode::FailedToStart);
        }
    }
    if let Err(e) = git.checkout(&listing.state.branch).await {
        print_failed_to_start(&format!(
            "resume {:?}: checking out {:?}: {e:#}",
            listing.run_id, listing.state.branch
        ));
        return Ok(ExitCode::FailedToStart);
    }

    let run_dir = match RunDir::open(&workspace, &listing.run_id) {
        Ok(d) => d,
        Err(e) => {
            print_failed_to_start(&format!(
                "resume {:?}: opening run directory: {e:#}",
                listing.run_id
            ));
            return Ok(ExitCode::FailedToStart);
        }
    };

    let cli_budgets = PlanBudgets {
        max_iterations: args.max_iterations,
        until: args.until,
        max_tokens: args.max_tokens,
        max_cost_usd: args.max_cost,
    };
    let budgets = resolve_budgets(&config.grind.budgets, &plan.budgets, &cli_budgets);
    let consecutive_failure_limit = config.grind.consecutive_failure_limit;

    let prompts_map = into_lookup(prompts);
    let runner_git = ShellGit::new(workspace.clone());
    let RunListing {
        run_id,
        state_path: _,
        state,
    } = listing;
    let mut runner = GrindRunner::resume(
        workspace.clone(),
        config,
        run_id.clone(),
        state.branch.clone(),
        plan,
        prompts_map,
        run_dir,
        agent,
        runner_git,
        budgets,
        consecutive_failure_limit,
        state.scheduler_state,
        state.budget_consumed,
        state.last_session_seq,
        state.started_at,
    );

    let shutdown = GrindShutdown::new();
    let signal_task = spawn_signal_handler(shutdown.clone());

    announce_resume(&run_id, &state.branch, state.last_session_seq);

    let result = runner.run(shutdown.clone()).await;

    signal_task.abort();
    let _ = signal_task.await;

    let outcome = result?;
    announce_finish(
        &outcome.run_id,
        &outcome.branch,
        &outcome.stop_reason,
        outcome.sessions.len(),
    );

    Ok(classify_outcome(&outcome.stop_reason, &outcome.sessions))
}

fn render_resume_error(e: &ResumeError) -> String {
    format!("resume: {e}")
}

/// Map a grind run's [`GrindStopReason`] plus its session list to the
/// documented [`ExitCode`]. Public for the integration test crate.
pub fn classify_outcome(stop_reason: &GrindStopReason, sessions: &[SessionRecord]) -> ExitCode {
    match stop_reason {
        GrindStopReason::Aborted => ExitCode::Aborted,
        GrindStopReason::BudgetExhausted(_) => ExitCode::BudgetExhausted,
        GrindStopReason::ConsecutiveFailureLimit { .. } => ExitCode::ConsecutiveFailures,
        GrindStopReason::Completed | GrindStopReason::Drained => {
            if sessions
                .iter()
                .any(|r| matches!(r.status, SessionStatus::Error | SessionStatus::Timeout))
            {
                ExitCode::MixedFailures
            } else {
                ExitCode::Success
            }
        }
    }
}

fn load_prompts(
    workspace: &Path,
    config: &Config,
    flag_override: Option<&Path>,
) -> Result<Vec<PromptDoc>> {
    let override_dir = flag_override
        .map(|p| p.to_path_buf())
        .or_else(|| config.grind.prompts_dir.clone());
    let opts = DiscoveryOptions {
        project_root: workspace.to_path_buf(),
        home_dir: std::env::var_os("HOME").map(PathBuf::from),
        override_dir,
    };
    let discovered = discover_prompts(opts);
    if !discovered.errors.is_empty() {
        let stderr = std::io::stderr();
        let c = style::use_color_stderr();
        let mut h = stderr.lock();
        for (path, error) in &discovered.errors {
            let _ = writeln!(
                h,
                "{} {}: {}",
                col(c, style::BOLD_RED, "warning:"),
                path.display(),
                error
            );
        }
    }
    if discovered.prompts.is_empty() {
        bail!("grind: no prompts discovered (run `pitboss prompts new <name>` to create one)");
    }
    Ok(discovered.prompts)
}

fn resolve_plan(
    workspace: &Path,
    config: &Config,
    cli_plan: Option<&str>,
    prompts: &[PromptDoc],
) -> Result<GrindPlan> {
    let plan_name = cli_plan.or(config.grind.default_plan.as_deref());
    let Some(name) = plan_name else {
        return Ok(default_plan_from_dir(prompts));
    };
    let path = workspace
        .join(".pitboss")
        .join("plans")
        .join(format!("{name}.toml"));
    load_plan(&path).with_context(|| format!("grind: loading plan {:?}", path))
}

fn into_lookup(prompts: Vec<PromptDoc>) -> BTreeMap<String, PromptDoc> {
    let mut out = BTreeMap::new();
    for p in prompts {
        out.insert(p.meta.name.clone(), p);
    }
    out
}

/// Wire `Ctrl-C` to the runner's two-stage shutdown. First signal sets drain
/// (the in-flight session finishes, then the loop exits). Second signal aborts
/// (the agent's cancel token fires).
fn spawn_signal_handler(shutdown: GrindShutdown) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if signal::ctrl_c().await.is_err() {
            warn!("grind: failed to install Ctrl-C handler; ignoring signal-driven drain");
            return;
        }
        announce_drain();
        shutdown.drain();

        if signal::ctrl_c().await.is_err() {
            return;
        }
        announce_abort();
        shutdown.abort();
    })
}

fn announce_start(run_id: &str, branch: &str) {
    let c = style::use_color_stderr();
    eprintln!(
        "{} grind run {} on branch {}",
        col(c, style::BOLD_CYAN, "[pitboss]"),
        col(c, style::BOLD_WHITE, run_id),
        col(c, style::CYAN, branch),
    );
    info!(run_id, branch, "grind: run started");
}

fn announce_resume(run_id: &str, branch: &str, last_session_seq: u32) {
    let c = style::use_color_stderr();
    eprintln!(
        "{} resuming grind run {} on branch {} (next session-{:04})",
        col(c, style::BOLD_CYAN, "[pitboss]"),
        col(c, style::BOLD_WHITE, run_id),
        col(c, style::CYAN, branch),
        last_session_seq.saturating_add(1),
    );
    info!(
        run_id,
        branch,
        last_session_seq,
        "grind: run resumed"
    );
}

fn announce_drain() {
    let c = style::use_color_stderr();
    eprintln!(
        "{} draining: finishing the current session, press Ctrl-C again to abort",
        col(c, style::BOLD_YELLOW, "[pitboss]"),
    );
}

fn announce_abort() {
    let c = style::use_color_stderr();
    eprintln!(
        "{} aborting: cancelling the current agent",
        col(c, style::BOLD_RED, "[pitboss]"),
    );
}

fn announce_finish(run_id: &str, branch: &str, reason: &GrindStopReason, sessions: usize) {
    let c = style::use_color_stderr();
    let (label, suffix) = match reason {
        GrindStopReason::Completed => (col(c, style::BOLD_GREEN, "completed"), String::new()),
        GrindStopReason::Drained => (col(c, style::BOLD_YELLOW, "drained"), String::new()),
        GrindStopReason::Aborted => (col(c, style::BOLD_RED, "aborted"), String::new()),
        GrindStopReason::BudgetExhausted(reason) => (
            col(c, style::BOLD_YELLOW, "BudgetExhausted"),
            format!(" ({reason})"),
        ),
        GrindStopReason::ConsecutiveFailureLimit { limit } => (
            col(c, style::BOLD_RED, "consecutive-failure-limit"),
            format!(" (limit={limit})"),
        ),
    };
    eprintln!(
        "{} grind run {} {}{} after {} session(s) on {}",
        col(c, style::BOLD_CYAN, "[pitboss]"),
        col(c, style::BOLD_WHITE, run_id),
        label,
        suffix,
        sessions,
        col(c, style::CYAN, branch),
    );
    // The spec requires a final `BudgetExhausted` log line so log scrapers
    // and supervising scripts have a stable string to match on.
    if let GrindStopReason::BudgetExhausted(reason) = reason {
        info!(run_id, %reason, "BudgetExhausted");
    }
}

fn print_failed_to_start(message: &str) {
    let c = style::use_color_stderr();
    eprintln!(
        "{} grind: {} (exit 4)",
        col(c, style::BOLD_RED, "[pitboss]"),
        message,
    );
}

fn print_dry_run_summary(
    workspace: &Path,
    config: &Config,
    plan: &GrindPlan,
    prompts: &[PromptDoc],
) -> Result<()> {
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    writeln!(h, "# pitboss grind --dry-run")?;
    writeln!(h, "workspace: {}", workspace.display())?;
    writeln!(
        h,
        "agent backend: {}",
        config
            .agent
            .backend
            .as_deref()
            .unwrap_or("claude_code (default)")
    )?;
    writeln!(h, "plan: {}", plan.name)?;
    writeln!(h, "prompts: {}", prompts.len())?;
    for p in prompts {
        writeln!(
            h,
            "  - {} (weight={}, every={}, verify={}, source={:?})",
            p.meta.name, p.meta.weight, p.meta.every, p.meta.verify, p.source_kind
        )?;
    }
    writeln!(h, "plan entries: {}", plan.prompts.len())?;
    for entry in &plan.prompts {
        writeln!(
            h,
            "  - {} (weight_override={:?}, every_override={:?}, max_runs_override={:?})",
            entry.name, entry.weight_override, entry.every_override, entry.max_runs_override
        )?;
    }
    if plan.prompts.is_empty() {
        return Err(anyhow!("grind --dry-run: plan has no prompt entries"));
    }
    writeln!(h, "max_parallel: {}", plan.max_parallel)?;
    Ok(())
}
