//! `pitboss grind` — sequential rotating prompt runner.
//!
//! This is the user-facing front-end on top of [`crate::grind::GrindRunner`]:
//! it loads `pitboss.toml`, discovers prompts from the project / global /
//! `--prompts-dir` precedence chain, picks (or synthesizes) a plan, opens a
//! per-run directory under `.pitboss/grind/<run-id>/`, creates and checks out
//! the run branch (`pitboss/grind/<run-id>`), wires up Ctrl-C handling, and
//! drives the runner to completion.
//!
//! Phase 07 ships the sequential MVP only — `--max-iterations`, `--until`,
//! `--max-cost`, `--max-tokens`, `--resume`, `--pr`, `--tui`, and hook
//! execution land in phases 08-13.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use tokio::signal;
use tracing::{info, warn};

use crate::agent::{self, Agent};
use crate::config::{self, Config};
use crate::git::{Git, ShellGit};
use crate::grind::{
    default_plan_from_dir, discover_prompts, generate_run_id, load_plan, run_branch_name,
    DiscoveryOptions, GrindPlan, GrindRunner, GrindShutdown, GrindStopReason, PromptDoc, RunDir,
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
}

/// Entry point invoked from `cli::dispatch`.
pub async fn run(workspace: PathBuf, args: GrindArgs) -> Result<()> {
    let config = config::load(&workspace)
        .with_context(|| format!("grind: loading config in {:?}", workspace))?;
    let prompts = load_prompts(&workspace, &config, args.prompts_dir.as_deref())?;
    let plan = resolve_plan(&workspace, &config, args.plan.as_deref(), &prompts)?;
    plan.validate_against(&prompts)
        .with_context(|| format!("grind: validating plan {:?}", plan.name))?;

    if args.dry_run {
        return print_dry_run_summary(&workspace, &config, &plan, &prompts);
    }

    let agent = agent::build_agent(&config)?;
    execute(workspace, config, plan, prompts, agent).await
}

async fn execute<A>(
    workspace: PathBuf,
    config: Config,
    plan: GrindPlan,
    prompts: Vec<PromptDoc>,
    agent: A,
) -> Result<()>
where
    A: Agent + 'static,
{
    let run_id = generate_run_id();
    let branch = run_branch_name(&run_id);

    let git = ShellGit::new(workspace.clone());
    git.create_branch(&branch).await.with_context(|| {
        format!(
            "grind: creating run branch {:?} (workspace must be a git repo)",
            branch
        )
    })?;
    git.checkout(&branch)
        .await
        .with_context(|| format!("grind: checking out {:?}", branch))?;

    let run_dir = RunDir::create(&workspace, &run_id)
        .with_context(|| format!("grind: creating run directory for {run_id}"))?;

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
    );

    let shutdown = GrindShutdown::new();
    let signal_task = spawn_signal_handler(shutdown.clone());

    announce_start(&run_id, &branch);

    let result = runner.run(shutdown.clone()).await;

    signal_task.abort();
    let _ = signal_task.await;

    let outcome = result?;
    announce_finish(&outcome.run_id, &outcome.branch, outcome.stop_reason, outcome.sessions.len());

    if outcome.stop_reason == GrindStopReason::Aborted {
        bail!("grind run {} aborted by signal", outcome.run_id);
    }
    Ok(())
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
        bail!(
            "grind: no prompts discovered (run `pitboss prompts new <name>` to create one)"
        );
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

fn announce_finish(run_id: &str, branch: &str, reason: GrindStopReason, sessions: usize) {
    let c = style::use_color_stderr();
    let label = match reason {
        GrindStopReason::Completed => col(c, style::BOLD_GREEN, "completed"),
        GrindStopReason::Drained => col(c, style::BOLD_YELLOW, "drained"),
        GrindStopReason::Aborted => col(c, style::BOLD_RED, "aborted"),
    };
    eprintln!(
        "{} grind run {} {} after {} session(s) on {}",
        col(c, style::BOLD_CYAN, "[pitboss]"),
        col(c, style::BOLD_WHITE, run_id),
        label,
        sessions,
        col(c, style::CYAN, branch),
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
