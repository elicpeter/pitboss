//! Grind orchestrator: pick a prompt, dispatch the agent, capture the
//! summary, commit, repeat — sequentially or across parallel worktrees.
//!
//! [`GrindRunner`] wires together the artifacts assembled in phases 01-10:
//! discovered prompts, a [`GrindPlan`], a [`Scheduler`], and an open
//! [`RunDir`]. One [`GrindRunner::run`] call drives the loop until the
//! scheduler is exhausted, the run is drained (one Ctrl-C), or the run is
//! aborted (second Ctrl-C, or any other [`CancellationToken::cancel`]).
//!
//! Phase 11 lifts the runner from "one session at a time" to a real
//! concurrency gate. Each prompt declares whether it is `parallel_safe`; the
//! plan declares its `max_parallel` ceiling. A `tokio::sync::Semaphore` of
//! `max_parallel` permits guards dispatch:
//!
//! - A `parallel_safe: true` prompt grabs one permit, runs in its own
//!   [`SessionWorktree`] off `<run-root>/worktrees/session-NNNN/`, and
//!   fast-forwards the run branch to its session tip on completion.
//! - A `parallel_safe: false` prompt grabs *all* `max_parallel` permits,
//!   effectively serializing it against any in-flight parallel sessions.
//!   It runs in the main workspace exactly the way phase 07 wired it up.
//!
//! The runner is intentionally agnostic to the surface that signals a stop:
//! it takes a [`GrindShutdown`] handle that carries an
//! [`AtomicBool`](std::sync::atomic::AtomicBool) drain flag and a
//! [`CancellationToken`] abort token. The CLI binds those to live `Ctrl-C`
//! events; the integration tests flip them by hand.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, Mutex as TokioMutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::agent::{Agent, AgentEvent, AgentRequest, Role, StopReason};
use crate::config::Config;
use crate::git::{CommitId, Git};
use crate::state::TokenUsage;
use crate::tests as project_tests;

use super::budget::{session_cost_usd, BudgetCheck, BudgetReason, BudgetTracker};
use super::hooks::{run_hook, HookKind};
use super::plan::{GrindPlan, Hooks, PlanBudgets};
use super::prompt::PromptDoc;
use super::run_dir::{RunDir, RunPaths, SessionRecord, SessionStatus};
use super::scheduler::{Scheduler, SchedulerState};
use super::state::{build_state, RunStatus};
use super::worktree::{
    merge_scratchpad_into_run, parallel_safe_violation_summary, SessionWorktree,
};

/// Raw markdown standing-instruction block prepended to every grind prompt.
/// Embedded at compile time so users do not have to author it themselves and
/// so the markers can be located, updated, or stripped later without parsing
/// the whole prompt body.
const STANDING_INSTRUCTION_TEMPLATE: &str = include_str!("standing_instruction.md");

/// Marker tag wrapping the auto-injected sessions.md tail in the user-prompt
/// the agent receives. Stable so a downstream parser (or the agent itself) can
/// locate or strip it.
const SESSION_LOG_OPEN: &str = "<!-- pitboss:session-log -->";
const SESSION_LOG_CLOSE: &str = "<!-- /pitboss:session-log -->";
const SCRATCHPAD_OPEN: &str = "<!-- pitboss:scratchpad -->";
const SCRATCHPAD_CLOSE: &str = "<!-- /pitboss:scratchpad -->";

/// Number of trailing `sessions.md` lines auto-injected as context. Bounded so
/// long-running grinds don't quietly consume the agent's whole context window
/// with backlog.
const SESSION_LOG_TAIL_LINES: usize = 50;

/// Default per-session wall-clock cap, applied when the prompt frontmatter
/// leaves `max_session_seconds` unset.
const DEFAULT_SESSION_TIMEOUT: Duration = Duration::from_secs(60 * 30);

/// Standing-instruction text rendered into the agent's prompt body. Stable so
/// callers (and tests) can grep for it. Public for snapshot tests; not part of
/// the supported API surface.
pub fn standing_instruction_block() -> &'static str {
    STANDING_INSTRUCTION_TEMPLATE
}

/// Two-stage shutdown handle.
///
/// `drain` flips on the first Ctrl-C: the runner finishes in-flight sessions
/// cleanly, then exits. `abort` flips on the second Ctrl-C (or any
/// caller-driven cancel): the in-flight agents are cancelled and their
/// sessions are recorded with [`SessionStatus::Aborted`].
///
/// Cloning is cheap — both handles share state across clones.
#[derive(Debug, Clone)]
pub struct GrindShutdown {
    drain: Arc<AtomicBool>,
    abort: CancellationToken,
}

impl GrindShutdown {
    /// Build a shutdown handle with both signals cleared.
    pub fn new() -> Self {
        Self {
            drain: Arc::new(AtomicBool::new(false)),
            abort: CancellationToken::new(),
        }
    }

    /// Has the drain signal fired?
    pub fn is_draining(&self) -> bool {
        self.drain.load(Ordering::Relaxed)
    }

    /// Trip the drain signal. Idempotent.
    pub fn drain(&self) {
        self.drain.store(true, Ordering::Relaxed);
    }

    /// Trip the abort signal. Idempotent. Implicitly trips drain too so the
    /// outer loop exits regardless of which path saw the abort first.
    pub fn abort(&self) {
        self.drain.store(true, Ordering::Relaxed);
        self.abort.cancel();
    }

    /// Borrow the cancel token the agent dispatch should honor.
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.abort
    }
}

impl Default for GrindShutdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a [`GrindRunner::run`] call returned.
#[derive(Debug, Clone, PartialEq)]
pub enum GrindStopReason {
    /// The scheduler had no further prompts to dispatch.
    Completed,
    /// The drain signal fired between sessions; the in-flight session (if any)
    /// finished cleanly.
    Drained,
    /// The abort signal fired during a session; the session was recorded as
    /// [`SessionStatus::Aborted`].
    Aborted,
    /// A run-level budget tripped. Carries the exhausted reason for log /
    /// CLI output.
    BudgetExhausted(BudgetReason),
    /// The consecutive-failure escape valve fired (see
    /// [`crate::config::GrindConfig::consecutive_failure_limit`]).
    ConsecutiveFailureLimit {
        /// The configured limit that was just reached.
        limit: u32,
    },
}

/// Outcome of a full [`GrindRunner::run`] invocation.
#[derive(Debug, Clone)]
pub struct GrindRunOutcome {
    /// The id of the run on disk under `.pitboss/grind/<run-id>/`.
    pub run_id: String,
    /// The git branch the runner committed on.
    pub branch: String,
    /// All session records appended during this run, in completion order.
    pub sessions: Vec<SessionRecord>,
    /// Why the loop exited.
    pub stop_reason: GrindStopReason,
}

/// Grind orchestrator. See module docs.
pub struct GrindRunner<A: Agent, G: Git> {
    workspace: PathBuf,
    config: Arc<Config>,
    run_id: String,
    branch: String,
    plan: GrindPlan,
    scheduler: Scheduler,
    run_dir: RunDir,
    agent: Arc<A>,
    git: Arc<G>,
    next_seq: u32,
    budgets: PlanBudgets,
    consecutive_failure_limit: u32,
    started_at: DateTime<Utc>,
    initial_budget: super::budget::BudgetSnapshot,
    /// Serializes operations that touch the run branch from the main
    /// workspace's checkout: ff-merge of session branches, scratchpad merge,
    /// and the post-merge cleanup. Sequential sessions also hold this lock
    /// for the duration of their commit step so a parallel session waiting
    /// on the run branch can't interleave.
    run_branch_lock: Arc<TokioMutex<()>>,
}

impl<A: Agent + 'static, G: Git + 'static> GrindRunner<A, G> {
    /// Build a runner ready to dispatch its first session. Caller has already
    /// created the per-run branch and checked it out.
    ///
    /// `budgets` holds the run-wide caps already resolved from
    /// `pitboss.toml`'s `[grind.budgets]`, the plan's `PlanBudgets`, and any
    /// CLI overrides via [`crate::grind::resolve_budgets`].
    /// `consecutive_failure_limit` defaults to
    /// [`crate::config::GrindConfig::consecutive_failure_limit`] (`3`) and
    /// trips [`GrindStopReason::ConsecutiveFailureLimit`] once that many
    /// failed sessions land in a row.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        workspace: PathBuf,
        config: Config,
        run_id: String,
        branch: String,
        plan: GrindPlan,
        prompts: BTreeMap<String, PromptDoc>,
        run_dir: RunDir,
        agent: A,
        git: G,
        budgets: PlanBudgets,
        consecutive_failure_limit: u32,
    ) -> Self {
        let scheduler = Scheduler::new(plan.clone(), prompts);
        Self {
            workspace,
            config: Arc::new(config),
            run_id,
            branch,
            plan,
            scheduler,
            run_dir,
            agent: Arc::new(agent),
            git: Arc::new(git),
            next_seq: 1,
            budgets,
            consecutive_failure_limit,
            started_at: Utc::now(),
            initial_budget: super::budget::BudgetSnapshot::default(),
            run_branch_lock: Arc::new(TokioMutex::new(())),
        }
    }

    /// Build a runner from a previously persisted resume state. The caller
    /// has already validated the prompt set, opened the existing run
    /// directory, and re-checked out the run branch.
    ///
    /// `scheduler_state` is fed straight into [`Scheduler::with_state`] so
    /// the next [`Scheduler::next`] call deterministically returns the same
    /// prompt the original loop would have. `initial_budget` seeds the
    /// in-memory [`BudgetTracker`] so cumulative caps stay accurate across
    /// the resume boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn resume(
        workspace: PathBuf,
        config: Config,
        run_id: String,
        branch: String,
        plan: GrindPlan,
        prompts: BTreeMap<String, PromptDoc>,
        run_dir: RunDir,
        agent: A,
        git: G,
        budgets: PlanBudgets,
        consecutive_failure_limit: u32,
        scheduler_state: SchedulerState,
        initial_budget: super::budget::BudgetSnapshot,
        last_session_seq: u32,
        started_at: DateTime<Utc>,
    ) -> Self {
        let scheduler = Scheduler::with_state(plan.clone(), prompts, scheduler_state);
        Self {
            workspace,
            config: Arc::new(config),
            run_id,
            branch,
            plan,
            scheduler,
            run_dir,
            agent: Arc::new(agent),
            git: Arc::new(git),
            next_seq: last_session_seq.saturating_add(1),
            budgets,
            consecutive_failure_limit,
            started_at,
            initial_budget,
            run_branch_lock: Arc::new(TokioMutex::new(())),
        }
    }

    /// Workspace this runner is rooted at.
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Run id under `.pitboss/grind/<run-id>/`.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Branch the runner is committing on.
    pub fn branch(&self) -> &str {
        &self.branch
    }

    /// Current plan.
    pub fn plan(&self) -> &GrindPlan {
        &self.plan
    }

    /// Wall-clock instant the run was originally started. Resumed runs
    /// inherit the original start; fresh runs use the construction time.
    pub fn started_at(&self) -> DateTime<Utc> {
        self.started_at
    }

    /// Drive the loop. Returns once the scheduler exhausts, drain trips,
    /// abort trips, a run-level budget exhausts, or the consecutive-failure
    /// limit is reached.
    pub async fn run(&mut self, shutdown: GrindShutdown) -> Result<GrindRunOutcome> {
        let mut sessions: Vec<SessionRecord> = Vec::new();
        let mut tracker = BudgetTracker::from_snapshot(
            self.budgets.clone(),
            self.consecutive_failure_limit,
            self.initial_budget,
        );
        let mut stop_reason = GrindStopReason::Completed;
        let max_parallel = self.plan.max_parallel.max(1);
        let semaphore = Arc::new(Semaphore::new(max_parallel as usize));
        let mut tasks: JoinSet<Result<SessionRecord>> = JoinSet::new();
        let mut max_completed_seq: u32 = self.next_seq.saturating_sub(1);

        // Stamp the initial state.json so a resume target exists from the
        // first moment a run is on disk, even if the host process dies before
        // a single session lands.
        if let Err(e) = self.write_state(&tracker, max_completed_seq, RunStatus::Active) {
            warn!(
                run_id = %self.run_id,
                error = %format!("{e:#}"),
                "grind: initial state.json write failed"
            );
        }

        'outer: loop {
            if shutdown.is_draining() {
                stop_reason = if shutdown.cancel_token().is_cancelled() {
                    GrindStopReason::Aborted
                } else {
                    GrindStopReason::Drained
                };
                break;
            }

            // Pre-flight budget check so we never start a session that would
            // immediately blow a cap that was already reached by a previous
            // session.
            if let BudgetCheck::Exhausted(reason) = tracker.check() {
                info!(
                    run_id = %self.run_id,
                    reason = %reason,
                    "grind: BudgetExhausted (pre-dispatch)"
                );
                stop_reason = GrindStopReason::BudgetExhausted(reason);
                break;
            }

            if tracker.consecutive_failure_limit_reached() {
                warn!(
                    run_id = %self.run_id,
                    limit = self.consecutive_failure_limit,
                    "grind: consecutive-failure limit reached"
                );
                stop_reason = GrindStopReason::ConsecutiveFailureLimit {
                    limit: self.consecutive_failure_limit,
                };
                break;
            }

            let Some(prompt) = self.scheduler.next() else {
                break;
            };

            let seq = self.next_seq;
            self.next_seq += 1;

            // `parallel_safe: true` takes one permit; everything else takes
            // every permit so it can't overlap any concurrent session.
            let permits_needed = if prompt.meta.parallel_safe {
                1
            } else {
                max_parallel
            };

            let permit_outcome = self
                .acquire_permit(
                    semaphore.clone(),
                    permits_needed,
                    &mut tasks,
                    &mut sessions,
                    &mut tracker,
                    &mut max_completed_seq,
                    &shutdown,
                )
                .await?;
            let permit = match permit_outcome {
                AcquireOutcome::Got(p) => p,
                AcquireOutcome::ShutdownTripped => {
                    stop_reason = if shutdown.cancel_token().is_cancelled() {
                        GrindStopReason::Aborted
                    } else {
                        GrindStopReason::Drained
                    };
                    break 'outer;
                }
                AcquireOutcome::BudgetTripped(reason) => {
                    info!(
                        run_id = %self.run_id,
                        reason = %reason,
                        "grind: BudgetExhausted (mid-acquire)"
                    );
                    stop_reason = GrindStopReason::BudgetExhausted(reason);
                    break 'outer;
                }
                AcquireOutcome::ConsecutiveFailures => {
                    warn!(
                        run_id = %self.run_id,
                        limit = self.consecutive_failure_limit,
                        "grind: consecutive-failure limit reached"
                    );
                    stop_reason = GrindStopReason::ConsecutiveFailureLimit {
                        limit: self.consecutive_failure_limit,
                    };
                    break 'outer;
                }
            };

            // Now committed to dispatch — bump the scheduler so the next
            // pick reflects this run.
            self.scheduler.record_run(&prompt.meta.name);

            info!(
                run_id = %self.run_id,
                seq,
                prompt = %prompt.meta.name,
                parallel_safe = prompt.meta.parallel_safe,
                "grind: dispatching session"
            );

            let input = self
                .prepare_session_input(seq, prompt, permit, &shutdown)
                .await
                .with_context(|| format!("grind: preparing session {seq}"))?;
            tasks.spawn(run_session_task(input));
        }

        // Drain in-flight tasks.
        while let Some(res) = tasks.join_next().await {
            match res {
                Ok(Ok(rec)) => self.handle_completion(
                    rec,
                    &mut sessions,
                    &mut tracker,
                    &mut max_completed_seq,
                )?,
                Ok(Err(e)) => return Err(e),
                Err(je) => {
                    return Err(anyhow::anyhow!("session task panicked: {je}"));
                }
            }
        }

        // Aborted sessions in the drain phase should propagate as Aborted
        // stop reason if we weren't already Aborted/Drained.
        if matches!(stop_reason, GrindStopReason::Completed)
            && sessions.iter().any(|r| r.status == SessionStatus::Aborted)
        {
            stop_reason = GrindStopReason::Aborted;
        }

        // Stamp the terminal state.json.
        let terminal_status = match &stop_reason {
            GrindStopReason::Completed => RunStatus::Completed,
            GrindStopReason::Drained | GrindStopReason::Aborted => RunStatus::Aborted,
            GrindStopReason::BudgetExhausted(_)
            | GrindStopReason::ConsecutiveFailureLimit { .. } => RunStatus::Failed,
        };
        if let Err(e) = self.write_state(&tracker, max_completed_seq, terminal_status) {
            warn!(
                run_id = %self.run_id,
                error = %format!("{e:#}"),
                "grind: terminal state.json write failed"
            );
        }

        Ok(GrindRunOutcome {
            run_id: self.run_id.clone(),
            branch: self.branch.clone(),
            sessions,
            stop_reason,
        })
    }

    /// Wait until `permits_needed` permits are available, draining session
    /// completions in the meantime so each finished record lands in the
    /// session log + state.json before the next dispatch is committed.
    #[allow(clippy::too_many_arguments)]
    async fn acquire_permit(
        &self,
        semaphore: Arc<Semaphore>,
        permits_needed: u32,
        tasks: &mut JoinSet<Result<SessionRecord>>,
        sessions: &mut Vec<SessionRecord>,
        tracker: &mut BudgetTracker,
        max_completed_seq: &mut u32,
        shutdown: &GrindShutdown,
    ) -> Result<AcquireOutcome> {
        loop {
            if shutdown.is_draining() {
                return Ok(AcquireOutcome::ShutdownTripped);
            }

            if let Ok(p) = semaphore.clone().try_acquire_many_owned(permits_needed) {
                return Ok(AcquireOutcome::Got(p));
            }

            if tasks.is_empty() {
                // Permits requested exceed the configured ceiling — block
                // unconditionally so the test for `permits_needed >
                // max_parallel` still resolves rather than spinning.
                return Ok(AcquireOutcome::Got(
                    semaphore.acquire_many_owned(permits_needed).await?,
                ));
            }

            let Some(res) = tasks.join_next().await else {
                continue;
            };
            match res {
                Ok(Ok(rec)) => {
                    self.handle_completion(rec, sessions, tracker, max_completed_seq)?
                }
                Ok(Err(e)) => return Err(e),
                Err(je) => return Err(anyhow::anyhow!("session task panicked: {je}")),
            }

            if let BudgetCheck::Exhausted(reason) = tracker.check() {
                return Ok(AcquireOutcome::BudgetTripped(reason));
            }
            if tracker.consecutive_failure_limit_reached() {
                return Ok(AcquireOutcome::ConsecutiveFailures);
            }
        }
    }

    fn handle_completion(
        &self,
        record: SessionRecord,
        sessions: &mut Vec<SessionRecord>,
        tracker: &mut BudgetTracker,
        max_completed_seq: &mut u32,
    ) -> Result<()> {
        let seq = record.seq;
        self.run_dir
            .log()
            .append(&record)
            .with_context(|| format!("grind: appending session {seq} record to log"))?;
        tracker.record_session(&record);
        if seq > *max_completed_seq {
            *max_completed_seq = seq;
        }
        sessions.push(record);
        if let Err(e) = self.write_state(tracker, *max_completed_seq, RunStatus::Active) {
            warn!(
                run_id = %self.run_id,
                seq,
                error = %format!("{e:#}"),
                "grind: state.json write failed"
            );
        }
        Ok(())
    }

    async fn prepare_session_input(
        &self,
        seq: u32,
        prompt: PromptDoc,
        permit: OwnedSemaphorePermit,
        shutdown: &GrindShutdown,
    ) -> Result<SessionTaskInput<A, G>> {
        let transcript_path = self.run_dir.paths().transcript_for(seq);
        let summary_path = self
            .run_dir
            .paths()
            .root
            .join(format!(".pitboss-summary-{seq:04}.txt"));
        // Make sure the summary path is empty so a stale value from a prior
        // session can never be misread as the agent's current output.
        let _ = std::fs::remove_file(&summary_path);

        let session_log_tail = self
            .read_session_log_tail()
            .unwrap_or_else(|e| format!("(failed to read sessions.md: {e})"));
        let scratchpad_seed = self
            .run_dir
            .scratchpad()
            .read()
            .unwrap_or_else(|e| format!("(failed to read scratchpad: {e})"));

        let mut base_env: HashMap<String, String> = HashMap::new();
        base_env.insert("PITBOSS_RUN_ID".into(), self.run_id.clone());
        base_env.insert("PITBOSS_PROMPT_NAME".into(), prompt.meta.name.clone());
        base_env.insert(
            "PITBOSS_SUMMARY_FILE".into(),
            summary_path.display().to_string(),
        );
        base_env.insert("PITBOSS_SESSION_SEQ".into(), seq.to_string());

        let (workdir_for_agent, scratchpad_path_for_agent, worktree_opt) =
            if prompt.meta.parallel_safe {
                let wt = SessionWorktree::create(
                    &*self.git,
                    self.run_dir.paths(),
                    &self.run_id,
                    &self.branch,
                    seq,
                    &scratchpad_seed,
                )
                .await
                .with_context(|| format!("grind: creating worktree for session {seq}"))?;
                let path = wt.path().to_path_buf();
                let pad = wt.scratchpad_path().to_path_buf();
                (path, pad, Some(wt))
            } else {
                let pad = self.run_dir.scratchpad().path_for_agent().to_path_buf();
                (self.workspace.clone(), pad, None)
            };
        base_env.insert(
            "PITBOSS_SCRATCHPAD".into(),
            scratchpad_path_for_agent.display().to_string(),
        );

        Ok(SessionTaskInput {
            repo_root: self.workspace.clone(),
            workdir_for_agent,
            config: self.config.clone(),
            run_id: self.run_id.clone(),
            run_branch: self.branch.clone(),
            plan_hooks: self.plan.hooks.clone(),
            run_paths: self.run_dir.paths().clone(),
            transcript_path,
            summary_path,
            seq,
            prompt,
            agent: self.agent.clone(),
            repo_git: self.git.clone(),
            worktree: worktree_opt,
            run_branch_lock: self.run_branch_lock.clone(),
            shutdown: shutdown.clone(),
            permit,
            session_log_tail,
            scratchpad_seed,
            base_env,
        })
    }

    fn write_state(
        &self,
        tracker: &BudgetTracker,
        last_session_seq: u32,
        status: RunStatus,
    ) -> Result<()> {
        let prompt_names: Vec<String> = self
            .plan
            .prompts
            .iter()
            .map(|p| p.name.clone())
            .collect();
        let state = build_state(
            self.run_id.clone(),
            self.branch.clone(),
            self.plan.name.clone(),
            prompt_names,
            self.scheduler.state().clone(),
            tracker.snapshot(),
            last_session_seq,
            self.started_at,
            status,
        );
        state.write(self.run_dir.paths())
    }

    fn read_session_log_tail(&self) -> Result<String> {
        let path = &self.run_dir.paths().sessions_md;
        let raw = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(String::new()),
            Err(e) => {
                return Err(anyhow::Error::new(e).context(format!("grind: reading {:?}", path)))
            }
        };
        Ok(tail_lines(&raw, SESSION_LOG_TAIL_LINES))
    }
}

/// Resolution of [`GrindRunner::acquire_permit`]. Distinguishes the various
/// "could not acquire" branches so the main loop maps each to the right
/// [`GrindStopReason`].
enum AcquireOutcome {
    Got(OwnedSemaphorePermit),
    ShutdownTripped,
    BudgetTripped(BudgetReason),
    ConsecutiveFailures,
}

/// Bundle of everything a spawned session task needs. Owned/`Arc` so the
/// task body has no borrow back to the runner.
struct SessionTaskInput<A: Agent, G: Git> {
    /// Path of the main workspace. Used for `relative_to` on transcripts and
    /// for ff-merge into the run branch.
    repo_root: PathBuf,
    /// Where the agent should run — main workspace for sequential, the
    /// session worktree for parallel.
    workdir_for_agent: PathBuf,
    config: Arc<Config>,
    run_id: String,
    run_branch: String,
    plan_hooks: Hooks,
    run_paths: RunPaths,
    transcript_path: PathBuf,
    summary_path: PathBuf,
    seq: u32,
    prompt: PromptDoc,
    agent: Arc<A>,
    repo_git: Arc<G>,
    worktree: Option<SessionWorktree>,
    run_branch_lock: Arc<TokioMutex<()>>,
    shutdown: GrindShutdown,
    permit: OwnedSemaphorePermit,
    session_log_tail: String,
    /// Snapshot of the run-level scratchpad at session start. Embedded into
    /// the agent's user prompt and used as the seed for the per-session
    /// scratchpad merge in parallel mode.
    scratchpad_seed: String,
    base_env: HashMap<String, String>,
}

/// Body of one dispatched session. Returns the resulting [`SessionRecord`];
/// the runner appends it to the log and folds it into the budget tracker.
async fn run_session_task<A: Agent + 'static, G: Git + 'static>(
    input: SessionTaskInput<A, G>,
) -> Result<SessionRecord> {
    let SessionTaskInput {
        repo_root,
        workdir_for_agent,
        config,
        run_id,
        run_branch,
        plan_hooks,
        run_paths,
        transcript_path,
        summary_path,
        seq,
        prompt,
        agent,
        repo_git,
        worktree,
        run_branch_lock,
        shutdown,
        permit,
        session_log_tail,
        scratchpad_seed,
        base_env,
    } = input;

    let started_at = Utc::now();
    let transcript_rel = relative_to(&repo_root, &transcript_path);

    let hook_timeout = Duration::from_secs(config.grind.hook_timeout_secs.max(1));

    // ---- pre_session hook ---------------------------------------------
    let mut skip_dispatch_reason: Option<String> = None;
    if let Some(cmd) = plan_hooks.pre_session.as_deref() {
        let mut env = base_env.clone();
        env.insert("PITBOSS_SESSION_PROMPT".into(), prompt.meta.name.clone());
        let outcome = run_hook(
            HookKind::PreSession,
            cmd,
            &env,
            hook_timeout,
            &transcript_path,
        )
        .await;
        if !outcome.is_success() {
            warn!(
                run_id = %run_id,
                seq,
                outcome = %outcome.description(),
                "grind: pre_session hook failed; skipping dispatch"
            );
            skip_dispatch_reason = Some(format!("pre_session hook {}", outcome.description()));
        }
    }

    let mut status: SessionStatus;
    let mut summary: String;
    let mut commit: Option<CommitId> = None;
    let mut tokens: TokenUsage = TokenUsage::default();
    let mut cost_usd: f64 = 0.0;
    let ended_at: DateTime<Utc>;

    if let Some(reason) = skip_dispatch_reason {
        status = SessionStatus::Error;
        summary = reason;
        ended_at = Utc::now();
    } else {
        let user_prompt = compose_user_prompt(
            STANDING_INSTRUCTION_TEMPLATE,
            &session_log_tail,
            &scratchpad_seed,
            &prompt.body,
        );

        let timeout = prompt
            .meta
            .max_session_seconds
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_SESSION_TIMEOUT);

        let model = config.models.implementer.clone();
        let request = AgentRequest {
            role: Role::Implementer,
            model: model.clone(),
            system_prompt: String::new(),
            user_prompt,
            workdir: workdir_for_agent.clone(),
            log_path: transcript_path.clone(),
            timeout,
            env: base_env.clone(),
        };

        let mut summary_override: Option<String> = None;
        let dispatch = match tokio::time::timeout(
            timeout,
            dispatch_agent(&*agent, request, shutdown.cancel_token()),
        )
        .await
        {
            Ok(res) => res?,
            Err(_) => {
                warn!(
                    run_id = %run_id,
                    seq,
                    timeout_secs = timeout.as_secs(),
                    "grind: per-prompt timeout fired"
                );
                summary_override = Some(format!(
                    "session exceeded max_session_seconds ({}s)",
                    timeout.as_secs()
                ));
                AgentDispatch {
                    stop_reason: StopReason::Timeout,
                    tokens: TokenUsage::default(),
                }
            }
        };
        ended_at = Utc::now();

        status = match &dispatch.stop_reason {
            StopReason::Completed => SessionStatus::Ok,
            StopReason::Timeout => SessionStatus::Timeout,
            StopReason::Cancelled => SessionStatus::Aborted,
            StopReason::Error(_) => SessionStatus::Error,
        };

        cost_usd = session_cost_usd(
            &config,
            &model,
            dispatch.tokens.input,
            dispatch.tokens.output,
        );

        if status == SessionStatus::Ok {
            if let Some(cap) = prompt.meta.max_session_cost_usd {
                if cost_usd > cap {
                    warn!(
                        run_id = %run_id,
                        seq,
                        cost = cost_usd,
                        cap,
                        "grind: per-prompt max_session_cost_usd exceeded"
                    );
                    status = SessionStatus::Error;
                    summary_override = Some(format!(
                        "session exceeded max_session_cost_usd: ${cost_usd:.4} > ${cap:.4}"
                    ));
                }
            }
        }

        summary = match summary_override {
            Some(s) => s,
            None => read_summary_or_fallback(&summary_path),
        };

        if status == SessionStatus::Ok && prompt.meta.verify {
            status = verify_session(
                seq,
                &prompt,
                &workdir_for_agent,
                config.tests.command.as_deref(),
                &transcript_path,
            )
            .await?;
        }

        // Commit + stash. Sequential and parallel sessions share the same
        // commit / stash logic but run it against different git handles —
        // sequential against the workspace-rooted runner git, parallel
        // against a worktree-scoped ShellGit owned by the SessionWorktree.
        // Parallel sessions also hold the run-branch lock for the entire
        // sync → commit → ff-merge dance so a sibling session cannot
        // interleave between the steps.
        if let Some(wt) = &worktree {
            let g = wt.worktree_git();
            let _guard = run_branch_lock.lock().await;

            // Step 1 — sync the worktree's session branch to the current
            // run-branch tip. When the session was created run_branch was at
            // commit A; another parallel session may have advanced it to A'
            // since. Replaying that fast-forward inside the worktree is what
            // makes the eventual run-branch ff-merge possible. If the FF
            // refuses (because the agent's uncommitted edits would be
            // overwritten by the incoming run-branch tip), the prompt
            // violated its `parallel_safe: true` claim — we mark the session
            // Error and skip the commit / merge entirely.
            let mut sync_ok = true;
            if status == SessionStatus::Ok || status == SessionStatus::Error {
                if let Err(e) = g.merge_ff_only(&run_branch).await {
                    warn!(
                        run_id = %run_id,
                        seq,
                        error = %format!("{e:#}"),
                        prompt = %prompt.meta.name,
                        "grind: parallel_safe contract violation (worktree sync)"
                    );
                    status = SessionStatus::Error;
                    summary = parallel_safe_violation_summary(&prompt.meta.name);
                    sync_ok = false;
                }
            }

            // Step 2 — commit on top of the synced HEAD. The per-session
            // scratchpad lives at the worktree root and must stay out of
            // the run-branch tree; the runner merges it back into the
            // run-level scratchpad below.
            let pitboss_rel = Path::new(".pitboss");
            let scratchpad_rel = Path::new("scratchpad.md");
            let parallel_exclusions: [&Path; 2] = [pitboss_rel, scratchpad_rel];
            if sync_ok {
                commit = match status {
                    SessionStatus::Ok | SessionStatus::Error => {
                        try_commit_session(g, seq, &prompt, &run_id, &parallel_exclusions).await?
                    }
                    _ => None,
                };
            }

            // Step 3 — fast-forward the run branch to the session tip. The
            // sync above guarantees this is a strict descendant unless
            // run_branch raced forward between sync and merge — but the
            // run_branch_lock prevents that.
            if sync_ok && commit.is_some() {
                if let Err(e) = repo_git.merge_ff_only(wt.branch()).await {
                    warn!(
                        run_id = %run_id,
                        seq,
                        error = %format!("{e:#}"),
                        prompt = %prompt.meta.name,
                        "grind: parallel_safe contract violation (run-branch ff)"
                    );
                    status = SessionStatus::Error;
                    summary = parallel_safe_violation_summary(&prompt.meta.name);
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
                let stash_label = format!("grind/{}/session-{:04}-leftover", run_id, seq);
                match g.stash_push(&stash_label, &parallel_exclusions).await {
                    Ok(true) => {
                        warn!(
                            run_id = %run_id,
                            seq,
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
                            seq,
                            error = %format!("{e:#}"),
                            "grind: stash_push failed (parallel)"
                        );
                    }
                }
            }
            drop(_guard);
        } else {
            // Sequential: hold the run-branch lock so a concurrent parallel
            // session cannot ff-merge while we're staging / committing.
            let _guard = run_branch_lock.lock().await;
            let pitboss_rel = Path::new(".pitboss");
            let sequential_exclusions: [&Path; 1] = [pitboss_rel];
            commit = match status {
                SessionStatus::Ok | SessionStatus::Error => {
                    try_commit_session(&*repo_git, seq, &prompt, &run_id, &sequential_exclusions)
                        .await?
                }
                _ => None,
            };
            let stash_label = format!("grind/{}/session-{:04}-leftover", run_id, seq);
            match repo_git.stash_push(&stash_label, &sequential_exclusions).await {
                Ok(true) => {
                    warn!(
                        run_id = %run_id,
                        seq,
                        stash = %stash_label,
                        "grind: leftover changes stashed"
                    );
                    if status == SessionStatus::Ok {
                        status = SessionStatus::Dirty;
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    warn!(
                        run_id = %run_id,
                        seq,
                        error = %format!("{e:#}"),
                        "grind: stash_push failed"
                    );
                }
            }
        }

        tokens = dispatch.tokens;
    }

    // ---- per-session scratchpad merge (parallel only) ----------------
    if let Some(wt) = &worktree {
        // The agent wrote into the per-session scratchpad view; fold those
        // edits back into the run-level scratchpad. Held under the run
        // branch lock to avoid interleaving with another session's merge.
        // Always attempted (even on Error) so an agent that produced useful
        // notes before failing doesn't silently lose them.
        let session_view = std::fs::read_to_string(wt.scratchpad_path()).unwrap_or_default();
        let _guard = run_branch_lock.lock().await;
        if let Err(e) = merge_scratchpad_into_run(
            &run_paths.scratchpad,
            &session_view,
            wt.scratchpad_seed(),
            seq,
        ) {
            warn!(
                run_id = %run_id,
                seq,
                error = %format!("{e:#}"),
                "grind: scratchpad merge failed"
            );
        }
        drop(_guard);
    }

    // ---- worktree teardown -------------------------------------------
    if let Some(wt) = worktree {
        let cleaned = if status == SessionStatus::Ok {
            wt.cleanup(&*repo_git).await
        } else {
            // Leave the worktree behind under worktrees/failed/ for triage.
            wt.quarantine(&*repo_git).await.map(|_| ())
        };
        if let Err(e) = cleaned {
            warn!(
                run_id = %run_id,
                seq,
                error = %format!("{e:#}"),
                "grind: worktree teardown failed"
            );
        }
    }

    // ---- post_session / on_failure hooks ------------------------------
    let mut hook_env = base_env.clone();
    hook_env.insert("PITBOSS_SESSION_PROMPT".into(), prompt.meta.name.clone());
    hook_env.insert("PITBOSS_SESSION_STATUS".into(), status.as_str().to_string());
    hook_env.insert("PITBOSS_SESSION_SUMMARY".into(), summary.clone());
    if let Some(cmd) = plan_hooks.post_session.as_deref() {
        let _ = run_hook(
            HookKind::PostSession,
            cmd,
            &hook_env,
            hook_timeout,
            &transcript_path,
        )
        .await;
    }
    if status != SessionStatus::Ok {
        if let Some(cmd) = plan_hooks.on_failure.as_deref() {
            let _ = run_hook(
                HookKind::OnFailure,
                cmd,
                &hook_env,
                hook_timeout,
                &transcript_path,
            )
            .await;
        }
    }

    // The run_branch is unused beyond this point; consume it to silence the
    // unused-variable warning while keeping the field on `SessionTaskInput`
    // for symmetry with other run-level handles.
    let _ = run_branch;

    drop(permit);

    Ok(SessionRecord {
        seq,
        run_id,
        prompt: prompt.meta.name.clone(),
        started_at,
        ended_at,
        status,
        summary: Some(summary),
        commit,
        tokens,
        cost_usd,
        transcript_path: transcript_rel,
    })
}

/// Stage and commit any code changes the session produced. Returns the new
/// commit id, or `None` if there was nothing code-side to commit (e.g., the
/// agent only edited `.pitboss/`).
///
/// `exclude` is the per-call exclusion set forwarded to
/// [`Git::stage_changes`]. Sequential sessions pass just `.pitboss/`;
/// parallel sessions also pass the per-session `scratchpad.md` so the
/// worktree-rooted scratchpad never lands in the run-branch tree (it lives
/// outside git's history; pitboss merges it back via [`merge_scratchpad_into_run`]).
async fn try_commit_session<G: Git + ?Sized>(
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

/// Auto-detect the project's test runner and run it once. Returns
/// [`SessionStatus::Ok`] when tests pass and [`SessionStatus::Error`] when
/// they fail. Reuse of the existing fixer cycle is deferred (see
/// `deferred.md`).
async fn verify_session(
    seq: u32,
    prompt: &PromptDoc,
    workdir: &Path,
    override_command: Option<&str>,
    transcript_path: &Path,
) -> Result<SessionStatus> {
    let Some(runner) = project_tests::detect(workdir, override_command) else {
        debug!(
            seq,
            prompt = %prompt.meta.name,
            "grind: verify requested but no test runner detected"
        );
        return Ok(SessionStatus::Ok);
    };
    let verify_log = transcript_path.with_extension("verify.log");
    let outcome = runner
        .run(verify_log)
        .await
        .with_context(|| format!("grind: verify run for session {seq}"))?;
    if outcome.passed {
        Ok(SessionStatus::Ok)
    } else {
        warn!(
            seq,
            prompt = %prompt.meta.name,
            summary = %outcome.summary,
            "grind: verify failed"
        );
        Ok(SessionStatus::Error)
    }
}

async fn dispatch_agent<A: Agent + ?Sized>(
    agent: &A,
    request: AgentRequest,
    cancel: &CancellationToken,
) -> Result<AgentDispatch> {
    let (events_tx, mut events_rx) = mpsc::channel::<AgentEvent>(64);
    let cancel_clone = cancel.clone();
    let drain_task = tokio::spawn(async move {
        while events_rx.recv().await.is_some() {
            // Phase 13 wires these into the TUI; for now we drop them so
            // the channel doesn't apply backpressure on the agent.
        }
    });

    let outcome = agent
        .run(request, events_tx, cancel_clone)
        .await
        .context("grind: agent dispatch failed")?;
    let _ = drain_task.await;

    Ok(AgentDispatch {
        stop_reason: outcome.stop_reason,
        tokens: outcome.tokens,
    })
}

struct AgentDispatch {
    stop_reason: StopReason,
    tokens: TokenUsage,
}

/// Compose the full user-prompt body the agent sees: standing instruction,
/// session-log tail, scratchpad snapshot, and the user-authored prompt body.
/// Each block is fenced with stable markers so a downstream tool (or another
/// agent) can locate or strip individual sections without reparsing.
pub fn compose_user_prompt(
    standing_instruction: &str,
    session_log: &str,
    scratchpad: &str,
    prompt_body: &str,
) -> String {
    let mut out = String::with_capacity(
        standing_instruction.len() + session_log.len() + scratchpad.len() + prompt_body.len() + 256,
    );
    out.push_str(standing_instruction.trim_end_matches('\n'));
    out.push_str("\n\n");
    out.push_str(SESSION_LOG_OPEN);
    out.push('\n');
    if session_log.trim().is_empty() {
        out.push_str("(no prior sessions)\n");
    } else {
        out.push_str(session_log.trim_end_matches('\n'));
        out.push('\n');
    }
    out.push_str(SESSION_LOG_CLOSE);
    out.push_str("\n\n");
    out.push_str(SCRATCHPAD_OPEN);
    out.push('\n');
    if scratchpad.trim().is_empty() {
        out.push_str("(scratchpad is empty)\n");
    } else {
        out.push_str(scratchpad.trim_end_matches('\n'));
        out.push('\n');
    }
    out.push_str(SCRATCHPAD_CLOSE);
    out.push_str("\n\n");
    out.push_str(prompt_body.trim_start_matches('\n'));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn read_summary_or_fallback(path: &Path) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        Ok(_) => {
            warn!(path = %path.display(), "grind: agent left $PITBOSS_SUMMARY_FILE empty");
            "(no summary provided)".to_string()
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!(path = %path.display(), "grind: agent did not write $PITBOSS_SUMMARY_FILE");
            "(no summary provided)".to_string()
        }
        Err(e) => {
            warn!(
                path = %path.display(),
                error = %format!("{e:#}"),
                "grind: failed to read $PITBOSS_SUMMARY_FILE"
            );
            "(no summary provided)".to_string()
        }
    }
}

fn relative_to(base: &Path, full: &Path) -> PathBuf {
    full.strip_prefix(base)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| full.to_path_buf())
}

fn tail_lines(text: &str, n: usize) -> String {
    let mut buf: Vec<&str> = Vec::new();
    for line in text.lines() {
        buf.push(line);
        if buf.len() > n {
            buf.remove(0);
        }
    }
    let mut out = buf.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Format the per-run branch name. Stable so worktree branches in
/// [`super::worktree::session_branch_name`] derive their names from the same
/// prefix.
pub fn run_branch_name(run_id: &str) -> String {
    format!("pitboss/grind/{run_id}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_branch_name_uses_canonical_prefix() {
        assert_eq!(
            run_branch_name("20260430T120000Z-1a2b"),
            "pitboss/grind/20260430T120000Z-1a2b"
        );
    }

    #[test]
    fn compose_user_prompt_stamps_all_three_marker_blocks() {
        let out = compose_user_prompt("STANDING", "session log row", "scratchpad note", "body");
        assert!(out.starts_with("STANDING\n\n"));
        assert!(out.contains(SESSION_LOG_OPEN));
        assert!(out.contains(SESSION_LOG_CLOSE));
        assert!(out.contains(SCRATCHPAD_OPEN));
        assert!(out.contains(SCRATCHPAD_CLOSE));
        assert!(out.contains("session log row"));
        assert!(out.contains("scratchpad note"));
        assert!(out.contains("body"));
        // Body lands at the end.
        let body_pos = out.find("body").unwrap();
        assert!(body_pos > out.find(SCRATCHPAD_CLOSE).unwrap());
    }

    #[test]
    fn compose_user_prompt_substitutes_empty_blocks() {
        let out = compose_user_prompt("STANDING", "", "", "do the thing");
        assert!(out.contains("(no prior sessions)"));
        assert!(out.contains("(scratchpad is empty)"));
        assert!(out.contains("do the thing"));
    }

    #[test]
    fn standing_instruction_block_carries_markers() {
        let s = standing_instruction_block();
        assert!(
            s.contains("<!-- pitboss:standing-instruction:start -->"),
            "instruction missing start marker"
        );
        assert!(
            s.contains("<!-- pitboss:standing-instruction:end -->"),
            "instruction missing end marker"
        );
        assert!(s.contains("$PITBOSS_SUMMARY_FILE"));
        assert!(s.contains("$PITBOSS_SCRATCHPAD"));
        assert!(s.contains("$PITBOSS_RUN_ID"));
    }

    #[test]
    fn tail_lines_keeps_last_n() {
        let text = "a\nb\nc\nd\ne\n";
        assert_eq!(tail_lines(text, 3), "c\nd\ne\n");
        assert_eq!(tail_lines(text, 10), "a\nb\nc\nd\ne\n");
        assert_eq!(tail_lines("", 5), "");
    }

    #[test]
    fn read_summary_falls_back_when_file_is_missing_or_empty() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.txt");
        assert_eq!(read_summary_or_fallback(&missing), "(no summary provided)");

        let empty = dir.path().join("empty.txt");
        std::fs::write(&empty, "   \n  \n").unwrap();
        assert_eq!(read_summary_or_fallback(&empty), "(no summary provided)");

        let real = dir.path().join("real.txt");
        std::fs::write(&real, "did the thing\n").unwrap();
        assert_eq!(read_summary_or_fallback(&real), "did the thing");
    }

    #[test]
    fn shutdown_drain_and_abort_signals_propagate() {
        let s = GrindShutdown::new();
        assert!(!s.is_draining());
        assert!(!s.cancel_token().is_cancelled());

        s.drain();
        assert!(s.is_draining());
        assert!(!s.cancel_token().is_cancelled());

        let s2 = s.clone();
        s2.abort();
        assert!(s.is_draining());
        assert!(s.cancel_token().is_cancelled());
    }
}
