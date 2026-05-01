//! Phase 08 acceptance: each documented `pitboss grind` exit code is produced
//! by the corresponding scenario.
//!
//! Tests drive [`GrindRunner`] directly via a scriptable [`MockAgent`], then
//! map the resolved [`GrindStopReason`] + session list to an [`ExitCode`]
//! through `cli::grind::classify_outcome` (the same translation `pitboss
//! grind` uses on the CLI side). [`ExitCode::FailedToStart`] is produced
//! upstream of the runner in `cli::grind::run`; coverage for that path is
//! tracked in `deferred.md`.

#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::{mpsc, Semaphore};
use tokio_util::sync::CancellationToken;

use pitboss::agent::{Agent, AgentEvent, AgentOutcome, AgentRequest, StopReason};
use pitboss::cli::grind::classify_outcome;
use pitboss::config::Config;
use pitboss::git::{Git, ShellGit};
use pitboss::grind::{
    default_plan_from_dir, ExitCode, GrindPlan, GrindRunner, GrindShutdown, GrindStopReason,
    PlanBudgets, PromptDoc, PromptMeta, PromptSource, RunDir, SessionStatus,
};
use pitboss::state::{RoleUsage, TokenUsage};

const RUN_ID: &str = "20260430T180000Z-exit";

/// Pluggable mock that returns a per-invocation script: status + token usage
/// plus an optional sleep before completing. Each tick of the script
/// consumes one entry; if the script runs dry the agent reports
/// `StopReason::Completed` with zero tokens.
///
/// Sleeps and any optional gate await race the dispatch [`CancellationToken`]
/// so a test that fires `shutdown.abort()` mid-dispatch sees the agent
/// short-circuit with [`StopReason::Cancelled`] — the same behavior the
/// production `ClaudeCodeAgent` exhibits when killed by a signal.
#[derive(Clone)]
struct ScriptedAgent {
    name: String,
    script: Arc<Vec<ScriptStep>>,
    invocations: Arc<AtomicU32>,
    /// Optional pair of channels for tests that need a deterministic sync
    /// point per session: the agent posts the 1-based dispatch number on
    /// `started_tx` after writing summary/marker, then awaits one permit on
    /// `proceed` (cancellable via `cancel`). Mirrors the gated MockAgent in
    /// `tests/grind_smoke.rs` so a test can pin the abort to a specific
    /// dispatch.
    gate: Option<(mpsc::UnboundedSender<u32>, Arc<Semaphore>)>,
}

#[derive(Clone)]
struct ScriptStep {
    /// Sleep before the agent reports completion. The grind runner wraps the
    /// dispatch in `tokio::time::timeout(prompt.max_session_seconds)`, so a
    /// long sleep here lets a tight per-prompt timeout fire.
    sleep: Option<Duration>,
    /// What the agent reports back when it does finish (independent of any
    /// outer timeout).
    stop: StopReason,
    /// Tokens billed for this dispatch. The runner prices these via
    /// `Config::budgets.pricing` keyed by `Config::models.implementer`.
    tokens: TokenUsage,
    /// When `true`, the mock writes a real file under `src/` so the session
    /// produces a commit. The exit-code tests don't care about the commit;
    /// disabled by default to keep the test tree small.
    write_commit_marker: bool,
    /// Optional summary text to write into `$PITBOSS_SUMMARY_FILE`. When
    /// `None`, the mock writes a generic placeholder.
    summary: Option<String>,
}

impl Default for ScriptStep {
    fn default() -> Self {
        Self {
            sleep: None,
            stop: StopReason::Completed,
            tokens: TokenUsage::default(),
            write_commit_marker: false,
            summary: None,
        }
    }
}

impl ScriptedAgent {
    fn new(steps: Vec<ScriptStep>, invocations: Arc<AtomicU32>) -> Self {
        Self {
            name: "scripted".into(),
            script: Arc::new(steps),
            invocations,
            gate: None,
        }
    }

    fn gated(
        steps: Vec<ScriptStep>,
        invocations: Arc<AtomicU32>,
        started_tx: mpsc::UnboundedSender<u32>,
        proceed: Arc<Semaphore>,
    ) -> Self {
        let mut me = Self::new(steps, invocations);
        me.gate = Some((started_tx, proceed));
        me
    }
}

#[async_trait]
impl Agent for ScriptedAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(
        &self,
        req: AgentRequest,
        events: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
    ) -> Result<AgentOutcome> {
        let n = self.invocations.fetch_add(1, Ordering::SeqCst) as usize;
        let step = self
            .script
            .get(n)
            .cloned()
            .unwrap_or_else(ScriptStep::default);

        if let Some(parent) = req.log_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(
            &req.log_path,
            format!("[scripted] dispatch {} ({})\n", n + 1, self.name).as_bytes(),
        )
        .ok();

        if step.write_commit_marker {
            let seq = req
                .env
                .get("PITBOSS_SESSION_SEQ")
                .map(String::as_str)
                .unwrap_or("0");
            let prompt = req
                .env
                .get("PITBOSS_PROMPT_NAME")
                .map(String::as_str)
                .unwrap_or("p");
            let marker = req
                .workdir
                .join(format!("src/scripted_session_{:0>4}_{}.rs", seq, prompt));
            if let Some(parent) = marker.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(
                &marker,
                format!("// scripted session {seq} for {prompt}\n").as_bytes(),
            )
            .ok();
        }

        // Always honor the summary contract so a session that "succeeds" lands
        // a non-empty summary in the record.
        if let Some(summary_file) = req.env.get("PITBOSS_SUMMARY_FILE") {
            let body = step
                .summary
                .clone()
                .unwrap_or_else(|| format!("scripted dispatch {}", n + 1));
            std::fs::write(summary_file, body).ok();
        }

        // Sleep is cancellable so a mid-dispatch abort short-circuits to
        // `StopReason::Cancelled` instead of waiting out the full duration
        // (or being killed by the runner's outer timeout, which would
        // surface as `Timeout` and miss the abort path entirely).
        if let Some(d) = step.sleep {
            tokio::select! {
                _ = tokio::time::sleep(d) => {}
                _ = cancel.cancelled() => {
                    return Ok(AgentOutcome {
                        exit_code: -1,
                        stop_reason: StopReason::Cancelled,
                        tokens: TokenUsage::default(),
                        log_path: req.log_path,
                    });
                }
            }
        }

        // Optional gate: signal to the test that dispatch reached the end of
        // the agent body, then block on a permit. The wait races the cancel
        // token so a test can deterministically prove that an abort fired
        // *while the agent was inside this dispatch* (not before it started
        // and not after it returned).
        if let Some((started_tx, proceed)) = &self.gate {
            let _ = started_tx.send(n as u32 + 1);
            let permit = tokio::select! {
                p = proceed.clone().acquire_owned() => p,
                _ = cancel.cancelled() => {
                    return Ok(AgentOutcome {
                        exit_code: -1,
                        stop_reason: StopReason::Cancelled,
                        tokens: TokenUsage::default(),
                        log_path: req.log_path,
                    });
                }
            }
            .expect("proceed semaphore closed unexpectedly");
            permit.forget();
        }

        let _ = events
            .send(AgentEvent::Stdout(format!("[scripted] dispatch {}", n + 1)))
            .await;

        let exit_code = match &step.stop {
            StopReason::Completed => 0,
            _ => -1,
        };
        Ok(AgentOutcome {
            exit_code,
            stop_reason: step.stop,
            tokens: step.tokens,
            log_path: req.log_path,
        })
    }
}

fn fake_prompt(name: &str, max_session_seconds: Option<u64>) -> PromptDoc {
    fake_prompt_with(name, max_session_seconds, None)
}

fn fake_prompt_with(
    name: &str,
    max_session_seconds: Option<u64>,
    max_runs: Option<u32>,
) -> PromptDoc {
    PromptDoc {
        meta: PromptMeta {
            name: name.into(),
            description: format!("desc for {name}"),
            weight: 1,
            every: 1,
            max_runs,
            verify: false,
            parallel_safe: false,
            tags: vec![],
            max_session_seconds,
            max_session_cost_usd: None,
        },
        body: format!("body for {name}"),
        source_path: PathBuf::from(format!("/fixture/{name}.md")),
        source_kind: PromptSource::Project,
    }
}

fn lookup(prompts: &[PromptDoc]) -> BTreeMap<String, PromptDoc> {
    prompts
        .iter()
        .map(|p| (p.meta.name.clone(), p.clone()))
        .collect()
}

fn init_git_repo(dir: &Path) {
    let status = Command::new("git")
        .args(["-c", "init.defaultBranch=main", "init", "-q"])
        .arg(dir)
        .status()
        .expect("git init");
    assert!(status.success());
    for (k, v) in [
        ("user.name", "pitboss-test"),
        ("user.email", "pitboss@test"),
    ] {
        Command::new("git")
            .args(["-C"])
            .arg(dir)
            .args(["config", k, v])
            .status()
            .unwrap();
    }
    let status = Command::new("git")
        .args(["-C"])
        .arg(dir)
        .args(["commit", "--allow-empty", "-m", "seed", "-q"])
        .status()
        .expect("seed commit");
    assert!(status.success());
}

#[allow(clippy::too_many_arguments)]
async fn build_runner(
    workspace: &Path,
    branch: &str,
    prompts: Vec<PromptDoc>,
    plan: GrindPlan,
    agent: ScriptedAgent,
    budgets: PlanBudgets,
    consecutive_failure_limit: u32,
) -> GrindRunner<ScriptedAgent, ShellGit> {
    init_git_repo(workspace);
    let git = ShellGit::new(workspace);
    git.create_branch(branch).await.unwrap();
    git.checkout(branch).await.unwrap();
    let run_dir = RunDir::create(workspace, RUN_ID).expect("create run dir");
    let runner_git = ShellGit::new(workspace);
    GrindRunner::new(
        workspace.to_path_buf(),
        Config::default(),
        RUN_ID.to_string(),
        branch.to_string(),
        plan,
        lookup(&prompts),
        run_dir,
        agent,
        runner_git,
        budgets,
        consecutive_failure_limit,
    )
}

/// Exit code 0: every session reported `Ok` and no budget tripped. Uses
/// per-prompt `max_runs=1` so the scheduler exhausts naturally — that way
/// the runner exits via [`GrindStopReason::Completed`] (not
/// `BudgetExhausted`), and the classifier reports `Success`.
#[tokio::test]
async fn success_when_all_sessions_resolve_ok() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![
        fake_prompt_with("alpha", None, Some(1)),
        fake_prompt_with("bravo", None, Some(1)),
    ];
    let plan = default_plan_from_dir(&prompts);
    let script = vec![
        ScriptStep {
            stop: StopReason::Completed,
            tokens: TokenUsage::default(),
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Completed,
            tokens: TokenUsage::default(),
            ..Default::default()
        },
    ];
    let agent = ScriptedAgent::new(script, invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets::default(),
        3,
    )
    .await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(outcome.sessions.len(), 2);
    assert!(outcome
        .sessions
        .iter()
        .all(|r| r.status == SessionStatus::Ok));
    assert_eq!(outcome.stop_reason, GrindStopReason::Completed);
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::Success);
}

/// Exit code 1: at least one session resolved as `Error` or `Timeout` but the
/// run otherwise completed naturally (scheduler exhausted, no budget hit).
#[tokio::test]
async fn mixed_failures_when_some_sessions_fail() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![
        fake_prompt_with("alpha", None, Some(1)),
        fake_prompt_with("bravo", None, Some(1)),
    ];
    let plan = default_plan_from_dir(&prompts);
    let script = vec![
        ScriptStep {
            stop: StopReason::Completed,
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Error("synthetic failure".into()),
            ..Default::default()
        },
    ];
    let agent = ScriptedAgent::new(script, invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets::default(),
        // Disable the consecutive-failure escape valve so we don't mask the
        // intent of this test (one error mixed with one success).
        0,
    )
    .await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(outcome.sessions.len(), 2);
    assert_eq!(outcome.stop_reason, GrindStopReason::Completed);
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::MixedFailures);
}

/// Exit code 2: the user aborted (here, simulated by tripping the abort
/// signal between sessions). We pre-trip the signal so the runner exits with
/// `Aborted` even before dispatching session 1.
#[tokio::test]
async fn aborted_when_shutdown_abort_fires() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt("alpha", None)];
    let plan = default_plan_from_dir(&prompts);
    let agent = ScriptedAgent::new(vec![], invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets::default(),
        3,
    )
    .await;
    let shutdown = GrindShutdown::new();
    shutdown.abort();
    let outcome = runner.run(shutdown).await.unwrap();
    assert_eq!(outcome.stop_reason, GrindStopReason::Aborted);
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::Aborted);
}

/// Exit code 2 again, but this time the abort fires *while the agent is
/// running* — the in-flight cancellation path the previous test could not
/// reach. Uses the gated [`ScriptedAgent`] so the test holds the agent inside
/// dispatch until `shutdown.abort()` lands; the agent's gate-await races the
/// cancel token, returns `StopReason::Cancelled`, and the runner records the
/// session as `SessionStatus::Aborted`.
#[tokio::test]
async fn aborted_when_shutdown_abort_fires_during_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt("alpha", None)];
    let plan = default_plan_from_dir(&prompts);

    // One step: agent reports `Completed` if the gate releases before the
    // abort, but the gated path lets the test guarantee the abort wins by
    // never adding a permit.
    let script = vec![ScriptStep {
        stop: StopReason::Completed,
        ..Default::default()
    }];

    let (started_tx, mut started_rx) = mpsc::unbounded_channel::<u32>();
    let proceed = Arc::new(Semaphore::new(0));
    let agent = ScriptedAgent::gated(script, invocations.clone(), started_tx, proceed.clone());
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets::default(),
        3,
    )
    .await;

    let shutdown = GrindShutdown::new();
    let runner_shutdown = shutdown.clone();
    let runner_handle = tokio::spawn(async move { runner.run(runner_shutdown).await.unwrap() });

    // Wait until dispatch 1 reports it has reached the gate, *then* abort.
    // The gate-await is racing the cancel token in the agent body, so the
    // abort lands while the agent is genuinely in flight.
    let n = started_rx.recv().await.expect("dispatch 1 start signal");
    assert_eq!(n, 1, "expected start signal from dispatch 1");
    shutdown.abort();

    let outcome = runner_handle.await.expect("runner task panicked");

    // The session was in flight when the abort fired, so it lands as
    // `Aborted` (not `Ok`, not `Error`, not `Timeout`).
    assert_eq!(
        outcome.sessions.len(),
        1,
        "expected exactly one session record, got {}",
        outcome.sessions.len()
    );
    assert_eq!(
        outcome.sessions[0].status,
        SessionStatus::Aborted,
        "in-flight cancellation must land as Aborted: {:?}",
        outcome.sessions[0]
    );
    assert_eq!(outcome.stop_reason, GrindStopReason::Aborted);
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::Aborted);
    assert_eq!(
        invocations.load(Ordering::SeqCst),
        1,
        "exactly one dispatch should have fired"
    );
}

/// Exit code 3: a run-level budget tripped. `--max-iterations 2` halts after
/// session 2 even though the scheduler would dispatch indefinitely.
#[tokio::test]
async fn budget_exhausted_halts_after_max_iterations() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt("alpha", None)];
    let plan = default_plan_from_dir(&prompts);
    // Provide enough script steps to demonstrate the cap really stops the
    // loop — if the cap isn't enforced, the test would dispatch a third
    // session.
    let script = vec![
        ScriptStep {
            stop: StopReason::Completed,
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Completed,
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Completed,
            ..Default::default()
        },
    ];
    let invocations_for_assert = invocations.clone();
    let agent = ScriptedAgent::new(script, invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets {
            max_iterations: Some(2),
            ..Default::default()
        },
        3,
    )
    .await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(
        invocations_for_assert.load(Ordering::SeqCst),
        2,
        "exactly two dispatches must fire under --max-iterations 2"
    );
    assert_eq!(outcome.sessions.len(), 2);
    assert!(matches!(
        outcome.stop_reason,
        GrindStopReason::BudgetExhausted(_)
    ));
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::BudgetExhausted);
}

/// Exit code 3: cumulative cost ceiling fires after the second session pushes
/// total spend over the configured cap.
#[tokio::test]
async fn budget_exhausted_halts_after_max_cost() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt("alpha", None)];
    let plan = default_plan_from_dir(&prompts);
    let mut by_role = std::collections::HashMap::new();
    by_role.insert(
        "implementer".to_string(),
        RoleUsage {
            input: 500_000,
            output: 0,
        },
    );
    let tokens = TokenUsage {
        input: 500_000,
        output: 0,
        by_role,
    };
    let script = vec![
        ScriptStep {
            stop: StopReason::Completed,
            tokens: tokens.clone(),
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Completed,
            tokens: tokens.clone(),
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Completed,
            tokens: tokens.clone(),
            ..Default::default()
        },
    ];
    let agent = ScriptedAgent::new(script, invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    // Default opus pricing is $15/M input, so two 500k-input dispatches =
    // $15. Set the cap at $10 so the budget trips after session 1's
    // post-flight check.
    let budgets = PlanBudgets {
        max_cost_usd: Some(10.0),
        max_iterations: Some(10),
        ..Default::default()
    };
    let mut runner = build_runner(dir.path(), &branch, prompts, plan, agent, budgets, 3).await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::BudgetExhausted);
    // Bound checks: session 1 (cost $7.50) doesn't trip, session 2 ($15
    // cumulative) does. Session 3 must not fire.
    assert!(
        outcome.sessions.len() <= 2,
        "expected at most 2 sessions, got {}",
        outcome.sessions.len()
    );
}

/// Exit code 5: three failures in a row trip the consecutive-failure escape
/// valve before any other budget can fire.
#[tokio::test]
async fn consecutive_failure_limit_trips_exit_5() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt("alpha", None)];
    let plan = default_plan_from_dir(&prompts);
    let script = vec![
        ScriptStep {
            stop: StopReason::Error("first".into()),
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Error("second".into()),
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Error("third".into()),
            ..Default::default()
        },
        ScriptStep {
            stop: StopReason::Completed,
            ..Default::default()
        },
    ];
    let agent = ScriptedAgent::new(script, invocations.clone());
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets {
            max_iterations: Some(20),
            ..Default::default()
        },
        3,
    )
    .await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(
        invocations.load(Ordering::SeqCst),
        3,
        "exactly three dispatches must fire before the limit trips"
    );
    assert!(matches!(
        outcome.stop_reason,
        GrindStopReason::ConsecutiveFailureLimit { limit: 3 }
    ));
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::ConsecutiveFailures);
}

/// Per-prompt enforcement: a sleeping mock with a 1-second
/// `max_session_seconds` cap is killed by the outer timeout. The session is
/// recorded as `Timeout` (which the runner tags as a failure for exit-code
/// purposes — `MixedFailures` here since the run otherwise completes
/// naturally via the prompt's `max_runs=1` cap).
#[tokio::test]
async fn per_prompt_max_session_seconds_is_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt_with("alpha", Some(1), Some(1))];
    let plan = default_plan_from_dir(&prompts);
    let script = vec![ScriptStep {
        sleep: Some(Duration::from_secs(10)),
        stop: StopReason::Completed,
        ..Default::default()
    }];
    let agent = ScriptedAgent::new(script, invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets::default(),
        // Disable the escape valve so the mixed-run classification surfaces.
        0,
    )
    .await;
    let started = std::time::Instant::now();
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(8),
        "outer timeout should have killed the dispatch well under 8s, took {elapsed:?}"
    );
    assert_eq!(outcome.sessions.len(), 1);
    assert_eq!(outcome.sessions[0].status, SessionStatus::Timeout);
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::MixedFailures);
}

/// Per-prompt enforcement: post-hoc cost compare. A session whose tokens
/// price out above `max_session_cost_usd` is marked `Error` with a clear
/// summary, even if the agent itself reported `Completed`.
#[tokio::test]
async fn per_prompt_max_session_cost_overrides_status() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let mut prompt = fake_prompt_with("alpha", None, Some(1));
    prompt.meta.max_session_cost_usd = Some(0.01);
    let prompts = vec![prompt];
    let plan = default_plan_from_dir(&prompts);
    let mut by_role = std::collections::HashMap::new();
    by_role.insert(
        "implementer".to_string(),
        RoleUsage {
            input: 1_000_000,
            output: 0,
        },
    );
    let tokens = TokenUsage {
        input: 1_000_000,
        output: 0,
        by_role,
    };
    let script = vec![ScriptStep {
        stop: StopReason::Completed,
        tokens,
        ..Default::default()
    }];
    let agent = ScriptedAgent::new(script, invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets::default(),
        0,
    )
    .await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(outcome.sessions.len(), 1);
    assert_eq!(outcome.sessions[0].status, SessionStatus::Error);
    let summary = outcome.sessions[0].summary.as_deref().unwrap_or("");
    assert!(
        summary.contains("max_session_cost_usd"),
        "summary should mention the cost cap: {summary}"
    );
}

/// Until-budget: when the cutoff is in the past, the runner trips
/// `BudgetExhausted` on the first pre-dispatch check without firing the
/// agent.
#[tokio::test]
async fn until_in_the_past_halts_immediately() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt("alpha", None)];
    let plan = default_plan_from_dir(&prompts);
    let agent = ScriptedAgent::new(vec![], invocations.clone());
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let past = Utc::now() - chrono::Duration::seconds(60);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets {
            until: Some(past),
            ..Default::default()
        },
        3,
    )
    .await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(invocations.load(Ordering::SeqCst), 0);
    assert!(outcome.sessions.is_empty());
    let code = classify_outcome(&outcome.stop_reason, &outcome.sessions);
    assert_eq!(code, ExitCode::BudgetExhausted);
}

/// Cost folding: a session whose tokens were billed under a known model
/// landed `cost_usd > 0` in its `SessionRecord`. (Closes a phase 07 deferred
/// item that left every session at $0.)
#[tokio::test]
async fn session_record_carries_priced_cost() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let prompts = vec![fake_prompt_with("alpha", None, Some(1))];
    let plan = default_plan_from_dir(&prompts);
    let mut by_role = std::collections::HashMap::new();
    by_role.insert(
        "implementer".to_string(),
        RoleUsage {
            input: 100_000,
            output: 50_000,
        },
    );
    let tokens = TokenUsage {
        input: 100_000,
        output: 50_000,
        by_role,
    };
    let script = vec![ScriptStep {
        stop: StopReason::Completed,
        tokens,
        ..Default::default()
    }];
    let agent = ScriptedAgent::new(script, invocations);
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let mut runner = build_runner(
        dir.path(),
        &branch,
        prompts,
        plan,
        agent,
        PlanBudgets::default(),
        3,
    )
    .await;
    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(outcome.sessions.len(), 1);
    let cost = outcome.sessions[0].cost_usd;
    // 100k input * $15/M + 50k output * $75/M = $1.50 + $3.75 = $5.25.
    assert!(
        (cost - 5.25).abs() < 1e-6,
        "expected priced cost ~$5.25, got {cost}"
    );
}
