//! Phase 09 acceptance: kill-and-resume preserves scheduler position and
//! cumulative budget across the boundary; default `--resume` lands on the
//! most-recent resumable run; resume rejects when the prompt set has changed.
//!
//! Tests drive [`GrindRunner`] directly. The "kill" is simulated by dropping
//! the original runner once it has written `state.json`; "resume" is a fresh
//! [`GrindRunner::resume`] built from that state. The runner's dispatch path
//! is otherwise identical to phases 07/08, so the only invariants under test
//! here are the persistence and reconstruction logic.

#![cfg(unix)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use pitboss::agent::{Agent, AgentEvent, AgentOutcome, AgentRequest, StopReason};
use pitboss::config::Config;
use pitboss::git::{Git, ShellGit};
use pitboss::grind::{
    default_plan_from_dir, list_runs, most_recent_resumable, resolve_target, run_branch_name,
    validate_resume, GrindRunner, GrindShutdown, GrindStopReason, PlanBudgets, PromptDoc,
    PromptMeta, PromptSource, ResumeError, RunDir, RunState, RunStatus, Scheduler,
};

/// Counts dispatches and lets a watcher trip a drain after N sessions land.
/// Each dispatch writes a real source file so the runner produces a commit;
/// resume tests need real commits in place so the run branch is in a
/// reasonable shape when the resumed runner re-checks it out.
struct CommitMockAgent {
    name: String,
    invocations: Arc<AtomicU32>,
}

impl CommitMockAgent {
    fn new(invocations: Arc<AtomicU32>) -> Self {
        Self {
            name: "resume-mock".into(),
            invocations,
        }
    }
}

#[async_trait]
impl Agent for CommitMockAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(
        &self,
        req: AgentRequest,
        events: mpsc::Sender<AgentEvent>,
        _cancel: CancellationToken,
    ) -> Result<AgentOutcome> {
        let n = self.invocations.fetch_add(1, Ordering::SeqCst) + 1;
        let prompt = req
            .env
            .get("PITBOSS_PROMPT_NAME")
            .cloned()
            .unwrap_or_default();
        let seq = req
            .env
            .get("PITBOSS_SESSION_SEQ")
            .cloned()
            .unwrap_or_default();
        if let Some(parent) = req.log_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(
            &req.log_path,
            format!("[mock] dispatch {n} prompt={prompt} seq={seq}\n").as_bytes(),
        )
        .ok();

        let marker = req.workdir.join(format!(
            "src/grind_resume_session_{:0>4}_{}.rs",
            seq.parse::<u32>().unwrap_or(0),
            prompt.replace('-', "_"),
        ));
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(
            &marker,
            format!("// dispatch {n} prompt={prompt} seq={seq}\n").as_bytes(),
        )
        .ok();

        if let Some(path) = req.env.get("PITBOSS_SUMMARY_FILE") {
            fs::write(path, format!("dispatch {n} for {prompt}#{seq}")).ok();
        }
        let _ = events
            .send(AgentEvent::Stdout(format!("[mock] {prompt}#{seq}")))
            .await;
        Ok(AgentOutcome {
            exit_code: 0,
            stop_reason: StopReason::Completed,
            tokens: pitboss::state::TokenUsage::default(),
            log_path: req.log_path,
        })
    }
}

fn fake_prompt(name: &str) -> PromptDoc {
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

async fn run_initial_until_drain(
    workspace: &Path,
    run_id: &str,
    prompts: &[PromptDoc],
    drain_after: u32,
) -> (
    GrindRunner<CommitMockAgent, ShellGit>,
    Vec<pitboss::grind::SessionRecord>,
) {
    init_git_repo(workspace);
    let branch = run_branch_name(run_id);
    let git = ShellGit::new(workspace);
    git.create_branch(&branch).await.unwrap();
    git.checkout(&branch).await.unwrap();

    let plan = default_plan_from_dir(prompts);
    let run_dir = RunDir::create(workspace, run_id).expect("create run dir");
    let invocations = Arc::new(AtomicU32::new(0));
    let runner_git = ShellGit::new(workspace);
    let mut runner = GrindRunner::new(
        workspace.to_path_buf(),
        Config::default(),
        run_id.to_string(),
        branch,
        plan,
        lookup(prompts),
        run_dir,
        CommitMockAgent::new(invocations.clone()),
        runner_git,
        PlanBudgets::default(),
        3,
    );

    let shutdown = GrindShutdown::new();
    let watch_invocations = invocations.clone();
    let watch_shutdown = shutdown.clone();
    let watcher = tokio::spawn(async move {
        loop {
            if watch_invocations.load(Ordering::SeqCst) >= drain_after {
                watch_shutdown.drain();
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    let outcome = runner.run(shutdown).await.unwrap();
    let _ = watcher.await;
    assert_eq!(outcome.stop_reason, GrindStopReason::Drained);
    (runner, outcome.sessions)
}

/// Resume after a clean drain: state.json must be on disk; the resumed runner
/// must continue from `last_session_seq + 1` and produce a non-empty session
/// stream when it next drains.
#[tokio::test]
async fn kill_and_resume_continues_from_persisted_state() {
    let dir = tempfile::tempdir().unwrap();
    let run_id = "20260430T180000Z-rsm0";
    let prompts = vec![
        fake_prompt("alpha"),
        fake_prompt("bravo"),
        fake_prompt("charlie"),
    ];
    let (_runner, first_sessions) = run_initial_until_drain(dir.path(), run_id, &prompts, 3).await;
    assert!(
        first_sessions.len() >= 3,
        "expected at least 3 sessions before drain, got {}",
        first_sessions.len()
    );

    let paths = pitboss::grind::RunPaths::for_run(dir.path(), run_id);
    let state = RunState::read(&paths).expect("state.json must exist after drain");
    assert_eq!(state.run_id, run_id);
    assert_eq!(state.last_session_seq, first_sessions.last().unwrap().seq);
    // Drained runs land terminal-status = Aborted (resumable).
    assert_eq!(state.status, RunStatus::Aborted);

    // Resume now. We use validate_resume to mirror the CLI path.
    let listing = resolve_target(dir.path(), Some(run_id)).unwrap();
    let prompt_names: Vec<String> = prompts.iter().map(|p| p.meta.name.clone()).collect();
    let listing = validate_resume(listing, "default", &prompt_names).unwrap();

    let git = ShellGit::new(dir.path());
    git.checkout(&listing.state.branch).await.unwrap();
    let run_dir = RunDir::open(dir.path(), run_id).expect("re-open run dir");
    let invocations = Arc::new(AtomicU32::new(0));
    let runner_git = ShellGit::new(dir.path());
    let plan = default_plan_from_dir(&prompts);
    let mut runner = GrindRunner::resume(
        dir.path().to_path_buf(),
        Config::default(),
        run_id.to_string(),
        listing.state.branch.clone(),
        plan,
        lookup(&prompts),
        run_dir,
        CommitMockAgent::new(invocations.clone()),
        runner_git,
        PlanBudgets::default(),
        3,
        listing.state.scheduler_state.clone(),
        listing.state.budget_consumed,
        listing.state.last_session_seq,
        listing.state.started_at,
    );

    let shutdown = GrindShutdown::new();
    let watch_invocations = invocations.clone();
    let watch_shutdown = shutdown.clone();
    let watcher = tokio::spawn(async move {
        loop {
            if watch_invocations.load(Ordering::SeqCst) >= 2 {
                watch_shutdown.drain();
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    let outcome = runner.run(shutdown).await.unwrap();
    let _ = watcher.await;

    // Resumed sessions begin at first_sessions.last().seq + 1.
    let first_resumed_seq = first_sessions.last().unwrap().seq + 1;
    assert!(
        !outcome.sessions.is_empty(),
        "resumed run should have produced sessions"
    );
    assert_eq!(
        outcome.sessions.first().unwrap().seq,
        first_resumed_seq,
        "resumed runner must continue from last_session_seq + 1"
    );

    // The full sessions.jsonl now has both halves of the run.
    let combined = std::fs::read_to_string(&paths.sessions_jsonl).unwrap();
    let lines: Vec<&str> = combined.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        first_sessions.len() + outcome.sessions.len(),
        "JSONL must accumulate sessions across the resume boundary"
    );
}

/// The resumed scheduler must emit the same next prompt the original loop
/// would have. We capture the original's "next" by cloning the scheduler
/// state at drain time, advancing it once, and comparing against the first
/// prompt the resumed runner dispatches.
#[tokio::test]
async fn resume_emits_same_next_prompt_the_original_would_have() {
    let dir = tempfile::tempdir().unwrap();
    let run_id = "20260430T180000Z-rsm1";
    let prompts = vec![
        fake_prompt("alpha"),
        fake_prompt("bravo"),
        fake_prompt("charlie"),
    ];
    let (_runner, _first_sessions) = run_initial_until_drain(dir.path(), run_id, &prompts, 4).await;

    let paths = pitboss::grind::RunPaths::for_run(dir.path(), run_id);
    let state = RunState::read(&paths).unwrap();

    // Build a "what would have been next" projection from the persisted
    // scheduler state alone, with no IO and no runner.
    let plan = default_plan_from_dir(&prompts);
    let mut projection = Scheduler::with_state(
        plan.clone(),
        lookup(&prompts),
        state.scheduler_state.clone(),
    );
    let projected_next = projection
        .next()
        .expect("scheduler must have at least one more candidate")
        .meta
        .name;

    // Now resume and dispatch exactly one session. The first prompt name
    // dispatched must match the projection.
    let git = ShellGit::new(dir.path());
    git.checkout(&state.branch).await.unwrap();
    let run_dir = RunDir::open(dir.path(), run_id).unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let runner_git = ShellGit::new(dir.path());
    let mut runner = GrindRunner::resume(
        dir.path().to_path_buf(),
        Config::default(),
        run_id.to_string(),
        state.branch.clone(),
        plan,
        lookup(&prompts),
        run_dir,
        CommitMockAgent::new(invocations.clone()),
        runner_git,
        PlanBudgets::default(),
        3,
        state.scheduler_state.clone(),
        state.budget_consumed,
        state.last_session_seq,
        state.started_at,
    );

    let shutdown = GrindShutdown::new();
    let watch_invocations = invocations.clone();
    let watch_shutdown = shutdown.clone();
    let watcher = tokio::spawn(async move {
        loop {
            if watch_invocations.load(Ordering::SeqCst) >= 1 {
                watch_shutdown.drain();
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    let outcome = runner.run(shutdown).await.unwrap();
    let _ = watcher.await;

    let first_dispatched = &outcome
        .sessions
        .first()
        .expect("at least one resumed session")
        .prompt;
    assert_eq!(
        first_dispatched, &projected_next,
        "resumed runner must dispatch the same prompt the original scheduler would have"
    );
}

/// Resume with no run-id picks the most-recent run whose status is `Active`
/// or `Aborted`, even if a fresher `Completed` run is on disk.
#[tokio::test]
async fn default_resume_picks_most_recent_resumable() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join(".pitboss/grind/runs")).unwrap();

    // Create two run dirs by hand, with different timestamps and statuses.
    fn write_state(repo: &Path, run_id: &str, status: RunStatus, ts: &str) {
        let paths = pitboss::grind::RunPaths::for_run(repo, run_id);
        fs::create_dir_all(&paths.root).unwrap();
        fs::create_dir_all(&paths.transcripts).unwrap();
        fs::create_dir_all(&paths.worktrees).unwrap();
        fs::write(&paths.sessions_jsonl, b"").unwrap();
        fs::write(&paths.sessions_md, b"").unwrap();
        fs::write(&paths.scratchpad, b"").unwrap();
        let state = RunState {
            run_id: run_id.to_string(),
            branch: format!("pitboss/grind/{run_id}"),
            plan_name: "default".into(),
            prompt_names: vec!["alpha".into()],
            scheduler_state: pitboss::grind::SchedulerState::default(),
            budget_consumed: pitboss::grind::BudgetSnapshot::default(),
            last_session_seq: 0,
            started_at: ts.parse().unwrap(),
            last_updated_at: ts.parse().unwrap(),
            status,
        };
        state.write(&paths).unwrap();
    }

    write_state(
        dir.path(),
        "rid-completed",
        RunStatus::Completed,
        "2026-04-30T18:00:00Z",
    );
    write_state(
        dir.path(),
        "rid-aborted",
        RunStatus::Aborted,
        "2026-04-30T17:30:00Z",
    );
    write_state(
        dir.path(),
        "rid-active",
        RunStatus::Active,
        "2026-04-30T17:00:00Z",
    );

    // The freshest *resumable* run is rid-aborted (Completed is terminal).
    let pick = most_recent_resumable(dir.path()).unwrap();
    assert_eq!(pick.run_id, "rid-aborted");

    // resolve_target with no run-id mirrors the CLI path.
    let listing = resolve_target(dir.path(), None).unwrap();
    assert_eq!(listing.run_id, "rid-aborted");
    assert_eq!(listing.state.status, RunStatus::Aborted);

    // list_runs returns all three (sorted desc by last_updated_at).
    let listings = list_runs(dir.path());
    assert_eq!(listings.len(), 3);
    assert_eq!(listings[0].run_id, "rid-completed");
    assert_eq!(listings[1].run_id, "rid-aborted");
    assert_eq!(listings[2].run_id, "rid-active");
}

/// Resume rejects when a prompt referenced by the original plan has been
/// removed.
#[tokio::test]
async fn resume_rejects_when_prompt_removed() {
    let dir = tempfile::tempdir().unwrap();
    let run_id = "20260430T180000Z-rsm2";
    let prompts = vec![fake_prompt("alpha"), fake_prompt("bravo")];
    let (_runner, _first_sessions) = run_initial_until_drain(dir.path(), run_id, &prompts, 2).await;

    // Discover only one prompt now (bravo removed).
    let listing = resolve_target(dir.path(), Some(run_id)).unwrap();
    let err = validate_resume(listing, "default", &["alpha".to_string()]).unwrap_err();
    match err {
        ResumeError::PromptSetChanged { added, removed, .. } => {
            assert!(added.is_empty(), "no prompts added; got: {added:?}");
            assert_eq!(removed, vec!["bravo".to_string()]);
        }
        other => panic!("expected PromptSetChanged, got {other:?}"),
    }
}
