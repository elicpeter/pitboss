//! Integration coverage for the `verify: true` fixer cycle in `pitboss grind`.
//!
//! Phase 07 shipped a one-shot test invocation behind `verify: true`: tests
//! pass → `SessionStatus::Ok`, tests fail → `SessionStatus::Error`. The
//! deferred follow-up wires that path into a bounded fixer loop modeled on
//! the runner's [`crate::config::RetryBudgets::fixer_max_attempts`]: a
//! failing test run dispatches the fixer agent up to N times, re-running
//! tests after each attempt, before giving up.
//!
//! The tests here drive [`GrindRunner`] against a [`CountingAgent`] that
//! records every dispatch and can be configured to "fix" the failing test
//! script the second time it is asked. The script itself is a sequence-aware
//! shell helper that fails its first invocation and succeeds (or keeps
//! failing) afterward depending on what the agent does.

#![cfg(unix)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use pitboss::agent::{Agent, AgentEvent, AgentOutcome, AgentRequest, Role, StopReason};
use pitboss::config::Config;
use pitboss::git::{Git, ShellGit};
use pitboss::grind::{
    default_plan_from_dir, GrindRunner, GrindShutdown, GrindStopReason, PlanBudgets, PromptDoc,
    PromptMeta, PromptSource, RunDir, SessionStatus,
};

/// Mock agent that counts dispatches by role and optionally creates a marker
/// file when invoked as the fixer. The marker file is what flips the test
/// script from "fail" to "pass" on the next invocation.
struct CountingAgent {
    name: String,
    implementer_calls: Arc<AtomicU32>,
    fixer_calls: Arc<AtomicU32>,
    /// When `Some`, the fixer writes the named marker file on every fixer
    /// dispatch so a sequence-aware test runner script can detect the
    /// repair and switch from failing to passing.
    fixer_marker: Option<PathBuf>,
}

impl CountingAgent {
    fn new(
        implementer_calls: Arc<AtomicU32>,
        fixer_calls: Arc<AtomicU32>,
        fixer_marker: Option<PathBuf>,
    ) -> Self {
        Self {
            name: "verify-fixer-mock".into(),
            implementer_calls,
            fixer_calls,
            fixer_marker,
        }
    }
}

#[async_trait]
impl Agent for CountingAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(
        &self,
        req: AgentRequest,
        events: mpsc::Sender<AgentEvent>,
        _cancel: CancellationToken,
    ) -> Result<AgentOutcome> {
        match req.role {
            Role::Implementer => {
                self.implementer_calls.fetch_add(1, Ordering::SeqCst);
            }
            Role::Fixer => {
                self.fixer_calls.fetch_add(1, Ordering::SeqCst);
                if let Some(marker) = &self.fixer_marker {
                    let path = req.workdir.join(marker);
                    if let Some(parent) = path.parent() {
                        fs::create_dir_all(parent).ok();
                    }
                    fs::write(&path, b"fixed by mock\n").expect("write fixer marker");
                }
            }
            other => {
                panic!("CountingAgent: unexpected role {other:?}");
            }
        }

        if let Some(parent) = req.log_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(&req.log_path, format!("[mock] role={:?}\n", req.role)).ok();

        if let Some(path) = req.env.get("PITBOSS_SUMMARY_FILE") {
            fs::write(path, format!("dispatched {:?}", req.role)).ok();
        }

        let _ = events
            .send(AgentEvent::Stdout(format!("[{:?}] dispatched", req.role)))
            .await;

        Ok(AgentOutcome {
            exit_code: 0,
            stop_reason: StopReason::Completed,
            tokens: pitboss::state::TokenUsage::default(),
            log_path: req.log_path,
        })
    }
}

fn verify_prompt() -> PromptDoc {
    PromptDoc {
        meta: PromptMeta {
            name: "verify-prompt".into(),
            description: "verify prompt".into(),
            weight: 1,
            every: 1,
            max_runs: Some(1),
            verify: true,
            parallel_safe: false,
            tags: vec![],
            max_session_seconds: None,
            max_session_cost_usd: None,
        },
        body: "do the thing".into(),
        source_path: PathBuf::from("/fixture/verify-prompt.md"),
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

/// Write a shell test runner that exits 1 unless `marker_path` exists.
fn write_marker_test_script(workspace: &Path, script: &Path, marker_path: &str) {
    let body = format!(
        "#!/bin/sh\nif [ -f {marker} ]; then\n  echo 'tests pass after fix'\n  exit 0\nelse\n  echo 'tests fail (marker missing)'\n  exit 1\nfi\n",
        marker = marker_path,
    );
    fs::write(script, body).unwrap();
    Command::new("chmod")
        .arg("+x")
        .arg(script)
        .current_dir(workspace)
        .status()
        .unwrap();
}

#[tokio::test]
async fn verify_failure_dispatches_fixer_and_recovers_when_tests_pass() {
    // The implementer leaves tests red. The fixer agent writes the marker
    // file the test runner is gated on; the second run of tests then exits 0
    // and the session resolves as `Ok`.
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(dir.path());

    let script = dir.path().join("test.sh");
    write_marker_test_script(dir.path(), &script, "fixer-marker");

    let cfg_path = dir.path().join(".pitboss/config.toml");
    fs::create_dir_all(cfg_path.parent().unwrap()).unwrap();
    fs::write(
        &cfg_path,
        "[tests]\ncommand = \"./test.sh\"\n[retries]\nfixer_max_attempts = 2\n",
    )
    .unwrap();

    let prompts = vec![verify_prompt()];
    let plan = default_plan_from_dir(&prompts);
    let run_id = "20260430T180000Z-vfx0";
    let branch = format!("pitboss/grind/{run_id}");
    let git = ShellGit::new(dir.path());
    git.create_branch(&branch).await.unwrap();
    git.checkout(&branch).await.unwrap();

    let run_dir = RunDir::create(dir.path(), run_id).unwrap();
    let implementer = Arc::new(AtomicU32::new(0));
    let fixer = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent::new(
        implementer.clone(),
        fixer.clone(),
        Some(PathBuf::from("fixer-marker")),
    );
    let mut config = Config::default();
    config.tests.command = Some("./test.sh".into());
    config.retries.fixer_max_attempts = 2;

    let mut runner = GrindRunner::new(
        dir.path().to_path_buf(),
        config,
        run_id.to_string(),
        branch.clone(),
        plan,
        lookup(&prompts),
        run_dir,
        agent,
        ShellGit::new(dir.path()),
        PlanBudgets::default(),
        3,
    );

    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert!(matches!(outcome.stop_reason, GrindStopReason::Completed));
    assert_eq!(outcome.sessions.len(), 1);
    assert_eq!(outcome.sessions[0].status, SessionStatus::Ok);
    assert_eq!(implementer.load(Ordering::SeqCst), 1);
    assert_eq!(fixer.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn verify_failure_exhausts_fixer_budget_and_records_error() {
    // The fixer agent never writes the marker, so tests stay red across the
    // entire fixer budget. The session record must end up `Error` and the
    // summary should call out how many attempts were burned.
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(dir.path());

    let script = dir.path().join("test.sh");
    write_marker_test_script(dir.path(), &script, "marker-never-written");

    let cfg_path = dir.path().join(".pitboss/config.toml");
    fs::create_dir_all(cfg_path.parent().unwrap()).unwrap();
    fs::write(
        &cfg_path,
        "[tests]\ncommand = \"./test.sh\"\n[retries]\nfixer_max_attempts = 2\n",
    )
    .unwrap();

    let prompts = vec![verify_prompt()];
    let plan = default_plan_from_dir(&prompts);
    let run_id = "20260430T180000Z-vfx1";
    let branch = format!("pitboss/grind/{run_id}");
    let git = ShellGit::new(dir.path());
    git.create_branch(&branch).await.unwrap();
    git.checkout(&branch).await.unwrap();

    let run_dir = RunDir::create(dir.path(), run_id).unwrap();
    let implementer = Arc::new(AtomicU32::new(0));
    let fixer = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent::new(implementer.clone(), fixer.clone(), None);
    let mut config = Config::default();
    config.tests.command = Some("./test.sh".into());
    config.retries.fixer_max_attempts = 2;

    let mut runner = GrindRunner::new(
        dir.path().to_path_buf(),
        config,
        run_id.to_string(),
        branch.clone(),
        plan,
        lookup(&prompts),
        run_dir,
        agent,
        ShellGit::new(dir.path()),
        PlanBudgets::default(),
        3,
    );

    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(outcome.sessions.len(), 1);
    assert_eq!(outcome.sessions[0].status, SessionStatus::Error);
    let summary = outcome.sessions[0].summary.as_deref().unwrap_or("");
    assert!(
        summary.contains("verify failed after 2 fixer attempts"),
        "summary should report fixer exhaustion, got {summary:?}"
    );
    assert_eq!(implementer.load(Ordering::SeqCst), 1);
    assert_eq!(fixer.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn verify_failure_with_fixer_disabled_records_error_immediately() {
    // `fixer_max_attempts = 0` disables the fixer entirely. A failing verify
    // run should land as `Error` without dispatching the fixer agent at all.
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(dir.path());

    let script = dir.path().join("test.sh");
    write_marker_test_script(dir.path(), &script, "marker-never-written");

    let cfg_path = dir.path().join(".pitboss/config.toml");
    fs::create_dir_all(cfg_path.parent().unwrap()).unwrap();
    fs::write(
        &cfg_path,
        "[tests]\ncommand = \"./test.sh\"\n[retries]\nfixer_max_attempts = 0\n",
    )
    .unwrap();

    let prompts = vec![verify_prompt()];
    let plan = default_plan_from_dir(&prompts);
    let run_id = "20260430T180000Z-vfx2";
    let branch = format!("pitboss/grind/{run_id}");
    let git = ShellGit::new(dir.path());
    git.create_branch(&branch).await.unwrap();
    git.checkout(&branch).await.unwrap();

    let run_dir = RunDir::create(dir.path(), run_id).unwrap();
    let implementer = Arc::new(AtomicU32::new(0));
    let fixer = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent::new(implementer.clone(), fixer.clone(), None);
    let mut config = Config::default();
    config.tests.command = Some("./test.sh".into());
    config.retries.fixer_max_attempts = 0;

    let mut runner = GrindRunner::new(
        dir.path().to_path_buf(),
        config,
        run_id.to_string(),
        branch.clone(),
        plan,
        lookup(&prompts),
        run_dir,
        agent,
        ShellGit::new(dir.path()),
        PlanBudgets::default(),
        3,
    );

    let outcome = runner.run(GrindShutdown::new()).await.unwrap();
    assert_eq!(outcome.sessions.len(), 1);
    assert_eq!(outcome.sessions[0].status, SessionStatus::Error);
    let summary = outcome.sessions[0].summary.as_deref().unwrap_or("");
    assert!(
        summary.starts_with("verify failed:"),
        "summary should call out verify failure with fixer disabled, got {summary:?}"
    );
    assert_eq!(implementer.load(Ordering::SeqCst), 1);
    assert_eq!(fixer.load(Ordering::SeqCst), 0);
}
