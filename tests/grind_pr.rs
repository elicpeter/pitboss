//! Phase 12 acceptance: `pitboss grind --pr` opens exactly one PR after a
//! successful run, via the same `git.open_pr` pathway `pitboss play --pr`
//! uses.
//!
//! Drives a [`GrindRunner`] end-to-end against the mock-agent pattern from
//! `grind_smoke`, then exercises [`cli::grind::open_post_run_grind_pr`]
//! against a [`MockGit`] so the title/body the runner produced are pinned
//! without spawning `gh`.

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
use pitboss::cli::grind::open_post_run_grind_pr;
use pitboss::config::Config;
use pitboss::git::{self, Git, MockGit, MockOp, ShellGit};
use pitboss::grind::{
    default_plan_from_dir, GrindRunner, GrindShutdown, GrindStopReason, PlanBudgets, PromptDoc,
    PromptMeta, PromptSource, RunDir, SessionStatus,
};

const RUN_ID: &str = "20260430T200000Z-pr01";

struct MockAgent {
    invocations: Arc<AtomicU32>,
}

#[async_trait]
impl Agent for MockAgent {
    fn name(&self) -> &str {
        "grind-pr-mock"
    }

    async fn run(
        &self,
        req: AgentRequest,
        events: mpsc::Sender<AgentEvent>,
        _cancel: CancellationToken,
    ) -> Result<AgentOutcome> {
        let n = self.invocations.fetch_add(1, Ordering::SeqCst) + 1;
        let prompt_name = req.env.get("PITBOSS_PROMPT_NAME").cloned().unwrap_or_default();
        let seq = req.env.get("PITBOSS_SESSION_SEQ").cloned().unwrap_or_default();

        // Land a real edit so the session produces a commit on the run branch.
        let marker = req
            .workdir
            .join(format!("src/grind_pr_session_{n}_{prompt_name}.rs"));
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(&marker, format!("// pr session {n}\n").as_bytes()).expect("write marker");

        // Transcript file the runner expects to exist.
        if let Some(parent) = req.log_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(&req.log_path, b"[mock] ran\n").ok();

        // Summary file the runner reads back into the session record.
        let summary = PathBuf::from(req.env.get("PITBOSS_SUMMARY_FILE").unwrap());
        if let Some(parent) = summary.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(&summary, format!("session {seq} ({prompt_name}) summary"))
            .expect("write summary");

        let _ = events.send(AgentEvent::Stdout("[mock] ok".into())).await;

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
        body: format!("{name} prompt body"),
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

async fn drive_successful_run(workspace: &Path) -> (String, String) {
    init_git_repo(workspace);

    let prompts = vec![fake_prompt("alpha"), fake_prompt("bravo")];
    let plan = default_plan_from_dir(&prompts);
    let plan_name = plan.name.clone();

    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let setup_git = ShellGit::new(workspace);
    setup_git.create_branch(&branch).await.unwrap();
    setup_git.checkout(&branch).await.unwrap();

    let invocations = Arc::new(AtomicU32::new(0));
    let run_dir = RunDir::create(workspace, RUN_ID).expect("create run dir");
    let runner_git = ShellGit::new(workspace);

    let mut runner = GrindRunner::new(
        workspace.to_path_buf(),
        Config::default(),
        RUN_ID.to_string(),
        branch.clone(),
        plan,
        lookup(&prompts),
        run_dir,
        MockAgent {
            invocations: invocations.clone(),
        },
        runner_git,
        PlanBudgets::default(),
        3,
    );

    let shutdown = GrindShutdown::new();
    let watch = invocations.clone();
    let watch_shutdown = shutdown.clone();
    let watcher = tokio::spawn(async move {
        loop {
            if watch.load(Ordering::SeqCst) >= 3 {
                watch_shutdown.drain();
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });

    let outcome = runner.run(shutdown).await.unwrap();
    let _ = watcher.await;

    assert_eq!(outcome.stop_reason, GrindStopReason::Drained);
    assert!(
        outcome
            .sessions
            .iter()
            .all(|s| s.status == SessionStatus::Ok),
        "every session should be Ok: {:?}",
        outcome.sessions
    );

    (plan_name, branch)
}

#[tokio::test]
async fn pr_helper_invokes_mock_git_with_grind_title_and_sessions_md_body() {
    let dir = tempfile::tempdir().unwrap();
    let (plan_name, _branch) = drive_successful_run(dir.path()).await;

    // Sanity: the grind run wrote a sessions.md the helper will inline.
    let sessions_md_path = dir
        .path()
        .join(".pitboss")
        .join("grind")
        .join(RUN_ID)
        .join("sessions.md");
    let sessions_md = fs::read_to_string(&sessions_md_path).unwrap();
    assert!(sessions_md.contains("# Sessions"), "{sessions_md}");
    assert!(
        sessions_md.contains("session-0001"),
        "sessions.md missing session 1: {sessions_md}"
    );

    let mock = MockGit::new();
    mock.set_open_pr_response("https://github.com/example/repo/pull/777");
    let url = open_post_run_grind_pr(&mock, dir.path(), RUN_ID, &plan_name)
        .await
        .expect("open_post_run_grind_pr");
    assert_eq!(url, "https://github.com/example/repo/pull/777");

    // Exactly one open_pr call, with the canonical grind title and the
    // run's sessions.md verbatim as the body.
    let ops = mock.ops();
    let pr_calls: Vec<&MockOp> = ops
        .iter()
        .filter(|o| matches!(o, MockOp::OpenPr { .. }))
        .collect();
    assert_eq!(pr_calls.len(), 1, "ops: {ops:?}");
    match pr_calls[0] {
        MockOp::OpenPr { title, body } => {
            assert_eq!(title, &git::grind_pr_title(&plan_name, RUN_ID));
            assert_eq!(body, &sessions_md);
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn pr_helper_propagates_gh_failure() {
    let dir = tempfile::tempdir().unwrap();
    let (plan_name, _branch) = drive_successful_run(dir.path()).await;

    let mock = MockGit::new();
    mock.set_open_pr_failure("no remote configured");
    let err = open_post_run_grind_pr(&mock, dir.path(), RUN_ID, &plan_name)
        .await
        .expect_err("expected helper to surface the gh error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("opening PR") && msg.contains("no remote configured"),
        "unexpected error chain: {msg}"
    );
}

#[test]
fn grind_pr_title_uses_documented_format() {
    // Acceptance criterion in plan.md phase 12: PR title is
    // `grind/<plan-or-default>: <run-id>`. Pin it here so a downstream
    // refactor can't accidentally drift the surface contract.
    assert_eq!(
        git::grind_pr_title("default", "20260430T200000Z-pr01"),
        "grind/default: 20260430T200000Z-pr01"
    );
    assert_eq!(
        git::grind_pr_title("nightly-rotation", "20260430T200000Z-pr01"),
        "grind/nightly-rotation: 20260430T200000Z-pr01"
    );
}
