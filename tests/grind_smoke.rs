//! Phase 07 grind acceptance: drive a [`GrindRunner`] end-to-end against a
//! scripted [`MockAgent`] and assert the per-run directory, session log, and
//! drain-on-Ctrl-C path all behave as the spec calls out.

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
use tokio::sync::{mpsc, Semaphore};
use tokio_util::sync::CancellationToken;

use pitboss::agent::{Agent, AgentEvent, AgentOutcome, AgentRequest, StopReason};
use pitboss::config::Config;
use pitboss::git::{Git, ShellGit};
use pitboss::grind::{
    default_plan_from_dir, GrindRunner, GrindShutdown, GrindStopReason, PlanBudgets, PromptDoc,
    PromptMeta, PromptSource, RunDir, SessionStatus,
};

const RUN_ID: &str = "20260430T180000Z-test";

/// Per-call mock that asserts the env vars pitboss promised the agent and
/// writes a session summary file plus a real source file (so each session
/// produces a commit).
struct MockAgent {
    name: String,
    write_marker: bool,
    summary_template: String,
    invocations: Arc<AtomicU32>,
    expected_run_id: String,
    /// Optional pair of channels for tests that need a deterministic sync
    /// point per session. When set, the agent: (1) sends the session seq on
    /// `started_tx` after writing summary/marker, then (2) acquires one
    /// permit from `proceed` before returning. The test drives the sequence
    /// by reading the started signal and adding permits as needed.
    gate: Option<(mpsc::UnboundedSender<u32>, Arc<Semaphore>)>,
}

impl MockAgent {
    fn new(invocations: Arc<AtomicU32>, expected_run_id: &str) -> Self {
        Self {
            name: "grind-mock".into(),
            write_marker: true,
            summary_template: "session ran for {prompt} #{seq}".into(),
            invocations,
            expected_run_id: expected_run_id.into(),
            gate: None,
        }
    }

    fn gated(
        invocations: Arc<AtomicU32>,
        expected_run_id: &str,
        started_tx: mpsc::UnboundedSender<u32>,
        proceed: Arc<Semaphore>,
    ) -> Self {
        let mut me = Self::new(invocations, expected_run_id);
        me.gate = Some((started_tx, proceed));
        me
    }
}

#[async_trait]
impl Agent for MockAgent {
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

        // Required env vars are all present and non-empty.
        for key in [
            "PITBOSS_RUN_ID",
            "PITBOSS_PROMPT_NAME",
            "PITBOSS_SUMMARY_FILE",
            "PITBOSS_SCRATCHPAD",
            "PITBOSS_SESSION_SEQ",
            "PITBOSS_WORKTREE",
        ] {
            let val = req
                .env
                .get(key)
                .unwrap_or_else(|| panic!("env var {key} missing on dispatch {n}"));
            assert!(!val.is_empty(), "env var {key} empty on dispatch {n}");
        }
        assert_eq!(
            req.env.get("PITBOSS_RUN_ID").map(String::as_str),
            Some(self.expected_run_id.as_str())
        );

        let prompt_name = req.env.get("PITBOSS_PROMPT_NAME").unwrap().clone();
        let seq = req.env.get("PITBOSS_SESSION_SEQ").unwrap().clone();

        // Writing to the agent-facing scratchpad must round-trip back to
        // pitboss the same way the production agent would write it. Each
        // session adds one labeled line so the scratchpad accumulates over
        // the run.
        let scratchpad_path = PathBuf::from(req.env.get("PITBOSS_SCRATCHPAD").unwrap());
        let mut existing = fs::read_to_string(&scratchpad_path).unwrap_or_default();
        existing.push_str(&format!("- session {seq} ({prompt_name})\n"));
        fs::write(&scratchpad_path, existing).expect("write scratchpad");

        // Auto-injected context blocks must reach the agent verbatim.
        assert!(
            req.user_prompt.contains("<!-- pitboss:session-log -->"),
            "missing session-log marker in prompt #{n}"
        );
        assert!(
            req.user_prompt.contains("<!-- pitboss:scratchpad -->"),
            "missing scratchpad marker in prompt #{n}"
        );
        assert!(
            req.user_prompt
                .contains("<!-- pitboss:standing-instruction:start -->"),
            "missing standing-instruction marker in prompt #{n}"
        );

        // Drop a real source-tree file so each session lands a commit.
        if self.write_marker {
            let marker = req.workdir.join(format!(
                "src/grind_session_{:04}_{}.rs",
                seq.parse::<u32>().unwrap_or(0),
                prompt_name.replace('-', "_")
            ));
            if let Some(parent) = marker.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::write(
                &marker,
                format!("// session {seq} for {prompt_name}\n").as_bytes(),
            )
            .expect("write marker");
        }

        // Materialize the transcript so the runner's expected log path exists
        // even with no real subprocess.
        if let Some(parent) = req.log_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(
            &req.log_path,
            format!("[mock] session {seq} for {prompt_name}\n").as_bytes(),
        )
        .ok();

        // Write the agent summary the way a production agent would.
        let summary_file = PathBuf::from(req.env.get("PITBOSS_SUMMARY_FILE").unwrap());
        if let Some(parent) = summary_file.parent() {
            fs::create_dir_all(parent).ok();
        }
        let summary = self
            .summary_template
            .replace("{prompt}", &prompt_name)
            .replace("{seq}", &seq);
        fs::write(&summary_file, summary).expect("write summary");

        let _ = events
            .send(AgentEvent::Stdout(format!("[mock] {prompt_name}#{seq}")))
            .await;

        // Gated mode: signal the test we've reached the end of dispatch, then
        // block until the test grants a permit. This gives the test a
        // deterministic point to (a) trip the drain flag before the runner
        // can spin up the next session, and (b) verify how many sessions
        // actually ran.
        if let Some((started_tx, proceed)) = &self.gate {
            let _ = started_tx.send(n);
            let permit = proceed
                .clone()
                .acquire_owned()
                .await
                .expect("proceed semaphore closed unexpectedly");
            permit.forget();
        }

        Ok(AgentOutcome {
            exit_code: 0,
            stop_reason: StopReason::Completed,
            tokens: pitboss::state::TokenUsage::default(),
            log_path: req.log_path,
        })
    }
}

fn fake_prompt(name: &str, body: &str) -> PromptDoc {
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
        body: body.into(),
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

async fn make_runner(
    workspace: &Path,
    branch: &str,
    invocations: Arc<AtomicU32>,
) -> (GrindRunner<MockAgent, ShellGit>, pitboss::grind::SessionLog) {
    init_git_repo(workspace);
    let git = ShellGit::new(workspace);
    git.create_branch(branch).await.unwrap();
    git.checkout(branch).await.unwrap();

    let prompts = vec![
        fake_prompt("alpha", "alpha prompt body"),
        fake_prompt("bravo", "bravo prompt body"),
        fake_prompt("charlie", "charlie prompt body"),
    ];
    let plan = default_plan_from_dir(&prompts);
    let run_dir = RunDir::create(workspace, RUN_ID).expect("create run dir");
    let log = run_dir.log().clone();
    let runner_git = ShellGit::new(workspace);
    let runner = GrindRunner::new(
        workspace.to_path_buf(),
        Config::default(),
        RUN_ID.to_string(),
        branch.to_string(),
        plan,
        lookup(&prompts),
        run_dir,
        MockAgent::new(invocations, RUN_ID),
        runner_git,
        PlanBudgets::default(),
        3,
    );
    (runner, log)
}

#[tokio::test]
async fn six_sessions_rotate_across_three_prompts() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let (mut runner, log) = make_runner(dir.path(), &branch, invocations.clone()).await;

    let shutdown = GrindShutdown::new();
    // Drain after exactly six sessions complete by spawning a watcher that
    // counts mock invocations and flips the drain flag once we hit the
    // target. The runner exits between sessions when drain is set.
    let watch_invocations = invocations.clone();
    let watch_shutdown = shutdown.clone();
    let watcher = tokio::spawn(async move {
        loop {
            if watch_invocations.load(Ordering::SeqCst) >= 6 {
                watch_shutdown.drain();
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });

    let outcome = runner.run(shutdown).await.unwrap();
    let _ = watcher.await;

    assert_eq!(outcome.run_id, RUN_ID);
    assert_eq!(outcome.branch, branch);
    assert_eq!(
        outcome.stop_reason,
        GrindStopReason::Drained,
        "runner should report drained, not completed (mock prompts have no max_runs cap)"
    );

    // Sessions should be exactly 6 once the watcher's drain trips between
    // sessions; the in-flight session may make it 7 if the drain races, so
    // accept 6..=7. The acceptance criterion is "six sessions" — the tighter
    // assertion is on the JSONL line count plus the rotation distribution.
    assert!(
        (6..=7).contains(&outcome.sessions.len()),
        "expected 6-7 sessions, got {}",
        outcome.sessions.len()
    );

    // Source-of-truth JSONL has every session.
    let records = log.records().unwrap();
    assert_eq!(records.len(), outcome.sessions.len());
    for (i, r) in records.iter().enumerate() {
        let want_seq = (i + 1) as u32;
        assert_eq!(r.seq, want_seq, "record {i} seq should be {want_seq}");
        assert_eq!(r.run_id, RUN_ID);
        assert_eq!(r.status, SessionStatus::Ok, "record {i} status");
        let summary = r.summary.as_deref().unwrap_or("");
        assert!(
            summary.contains(&r.prompt) && summary.contains(&r.seq.to_string()),
            "summary should mention prompt name and seq: {summary}"
        );
        assert!(
            r.commit.is_some(),
            "session {} should have produced a commit",
            r.seq,
        );
    }

    // Round-robin across the three prompts: at least one of each in the
    // first six sessions.
    let names: Vec<&str> = records.iter().map(|r| r.prompt.as_str()).collect();
    for expected in ["alpha", "bravo", "charlie"] {
        assert!(
            names.contains(&expected),
            "rotation should include {expected}: {names:?}"
        );
    }
}

#[tokio::test]
async fn run_directory_layout_matches_spec() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let branch = pitboss::grind::run_branch_name(RUN_ID);
    let (mut runner, _log) = make_runner(dir.path(), &branch, invocations.clone()).await;

    let shutdown = GrindShutdown::new();
    let watch_invocations = invocations.clone();
    let watch_shutdown = shutdown.clone();
    let watcher = tokio::spawn(async move {
        loop {
            if watch_invocations.load(Ordering::SeqCst) >= 3 {
                watch_shutdown.drain();
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });

    runner.run(shutdown).await.unwrap();
    let _ = watcher.await;

    let root = dir.path().join(".pitboss/grind/runs").join(RUN_ID);
    assert!(root.is_dir(), "run root must exist: {:?}", root);
    assert!(root.join("sessions.jsonl").is_file());
    assert!(root.join("sessions.md").is_file());
    assert!(root.join("scratchpad.md").is_file());
    assert!(root.join("transcripts").is_dir());
    assert!(root.join("worktrees").is_dir());

    // The mock writes one labeled line per session into the scratchpad.
    // Reading must return that content (the agent owns the file; pitboss
    // never trims it).
    let pad = fs::read_to_string(root.join("scratchpad.md")).unwrap();
    assert!(
        pad.lines().count() >= 3,
        "scratchpad should accumulate session entries: {pad}"
    );
    assert!(pad.contains("session 1"));
}

/// A drain triggered after session 2 finishes must let session 2 land cleanly
/// and prevent session 3 from starting. Uses a gated mock-agent so the test
/// can deterministically interleave drain-vs-dispatch: drain is set while the
/// agent is still inside session 2's dispatch, so the next-iteration drain
/// check is guaranteed to see it before any session 3 spawns.
#[tokio::test]
async fn drain_after_session_two_skips_session_three() {
    let dir = tempfile::tempdir().unwrap();
    let invocations = Arc::new(AtomicU32::new(0));
    let branch = pitboss::grind::run_branch_name(RUN_ID);

    // Build a gated runner by hand (the make_runner helper uses the
    // ungated MockAgent constructor).
    init_git_repo(dir.path());
    let git = ShellGit::new(dir.path());
    git.create_branch(&branch).await.unwrap();
    git.checkout(&branch).await.unwrap();

    let prompts = vec![
        fake_prompt("alpha", "alpha prompt body"),
        fake_prompt("bravo", "bravo prompt body"),
        fake_prompt("charlie", "charlie prompt body"),
    ];
    let plan = pitboss::grind::default_plan_from_dir(&prompts);
    let run_dir = RunDir::create(dir.path(), RUN_ID).expect("create run dir");
    let log = run_dir.log().clone();

    let (started_tx, mut started_rx) = mpsc::unbounded_channel::<u32>();
    let proceed = Arc::new(Semaphore::new(0));
    let runner_git = ShellGit::new(dir.path());
    let mut runner = GrindRunner::new(
        dir.path().to_path_buf(),
        Config::default(),
        RUN_ID.to_string(),
        branch.clone(),
        plan,
        lookup(&prompts),
        run_dir,
        MockAgent::gated(invocations.clone(), RUN_ID, started_tx, proceed.clone()),
        runner_git,
        PlanBudgets::default(),
        3,
    );

    let shutdown = GrindShutdown::new();

    // Drive runner concurrently so the test thread can step it through
    // sessions one at a time.
    let runner_shutdown = shutdown.clone();
    let runner_handle = tokio::spawn(async move { runner.run(runner_shutdown).await.unwrap() });

    // Step session 1: receive the entry signal, then release.
    let n1 = started_rx.recv().await.expect("session 1 start signal");
    assert_eq!(n1, 1);
    proceed.add_permits(1);

    // Step session 2: receive the entry signal. Set drain BEFORE releasing
    // the agent so the runner's next-iteration drain check is guaranteed to
    // see it. (If we released first, the runner would race ahead and could
    // dispatch session 3 between our log-poll and our drain call — which is
    // exactly the flake the deferred item called out.)
    let n2 = started_rx.recv().await.expect("session 2 start signal");
    assert_eq!(n2, 2);
    shutdown.drain();
    proceed.add_permits(1);

    // Drain rx so we can prove session 3 never started. Closing the test
    // channel is what would happen if a 3rd dispatch tried to send on a
    // dropped sender, but here we keep the sender alive in the agent — so
    // the right thing to assert is that the runner finishes and *then*
    // started_rx is empty.
    let outcome = runner_handle.await.expect("runner task panicked");

    // Strict bound: exactly two sessions ran. Earlier this was 2..=3 because
    // the polling watcher couldn't win the race; the gated mock removes the
    // race entirely.
    assert_eq!(
        invocations.load(Ordering::SeqCst),
        2,
        "exactly two dispatches should have fired; session 3 must not start"
    );
    assert_eq!(
        outcome.sessions.len(),
        2,
        "expected exactly 2 sessions after drain, got {}",
        outcome.sessions.len()
    );
    assert_eq!(outcome.stop_reason, GrindStopReason::Drained);

    // No further start signals were emitted (session 3 never began dispatch).
    assert!(
        started_rx.try_recv().is_err(),
        "no start signal should have been emitted for a third session"
    );

    // JSONL log mirrors the count.
    let records = log.records().unwrap();
    assert_eq!(records.len(), 2);
    for r in &records {
        assert_eq!(r.status, SessionStatus::Ok);
    }
}
