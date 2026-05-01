//! Integration coverage for `pitboss grind` exit code 4
//! ([`pitboss::grind::ExitCode::FailedToStart`]).
//!
//! `tests/grind_exit_codes.rs` exercises every other documented code by
//! driving `GrindRunner` directly. The FailedToStart path lives upstream of
//! the runner in `cli::grind::run` and is reached when the pre-flight
//! refuses the run: missing prompts, non-git workspace, dirty tree on
//! resume, missing run id on resume. Each of those scenarios is wired
//! through this file via `assert_cmd` so the actual binary path stays
//! covered.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use predicates::str::contains;
use tempfile::tempdir;

fn pitboss() -> Command {
    Command::cargo_bin("pitboss").expect("pitboss binary should be built")
}

fn isolated(workspace: &Path, home: &Path) -> Command {
    let mut cmd = pitboss();
    cmd.current_dir(workspace)
        .env("HOME", home)
        .env_remove("NO_COLOR")
        .env("NO_COLOR", "1");
    cmd
}

fn write_prompt(dir: &Path, file: &str, body: &str) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join(file), body).unwrap();
}

fn seed_one_prompt(workspace: &Path) {
    let dir = workspace.join(".pitboss/grind/prompts");
    write_prompt(
        &dir,
        "alpha.md",
        "---\nname: alpha\ndescription: only prompt\n---\nalpha body\n",
    );
}

fn init_git_repo(dir: &Path) {
    let status = StdCommand::new("git")
        .args(["-c", "init.defaultBranch=main", "init", "-q"])
        .arg(dir)
        .status()
        .expect("git init");
    assert!(status.success());
    for (k, v) in [
        ("user.name", "pitboss-test"),
        ("user.email", "pitboss@test"),
    ] {
        StdCommand::new("git")
            .args(["-C"])
            .arg(dir)
            .args(["config", k, v])
            .status()
            .unwrap();
    }
    StdCommand::new("git")
        .args(["-C"])
        .arg(dir)
        .args(["commit", "--allow-empty", "-m", "seed", "-q"])
        .status()
        .unwrap();
}

#[test]
fn non_git_workspace_exits_failed_to_start() {
    // Workspace has prompts but is not a git repo, so `git create_branch`
    // refuses and the CLI returns code 4 with a clear stderr line about
    // the run branch.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_one_prompt(work.path());

    isolated(work.path(), home.path())
        .args(["grind", "--max-iterations", "1"])
        .assert()
        .code(4)
        .stderr(contains("creating run branch"));
}

#[test]
fn missing_prompts_exits_failed_to_start() {
    // No prompts at all → discovery validation refuses up-front.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    init_git_repo(work.path());

    isolated(work.path(), home.path())
        .args(["grind", "--max-iterations", "1"])
        .assert()
        .code(4)
        .stderr(contains("no prompts discovered"));
}

#[test]
fn resume_with_missing_run_id_exits_failed_to_start() {
    // No prior runs under `.pitboss/grind/runs/` — `--resume` (with no argument)
    // cannot find a target and returns code 4. With an explicit unknown id
    // the same exit code applies.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_one_prompt(work.path());
    init_git_repo(work.path());

    isolated(work.path(), home.path())
        .args(["grind", "--resume"])
        .assert()
        .code(4);

    isolated(work.path(), home.path())
        .args(["grind", "--resume", "no-such-run"])
        .assert()
        .code(4);
}

#[test]
fn resume_against_dirty_tree_exits_failed_to_start() {
    // Stage a fake run directory so the resume target resolves, then leave
    // an uncommitted change in the working tree. `cli::grind::execute_resume`
    // pre-flights `git is_clean()` and translates a dirty tree into
    // `ExitCode::FailedToStart`.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_one_prompt(work.path());
    init_git_repo(work.path());

    // Minimal run directory layout the resume code path expects: a valid
    // `state.json` whose `run_id` and `branch` line up, plus an empty
    // `sessions.jsonl`. We construct the state via the library's
    // `RunState` so the JSON shape stays in sync with the source of truth.
    let run_id = "20260101T000000Z-dirt";
    let branch = format!("pitboss/grind/{run_id}");
    let run_dir = pitboss::grind::RunDir::create(work.path(), run_id).unwrap();
    let state = pitboss::grind::RunState {
        run_id: run_id.to_string(),
        branch: branch.clone(),
        plan_name: "default".to_string(),
        prompt_names: vec!["alpha".to_string()],
        scheduler_state: Default::default(),
        budget_consumed: pitboss::grind::BudgetSnapshot::default(),
        last_session_seq: 0,
        started_at: chrono::Utc::now(),
        last_updated_at: chrono::Utc::now(),
        status: pitboss::grind::RunStatus::Aborted,
    };
    state.write(run_dir.paths()).unwrap();

    // Create the run branch so checkout succeeds, then dirty the tree.
    StdCommand::new("git")
        .args(["-C"])
        .arg(work.path())
        .args(["branch", &branch])
        .status()
        .unwrap();
    fs::write(work.path().join("dirty.txt"), "uncommitted").unwrap();

    isolated(work.path(), home.path())
        .args(["grind", "--resume", run_id])
        .assert()
        .code(4)
        .stderr(contains("working tree"));
}
