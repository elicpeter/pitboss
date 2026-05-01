//! End-to-end CLI coverage for `pitboss grind --resume`.
//!
//! `tests/grind_resume.rs` exercises the resume *mechanics* by constructing
//! `GrindRunner::resume` directly. This file drives the same path through
//! the binary so the wiring inside `cli::grind::execute_resume` —
//! `git is_clean`, `git checkout`, `announce_resume`, the `ResumeError`
//! translation, and the `--max-iterations` budget layering on top of the
//! persisted snapshot — stays covered.

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
    // Mirror real pitboss workspaces: `.pitboss/` is gitignored so the
    // per-run directory we seed below does not leave `is_clean` reporting
    // dirty.
    fs::write(dir.join(".gitignore"), ".pitboss/\n").unwrap();
    StdCommand::new("git")
        .args(["-C"])
        .arg(dir)
        .args(["add", ".gitignore"])
        .status()
        .unwrap();
    StdCommand::new("git")
        .args(["-C"])
        .arg(dir)
        .args(["commit", "-m", "seed", "-q"])
        .status()
        .unwrap();
}

/// Materialize the on-disk shape `cli::grind::execute_resume` expects:
/// a clean git repo, an existing run branch, a valid `state.json`, and an
/// empty `sessions.jsonl`. Returns the run id.
fn seed_resumable_run(workspace: &Path, status: pitboss::grind::RunStatus) -> String {
    let run_id = "20260101T000000Z-cli";
    let branch = format!("pitboss/grind/{run_id}");

    // The run branch must already exist for `execute_resume`'s checkout.
    StdCommand::new("git")
        .args(["-C"])
        .arg(workspace)
        .args(["branch", &branch])
        .status()
        .unwrap();

    let run_dir = pitboss::grind::RunDir::create(workspace, run_id).unwrap();
    let state = pitboss::grind::RunState {
        run_id: run_id.to_string(),
        branch,
        plan_name: "default".to_string(),
        prompt_names: vec!["alpha".to_string()],
        scheduler_state: Default::default(),
        budget_consumed: pitboss::grind::BudgetSnapshot::default(),
        last_session_seq: 0,
        started_at: chrono::Utc::now(),
        last_updated_at: chrono::Utc::now(),
        status,
    };
    state.write(run_dir.paths()).unwrap();
    run_id.to_string()
}

#[test]
fn resume_with_explicit_run_id_announces_and_reaches_runner() {
    // Happy-path resume of a clean tree. We force the session-budget to zero
    // via `--max-iterations 0` so the budget tracker trips before any agent
    // dispatch — the resume code path still has to traverse `is_clean`,
    // `checkout`, run-dir open, runner construction, and `announce_resume`
    // before the runner returns `BudgetExhausted` (exit 3). Reaching that
    // exit code is the proof that `execute_resume` ran end-to-end.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_one_prompt(work.path());
    init_git_repo(work.path());
    let run_id = seed_resumable_run(work.path(), pitboss::grind::RunStatus::Aborted);

    isolated(work.path(), home.path())
        .args(["grind", "--resume", &run_id, "--max-iterations", "0"])
        .assert()
        .code(3)
        .stderr(contains("resuming"));
}

#[test]
fn resume_with_no_argument_picks_latest_run() {
    // The bare `--resume` form (no positional arg) must resolve to the most
    // recent resumable run on disk. Same exit-3 trick as above proves the
    // selection landed on our seeded run.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_one_prompt(work.path());
    init_git_repo(work.path());
    let _ = seed_resumable_run(work.path(), pitboss::grind::RunStatus::Aborted);

    isolated(work.path(), home.path())
        .args(["grind", "--resume", "--max-iterations", "0"])
        .assert()
        .code(3)
        .stderr(contains("resuming"));
}

#[test]
fn resume_refuses_completed_run() {
    // `RunStatus::Completed` is not resumable. The CLI must surface this as
    // a `FailedToStart` (exit 4) rather than silently re-driving a run that
    // already wrapped up.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_one_prompt(work.path());
    init_git_repo(work.path());
    let run_id = seed_resumable_run(work.path(), pitboss::grind::RunStatus::Completed);

    isolated(work.path(), home.path())
        .args(["grind", "--resume", &run_id])
        .assert()
        .code(4);
}
