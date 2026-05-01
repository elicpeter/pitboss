//! Integration tests for `pitboss grind --dry-run` (phase 12).
//!
//! Drives the binary via `assert_cmd` against a temp workspace seeded with
//! prompt files but no git repo. The dry-run path must:
//!
//! - exit `0`
//! - print the deterministic header (`=== pitboss grind --dry-run ===`)
//! - never create a `.pitboss/grind/runs/<run-id>/` directory
//! - never invoke git (proven by the test workspace not being a git repo —
//!   if dry-run accidentally tried to `git init` / `git checkout` the
//!   subprocess would error out)
//!
//! The exact format of the report is pinned by the unit-level snapshot in
//! `src/grind/dry_run.rs::tests::dry_run_report_snapshot_full_fixture`. These
//! tests are about the CLI plumbing and the no-side-effects contract.

use std::fs;
use std::path::Path;

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

fn seed_three_prompts(workspace: &Path) {
    let dir = workspace.join(".pitboss/grind/prompts");
    write_prompt(
        &dir,
        "alpha.md",
        "---\nname: alpha\ndescription: first\nweight: 2\nevery: 1\n---\nalpha body\n",
    );
    write_prompt(
        &dir,
        "bravo.md",
        "---\nname: bravo\ndescription: second\nweight: 1\nevery: 1\n---\nbravo body\n",
    );
    write_prompt(
        &dir,
        "charlie.md",
        "---\nname: charlie\ndescription: third\nevery: 3\n---\ncharlie body\n",
    );
}

#[test]
fn dry_run_prints_header_and_exits_zero() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_three_prompts(work.path());

    isolated(work.path(), home.path())
        .args(["grind", "--dry-run"])
        .assert()
        .success()
        .stdout(contains("=== pitboss grind --dry-run ==="))
        .stdout(contains("version: 1"))
        .stdout(contains("alpha"))
        .stdout(contains("bravo"))
        .stdout(contains("charlie"))
        .stdout(contains("Scheduler preview"));
}

#[test]
fn dry_run_creates_no_run_directory_and_no_branch() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_three_prompts(work.path());

    // Workspace is intentionally not a git repo. `pitboss grind` (without
    // --dry-run) requires a git repo to create the run branch; the dry-run
    // path must never reach the git layer at all, so the absence of a repo
    // is the proof that no commits or branches were attempted.
    isolated(work.path(), home.path())
        .args(["grind", "--dry-run"])
        .assert()
        .success();

    let grind_runs_root = work.path().join(".pitboss/grind/runs");
    assert!(
        !grind_runs_root.exists() || fs::read_dir(&grind_runs_root).unwrap().next().is_none(),
        "no per-run directory should be created: {:?}",
        grind_runs_root
    );

    // Sanity: this is not a git repo, so the path that would skip the
    // dry-run guard would have failed. Re-run without --dry-run as a
    // smoke check that the same workspace would otherwise refuse to start.
    isolated(work.path(), home.path())
        .args(["grind"])
        .assert()
        .failure();
}

#[test]
fn dry_run_with_no_prompts_exits_failed_to_start() {
    // No prompts means there's nothing to dry-run; pitboss should report
    // failed-to-start (exit 4) rather than silently dumping an empty
    // report. This pins the early-exit ordering: discovery validation
    // runs before the --dry-run branch.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    isolated(work.path(), home.path())
        .args(["grind", "--dry-run"])
        .assert()
        .code(4)
        .stderr(contains("no prompts discovered"));
}

#[test]
fn dry_run_with_resume_no_runs_reports_failed_to_start() {
    // `--dry-run --resume` is allowed but still needs a resumable run on disk
    // — without one we surface the same "no resumable grind run found" error
    // the live resume path produces, mapped to exit 4. This pins the CLI
    // wiring after the rejection from earlier phases was lifted.
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_three_prompts(work.path());

    isolated(work.path(), home.path())
        .args(["grind", "--dry-run", "--resume"])
        .assert()
        .code(4)
        .stderr(contains("no resumable grind run found"));
}

#[test]
fn dry_run_with_resume_seeds_preview_from_persisted_state() {
    // Happy path: a resumable run exists on disk. `--dry-run --resume` should
    // load it, validate the prompt set, and emit a report whose `## Resume`
    // section mirrors the persisted budget snapshot. The preview reflects the
    // resumed scheduler state rather than starting at rotation 0.
    use chrono::Utc;
    use std::collections::{BTreeMap, HashMap};
    use std::path::PathBuf;

    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    seed_three_prompts(work.path());

    let run_id = "20260430T180000Z-dryresume";
    let run_dir = pitboss::grind::RunDir::create(work.path(), run_id).unwrap();

    // Seed three matching session records into sessions.jsonl so the
    // reconciler (which compares state.last_session_seq against the JSONL
    // tail) sees an aligned snapshot.
    let log = run_dir.log();
    for (seq, prompt) in [(1u32, "alpha"), (2u32, "bravo"), (3u32, "alpha")] {
        let rec = pitboss::grind::SessionRecord {
            seq,
            run_id: run_id.to_string(),
            prompt: prompt.to_string(),
            started_at: Utc::now(),
            ended_at: Utc::now(),
            status: pitboss::grind::SessionStatus::Ok,
            summary: Some(format!("session {seq}")),
            commit: None,
            tokens: pitboss::state::TokenUsage {
                input: 1000,
                output: 500,
                by_role: HashMap::new(),
            },
            cost_usd: 0.4,
            transcript_path: PathBuf::from(format!("transcripts/session-{seq:04}.log")),
        };
        log.append(&rec).unwrap();
    }

    let mut runs: BTreeMap<String, u32> = BTreeMap::new();
    runs.insert("alpha".to_string(), 2);
    runs.insert("bravo".to_string(), 1);
    let state = pitboss::grind::RunState {
        run_id: run_id.to_string(),
        branch: format!("pitboss/grind/{run_id}"),
        plan_name: "default".to_string(),
        prompt_names: vec![
            "alpha".to_string(),
            "bravo".to_string(),
            "charlie".to_string(),
        ],
        scheduler_state: pitboss::grind::SchedulerState {
            rotation: 3,
            runs_per_prompt: runs,
        },
        budget_consumed: pitboss::grind::BudgetSnapshot {
            iterations: 3,
            tokens_input: 3000,
            tokens_output: 1500,
            cost_usd: 1.2,
            consecutive_failures: 0,
        },
        last_session_seq: 3,
        started_at: Utc::now(),
        last_updated_at: Utc::now(),
        status: pitboss::grind::RunStatus::Aborted,
    };
    state.write(run_dir.paths()).unwrap();

    isolated(work.path(), home.path())
        .args(["grind", "--dry-run", "--resume", run_id])
        .assert()
        .success()
        .stdout(contains("## Resume"))
        .stdout(contains("last_session_seq: 3"))
        .stdout(contains("iterations_consumed: 3"))
        .stdout(contains("resumed scheduler state"));
}

#[test]
fn dry_run_help_documents_pr_and_dry_run_flags() {
    // Acceptance: `pitboss grind --help` documents every new flag added by
    // phases 08-12. This is the cheapest gate against accidentally dropping
    // either flag from the clap definition.
    let assert = pitboss().args(["grind", "--help"]).assert().success();
    let out = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(out.contains("--dry-run"), "missing --dry-run: {out}");
    assert!(out.contains("--pr"), "missing --pr: {out}");
}
