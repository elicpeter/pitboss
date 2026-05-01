//! Integration tests for `pitboss grind --dry-run` (phase 12).
//!
//! Drives the binary via `assert_cmd` against a temp workspace seeded with
//! prompt files but no git repo. The dry-run path must:
//!
//! - exit `0`
//! - print the deterministic header (`=== pitboss grind --dry-run ===`)
//! - never create a `.pitboss/grind/<run-id>/` directory
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
    let dir = workspace.join(".pitboss").join("prompts");
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

    let grind_root = work.path().join(".pitboss").join("grind");
    assert!(
        !grind_root.exists() || fs::read_dir(&grind_root).unwrap().next().is_none(),
        "no per-run directory should be created: {:?}",
        grind_root
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
fn dry_run_help_documents_pr_and_dry_run_flags() {
    // Acceptance: `pitboss grind --help` documents every new flag added by
    // phases 08-12. This is the cheapest gate against accidentally dropping
    // either flag from the clap definition.
    let assert = pitboss().args(["grind", "--help"]).assert().success();
    let out = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(out.contains("--dry-run"), "missing --dry-run: {out}");
    assert!(out.contains("--pr"), "missing --pr: {out}");
}
