//! Integration tests for `pitboss prompts {ls, validate, new}` (phase 03).
//!
//! Drives the binary via `assert_cmd` against a temp workspace. Each test
//! points `HOME` at its own temp dir so the user's real
//! `~/.pitboss/grind/prompts/` cannot leak into discovery.

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

#[test]
fn help_lists_all_three_actions() {
    let assert = pitboss().args(["prompts", "--help"]).assert().success();
    let out = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(out.contains("ls"), "help output missing `ls`: {out}");
    assert!(
        out.contains("validate"),
        "help output missing `validate`: {out}"
    );
    assert!(out.contains("new"), "help output missing `new`: {out}");
}

#[test]
fn ls_in_fresh_repo_with_no_prompts_says_so() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    isolated(work.path(), home.path())
        .args(["prompts", "ls"])
        .assert()
        .success()
        .stdout(contains("no prompts discovered"));
}

#[test]
fn ls_renders_table_with_discovered_prompts() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    let project_dir = work.path().join(".pitboss/grind/prompts");
    write_prompt(
        &project_dir,
        "alpha.md",
        "---\nname: alpha\ndescription: first prompt\nweight: 2\nevery: 1\nverify: true\n---\nbody\n",
    );
    write_prompt(
        &project_dir,
        "bravo.md",
        "---\nname: bravo\ndescription: second prompt\n---\nbody\n",
    );

    let assert = isolated(work.path(), home.path())
        .args(["prompts", "ls"])
        .assert()
        .success();
    let out = String::from_utf8(assert.get_output().stdout.clone()).unwrap();

    assert!(out.contains("NAME"), "header missing: {out}");
    assert!(out.contains("SOURCE"), "header missing: {out}");
    assert!(out.contains("WEIGHT"), "header missing: {out}");
    assert!(out.contains("EVERY"), "header missing: {out}");
    assert!(out.contains("VERIFY"), "header missing: {out}");
    assert!(out.contains("PATH"), "header missing: {out}");
    assert!(out.contains("alpha"), "row missing: {out}");
    assert!(out.contains("bravo"), "row missing: {out}");
    assert!(out.contains("project"), "source missing: {out}");
}

#[test]
fn validate_zero_prompts_exits_ok_with_zero_summary() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    isolated(work.path(), home.path())
        .args(["prompts", "validate"])
        .assert()
        .success()
        .stdout(contains("0 prompt(s) ok, 0 error(s)"));
}

#[test]
fn validate_reports_each_bad_file_and_exits_nonzero() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    let dir = work.path().join(".pitboss/grind/prompts");
    write_prompt(
        &dir,
        "good.md",
        "---\nname: good\ndescription: ok\n---\nbody\n",
    );
    write_prompt(&dir, "bad-no-fence.md", "no frontmatter here\n");
    write_prompt(
        &dir,
        "bad-name.md",
        "---\nname: Bad Name\ndescription: nope\n---\nbody\n",
    );

    let assert = isolated(work.path(), home.path())
        .args(["prompts", "validate"])
        .assert()
        .failure();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();

    assert!(
        stdout.contains("1 prompt(s) ok, 2 error(s)"),
        "summary missing or wrong, stdout={stdout}"
    );
    assert!(
        stderr.contains("bad-no-fence.md"),
        "expected bad-no-fence.md in errors, got: {stderr}"
    );
    assert!(
        stderr.contains("bad-name.md"),
        "expected bad-name.md in errors, got: {stderr}"
    );
    let error_lines = stderr.lines().filter(|l| l.contains("error:")).count();
    assert_eq!(
        error_lines, 2,
        "expected one error line per bad file, got: {stderr}"
    );
}

#[test]
fn new_creates_project_prompt_and_refuses_to_overwrite() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    isolated(work.path(), home.path())
        .args(["prompts", "new", "fp-hunter"])
        .assert()
        .success()
        .stdout(contains("created"))
        .stdout(contains(".pitboss/grind/prompts/fp-hunter.md"));

    let target = work.path().join(".pitboss/grind/prompts/fp-hunter.md");
    assert!(target.exists(), "new should write the file");
    let body = fs::read_to_string(&target).unwrap();
    assert!(body.contains("name: fp-hunter"), "body: {body}");
    assert!(
        !body.contains("__NAME__"),
        "placeholder should be substituted: {body}"
    );

    let second = isolated(work.path(), home.path())
        .args(["prompts", "new", "fp-hunter"])
        .assert()
        .failure();
    let stderr = String::from_utf8(second.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("refusing to overwrite"),
        "expected refusal message, got: {stderr}"
    );
}

#[test]
fn new_global_writes_under_home_pitboss_prompts() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    isolated(work.path(), home.path())
        .args(["prompts", "new", "triage", "--global"])
        .assert()
        .success();

    let target = home.path().join(".pitboss/grind/prompts/triage.md");
    assert!(
        target.exists(),
        "global new should write under $HOME/.pitboss/grind/prompts"
    );
    let project_target = work.path().join(".pitboss/grind/prompts/triage.md");
    assert!(
        !project_target.exists(),
        "global new must not touch the project prompts dir"
    );
}

#[test]
fn new_dir_override_writes_to_explicit_directory() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    let custom = work.path().join("custom-prompts");

    isolated(work.path(), home.path())
        .args(["prompts", "new", "lint-sweep", "--dir"])
        .arg(&custom)
        .assert()
        .success();

    assert!(custom.join("lint-sweep.md").exists());
}

#[test]
fn new_then_validate_round_trip_passes() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    isolated(work.path(), home.path())
        .args(["prompts", "new", "round-trip"])
        .assert()
        .success();

    isolated(work.path(), home.path())
        .args(["prompts", "validate"])
        .assert()
        .success()
        .stdout(contains("1 prompt(s) ok, 0 error(s)"));
}

#[test]
fn new_rejects_invalid_name() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();

    isolated(work.path(), home.path())
        .args(["prompts", "new", "Has Caps"])
        .assert()
        .failure();
}
