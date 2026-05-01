//! Integration tests for `pitboss init`.
//!
//! Drives the binary via `assert_cmd` against a temp directory so we exercise
//! the full clap-dispatch path (and prove `init` honors `current_dir()`).

use std::fs;

use assert_cmd::Command;
use predicates::str::contains;
use tempfile::tempdir;

fn pitboss() -> Command {
    Command::cargo_bin("pitboss").expect("pitboss binary should be built")
}

#[test]
fn fresh_init_creates_every_artifact() {
    let dir = tempdir().unwrap();

    pitboss()
        .arg("init")
        .current_dir(dir.path())
        .assert()
        .success()
        .stdout(contains("created .pitboss/play/plan.md"))
        .stdout(contains("created .pitboss/play/deferred.md"))
        .stdout(contains("created .pitboss/config.toml"))
        .stdout(contains("created .pitboss/play/snapshots/"))
        .stdout(contains("created .pitboss/play/logs/"))
        .stdout(contains("created .pitboss/play/state.json"))
        .stdout(contains("created .pitboss/grind/prompts/"))
        .stdout(contains("created .pitboss/grind/rotations/"))
        .stdout(contains("created .pitboss/grind/runs/"));

    for rel in [
        ".pitboss",
        ".pitboss/config.toml",
        ".pitboss/play",
        ".pitboss/play/plan.md",
        ".pitboss/play/deferred.md",
        ".pitboss/play/state.json",
        ".pitboss/play/snapshots",
        ".pitboss/play/logs",
        ".pitboss/grind/prompts",
        ".pitboss/grind/rotations",
        ".pitboss/grind/runs",
        ".gitignore",
    ] {
        assert!(
            dir.path().join(rel).exists(),
            "expected {:?} after init",
            rel
        );
    }
}

#[test]
fn rerun_init_is_idempotent_and_prints_skipped() {
    let dir = tempdir().unwrap();

    pitboss()
        .arg("init")
        .current_dir(dir.path())
        .assert()
        .success();

    let snapshot_paths = [
        ".pitboss/config.toml",
        ".pitboss/play/plan.md",
        ".pitboss/play/deferred.md",
        ".pitboss/play/state.json",
        ".gitignore",
    ];
    let before: Vec<Vec<u8>> = snapshot_paths
        .iter()
        .map(|p| fs::read(dir.path().join(p)).unwrap())
        .collect();

    pitboss()
        .arg("init")
        .current_dir(dir.path())
        .assert()
        .success()
        .stdout(contains("skipped .pitboss/play/plan.md (already exists)"))
        .stdout(contains(
            "skipped .pitboss/play/deferred.md (already exists)",
        ))
        .stdout(contains("skipped .pitboss/config.toml (already exists)"))
        .stdout(contains(
            "skipped .pitboss/play/state.json (already exists)",
        ))
        .stdout(contains("skipped .gitignore (already exists)"));

    let after: Vec<Vec<u8>> = snapshot_paths
        .iter()
        .map(|p| fs::read(dir.path().join(p)).unwrap())
        .collect();
    assert_eq!(before, after, "rerun must not modify any artifact");
}

#[test]
fn preexisting_plan_md_survives_byte_for_byte_with_warning_on_stderr() {
    let dir = tempdir().unwrap();
    let custom = "---\ncurrent_phase: \"05\"\n---\n\n# Phase 05: Custom\n\nhand-written body.\n";
    let plan_path = dir.path().join(".pitboss/play/plan.md");
    fs::create_dir_all(plan_path.parent().unwrap()).unwrap();
    fs::write(&plan_path, custom).unwrap();

    pitboss()
        .arg("init")
        .current_dir(dir.path())
        .assert()
        .success()
        .stdout(contains("skipped .pitboss/play/plan.md (already exists)"))
        .stderr(contains(
            "warning: .pitboss/play/plan.md already exists, leaving it alone",
        ));

    let after = fs::read_to_string(&plan_path).unwrap();
    assert_eq!(after, custom, "init must not touch a pre-existing plan.md");
}

#[test]
fn gitignore_is_updated_idempotently() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join(".gitignore"), "/target\n").unwrap();

    for _ in 0..3 {
        pitboss()
            .arg("init")
            .current_dir(dir.path())
            .assert()
            .success();
    }

    let gi = fs::read_to_string(dir.path().join(".gitignore")).unwrap();
    assert!(gi.starts_with("/target\n"), "preserves existing entry");
    let occurrences = gi
        .lines()
        .filter(|l| l.trim().trim_start_matches('/').trim_end_matches('/') == ".pitboss")
        .count();
    assert_eq!(occurrences, 1, ".pitboss entry must appear exactly once");
}

#[test]
fn preexisting_gitignore_with_pitboss_entry_is_left_alone() {
    let dir = tempdir().unwrap();
    let original = "/target\n.pitboss/\n";
    fs::write(dir.path().join(".gitignore"), original).unwrap();

    pitboss()
        .arg("init")
        .current_dir(dir.path())
        .assert()
        .success()
        .stdout(contains("skipped .gitignore (already exists)"));

    let gi = fs::read_to_string(dir.path().join(".gitignore")).unwrap();
    assert_eq!(gi, original);
}
