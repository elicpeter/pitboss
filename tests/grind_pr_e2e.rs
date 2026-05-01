//! End-to-end coverage for `pitboss grind --pr`.
//!
//! The mock-git variant lives in `tests/grind_pr.rs` and pins the
//! title/body contract by exercising
//! [`pitboss::cli::grind::open_post_run_grind_pr`] against a `MockGit`. This
//! file complements that with a binary-driven path: `pitboss grind --pr` is
//! launched via `assert_cmd` against a real workspace, with the
//! `tests/fixtures/fake-gh-success.sh` shim wired in via PATH and
//! `tests/fixtures/fake-claude-success.sh` wired in via
//! `[agent.claude_code] binary` in `.pitboss/config.toml`. That covers the
//! `if args.pr && exit == ExitCode::Success` branch in `cli::grind::execute`
//! that the mock variant cannot reach.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;

use assert_cmd::Command;
use predicates::str::contains;
use tempfile::tempdir;

fn pitboss() -> Command {
    Command::cargo_bin("pitboss").expect("pitboss binary should be built")
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn write_prompt(dir: &Path, file: &str, body: &str) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join(file), body).unwrap();
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
    fs::write(dir.join(".gitignore"), ".pitboss/\n.gh-fake-log\n").unwrap();
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

/// Stage a tempdir bin/ that contains a `gh` shim resolving to
/// `tests/fixtures/fake-gh-success.sh`. Returns the bin path; the caller
/// prepends it to `PATH` for the assert_cmd invocation.
fn stage_fake_gh_on_path(bin_dir: &Path) {
    fs::create_dir_all(bin_dir).unwrap();
    let target = fixture_path("fake-gh-success.sh");
    let link = bin_dir.join("gh");
    // Symlink so the fake script still finds itself via $0 if it ever needs
    // to. `Command::new("gh")` resolves through PATH and follows the link.
    std::os::unix::fs::symlink(&target, &link).unwrap();
}

#[test]
fn grind_with_pr_invokes_fake_gh_and_succeeds() {
    let work = tempdir().unwrap();
    let home = tempdir().unwrap();
    let bin = tempdir().unwrap();

    init_git_repo(work.path());
    stage_fake_gh_on_path(bin.path());

    // One-prompt plan with `max_runs: 1` so the scheduler exhausts after a
    // single dispatch and the runner reports `GrindStopReason::Completed`
    // (which classifies as `ExitCode::Success` provided the session itself
    // is `Ok`). `--max-iterations 1` would also stop after one session, but
    // it trips `BudgetExhausted` (exit 3) and the `--pr` branch is gated
    // on Success.
    //
    // The fake claude binary returns success without writing to the
    // workspace, so the session resolves Ok with the "(no summary
    // provided)" fallback.
    write_prompt(
        &work.path().join(".pitboss/grind/prompts"),
        "alpha.md",
        "---\nname: alpha\ndescription: only prompt\nmax_runs: 1\n---\nalpha body\n",
    );

    let claude_bin = fixture_path("fake-claude-success.sh");
    let toml = format!(
        "[agent.claude_code]\nbinary = \"{}\"\n",
        claude_bin.display()
    );
    let config_path = work.path().join(".pitboss/config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    fs::write(&config_path, toml).unwrap();

    let original_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin.path().display(), original_path);

    pitboss()
        .current_dir(work.path())
        .env("HOME", home.path())
        .env_remove("NO_COLOR")
        .env("NO_COLOR", "1")
        .env("PATH", new_path)
        .args(["grind", "--pr"])
        .assert()
        .success()
        .stdout(contains("opened PR"))
        .stdout(contains("https://github.com/example/repo/pull/42"));

    // The fake-gh wrote its argv into `.gh-fake-log` in cwd. cwd is the
    // workspace because ShellGit::open_pr sets current_dir to its workspace.
    let log = fs::read_to_string(work.path().join(".gh-fake-log")).unwrap();
    assert!(log.contains("--title"), "fake-gh log: {log}");
    assert!(
        log.contains("grind/"),
        "PR title should carry grind/<plan>: prefix: {log}"
    );
    assert!(log.contains("--body"), "fake-gh log: {log}");
}
