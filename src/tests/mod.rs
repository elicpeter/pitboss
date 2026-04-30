//! Project test runner detection and execution.
//!
//! Despite the module name, this is *not* the crate's integration test
//! directory (that lives at `<crate-root>/tests/`). It's the per-project test
//! runner the foreman runner invokes after each phase to decide whether the
//! agent's changes pass.
//!
//! ## Surface
//!
//! [`detect`] probes the workspace for a recognized project layout and returns
//! a [`TestRunner`] preconfigured to invoke the right command. The probe is
//! best-effort; foreman.toml's `[tests] command = "..."` overrides detection
//! entirely, in which case the configured command is used verbatim.
//!
//! [`TestRunner::run`] executes the runner, tees combined stdout+stderr to a
//! per-phase log file, and returns a [`TestOutcome`] with a short summary
//! suitable for surfacing in CLI output and feeding into the fixer prompt.
//!
//! ## Detection priority
//!
//! 1. `Cargo.toml`             → `cargo test`
//! 2. `package.json` (with a `test` script) → `npm` / `pnpm` / `yarn` `test`
//!    (chosen by the lock file present)
//! 3. `pyproject.toml` or `setup.py` → `pytest`
//! 4. `go.mod`                 → `go test ./...`
//!
//! Detection stops at the first match. A workspace with both `Cargo.toml` and
//! `package.json` resolves to cargo — pick whichever language is canonical and
//! set `[tests] command = "..."` to override.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

/// Maximum number of trailing log lines included in [`TestOutcome::summary`]
/// when a test run fails. Keeps prompts and CLI output bounded; the full log
/// is always available at [`TestOutcome::log_path`].
const FAILURE_TAIL_LINES: usize = 40;

/// Result of executing a [`TestRunner`].
///
/// `summary` is a short, human-readable description suitable for CLI output
/// and fixer prompts. The full transcript is at `log_path`.
#[derive(Debug, Clone)]
pub struct TestOutcome {
    /// `true` when the underlying process exited with status 0.
    pub passed: bool,
    /// Short description of the run. On success: `"<runner>: passed (N lines
    /// captured)"`. On failure: `"<runner>: failed (exit C)\n<last K lines>"`.
    pub summary: String,
    /// Path to the combined stdout+stderr log written during the run.
    pub log_path: PathBuf,
}

/// Which built-in detector matched, if any.
///
/// Independent of `program`/`args` because callers occasionally want to log or
/// branch on the kind without re-parsing the command line. [`TestRunnerKind::Override`]
/// signals that the runner came from `foreman.toml` rather than detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestRunnerKind {
    /// `cargo test` — chosen when `Cargo.toml` is present.
    Cargo,
    /// `npm test` — JS workspace with no other lock file.
    Npm,
    /// `pnpm test` — JS workspace with `pnpm-lock.yaml`.
    Pnpm,
    /// `yarn test` — JS workspace with `yarn.lock`.
    Yarn,
    /// `pytest` — Python workspace (`pyproject.toml` or `setup.py`).
    Pytest,
    /// `go test ./...` — Go workspace (`go.mod`).
    Go,
    /// User-supplied `[tests] command = "..."`. Bypassed detection.
    Override,
}

impl TestRunnerKind {
    /// Short name used in summaries and log lines.
    pub fn label(self) -> &'static str {
        match self {
            TestRunnerKind::Cargo => "cargo test",
            TestRunnerKind::Npm => "npm test",
            TestRunnerKind::Pnpm => "pnpm test",
            TestRunnerKind::Yarn => "yarn test",
            TestRunnerKind::Pytest => "pytest",
            TestRunnerKind::Go => "go test",
            TestRunnerKind::Override => "tests",
        }
    }
}

/// A resolved test invocation: program, arguments, and the workspace to run
/// it from. Construct via [`detect`] (or [`TestRunner::from_override`] when
/// the user has supplied an explicit command).
#[derive(Debug, Clone)]
pub struct TestRunner {
    /// Detector that produced this runner, or `Override` for user-supplied.
    pub kind: TestRunnerKind,
    /// Program to spawn (e.g., `"cargo"`, `"sh"` if the user wrapped it).
    pub program: String,
    /// Arguments passed to the program.
    pub args: Vec<String>,
    /// Working directory the process is spawned in.
    pub workdir: PathBuf,
}

impl TestRunner {
    /// Build a runner from a user-supplied shell-style command line.
    ///
    /// The command is whitespace-split into program + args; shell features
    /// (pipes, env-var assignments, glob expansion) require an explicit
    /// `sh -c "..."` wrapper. Returns `None` when the command is empty or
    /// contains only whitespace.
    pub fn from_override(command: &str, workdir: impl Into<PathBuf>) -> Option<Self> {
        let mut parts = command.split_whitespace().map(str::to_string);
        let program = parts.next()?;
        let args: Vec<String> = parts.collect();
        Some(Self {
            kind: TestRunnerKind::Override,
            program,
            args,
            workdir: workdir.into(),
        })
    }

    /// Spawn the configured command, tee combined stdout+stderr to `log_path`,
    /// and wait for it to exit. Returns a [`TestOutcome`] describing the run.
    ///
    /// `log_path`'s parent directory is created if missing. The log file is
    /// truncated on each call so a re-run produces a clean transcript;
    /// callers wanting per-attempt logs must vary the path.
    pub async fn run(&self, log_path: impl Into<PathBuf>) -> Result<TestOutcome> {
        let log_path = log_path.into();
        if let Some(parent) = log_path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .with_context(|| format!("tests: create log dir {:?}", parent))?;
            }
        }
        let mut log_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&log_path)
            .await
            .with_context(|| format!("tests: open log {:?}", log_path))?;

        // Header line so the log file is self-describing — useful when an
        // oncall is reading just `phase-NN.log` without other context.
        let header = format!(
            "$ {}{}{} (cwd: {})\n",
            self.program,
            if self.args.is_empty() { "" } else { " " },
            self.args.join(" "),
            self.workdir.display(),
        );
        log_file
            .write_all(header.as_bytes())
            .await
            .with_context(|| format!("tests: write header {:?}", log_path))?;

        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args)
            .current_dir(&self.workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("tests: spawn {:?}", self.program))?;
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        // Single-tasked merge: tokio::select! over both readers so we keep
        // strict relative ordering from each stream while writing serially to
        // the log file. Captures every line (no truncation) so the tail-on-
        // failure summary has the real terminal output to draw from.
        let mut line_count: usize = 0;
        let mut tail: std::collections::VecDeque<String> =
            std::collections::VecDeque::with_capacity(FAILURE_TAIL_LINES);
        let mut stdout_done = false;
        let mut stderr_done = false;

        loop {
            tokio::select! {
                line = stdout_reader.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(l)) => {
                            log_file.write_all(l.as_bytes()).await.ok();
                            log_file.write_all(b"\n").await.ok();
                            push_tail(&mut tail, l);
                            line_count += 1;
                        }
                        Ok(None) | Err(_) => stdout_done = true,
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(l)) => {
                            log_file.write_all(b"[stderr] ").await.ok();
                            log_file.write_all(l.as_bytes()).await.ok();
                            log_file.write_all(b"\n").await.ok();
                            push_tail(&mut tail, format!("[stderr] {l}"));
                            line_count += 1;
                        }
                        Ok(None) | Err(_) => stderr_done = true,
                    }
                }
                else => break,
            }
        }

        let status = child.wait().await.context("tests: waiting for child")?;
        log_file.flush().await.ok();

        let passed = status.success();
        let summary = if passed {
            format!(
                "{}: passed ({} lines captured)",
                self.kind.label(),
                line_count
            )
        } else {
            let exit = status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string());
            let mut s = format!(
                "{}: failed (exit {}, {} lines captured)",
                self.kind.label(),
                exit,
                line_count
            );
            if !tail.is_empty() {
                s.push('\n');
                for line in &tail {
                    s.push_str(line);
                    s.push('\n');
                }
            }
            s
        };

        Ok(TestOutcome {
            passed,
            summary,
            log_path,
        })
    }
}

/// Bounded ring of trailing lines used to build the failure summary.
fn push_tail(tail: &mut std::collections::VecDeque<String>, line: String) {
    if tail.len() == FAILURE_TAIL_LINES {
        tail.pop_front();
    }
    tail.push_back(line);
}

/// Pick a [`TestRunner`] for `workdir`.
///
/// `override_command` is consulted first: when `Some`, the result is built via
/// [`TestRunner::from_override`] without any filesystem probing. When `None`
/// (or when the override is empty), the function probes the workspace in the
/// order listed in the module docs and returns the first match.
///
/// Returns `None` only when no override is given and no recognized layout is
/// present — the runner treats that as "tests skipped" rather than a failure.
pub fn detect(workdir: impl AsRef<Path>, override_command: Option<&str>) -> Option<TestRunner> {
    let workdir = workdir.as_ref();
    if let Some(cmd) = override_command {
        if let Some(runner) = TestRunner::from_override(cmd, workdir) {
            return Some(runner);
        }
    }

    if workdir.join("Cargo.toml").is_file() {
        return Some(TestRunner {
            kind: TestRunnerKind::Cargo,
            program: "cargo".into(),
            args: vec!["test".into()],
            workdir: workdir.to_path_buf(),
        });
    }

    if let Some(runner) = detect_node(workdir) {
        return Some(runner);
    }

    if workdir.join("pyproject.toml").is_file() || workdir.join("setup.py").is_file() {
        return Some(TestRunner {
            kind: TestRunnerKind::Pytest,
            program: "pytest".into(),
            args: Vec::new(),
            workdir: workdir.to_path_buf(),
        });
    }

    if workdir.join("go.mod").is_file() {
        return Some(TestRunner {
            kind: TestRunnerKind::Go,
            program: "go".into(),
            args: vec!["test".into(), "./...".into()],
            workdir: workdir.to_path_buf(),
        });
    }

    None
}

/// Detect a JS/TS workspace. Requires `package.json` with a `scripts.test`
/// entry; without one we have no command to invoke. The package manager is
/// chosen from the lock file present (pnpm > yarn > npm), defaulting to npm.
fn detect_node(workdir: &Path) -> Option<TestRunner> {
    let pkg = workdir.join("package.json");
    if !pkg.is_file() {
        return None;
    }
    if !package_json_has_test_script(&pkg) {
        return None;
    }

    let (kind, program) = if workdir.join("pnpm-lock.yaml").is_file() {
        (TestRunnerKind::Pnpm, "pnpm")
    } else if workdir.join("yarn.lock").is_file() {
        (TestRunnerKind::Yarn, "yarn")
    } else {
        (TestRunnerKind::Npm, "npm")
    };

    // npm uniquely requires `--` to forward extra flags; no extras here so
    // the bare `npm test` form is fine. pnpm and yarn use `<pm> test`.
    let args = vec!["test".into()];

    Some(TestRunner {
        kind,
        program: program.into(),
        args,
        workdir: workdir.to_path_buf(),
    })
}

/// `true` when `package.json` parses as JSON and contains a non-empty
/// `scripts.test` string. A malformed `package.json` is treated as "no
/// recognizable test script" rather than a hard error — detection stays
/// best-effort.
fn package_json_has_test_script(path: &Path) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    value
        .get("scripts")
        .and_then(|s| s.get("test"))
        .and_then(|t| t.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn touch(dir: &Path, rel: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"").unwrap();
    }

    fn write(dir: &Path, rel: &str, contents: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn detect_returns_none_for_unrecognized_layout() {
        let dir = tempdir().unwrap();
        assert!(detect(dir.path(), None).is_none());
    }

    #[test]
    fn detect_cargo_when_cargo_toml_present() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "Cargo.toml");
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Cargo);
        assert_eq!(runner.program, "cargo");
        assert_eq!(runner.args, vec!["test".to_string()]);
        assert_eq!(runner.workdir, dir.path());
    }

    #[test]
    fn detect_pytest_via_pyproject_toml() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "pyproject.toml");
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Pytest);
        assert_eq!(runner.program, "pytest");
        assert!(runner.args.is_empty());
    }

    #[test]
    fn detect_pytest_via_setup_py_when_pyproject_missing() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "setup.py");
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Pytest);
    }

    #[test]
    fn detect_go_when_go_mod_present() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "go.mod");
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Go);
        assert_eq!(runner.program, "go");
        assert_eq!(runner.args, vec!["test".to_string(), "./...".to_string()]);
    }

    #[test]
    fn detect_npm_when_package_json_has_test_script() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"scripts":{"test":"jest"}}"#);
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Npm);
        assert_eq!(runner.program, "npm");
        assert_eq!(runner.args, vec!["test".to_string()]);
    }

    #[test]
    fn detect_pnpm_when_pnpm_lock_present() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "package.json",
            r#"{"scripts":{"test":"vitest"}}"#,
        );
        touch(dir.path(), "pnpm-lock.yaml");
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Pnpm);
        assert_eq!(runner.program, "pnpm");
    }

    #[test]
    fn detect_yarn_when_yarn_lock_present_but_no_pnpm() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"scripts":{"test":"jest"}}"#);
        touch(dir.path(), "yarn.lock");
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Yarn);
        assert_eq!(runner.program, "yarn");
    }

    #[test]
    fn detect_pnpm_wins_over_yarn_when_both_lockfiles_present() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "package.json",
            r#"{"scripts":{"test":"vitest"}}"#,
        );
        touch(dir.path(), "pnpm-lock.yaml");
        touch(dir.path(), "yarn.lock");
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Pnpm);
    }

    #[test]
    fn detect_skips_node_when_no_test_script() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"scripts":{"build":"tsc"}}"#);
        // No fallback in this layout, so detection returns None entirely.
        assert!(detect(dir.path(), None).is_none());
    }

    #[test]
    fn detect_skips_node_when_test_script_is_empty_string() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"scripts":{"test":"   "}}"#);
        assert!(detect(dir.path(), None).is_none());
    }

    #[test]
    fn detect_treats_malformed_package_json_as_no_match() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", "{ not valid json");
        assert!(detect(dir.path(), None).is_none());
    }

    #[test]
    fn detect_priority_cargo_over_node() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "Cargo.toml");
        write(dir.path(), "package.json", r#"{"scripts":{"test":"jest"}}"#);
        let runner = detect(dir.path(), None).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Cargo);
    }

    #[test]
    fn override_bypasses_detection_entirely() {
        let dir = tempdir().unwrap();
        // Cargo would otherwise win.
        touch(dir.path(), "Cargo.toml");
        let runner = detect(dir.path(), Some("make check")).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Override);
        assert_eq!(runner.program, "make");
        assert_eq!(runner.args, vec!["check".to_string()]);
    }

    #[test]
    fn override_with_only_whitespace_falls_back_to_detection() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "Cargo.toml");
        // Empty/whitespace override yields no runner, so detection runs.
        let runner = detect(dir.path(), Some("   ")).unwrap();
        assert_eq!(runner.kind, TestRunnerKind::Cargo);
    }

    #[test]
    fn override_with_no_args_uses_program_only() {
        let runner = TestRunner::from_override("./run-tests", "/tmp").unwrap();
        assert_eq!(runner.program, "./run-tests");
        assert!(runner.args.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_succeeds_for_zero_exit() {
        let dir = tempdir().unwrap();
        let runner = TestRunner::from_override("/bin/sh -c true", dir.path()).unwrap();
        let outcome = runner.run(dir.path().join("test.log")).await.unwrap();
        assert!(outcome.passed);
        assert!(outcome.summary.contains("passed"));
        assert!(outcome.log_path.is_file());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_fails_with_tail_summary_for_nonzero_exit() {
        let dir = tempdir().unwrap();
        // Built directly because shell-style quoting can't survive the
        // whitespace-split heuristic in `from_override`.
        let runner = TestRunner {
            kind: TestRunnerKind::Override,
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "echo failure-marker; exit 7".into()],
            workdir: dir.path().to_path_buf(),
        };
        let outcome = runner.run(dir.path().join("test.log")).await.unwrap();
        assert!(!outcome.passed);
        assert!(outcome.summary.contains("failed"));
        assert!(
            outcome.summary.contains("failure-marker"),
            "summary should include tail; got: {}",
            outcome.summary
        );
        assert!(outcome.summary.contains("exit 7"));
        // Log file contains the same line.
        let log = std::fs::read_to_string(&outcome.log_path).unwrap();
        assert!(log.contains("failure-marker"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_logs_header_with_command_and_cwd() {
        let dir = tempdir().unwrap();
        let runner = TestRunner::from_override("/bin/sh -c true", dir.path()).unwrap();
        let log_path = dir.path().join("nested").join("test.log");
        runner.run(&log_path).await.unwrap();
        let log = std::fs::read_to_string(&log_path).unwrap();
        assert!(log.starts_with("$ /bin/sh -c true"));
        assert!(log.contains(&format!("cwd: {}", dir.path().display())));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_truncates_existing_log_file() {
        let dir = tempdir().unwrap();
        let log_path = dir.path().join("test.log");
        std::fs::write(&log_path, "stale contents from prior run\n").unwrap();
        let runner = TestRunner::from_override("/bin/sh -c true", dir.path()).unwrap();
        runner.run(&log_path).await.unwrap();
        let log = std::fs::read_to_string(&log_path).unwrap();
        assert!(
            !log.contains("stale contents"),
            "log not truncated: {log:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_failure_summary_is_bounded_to_tail_lines() {
        // Emit many more lines than FAILURE_TAIL_LINES; the summary tail must
        // include only the final FAILURE_TAIL_LINES and never grow unbounded.
        let dir = tempdir().unwrap();
        let lines_to_emit = FAILURE_TAIL_LINES + 50;
        let script = format!(
            "for i in $(seq 1 {n}); do echo line-$i; done; exit 1",
            n = lines_to_emit
        );
        let runner = TestRunner {
            kind: TestRunnerKind::Override,
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script],
            workdir: dir.path().to_path_buf(),
        };
        let outcome = runner.run(dir.path().join("test.log")).await.unwrap();
        assert!(!outcome.passed);
        // Last line ("line-N") must appear; an early line ("line-1") must not.
        assert!(outcome.summary.contains(&format!("line-{}", lines_to_emit)));
        assert!(
            !outcome.summary.contains("line-1\n"),
            "summary should not include the very first line"
        );
        // Sanity: the summary contains a bounded number of newline-delimited
        // lines (header + tail), well below the total emitted.
        let summary_lines = outcome.summary.lines().count();
        assert!(summary_lines <= FAILURE_TAIL_LINES + 4);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_captures_stderr_with_marker() {
        let dir = tempdir().unwrap();
        let runner = TestRunner {
            kind: TestRunnerKind::Override,
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "echo on-stderr 1>&2; exit 1".into()],
            workdir: dir.path().to_path_buf(),
        };
        let outcome = runner.run(dir.path().join("test.log")).await.unwrap();
        assert!(!outcome.passed);
        let log = std::fs::read_to_string(&outcome.log_path).unwrap();
        assert!(
            log.contains("[stderr] on-stderr"),
            "log should mark stderr lines: {log:?}"
        );
    }

    #[tokio::test]
    async fn run_surfaces_spawn_failure() {
        let dir = tempdir().unwrap();
        let runner = TestRunner {
            kind: TestRunnerKind::Override,
            program: "/this/binary/does/not/exist".into(),
            args: Vec::new(),
            workdir: dir.path().to_path_buf(),
        };
        let err = runner.run(dir.path().join("test.log")).await.unwrap_err();
        assert!(
            format!("{err:#}").contains("spawn"),
            "expected spawn failure, got: {err:#}"
        );
    }
}
