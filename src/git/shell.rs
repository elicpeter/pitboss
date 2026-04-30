//! Shell-out [`Git`] implementation backed by the `git` CLI.
//!
//! [`ShellGit::new`] takes the workspace path; every command is run with
//! `git -C <workspace>` so the implementation is independent of the calling
//! process's current directory.
//!
//! Commit operations supply an inline `user.name` / `user.email` via `-c`
//! flags so foreman never depends on global git config — runs are reproducible
//! on hosts that have never seen a `~/.gitconfig`.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::process::Command;

use super::{CommitId, DiffStat, Git, GitError};

/// `Git` impl that invokes the local `git` CLI against a fixed workspace.
pub struct ShellGit {
    workspace: PathBuf,
}

impl ShellGit {
    /// Build a `ShellGit` rooted at `workspace`. The workspace must already be
    /// a git repository — `ShellGit` does not run `git init` itself.
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
        }
    }

    /// Workspace this `ShellGit` was constructed against.
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    fn cmd(&self) -> Command {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(&self.workspace);
        // Refuse to prompt on TTY-less environments (CI, the runner).
        cmd.env("GIT_TERMINAL_PROMPT", "0");
        cmd
    }

    async fn run(&self, op: &'static str, args: &[&str]) -> Result<CommandOut> {
        let mut cmd = self.cmd();
        cmd.args(args);
        let output = cmd
            .stdin(Stdio::null())
            .output()
            .await
            .with_context(|| format!("git {op}: spawning child"))?;
        Ok(CommandOut::from(output))
    }

    async fn run_succeed(&self, op: &'static str, args: &[&str]) -> Result<CommandOut> {
        let out = self.run(op, args).await?;
        if !out.success {
            return Err(GitError::Command {
                operation: op.into(),
                exit: out.status,
                stderr: out.stderr.clone(),
            }
            .into());
        }
        Ok(out)
    }
}

struct CommandOut {
    success: bool,
    status: Option<i32>,
    stdout: String,
    stderr: String,
}

impl From<std::process::Output> for CommandOut {
    fn from(o: std::process::Output) -> Self {
        Self {
            success: o.status.success(),
            status: o.status.code(),
            stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
        }
    }
}

#[async_trait]
impl Git for ShellGit {
    async fn is_clean(&self) -> Result<bool> {
        let out = self
            .run_succeed("status", &["status", "--porcelain"])
            .await?;
        Ok(out.stdout.trim().is_empty())
    }

    async fn current_branch(&self) -> Result<String> {
        let out = self
            .run_succeed("current_branch", &["branch", "--show-current"])
            .await?;
        let name = out.stdout.trim().to_string();
        if name.is_empty() {
            return Err(GitError::UnexpectedOutput {
                operation: "current_branch".into(),
                output: "(detached HEAD or unborn branch)".into(),
            }
            .into());
        }
        Ok(name)
    }

    async fn create_branch(&self, name: &str) -> Result<()> {
        self.run_succeed("create_branch", &["branch", name]).await?;
        Ok(())
    }

    async fn checkout(&self, name: &str) -> Result<()> {
        self.run_succeed("checkout", &["checkout", name]).await?;
        Ok(())
    }

    async fn stage_changes(&self, exclude: &[&Path]) -> Result<()> {
        // Pathspec form: `git add -A -- . :!<excluded>...`. The leading `--`
        // makes everything after it a path/pathspec and disables further
        // option parsing, which protects us if a user-supplied path starts
        // with `-`.
        let mut args: Vec<String> = vec!["add".into(), "-A".into(), "--".into(), ".".into()];
        for p in exclude {
            args.push(format!(":!{}", p.display()));
        }
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        self.run_succeed("stage_changes", &arg_refs).await?;
        Ok(())
    }

    async fn has_staged_changes(&self) -> Result<bool> {
        // `git diff --cached --quiet` exits 0 when index == HEAD (no staged
        // changes), 1 when there *are* staged changes, anything else on real
        // failure. Compares against the empty tree on an unborn HEAD, so it
        // remains correct before the very first commit.
        let out = self
            .run("has_staged_changes", &["diff", "--cached", "--quiet"])
            .await?;
        match out.status {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(GitError::Command {
                operation: "has_staged_changes".into(),
                exit: out.status,
                stderr: out.stderr,
            }
            .into()),
        }
    }

    async fn commit(&self, message: &str) -> Result<CommitId> {
        let mut cmd = self.cmd();
        cmd.args([
            "-c",
            "user.name=foreman",
            "-c",
            "user.email=foreman@local",
        ])
        .arg("commit")
        .arg("-m")
        .arg(message);
        let out: CommandOut = cmd
            .stdin(Stdio::null())
            .output()
            .await
            .context("git commit: spawning child")?
            .into();
        if !out.success {
            return Err(GitError::Command {
                operation: "commit".into(),
                exit: out.status,
                stderr: out.stderr,
            }
            .into());
        }
        let head = self.run_succeed("rev-parse", &["rev-parse", "HEAD"]).await?;
        Ok(CommitId::new(head.stdout.trim().to_string()))
    }

    async fn diff_stat(&self, from: &str, to: &str) -> Result<DiffStat> {
        let range = format!("{from}..{to}");
        let out = self
            .run_succeed("diff_stat", &["diff", "--shortstat", &range])
            .await?;
        Ok(parse_shortstat(&out.stdout))
    }
}

/// Parse `git diff --shortstat` output. Examples:
///
/// ```text
/// (empty)
///  3 files changed, 22 insertions(+), 7 deletions(-)
///  1 file changed, 5 insertions(+)
///  1 file changed, 2 deletions(-)
/// ```
///
/// Unrecognized fragments are ignored rather than failing, since git's exact
/// wording can shift between versions.
fn parse_shortstat(s: &str) -> DiffStat {
    let mut stat = DiffStat::default();
    for piece in s.split(',') {
        let piece = piece.trim();
        let Some((n, label)) = piece.split_once(' ') else {
            continue;
        };
        let Ok(n) = n.parse::<u64>() else {
            continue;
        };
        if label.starts_with("file") {
            stat.files_changed = n;
        } else if label.starts_with("insertion") {
            stat.insertions = n;
        } else if label.starts_with("deletion") {
            stat.deletions = n;
        }
    }
    stat
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Initialize a fresh repo and seed an initial commit so HEAD points
    /// somewhere — `git branch <name>` and most other operations require it.
    async fn fresh_repo() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        // `init.defaultBranch` is set inline so this works on any git version
        // (>=2.28 honors it; older versions ignore it and create `master`,
        // which our tests handle by reading the branch name dynamically).
        let status = std::process::Command::new("git")
            .args(["-c", "init.defaultBranch=main", "init", "-q"])
            .arg(dir.path())
            .status()
            .unwrap();
        assert!(status.success());
        // Identity for the seed commit.
        for (k, v) in [("user.name", "foreman-test"), ("user.email", "foreman@test")] {
            std::process::Command::new("git")
                .args(["-C"])
                .arg(dir.path())
                .args(["config", k, v])
                .status()
                .unwrap();
        }
        // Empty seed commit so HEAD exists.
        std::process::Command::new("git")
            .args(["-C"])
            .arg(dir.path())
            .args(["commit", "--allow-empty", "-m", "seed", "-q"])
            .status()
            .unwrap();
        dir
    }

    #[tokio::test]
    async fn is_clean_distinguishes_clean_and_dirty() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());
        assert!(git.is_clean().await.unwrap());
        fs::write(dir.path().join("untracked.txt"), b"hi").unwrap();
        assert!(!git.is_clean().await.unwrap());
    }

    #[tokio::test]
    async fn current_branch_returns_initial_branch() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());
        let name = git.current_branch().await.unwrap();
        // We don't assert "main" because git <2.28 ignores init.defaultBranch.
        assert!(!name.is_empty(), "branch should be non-empty");
        assert!(
            name == "main" || name == "master",
            "unexpected initial branch {name:?}",
        );
    }

    #[tokio::test]
    async fn create_and_checkout_branch_round_trip() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());
        let starting = git.current_branch().await.unwrap();

        git.create_branch("foreman/run-test").await.unwrap();
        // create_branch must not switch.
        assert_eq!(git.current_branch().await.unwrap(), starting);

        git.checkout("foreman/run-test").await.unwrap();
        assert_eq!(git.current_branch().await.unwrap(), "foreman/run-test");
    }

    #[tokio::test]
    async fn stage_changes_excludes_planning_artifacts() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        // Mirror the runner's per-phase situation: the agent left planning
        // artifacts and `.foreman/` updates in the working tree alongside one
        // real source change. Only the source change should be staged.
        fs::write(dir.path().join("plan.md"), "plan body\n").unwrap();
        fs::write(dir.path().join("deferred.md"), "deferred body\n").unwrap();
        fs::create_dir_all(dir.path().join(".foreman")).unwrap();
        fs::write(dir.path().join(".foreman/state.json"), "{}\n").unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "fn main() {}\n").unwrap();

        let plan_path = Path::new("plan.md");
        let deferred_path = Path::new("deferred.md");
        let foreman_path = Path::new(".foreman");
        git.stage_changes(&[plan_path, deferred_path, foreman_path])
            .await
            .unwrap();

        // Inspect the index directly: only `src/foo.rs` should be there.
        let staged = std::process::Command::new("git")
            .args(["-C"])
            .arg(dir.path())
            .args(["diff", "--cached", "--name-only"])
            .output()
            .unwrap();
        let staged = String::from_utf8(staged.stdout).unwrap();
        let lines: Vec<&str> = staged.lines().collect();
        assert_eq!(lines, vec!["src/foo.rs"], "staged set: {lines:?}");
    }

    #[tokio::test]
    async fn has_staged_changes_reflects_index_state() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        // Empty index after seed commit.
        assert!(!git.has_staged_changes().await.unwrap());

        // Untracked alone is not staged.
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "fn main() {}\n").unwrap();
        assert!(!git.has_staged_changes().await.unwrap());

        // Stage it.
        git.stage_changes(&[]).await.unwrap();
        assert!(git.has_staged_changes().await.unwrap());
    }

    #[tokio::test]
    async fn empty_commit_path_when_only_excluded_files_changed() {
        // The runner contract: if the agent only modified `.foreman/`,
        // `plan.md`, or `deferred.md`, `stage_changes` finds nothing to stage
        // and `has_staged_changes` returns false. Runner skips commit.
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        fs::write(dir.path().join("plan.md"), "plan body\n").unwrap();
        fs::write(dir.path().join("deferred.md"), "deferred body\n").unwrap();
        fs::create_dir_all(dir.path().join(".foreman")).unwrap();
        fs::write(dir.path().join(".foreman/state.json"), "{}\n").unwrap();

        git.stage_changes(&[
            Path::new("plan.md"),
            Path::new("deferred.md"),
            Path::new(".foreman"),
        ])
        .await
        .unwrap();

        assert!(
            !git.has_staged_changes().await.unwrap(),
            "only-excluded changes should produce an empty index"
        );
    }

    #[tokio::test]
    async fn commit_returns_resolvable_commit_id() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "fn main() {}\n").unwrap();
        git.stage_changes(&[]).await.unwrap();

        let id = git.commit("[foreman] phase 01: seed").await.unwrap();
        assert_eq!(id.as_str().len(), 40, "commit id: {id}");

        // The id resolves to the same commit `git rev-parse` returns.
        let head = std::process::Command::new("git")
            .args(["-C"])
            .arg(dir.path())
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap();
        let head = String::from_utf8(head.stdout).unwrap().trim().to_string();
        assert_eq!(head, id.as_str());
    }

    #[tokio::test]
    async fn commit_with_empty_index_errors() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());
        let err = git.commit("[foreman] empty").await.unwrap_err();
        let chain = format!("{err:#}");
        assert!(chain.contains("commit"), "err chain: {chain}");
    }

    #[tokio::test]
    async fn diff_stat_reports_change_summary() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        // First commit on top of seed.
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "a\nb\nc\n").unwrap();
        git.stage_changes(&[]).await.unwrap();
        let from = git.commit("[foreman] phase 01: a").await.unwrap();

        // Second commit modifying the same file.
        fs::write(dir.path().join("src/foo.rs"), "a\nB\nc\nd\n").unwrap();
        git.stage_changes(&[]).await.unwrap();
        let to = git.commit("[foreman] phase 02: b").await.unwrap();

        let stat = git.diff_stat(from.as_str(), to.as_str()).await.unwrap();
        assert_eq!(stat.files_changed, 1);
        assert!(stat.insertions >= 1);
        assert!(stat.deletions >= 1);
    }

    #[tokio::test]
    async fn diff_stat_empty_range_is_default() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());
        // Seed..seed is empty.
        let head = std::process::Command::new("git")
            .args(["-C"])
            .arg(dir.path())
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap();
        let head = String::from_utf8(head.stdout).unwrap().trim().to_string();
        let stat = git.diff_stat(&head, &head).await.unwrap();
        assert_eq!(stat, DiffStat::default());
    }

    #[test]
    fn parse_shortstat_handles_known_shapes() {
        assert_eq!(parse_shortstat(""), DiffStat::default());
        assert_eq!(
            parse_shortstat(" 3 files changed, 22 insertions(+), 7 deletions(-)\n"),
            DiffStat {
                files_changed: 3,
                insertions: 22,
                deletions: 7,
            }
        );
        assert_eq!(
            parse_shortstat(" 1 file changed, 5 insertions(+)\n"),
            DiffStat {
                files_changed: 1,
                insertions: 5,
                deletions: 0,
            }
        );
        assert_eq!(
            parse_shortstat(" 1 file changed, 2 deletions(-)\n"),
            DiffStat {
                files_changed: 1,
                insertions: 0,
                deletions: 2,
            }
        );
    }

    #[tokio::test]
    async fn command_failure_surfaces_typed_error() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());
        let err = git.checkout("does-not-exist").await.unwrap_err();
        // anyhow chain should surface our typed error message.
        let chain = format!("{err:#}");
        assert!(chain.contains("checkout"), "chain: {chain}");
    }
}
