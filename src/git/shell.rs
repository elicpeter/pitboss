//! Shell-out [`Git`] implementation backed by the `git` CLI.
//!
//! [`ShellGit::new`] takes the workspace path; every command is run with
//! `git -C <workspace>` so the implementation is independent of the calling
//! process's current directory.
//!
//! Commit operations supply an inline `user.name` / `user.email` via `-c`
//! flags so pitboss never depends on global git config — runs are reproducible
//! on hosts that have never seen a `~/.gitconfig`.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::process::Command;

use super::{CommitId, DiffStat, Git, GitError};

/// `gh` binary name resolved against `PATH`. Overridable per `ShellGit`
/// instance via [`ShellGit::with_gh_binary`] so tests can point at a fixture
/// script without touching `PATH`.
const DEFAULT_GH_BINARY: &str = "gh";

/// `Git` impl that invokes the local `git` CLI against a fixed workspace.
pub struct ShellGit {
    workspace: PathBuf,
    gh_binary: PathBuf,
}

impl ShellGit {
    /// Build a `ShellGit` rooted at `workspace`. The workspace must already be
    /// a git repository — `ShellGit` does not run `git init` itself.
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            gh_binary: PathBuf::from(DEFAULT_GH_BINARY),
        }
    }

    /// Override the `gh` binary path. Tests use this to substitute a fixture
    /// script; production callers should leave it at the default so `gh` is
    /// resolved against `PATH` like any other tool.
    pub fn with_gh_binary(mut self, binary: impl Into<PathBuf>) -> Self {
        self.gh_binary = binary.into();
        self
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

    /// Whether `path` is currently covered by a `.gitignore` rule. Exit 0 from
    /// `git check-ignore -q` means ignored, 1 means not, anything else is a
    /// real failure.
    async fn path_is_ignored(&self, path: &Path) -> Result<bool> {
        let path_str = path.to_string_lossy();
        let out = self
            .run("check_ignore", &["check-ignore", "-q", "--", &path_str])
            .await?;
        match out.status {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => Err(GitError::Command {
                operation: "check_ignore".into(),
                exit: out.status,
                stderr: out.stderr,
            }
            .into()),
        }
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
        //
        // Drop exclude entries already covered by `.gitignore`: naming an
        // ignored path in a `:!<path>` pathspec still trips `git add`'s
        // "paths are ignored" warning and a non-zero exit, even though the
        // path would be skipped silently anyway.
        let mut effective: Vec<&Path> = Vec::with_capacity(exclude.len());
        for p in exclude {
            if !self.path_is_ignored(p).await? {
                effective.push(p);
            }
        }
        let mut args: Vec<String> = vec!["add".into(), "-A".into(), "--".into(), ".".into()];
        for p in effective {
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
        cmd.args(["-c", "user.name=pitboss", "-c", "user.email=pitboss@local"])
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
        let head = self
            .run_succeed("rev-parse", &["rev-parse", "HEAD"])
            .await?;
        Ok(CommitId::new(head.stdout.trim().to_string()))
    }

    async fn diff_stat(&self, from: &str, to: &str) -> Result<DiffStat> {
        let range = format!("{from}..{to}");
        let out = self
            .run_succeed("diff_stat", &["diff", "--shortstat", &range])
            .await?;
        Ok(parse_shortstat(&out.stdout))
    }

    async fn staged_diff(&self) -> Result<String> {
        let out = self
            .run_succeed("staged_diff", &["diff", "--cached"])
            .await?;
        Ok(out.stdout)
    }

    async fn stash_push(&self, message: &str, exclude: &[&Path]) -> Result<bool> {
        // Pre-flight: a clean tree means there is nothing to stash. `git stash
        // push` on a clean tree exits 0 and prints "No local changes to save",
        // but emitting that as `false` lets callers skip the noise.
        if self.is_clean().await? {
            return Ok(false);
        }
        // Same `:!<path>` exclusion machinery `stage_changes` uses so callers
        // can keep paths like `.pitboss/` out of the stash. Skip excludes that
        // are already covered by `.gitignore`.
        let mut effective: Vec<&Path> = Vec::with_capacity(exclude.len());
        for p in exclude {
            if !self.path_is_ignored(p).await? {
                effective.push(p);
            }
        }
        let mut args: Vec<String> = vec![
            "stash".into(),
            "push".into(),
            "--include-untracked".into(),
            "-m".into(),
            message.to_string(),
            "--".into(),
            ".".into(),
        ];
        for p in effective {
            args.push(format!(":!{}", p.display()));
        }
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let out = self.run("stash_push", &arg_refs).await?;
        if !out.success {
            return Err(GitError::Command {
                operation: "stash_push".into(),
                exit: out.status,
                stderr: out.stderr,
            }
            .into());
        }
        // The pathspec form can succeed but leave nothing stashed when the
        // only dirty paths were excluded. Detect that by re-checking
        // cleanliness against the same exclusion set: if we're still dirty,
        // the residue is just the excluded paths and we report no stash.
        if self.is_clean().await? {
            // Working tree is fully clean now → something landed in the stash.
            Ok(true)
        } else {
            // Still dirty: confirm by checking whether the stash list has a
            // matching message. Cheaper than re-running pathspec logic.
            let stashes = self
                .run_succeed("stash_list", &["stash", "list"])
                .await?;
            Ok(stashes.stdout.contains(message))
        }
    }

    async fn open_pr(&self, title: &str, body: &str) -> Result<String> {
        // `gh` resolves the target repository from its working directory's git
        // remotes — there is no `-C` flag, so the workspace is passed via
        // `current_dir` instead of an argv entry.
        let mut cmd = Command::new(&self.gh_binary);
        cmd.current_dir(&self.workspace)
            .arg("pr")
            .arg("create")
            .arg("--title")
            .arg(title)
            .arg("--body")
            .arg(body)
            .env("GIT_TERMINAL_PROMPT", "0");
        let output = cmd.stdin(Stdio::null()).output().await.with_context(|| {
            format!("gh pr create: spawning child (binary {:?})", self.gh_binary)
        })?;
        let out: CommandOut = output.into();
        if !out.success {
            return Err(GitError::Command {
                operation: "open_pr".into(),
                exit: out.status,
                stderr: out.stderr,
            }
            .into());
        }
        // `gh pr create` prints the PR URL on the final stdout line. Take the
        // last non-empty line so any preamble (`Creating pull request...`)
        // doesn't leak into the returned URL.
        let url = out
            .stdout
            .lines()
            .map(str::trim)
            .rfind(|l| !l.is_empty())
            .unwrap_or("")
            .to_string();
        if url.is_empty() {
            return Err(GitError::UnexpectedOutput {
                operation: "open_pr".into(),
                output: "(gh pr create produced no URL on stdout)".into(),
            }
            .into());
        }
        Ok(url)
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
        for (k, v) in [
            ("user.name", "pitboss-test"),
            ("user.email", "pitboss@test"),
        ] {
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

        git.create_branch("pitboss/run-test").await.unwrap();
        // create_branch must not switch.
        assert_eq!(git.current_branch().await.unwrap(), starting);

        git.checkout("pitboss/run-test").await.unwrap();
        assert_eq!(git.current_branch().await.unwrap(), "pitboss/run-test");
    }

    #[tokio::test]
    async fn stage_changes_excludes_planning_artifacts() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        // Mirror the runner's per-phase situation: the agent left planning
        // artifacts and `.pitboss/` updates in the working tree alongside one
        // real source change. Only the source change should be staged.
        fs::write(dir.path().join("plan.md"), "plan body\n").unwrap();
        fs::write(dir.path().join("deferred.md"), "deferred body\n").unwrap();
        fs::create_dir_all(dir.path().join(".pitboss")).unwrap();
        fs::write(dir.path().join(".pitboss/state.json"), "{}\n").unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "fn main() {}\n").unwrap();

        let plan_path = Path::new("plan.md");
        let deferred_path = Path::new("deferred.md");
        let pitboss_path = Path::new(".pitboss");
        git.stage_changes(&[plan_path, deferred_path, pitboss_path])
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
    async fn stage_changes_tolerates_excluded_path_already_in_gitignore() {
        // Regression: when `.pitboss/` is in `.gitignore` AND on disk, naming
        // it in a `:!` exclude pathspec used to trip git's "paths are ignored"
        // error and fail the stage. `stage_changes` now filters such entries
        // out so the call succeeds and only the real source change is staged.
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        fs::write(dir.path().join(".gitignore"), ".pitboss/\n").unwrap();
        fs::create_dir_all(dir.path().join(".pitboss")).unwrap();
        fs::write(dir.path().join(".pitboss/state.json"), "{}\n").unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "fn main() {}\n").unwrap();

        git.stage_changes(&[
            Path::new("plan.md"),
            Path::new("deferred.md"),
            Path::new(".pitboss"),
        ])
        .await
        .unwrap();

        let staged = std::process::Command::new("git")
            .args(["-C"])
            .arg(dir.path())
            .args(["diff", "--cached", "--name-only"])
            .output()
            .unwrap();
        let staged = String::from_utf8(staged.stdout).unwrap();
        let mut lines: Vec<&str> = staged.lines().collect();
        lines.sort();
        assert_eq!(lines, vec![".gitignore", "src/foo.rs"], "staged: {lines:?}");
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
        // The runner contract: if the agent only modified `.pitboss/`,
        // `plan.md`, or `deferred.md`, `stage_changes` finds nothing to stage
        // and `has_staged_changes` returns false. Runner skips commit.
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        fs::write(dir.path().join("plan.md"), "plan body\n").unwrap();
        fs::write(dir.path().join("deferred.md"), "deferred body\n").unwrap();
        fs::create_dir_all(dir.path().join(".pitboss")).unwrap();
        fs::write(dir.path().join(".pitboss/state.json"), "{}\n").unwrap();

        git.stage_changes(&[
            Path::new("plan.md"),
            Path::new("deferred.md"),
            Path::new(".pitboss"),
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

        let id = git.commit("[pitboss] phase 01: seed").await.unwrap();
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
        let err = git.commit("[pitboss] empty").await.unwrap_err();
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
        let from = git.commit("[pitboss] phase 01: a").await.unwrap();

        // Second commit modifying the same file.
        fs::write(dir.path().join("src/foo.rs"), "a\nB\nc\nd\n").unwrap();
        git.stage_changes(&[]).await.unwrap();
        let to = git.commit("[pitboss] phase 02: b").await.unwrap();

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

    #[tokio::test]
    async fn staged_diff_reflects_index_contents() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        // No index → empty diff.
        assert_eq!(git.staged_diff().await.unwrap().trim(), "");

        // Stage a new file and an edit; both should show up in the diff.
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "fn main() {}\n").unwrap();
        git.stage_changes(&[]).await.unwrap();
        let diff = git.staged_diff().await.unwrap();
        assert!(diff.contains("src/foo.rs"), "diff: {diff}");
        assert!(diff.contains("+fn main()"), "diff: {diff}");
    }

    #[tokio::test]
    async fn staged_diff_excludes_paths_excluded_by_stage_changes() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path());

        // Mirror the runner's pre-audit setup: implementer touched both
        // planning artifacts and code; only code should make it into the diff.
        fs::write(dir.path().join("plan.md"), "plan body\n").unwrap();
        fs::write(dir.path().join("deferred.md"), "deferred body\n").unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/foo.rs"), "fn main() {}\n").unwrap();
        git.stage_changes(&[
            Path::new("plan.md"),
            Path::new("deferred.md"),
            Path::new(".pitboss"),
        ])
        .await
        .unwrap();

        let diff = git.staged_diff().await.unwrap();
        assert!(diff.contains("src/foo.rs"), "diff: {diff}");
        assert!(!diff.contains("plan.md"), "diff leaked plan.md: {diff}");
        assert!(
            !diff.contains("deferred.md"),
            "diff leaked deferred.md: {diff}"
        );
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

    /// Resolve the path of a fixture script under `tests/fixtures/`.
    fn fixture_path(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    #[tokio::test]
    async fn open_pr_returns_url_and_passes_title_and_body_to_gh() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path()).with_gh_binary(fixture_path("fake-gh-success.sh"));

        let url = git
            .open_pr(
                "pitboss: phase 01 — Foundation",
                "## Run\n\nbody body body\n",
            )
            .await
            .unwrap();
        assert_eq!(url, "https://github.com/example/repo/pull/42");

        // The fake logs its invocation into `.gh-fake-log` inside cwd, which
        // confirms both the argv it saw and that we ran it from the workspace.
        let log_path = dir.path().join(".gh-fake-log");
        let log = std::fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("pr"), "fake log: {log}");
        assert!(log.contains("create"), "fake log: {log}");
        assert!(log.contains("--title"), "fake log: {log}");
        assert!(
            log.contains("pitboss: phase 01 — Foundation"),
            "fake log: {log}"
        );
        assert!(log.contains("--body"), "fake log: {log}");
        assert!(log.contains("body body body"), "fake log: {log}");
        let workspace_real = std::fs::canonicalize(dir.path()).unwrap();
        let logged_cwd_line = log
            .lines()
            .find(|l| l.starts_with("cwd:"))
            .expect("cwd line in fake log");
        let logged_cwd = logged_cwd_line.trim_start_matches("cwd:").trim();
        let logged_cwd = std::fs::canonicalize(logged_cwd).unwrap();
        assert_eq!(logged_cwd, workspace_real);
    }

    #[tokio::test]
    async fn open_pr_surfaces_failure_with_stderr() {
        let dir = fresh_repo().await;
        let git = ShellGit::new(dir.path()).with_gh_binary(fixture_path("fake-gh-failure.sh"));
        let err = git.open_pr("title", "body").await.unwrap_err();
        let chain = format!("{err:#}");
        assert!(chain.contains("open_pr"), "chain: {chain}");
        assert!(
            chain.contains("could not determine the base repository"),
            "chain: {chain}"
        );
    }
}
