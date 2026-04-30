//! In-memory [`Git`] implementation used by runner tests.
//!
//! `MockGit` synthesizes the slice of git state the runner cares about:
//! a current branch, a set of branches that exist, a "working tree" of
//! pseudo-modified paths, an index, and a commit log. Tests drive the working
//! tree directly with [`MockGit::touch`] / [`MockGit::clear`]; every trait
//! call appends to a journal so tests can assert exactly which operations the
//! runner performed and in what order.
//!
//! `MockGit` is **always** compiled, not gated behind `#[cfg(test)]`, because
//! integration tests under `tests/` consume it as a regular dependency.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use super::{CommitId, DiffStat, Git};

/// One entry in the [`MockGit`] operation journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockOp {
    /// `is_clean` was called.
    IsClean,
    /// `current_branch` was called.
    CurrentBranch,
    /// `create_branch(name)` was called.
    CreateBranch(String),
    /// `checkout(name)` was called.
    Checkout(String),
    /// `stage_changes(exclude)` was called; carries the exclusion set.
    StageChanges(Vec<PathBuf>),
    /// `has_staged_changes` was called.
    HasStagedChanges,
    /// `commit(message)` was called.
    Commit(String),
    /// `diff_stat(from, to)` was called.
    DiffStat(String, String),
    /// `staged_diff` was called.
    StagedDiff,
    /// `open_pr(title, body)` was called.
    OpenPr {
        /// PR title passed in.
        title: String,
        /// PR body passed in.
        body: String,
    },
    /// `stash_push(message)` was called.
    StashPush(String),
}

/// Record of a single commit in [`MockGit`]'s in-memory log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockCommit {
    /// Synthesized commit hash (`mock0000...0001` style).
    pub id: CommitId,
    /// Branch the commit landed on.
    pub branch: String,
    /// Commit message verbatim.
    pub message: String,
    /// Files included in the commit, in sorted order.
    pub files: Vec<PathBuf>,
}

#[derive(Debug)]
struct MockState {
    current_branch: String,
    branches: HashSet<String>,
    working_tree: HashSet<PathBuf>,
    staged: HashSet<PathBuf>,
    commits: Vec<MockCommit>,
    ops: Vec<MockOp>,
    next_commit_seq: u64,
    /// Synthetic `git diff --cached` output. Tests set this to feed the runner
    /// auditor a representative diff; the mock has no real index/working tree
    /// to compute one from.
    staged_diff: String,
    /// Result the next `open_pr` call should produce. Defaults to a synthetic
    /// success URL so tests that don't care about the value still get a
    /// non-error return.
    open_pr_response: Result<String, String>,
}

impl MockState {
    fn new(branch: String) -> Self {
        let mut branches = HashSet::new();
        branches.insert(branch.clone());
        Self {
            current_branch: branch,
            branches,
            working_tree: HashSet::new(),
            staged: HashSet::new(),
            commits: Vec::new(),
            ops: Vec::new(),
            next_commit_seq: 0,
            staged_diff: String::new(),
            open_pr_response: Ok("https://github.com/example/repo/pull/1".to_string()),
        }
    }
}

/// In-memory test double for [`Git`]. Cheap to construct; thread-safe via
/// internal `Mutex`. See module docs for the expected usage pattern.
pub struct MockGit {
    state: Mutex<MockState>,
}

impl MockGit {
    /// New mock starting on a `main` branch with no working-tree changes.
    pub fn new() -> Self {
        Self::with_branch("main")
    }

    /// New mock starting on a custom initial branch.
    pub fn with_branch(branch: impl Into<String>) -> Self {
        Self {
            state: Mutex::new(MockState::new(branch.into())),
        }
    }

    /// Mark `path` as a working-tree change so the next `stage_changes` call
    /// will pick it up (modulo exclusions).
    pub fn touch(&self, path: impl Into<PathBuf>) {
        self.state.lock().unwrap().working_tree.insert(path.into());
    }

    /// Drop a path from the synthetic working tree, e.g., to simulate a user
    /// reverting an edit.
    pub fn clear(&self, path: impl AsRef<Path>) {
        self.state
            .lock()
            .unwrap()
            .working_tree
            .remove(path.as_ref());
    }

    /// Snapshot of every commit recorded so far, oldest first.
    pub fn commits(&self) -> Vec<MockCommit> {
        self.state.lock().unwrap().commits.clone()
    }

    /// Snapshot of the operation journal in call order.
    pub fn ops(&self) -> Vec<MockOp> {
        self.state.lock().unwrap().ops.clone()
    }

    /// Set the canned `git diff --cached` output the next [`Git::staged_diff`]
    /// call returns. Tests use this to drive the runner's audit branch with a
    /// representative diff string.
    pub fn set_staged_diff(&self, diff: impl Into<String>) {
        self.state.lock().unwrap().staged_diff = diff.into();
    }

    /// Make the next [`Git::open_pr`] call return `Ok(url)`.
    pub fn set_open_pr_response(&self, url: impl Into<String>) {
        self.state.lock().unwrap().open_pr_response = Ok(url.into());
    }

    /// Make the next [`Git::open_pr`] call fail with `message`.
    pub fn set_open_pr_failure(&self, message: impl Into<String>) {
        self.state.lock().unwrap().open_pr_response = Err(message.into());
    }

    /// Most recent exclusion set passed to `stage_changes`, or `None` if it
    /// was never called.
    pub fn last_exclusions(&self) -> Option<Vec<PathBuf>> {
        self.state
            .lock()
            .unwrap()
            .ops
            .iter()
            .rev()
            .find_map(|op| match op {
                MockOp::StageChanges(p) => Some(p.clone()),
                _ => None,
            })
    }
}

impl Default for MockGit {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Git for MockGit {
    async fn is_clean(&self) -> Result<bool> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::IsClean);
        Ok(s.working_tree.is_empty() && s.staged.is_empty())
    }

    async fn current_branch(&self) -> Result<String> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::CurrentBranch);
        Ok(s.current_branch.clone())
    }

    async fn create_branch(&self, name: &str) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::CreateBranch(name.to_string()));
        if !s.branches.insert(name.to_string()) {
            return Err(anyhow!("mock-git: branch {name:?} already exists"));
        }
        Ok(())
    }

    async fn checkout(&self, name: &str) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::Checkout(name.to_string()));
        if !s.branches.contains(name) {
            return Err(anyhow!("mock-git: cannot checkout unknown branch {name:?}"));
        }
        s.current_branch = name.to_string();
        Ok(())
    }

    async fn stage_changes(&self, exclude: &[&Path]) -> Result<()> {
        let exclude_paths: Vec<PathBuf> = exclude.iter().map(|p| p.to_path_buf()).collect();
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::StageChanges(exclude_paths.clone()));
        let exclude_set: HashSet<PathBuf> = exclude_paths.into_iter().collect();
        let to_stage: Vec<PathBuf> = s
            .working_tree
            .iter()
            .filter(|p| !is_excluded(p, &exclude_set))
            .cloned()
            .collect();
        for p in to_stage {
            s.working_tree.remove(&p);
            s.staged.insert(p);
        }
        Ok(())
    }

    async fn has_staged_changes(&self) -> Result<bool> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::HasStagedChanges);
        Ok(!s.staged.is_empty())
    }

    async fn commit(&self, message: &str) -> Result<CommitId> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::Commit(message.to_string()));
        if s.staged.is_empty() {
            return Err(anyhow!("mock-git: commit with empty index"));
        }
        s.next_commit_seq += 1;
        let id = CommitId::new(format!("mock{:040}", s.next_commit_seq));
        let mut files: Vec<PathBuf> = s.staged.drain().collect();
        files.sort();
        let branch = s.current_branch.clone();
        s.commits.push(MockCommit {
            id: id.clone(),
            branch,
            message: message.to_string(),
            files,
        });
        Ok(id)
    }

    async fn diff_stat(&self, from: &str, to: &str) -> Result<DiffStat> {
        let mut s = self.state.lock().unwrap();
        s.ops
            .push(MockOp::DiffStat(from.to_string(), to.to_string()));
        Ok(DiffStat::default())
    }

    async fn staged_diff(&self) -> Result<String> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::StagedDiff);
        Ok(s.staged_diff.clone())
    }

    async fn stash_push(&self, message: &str, exclude: &[&Path]) -> Result<bool> {
        let exclude_paths: Vec<PathBuf> = exclude.iter().map(|p| p.to_path_buf()).collect();
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::StashPush(message.to_string()));
        if s.working_tree.is_empty() && s.staged.is_empty() {
            return Ok(false);
        }
        let exclude_set: std::collections::HashSet<PathBuf> = exclude_paths.into_iter().collect();
        let mut moved = false;
        let to_clear: Vec<PathBuf> = s
            .working_tree
            .iter()
            .filter(|p| !is_excluded(p, &exclude_set))
            .cloned()
            .collect();
        for p in to_clear {
            s.working_tree.remove(&p);
            moved = true;
        }
        let staged_clear: Vec<PathBuf> = s
            .staged
            .iter()
            .filter(|p| !is_excluded(p, &exclude_set))
            .cloned()
            .collect();
        for p in staged_clear {
            s.staged.remove(&p);
            moved = true;
        }
        Ok(moved)
    }

    async fn open_pr(&self, title: &str, body: &str) -> Result<String> {
        let mut s = self.state.lock().unwrap();
        s.ops.push(MockOp::OpenPr {
            title: title.to_string(),
            body: body.to_string(),
        });
        match &s.open_pr_response {
            Ok(url) => Ok(url.clone()),
            Err(msg) => Err(anyhow!("mock-git: open_pr failed: {msg}")),
        }
    }
}

/// A path is excluded if it equals or is nested under any path in `exclude`.
/// Mirrors the behavior of `git add -- :!<path>` for directory exclusions.
fn is_excluded(path: &Path, exclude: &HashSet<PathBuf>) -> bool {
    exclude
        .iter()
        .any(|ex| path == ex.as_path() || path.starts_with(ex))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fresh_mock_is_clean_on_new_branch() {
        let git = MockGit::new();
        assert!(git.is_clean().await.unwrap());
        assert_eq!(git.current_branch().await.unwrap(), "main");
    }

    #[tokio::test]
    async fn create_then_checkout_switches_current_branch() {
        let git = MockGit::new();
        git.create_branch("pitboss/run-x").await.unwrap();
        assert_eq!(git.current_branch().await.unwrap(), "main");
        git.checkout("pitboss/run-x").await.unwrap();
        assert_eq!(git.current_branch().await.unwrap(), "pitboss/run-x");
    }

    #[tokio::test]
    async fn duplicate_branch_creation_errors() {
        let git = MockGit::new();
        git.create_branch("dup").await.unwrap();
        let err = git.create_branch("dup").await.unwrap_err();
        assert!(format!("{err}").contains("already exists"));
    }

    #[tokio::test]
    async fn checkout_unknown_branch_errors() {
        let git = MockGit::new();
        let err = git.checkout("missing").await.unwrap_err();
        assert!(format!("{err}").contains("unknown branch"));
    }

    #[tokio::test]
    async fn stage_changes_records_exclusions_and_filters_working_tree() {
        let git = MockGit::new();
        git.touch("src/foo.rs");
        git.touch("plan.md");
        git.touch("deferred.md");
        git.touch(".pitboss/state.json");

        let plan = Path::new("plan.md");
        let deferred = Path::new("deferred.md");
        let pitboss = Path::new(".pitboss");
        git.stage_changes(&[plan, deferred, pitboss]).await.unwrap();

        let exclusions = git.last_exclusions().unwrap();
        assert_eq!(
            exclusions,
            vec![
                PathBuf::from("plan.md"),
                PathBuf::from("deferred.md"),
                PathBuf::from(".pitboss"),
            ]
        );

        // Index should now hold only `src/foo.rs`.
        assert!(git.has_staged_changes().await.unwrap());
        let id = git.commit("[pitboss] phase 01: code only").await.unwrap();
        let commits = git.commits();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].id, id);
        assert_eq!(commits[0].files, vec![PathBuf::from("src/foo.rs")]);
        assert_eq!(commits[0].branch, "main");
        assert_eq!(commits[0].message, "[pitboss] phase 01: code only");
    }

    #[tokio::test]
    async fn empty_index_path_when_only_excluded_files_changed() {
        let git = MockGit::new();
        git.touch("plan.md");
        git.touch(".pitboss/state.json");
        git.stage_changes(&[Path::new("plan.md"), Path::new(".pitboss")])
            .await
            .unwrap();
        assert!(!git.has_staged_changes().await.unwrap());
    }

    #[tokio::test]
    async fn commit_with_empty_index_errors() {
        let git = MockGit::new();
        let err = git.commit("nothing").await.unwrap_err();
        assert!(format!("{err}").contains("empty index"));
    }

    #[tokio::test]
    async fn ops_journal_records_each_call_in_order() {
        let git = MockGit::new();
        git.touch("src/foo.rs");
        git.is_clean().await.unwrap();
        git.create_branch("b").await.unwrap();
        git.checkout("b").await.unwrap();
        git.stage_changes(&[Path::new("plan.md")]).await.unwrap();
        git.has_staged_changes().await.unwrap();
        git.commit("msg").await.unwrap();
        git.diff_stat("a", "b").await.unwrap();

        let ops = git.ops();
        assert_eq!(
            ops,
            vec![
                MockOp::IsClean,
                MockOp::CreateBranch("b".into()),
                MockOp::Checkout("b".into()),
                MockOp::StageChanges(vec![PathBuf::from("plan.md")]),
                MockOp::HasStagedChanges,
                MockOp::Commit("msg".into()),
                MockOp::DiffStat("a".into(), "b".into()),
            ]
        );
    }

    #[tokio::test]
    async fn diff_stat_returns_default_on_mock() {
        let git = MockGit::new();
        let stat = git.diff_stat("x", "y").await.unwrap();
        assert_eq!(stat, DiffStat::default());
    }

    #[tokio::test]
    async fn open_pr_records_op_and_returns_canned_url() {
        let git = MockGit::new();
        // Default response is a synthetic URL.
        let url = git.open_pr("title", "body").await.unwrap();
        assert_eq!(url, "https://github.com/example/repo/pull/1");

        // Override the canned response.
        git.set_open_pr_response("https://github.com/example/repo/pull/77");
        let url = git.open_pr("t2", "b2").await.unwrap();
        assert_eq!(url, "https://github.com/example/repo/pull/77");

        let ops = git.ops();
        assert_eq!(
            ops,
            vec![
                MockOp::OpenPr {
                    title: "title".into(),
                    body: "body".into()
                },
                MockOp::OpenPr {
                    title: "t2".into(),
                    body: "b2".into()
                },
            ]
        );
    }

    #[tokio::test]
    async fn open_pr_failure_response_surfaces_error() {
        let git = MockGit::new();
        git.set_open_pr_failure("no remote");
        let err = git.open_pr("t", "b").await.unwrap_err();
        assert!(format!("{err}").contains("no remote"));
    }

    #[tokio::test]
    async fn staged_diff_returns_canned_text_and_records_op() {
        let git = MockGit::new();
        assert_eq!(git.staged_diff().await.unwrap(), "");
        git.set_staged_diff("diff --git a/foo b/foo\n+new line\n");
        let diff = git.staged_diff().await.unwrap();
        assert!(diff.contains("+new line"));
        let ops = git.ops();
        assert_eq!(
            ops,
            vec![MockOp::StagedDiff, MockOp::StagedDiff],
            "ops: {ops:?}"
        );
    }

    #[test]
    fn is_excluded_treats_directory_paths_as_prefixes() {
        let mut set = HashSet::new();
        set.insert(PathBuf::from(".pitboss"));
        assert!(is_excluded(Path::new(".pitboss"), &set));
        assert!(is_excluded(Path::new(".pitboss/state.json"), &set));
        assert!(is_excluded(Path::new(".pitboss/logs/x.log"), &set));
        assert!(!is_excluded(Path::new("src/foo.rs"), &set));
        assert!(!is_excluded(Path::new(".pitbossx"), &set));
    }
}
