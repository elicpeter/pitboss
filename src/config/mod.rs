//! `foreman.toml` schema and loader.
//!
//! `Config` is the typed view of the project's `foreman.toml`. Every field has
//! a sensible default, so a missing file or missing section round-trips to the
//! same shape as a fully populated one. Unknown keys are logged via
//! [`tracing::warn`] and otherwise ignored, so a forward-compatible config
//! written by a newer foreman can still be loaded by an older binary.
//!
//! [`load`] reads `<workspace>/foreman.toml`, returning [`Config::default()`]
//! when the file is missing. [`parse`] is the same logic against an in-memory
//! string, used by both [`load`] and the unit tests.
//!
//! Type errors (wrong-type values, malformed TOML) surface as anyhow errors
//! with file context. The runner halts on these rather than silently falling
//! back to defaults, because a typo'd budget or model name should never be
//! silently lost.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Path of the workspace's configuration file (`<workspace>/foreman.toml`).
pub fn config_path(workspace: impl AsRef<Path>) -> PathBuf {
    workspace.as_ref().join("foreman.toml")
}

/// Fully resolved foreman configuration. Every section has a [`Default`] so
/// `Config::default()` is a valid runtime config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Per-role model selection.
    pub models: ModelRoles,
    /// Bounded-retry budgets for the runner loop.
    pub retries: RetryBudgets,
    /// Auditor pass configuration.
    pub audit: AuditConfig,
    /// Git integration tunables (branch naming, PR creation).
    pub git: GitConfig,
}

/// Model identifiers used for each agent role. Strings are passed verbatim to
/// the configured `Agent` implementation, so they must match whatever model id
/// the active agent (e.g., `claude` CLI) accepts.
///
/// Every role defaults to `claude-opus-4-7`. Users wanting a cheaper
/// implementer/fixer split can override per role in `foreman.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelRoles {
    /// Model used by `foreman plan` to generate `plan.md` from a goal.
    pub planner: String,
    /// Model used for the per-phase implementation pass — the bulk of token
    /// spend for most runs.
    pub implementer: String,
    /// Model used for the post-phase audit pass (see [`AuditConfig`]).
    pub auditor: String,
    /// Model used to fix failing tests, retried up to
    /// [`RetryBudgets::fixer_max_attempts`] times per phase.
    pub fixer: String,
}

impl Default for ModelRoles {
    fn default() -> Self {
        Self {
            planner: "claude-opus-4-7".to_string(),
            implementer: "claude-opus-4-7".to_string(),
            auditor: "claude-opus-4-7".to_string(),
            fixer: "claude-opus-4-7".to_string(),
        }
    }
}

/// Bounded-retry budgets. Foreman never loops indefinitely; once a budget is
/// exhausted the runner halts and surfaces the failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetryBudgets {
    /// Maximum number of fixer dispatches the runner will issue for a single
    /// phase before halting. Set to `0` to disable the fixer entirely.
    pub fixer_max_attempts: u32,
    /// Maximum total agent dispatches per phase across all roles
    /// (implementer + fixer + auditor). Once exceeded the runner halts even
    /// if `fixer_max_attempts` would otherwise allow another retry.
    pub max_phase_attempts: u32,
}

impl Default for RetryBudgets {
    fn default() -> Self {
        Self {
            fixer_max_attempts: 2,
            max_phase_attempts: 3,
        }
    }
}

/// Auditor-role tunables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditConfig {
    /// When `false`, the auditor pass is skipped entirely and the runner
    /// commits straight after tests pass.
    pub enabled: bool,
    /// Threshold (in changed lines of diff) that distinguishes "small fix"
    /// from "large enough to defer". The auditor inlines fixes at or below
    /// this size and writes anything larger to `deferred.md`.
    pub small_fix_line_limit: u32,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            small_fix_line_limit: 30,
        }
    }
}

/// Git integration tunables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GitConfig {
    /// Prefix prepended to the per-run branch name. The full branch is
    /// `<branch_prefix><utc_timestamp>` (e.g., `foreman/run-20260429T143022Z`).
    pub branch_prefix: String,
    /// When `true`, the runner shells out to `gh pr create` after the final
    /// phase commits. Equivalent to passing `--pr` on the CLI.
    pub create_pr: bool,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            branch_prefix: "foreman/run-".to_string(),
            create_pr: false,
        }
    }
}

/// Read the workspace's `foreman.toml`.
///
/// A missing file returns [`Config::default()`] — foreman is usable without
/// any config — but a present-but-malformed file is an error. Unknown keys
/// emit a [`tracing::warn`] and are otherwise ignored.
pub fn load(workspace: impl AsRef<Path>) -> Result<Config> {
    let path = config_path(workspace.as_ref());
    let text = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Config::default()),
        Err(e) => {
            return Err(
                anyhow::Error::new(e).context(format!("config::load: reading {:?}", path))
            );
        }
    };
    parse(&text).with_context(|| format!("config::load: parsing {:?}", path))
}

/// Parse a `foreman.toml` body. Empty / whitespace-only input yields
/// [`Config::default()`]. Unknown keys are logged at warn level.
pub fn parse(text: &str) -> Result<Config> {
    if text.trim().is_empty() {
        return Ok(Config::default());
    }
    let value: toml::Value = toml::from_str(text).context("foreman.toml is not valid TOML")?;
    for unknown in find_unknown_keys(&value) {
        warn!(key = %unknown, "foreman.toml: unknown key {:?} (ignored)", unknown);
    }
    let cfg: Config = value
        .try_into()
        .context("foreman.toml does not match the expected schema")?;
    Ok(cfg)
}

/// Walk a parsed `foreman.toml` value and return any keys not in the schema.
/// Returned in `section.key` form for nested keys, bare `section` for unknown
/// top-level keys. Order follows the document.
fn find_unknown_keys(value: &toml::Value) -> Vec<String> {
    let mut out = Vec::new();
    let toml::Value::Table(top) = value else {
        return out;
    };
    for (section, sub) in top {
        let known_subkeys: &[&str] = match section.as_str() {
            "models" => &["planner", "implementer", "auditor", "fixer"],
            "retries" => &["fixer_max_attempts", "max_phase_attempts"],
            "audit" => &["enabled", "small_fix_line_limit"],
            "git" => &["branch_prefix", "create_pr"],
            _ => {
                out.push(section.clone());
                continue;
            }
        };
        if let toml::Value::Table(sub_table) = sub {
            for sub_key in sub_table.keys() {
                if !known_subkeys.contains(&sub_key.as_str()) {
                    out.push(format!("{}.{}", section, sub_key));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn defaults_are_self_consistent() {
        let cfg = Config::default();
        assert_eq!(cfg.models.planner, "claude-opus-4-7");
        assert_eq!(cfg.models.implementer, "claude-opus-4-7");
        assert_eq!(cfg.models.auditor, "claude-opus-4-7");
        assert_eq!(cfg.models.fixer, "claude-opus-4-7");
        assert_eq!(cfg.retries.fixer_max_attempts, 2);
        assert_eq!(cfg.retries.max_phase_attempts, 3);
        assert!(cfg.audit.enabled);
        assert_eq!(cfg.audit.small_fix_line_limit, 30);
        assert_eq!(cfg.git.branch_prefix, "foreman/run-");
        assert!(!cfg.git.create_pr);
    }

    #[test]
    fn empty_input_yields_defaults() {
        assert_eq!(parse("").unwrap(), Config::default());
        assert_eq!(parse("   \n\t\n").unwrap(), Config::default());
    }

    #[test]
    fn full_input_overrides_every_field() {
        let text = "
[models]
planner = \"a\"
implementer = \"b\"
auditor = \"c\"
fixer = \"d\"

[retries]
fixer_max_attempts = 7
max_phase_attempts = 11

[audit]
enabled = false
small_fix_line_limit = 5

[git]
branch_prefix = \"work/\"
create_pr = true
";
        let cfg = parse(text).unwrap();
        assert_eq!(cfg.models.planner, "a");
        assert_eq!(cfg.models.implementer, "b");
        assert_eq!(cfg.models.auditor, "c");
        assert_eq!(cfg.models.fixer, "d");
        assert_eq!(cfg.retries.fixer_max_attempts, 7);
        assert_eq!(cfg.retries.max_phase_attempts, 11);
        assert!(!cfg.audit.enabled);
        assert_eq!(cfg.audit.small_fix_line_limit, 5);
        assert_eq!(cfg.git.branch_prefix, "work/");
        assert!(cfg.git.create_pr);
    }

    #[test]
    fn partial_input_fills_remaining_with_defaults() {
        let text = "
[git]
create_pr = true
";
        let cfg = parse(text).unwrap();
        // Specified field took effect.
        assert!(cfg.git.create_pr);
        // Untouched field within the same section stays at default.
        assert_eq!(cfg.git.branch_prefix, "foreman/run-");
        // Whole sections missing → defaults.
        assert_eq!(cfg.models, ModelRoles::default());
        assert_eq!(cfg.retries, RetryBudgets::default());
        assert_eq!(cfg.audit, AuditConfig::default());
    }

    #[test]
    fn partial_section_fills_missing_subkeys() {
        let text = "
[models]
implementer = \"custom-impl\"
";
        let cfg = parse(text).unwrap();
        assert_eq!(cfg.models.implementer, "custom-impl");
        // Other model fields still at default.
        assert_eq!(cfg.models.planner, ModelRoles::default().planner);
        assert_eq!(cfg.models.auditor, ModelRoles::default().auditor);
        assert_eq!(cfg.models.fixer, ModelRoles::default().fixer);
    }

    #[test]
    fn malformed_toml_is_an_error() {
        let err = parse("[models\nplanner = \"x\"").unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("not valid TOML"), "msg: {msg}");
    }

    #[test]
    fn wrong_value_type_is_an_error() {
        let text = "
[retries]
fixer_max_attempts = \"two\"
";
        let err = parse(text).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("expected schema"),
            "expected schema error, got: {msg}"
        );
    }

    #[test]
    fn unknown_keys_are_collected_not_errored() {
        let text = "
something_extra = 1

[models]
planner = \"p\"
new_role = \"x\"

[telemetry]
sink = \"stdout\"
";
        let cfg = parse(text).unwrap();
        // Known fields still loaded.
        assert_eq!(cfg.models.planner, "p");
        // Unknown fields surfaced by the helper.
        let toml_value: toml::Value = toml::from_str(text).unwrap();
        let unknown = find_unknown_keys(&toml_value);
        assert!(unknown.contains(&"something_extra".to_string()));
        assert!(unknown.contains(&"models.new_role".to_string()));
        assert!(unknown.contains(&"telemetry".to_string()));
    }

    #[test]
    fn no_unknown_keys_for_canonical_input() {
        let text = "
[models]
planner = \"p\"
implementer = \"i\"
auditor = \"a\"
fixer = \"f\"

[retries]
fixer_max_attempts = 1
max_phase_attempts = 2

[audit]
enabled = true
small_fix_line_limit = 10

[git]
branch_prefix = \"x/\"
create_pr = false
";
        let value: toml::Value = toml::from_str(text).unwrap();
        assert!(find_unknown_keys(&value).is_empty());
    }

    #[test]
    fn load_returns_defaults_when_file_missing() {
        let dir = tempdir().unwrap();
        let cfg = load(dir.path()).unwrap();
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn load_reads_file_from_workspace() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("foreman.toml"),
            "[git]\nbranch_prefix = \"loaded/\"\n",
        )
        .unwrap();
        let cfg = load(dir.path()).unwrap();
        assert_eq!(cfg.git.branch_prefix, "loaded/");
    }

    #[test]
    fn load_surfaces_parse_errors_with_path_context() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("foreman.toml"), "[broken").unwrap();
        let err = load(dir.path()).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("foreman.toml"), "msg: {msg}");
    }

    #[test]
    fn init_template_round_trips_through_loader() {
        // The seed `foreman.toml` written by `foreman init` must parse cleanly
        // and produce defaults equivalent to `Config::default()`. If the two
        // ever drift this test catches it.
        let dir = tempdir().unwrap();
        crate::cli::init::run(dir.path()).unwrap();
        let cfg = load(dir.path()).unwrap();
        assert_eq!(cfg, Config::default());
    }
}
