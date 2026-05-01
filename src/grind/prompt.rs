//! Grind prompt files: YAML frontmatter + markdown body.
//!
//! A grind prompt is a user-authored markdown file. The frontmatter declares
//! how the prompt participates in a rotation (weight, cadence, run cap, time
//! and cost ceilings) and the body is the instruction text the agent receives.
//!
//! This is deliberately distinct from [`crate::prompts`], which holds the
//! built-in LLM templates the runner feeds the agent. Those live in the
//! binary; these live on disk and are authored by the user.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Where a [`PromptDoc`] was loaded from. Drives precedence in phase 02
/// discovery (override > project > global).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptSource {
    /// `./.pitboss/grind/prompts/<name>.md` inside the current repo.
    Project,
    /// `~/.pitboss/grind/prompts/<name>.md` in the user's home directory.
    Global,
    /// Explicit `--prompts-dir <path>` override on the command line.
    Override,
}

/// Frontmatter metadata for a grind prompt.
///
/// Field defaults are applied during deserialization so a prompt can declare
/// only `name` and `description` and still parse. Run-time invariants beyond
/// what serde can express live in [`PromptMeta::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptMeta {
    /// Stable identifier; matches `^[a-z0-9][a-z0-9_-]*$`. Used as the prompt's
    /// filename stem and as the key for plan overrides and run-count tracking.
    pub name: String,
    /// One-line human description. Surfaced in `pitboss prompts ls`.
    pub description: String,
    /// Relative weight for the rotation scheduler. Defaults to `1` when omitted.
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Run once every N rotations. Defaults to `1` (every rotation).
    #[serde(default = "default_every")]
    pub every: u32,
    /// Optional ceiling on the number of times this prompt is dispatched in a
    /// run. `None` means no cap.
    #[serde(default)]
    pub max_runs: Option<u32>,
    /// If `true`, the runner reuses the existing test/fixer cycle after this
    /// prompt's session.
    #[serde(default)]
    pub verify: bool,
    /// If `true`, multiple sessions of this prompt can run concurrently in
    /// separate worktrees once parallelism lands in phase 11.
    #[serde(default)]
    pub parallel_safe: bool,
    /// Free-form labels for filtering and reporting.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Per-prompt wall-clock cap (seconds). `None` means inherit.
    #[serde(default)]
    pub max_session_seconds: Option<u64>,
    /// Per-prompt cost cap (USD). `None` means inherit.
    #[serde(default)]
    pub max_session_cost_usd: Option<f64>,
}

fn default_weight() -> u32 {
    1
}

fn default_every() -> u32 {
    1
}

/// A parsed grind prompt: frontmatter + body + provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptDoc {
    /// Parsed and validated frontmatter.
    pub meta: PromptMeta,
    /// Markdown body following the closing `---` fence. Preserved verbatim,
    /// including any leading newline.
    pub body: String,
    /// Path the prompt was loaded from.
    pub source_path: PathBuf,
    /// Which discovery source produced this prompt.
    pub source_kind: PromptSource,
}

/// Validation failures for [`PromptMeta::validate`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PromptMetaValidationError {
    /// `name` did not match `^[a-z0-9][a-z0-9_-]*$`.
    #[error("invalid name {0:?}: must match ^[a-z0-9][a-z0-9_-]*$")]
    InvalidName(String),
    /// `weight` was zero.
    #[error("weight must be >= 1")]
    WeightTooSmall,
    /// `every` was zero.
    #[error("every must be >= 1")]
    EveryTooSmall,
    /// `max_session_cost_usd` was negative.
    #[error("max_session_cost_usd must be >= 0")]
    NegativeCost,
}

/// Errors produced by [`parse_prompt_file`] and the in-memory parser.
#[derive(Debug, Error)]
pub enum PromptParseError {
    /// The file could not be read.
    #[error("failed to read prompt file {path}: {source}")]
    Io {
        /// Display path of the offending file.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// No leading `---\n…\n---` block was found.
    #[error("{path}: missing YAML frontmatter (expected ---\\n…\\n---\\n)")]
    MissingFrontmatter {
        /// Display path of the offending file.
        path: String,
    },
    /// `name` was missing from the frontmatter.
    #[error("{path}: frontmatter is missing required field `name`")]
    MissingName {
        /// Display path of the offending file.
        path: String,
    },
    /// The same key appeared twice in the frontmatter.
    #[error("{path}: duplicate frontmatter field `{field}`")]
    DuplicateField {
        /// Display path of the offending file.
        path: String,
        /// Name of the duplicated key.
        field: String,
    },
    /// The frontmatter parsed as YAML but did not match the expected shape, or
    /// YAML itself was malformed.
    #[error("{path}: invalid frontmatter: {message}")]
    BadFrontmatter {
        /// Display path of the offending file.
        path: String,
        /// One-line diagnostic.
        message: String,
    },
    /// A field passed YAML decoding but failed semantic validation.
    #[error("{path}: invalid frontmatter: {source}")]
    InvalidMeta {
        /// Display path of the offending file.
        path: String,
        /// Underlying validation error.
        #[source]
        source: PromptMetaValidationError,
    },
}

impl PartialEq for PromptParseError {
    fn eq(&self, other: &Self) -> bool {
        use PromptParseError::*;
        match (self, other) {
            (Io { path: a, .. }, Io { path: b, .. }) => a == b,
            (MissingFrontmatter { path: a }, MissingFrontmatter { path: b }) => a == b,
            (MissingName { path: a }, MissingName { path: b }) => a == b,
            (DuplicateField { path: a, field: af }, DuplicateField { path: b, field: bf }) => {
                a == b && af == bf
            }
            (
                BadFrontmatter {
                    path: a,
                    message: am,
                },
                BadFrontmatter {
                    path: b,
                    message: bm,
                },
            ) => a == b && am == bm,
            (
                InvalidMeta {
                    path: a,
                    source: as_,
                },
                InvalidMeta {
                    path: b,
                    source: bs,
                },
            ) => a == b && as_ == bs,
            _ => false,
        }
    }
}

const FENCE: &str = "---\n";

impl PromptMeta {
    /// Run semantic checks beyond what the serde derive enforces.
    pub fn validate(&self) -> Result<(), PromptMetaValidationError> {
        if !is_valid_name(&self.name) {
            return Err(PromptMetaValidationError::InvalidName(self.name.clone()));
        }
        if self.weight < 1 {
            return Err(PromptMetaValidationError::WeightTooSmall);
        }
        if self.every < 1 {
            return Err(PromptMetaValidationError::EveryTooSmall);
        }
        if let Some(cost) = self.max_session_cost_usd {
            if cost < 0.0 {
                return Err(PromptMetaValidationError::NegativeCost);
            }
        }
        Ok(())
    }
}

fn is_valid_name(s: &str) -> bool {
    let mut bytes = s.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Read and parse a prompt file from disk.
pub fn parse_prompt_file(path: &Path) -> Result<PromptDoc, PromptParseError> {
    let display = path.display().to_string();
    let raw = fs::read_to_string(path).map_err(|e| PromptParseError::Io {
        path: display.clone(),
        source: e,
    })?;
    parse_prompt_str(&raw, path, PromptSource::Project)
}

/// Parse a prompt from an in-memory string. Used by [`parse_prompt_file`] and
/// kept public-in-crate for tests and future discovery code (phase 02).
pub(crate) fn parse_prompt_str(
    input: &str,
    source_path: &Path,
    source_kind: PromptSource,
) -> Result<PromptDoc, PromptParseError> {
    let display = source_path.display().to_string();
    let normalized;
    let text = if input.contains('\r') {
        normalized = input.replace("\r\n", "\n");
        normalized.as_str()
    } else {
        input
    };

    let after_open =
        text.strip_prefix(FENCE)
            .ok_or_else(|| PromptParseError::MissingFrontmatter {
                path: display.clone(),
            })?;
    let close_idx =
        find_closing_fence(after_open).ok_or_else(|| PromptParseError::MissingFrontmatter {
            path: display.clone(),
        })?;

    let frontmatter_raw = &after_open[..close_idx];
    let body = &after_open[close_idx + FENCE.len()..];

    check_no_duplicate_keys(frontmatter_raw, &display)?;

    let meta: PromptMeta = match serde_yaml::from_str(frontmatter_raw) {
        Ok(m) => m,
        Err(err) => {
            return Err(classify_yaml_error(err, &display));
        }
    };
    meta.validate().map_err(|e| PromptParseError::InvalidMeta {
        path: display.clone(),
        source: e,
    })?;

    Ok(PromptDoc {
        meta,
        body: body.to_string(),
        source_path: source_path.to_path_buf(),
        source_kind,
    })
}

fn find_closing_fence(after_open: &str) -> Option<usize> {
    if after_open.starts_with(FENCE) {
        return Some(0);
    }
    after_open.find("\n---\n").map(|idx| idx + 1)
}

fn classify_yaml_error(err: serde_yaml::Error, path: &str) -> PromptParseError {
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("missing field `name`") {
        return PromptParseError::MissingName {
            path: path.to_string(),
        };
    }
    PromptParseError::BadFrontmatter {
        path: path.to_string(),
        message: one_line(&msg),
    }
}

fn one_line(s: &str) -> String {
    s.lines().next().unwrap_or(s).trim().to_string()
}

fn check_no_duplicate_keys(frontmatter: &str, path: &str) -> Result<(), PromptParseError> {
    let mut seen: Vec<String> = Vec::new();
    for line in frontmatter.lines() {
        if let Some(key) = top_level_key(line) {
            if seen.iter().any(|k| k == key) {
                return Err(PromptParseError::DuplicateField {
                    path: path.to_string(),
                    field: key.to_string(),
                });
            }
            seen.push(key.to_string());
        }
    }
    Ok(())
}

fn top_level_key(line: &str) -> Option<&str> {
    if line.starts_with(' ') || line.starts_with('\t') || line.starts_with('-') {
        return None;
    }
    let trimmed = line.trim_end();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let colon = trimmed.find(':')?;
    let key = trimmed[..colon].trim();
    if key.is_empty() {
        return None;
    }
    if key.bytes().any(|b| b == b'"' || b == b'\'' || b == b' ') {
        return None;
    }
    Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_path() -> PathBuf {
        PathBuf::from("/fixture/prompt.md")
    }

    fn parse(input: &str) -> Result<PromptDoc, PromptParseError> {
        parse_prompt_str(input, &fake_path(), PromptSource::Project)
    }

    #[test]
    fn well_formed_prompt_round_trips() {
        let input = "\
---
name: fp-hunter
description: Find and remove false positives.
weight: 3
every: 2
max_runs: 10
verify: true
parallel_safe: true
tags: [cleanup, lint]
max_session_seconds: 600
max_session_cost_usd: 1.5
---
Hunt for spurious failures, then file a deferred item describing each one.
";
        let doc = parse(input).expect("parse should succeed");
        assert_eq!(doc.meta.name, "fp-hunter");
        assert_eq!(doc.meta.description, "Find and remove false positives.");
        assert_eq!(doc.meta.weight, 3);
        assert_eq!(doc.meta.every, 2);
        assert_eq!(doc.meta.max_runs, Some(10));
        assert!(doc.meta.verify);
        assert!(doc.meta.parallel_safe);
        assert_eq!(doc.meta.tags, vec!["cleanup".to_string(), "lint".into()]);
        assert_eq!(doc.meta.max_session_seconds, Some(600));
        assert_eq!(doc.meta.max_session_cost_usd, Some(1.5));
        assert!(doc.body.starts_with("Hunt for spurious"));
        assert_eq!(doc.source_kind, PromptSource::Project);
        assert_eq!(doc.source_path, fake_path());
    }

    #[test]
    fn defaults_apply_when_optional_fields_omitted() {
        let input = "\
---
name: triage
description: Walk the issue queue.
---
Body.
";
        let doc = parse(input).expect("parse should succeed");
        assert_eq!(doc.meta.weight, 1);
        assert_eq!(doc.meta.every, 1);
        assert_eq!(doc.meta.max_runs, None);
        assert!(!doc.meta.verify);
        assert!(!doc.meta.parallel_safe);
        assert!(doc.meta.tags.is_empty());
        assert_eq!(doc.meta.max_session_seconds, None);
        assert_eq!(doc.meta.max_session_cost_usd, None);
    }

    #[test]
    fn verify_and_parallel_safe_parse_independently() {
        let input = "\
---
name: lint-sweep
description: Lint pass.
verify: true
parallel_safe: false
---
";
        let doc = parse(input).expect("parse should succeed");
        assert!(doc.meta.verify);
        assert!(!doc.meta.parallel_safe);
    }

    #[test]
    fn missing_frontmatter_is_rejected() {
        let input = "no fence here\nname: foo\n";
        let err = parse(input).unwrap_err();
        assert!(matches!(err, PromptParseError::MissingFrontmatter { .. }));
    }

    #[test]
    fn unterminated_frontmatter_is_rejected() {
        let input = "---\nname: foo\ndescription: bar\n";
        let err = parse(input).unwrap_err();
        assert!(matches!(err, PromptParseError::MissingFrontmatter { .. }));
    }

    #[test]
    fn missing_name_is_rejected() {
        let input = "\
---
description: only description, no name.
---
body
";
        let err = parse(input).unwrap_err();
        assert!(
            matches!(err, PromptParseError::MissingName { .. }),
            "expected MissingName, got {err:?}"
        );
    }

    #[test]
    fn malformed_yaml_is_rejected() {
        let input = "\
---
name: foo
description: bar
weight: : :
---
";
        let err = parse(input).unwrap_err();
        assert!(matches!(err, PromptParseError::BadFrontmatter { .. }));
    }

    #[test]
    fn duplicate_field_is_rejected() {
        let input = "\
---
name: foo
description: bar
weight: 1
weight: 2
---
";
        let err = parse(input).unwrap_err();
        match err {
            PromptParseError::DuplicateField { field, .. } => assert_eq!(field, "weight"),
            other => panic!("expected DuplicateField, got {other:?}"),
        }
    }

    #[test]
    fn invalid_name_is_rejected() {
        let input = "\
---
name: Bad Name
description: nope
---
";
        let err = parse(input).unwrap_err();
        match err {
            PromptParseError::InvalidMeta { source, .. } => {
                assert!(matches!(source, PromptMetaValidationError::InvalidName(_)));
            }
            other => panic!("expected InvalidMeta(InvalidName), got {other:?}"),
        }
    }

    #[test]
    fn name_starting_with_dash_is_rejected() {
        let input = "\
---
name: -leading-dash
description: nope
---
";
        let err = parse(input).unwrap_err();
        assert!(matches!(err, PromptParseError::InvalidMeta { .. }));
    }

    #[test]
    fn name_with_underscore_and_digits_is_accepted() {
        let input = "\
---
name: 9lives_v2-rev
description: ok
---
";
        let doc = parse(input).expect("name should be valid");
        assert_eq!(doc.meta.name, "9lives_v2-rev");
    }

    #[test]
    fn weight_zero_is_rejected() {
        let input = "\
---
name: foo
description: bar
weight: 0
---
";
        let err = parse(input).unwrap_err();
        match err {
            PromptParseError::InvalidMeta { source, .. } => {
                assert_eq!(source, PromptMetaValidationError::WeightTooSmall);
            }
            other => panic!("expected WeightTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn every_zero_is_rejected() {
        let input = "\
---
name: foo
description: bar
every: 0
---
";
        let err = parse(input).unwrap_err();
        match err {
            PromptParseError::InvalidMeta { source, .. } => {
                assert_eq!(source, PromptMetaValidationError::EveryTooSmall);
            }
            other => panic!("expected EveryTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn negative_cost_is_rejected() {
        let input = "\
---
name: foo
description: bar
max_session_cost_usd: -0.5
---
";
        let err = parse(input).unwrap_err();
        match err {
            PromptParseError::InvalidMeta { source, .. } => {
                assert_eq!(source, PromptMetaValidationError::NegativeCost);
            }
            other => panic!("expected NegativeCost, got {other:?}"),
        }
    }

    #[test]
    fn unknown_field_is_rejected() {
        let input = "\
---
name: foo
description: bar
mystery_field: 42
---
";
        let err = parse(input).unwrap_err();
        assert!(matches!(err, PromptParseError::BadFrontmatter { .. }));
    }

    #[test]
    fn crlf_line_endings_are_normalized() {
        let input = "---\r\nname: foo\r\ndescription: bar\r\n---\r\nbody\r\n";
        let doc = parse(input).expect("CRLF should be normalized");
        assert_eq!(doc.meta.name, "foo");
        assert!(doc.body.contains("body"));
    }

    #[test]
    fn parse_prompt_file_reads_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("triage.md");
        std::fs::write(
            &path,
            "---\nname: triage\ndescription: walk queue\n---\nbody\n",
        )
        .unwrap();
        let doc = parse_prompt_file(&path).expect("file should parse");
        assert_eq!(doc.meta.name, "triage");
        assert_eq!(doc.source_path, path);
    }

    #[test]
    fn parse_prompt_file_reports_io_error_for_missing_path() {
        let err = parse_prompt_file(Path::new("/no/such/prompt.md")).unwrap_err();
        assert!(matches!(err, PromptParseError::Io { .. }));
    }
}
