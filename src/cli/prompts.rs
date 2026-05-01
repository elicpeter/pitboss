//! `pitboss prompts` — author and inspect grind prompt files.
//!
//! Three thin actions, all backed by phase 02's `discover_prompts`:
//!
//! - `ls` prints a tabular view of the discovered prompts and their source.
//! - `validate` re-runs discovery and reports per-file parse errors, exiting
//!   non-zero if any file failed.
//! - `new <name>` writes a templated `<name>.md` into the project (default) or
//!   user-global prompts directory, refusing to overwrite an existing file.
//!
//! `new` does not invoke discovery; the new file is intended to be edited
//! before `validate` is run.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};

use crate::grind::{
    discover_prompts, resolve_home_prompts_dir, templates, DiscoveryOptions, PromptDoc,
    PromptSource,
};
use crate::style::{self, col};
use crate::util::paths::grind_prompts_dir;
use crate::util::write_atomic;

/// Arguments for `pitboss prompts <action>`.
#[derive(Debug, Args)]
pub struct PromptsArgs {
    /// What to do with the discovered prompts.
    #[command(subcommand)]
    pub action: PromptsAction,
}

/// Actions available under `pitboss prompts`.
#[derive(Debug, Subcommand)]
pub enum PromptsAction {
    /// List discovered prompts and where each was loaded from.
    Ls,
    /// Re-run discovery and report each malformed file. Exits non-zero on any
    /// error.
    Validate,
    /// Write a templated `<name>.md` into the prompts directory.
    New {
        /// Stable prompt identifier. Becomes the filename stem and the
        /// `name:` field in the template.
        name: String,
        /// Override the destination directory. When set, neither
        /// `.pitboss/grind/prompts/` nor `~/.pitboss/grind/prompts/` is used.
        #[arg(long = "dir")]
        dir: Option<PathBuf>,
        /// Write into `~/.pitboss/grind/prompts/` instead of the project's
        /// `.pitboss/grind/prompts/` directory.
        #[arg(long, conflicts_with = "dir")]
        global: bool,
    },
}

/// Top-level dispatcher invoked from `cli::dispatch`.
pub fn run(workspace: PathBuf, args: PromptsArgs) -> Result<()> {
    match args.action {
        PromptsAction::Ls => run_ls(&workspace),
        PromptsAction::Validate => run_validate(&workspace),
        PromptsAction::New { name, dir, global } => {
            run_new(&workspace, &name, dir.as_deref(), global)
        }
    }
}

fn run_ls(workspace: &Path) -> Result<()> {
    let result = discover_prompts(default_options(workspace));
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let c = style::use_color_stdout();

    if result.prompts.is_empty() {
        let _ = writeln!(handle, "no prompts discovered");
        return Ok(());
    }

    let _ = handle.write_all(render_table(&result.prompts, c).as_bytes());
    Ok(())
}

fn run_validate(workspace: &Path) -> Result<()> {
    let result = discover_prompts(default_options(workspace));
    let stderr = std::io::stderr();
    let stdout = std::io::stdout();
    let c_err = style::use_color_stderr();

    {
        let mut err = stderr.lock();
        for (path, error) in &result.errors {
            let _ = writeln!(
                err,
                "{} {}: {}",
                col(c_err, style::BOLD_RED, "error:"),
                path.display(),
                one_line(&error.to_string()),
            );
        }
    }

    let ok_count = result.prompts.len();
    let err_count = result.errors.len();
    {
        let mut out = stdout.lock();
        let _ = writeln!(out, "{} prompt(s) ok, {} error(s)", ok_count, err_count);
    }

    if err_count > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn run_new(workspace: &Path, name: &str, dir_override: Option<&Path>, global: bool) -> Result<()> {
    if !is_valid_prompt_name(name) {
        bail!("prompts new: invalid name {name:?}: must match ^[a-z0-9][a-z0-9_-]*$");
    }

    let target_dir = resolve_target_dir(workspace, dir_override, global)?;
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("prompts new: creating {:?}", target_dir))?;

    let target = target_dir.join(format!("{name}.md"));
    if target.exists() {
        bail!(
            "prompts new: refusing to overwrite existing prompt file {:?}",
            target
        );
    }

    let body = templates::render_new_prompt(name);
    write_atomic(&target, body.as_bytes())?;

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let c = style::use_color_stdout();
    let _ = writeln!(
        handle,
        "{} {}",
        col(c, style::GREEN, "created"),
        display_relative(workspace, &target)
    );
    Ok(())
}

fn resolve_target_dir(
    workspace: &Path,
    dir_override: Option<&Path>,
    global: bool,
) -> Result<PathBuf> {
    if let Some(dir) = dir_override {
        return Ok(if dir.is_absolute() {
            dir.to_path_buf()
        } else {
            workspace.join(dir)
        });
    }
    if global {
        return resolve_home_prompts_dir().ok_or_else(|| {
            anyhow::anyhow!(
                "prompts new --global: HOME is unset; cannot locate ~/.pitboss/grind/prompts/"
            )
        });
    }
    Ok(grind_prompts_dir(workspace))
}

fn default_options(workspace: &Path) -> DiscoveryOptions {
    DiscoveryOptions {
        project_root: workspace.to_path_buf(),
        home_dir: std::env::var_os("HOME").map(PathBuf::from),
        override_dir: None,
    }
}

fn is_valid_prompt_name(s: &str) -> bool {
    let mut bytes = s.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Render the `ls` table. Column widths grow to fit the longest cell so the
/// output stays aligned without wrapping. `color` controls ANSI styling so the
/// renderer is testable without probing terminal state.
pub(crate) fn render_table(prompts: &[PromptDoc], color: bool) -> String {
    let headers = ["NAME", "SOURCE", "WEIGHT", "EVERY", "VERIFY", "PATH"];
    let mut rows: Vec<[String; 6]> = Vec::with_capacity(prompts.len());
    for p in prompts {
        rows.push([
            p.meta.name.clone(),
            source_label(p.source_kind).to_string(),
            p.meta.weight.to_string(),
            p.meta.every.to_string(),
            if p.meta.verify { "yes" } else { "no" }.to_string(),
            p.source_path.display().to_string(),
        ]);
    }

    let mut widths = headers.map(|h| h.len());
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            if cell.len() > widths[i] {
                widths[i] = cell.len();
            }
        }
    }

    let mut out = String::new();
    for (i, h) in headers.iter().enumerate() {
        let padded = pad(h, widths[i]);
        out.push_str(&col(color, style::BOLD, &padded));
        if i + 1 < headers.len() {
            out.push_str("  ");
        }
    }
    out.push('\n');
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            out.push_str(&pad(cell, widths[i]));
            if i + 1 < row.len() {
                out.push_str("  ");
            }
        }
        out.push('\n');
    }
    out
}

fn pad(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        let mut out = String::with_capacity(width);
        out.push_str(s);
        for _ in s.len()..width {
            out.push(' ');
        }
        out
    }
}

fn source_label(kind: PromptSource) -> &'static str {
    match kind {
        PromptSource::Project => "project",
        PromptSource::Global => "global",
        PromptSource::Override => "override",
    }
}

fn one_line(s: &str) -> String {
    s.lines().next().unwrap_or(s).trim().to_string()
}

fn display_relative(workspace: &Path, target: &Path) -> String {
    target
        .strip_prefix(workspace)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| target.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grind::{PromptMeta, PromptSource};
    use std::path::PathBuf;

    fn doc(
        name: &str,
        source: PromptSource,
        weight: u32,
        every: u32,
        verify: bool,
        path: &str,
    ) -> PromptDoc {
        PromptDoc {
            meta: PromptMeta {
                name: name.to_string(),
                description: "test".into(),
                weight,
                every,
                max_runs: None,
                verify,
                parallel_safe: false,
                tags: vec![],
                max_session_seconds: None,
                max_session_cost_usd: None,
            },
            body: String::new(),
            source_path: PathBuf::from(path),
            source_kind: source,
        }
    }

    #[test]
    fn render_table_aligns_columns() {
        let prompts = vec![
            doc("alpha", PromptSource::Project, 1, 1, false, "/p/alpha.md"),
            doc(
                "triage-much-longer-name",
                PromptSource::Global,
                5,
                3,
                true,
                "/g/triage-much-longer-name.md",
            ),
        ];
        let table = render_table(&prompts, false);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("NAME"));
        assert!(lines[0].contains("SOURCE"));
        assert!(lines[0].contains("WEIGHT"));
        assert!(lines[0].contains("EVERY"));
        assert!(lines[0].contains("VERIFY"));
        assert!(lines[0].contains("PATH"));
        // Headers and rows share their column widths.
        assert_eq!(lines[0].len(), lines[1].len());
        assert_eq!(lines[1].len(), lines[2].len());
        assert!(lines[1].contains("alpha"));
        assert!(lines[1].contains("project"));
        assert!(lines[2].contains("triage-much-longer-name"));
        assert!(lines[2].contains("global"));
        assert!(lines[2].contains("yes"));
    }

    #[test]
    fn is_valid_prompt_name_matches_regex() {
        assert!(is_valid_prompt_name("alpha"));
        assert!(is_valid_prompt_name("a-b_c"));
        assert!(is_valid_prompt_name("9lives"));
        assert!(!is_valid_prompt_name(""));
        assert!(!is_valid_prompt_name("-leading"));
        assert!(!is_valid_prompt_name("Has Caps"));
        assert!(!is_valid_prompt_name("with space"));
        assert!(!is_valid_prompt_name("dot.bad"));
    }
}
