//! Pure rendering for `pitboss grind --dry-run`.
//!
//! Phase 12 ships `--dry-run` so users can verify the resolved configuration
//! — discovered prompts and their sources, the selected (or synthesized)
//! plan, the layered budgets, the active hooks, the parallelism cap — and
//! preview which prompts the scheduler will dispatch first, all without
//! touching git, the agent, or any on-disk run directory.
//!
//! The report is split into a deterministic machine-readable header (a
//! single `=== pitboss grind --dry-run ===` line followed by a `version: 1`
//! key so external scrapers can skip-detect), then a human report. Both halves
//! come out of [`render_dry_run_report`] as one string, so the CLI layer just
//! prints whatever this returns.
//!
//! The render is a pure function over its inputs so tests can pin the format
//! with `insta` and exercise edge cases (empty plan, hooks unset, every cap
//! hit) without spinning up a runner.

use std::collections::BTreeMap;
use std::path::Path;

use chrono::{DateTime, Utc};

use super::budget::BudgetSnapshot;
use super::plan::{GrindPlan, PlanBudgets};
use super::prompt::{PromptDoc, PromptSource};
use super::scheduler::{Scheduler, SchedulerState};

/// Stable header line at the top of every dry-run report. Intentionally a
/// single literal so a wrapping script can match on it deterministically.
pub const DRY_RUN_HEADER: &str = "=== pitboss grind --dry-run ===";

/// Schema version for the dry-run report. Bumped if the section layout below
/// changes in a breaking way; keep it as the second line of the report so
/// scrapers can branch on it.
pub const DRY_RUN_VERSION: &str = "1";

/// How many scheduler picks to preview in the dry-run report.
pub const PREVIEW_PICK_COUNT: usize = 10;

/// Inputs to [`render_dry_run_report`]. Everything is borrowed; the caller
/// owns the originals and we never mutate.
pub struct DryRunInputs<'a> {
    /// Workspace root path for display only. Stored as a `&Path` so callers
    /// can pass either an absolute or workspace-relative path; the renderer
    /// stringifies via `display()`.
    pub workspace: &'a Path,
    /// Resolved agent backend label for display. `None` collapses to
    /// `(default)` so the report tells the user explicitly that no override
    /// was set rather than silently omitting the line.
    pub agent_backend: Option<&'a str>,
    /// All prompts that survived discovery, in discovery order.
    pub prompts: &'a [PromptDoc],
    /// The selected plan, after defaulting / file load and validation.
    pub plan: &'a GrindPlan,
    /// Run-level budgets after layering config.toml `[grind.budgets]`, the
    /// plan's `PlanBudgets`, and any CLI overrides.
    pub budgets: &'a PlanBudgets,
    /// Maximum number of consecutive failed sessions before the
    /// consecutive-failure escape valve fires.
    pub consecutive_failure_limit: u32,
    /// Resume target if `--resume` is set. `None` for a fresh dry-run.
    pub resume_target: Option<&'a str>,
    /// Persisted scheduler state when `--resume` is set. Seeds the preview
    /// scheduler so the picks reflect where the resumed loop would actually
    /// land instead of a fresh rotation. `None` for a fresh dry-run.
    pub resume_scheduler_state: Option<&'a SchedulerState>,
    /// Persisted cumulative budget consumption when `--resume` is set. Shown
    /// in the `## Resume` section so the report surfaces what's already been
    /// spent. `None` for a fresh dry-run.
    pub resume_budget_consumed: Option<&'a BudgetSnapshot>,
    /// Sequence number of the last session recorded for the resumed run, so
    /// the user can see at a glance where the resumed loop will pick up.
    /// `None` for a fresh dry-run.
    pub resume_last_session_seq: Option<u32>,
}

/// Render the full dry-run report. Pure: depends only on `inputs`.
///
/// Layout:
/// 1. `=== pitboss grind --dry-run ===` then `version: 1`.
/// 2. `## Workspace` — paths and agent backend.
/// 3. `## Prompts` — every discovered prompt with source and frontmatter
///    summary.
/// 4. `## Plan` — name, parallelism cap, plan-level prompt entries.
/// 5. `## Budgets` — every cap that resolved (or `(unset)` placeholders).
/// 6. `## Hooks` — every hook command (or `(unset)` placeholders).
/// 7. `## Scheduler preview` — the next [`PREVIEW_PICK_COUNT`] picks the
///    scheduler would emit on the first `next()` call cycle.
pub fn render_dry_run_report(inputs: &DryRunInputs<'_>) -> String {
    let mut out = String::new();
    out.push_str(DRY_RUN_HEADER);
    out.push('\n');
    out.push_str(&format!("version: {DRY_RUN_VERSION}\n"));
    out.push('\n');

    out.push_str("## Workspace\n\n");
    out.push_str(&format!("- path: {}\n", inputs.workspace.display()));
    out.push_str(&format!(
        "- agent backend: {}\n",
        inputs.agent_backend.unwrap_or("(default)")
    ));
    if let Some(target) = inputs.resume_target {
        let label = if target.is_empty() {
            "(latest)".to_string()
        } else {
            target.to_string()
        };
        out.push_str(&format!("- resume target: {label}\n"));
    }
    out.push('\n');

    if let Some(snap) = inputs.resume_budget_consumed {
        out.push_str("## Resume\n\n");
        if let Some(seq) = inputs.resume_last_session_seq {
            out.push_str(&format!("- last_session_seq: {seq}\n"));
        }
        if let Some(state) = inputs.resume_scheduler_state {
            out.push_str(&format!("- scheduler_rotation: {}\n", state.rotation));
        }
        out.push_str(&format!("- iterations_consumed: {}\n", snap.iterations));
        out.push_str(&format!(
            "- tokens_consumed: {} (input={}, output={})\n",
            snap.tokens_input.saturating_add(snap.tokens_output),
            snap.tokens_input,
            snap.tokens_output,
        ));
        out.push_str(&format!("- cost_consumed_usd: ${:.4}\n", snap.cost_usd));
        out.push_str(&format!(
            "- consecutive_failures: {}\n",
            snap.consecutive_failures
        ));
        out.push('\n');
    }

    out.push_str("## Prompts\n\n");
    if inputs.prompts.is_empty() {
        out.push_str("_No prompts discovered._\n");
    } else {
        for p in inputs.prompts {
            out.push_str(&format!(
                "- {} (source={}, weight={}, every={}, max_runs={}, verify={}, parallel_safe={})\n",
                p.meta.name,
                source_label(p.source_kind),
                p.meta.weight,
                p.meta.every,
                opt_u32(p.meta.max_runs),
                p.meta.verify,
                p.meta.parallel_safe,
            ));
        }
    }
    out.push('\n');

    out.push_str("## Plan\n\n");
    out.push_str(&format!("- name: {}\n", inputs.plan.name));
    out.push_str(&format!("- max_parallel: {}\n", inputs.plan.max_parallel));
    out.push_str(&format!(
        "- consecutive_failure_limit: {}\n",
        inputs.consecutive_failure_limit
    ));
    out.push_str(&format!("- entries: {}\n", inputs.plan.prompts.len()));
    for entry in &inputs.plan.prompts {
        out.push_str(&format!(
            "  - {} (weight_override={}, every_override={}, max_runs_override={})\n",
            entry.name,
            opt_u32(entry.weight_override),
            opt_u32(entry.every_override),
            opt_u32(entry.max_runs_override),
        ));
    }
    out.push('\n');

    out.push_str("## Budgets\n\n");
    out.push_str(&format!(
        "- max_iterations: {}\n",
        opt_u32(inputs.budgets.max_iterations),
    ));
    out.push_str(&format!("- until: {}\n", opt_until(inputs.budgets.until)));
    out.push_str(&format!(
        "- max_tokens: {}\n",
        opt_u64(inputs.budgets.max_tokens),
    ));
    out.push_str(&format!(
        "- max_cost_usd: {}\n",
        opt_f64_usd(inputs.budgets.max_cost_usd),
    ));
    out.push('\n');

    out.push_str("## Hooks\n\n");
    out.push_str(&format!(
        "- pre_session: {}\n",
        opt_str(inputs.plan.hooks.pre_session.as_deref()),
    ));
    out.push_str(&format!(
        "- post_session: {}\n",
        opt_str(inputs.plan.hooks.post_session.as_deref()),
    ));
    out.push_str(&format!(
        "- on_failure: {}\n",
        opt_str(inputs.plan.hooks.on_failure.as_deref()),
    ));
    out.push('\n');

    out.push_str("## Scheduler preview\n\n");
    let preview_label = if inputs.resume_scheduler_state.is_some() {
        format!("Next {PREVIEW_PICK_COUNT} picks (resumed scheduler state):")
    } else {
        format!("Next {PREVIEW_PICK_COUNT} picks (frontmatter rules + plan overrides):")
    };
    out.push_str(&preview_label);
    out.push('\n');
    let picks = preview_picks_from_state(
        inputs.plan,
        inputs.prompts,
        inputs.resume_scheduler_state,
        PREVIEW_PICK_COUNT,
    );
    if picks.is_empty() {
        out.push_str("- (none — scheduler is exhausted from the first call)\n");
    } else {
        for (i, pick) in picks.iter().enumerate() {
            let label = match pick {
                Some(name) => name.as_str(),
                None => "(no eligible prompt this rotation)",
            };
            out.push_str(&format!("  {:>2}. {label}\n", i + 1));
        }
    }
    out.push('\n');

    out
}

/// Run the scheduler `count` times against a fresh state and return the
/// picked prompt name (or `None` for a rotation that yields nothing). Used by
/// the dry-run report and by tests that want to introspect the rotation.
pub fn preview_picks(plan: &GrindPlan, prompts: &[PromptDoc], count: usize) -> Vec<Option<String>> {
    preview_picks_from_state(plan, prompts, None, count)
}

/// Variant of [`preview_picks`] that seeds the scheduler from an explicit
/// [`SchedulerState`] when `seed` is `Some`. Used by `--dry-run --resume` so
/// the preview reflects the resumed loop's actual starting position.
pub fn preview_picks_from_state(
    plan: &GrindPlan,
    prompts: &[PromptDoc],
    seed: Option<&SchedulerState>,
    count: usize,
) -> Vec<Option<String>> {
    let lookup: BTreeMap<String, PromptDoc> = prompts
        .iter()
        .map(|p| (p.meta.name.clone(), p.clone()))
        .collect();
    let mut sched = match seed {
        Some(s) => Scheduler::with_state(plan.clone(), lookup, s.clone()),
        None => Scheduler::new(plan.clone(), lookup),
    };
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let picked = sched.next();
        let name = picked.as_ref().map(|p| p.meta.name.clone());
        if let Some(p) = picked {
            sched.record_run(&p.meta.name);
        }
        out.push(name);
    }
    out
}

fn source_label(s: PromptSource) -> &'static str {
    match s {
        PromptSource::Project => "project",
        PromptSource::Global => "global",
        PromptSource::Override => "override",
    }
}

fn opt_u32(v: Option<u32>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "(unset)".into())
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "(unset)".into())
}

fn opt_f64_usd(v: Option<f64>) -> String {
    v.map(|n| format!("${n:.4}"))
        .unwrap_or_else(|| "(unset)".into())
}

fn opt_until(v: Option<DateTime<Utc>>) -> String {
    v.map(|d| d.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_else(|| "(unset)".into())
}

fn opt_str(v: Option<&str>) -> String {
    v.map(|s| format!("`{s}`"))
        .unwrap_or_else(|| "(unset)".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grind::plan::{Hooks, PlanPromptRef};
    use crate::grind::prompt::PromptMeta;
    use std::path::PathBuf;

    fn fixture_prompt(name: &str, weight: u32, every: u32) -> PromptDoc {
        PromptDoc {
            meta: PromptMeta {
                name: name.into(),
                description: format!("desc for {name}"),
                weight,
                every,
                max_runs: None,
                verify: false,
                parallel_safe: false,
                tags: vec![],
                max_session_seconds: None,
                max_session_cost_usd: None,
            },
            body: String::new(),
            source_path: PathBuf::from(format!("/fixture/{name}.md")),
            source_kind: PromptSource::Project,
        }
    }

    fn fixture_plan(prompts: &[PromptDoc]) -> GrindPlan {
        GrindPlan {
            name: "fixture-plan".into(),
            prompts: prompts
                .iter()
                .map(|p| PlanPromptRef {
                    name: p.meta.name.clone(),
                    weight_override: None,
                    every_override: None,
                    max_runs_override: None,
                })
                .collect(),
            max_parallel: 2,
            hooks: Hooks {
                pre_session: Some("./pre.sh".into()),
                post_session: None,
                on_failure: Some("./fail.sh".into()),
            },
            budgets: PlanBudgets {
                max_iterations: Some(50),
                until: None,
                max_cost_usd: Some(12.5),
                max_tokens: None,
            },
        }
    }

    #[test]
    fn dry_run_report_snapshot_full_fixture() {
        let prompts = vec![
            fixture_prompt("alpha", 2, 1),
            fixture_prompt("bravo", 1, 1),
            fixture_prompt("charlie", 1, 3),
        ];
        let plan = fixture_plan(&prompts);
        let budgets = PlanBudgets {
            max_iterations: Some(50),
            until: Some("2026-04-30T23:59:00Z".parse().unwrap()),
            max_cost_usd: Some(12.5),
            max_tokens: Some(1_000_000),
        };
        let inputs = DryRunInputs {
            workspace: Path::new("/tmp/fixture-workspace"),
            agent_backend: Some("claude_code"),
            prompts: &prompts,
            plan: &plan,
            budgets: &budgets,
            consecutive_failure_limit: 3,
            resume_target: None,
            resume_scheduler_state: None,
            resume_budget_consumed: None,
            resume_last_session_seq: None,
        };
        let report = render_dry_run_report(&inputs);
        insta::assert_snapshot!("dry_run_report_full_fixture", report);
    }

    #[test]
    fn dry_run_report_snapshot_minimal_defaults() {
        // Empty plan, no hooks, no budgets — exercises the `(unset)` and
        // `(none — scheduler is exhausted)` branches together.
        let plan = GrindPlan {
            name: "default".into(),
            prompts: vec![],
            max_parallel: 1,
            hooks: Hooks::default(),
            budgets: PlanBudgets::default(),
        };
        let inputs = DryRunInputs {
            workspace: Path::new("/tmp/empty"),
            agent_backend: None,
            prompts: &[],
            plan: &plan,
            budgets: &PlanBudgets::default(),
            consecutive_failure_limit: 3,
            resume_target: Some(""),
            resume_scheduler_state: None,
            resume_budget_consumed: None,
            resume_last_session_seq: None,
        };
        let report = render_dry_run_report(&inputs);
        insta::assert_snapshot!("dry_run_report_minimal_defaults", report);
    }

    #[test]
    fn dry_run_report_snapshot_resume_preview() {
        // `--dry-run --resume` should surface the resume target, the consumed
        // budget snapshot, and a scheduler preview seeded from the persisted
        // state — not a fresh rotation.
        let prompts = vec![fixture_prompt("alpha", 2, 1), fixture_prompt("bravo", 1, 1)];
        let plan = fixture_plan(&prompts);
        let budgets = PlanBudgets {
            max_iterations: Some(50),
            until: None,
            max_cost_usd: Some(12.5),
            max_tokens: None,
        };
        // Seeded mid-rotation: alpha has been picked twice, bravo once. The
        // preview should reflect this skew.
        let mut runs = std::collections::BTreeMap::new();
        runs.insert("alpha".to_string(), 2u32);
        runs.insert("bravo".to_string(), 1u32);
        let scheduler_state = SchedulerState {
            rotation: 3,
            runs_per_prompt: runs,
        };
        let snapshot = BudgetSnapshot {
            iterations: 3,
            tokens_input: 4000,
            tokens_output: 2000,
            cost_usd: 1.2345,
            consecutive_failures: 0,
        };
        let inputs = DryRunInputs {
            workspace: Path::new("/tmp/fixture-workspace"),
            agent_backend: Some("claude_code"),
            prompts: &prompts,
            plan: &plan,
            budgets: &budgets,
            consecutive_failure_limit: 3,
            resume_target: Some("20260430T180000Z-rsm1"),
            resume_scheduler_state: Some(&scheduler_state),
            resume_budget_consumed: Some(&snapshot),
            resume_last_session_seq: Some(3),
        };
        let report = render_dry_run_report(&inputs);
        insta::assert_snapshot!("dry_run_report_resume_preview", report);
    }

    #[test]
    fn preview_picks_respects_max_runs_cap() {
        // Single prompt with max_runs=2 should emit two picks then None
        // forever. Confirms preview_picks calls record_run between iterations.
        let mut p = fixture_prompt("solo", 1, 1);
        p.meta.max_runs = Some(2);
        let plan = GrindPlan {
            name: "cap".into(),
            prompts: vec![PlanPromptRef {
                name: "solo".into(),
                weight_override: None,
                every_override: None,
                max_runs_override: None,
            }],
            max_parallel: 1,
            hooks: Hooks::default(),
            budgets: PlanBudgets::default(),
        };
        let picks = preview_picks(&plan, &[p], 5);
        assert_eq!(picks.len(), 5);
        assert_eq!(picks[0].as_deref(), Some("solo"));
        assert_eq!(picks[1].as_deref(), Some("solo"));
        for slot in &picks[2..] {
            assert!(slot.is_none(), "expected None after cap, got {slot:?}");
        }
    }

    #[test]
    fn header_and_version_appear_first() {
        // External scrapers depend on the literal header + version on the
        // first two lines; pin both here so a casual format change can't
        // break them.
        let plan = GrindPlan {
            name: "default".into(),
            prompts: vec![],
            max_parallel: 1,
            hooks: Hooks::default(),
            budgets: PlanBudgets::default(),
        };
        let inputs = DryRunInputs {
            workspace: Path::new("/tmp/x"),
            agent_backend: None,
            prompts: &[],
            plan: &plan,
            budgets: &PlanBudgets::default(),
            consecutive_failure_limit: 3,
            resume_target: None,
            resume_scheduler_state: None,
            resume_budget_consumed: None,
            resume_last_session_seq: None,
        };
        let report = render_dry_run_report(&inputs);
        let mut lines = report.lines();
        assert_eq!(lines.next(), Some(DRY_RUN_HEADER));
        assert_eq!(lines.next(), Some("version: 1"));
    }
}
