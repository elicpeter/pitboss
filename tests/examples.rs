//! Pin the shipped `examples/` corpus against the live parsers.
//!
//! Phase 20 ships at least one example plan and walkthrough under `examples/`.
//! These tests guard against silent rot: if the parser ever tightens, or an
//! example is hand-edited into something pitboss can no longer load, this
//! suite fails loudly instead of letting users discover the breakage when
//! they paste an example into a fresh workspace.
//!
//! Adding a new example? Add a matching pair of asserts here so it stays
//! covered.

use std::fs;
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

#[test]
fn todo_cli_plan_parses_and_round_trips() {
    let path = examples_dir().join("todo-cli").join("plan.md");
    let text = fs::read_to_string(&path).expect("read plan.md");
    let plan =
        pitboss::plan::parse(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    assert!(
        !plan.phases.is_empty(),
        "example plan should declare at least one phase"
    );
    // current_phase points at a real heading.
    assert!(
        plan.phases.iter().any(|p| p.id == plan.current_phase),
        "current_phase {:?} must match a phase heading",
        plan.current_phase
    );
    // Round-trips byte-for-byte so an agent that re-serializes does not
    // accidentally rewrite the file.
    assert_eq!(pitboss::plan::serialize(&plan), text);
}

#[test]
fn todo_cli_pitboss_toml_parses() {
    let path = examples_dir().join("todo-cli").join("config.toml");
    let text = fs::read_to_string(&path).expect("read config.toml");
    let cfg =
        pitboss::config::parse(&text).unwrap_or_else(|e| panic!("parse {}: {e:#}", path.display()));
    // Spot-check a couple of fields the example deliberately overrides.
    assert!(cfg.audit.enabled);
    assert_eq!(cfg.retries.fixer_max_attempts, 3);
    assert!(
        cfg.budgets.max_total_usd.is_some(),
        "example config should ship a USD budget cap"
    );
}
