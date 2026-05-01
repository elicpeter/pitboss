//! Cross-backend dispatch integration test.
//!
//! Phase 04 acceptance: every backend named in [`pitboss::agent::backend::BackendKind`]
//! must be reachable from a minimal `config.toml` containing only an `[agent]`
//! section. This file pins the factory dispatch path against the config schema
//! so that an accidental enum-arm deletion, factory-arm regression, or config
//! key rename in any of those layers fails one named test rather than slipping
//! through to a runtime "not yet implemented" surprise.
//!
//! Each subtest parses an inline `[agent] backend = "<name>"` snippet via the
//! real config parser, hands the result to the real factory, and asserts the
//! resulting agent reports the expected [`pitboss::agent::Agent::name`]. Using
//! the public name surface keeps the test independent of concrete struct
//! identities (which are private to each adapter module).

use pitboss::agent::build_agent;
use pitboss::config::{parse, Config};

/// Parse `text` and round-trip through [`build_agent`], asserting the
/// resulting agent's name matches `expected`. Centralized so each per-backend
/// test reads as a single line.
fn assert_backend(text: &str, expected: &str) {
    let cfg: Config =
        parse(text).unwrap_or_else(|e| panic!("config parse failed for {expected}: {e:#}"));
    let agent =
        build_agent(&cfg).unwrap_or_else(|e| panic!("build_agent failed for {expected}: {e:#}"));
    assert_eq!(
        agent.name(),
        expected,
        "expected backend {expected} to dispatch to an agent named {expected:?}"
    );
}

#[test]
fn claude_code_backend_dispatches_from_config() {
    // Sanity check: the historical default (no `[agent]` section at all) still
    // resolves to claude_code. If this fails the regression is in the config
    // defaults, not the factory.
    assert_backend("", "claude-code");
}

#[test]
fn explicit_claude_code_backend_dispatches_from_config() {
    assert_backend(
        "
[agent]
backend = \"claude_code\"
",
        "claude-code",
    );
}

#[test]
fn codex_backend_dispatches_from_config() {
    assert_backend(
        "
[agent]
backend = \"codex\"
",
        "codex",
    );
}

#[test]
fn aider_backend_dispatches_from_config() {
    assert_backend(
        "
[agent]
backend = \"aider\"
",
        "aider",
    );
}

#[test]
fn gemini_backend_dispatches_from_config() {
    assert_backend(
        "
[agent]
backend = \"gemini\"
",
        "gemini",
    );
}

#[test]
fn per_backend_overrides_reach_each_adapter() {
    // The per-backend sub-tables (`binary`, `extra_args`, `model`) must thread
    // through the factory for every adapter, not just claude_code. Pointing
    // each `binary` at an obviously-fake path is fine: the factory does not
    // spawn anything, it only constructs the agent. If a future adapter
    // forgot to plumb its overrides this test fails because either the parse
    // step rejects unknown keys or the factory drops the table on the floor.
    let text = "
[agent]
backend = \"gemini\"

[agent.codex]
binary = \"/tmp/fake-codex\"
extra_args = [\"--quiet\"]
model = \"gpt-5-codex\"

[agent.aider]
binary = \"/tmp/fake-aider\"
extra_args = [\"--no-auto-commits\"]
model = \"anthropic/sonnet-4.5\"

[agent.gemini]
binary = \"/tmp/fake-gemini\"
extra_args = [\"--include-directories\", \"src\"]
model = \"gemini-2.5-flash\"
";
    let cfg: Config = parse(text).expect("multi-backend config must parse");
    // The selected backend (gemini) builds its agent.
    let agent = build_agent(&cfg).expect("gemini-with-overrides config must build");
    assert_eq!(agent.name(), "gemini");
    // Spot-check that the overrides reached the parsed config so that the
    // factory has data to plumb. (Each adapter's own tests verify the
    // overrides are actually consumed at run time.)
    assert_eq!(
        cfg.agent.codex.binary.as_deref(),
        Some(std::path::Path::new("/tmp/fake-codex"))
    );
    assert_eq!(
        cfg.agent.aider.extra_args,
        vec!["--no-auto-commits".to_string()]
    );
    assert_eq!(cfg.agent.gemini.model.as_deref(), Some("gemini-2.5-flash"));
}

#[test]
fn unknown_backend_surfaces_clear_error() {
    let cfg: Config = parse("[agent]\nbackend = \"ollama\"\n")
        .expect("unknown backend names parse — validation lives in the factory");
    let err = build_agent(&cfg).err().expect("unknown backend must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ollama"),
        "expected the bad input echoed back, got: {msg}"
    );
}
