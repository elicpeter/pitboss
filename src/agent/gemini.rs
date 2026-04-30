//! `GeminiAgent` — production [`Agent`] that drives Google's `gemini` CLI in
//! non-interactive mode and parses its `--output-format json` document into
//! pitboss's [`AgentEvent`] / [`AgentOutcome`] vocabulary.
//!
//! ## How to install / configure `gemini`
//!
//! Pitboss shells out to whatever `gemini` binary is on `PATH` (or the path you
//! pass via `[agent.gemini] binary` in `pitboss.toml`). Install per Google's
//! Gemini CLI docs (`npm install -g @google/gemini-cli`) and configure
//! authentication (`GEMINI_API_KEY` env var, or `gemini auth`) before running
//! pitboss.
//!
//! Pitboss runs the agent under `--yolo` so it auto-approves every tool call,
//! `--output-format json` so the parser gets a structured terminal document
//! rather than ANSI-decorated text, and explicit `--model` / `--prompt`
//! arguments. Override via `[agent.gemini] extra_args = […]` if a workspace
//! needs different defaults.
//!
//! ## Prompt assembly
//!
//! Gemini has no separate system-prompt channel, so [`AgentRequest::system_prompt`]
//! and [`AgentRequest::user_prompt`] are concatenated — system first, blank line,
//! then user — and the whole payload is passed via `--prompt <body>`. The OS's
//! `ARG_MAX` (≥256 KB on every platform pitboss supports) is comfortably above
//! the prompts the runner produces today.
//!
//! ## Event mapping
//!
//! `gemini --output-format json` writes a single JSON document to stdout once
//! the run finishes. Two terminal shapes are supported:
//!
//! - Success — `{"response": "...", "stats": {...}}`. The `response` field
//!   becomes one [`AgentEvent::Stdout`]; each entry under `stats.tools.byName`
//!   becomes one [`AgentEvent::ToolUse`] carrying the tool name; the summed
//!   `stats.models.*.tokens.{prompt,candidates}` produces a single terminal
//!   [`AgentEvent::TokenDelta`] so the runner's accumulator doesn't double-
//!   count.
//! - Failure — `{"error": {"message": "...", ...}}`. The embedded message is
//!   captured and surfaced via [`StopReason::Error`].
//!
//! If stdout is not a parseable JSON document (older Gemini CLI, plain-text
//! mode, partial output), the buffered text is forwarded verbatim as a single
//! [`AgentEvent::Stdout`] so unexpected output stays visible. Stderr is
//! forwarded line-by-line and the trailing few lines are quoted into the
//! `StopReason::Error` message on a non-zero exit.
//!
//! Exit codes are interpreted per Gemini CLI's convention — known buckets
//! (auth, quota, network, usage, tool) get a short label appended to the error
//! message; unrecognized codes fall through to the bare numeric form.

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::state::TokenUsage;

use super::{
    subprocess::{self, SubprocessOutcome},
    Agent, AgentEvent, AgentOutcome, AgentRequest, StopReason,
};

/// Default binary name. Resolved against `PATH` by the OS.
const DEFAULT_BINARY: &str = "gemini";

/// How many trailing stderr lines to attach to a [`StopReason::Error`] when
/// the process exits non-zero. Bounded so a chatty error doesn't flood the
/// runner log.
const ERROR_TAIL_LINES: usize = 8;

/// Production [`Agent`] that drives the `gemini` CLI.
#[derive(Debug, Clone)]
pub struct GeminiAgent {
    binary: PathBuf,
    extra_args: Vec<String>,
    model_override: Option<String>,
}

impl GeminiAgent {
    /// Construct an agent that resolves `gemini` from `PATH`.
    pub fn new() -> Self {
        Self {
            binary: PathBuf::from(DEFAULT_BINARY),
            extra_args: Vec::new(),
            model_override: None,
        }
    }

    /// Construct an agent that invokes a specific binary path. Useful for
    /// tests (point at a fixture script) and for users with a non-standard
    /// install location.
    pub fn with_binary(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            extra_args: Vec::new(),
            model_override: None,
        }
    }

    /// Append extra argv that gets spliced in just before the `--prompt` flag
    /// on every invocation. Mirrors `[agent.gemini] extra_args` in
    /// `pitboss.toml`.
    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }

    /// Override the model identifier with a value from `[agent.gemini] model`.
    /// When set this beats the per-role model in [`AgentRequest::model`] —
    /// users who configure a backend-specific model expect it to be used for
    /// every dispatch through that backend.
    pub fn with_model_override(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }

    /// Path to the binary this agent will invoke.
    pub fn binary(&self) -> &PathBuf {
        &self.binary
    }
}

impl Default for GeminiAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for GeminiAgent {
    fn name(&self) -> &str {
        "gemini"
    }

    async fn run(
        &self,
        req: AgentRequest,
        events: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
    ) -> Result<AgentOutcome> {
        let log_path = req.log_path.clone();
        let cmd = self.build_command(&req);

        // Raw subprocess events flow through `raw_*` then a forwarder buffers
        // every stdout line (the JSON document arrives as a single chunk at
        // end-of-run, so per-line streaming has nothing useful to forward) and
        // tees stderr into a tail buffer for error attribution. Once the
        // child exits the buffered stdout is parsed once and the structured
        // events are emitted to the caller.
        let (raw_tx, mut raw_rx) = mpsc::channel::<AgentEvent>(64);
        let outbound = events.clone();
        let forwarder = tokio::spawn(async move {
            let mut stdout_buf = String::new();
            let mut stderr_tail: Vec<String> = Vec::new();
            while let Some(ev) = raw_rx.recv().await {
                match ev {
                    AgentEvent::Stdout(line) => {
                        if !stdout_buf.is_empty() {
                            stdout_buf.push('\n');
                        }
                        stdout_buf.push_str(&line);
                    }
                    AgentEvent::Stderr(line) => {
                        push_tail(&mut stderr_tail, line.clone(), ERROR_TAIL_LINES);
                        let _ = outbound.send(AgentEvent::Stderr(line)).await;
                    }
                    other => {
                        let _ = outbound.send(other).await;
                    }
                }
            }

            let parsed = parse_gemini_output(&stdout_buf);
            let mut tokens = TokenUsage::default();
            let mut error_message: Option<String> = None;
            match parsed {
                ParsedOutput::Success {
                    response,
                    tools,
                    token_usage,
                } => {
                    if let Some(text) = response {
                        if !text.is_empty() {
                            let _ = outbound.send(AgentEvent::Stdout(text)).await;
                        }
                    }
                    for tool in tools {
                        let _ = outbound.send(AgentEvent::ToolUse(tool)).await;
                    }
                    tokens = token_usage;
                }
                ParsedOutput::Error { message } => {
                    error_message = Some(message);
                }
                ParsedOutput::Unparseable => {
                    if !stdout_buf.is_empty() {
                        let _ = outbound.send(AgentEvent::Stdout(stdout_buf.clone())).await;
                    }
                }
            }
            if tokens.input > 0 || tokens.output > 0 {
                let _ = outbound.send(AgentEvent::TokenDelta(tokens.clone())).await;
            }
            ForwarderResult {
                tokens,
                error_message,
                stderr_tail,
            }
        });

        let sub_outcome: SubprocessOutcome =
            subprocess::run_logged(cmd, &log_path, raw_tx, cancel, req.timeout).await?;
        let ForwarderResult {
            mut tokens,
            error_message,
            stderr_tail,
        } = forwarder.await.unwrap_or(ForwarderResult {
            tokens: TokenUsage::default(),
            error_message: None,
            stderr_tail: Vec::new(),
        });
        // by_role isn't populated by the model itself — re-key once here so
        // the runner doesn't have to special-case Gemini's outcome shape.
        if tokens.input > 0 || tokens.output > 0 {
            tokens
                .by_role
                .entry(req.role.as_str().to_string())
                .or_default();
            let entry = tokens
                .by_role
                .get_mut(req.role.as_str())
                .expect("just inserted");
            entry.input = tokens.input;
            entry.output = tokens.output;
        }

        let stop_reason = match sub_outcome.stop_reason {
            StopReason::Completed => {
                if sub_outcome.exit_code == 0 && error_message.is_none() {
                    StopReason::Completed
                } else {
                    StopReason::Error(format_error_message(
                        sub_outcome.exit_code,
                        error_message.as_deref(),
                        &stderr_tail,
                    ))
                }
            }
            // Pass the terminator decided by the subprocess helper through
            // unchanged — timeout and cancel beat any error inferred from
            // partial output.
            other => other,
        };

        Ok(AgentOutcome {
            exit_code: sub_outcome.exit_code,
            stop_reason,
            tokens,
            log_path,
        })
    }
}

impl GeminiAgent {
    fn build_command(&self, req: &AgentRequest) -> Command {
        let mut cmd = Command::new(&self.binary);
        cmd.current_dir(&req.workdir);
        // Unattended-friendly defaults: auto-approve every tool call, emit a
        // structured JSON document instead of ANSI-decorated text. Anything a
        // workspace needs to flip can be supplied via [`Self::with_extra_args`].
        cmd.args(["--yolo", "--output-format", "json"]);
        let model = self.model_override.as_deref().unwrap_or(&req.model);
        cmd.args(["--model", model]);
        for arg in &self.extra_args {
            cmd.arg(arg);
        }
        cmd.arg("--prompt").arg(build_prompt_payload(req));
        cmd
    }
}

fn build_prompt_payload(req: &AgentRequest) -> String {
    let mut out = String::new();
    if !req.system_prompt.is_empty() {
        out.push_str(&req.system_prompt);
        out.push_str("\n\n");
    }
    out.push_str(&req.user_prompt);
    out
}

struct ForwarderResult {
    tokens: TokenUsage,
    error_message: Option<String>,
    stderr_tail: Vec<String>,
}

enum ParsedOutput {
    Success {
        response: Option<String>,
        tools: Vec<String>,
        token_usage: TokenUsage,
    },
    Error {
        message: String,
    },
    Unparseable,
}

fn parse_gemini_output(buf: &str) -> ParsedOutput {
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return ParsedOutput::Unparseable;
    }
    let value: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => return ParsedOutput::Unparseable,
    };
    if let Some(err_obj) = value.get("error") {
        let message = err_obj
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| "gemini reported an error".to_string());
        return ParsedOutput::Error { message };
    }
    let response = value
        .get("response")
        .and_then(Value::as_str)
        .map(str::to_string);
    let tools = extract_tool_calls(&value);
    let token_usage = extract_token_usage(&value);
    ParsedOutput::Success {
        response,
        tools,
        token_usage,
    }
}

fn extract_tool_calls(value: &Value) -> Vec<String> {
    let by_name = match value
        .get("stats")
        .and_then(|s| s.get("tools"))
        .and_then(|t| t.get("byName"))
        .and_then(Value::as_object)
    {
        Some(m) => m,
        None => return Vec::new(),
    };
    let mut out = Vec::with_capacity(by_name.len());
    // Iterate by insertion order (serde_json::Map preserves it when the
    // `preserve_order` feature isn't on, BTreeMap order otherwise — either way
    // is stable across runs of the same input, which is what tests need).
    for (name, entry) in by_name {
        let count = entry
            .get("count")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1);
        for _ in 0..count {
            out.push(name.clone());
        }
    }
    out
}

fn extract_token_usage(value: &Value) -> TokenUsage {
    let mut usage = TokenUsage::default();
    let models = match value
        .get("stats")
        .and_then(|s| s.get("models"))
        .and_then(Value::as_object)
    {
        Some(m) => m,
        None => return usage,
    };
    for (_, entry) in models {
        let tokens = match entry.get("tokens") {
            Some(t) => t,
            None => continue,
        };
        usage.input += read_u64(tokens, "prompt") + read_u64(tokens, "cached");
        usage.output += read_u64(tokens, "candidates") + read_u64(tokens, "thoughts");
    }
    usage
}

fn read_u64(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn push_tail(buf: &mut Vec<String>, line: String, max: usize) {
    if buf.len() == max {
        buf.remove(0);
    }
    buf.push(line);
}

fn format_error_message(exit_code: i32, parsed: Option<&str>, stderr_tail: &[String]) -> String {
    let label = exit_code_label(exit_code);
    let mut out = match (parsed, label) {
        (Some(m), Some(l)) if !m.is_empty() => {
            format!("gemini: {} ({}, exit {})", m, l, exit_code)
        }
        (Some(m), None) if !m.is_empty() => format!("gemini: {} (exit {})", m, exit_code),
        (_, Some(l)) => format!("gemini exited with code {} ({})", exit_code, l),
        (_, None) => format!("gemini exited with code {}", exit_code),
    };
    if !stderr_tail.is_empty() {
        out.push_str("\nstderr tail:\n");
        for line in stderr_tail {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Map a Gemini CLI exit code to a short human-readable label, or `None` for
/// unrecognized codes. The set covers Gemini CLI's documented buckets and is
/// intentionally narrow — unknown codes fall through to the bare numeric form
/// rather than getting a misleading label.
fn exit_code_label(code: i32) -> Option<&'static str> {
    match code {
        41 => Some("usage error"),
        42 => Some("authentication error"),
        43 => Some("quota exceeded"),
        44 => Some("network error"),
        53 => Some("tool error"),
        _ => None,
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::agent::Role;
    use std::path::PathBuf;
    use std::time::Duration;

    fn fixture_path(name: &str) -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest
            .join("tests")
            .join("fixtures")
            .join("gemini")
            .join(name)
    }

    fn req_with_log(log_path: PathBuf, timeout: Duration) -> AgentRequest {
        AgentRequest {
            role: Role::Implementer,
            model: "gemini-2.5-pro".into(),
            system_prompt: "be brief".into(),
            user_prompt: "say hi".into(),
            workdir: std::env::temp_dir(),
            log_path,
            timeout,
        }
    }

    async fn drain<T>(mut rx: mpsc::Receiver<T>) -> Vec<T> {
        let mut out = Vec::new();
        while let Some(v) = rx.recv().await {
            out.push(v);
        }
        out
    }

    #[tokio::test]
    async fn parses_response_tool_calls_and_token_stats() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let agent = GeminiAgent::with_binary(fixture_path("fake-gemini-success.sh"));
        let (tx, rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let outcome = agent
            .run(
                req_with_log(log.clone(), Duration::from_secs(5)),
                tx,
                cancel,
            )
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, StopReason::Completed);
        assert_eq!(outcome.exit_code, 0);

        let evs = drain(rx).await;
        let stdouts: Vec<&str> = evs
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Stdout(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let tool_uses: Vec<&str> = evs
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolUse(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let token_deltas: Vec<&TokenUsage> = evs
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TokenDelta(t) => Some(t),
                _ => None,
            })
            .collect();

        assert!(
            stdouts.iter().any(|s| s.contains("Hello from Gemini")),
            "missing assistant text: {stdouts:?}"
        );
        // Three tool calls total: one `list_directory` and two `edit_file`
        // (count=2). Document-order traversal of the byName map.
        assert_eq!(tool_uses.len(), 3);
        assert!(tool_uses.contains(&"list_directory"));
        assert_eq!(
            tool_uses.iter().filter(|t| **t == "edit_file").count(),
            2,
            "expected two edit_file tool-use events, got {tool_uses:?}"
        );
        assert_eq!(token_deltas.len(), 1);
        let total = token_deltas[0];
        // From fixture: prompt=1200, candidates=800.
        assert_eq!(total.input, 1200);
        assert_eq!(total.output, 800);

        // by_role re-keyed onto Role::Implementer at the agent level.
        assert_eq!(outcome.tokens.input, 1200);
        assert_eq!(outcome.tokens.output, 800);
        let role_usage = outcome
            .tokens
            .by_role
            .get("implementer")
            .expect("implementer role usage");
        assert_eq!(role_usage.input, 1200);
        assert_eq!(role_usage.output, 800);

        // Log file should contain the raw JSON document for post-mortem.
        let log_text = std::fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("\"response\""), "{log_text}");
        assert!(log_text.contains("edit_file"), "{log_text}");
    }

    #[tokio::test]
    async fn partial_output_with_no_stats_still_completes() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let agent = GeminiAgent::with_binary(fixture_path("fake-gemini-partial.sh"));
        let (tx, rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let outcome = agent
            .run(req_with_log(log, Duration::from_secs(5)), tx, cancel)
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, StopReason::Completed);

        let evs = drain(rx).await;
        let stdouts: Vec<&str> = evs
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Stdout(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let tool_uses: Vec<&str> = evs
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolUse(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let token_deltas: Vec<&TokenUsage> = evs
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TokenDelta(t) => Some(t),
                _ => None,
            })
            .collect();
        assert!(
            stdouts.iter().any(|s| s.contains("Nothing to change")),
            "expected response text, got {stdouts:?}"
        );
        // No `stats.tools` → no ToolUse events. No `stats.models` → no token
        // events. The agent must not invent or panic on the missing fields.
        assert!(
            tool_uses.is_empty(),
            "partial run should produce no tool-use events, got {tool_uses:?}"
        );
        assert!(
            token_deltas.is_empty(),
            "partial run should produce no token deltas, got {token_deltas:?}"
        );
        assert_eq!(outcome.tokens.input, 0);
        assert_eq!(outcome.tokens.output, 0);
    }

    #[tokio::test]
    async fn error_event_maps_to_error_stop_reason_with_exit_label() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let agent = GeminiAgent::with_binary(fixture_path("fake-gemini-error.sh"));
        let (tx, _rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let outcome = agent
            .run(req_with_log(log, Duration::from_secs(5)), tx, cancel)
            .await
            .unwrap();
        match outcome.stop_reason {
            StopReason::Error(msg) => {
                assert!(
                    msg.contains("GEMINI_API_KEY"),
                    "expected embedded message, got: {msg}"
                );
                // Exit code 42 → "authentication error" label per
                // [`exit_code_label`].
                assert!(
                    msg.contains("authentication error"),
                    "expected exit-code label, got: {msg}"
                );
            }
            other => panic!("expected Error, got {other:?}"),
        }
        assert_eq!(outcome.exit_code, 42);
    }

    #[tokio::test]
    async fn nonzero_exit_without_json_falls_back_to_stderr_tail() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let agent = GeminiAgent::with_binary(fixture_path("fake-gemini-crash.sh"));
        let (tx, _rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let outcome = agent
            .run(req_with_log(log, Duration::from_secs(5)), tx, cancel)
            .await
            .unwrap();
        match outcome.stop_reason {
            StopReason::Error(msg) => {
                assert!(msg.contains("exit"), "{msg}");
                assert!(
                    msg.contains("settings file"),
                    "expected stderr tail in error message, got: {msg}"
                );
            }
            other => panic!("expected Error, got {other:?}"),
        }
        assert_eq!(outcome.exit_code, 1);
    }

    #[tokio::test]
    async fn cancellation_propagates_to_child_process() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let agent = GeminiAgent::with_binary(fixture_path("fake-gemini-hang.sh"));
        let (tx, _rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let canceler = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(80)).await;
            canceler.cancel();
        });
        let outcome = agent
            .run(req_with_log(log, Duration::from_secs(30)), tx, cancel)
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, StopReason::Cancelled);
        assert_eq!(outcome.exit_code, -1);
    }

    #[tokio::test]
    async fn build_command_includes_required_flags_and_workdir() {
        let agent = GeminiAgent::with_binary("/usr/local/bin/gemini")
            .with_extra_args(vec!["--include-directories".into(), "src".into()])
            .with_model_override("gemini-2.5-flash");
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let req = AgentRequest {
            role: Role::Auditor,
            model: "ignored-because-override".into(),
            system_prompt: "system body".into(),
            user_prompt: "user body".into(),
            workdir: dir.path().to_path_buf(),
            log_path: log,
            timeout: Duration::from_secs(1),
        };
        let cmd = agent.build_command(&req);
        let std_cmd = cmd.as_std();
        let args: Vec<String> = std_cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        // Unattended defaults are present.
        assert!(args.iter().any(|a| a == "--yolo"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--output-format" && w[1] == "json"));
        // Model override beats AgentRequest::model.
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--model" && w[1] == "gemini-2.5-flash"));
        assert!(!args.iter().any(|a| a == "ignored-because-override"));
        // Extra args spliced in before --prompt, in declared order.
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--include-directories" && w[1] == "src"));
        // The prompt body is concatenated system + blank line + user, passed
        // via the trailing `--prompt` flag.
        let prompt_idx = args
            .iter()
            .position(|a| a == "--prompt")
            .expect("--prompt flag must be present");
        let body = &args[prompt_idx + 1];
        assert!(body.starts_with("system body\n\n"));
        assert!(body.ends_with("user body"));
        assert_eq!(std_cmd.get_program(), "/usr/local/bin/gemini");
        assert_eq!(std_cmd.get_current_dir(), Some(dir.path()));
    }

    #[tokio::test]
    async fn build_command_uses_request_model_when_no_override() {
        let agent = GeminiAgent::with_binary("gemini");
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let req = AgentRequest {
            role: Role::Implementer,
            model: "gemini-2.5-pro".into(),
            system_prompt: String::new(),
            user_prompt: "u".into(),
            workdir: dir.path().to_path_buf(),
            log_path: log,
            timeout: Duration::from_secs(1),
        };
        let cmd = agent.build_command(&req);
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--model" && w[1] == "gemini-2.5-pro"));
    }

    #[test]
    fn build_prompt_payload_concatenates_system_and_user_with_blank_line() {
        let req = AgentRequest {
            role: Role::Implementer,
            model: "x".into(),
            system_prompt: "you are a careful engineer".into(),
            user_prompt: "implement phase 04".into(),
            workdir: std::env::temp_dir(),
            log_path: std::env::temp_dir().join("never.log"),
            timeout: Duration::from_secs(1),
        };
        let payload = build_prompt_payload(&req);
        assert!(payload.starts_with("you are a careful engineer\n\n"));
        assert!(payload.contains("implement phase 04"));
    }

    #[test]
    fn build_prompt_payload_omits_system_when_empty() {
        let req = AgentRequest {
            role: Role::Implementer,
            model: "x".into(),
            system_prompt: String::new(),
            user_prompt: "just the user body".into(),
            workdir: std::env::temp_dir(),
            log_path: std::env::temp_dir().join("never.log"),
            timeout: Duration::from_secs(1),
        };
        let payload = build_prompt_payload(&req);
        assert_eq!(payload, "just the user body");
    }

    #[test]
    fn parse_gemini_output_handles_success_shape() {
        let buf = r#"{"response":"hi","stats":{"models":{"gemini-2.5-pro":{"tokens":{"prompt":10,"candidates":20,"cached":5,"thoughts":3}}},"tools":{"byName":{"a":{"count":1},"b":{"count":2}}}}}"#;
        match parse_gemini_output(buf) {
            ParsedOutput::Success {
                response,
                tools,
                token_usage,
            } => {
                assert_eq!(response.as_deref(), Some("hi"));
                // input = prompt(10) + cached(5); output = candidates(20) + thoughts(3).
                assert_eq!(token_usage.input, 15);
                assert_eq!(token_usage.output, 23);
                // 1 + 2 = 3 tool-use events; both names present.
                assert_eq!(tools.len(), 3);
                assert!(tools.contains(&"a".to_string()));
                assert_eq!(tools.iter().filter(|t| t.as_str() == "b").count(), 2);
            }
            other => panic!("expected Success, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn parse_gemini_output_handles_error_shape() {
        let buf = r#"{"error":{"type":"AuthError","message":"missing key"}}"#;
        match parse_gemini_output(buf) {
            ParsedOutput::Error { message } => {
                assert_eq!(message, "missing key");
            }
            _ => panic!("expected Error variant"),
        }
    }

    #[test]
    fn parse_gemini_output_treats_non_json_as_unparseable() {
        match parse_gemini_output("not json at all") {
            ParsedOutput::Unparseable => {}
            _ => panic!("expected Unparseable variant for non-JSON input"),
        }
        match parse_gemini_output("") {
            ParsedOutput::Unparseable => {}
            _ => panic!("expected Unparseable variant for empty input"),
        }
    }

    #[test]
    fn exit_code_label_covers_known_buckets() {
        assert_eq!(exit_code_label(42), Some("authentication error"));
        assert_eq!(exit_code_label(43), Some("quota exceeded"));
        assert_eq!(exit_code_label(44), Some("network error"));
        assert_eq!(exit_code_label(53), Some("tool error"));
        // Unrecognized codes return None — the formatter falls back to the
        // bare numeric form rather than guessing.
        assert_eq!(exit_code_label(1), None);
        assert_eq!(exit_code_label(99), None);
    }

    /// Real end-to-end test against the actual `gemini` binary on PATH.
    /// Skipped unless `PITBOSS_REAL_AGENT_TESTS=1` so CI doesn't burn tokens.
    #[tokio::test]
    async fn real_gemini_smoke_test() {
        if std::env::var("PITBOSS_REAL_AGENT_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping real_gemini_smoke_test (set PITBOSS_REAL_AGENT_TESTS=1 to run)");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("run.log");
        let agent = GeminiAgent::new();
        let (tx, _rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let req = AgentRequest {
            role: Role::Implementer,
            model: "gemini-2.5-pro".into(),
            system_prompt: String::new(),
            user_prompt: "respond with the single word OK".into(),
            workdir: dir.path().to_path_buf(),
            log_path: log,
            timeout: Duration::from_secs(120),
        };
        let outcome = agent.run(req, tx, cancel).await.unwrap();
        assert!(
            matches!(outcome.stop_reason, StopReason::Completed),
            "real gemini run did not complete: {:?}",
            outcome.stop_reason
        );
        assert_eq!(outcome.exit_code, 0);
    }
}
