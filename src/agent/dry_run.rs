//! Deterministic stand-in [`Agent`] used by every test downstream of phase 7.
//!
//! Every later phase exercises the runner against `DryRunAgent` rather than a
//! real `claude` invocation, so the runner's behavior under success, failure,
//! timeout, and cancellation is testable with no model spend and no flaky
//! subprocess timing.
//!
//! Build one with [`DryRunAgent::new`] and chain `.emit(...)`, `.wait(...)`,
//! and `.finish(...)` to script the run. The script is replayed in order,
//! then the configured [`DryRunFinal`] determines the outcome:
//!
//! ```ignore
//! let agent = DryRunAgent::new("test")
//!     .emit(AgentEvent::Stdout("starting".into()))
//!     .wait(Duration::from_millis(5))
//!     .emit(AgentEvent::ToolUse("write".into()))
//!     .finish(DryRunFinal::Success { exit_code: 0, tokens: TokenUsage::default() });
//! ```

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::state::TokenUsage;

use super::{Agent, AgentEvent, AgentOutcome, AgentRequest, StopReason};

/// One step in a [`DryRunAgent`] script.
#[derive(Debug, Clone)]
pub enum DryRunStep {
    /// Send an event on the agent's channel.
    Emit(AgentEvent),
    /// Sleep — useful to push events past the cancel/timeout deadline.
    Wait(Duration),
}

/// What the agent does after its scripted [`DryRunStep`]s finish.
#[derive(Debug, Clone)]
pub enum DryRunFinal {
    /// Resolve normally with the supplied tokens and exit code.
    Success {
        /// Process exit code reported in the [`AgentOutcome`].
        exit_code: i32,
        /// Total token usage reported in the [`AgentOutcome`].
        tokens: TokenUsage,
    },
    /// Resolve with [`StopReason::Error`] carrying `message`.
    Error(String),
    /// Hang forever, leaving the cancel/timeout branch to fire. Used to
    /// test the runner's terminator paths.
    Hang,
}

/// Test-only agent. See module docs for usage.
pub struct DryRunAgent {
    name: String,
    script: Vec<DryRunStep>,
    finish: DryRunFinal,
}

impl DryRunAgent {
    /// New scriptable agent with no events queued and a default success
    /// outcome (`exit_code: 0`, zero tokens).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            script: Vec::new(),
            finish: DryRunFinal::Success {
                exit_code: 0,
                tokens: TokenUsage::default(),
            },
        }
    }

    /// Append an event to the script.
    pub fn emit(mut self, event: AgentEvent) -> Self {
        self.script.push(DryRunStep::Emit(event));
        self
    }

    /// Append a sleep to the script.
    pub fn wait(mut self, d: Duration) -> Self {
        self.script.push(DryRunStep::Wait(d));
        self
    }

    /// Replace the configured terminal behavior.
    pub fn finish(mut self, finish: DryRunFinal) -> Self {
        self.finish = finish;
        self
    }
}

#[async_trait]
impl Agent for DryRunAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(
        &self,
        req: AgentRequest,
        events: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
    ) -> Result<AgentOutcome> {
        let log_path = req.log_path.clone();

        let work = async {
            for step in &self.script {
                match step {
                    DryRunStep::Emit(e) => {
                        let _ = events.send(e.clone()).await;
                    }
                    DryRunStep::Wait(d) => tokio::time::sleep(*d).await,
                }
            }
            match &self.finish {
                DryRunFinal::Success { exit_code, tokens } => AgentOutcome {
                    exit_code: *exit_code,
                    stop_reason: StopReason::Completed,
                    tokens: tokens.clone(),
                    log_path: log_path.clone(),
                },
                DryRunFinal::Error(msg) => AgentOutcome {
                    exit_code: 1,
                    stop_reason: StopReason::Error(msg.clone()),
                    tokens: TokenUsage::default(),
                    log_path: log_path.clone(),
                },
                DryRunFinal::Hang => {
                    std::future::pending::<()>().await;
                    unreachable!("std::future::pending never resolves");
                }
            }
        };

        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => AgentOutcome {
                exit_code: -1,
                stop_reason: StopReason::Cancelled,
                tokens: TokenUsage::default(),
                log_path: log_path.clone(),
            },
            _ = tokio::time::sleep(req.timeout) => AgentOutcome {
                exit_code: -1,
                stop_reason: StopReason::Timeout,
                tokens: TokenUsage::default(),
                log_path: log_path.clone(),
            },
            o = work => o,
        };
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Role;
    use std::path::PathBuf;

    fn req(timeout: Duration) -> AgentRequest {
        AgentRequest {
            role: Role::Implementer,
            model: "test-model".into(),
            system_prompt: String::new(),
            user_prompt: String::new(),
            workdir: PathBuf::from("/tmp"),
            log_path: PathBuf::from("/tmp/dry-run.log"),
            timeout,
            env: std::collections::HashMap::new(),
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
    async fn success_path_streams_events_and_returns_completed() {
        let tokens = TokenUsage {
            input: 10,
            output: 5,
            ..Default::default()
        };
        let agent = DryRunAgent::new("test")
            .emit(AgentEvent::Stdout("hello".into()))
            .emit(AgentEvent::ToolUse("write".into()))
            .finish(DryRunFinal::Success {
                exit_code: 0,
                tokens: tokens.clone(),
            });
        let (tx, rx) = mpsc::channel(8);
        let cancel = CancellationToken::new();
        let outcome = agent
            .run(req(Duration::from_secs(5)), tx, cancel)
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, StopReason::Completed);
        assert_eq!(outcome.exit_code, 0);
        assert_eq!(outcome.tokens, tokens);
        assert_eq!(outcome.log_path, PathBuf::from("/tmp/dry-run.log"));
        let events = drain(rx).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], AgentEvent::Stdout(ref s) if s == "hello"));
        assert!(matches!(events[1], AgentEvent::ToolUse(ref s) if s == "write"));
    }

    #[tokio::test]
    async fn failure_path_returns_error_stop_reason() {
        let agent = DryRunAgent::new("test").finish(DryRunFinal::Error("boom".into()));
        let (tx, _rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let outcome = agent
            .run(req(Duration::from_secs(1)), tx, cancel)
            .await
            .unwrap();
        match outcome.stop_reason {
            StopReason::Error(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected Error, got {:?}", other),
        }
        assert_eq!(outcome.exit_code, 1);
    }

    #[tokio::test]
    async fn timeout_path_fires_when_agent_hangs() {
        let agent = DryRunAgent::new("test").finish(DryRunFinal::Hang);
        let (tx, _rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let outcome = agent
            .run(req(Duration::from_millis(40)), tx, cancel)
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, StopReason::Timeout);
        assert_eq!(outcome.exit_code, -1);
    }

    #[tokio::test]
    async fn cancellation_path_aborts_a_hanging_agent() {
        let agent = DryRunAgent::new("test").finish(DryRunFinal::Hang);
        let (tx, _rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let canceler = cancel.clone();
        let trigger = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            canceler.cancel();
        });
        let outcome = agent
            .run(req(Duration::from_secs(60)), tx, cancel)
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, StopReason::Cancelled);
        assert_eq!(outcome.exit_code, -1);
        trigger.await.unwrap();
    }

    #[tokio::test]
    async fn cancellation_wins_when_already_signalled_at_start() {
        // Pre-cancelled token must short-circuit the script entirely.
        let agent = DryRunAgent::new("test")
            .emit(AgentEvent::Stdout("never sent".into()))
            .finish(DryRunFinal::Success {
                exit_code: 0,
                tokens: TokenUsage::default(),
            });
        let (tx, _rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let outcome = agent
            .run(req(Duration::from_secs(5)), tx, cancel)
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, StopReason::Cancelled);
    }
}
