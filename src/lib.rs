//! Pitboss library — orchestrates coding agents through a phased implementation plan.
//!
//! See `plan.md` for the full design. This crate exposes the modules the CLI and
//! integration tests build on.

pub mod agent;
pub mod cli;
pub mod config;
pub mod deferred;
pub mod git;
pub mod grind;
pub mod plan;
pub mod prompts;
pub mod runner;
pub mod state;
pub mod style;
pub mod tests;
pub mod tui;
pub mod util;
