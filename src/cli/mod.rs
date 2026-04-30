//! Clap command definitions and dispatch.
//!
//! `init` is implemented (phase 5); `run` (phase 12), `plan` (phase 15), and
//! the lifecycle trio `status` / `resume` / `abort` (phase 17) round out the
//! current surface. Later phases plug into the same dispatch table.

use anyhow::Result;
use clap::{Parser, Subcommand};

pub mod abort;
pub mod init;
pub mod plan;
pub mod resume;
pub mod run;
pub mod status;

#[derive(Debug, Parser)]
#[command(
    name = "foreman",
    version,
    about = "Orchestrate coding agents through a phased plan"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Scaffold a new foreman workspace in the current directory.
    Init,
    /// Generate a `plan.md` for a goal using the planner agent.
    Plan {
        /// Free-form description of what to build.
        goal: String,
        /// Overwrite an existing `plan.md`. Without this flag the command
        /// refuses to clobber a hand-written or `foreman init` seed file.
        #[arg(long)]
        force: bool,
    },
    /// Execute the plan, advancing through phases until done or halted.
    Run {
        /// Render a live `ratatui` dashboard instead of the plain logger.
        #[arg(long)]
        tui: bool,
    },
    /// Print a summary of the current run.
    Status,
    /// Resume a halted run from where it left off.
    Resume {
        /// Render a live `ratatui` dashboard instead of the plain logger.
        #[arg(long)]
        tui: bool,
    },
    /// Mark the active run as aborted. `foreman run` and `foreman resume`
    /// refuse to operate on an aborted state.
    Abort {
        /// After marking the run aborted, switch HEAD back to the branch that
        /// was checked out when the run began (when known).
        #[arg(long)]
        checkout_original: bool,
    },
}

/// Dispatch a parsed CLI invocation.
pub async fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Init => init::run(std::env::current_dir()?),
        Command::Plan { goal, force } => plan::run(std::env::current_dir()?, goal, force).await,
        Command::Run { tui } => run::run(std::env::current_dir()?, tui).await,
        Command::Status => status::run(std::env::current_dir()?),
        Command::Resume { tui } => resume::run(std::env::current_dir()?, tui).await,
        Command::Abort { checkout_original } => {
            abort::run(std::env::current_dir()?, checkout_original).await
        }
    }
}
