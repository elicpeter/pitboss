#!/usr/bin/env python3
"""Capture the README screenshots: TUI dashboard, `foreman status`, halted run.

Each scene is captured with `vhs` (charm.sh), then framed with a cyan
gradient that anchors on the wordmark color (#06b6d4 cyan-500). Output
PNGs land in `assets/`.

Pipeline per scene:
  1. set up whatever the scene needs
     - tui / halt: a temp Rust crate that depends on `foreman` via path
       and renders a single ratatui frame from a deterministic App state,
       then sleeps long enough for vhs to screenshot it
     - status: a temp foreman workspace (real `git init`, plan.md,
       deferred.md, foreman.toml, .foreman/state.json, one phase 01
       commit) so `foreman status` produces realistic output
  2. write a vhs tape, run vhs to produce a PNG
  3. wrap the PNG in the cyan gradient frame and save into assets/

Re-run any time. Everything below /tmp/foreman-screenshots-* is torn
down on exit (success or failure).

Prerequisites: vhs, python3+Pillow, rust toolchain, git.
Usage: python3 scripts/capture-screenshots.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FOREMAN_BIN = ROOT / "target" / "release" / "foreman"

# Frame geometry.
PAD_L, PAD_T, PAD_R, PAD_B = 80, 50, 80, 50
CORNER_RADIUS = 12

# Cyan gradient anchored on the wordmark cyan (#06b6d4). Lightest in TL,
# darkest in BR — the off-diagonal corners hold the wordmark color so
# the frame reads as a tonal range *of* the brand color, not a wash
# next to it.
GRAD_TL = (0xcf, 0xfa, 0xfe)  # cyan-100  — very light, airy
GRAD_TR = (0x38, 0xbd, 0xf8)  # sky-400   — vivid sky blue
GRAD_BL = (0x22, 0xd3, 0xee)  # cyan-400  — vivid cyan
GRAD_BR = (0x0e, 0x74, 0x90)  # cyan-700  — deep teal

# vhs visuals. One source of truth so every scene matches.
VHS_THEME = "Dracula"
VHS_FONT_SIZE = 14
VHS_PADDING = 24


def log(msg: str) -> None:
    print(f"[capture] {msg}", file=sys.stderr)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# Frame compositing
# ---------------------------------------------------------------------------


def make_gradient(w: int, h: int) -> Image.Image:
    """Bilinear gradient between the four GRAD_* corners.

    Built row-by-row by interpolating two edge gradients (top and bottom),
    then vertically blending — keeps a 1700x600 canvas under a few hundred
    ms vs the ten-second pure-Python pixel loop.
    """

    def edge(left, right):
        row = Image.new("RGB", (w, 1))
        px = row.load()
        for x in range(w):
            t = x / (w - 1) if w > 1 else 0.0
            px[x, 0] = (
                int(left[0] + (right[0] - left[0]) * t),
                int(left[1] + (right[1] - left[1]) * t),
                int(left[2] + (right[2] - left[2]) * t),
            )
        return row

    top, bot = edge(GRAD_TL, GRAD_TR), edge(GRAD_BL, GRAD_BR)
    out = Image.new("RGB", (w, h))
    for y in range(h):
        t = y / (h - 1) if h > 1 else 0.0
        out.paste(Image.blend(top, bot, t), (0, y))
    return out


def round_corners(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, img.size[0], img.size[1]), radius=radius, fill=255
    )
    out = img.convert("RGBA")
    out.putalpha(mask)
    return out


def frame_png(src: Path, dst: Path) -> None:
    inner = Image.open(src).convert("RGB")
    iw, ih = inner.size
    ow, oh = iw + PAD_L + PAD_R, ih + PAD_T + PAD_B
    bg = make_gradient(ow, oh).convert("RGBA")
    rounded = round_corners(inner, CORNER_RADIUS)
    bg.paste(rounded, (PAD_L, PAD_T), rounded)
    bg.convert("RGB").save(dst, "PNG", optimize=True)
    log(f"framed → {dst.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Demo binary (TUI + halt scenes)
# ---------------------------------------------------------------------------


DEMO_CARGO_TOML = f"""[package]
name = "foreman-screenshots-demo"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
foreman = {{ path = "{ROOT}" }}
ratatui = "0.30"
crossterm = "0.29"
"""

DEMO_MAIN_RS = r"""
use std::env;
use std::io;
use std::thread::sleep;
use std::time::Duration;

use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use foreman::git::CommitId;
use foreman::plan::{Phase, PhaseId, Plan};
use foreman::runner::{Event, HaltReason};
use foreman::state::RunState;
use foreman::tui::App;

fn pid(s: &str) -> PhaseId {
    PhaseId::parse(s).unwrap()
}

fn three_phase_plan() -> Plan {
    Plan::new(
        pid("01"),
        vec![
            Phase {
                id: pid("01"),
                title: "Project foundation".into(),
                body: String::new(),
            },
            Phase {
                id: pid("02"),
                title: "Domain types".into(),
                body: String::new(),
            },
            Phase {
                id: pid("03"),
                title: "Plan parser".into(),
                body: String::new(),
            },
        ],
    )
}

fn fresh_state() -> RunState {
    RunState::new(
        "20260430T120000Z",
        "foreman/run-20260430T120000Z",
        pid("01"),
    )
}

fn build_tui() -> App {
    let mut app = App::new(three_phase_plan(), fresh_state());
    app.handle_event(Event::PhaseStarted {
        phase_id: pid("01"),
        title: "Project foundation".into(),
        attempt: 1,
    });
    app.handle_event(Event::AgentStdout("Reading plan.md".into()));
    app.handle_event(Event::AgentStdout("Editing src/lib.rs".into()));
    app.handle_event(Event::TestStarted);
    app.handle_event(Event::TestFinished {
        passed: true,
        summary: "12 passed".into(),
    });
    app.handle_event(Event::PhaseCommitted {
        phase_id: pid("01"),
        commit: Some(CommitId::new("abc1234")),
    });
    app.handle_event(Event::PhaseStarted {
        phase_id: pid("02"),
        title: "Domain types".into(),
        attempt: 1,
    });
    app.handle_event(Event::AgentStdout("Defining PhaseId, Phase, Plan in src/plan/mod.rs".into()));
    app.handle_event(Event::AgentStdout("Adding parse() entry point with thiserror variants".into()));
    app.handle_event(Event::AgentStdout("Wiring serde derives onto Phase and Plan".into()));
    app.handle_event(Event::AgentStdout("Writing roundtrip test for plan::parse".into()));
    app.handle_event(Event::AgentStdout("cargo build --quiet (incremental)".into()));
    app
}

fn build_halt() -> App {
    let mut app = App::new(three_phase_plan(), fresh_state());
    app.handle_event(Event::PhaseStarted {
        phase_id: pid("02"),
        title: "Domain types".into(),
        attempt: 1,
    });
    app.handle_event(Event::AgentStdout("Defining PhaseId, Phase, Plan in src/plan/mod.rs".into()));
    app.handle_event(Event::AgentStdout("Adjusting parse() for trailing whitespace".into()));
    app.handle_event(Event::AgentStdout("Tightening edge case in dedent".into()));
    app.handle_event(Event::PhaseHalted {
        phase_id: pid("02"),
        reason: HaltReason::BudgetExceeded(
            "USD budget exhausted: $5.0123 spent, cap $5.0000".into(),
        ),
    });
    app
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = env::args().nth(1).unwrap_or_else(|| "tui".to_string());
    let app = match scene.as_str() {
        "tui" => build_tui(),
        "halt" => build_halt(),
        other => return Err(format!("unknown scene: {other}").into()),
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result: Result<(), Box<dyn std::error::Error>> = (|| {
        terminal.draw(|f| app.render(f))?;
        sleep(Duration::from_secs(6));
        Ok(())
    })();

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}
"""


def build_demo_binary(workdir: Path) -> Path:
    """Write Cargo.toml + src/main.rs into `workdir`, build release, return
    path to the compiled binary."""
    src = workdir / "src"
    src.mkdir(parents=True, exist_ok=True)
    (workdir / "Cargo.toml").write_text(DEMO_CARGO_TOML)
    (src / "main.rs").write_text(DEMO_MAIN_RS)
    log("building demo binary")
    run(["cargo", "build", "--release", "--quiet"], cwd=workdir)
    return workdir / "target" / "release" / "foreman-screenshots-demo"


# ---------------------------------------------------------------------------
# Foreman status workspace
# ---------------------------------------------------------------------------


STATUS_PLAN_MD = """---
current_phase: "02"
---

# Phase 01: Project foundation

Scaffold the crate, set up CI, drop in a CLAUDE.md.

# Phase 02: Domain types

Define PhaseId, Phase, Plan and their parser.

# Phase 03: Plan parser

Wire the parser into `foreman init` and `foreman run`.
"""

STATUS_DEFERRED_MD = """## Deferred items

- [ ] Stricter PhaseId validation (reject leading zeros)
- [x] Document the phase id grammar in plan.md

## Deferred phases

### From phase 02: rework parser entry point
"""

STATUS_FOREMAN_TOML = """[models]
implementer = "claude-opus-4-7"
auditor     = "claude-sonnet-4-6"
fixer       = "claude-sonnet-4-6"

[budgets]
max_total_tokens = 1000000
max_total_usd    = 5.00
"""


def setup_status_workspace(workdir: Path) -> None:
    """Build a workspace that produces a realistic `foreman status` output:
    one completed phase committed on the run branch, deferred items, real
    token usage so the budget block has numbers in it."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "plan.md").write_text(STATUS_PLAN_MD)
    (workdir / "deferred.md").write_text(STATUS_DEFERRED_MD)
    (workdir / "foreman.toml").write_text(STATUS_FOREMAN_TOML)

    # Real git history so `last commit:` resolves to a sensible value.
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "foreman",
        "GIT_AUTHOR_EMAIL": "foreman@example.com",
        "GIT_COMMITTER_NAME": "foreman",
        "GIT_COMMITTER_EMAIL": "foreman@example.com",
    }
    run(["git", "init", "--quiet", "--initial-branch=main"], cwd=workdir)
    run(["git", "checkout", "--quiet", "-b", "foreman/run-20260429T143022Z"], cwd=workdir, env=env)
    (workdir / "src").mkdir()
    (workdir / "src" / "lib.rs").write_text("// scaffold\n")
    run(["git", "add", "src/lib.rs"], cwd=workdir, env=env)
    run(
        ["git", "commit", "--quiet", "-m", "[foreman] phase 01: Project foundation"],
        cwd=workdir,
        env=env,
    )

    state_dir = workdir / ".foreman"
    state_dir.mkdir()
    state = {
        "run_id": "20260429T143022Z",
        "branch": "foreman/run-20260429T143022Z",
        "original_branch": "main",
        "started_at": "2026-04-29T14:30:22Z",
        "started_phase": "01",
        "completed": ["01"],
        "attempts": {"02": 2},
        "token_usage": {
            "input": 12850,
            "output": 4210,
            "by_role": {
                "implementer": {"input": 9000, "output": 2910},
                "auditor": {"input": 2100, "output": 480},
                "fixer": {"input": 1750, "output": 820},
            },
        },
        "aborted": False,
    }
    (state_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


# ---------------------------------------------------------------------------
# vhs orchestration
# ---------------------------------------------------------------------------


def write_tape(
    tape_path: Path,
    output_png: Path,
    command: str,
    *,
    width: int,
    height: int,
    pre_screenshot_sleep_ms: int = 2500,
    post_command_sleep_ms: int = 1500,
    cwd: str | None = None,
) -> None:
    """Render a vhs tape that types `command`, waits for it to settle, and
    screenshots once. Setup steps (cd, PATH export, clear) run inside a
    Hide block so the captured frame is just `> {command}` and its
    output."""
    setup = ""
    if cwd:
        setup += f'  Type "cd {cwd}"\n  Enter\n  Sleep 200ms\n'
    foreman_bin_dir = str(FOREMAN_BIN.parent)
    setup += (
        f'  Type "export PATH={foreman_bin_dir}:$PATH"\n  Enter\n  Sleep 200ms\n'
    )
    setup += '  Type "clear"\n  Enter\n  Sleep 200ms\n'
    tape = f"""Output "{output_png}"

Set Theme "{VHS_THEME}"
Set FontSize {VHS_FONT_SIZE}
Set Width {width}
Set Height {height}
Set Padding {VHS_PADDING}
Set Shell "bash"

Hide
{setup}  Type "{command}"
  Enter
  Sleep {post_command_sleep_ms}ms
Show
Sleep {pre_screenshot_sleep_ms}ms
Screenshot "{output_png}"
Sleep 300ms
"""
    tape_path.write_text(tape)


def run_vhs(tape: Path) -> None:
    log(f"vhs ← {tape.name}")
    run(["vhs", str(tape)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Scene driver
# ---------------------------------------------------------------------------


def ensure_foreman_built() -> None:
    if FOREMAN_BIN.exists():
        return
    log("building foreman release binary")
    run(["cargo", "build", "--release", "--quiet"], cwd=ROOT)


def capture_all() -> None:
    ensure_foreman_built()
    ASSETS.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="foreman-screenshots-") as tmp:
        tmp = Path(tmp)
        demo_dir = tmp / "demo"
        status_dir = tmp / "status-workspace"
        captures = tmp / "captures"
        captures.mkdir()

        demo_bin = build_demo_binary(demo_dir)
        setup_status_workspace(status_dir)

        scenes: list[tuple[str, str, int, int]] = [
            # name           command                w     h
            ("foreman-tui",    f"{demo_bin} tui",   1500, 480),
            ("foreman-halt",   f"{demo_bin} halt",  1500, 460),
            ("foreman-status", "foreman status",     880, 360),
        ]

        for name, command, w, h in scenes:
            tape = captures / f"{name}.tape"
            png = captures / f"{name}.png"
            cwd = str(status_dir) if name == "foreman-status" else None
            write_tape(tape, png, command, width=w, height=h, cwd=cwd)
            run_vhs(tape)
            frame_png(png, ASSETS / f"{name}.png")

    log("done")


if __name__ == "__main__":
    capture_all()
