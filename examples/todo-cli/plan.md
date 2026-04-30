---
current_phase: "01"
project: todo-cli
---

# Todo CLI Implementation Plan

A tiny Rust binary that reads and writes a flat JSON file on disk and exposes
`add`, `list`, `done`, and `rm` subcommands. Designed as a pitboss dogfood
target: small enough to finish in a single run, large enough that the
implementer / fixer / auditor loop has something real to chew on.

# Phase 01: Cargo skeleton & CLI parsing

**Scope.** Stand up the crate, pin dependencies, parse the four subcommands
with `clap`. No business logic yet.

**Deliverables.**
- `Cargo.toml` with `clap = { version = "4", features = ["derive"] }`,
  `serde`, `serde_json`, and `anyhow`.
- `src/main.rs` parses `add <text>`, `list`, `done <id>`, `rm <id>`.
- Each subcommand prints a placeholder line (`unimplemented: add "..."`).
- `cargo build` and `cargo test` both pass clean (no tests yet).

**Acceptance.**
- `cargo run -- --help` lists all four subcommands.
- `cargo run -- add "buy milk"` prints `unimplemented: add "buy milk"` and
  exits 0.

# Phase 02: Storage layer

**Scope.** A single `Store` type that loads, mutates, and saves a JSON file.
File path defaults to `./todos.json`; configurable via `--file <path>`.

**Deliverables.**
- `src/store.rs` with:
  - `pub struct Todo { id: u32, text: String, done: bool }`.
  - `pub struct Store { path: PathBuf, items: Vec<Todo> }`.
  - `Store::load`, `Store::save` (atomic: write to `.tmp`, rename), `add`,
    `mark_done`, `remove`.
- Auto-incrementing `id` (max existing + 1, starting at 1).
- A missing file loads as an empty store; `save` creates parent dirs.
- Unit tests in `src/store.rs` covering the round-trip, empty load, and the
  three mutators.

**Acceptance.**
- `cargo test --lib store` passes with at least 4 tests.
- Saving and re-loading reproduces the same `Vec<Todo>`.

# Phase 03: Wire subcommands to the store

**Scope.** Replace the placeholder prints from phase 01 with real logic
backed by phase 02's `Store`.

**Deliverables.**
- `add <text>` → loads, appends, saves, prints `added #<id>: <text>`.
- `list` → prints one line per item, marking done items with `[x]`.
- `done <id>` → marks the item, errors if no match.
- `rm <id>` → removes, errors if no match.
- Errors surface as `anyhow::Result` and bubble out of `main` as non-zero
  exits.

**Acceptance.**
- A scripted shell sequence (in `tests/cli.rs`) exercises `add`, `list`,
  `done`, `rm` and asserts on the printed output.
- `cargo test` passes.

# Phase 04: Polish

**Scope.** Round-out UX (exit codes, color, `--help` examples) and document
what the binary does.

**Deliverables.**
- A short `README.md` explaining the four subcommands and the JSON file
  layout.
- `--version` flag (clap auto-wires this from `Cargo.toml`).
- Exit code conventions: 0 on success, 1 on usage / not-found errors.
- `cargo doc --no-deps` produces clean docs (every public item has a
  rustdoc comment).

**Acceptance.**
- `cargo doc --no-deps` is warning-free.
- `cargo test` and `cargo clippy -- -D warnings` both pass.
