#!/bin/sh
# Fake `codex` binary used by CodexAgent unit tests.
# Emits a representative subset of `codex exec --json` events for a
# successful run. Reads (and discards) the prompt body from stdin so that
# tests exercise the stdin-payload path.
set -eu

# Drain stdin so a full pipe doesn't stall the parent.
cat >/dev/null

cat <<'JSON'
{"id":"t-1","msg":{"type":"task_started"}}
{"id":"t-1","msg":{"type":"agent_reasoning","text":"reasoning..."}}
{"id":"t-1","msg":{"type":"agent_message","message":"Hello from Codex"}}
{"id":"t-1","msg":{"type":"exec_command_begin","call_id":"c1","command":["bash","-c","ls"],"cwd":"/tmp"}}
{"id":"t-1","msg":{"type":"exec_command_end","call_id":"c1","exit_code":0,"stdout":"file\n","stderr":""}}
{"id":"t-1","msg":{"type":"patch_apply_begin","call_id":"p1","auto_approved":true,"changes":{"src/foo.rs":{"add":{"content":"fn x(){}"}}}}}
{"id":"t-1","msg":{"type":"patch_apply_end","call_id":"p1","success":true,"stdout":"","stderr":""}}
{"id":"t-1","msg":{"type":"token_count","info":{"total_token_usage":{"input_tokens":12,"cached_input_tokens":8,"output_tokens":37}}}}
{"id":"t-1","msg":{"type":"task_complete","last_agent_message":"done"}}
JSON
exit 0
