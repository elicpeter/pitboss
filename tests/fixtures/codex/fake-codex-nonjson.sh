#!/bin/sh
# Emits a non-JSON stdout line followed by a normal task_complete, exercising
# the parser fallback path in CodexAgent.
set -eu

cat >/dev/null

printf 'not-json output line\n'
cat <<'JSON'
{"id":"t-3","msg":{"type":"agent_message","message":"after a stray line"}}
{"id":"t-3","msg":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}}}
{"id":"t-3","msg":{"type":"task_complete","last_agent_message":"ok"}}
JSON
exit 0
