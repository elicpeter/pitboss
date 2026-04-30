#!/bin/sh
# Emits a JSON `error` event and exits non-zero so CodexAgent maps the run
# to StopReason::Error.
set -eu

cat >/dev/null

cat <<'JSON'
{"id":"t-2","msg":{"type":"task_started"}}
{"id":"t-2","msg":{"type":"error","message":"rate limit exceeded — try again later"}}
JSON
exit 2
