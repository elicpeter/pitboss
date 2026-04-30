#!/bin/sh
# Emits a JSON document missing the `stats.tools` and `stats.models` fields,
# simulating a successful run that produced no tool calls and no model stats
# (e.g. cached / short response). The parser must still surface the response
# text and not panic on the missing fields.
set -eu

cat <<'JSON'
{
  "response": "Nothing to change — file already implements what you asked."
}
JSON
exit 0
