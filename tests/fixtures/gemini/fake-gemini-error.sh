#!/bin/sh
# Emits a JSON document with the error variant and exits with code 42 (the
# Gemini CLI's "auth error" exit code). The parser must surface the embedded
# message and the exit-code label in the StopReason::Error.
set -eu

cat <<'JSON'
{
  "error": {
    "type": "AuthError",
    "message": "missing GEMINI_API_KEY — run `gemini auth` first",
    "code": 42
  }
}
JSON
exit 42
