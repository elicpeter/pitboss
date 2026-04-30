#!/bin/sh
# Fake `gemini` binary used by GeminiAgent unit tests.
# Emits the single JSON document `gemini --output-format json` produces on a
# successful non-interactive run, including model token stats and a tools
# summary so the parser exercises both fields.
set -eu

cat <<'JSON'
{
  "response": "Hello from Gemini — implementing the change now.",
  "stats": {
    "models": {
      "gemini-2.5-pro": {
        "tokens": {
          "prompt": 1200,
          "candidates": 800,
          "total": 2000,
          "cached": 0,
          "thoughts": 0,
          "tool": 0
        },
        "api": {
          "totalRequests": 1
        }
      }
    },
    "tools": {
      "totalCalls": 3,
      "totalSuccess": 3,
      "totalFail": 0,
      "totalDurationMs": 142,
      "byName": {
        "list_directory": {
          "count": 1,
          "success": 1,
          "fail": 0
        },
        "edit_file": {
          "count": 2,
          "success": 2,
          "fail": 0
        }
      }
    },
    "files": {
      "totalLinesAdded": 12,
      "totalLinesRemoved": 3
    }
  }
}
JSON
exit 0
