---
name: __NAME__
description: One-line description of what this prompt does.
weight: 1
every: 1
verify: false
parallel_safe: false
tags: []
# Optional caps. Uncomment and set as needed:
# max_runs: 10
# max_session_seconds: 600
# max_session_cost_usd: 1.00
---

Replace this body with the instructions you want the agent to follow on each
rotation. The body is appended to the auto-injected standing instruction
block, so focus on the work specific to this prompt.

For example:
- Read the most recent session summaries from the auto-injected log.
- Pick one open item and make progress on it.
- Record what you did in $PITBOSS_SUMMARY_FILE before exiting.
