<!-- pitboss:standing-instruction:start -->
You are running inside a `pitboss grind` session. A few standing rules apply
to every grind prompt regardless of what the user-authored body asks you to
do.

**Session summary.** Before you exit, write a short summary of what you did
this session to the file at `$PITBOSS_SUMMARY_FILE`. One paragraph is plenty:
what you changed, what you decided to skip, what the next session should pick
up. If you accomplished nothing, say so explicitly. The summary is the only
record that survives into the next session's context, so it has to stand on
its own without the transcript.

**Scratchpad.** A long-lived freeform notes file lives at
`$PITBOSS_SCRATCHPAD`. It is shared across every session in this run. Read it
at the start to see what previous sessions left behind, and append (do not
overwrite) anything the next session will need: hypotheses you ruled out,
flaky areas of the code, mental models, TODOs that are too small for
`deferred.md`. Keep entries dated and short.

**Session log.** A markdown projection of every prior session is auto-injected
into your prompt under the `<!-- pitboss:session-log -->` marker — you do not
need to read it from disk. The full source-of-truth lives at
`.pitboss/grind/runs/$PITBOSS_RUN_ID/sessions.jsonl` if you ever need to grep it.

**Identity.** This session's prompt is `$PITBOSS_PROMPT_NAME`, run id
`$PITBOSS_RUN_ID`, sequence `$PITBOSS_SESSION_SEQ`. Pitboss handles the git
commit at the end of the session — focus on the work, not on bookkeeping.

**Working directory.** The path of the tree you are editing is in
`$PITBOSS_WORKTREE`. For sequential sessions this is the workspace root; for
parallel-safe sessions it is a per-session git worktree under
`.pitboss/grind/runs/$PITBOSS_RUN_ID/worktrees/session-NNNN/`. Use this when
inspecting or rebuilding the tree from a hook or sub-tool.
<!-- pitboss:standing-instruction:end -->
