---
description: Create a git commit from the current working tree changes, following repo conventions.
---

You are executing `/commit`. This creates a single commit from the current working tree. **No remote operations** — this is a local commit only.

## Step 1 — Inspect state

Run these in parallel:

- `git status` (never use `-uall`) — see tracked modifications and untracked files.
- `git diff` — staged and unstaged changes combined.
- `git log --oneline -10` — match the repo's commit message style.

## Step 2 — Decide what to stage

Analyze the changes and group them intelligently:

- If tracked modifications and untracked files clearly belong to the same logical change, stage them together by **explicit file name**. Never `git add -A` or `git add .`.
- If the changes look like **unrelated concerns** that should be separate commits, stop and ask the user which commit to make first (or whether to split into multiple).
- If untracked files look like stray artifacts (`.log`, scratch files, build output) or anything that smells like a secret (`.env*`, `credentials*`, keys), **stop and ask** — do not guess.
- Never commit files that could contain secrets without explicit confirmation.

## Step 3 — Draft the message

Match the style in `git log --oneline -10`. This repo uses conventional-style prefixes:

- `feat:` new functionality
- `fix:` bug fix
- `refactor:` restructuring without behavior change
- `test:` test additions or changes
- `docs:` documentation only
- `merge:` merge commits with context

Keep the subject line short (under ~70 chars) and focus on the **why**, not a mechanical list of what changed. One or two sentences is usually enough. If the change is small and self-explanatory, a single subject line is fine — no body needed.

### Test Suite Status trailer

Every commit must include a `Test Suite Status:` line near the end of the message, just above the `Co-Authored-By` trailer.

- If you have run the full test suite (`make test` with no `FILE=` or `-k` filter) **since the last code modification in this working tree**, include the actual counts:

  ```
  Test Suite Status: X passed, Y failed, Z skipped.
  ```

  Read the counts from `/tmp/torchwright-test.log` (or the log path printed by the latest `make test` run). Do **not** re-run tests just to produce this line — if the log is from before your most recent edit, treat the suite as not-run and use `Unknown` instead.

- If you have not run the full test suite since the last code change, or ran it only with filters (`FILE=` / `-k`), write:

  ```
  Test Suite Status: Unknown
  ```

Never fabricate numbers. "Unknown" is the correct answer whenever you aren't certain the counts reflect the committed code.

## Step 4 — Create the commit

Stage the chosen files explicitly, then commit using a HEREDOC for correct formatting:

```
git commit -m "$(cat <<'EOF'
<subject line>

<optional body focused on why>

Test Suite Status: X passed, Y failed, Z skipped.
Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Then run `git status` to confirm the commit landed and the tree is clean (or shows only the intended remaining changes).

## Step 5 — Handle hook failures

If a pre-commit hook fails:

1. The commit did **not** happen.
2. Read the hook output, fix the underlying issue.
3. Re-stage the fixes and create a **new** commit — never `--amend` after a hook failure.
4. Never bypass with `--no-verify`.

## Rules

- **Never push.** No `git push`, no `git fetch`.
- **Never force or reset.** No `--force`, `reset --hard`, `checkout --`, `clean -f`.
- **Never amend** unless the user explicitly asks.
- **Never skip hooks.** No `--no-verify`.
- **Never stage with `-A` or `.`** — always explicit file names.
- When in doubt, stop and ask.
