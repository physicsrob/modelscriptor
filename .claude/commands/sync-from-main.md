---
description: Commit any WIP, then merge local main into the current branch. Does NOT advance main.
---

You are executing `/sync-from-main`. This pulls the latest local `main` into a feature-branch worktree. It does **not** advance `main` afterward — this is a one-way sync from `main` into the current branch. **No remote operations** — everything is local.

The main worktree lives at `/home/rob/workspace/torchwright`. Feature branches live in sibling worktrees (typically under `.claude/worktrees/<name>/`).

## Step 1 — Guard: refuse to run on main

Run `git rev-parse --abbrev-ref HEAD`. If it is `main`, print a friendly message like:

> `/sync-from-main` only runs from a feature branch — you're on `main`. Switch to a feature worktree and try again.

Then stop. Do not proceed.

Capture the current branch name as `$BRANCH` for later steps.

## Step 2 — Commit any uncommitted work

Run `git status` and `git diff` (both staged and unstaged, and check for untracked files).

- If the working tree is already clean, skip to Step 3.
- Otherwise, commit everything following the `/commit` protocol (the full protocol from your system prompt): analyze the diff, follow the repo's commit message style by checking `git log`, draft a concise "why"-focused message, and include the `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` trailer via HEREDOC.

**Intelligently decide what to stage:**

- Analyze tracked modifications and untracked files. If they clearly belong to the same logical change, stage them together (by explicit file names — never `git add -A` or `git add .`).
- If the changes look like **unrelated concerns** that should be separate commits, or if untracked files look like stray artifacts (e.g., `.log`, local scratch files, anything that smells like a secret such as `.env*`, `credentials*`), **stop and ask the user** how to proceed. Do not guess.
- Never commit files that could contain secrets without explicit confirmation.

After committing, verify with `git status` that the working tree is clean before continuing.

## Step 3 — Merge local `main` into `$BRANCH`

From the current worktree, run:

```
git merge main --no-edit
```

Note: this merges **local** `main`, not `origin/main`. Do not fetch.

If the merge is clean, print the result of `git log --oneline -3` so the user can confirm the new tip. Done.

If there are conflicts:

1. Attempt obvious auto-resolutions only — cases where the "right" answer is unambiguous:
   - Both sides added the identical content
   - One side's change is a strict superset of the other in a way git can verify
   - Trivial whitespace-only conflicts
2. For **anything non-trivial** (overlapping real edits, deleted-vs-modified, anything requiring a judgment call about intent), stop and ask the user for guidance. Show them the conflicting files and a short summary of each conflict. Do not invent resolutions.
3. Once all conflicts are resolved, stage the fixes and complete the merge with a standard merge commit (keep the default message).

If the user aborts the merge, run `git merge --abort` and stop.

## Rules

- **Never advance `main`.** This command is one-way only: `main` → `$BRANCH`. Do not fast-forward `main` to `$BRANCH` afterward — that's what `/sync` is for.
- **Never push.** No `git push`, no `git fetch`, no remote interaction of any kind.
- **Never force or reset.** No `--force`, no `reset --hard`, no `checkout --`, no `clean -f`.
- **Never skip hooks.** No `--no-verify`.
- **Never switch branches in the current worktree.** Stay on `$BRANCH` throughout.
- When in doubt, stop and ask.
