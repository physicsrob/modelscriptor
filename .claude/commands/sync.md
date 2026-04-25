---
description: Commit any WIP, merge local main into the current branch, then fast-forward main to match.
---

You are executing `/sync`. This syncs a feature-branch worktree with local `main`, advances local `main` to include the branch, and then pushes `main` to `origin`. The only remote operation is the final `main` push — no fetch, no branch push.

The main worktree lives at `/home/rob/workspace/torchwright`. Feature branches live in sibling worktrees (typically under `.claude/worktrees/<name>/`).

## Step 1 — Guard: refuse to run on main

Run `git rev-parse --abbrev-ref HEAD`. If it is `main`, print a friendly message like:

> `/sync` only runs from a feature branch — you're on `main`. Switch to a feature worktree and try again.

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

If the merge is clean, continue to Step 4.

If there are conflicts:

1. Attempt obvious auto-resolutions only — cases where the "right" answer is unambiguous:
   - Both sides added the identical content
   - One side's change is a strict superset of the other in a way git can verify
   - Trivial whitespace-only conflicts
2. For **anything non-trivial** (overlapping real edits, deleted-vs-modified, anything requiring a judgment call about intent), stop and ask the user for guidance. Show them the conflicting files and a short summary of each conflict. Do not invent resolutions.
3. Once all conflicts are resolved, stage the fixes and complete the merge with a standard merge commit (keep the default message).

If the user aborts the merge, run `git merge --abort` and stop.

## Step 4 — Fast-forward local `main` to `$BRANCH`

Run, from the main worktree path:

```
cd /home/rob/workspace/torchwright && git merge --ff-only $BRANCH && git log --oneline -3
```

(Substitute the actual branch name for `$BRANCH`.)

Interpreting the result:

- **Success** — print the `git log --oneline -3` output so the user can confirm main now points at the branch tip. Continue to Step 5.
- **Failure because local `main` moved** (non-fast-forward — i.e., `main` has commits that `$BRANCH` doesn't) — return to Step 3 and retry the whole sync. This can happen if another worktree advanced `main` during sync. Retry at most **once**; if it fails a second time, stop and report.
- **Any other failure** — stop and report the error to the user. Do not attempt recovery.

## Step 5 — Push `main` to `origin`

From the main worktree path, run:

```
cd /home/rob/workspace/torchwright && git push origin main
```

Print the push output so the user can see the result.

- **Success** — print "Safe to delete this worktree." and done.
- **Failure because `origin/main` has diverged** (non-fast-forward) — stop and report. Do not force-push, do not fetch-and-retry. The user must resolve manually.
- **Any other failure** (network, auth, hook) — stop and report the error. Do not attempt recovery.

## Rules

- **Only push `main`.** Never push the feature branch. Never `git fetch`. The `main` push in Step 5 is the only remote interaction.
- **Never force or reset.** No `--force`, no `reset --hard`, no `checkout --`, no `clean -f`.
- **Never skip hooks.** No `--no-verify`.
- **Never switch branches in the current worktree.** Step 4 uses `cd` into the main worktree so the feature worktree stays on `$BRANCH`.
- When in doubt, stop and ask.
