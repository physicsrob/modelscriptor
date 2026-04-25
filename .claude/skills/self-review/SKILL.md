---
name: self-review
description: Run a multi-reviewer Socratic self-review on a draft before shipping it to the user. Spawns up to four narrow-mandate reviewer subagents in parallel (terminology, self-containment, verification, invariant), each issuing probing questions; the main agent answers, the reviewers approve or deny with feedback, the main agent refines and re-submits up to one round, and any standing denials are surfaced to the user. INVOKE before shipping a load-bearing draft — design proposals, mechanism explanations, root-cause claims, scope summaries, work-completion summaries with factual claims, canonized documentation (CLAUDE.md additions, postmortems, design docs). DO NOT invoke for short factual answers, tool-call narration, progress updates, or clarifying questions.
---

# self-review skill — orchestrator

When a draft is load-bearing, spawn narrow-mandate reviewer subagents in parallel and run a bounded probe-and-refine loop before shipping. This exists because draft errors cluster at specific moments (root-cause claims, design proposals, mechanism explanations, canonized documentation) and most of them are detectable by a careful reader who is allowed to ask probing questions.

## When to invoke

Invoke when you are about to ship one of the following to the user:

- A **plan or design proposal** (architectural decisions, dataclass definitions, new token types, phase structures).
- A **mechanism explanation** (how a compiled op works, why a regression happens, what a layer does).
- A **research or investigation result** (root-cause claim, bisect outcome, summary of what was checked).
- **Canonized documentation** — text that will persist beyond this session: CLAUDE.md additions, postmortems, design docs, commit messages that describe behavior.
- A **work-completion summary** that claims something factual ("3 walls rendered", "clean merge", "all tests pass"). These are historically a high-error category.

Do not invoke on: short factual answers, tool-call narration, progress updates, clarifying questions back to the user, or internal investigation-in-progress messages you have no intent to ship.

## How to invoke

Spawn up to four reviewer subagents **in parallel**, each as a named agent you can resume via SendMessage. Each reviewer's prompt is the contents of the corresponding file in `reviewers/`:

- `reviewer-terminology` — `reviewers/terminology.md`
- `reviewer-self-containment` — `reviewers/self_containment.md`
- `reviewer-verification` — `reviewers/verification.md`
- `reviewer-invariant` — `reviewers/invariant.md` (skip if the session has no established invariants and no project rules plausibly apply)

**Capture the `agentId` returned by each spawn.** In Claude Code's harness, an async agent terminates after returning its result; SendMessage to the agent's *name* may fail once it has completed a turn, but SendMessage to its *agentId* resumes it from transcript. Track a `{name → agentId}` map before round 1 and use the agentId for all subsequent SendMessage calls.

Each reviewer gets the **full draft** as its input. Only `reviewer-invariant` additionally gets the **invariant list** — the decisions and constraints established in this session, plus the project rules (dumb-host principle, D1–D8 doctrine, and any other CLAUDE.md rule you judge relevant). Compile this list yourself before spawning; it is the context the invariant reviewer cannot reconstruct alone.

The reviewers do not know about each other. Do not tell them about each other.

## Protocol

**Round 1 — questions.** Each reviewer returns either `QUESTIONS: none` (proceed to its verdict immediately) or a list of questions. Answer each reviewer's questions separately via SendMessage to that reviewer's agentId. Answer honestly: when a question has no good answer, say "I didn't check" rather than constructing a plausible story — the reviewers exist to catch exactly that.

**Round 1 verdict.** Each reviewer returns `VERDICT: approve` or `VERDICT: deny` with feedback.

**Round 2 — refinement.** For each reviewer that denied, refine the draft to address the feedback. A single refinement pass should address all denying reviewers' feedback at once. Resubmit the refined draft to the denying reviewers only (approvers do not need to re-see). They return a new verdict.

**Round 2 verdict.** If all reviewers now approve, ship.

**Round 3 decision (your judgment).** If one or more reviewers still deny after round 2, you decide:

- **Iterate** — another round of refinement, if the objections are addressable and you have something new to try.
- **Proceed and surface** — ship the draft to the user, but in the same message, surface the reviewers' standing objections verbatim. Do not hide a denied review. Phrase as: "reviewer X is not satisfied: <its feedback>. I'm proceeding because <reason>."

Do not ship silently over a denied verdict. The user needs to know what was contested.

## When answering reviewer questions

Do not argue with the reviewer. Answer the question. If an honest answer is "I didn't check", give that answer — the reviewer's next move will be to deny with feedback naming what to check, and that feedback is exactly what you wanted.

If a reviewer asks a question that is wrong (misreads the draft, asks about a term that is actually defined one paragraph up), quote the passage back to correct them. Do not capitulate to a wrong question.

## Cost and scope

Each invocation spawns up to four subagents running up to two rounds. Budget accordingly. If a draft is clearly low-stakes, do not invoke; the triggers above are deliberately scoped to high-stakes moments.
