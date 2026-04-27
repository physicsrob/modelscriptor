# DOOM Sandbox — Platform Overview

This document is the platform-side counterpart to
`doom_sandbox/CLAUDE.md`. It explains what the sandbox is, why it
exists, how it relates to `torchwright/doom/`, and how the two-sided
maintenance model works. The sandbox-resident agent does not see this
file. You (the platform maintainer) do.

## What the sandbox is

A constrained Python environment for designing DOOM-rendering
algorithms. The agent working inside writes per-position `forward()`
functions, runs them through an autoregressive loop in pure Python,
and verifies output against pure-Python reference computations. No
GPU, no compilation, no transformer.

The constraints inside the sandbox are designed to mirror what the
real transformer enforces: cross-position state flows only through
attention-shaped primitives, all compute is piecewise-linear, the
discrete deembed/reembed bottleneck is enforced at every position.
Designs that work in the sandbox should port to `torchwright/doom/`
without surprise.

## Why it exists

The real graph is hard to iterate in. Compile cycles, GPU costs,
flaky tests, plus a large surface area (compiler invariants, PWL
noise budgets, residual-column scheduling) that the renderer-design
question shouldn't have to grapple with. Sessions with the real graph
routinely run to ~500k tokens of context.

The sandbox isolates the *design* problem from those concerns. The
sandbox-resident agent's complete context is `doom_sandbox/CLAUDE.md`
plus the phase they're working on. The cognitive surface is small
enough to keep ambitious design work tractable.

## Two-sided maintenance

| Side | Audience | What they see | What they own |
|------|----------|---------------|---------------|
| **Sandbox** | the resident agent | `doom_sandbox/CLAUDE.md` + their phase's directory | per-phase `setup.py` / `prefill.py` / `forward.py` / `extract.py` / `reference.py` / tests |
| **Platform** | us | `docs/sandbox/*.md` + the framework runtime + this overview | `doom_sandbox/api/` (the API surface), `doom_sandbox/runtime/` (framework internals), `doom_sandbox/types/` (pydantic schemas), `doom_sandbox/fixtures/` (JSON-serialized scenes), framework tests colocated with the code they test (e.g. `doom_sandbox/api/test_*.py`), `docs/sandbox/` (these docs) |

The sandbox-resident agent treats the platform side as a black box.
We treat the sandbox side as a contract — we don't read the agent's
phase code while building the framework, but we make sure their
contract holds.

## Layout

```
doom_sandbox/
├── CLAUDE.md                       # sandbox agent's complete context
├── api/                            # agent-facing API
├── types/                          # pydantic schemas (MapSubset, GameState, ...)
├── runtime/                        # framework internals (agent does NOT import)
├── fixtures/                       # serialized MapSubsets
└── phases/
    └── <phase_name>/
        ├── PHASE.md
        ├── setup.py
        ├── prefill.py
        ├── forward.py
        ├── extract.py
        ├── reference.py
        └── test_<phase>.py

docs/sandbox/                       # platform docs (this directory)
scripts/build_sandbox_fixtures.py   # platform-side fixture generator
                                    # (framework tests live colocated as
                                    #  doom_sandbox/<area>/test_*.py)
```

## Phase lifecycle

1. **Write `PHASE.md`** — define what the phase computes, the test
   contract, and any new vocab the phase needs.
2. **Write `reference.py`** — pure-Python ground truth for the phase's
   output. No framework dependencies.
3. **Sandbox agent implements** `setup.py`, `prefill.py`, `forward.py`,
   `extract.py`, `test_<phase>.py`. The test asserts
   `extract_<thing>(run(config, prefill, forward)) == expected_<thing>(...)`.
4. **Once green, port to `torchwright/doom/`.** The translation table
   (`docs/sandbox/translation_table.md`) maps each sandbox primitive
   to its real-graph counterpart.

## Maintenance discipline

These rules keep the sandbox honest as it grows:

1. **Translation-table sync.** Adding or changing a sandbox primitive
   in `doom_sandbox/api/` updates `docs/sandbox/translation_table.md`
   in the same commit. Otherwise the table goes stale and porting
   becomes guesswork.

2. **Schema round-trip tests.** Changes to types in
   `doom_sandbox/types/` are accompanied by a project-side test that
   serializes a real `MapSubset` from `torchwright/doom/` and
   deserializes it through the sandbox schema. Catches drift between
   the project's source-of-truth class and the sandbox's mirror.

3. **Per-phase isolation is real.** When adding a new phase, do not
   import from a previous phase's `setup.py` / `forward.py`. The
   sandbox-resident agent is told phases are independent
   re-implementations; the platform side must respect that — the
   per-phase test runner doesn't auto-add other phases to the path.

4. **CLAUDE.md changes go through review.** The sandbox agent's
   complete context is one file. Changes there are load-bearing for
   every phase that follows. Review like API changes — small, careful
   commits, ideally with a brief rationale.

5. **Negative tests for invariants.** The framework enforces several
   invariants — `pwl_def` construction frozen during `forward()`,
   `publish` only inside forward, names starting with `input.` are
   reserved, etc. Each invariant has a negative test colocated with
   the relevant module (e.g. `doom_sandbox/api/test_run.py` for the
   forward-lifecycle invariants) that asserts the framework raises
   when violated. Adding an invariant without its negative test is
   forbidden.

## Where to start, when

| You want to | Read | Edit |
|-------------|------|------|
| Understand the sandbox | this file + `doom_sandbox/CLAUDE.md` | nothing |
| Add or change a primitive | `docs/sandbox/translation_table.md` | `doom_sandbox/api/`, the table, the colocated `test_*.py` next to the changed file |
| Update the type schemas | `torchwright/doom/map_subset.py` (source of truth) | `doom_sandbox/types/`, project-side round-trip test |
| Implement framework runtime | API stubs in `doom_sandbox/api/`, this file | `doom_sandbox/runtime/`, the colocated `test_*.py` files |
| Spec a new phase | existing `PHASE.md` files for tone/shape | new `doom_sandbox/phases/<name>/PHASE.md` + `reference.py` |
