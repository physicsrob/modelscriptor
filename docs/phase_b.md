# Phase B

Phase B closes the gap between Phase A's current state and a
depth-reduced autoregressive rendering pipeline. The thinking-token
architecture is already landed; Phase B restructures how SORTED and
RESOLVED consume thinking-token values, deletes the prefill WALL
stage's computation, and reduces the compiled critical path inside
each thinking step.

## Goals

- **Reduce compiled depth.** Today's graph is 115 layers, driven by
  the internal chain of the deepest thinking step (~58 ops). Phase B
  targets a meaningful drop. Exact number measured on landing.
- **Eliminate overlay feedback for wall identity.** Move wall_index
  from overlay to an autoregressive VALUE token emitted by SORTED.
- **Gut prefill WALL's computation.** After Phase B, the WALL
  prefill stage carries raw geometry only — no collision detection,
  no visibility columns, no bsp_rank.

## Parts

| Part | Topic | Parallel with | Depends on |
|------|-------|---------------|------------|
| [Part 1](phase_b_part1_thinking_split.md) | Thinking-slot split, running accumulators, factor cascade restructure | Part 2 | — |
| [Part 2](phase_b_part2_sorted_render.md) | SORTED via quadratic-equality attention, SORTED emits wall_index as VALUE, RENDER reads via attention | Part 1 | — |
| [Part 3](phase_b_part3_resolved_and_wall_gut.md) | RESOLVED reads running accumulators, prefill WALL gutted to data carrier | — | Parts 1 + 2 |
| [Part 4](phase_b_part4_validation.md) | Walkthrough reference match, noise pipeline, depth measurement, brief completion note | — | Part 3 |

Dependency graph:

```
   Part 1 ──┐
            ├──> Part 3 ──> Part 4
   Part 2 ──┘
```

Parts 1 and 2 run concurrently. Part 3 merges both. Part 4 is a
lightweight validation tail.

## Development speed

Phase B optimizes for development speed. Each part ships with
minimal test coverage beyond the existing suite. Smoke test =
walkthrough matches reference. No dual-path test expansion, no
formal completion summaries, no exhaustive regression analysis.
Each part's doc has a "Development speed" section stating what is
specifically skipped.

## Explicitly out of scope for Phase B

- WALL-as-identifier-value-pairs prefill refactor (Phase C).
- Per-pixel RENDER output / eliminating chunked RENDER (Phase C).
- 16-bit vocabulary unification (eliminate E8 codes) (Phase C).
- Ceiling/floor pixel decision in transformer (Phase C).
- Further shallowing the `factor_q_to_embedding` cascade beyond
  Part 1's hi/lo split (Phase C).
- Reduction of RENDER's per-token compute chain (Phase C).

## Reading each part cold

Each part doc is self-contained: prerequisites are stated, shared
concepts (running accumulator, quadratic-equality attention, content
attention by wall_index) are re-explained where they appear. A
reader who lands on any one part doc does not need to read the
others to understand what the part changes.
