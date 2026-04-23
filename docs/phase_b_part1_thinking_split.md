# Phase B, Part 1 — Thinking-slot split, running accumulators, factor cascade

Three reductions to the thinking-token pipeline that together
shallow the compiled graph's critical path. Self-contained inside
`stages/thinking_wall.py` and `thinking_readback.py`; no external
consumers change in this part.

## Prerequisites

- `main` at commit `bf6ce48` or later (Phase A Part 4).
- The current thinking state machine carries 16 identifier slots
  (`BSP_RANK`, `IS_RENDERABLE`, `CROSS_A`, `DOT_A`, `CROSS_B`,
  `DOT_B`, `T_LO`, `T_HI`, `VIS_LO`, `VIS_HI`, `HIT_FULL`,
  `HIT_X`, `HIT_Y`, `RESOLVED_X`, `RESOLVED_Y`, `RESOLVED_ANGLE`),
  emits each value as a 72-wide VALUE embedding, and provides the
  Part 3 readback helper (`thinking_readback.get_value_after_last`).
- No dependency on Part 2. Parts 1 and 2 can land in either order
  or concurrently.

## Development speed

Skip: expanded dual-path tests for new slots, formal validation
ceremony, noise re-measurement (Part 4 owns that). The existing
dual-path tests for `T_LO`, `T_HI`, `VIS_LO`, `VIS_HI`, `HIT_*`
continue to apply after the change — if they still pass, the split
is correct. Smoke test = `make walkthrough` matches reference.

## What's landing

Three changes, all inside the thinking-token emit pipeline. Each is
independently useful; they ship together because they all touch
`thinking_wall.py` and share validation infrastructure.

### 1. Four new intermediate thinking slots

Today every thinking position's computation graph contains the full
`T_LO` / `T_HI` / `VIS_LO` / `VIS_HI` math:

- Rotate the wall's endpoints into the player's frame
  (already done at `CROSS_A`, `DOT_A`, `CROSS_B`, `DOT_B` steps).
- Per-plane FOV clip: compute `t_star = f_a / (f_a - f_b)` via
  `reciprocal + multiply_2d`, one per plane (left, right). ~8 ops
  serial each.
- Aggregate: `t_lo = max(0, t_lo_L, t_lo_R)`, `t_hi = min(1, ...)`.
- Project each clipped endpoint to a screen column via
  `low_rank_2d` (rank-3 SVD approximation of `atan`). ~12 ops
  serial.
- Column-fold selects for clipped-vs-interior column choice.
- Gate by `is_renderable`.

Combined per-step critical path: ~39 ops. Plus the per-position
select cascade (~3 ops) and the 72-wide VALUE factor cascade (~19
ops). Total ~58 ops per thinking step.

The graph is the same at every thinking position (one graph fits
all positions); every thinking step pays that depth, even at
positions whose own emitted value is shallow (e.g. `BSP_RANK` at
~5 ops of actual math).

**Part 1 splits the deep portion into four new thinking slots.**
Each new slot is an identifier + VALUE pair, emitted at its own
autoregressive step, reading upstream values from the KV cache via
`attend_most_recent_matching` (the Part 3 readback primitive).

| New slot | Computes | Reads from KV | Own depth |
|----------|----------|---------------|-----------|
| `T_STAR_L` | Left-plane clip contrib: `reciprocal + multiply_2d` → `t_star_L` | `CROSS_A`, `DOT_A`, `CROSS_B`, `DOT_B` | ~10 ops |
| `T_STAR_R` | Right-plane clip contrib | Same | ~10 ops |
| `COL_A` | Endpoint-A projected column via `low_rank_2d` atan + column-fold select | `CROSS_A`, `DOT_A` | ~12 ops |
| `COL_B` | Endpoint-B projected column | `CROSS_B`, `DOT_B` | ~12 ops |

The existing four slots become shallow consumers of the new ones:

| Existing slot | New computation | Own depth |
|---------------|-----------------|-----------|
| `T_LO` | Read `T_STAR_L`, `T_STAR_R`; `max` with 0 | ~4 ops |
| `T_HI` | Read `T_STAR_L`, `T_STAR_R`; `min` with 1 | ~4 ops |
| `VIS_LO` | Read `COL_A`, `COL_B`; `min`; gate by `is_renderable` + `is_empty` | ~5 ops |
| `VIS_HI` | Read `COL_A`, `COL_B`; `max`; gate | ~5 ops |

The deepest slot after the split is `COL_A` or `COL_B` at ~12 ops.
Since every thinking position still contains the graph for all
slots, the per-position critical path drops from ~58 ops
(`t_and_vis` 39 + select 3 + factor 19) to ~34 ops (`COL` 12 +
select 3 + factor 19).

The factor cascade restructure below drops the 19 further.

**Next-identifier ordering.** The wall's thinking-token sequence
becomes:

```
BSP_RANK → IS_RENDERABLE → CROSS_A → DOT_A → CROSS_B → DOT_B →
T_STAR_L → T_STAR_R → T_LO → T_HI → COL_A → COL_B →
VIS_LO → VIS_HI → HIT_FULL → HIT_X → HIT_Y → (wall transition)
```

17 identifiers per wall instead of 13. With 8 walls, that's 64
extra thinking tokens per frame (current ~222 → ~286). Per CLAUDE.md
the depth win is well worth the ~30% token-count growth.

**Prev-id storage widens.** Currently 16-wide (to tag which of the
16 identifiers preceded the current VALUE). Grows to 20-wide
(17 per-wall + 3 RESOLVED). M4 tuned the prev-id `attend_most_recent_matching`
match-gain at 12000 for 16-wide; 20-wide may need re-tuning. Verify
softmax concentration stays > 0.99 during implementation.

### 2. Running accumulators for HIT_FULL, HIT_X, HIT_Y

Today each HIT_FULL thinking token emits its own wall's collision
flag (0 or 1). RESOLVED aggregates across 8 walls via
`attend_mean_where(is_wall, ...)` over prefill WALL positions.

Part 1 changes each HIT_FULL thinking token to emit not
`hit_full_W` but `hit_full_W OR running_or_through_previous_walls`.
Mechanism: read the previous HIT_FULL value via
`attend_most_recent_matching(is_HIT_FULL_value)` (Part 3 primitive,
already wired), combine via a Linear + saturate to implement OR
over 0/1 inputs, emit the result.

By the time wall 7's HIT_FULL thinking token fires, its emitted
value is the global OR across all 8 walls. Same structure for
HIT_X and HIT_Y.

**Why this matters.** Part 3 (not Part 1) uses the accumulator:
RESOLVED reads `get_value_after_last("HIT_FULL")` and gets the
global aggregate as a single scalar. No cross-position aggregation
at RESOLVED time. But the emit-side change belongs in Part 1
because it's a narrow thinking-token emit change; bundling with the
slot split keeps consumer changes isolated to Part 3.

**Initial condition.** Wall 0's HIT_FULL has no previous value.
`attend_most_recent_matching` returns zero when nothing matches;
zero is the OR identity. The first wall's HIT_FULL emits
`hit_full_0 OR 0 = hit_full_0`. Correct.

**Numerical sanity.** HIT values are near-integer 0 or 1 (from
`bool_to_01`). The running OR is a linear combination
`saturate(a + b)` that preserves 0/1 semantics. Eight applications
through the chain stay within 0/1 bounds because saturation clamps
at 1. No noise accumulation that would drift across the 0.5
quantize boundary.

No consumer uses the accumulator in Part 1. The HIT_* dual-path
tests (which verify each wall's individual flag) continue to pass
because wall 0's accumulator equals the individual flag; later
walls' accumulators equal `flag_W OR prior_accumulator`. If the
dual-path test inspects only wall 0's HIT value, it's unchanged; if
it inspects wall 7's HIT value, the test needs to expect the global
OR rather than wall 7's individual flag. Adjust the test
expectation if needed.

### 3. factor_q_to_embedding — hi/lo split for parallel digit extraction

Every thinking-token VALUE emit goes through `factor_q_to_embedding`
to convert a 16-bit integer `q ∈ [0, 65535]` into a 4+4+4+4
factored one-hot on the 72-wide embedding. Today that's four
hex-digit extractions done serially:

```
h3 = q  // 4096          ;  r3 = q  - h3 * 4096
h2 = r3 // 256           ;  r2 = r3 - h2 * 256
h1 = r2 // 16            ;  h0 = r2 - h1 * 16
```

Each `//` is a `thermometer_floor_div` — a deep primitive. Three
serial `thermometer_floor_div` calls are the bulk of the factor
cascade's 19-op depth.

Restructure to split `q` into hi/lo bytes once, then extract digits
from each byte in parallel:

```
q_hi = q    // 256       # thermometer_floor_div, serial dep
q_lo = q    - q_hi * 256
h3   = q_hi // 16 ; h2 = q_hi - h3 * 16    ┐
                                            ├── parallel
h1   = q_lo // 16 ; h0 = q_lo - h1 * 16    ┘
```

Two serial `thermometer_floor_div` calls instead of three. The
factor cascade's per-step depth drops from ~19 ops to ~15. Saved
across every thinking-token step.

Both the old and new structures produce the same 4+4+4+4 factored
one-hot. Downstream (the host's argmax-against-W_embed) sees
identical output.

## Scope

- Add `T_STAR_L`, `T_STAR_R`, `COL_A`, `COL_B` to the identifier
  vocabulary and prev-id storage (16-wide → 20-wide).
- Update the `NEXT_IDENTIFIER` table in
  `stages/thinking_wall.py` to route through the new slots in the
  ordering above.
- Refactor `_compute_clip_and_project` in `stages/thinking_wall.py`:
  split into four intermediate emit steps (`T_STAR_L`, `T_STAR_R`,
  `COL_A`, `COL_B`), each emitting its own scalar value; update
  `T_LO`, `T_HI`, `VIS_LO`, `VIS_HI` to read from KV via
  `readback.get_value_after_last`.
- Extend `VALUE_RANGE_BY_NAME` with ranges for the new slots.
  `T_STAR_L` / `T_STAR_R` range is ~`[-2, 2]` (clamp targets under
  clip edge cases). `COL_A` / `COL_B` range is ~`[-2, W+2]` (same
  sentinel as `VIS_LO` / `VIS_HI`).
- Modify `HIT_FULL`, `HIT_X`, `HIT_Y` emit logic: read previous
  same-type value, OR with locally-computed flag, emit the
  aggregate.
- Restructure `factor_q_to_embedding` in
  `torchwright/doom/thinking_readback.py`: split `q` into hi/lo
  bytes, extract digits in parallel.
- Verify prev-id match-gain at 20-wide. If softmax concentration
  falls below 0.99, increase match-gain.

## Not in scope

- Any changes to SORTED, RENDER, or RESOLVED. They still read from
  their Phase A sources.
- Deleting prefill WALL's computation (Part 3 owns that).
- Wiring HIT_* accumulator consumers (Part 3 wires RESOLVED).
- Wiring SORTED to read from BSP_RANK thinking tokens (Part 2).
- Further shallowing the factor cascade (e.g., 4-way parallel
  decomposition) — Phase C territory.
- Skipping T/VIS computation at non-T/VIS positions (requires
  compiler-level position-conditional elaboration; Phase C).
- Adding consumers for the 7 thinking values that still have none
  after Phase A (`BSP_RANK`, `IS_RENDERABLE`, `VIS_LO`, `VIS_HI`,
  `HIT_FULL`, `HIT_X`, `HIT_Y`). Part 2 and Part 3 wire these.

## Smoke test

`make walkthrough` produces a reference-matching frame. Existing
dual-path tests for `T_LO`, `T_HI`, `VIS_LO`, `VIS_HI`, `HIT_*`
continue to pass (their emitted values are now computed from KV
readbacks but end-to-end equivalence holds). New `T_STAR_L/R` and
`COL_A/B` values can be spot-checked against hand-computed expected
values for one canonical scene, but no dedicated dual-path harness
is required.

## Interaction with Part 2

Both parts edit `stages/thinking_wall.py`:

- Part 1 adds slot computations (new emit steps for `T_STAR_L/R`,
  `COL_A/B`) and restructures the factor cascade.
- Part 2 adds scalar channels to the `BSP_RANK` thinking token's
  emit (for SORTED's quadratic-equality attention).

The edits touch different functions and different identifier slots;
merge conflicts unlikely. Neither part depends on the other.

## Open questions

- **Prev-id match-gain at 20-wide.** Start with the current 12000;
  bump if softmax concentration falls below 0.99 in the walkthrough.
- **Quantization range for T_STAR_L/R.** Nominal `[-1, 1]` under
  clean geometry; wider under clipping edge cases. Pick the range
  that keeps the existing T_LO/T_HI dual-path tests passing.

## High-level task list

1. Extend identifier vocabulary and prev-id storage: add
   `T_STAR_L`, `T_STAR_R`, `COL_A`, `COL_B`; widen to 20-wide;
   update `NEXT_IDENTIFIER` routing.
2. Refactor `_compute_clip_and_project`: per-plane clip contribs
   and endpoint-to-column emitted at their own slots.
3. Shallow `T_LO`, `T_HI`, `VIS_LO`, `VIS_HI` to read upstream
   intermediates from KV.
4. Add running-OR emit logic for `HIT_FULL`, `HIT_X`, `HIT_Y`.
5. Restructure `factor_q_to_embedding` with hi/lo split.
6. Adjust `VALUE_RANGE_BY_NAME` entries for new slots.
7. Run existing dual-path tests for T/VIS/HIT values; adjust
   test expectations for running accumulators on HIT_* if needed.
8. `make walkthrough` — reference match.
