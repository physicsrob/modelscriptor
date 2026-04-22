# Phase A, Part 2 — State machine skeleton at full width

With the embedding carrier in place (Part 1), this part expands the
thinking-phase state machine from M4's 3-identifier cascade to the
full 16-identifier vocabulary. No new value math is computed in this
part: the 10 new per-wall slots and the 3 RESOLVED slots are stubs
that emit placeholder VALUE IDs. HIT_FULL/HIT_X/HIT_Y continue to
produce real values.

The goal is a structurally correct transformer that walks the full
token sequence end to end — every identifier is emitted in the right
order, wall rollover fires correctly, the RESOLVED chain routes to
SORTED_WALL — even while most of the numbers it emits are
placeholders.

## Strong opinions this part commits to

### The computation step is the identifier step, not the value step

This is a semantic shift from M4. In M4, the computation (hit_full,
etc.) ran at the VALUE step (input token = THINKING_VALUE), and the
separate `thinking_value` overlaid field carried the number.

Under the embedding architecture, each step emits exactly one ID.
At an identifier step (input = `HIT_FULL_ID`, `CROSS_A_ID`, etc.), the
graph computes the value and emits the VALUE ID encoding it. At the
following step (input = that VALUE), the graph emits the next
identifier.

The wire-format token sequence is unchanged from M4:
`[marker][IDENTIFIER][VALUE][IDENTIFIER][VALUE]…`. What shifts is
which step is doing the arithmetic. Implementation must consciously
reflect this — a 1:1 port of M4's VALUE-step computation path will
not work in the embedding carrier.

### Two detector patterns: specific-ID vs category-only

Identifier / marker / decode tokens each have a unique 72-wide
embedding row; detection for these uses the usual
`equals_vector(embedding, embed_lookup("X"))` pattern.

VALUE positions have embeddings that vary by ID — detection cannot
compare full width. Instead, compare only the category-code columns:

```
category_code = extract_from(embedding, 72, 0, 8, "category_code")
is_value      = equals_vector(category_code, E8_VALUE)
```

Any detector that "matches a category regardless of payload" uses
this pattern. Any detector that "matches a specific token" uses the
full-embedding comparison.

### Prev-id attention stores a 16-wide slot one-hot at every identifier position

The value step's "which identifier preceded me" lookup reads via
`attend_most_recent_matching` with `key_vector = is_any_identifier`
and `value` = the 16-wide slot one-hot. To make that value available,
every identifier step must store its slot one-hot in its residual —
specifically, a vector where slot `i` = 1 at `IDENTIFIER_i` position,
0 elsewhere.

This is a structural requirement on every identifier position, not
just the 3 HIT_* identifiers M4 handled. Must be wired uniformly
for all 16.

### HIT_FULL / HIT_X / HIT_Y math is preserved, only the host-step shifts

The ray-segment intersection math from M4's `_compute_hit_flags` runs
unchanged. What changes is *where* in the autoregressive loop the
math runs: previously at the VALUE step (input = THINKING_VALUE),
now at the identifier step (input = HIT_FULL_ID etc.). The numerical
results at the corresponding trace-log positions must match M4's
dual-path reference bit for bit.

### Stub slots emit VALUE ID 0 uniformly

All 10 Phase-III-deferred slots (BSP_RANK, IS_RENDERABLE,
CROSS_A/DOT_A/CROSS_B/DOT_B, T_LO/T_HI, VIS_LO/VIS_HI) and all 3
Phase-IV-deferred slots (RESOLVED_X/Y/ANGLE) emit literal VALUE ID
0 from their identifier step. The embedding of VALUE ID 0 is a
well-defined constant (category E8_VALUE + four one-hots all with
slot 0 set).

No per-slot variation in stub values. Predictable, uniform, trivial
for tests to assert.

### RESOLVED identifiers exist as detectors; their computations do not

The detectors `is_resolved_x_id`, `is_resolved_y_id`,
`is_resolved_angle_id` are present in Part 2 so the state machine
can route through them (HIT_Y-of-last-wall → RESOLVED_X_ID → VALUE →
RESOLVED_Y_ID → VALUE → RESOLVED_ANGLE_ID → VALUE → SORTED_WALL).

Their identifier-step computations are stubbed (emit VALUE ID 0).
RESOLVED_X/Y/ANGLE real math lands in Part 4 alongside EOS gutting
and RENDER's migration to read resolved position.

## Scope

- Extend vocabulary with 10 new per-wall identifier IDs plus 3
  RESOLVED identifier IDs (design-doc order: BSP_RANK,
  IS_RENDERABLE, CROSS_A, DOT_A, CROSS_B, DOT_B, T_LO, T_HI,
  VIS_LO, VIS_HI, HIT_FULL, HIT_X, HIT_Y, RESOLVED_X, RESOLVED_Y,
  RESOLVED_ANGLE). The corresponding rows are added to `W_embed`
  as 8-wide unique type codes with all four one-hot sections zero.
- Rebuild `_detect_token_types` to produce `is_identifier_n` (length
  16), `is_any_identifier`, and `is_value` (category-only) alongside
  the existing markers and prompt-position detectors.
- Wire every identifier step's slot one-hot storage for the prev-id
  attention at 16-wide.
- Implement the full 16-way state-machine cascade:
  - Marker step → emits `BSP_RANK_ID`.
  - Identifier step → computes (real for HIT_*, stub for others),
    emits the VALUE ID encoding the value.
  - Value step → prev-id lookup → emits the next identifier (or
    next marker, or RESOLVED_X, or SORTED_WALL).
- Port M4's `_compute_hit_flags` math to run at HIT_FULL_ID /
  HIT_X_ID / HIT_Y_ID identifier steps. Integrate with the 4+4+4+4
  bit-factoring emit path.
- Trace test: walk the full thinking sequence for one frame, assert
  the emitted token ID at every step matches the expected
  sequence (exact for identifiers/markers/decode tokens, range-check
  for VALUE stubs, exact value for HIT_* positions).

## Not in scope

- Any of the 10 new value computations (Part 3).
- RESOLVED_X/Y/ANGLE math (Part 4).
- EOS gutting, RENDER migration to resolved position (Part 4).
- Cleaning up the `THINKING_WALL_0` host-injection hack at the
  prefill→thinking boundary (stays as-is from M4).

## Acceptance criteria

1. Trace test walks through one full frame's thinking sequence (8
   walls × 27 steps + 6 RESOLVED steps) and asserts:
   - Each marker step outputs `BSP_RANK_ID`.
   - Each identifier step outputs a token ID in the VALUE range.
   - Each value step outputs the correct next identifier, next
     marker, RESOLVED_X, or SORTED_WALL per the state machine.
   - The last step before SORTED emits `SORTED_WALL`.
2. HIT_FULL/HIT_X/HIT_Y values at their identifier steps in the
   trace log match M4's pure-Python reference within the
   quantization tolerance (2 / 65535, single LSB).
3. The prev-id attention compiles with
   `assert_hardness_gt(0.99)` and passes.
4. No assertion failures in the compiled forward pass for a single
   box-room frame.
5. `tests/doom/test_thinking_wall.py` (updated for the new step
   offsets and the role-shift) passes its HIT_* assertions.
6. `make walkthrough` renders the same output as Part 1 (stubs
   for the 10 new slots are *not* consumed by SORTED or RENDER —
   SORTED reads bsp_rank / is_renderable from the prefill WALL
   stage, RENDER reads player_x/y from the PLAYER broadcast, so
   Part 2's rendered frame matches Part 1's within existing
   tolerances).
7. `make test` passes every pre-existing test that doesn't
   depend on thinking-token numerical values beyond HIT_*.

## Open questions

These are left for the implementation session to resolve. None
should require revisiting the strong opinions above.

### Match-gain tuning at 16-wide prev-id attention

M4 used `match_gain = 12000` at 3-wide. The same value should work
at 16-wide by the same reasoning (match-swing dominates position
recency), but needs empirical confirmation via the `assert_hardness_gt`
assertion. If it doesn't resolve cleanly, either bump match_gain or
revisit the one-hot storage format (e.g., widen the value swing).

### Depth budget check

HIT_FULL_ID's identifier step runs ray-segment validity (~4 layers)
plus 4+4+4+4 bit-factoring emit (~4 layers) ≈ ~8 layers per step.
Well under the 15-layer RENDER precompute budget. Want to measure
with `make graph-stats` during implementation and flag if it comes
out surprising.

### Stub values do not cascade into SORTED / RENDER

Worth stating explicitly: the 10 stubbed thinking values in Part 2
are not consumed by any downstream stage. SORTED reads bsp_rank
and is_renderable from the prefill WALL stage (not from thinking
tokens); RENDER reads player_x/y from the PLAYER broadcast (not
from RESOLVED thinking tokens — that migration happens in Part 4);
no stage attends to CROSS/DOT/T_LO/T_HI/VIS thinking values. Stubs
emit valid VALUE-range IDs into the KV cache where they sit unread.
The rendered frame in Part 2 should match Part 1's output.

### Placement of the value-computation math

The identifier-step computations for HIT_* (and later for the 10
new values in Part 3) can live in one large `thinking_identifier.py`
file, or be split across multiple stage files
(`thinking_bsp_rank.py`, `thinking_cross_dot.py`, etc.). Either
works — decide during implementation based on code volume and
maintenance preference.

## High-level task list (not exhaustive)

1. Extend `torchwright/doom/embedding.py` with the 13 new identifier
   token IDs and 3 RESOLVED identifier token IDs; add their rows
   to `W_embed`.
2. Add the category-only detector helper (`extract_category(embedding)`
   + `is_value` detector) to whatever file owns token-type detection.
3. Rebuild `_detect_token_types` in `game_graph.py`: 16-slot
   identifier detector list plus the category-only VALUE detector.
4. Rewire `thinking_wall.py`:
   - Move HIT_* math to run at the corresponding identifier steps
     (input = HIT_FULL_ID etc., not input = THINKING_VALUE).
   - At every identifier step, store the slot one-hot for prev-id
     attention.
   - At VALUE steps, run prev-id attention and emit the next
     identifier / marker / RESOLVED_X / SORTED_WALL per the full
     state-machine cascade.
   - Stub the 10 new per-wall and 3 RESOLVED slots to emit
     VALUE ID 0.
5. Update the trace test: walk 8 walls × 27 steps + 6 RESOLVED
   steps + 1 handoff; assert correct ID at each position; assert
   HIT_* values match M4 reference.
6. Run `make test` and the trace test; flag any prev-id hardness
   failures or depth surprises.
