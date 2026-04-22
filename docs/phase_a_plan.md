# Phase A master plan

Phase A moves the WALL stage's computation from the prefill into
autoregressive "thinking tokens." The architecture is token-centric:
one discrete ID per step, embedding-matrix lookup, dot-product
unembed, explicit bypass channels for non-pure I/O. Five parts
deliver this incrementally while keeping M4's HIT_FULL/HIT_X/HIT_Y
behavior intact throughout.

This plan is the authoritative source for representation and
execution details. `docs/design_byte_token_renderer_phase_a.md`
describes the post-Phase-A architecture at the token-sequence level
and remains correct as a sequence reference; where the design doc
and this plan disagree on representation (e.g., "thinking_value" as
an overlaid field vs. the embedding's payload bits), **this plan
supersedes**.

## Cross-cutting architectural decisions

### Token architecture

- Every autoregressive step consumes one token ID and emits one
  token ID.
- Embedding: matrix lookup via `torchwright.graph.Embedding` (the
  existing class, generalized to accept arbitrary `d_embed` and a
  caller-supplied table).
- Deembed: dot-product LM head. `logits = slice @ W_unembed.T`;
  argmax picks the next ID. Tied weights (`W_unembed = W_embed`).
- No overlaid "value" fields alongside token IDs. Values live inside
  the embedding's payload columns.

### Embedding layout (`d_embed = 72`)

```
cols [0:8]   = E8 category code (distinct per category)
cols [8:24]  = one_hot(h3) — bits 12..15 of VALUE ID (zero for non-VALUE)
cols [24:40] = one_hot(h2) — bits 8..11
cols [40:56] = one_hot(h1) — bits 4..7
cols [56:72] = one_hot(h0) — bits 0..3
```

A 4+4+4+4 factored one-hot for the 16-bit VALUE payload. Sharp
argmax at deembed (four independent 16-way argmaxes). Sequential
thermometer extraction at emit (~4 sublayers).

### Vocabulary ID ranges

| Range | Category | Count |
|---|---|---|
| 0–65535 | VALUE (quantized 16-bit integers) | 65,536 |
| 65536–65543 | THINKING_WALL markers 0..7 | 8 |
| 65544–65556 | Per-wall identifiers (BSP_RANK, IS_RENDERABLE, CROSS_A, DOT_A, CROSS_B, DOT_B, T_LO, T_HI, VIS_LO, VIS_HI, HIT_FULL, HIT_X, HIT_Y) | 13 |
| 65557–65559 | RESOLVED identifiers (RESOLVED_X, RESOLVED_Y, RESOLVED_ANGLE) | 3 |
| 65560–65562 | Decode tokens (SORTED_WALL, RENDER, DONE) | 3 |
| 65563–65570 | Prompt-position categories (INPUT, BSP_NODE, WALL, EOS, TEX_COL, PLAYER_X, PLAYER_Y, PLAYER_ANGLE) | 8 |

Total V ≈ 65,571. The shared `E8_VALUE` category code in cols [0:8]
is reused across all 65,536 VALUE IDs regardless of context (wall
input values, thinking payloads, future RENDER payloads — same
category, different position).

### Bypass convention

Non-pure I/O is explicit. Each bypass is a named leaf node in the
residual stream with its own column allocation, sibling to the
`Embedding` node. No graph-level "join" between embeddings and
bypasses — they coexist in the residual, ops read whichever
leaf they need.

**Prompt bypasses** — host supplies per-position at prefill time:
texture data (`tex_pixels`, `texture_id_e8`, `tex_col_input`), wall
geometry (`wall_ax/ay/bx/by`, `wall_tex_id`, `wall_index`,
`wall_bsp_coeffs`, `wall_bsp_const`), BSP-plane coefficients, INPUT
controls, PLAYER_X/Y/ANGLE values.

**Autoregressive bypasses (overlaid)** — host copies from output
to next-input via delta transfer: `render_col`, `render_chunk_k`,
`wall_counter`.

**Autoregressive bypass outputs (overflow)** — host reads for
framebuffer / state update: `pixels`, `col`, `start`, `length`,
`done`, `advance_wall`, `sort_done`, `sort_vis_hi`,
`sort_wall_index`.

### Thinking-value readback

Cross-step reads of thinking-token values go through one helper:

```python
readback = build_thinking_readback(
    embedding=..., prev_id_slots=...,
    is_value_category=..., pos_encoding=...,
)
value_float = readback.get_value_after_last("CROSS_A")
```

Internally: build `is_X_value` flag per position, `attend_most_recent_matching`
on that flag, decode the 4+4+4+4 factored embedding to an integer
ID, dequantize via the identifier's design-doc range
(`VALUE_RANGE_BY_IDX[IDX_X]`).

Consumers read thinking-token values at layer 0 of their own step —
the value is already in the KV cache's embedding columns, no deep
computation waiting. Shallow Linears do the dequant.

### Computation happens at the identifier step

Under the one-ID-per-step model, the computation of a value happens
at the *identifier* step (input = X_ID, output = the quantized
value), not at the VALUE step as M4 had it. The VALUE step that
follows reads "which identifier preceded me" via prev-id attention
and emits the next identifier.

Wire-format token sequence is unchanged from M4 (`[marker][ID][VALUE][ID][VALUE]…`);
semantics shift. M4 code cannot port 1:1; the computation moves one
step forward.

### Detector patterns

Two flavors:

- **Specific-ID detector** — full 72-wide embedding comparison
  against a known row. Used for identifiers, markers, decode tokens,
  prompt categories.
- **Category-only detector** — compare just the first 8 columns
  (category code) against `E8_VALUE`. Used for "is this a VALUE
  position" independent of which specific VALUE.

Both use `equals_vector`; the category-only variant wraps with an
`extract_from(embedding, 72, 0, 8, ...)` to restrict the comparison.

### 16-wide prev-id storage

Every identifier step (13 per-wall + 3 RESOLVED = 16) stores its
slot one-hot in its residual at a consistent 16-wide location. The
prev-id attention at VALUE steps reads this back as a 16-wide one-hot
via `attend_most_recent_matching(key=is_any_identifier, value=onehot_slot)`.

Match-gain tuning (M4's 12000 should carry, unverified at 16-wide)
is an open question resolved during Part 2 implementation.

## Parts index

### Part 1 — Token architecture foundation
`docs/phase_a_part1_embedding.md`

Generalizes `Embedding`, builds DOOM's vocabulary + `W_embed`,
introduces bypass conventions, swaps DOOM's compile path to feed
token IDs + bypasses. M4's HIT_FULL/HIT_X/HIT_Y continues to work
end-to-end on the new carrier. No new DOOM behavior.

**Acceptance:** existing non-DOOM examples pass unchanged; DOOM
walkthrough matches reference; M4 dual-path tests pass; the
`thinking_value` overlaid field is gone from the codebase.

### Part 2 — State machine skeleton at full width
`docs/phase_a_part2_state_machine.md`

Extends the state machine from M4's 3-identifier cascade to the
full 16-identifier vocabulary. 10 new per-wall slots + 3 RESOLVED
slots get detectors and state-machine routing. Their value
computations are stubbed (emit VALUE ID 0 uniformly). Introduces
the semantic shift — computation at identifier step, not VALUE
step.

**Acceptance:** trace test asserts correct token-ID sequence
through one full frame; HIT_* values match M4 reference; prev-id
attention resolves with hardness > 0.99; walkthrough matches Part 1
(stubs don't cascade into SORTED/RENDER).

### Part 3 — Per-value math
`docs/phase_a_part3_values.md`

Fills in the 10 stub slots with real math. Base values (BSP_RANK,
IS_RENDERABLE, CROSS/DOT, HIT_*) compute from first principles.
Derived values (T_LO, T_HI, VIS_LO, VIS_HI) attend at layer 0 to
their upstream thinking values via the new `thinking_readback`
helper. Introduces and tests the helper.

**Acceptance:** 10 dual-path tests pass within accumulated
quantization tolerance; 5 helper unit tests pass; walkthrough still
matches (values have no downstream consumers in Phase A); depth
per identifier step ≤ 15.

### Part 4 — RESOLVED + EOS gutting + RENDER migration
`docs/phase_a_part4_resolved.md`

First part where rendered frame depends on thinking values. Three
interlocked changes, one atomic commit: RESOLVED math (aggregates
collision from prefill WALL), EOS computation deleted (marker
retained), RENDER reads post-collision player position from RESOLVED
via the helper. PLAYER broadcast flips to pre-collision; the
pre-collision player bypass is removed; host no longer injects
THINKING_WALL_0 (PLAYER_ANGLE step emits it).

**Acceptance:** RESOLVED dual-path tests pass; collision-scenario
walkthrough matches reference; EOS collision math is gone; host
reads resolved state from argmax output at known step offsets.

### Part 5 — Validation & carryover
`docs/phase_a_part5_validation.md`

Closure. Diagnoses and fixes the M4 regressions (SORTED softmax
dilution, affine-bound looseness). Runs `make measure-noise` per
CLAUDE.md. Records compiled depth without chasing a target.
Enumerates what Phase A explicitly does not deliver.

**Acceptance:** M4 regressions resolved with regression tests; noise
docs in sync; full `make test` + walkthroughs pass; Phase A
completion summary written.

## Dependencies between parts

```
Part 1 ──▶ Part 2 ──▶ Part 3 ──▶ Part 4
                                    │
                                    ▼
                                  Part 5
```

Parts 2–4 are strictly sequential. Part 5's validation is
*continuous* across Parts 2–4 in practice (measurement tools run as
each part lands), even though formal acceptance happens at the end.

## Cross-cutting budgets

### Compiled depth per identifier step

Target: ≤ 15 layers per identifier step (matches RENDER precompute
budget). Estimated per-step depths:

| Step | Estimated depth |
|---|---|
| Marker | ~1 |
| Base-value IDs (BSP_RANK, IS_RENDERABLE, CROSS/DOT) | 6–8 |
| HIT_* IDs | ~8 |
| Derived IDs (T_LO/T_HI) | 11–12 |
| Derived IDs (VIS_LO/VIS_HI) | 13–14 |
| RESOLVED_X/Y IDs | ~9 |
| RESOLVED_ANGLE ID | ~5 |
| VALUE steps (just next-ID emission) | ~3 |

Measure via `make graph-stats`; flag any step exceeding 15.

### Residual width (rough)

`d_model = 2048`. Token embedding takes 72 columns (3.5%). Bypasses
are substantial: wall BSP coeffs alone are 48 columns, tex_pixels is
`tex_h * 3` (~24 for 8-wide tex), plus overlaid bypasses and
overflow. Total non-graph-intermediate residual consumption is
perhaps 150–200 columns — well under 10% of d_model. Graph
intermediates consume the rest.

### Quantization error

Single-boundary quantization error per value is bounded by its LSB
(design-doc table). Compound through the derived chain (CROSS_A →
T_LO → VIS_LO): ~3 LSBs through VIS. Needs empirical validation in
Part 3. Mitigation if tight: widen T_LO/T_HI to 32-bit.

## What Phase A explicitly does not deliver

Enumerated in one place so scope is unambiguous:

- **Prefill WALL stage stays.** Its deletion is post-Phase-A.
- **Compiled depth is not reduced.** Phase A ships in roughly the
  same layer count as pre-Phase-A; the design-doc target of ~25–30
  layers requires the prefill WALL deletion.
- **RENDER overlaid state (col, chunk_k, wall_counter) remains.**
  The per-pixel-output phase that eliminates it is post-Phase-A.
- **WALL-as-identifier-value-pairs refactor is not done.** WALL is
  a single-token-per-wall prompt encoding through Phase A.
- **"Aggregate values across walls" helper is not built.** RESOLVED
  in Phase A uses prefill WALL aggregation. The thinking-token-only
  aggregation path awaits prefill WALL deletion.
- **Seven thinking-token values have no Phase-A consumer.**
  BSP_RANK, IS_RENDERABLE, VIS_LO, VIS_HI, HIT_FULL, HIT_X, HIT_Y
  are computed and validated via dual-path tests but never read by
  SORTED or RENDER in Phase A. They're correct and ready for the
  future phase that deletes prefill WALL.

## Open questions spanning the plan

### Design-doc update

Whether to update `docs/design_byte_token_renderer_phase_a.md`'s
"thinking value" language to reflect the embedding-matrix
representation. Current position: leave the design doc as-is, this
plan supersedes on representation. Revisit if it becomes a source
of confusion.

### Compile-path extension mechanism

Part 1's open question — extending `compile_headless` to recognize
`Embedding` as another leaf source vs. using the existing
token-transformer export path. Decide during Part 1 implementation;
either works.

### Helper API shape

Part 3's open question — the `thinking_readback` module's concrete
API (registry-object vs. free-function). Decide during Part 3
implementation.

### SORTED dilution fix mechanism

Part 5's open question — whether the fix is match_gain tuning,
position masking, or exhaustion-signal redesign. Decide after
root-cause diagnosis.

### Trace field renaming

Whether to rename `FrameTrace.eos_resolved_x/y/angle` to
`resolved_x/y/angle` in Part 4. Cosmetic; decide based on scope
pressure.

## Writing the parts — status

All five part docs are drafted:

- `docs/phase_a_part1_embedding.md`
- `docs/phase_a_part2_state_machine.md`
- `docs/phase_a_part3_values.md`
- `docs/phase_a_part4_resolved.md`
- `docs/phase_a_part5_validation.md`

Each has its own strong opinions, scope, acceptance criteria, open
questions, and high-level task list. This master plan indexes them
and captures the architectural decisions that span multiple parts.
