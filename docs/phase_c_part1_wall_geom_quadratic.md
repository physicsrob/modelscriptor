# Phase C, Part 1 — Quadratic-equality wall_geom_attention

Swap RENDER's wall-geometry attention from an 8-wide one-hot match
to a 2-wide quadratic-equality dot product, and consume the
SORT_RESULT VALUE token's raw slot directly (no scalar decode + no
`in_range` one-hot conversion). Eliminates the 19-layer Q-prep
cascade that currently delays `wall_geom_attention` to layer 23.

## Prerequisites

- Phase B Part 2 landed: SORTED emits `wall_index` as a
  `SORT_RESULT` VALUE token, and RENDER consumes it via
  `attend_most_recent_matching(is_SORT_RESULT_value)`.
- Phase B Part 3 landed: VALUE tokens use the 25-wide
  `[E8_VALUE(8) | raw(1) | gray(16)]` embedding; the raw slot
  carries `(2k + 1) / 131072` for `VALUE_k`. The scalar decode
  `value = lo + (raw · 65536 − 0.5) · LSB` is a single affine.
- No dependency on any other Phase C part.

## Development speed

Skip: generic quadratic-attention primitive refactor (SORTED has
its own inline code path; wall_geom gets its own), wall_index
round-trip tests (the end-to-end render match covers it). Smoke
test = `make walkthrough` matches reference after the swap.

## Context: why wall_geom_attention is slow today

Phase B Part 2 added the SORT_RESULT readback:

```
SORTED  → SORT_RESULT id  → VALUE(wall_index)  → RENDER_0  → RENDER_1 …
```

RENDER reads `wall_index` from the most-recent SORT_RESULT VALUE
position via `attend_most_recent_matching`. The readback returns
the VALUE's 1-wide raw slot, decoded through the standard
dequantize affine into a scalar `wall_index` (a near-integer float
in `[0, max_walls − 1]`).

`wall_geom_attention` then needs an 8-wide one-hot query
(`wall_j_onehot`) to match the 8-wide `wall_position_onehot` at
WALL prefill positions. Building that one-hot from the readback
scalar costs:

```
layer 0–13   wall_index_readback  (attention + scalar decode)
layer 14–19  in_range(wall_index, wall_index + 1, max_walls)
                → wall_j_onehot   (5-layer scalar → one-hot)
layer 19–23  wall_geom_attention fires
```

19 layers of Q-prep, then the attention itself. Every downstream
RENDER op that needs wall geometry waits until layer 23+.

## The quadratic-equality swap

Today's wall_geom_attention matches on one-hot equality:

```python
wall_geom = attend_argmax_dot(
    query_vector = cond_gate(is_render, wall_j_onehot),          # 8 wide
    key_vector   = cond_gate(is_wall,   wall_position_onehot),   # 8 wide
    value        = cond_gate(is_wall,
                             Concatenate([ax, ay, bx, by, tex_id])),
    match_gain   = 1000.0,
)
```

The query at RENDER is available at layer 19; the key at WALL is
available at layer 3 (from `_compute_position_onehot`, which runs
`in_range` on host-fed `wall_index`).

Swap to the quadratic-equality form the SORTED stage already uses
(`torchwright/doom/stages/sorted.py` §`sort/quad_attention`):

```
score(wall_k) = −(wall_k − wall_j)²
              = −wall_k²   +   2 · wall_j · wall_k   −   wall_j²

Q at RENDER:  [ 2 · wall_j, 1 ]                                   (2 wide)
K at WALL:    [ wall_k,      −wall_k² ]                           (2 wide)
```

The `−wall_j²` term is query-side constant and drops out of
softmax.  Dot product equals `−(wall_k − wall_j)²`.  Score peaks
at `wall_k == wall_j`; adjacent walls differ by `−1`.  Same match
gain regime as SORTED's quad attention — `match_gain = 20` over a
unit score gap saturates softmax to `≥ 0.999` on the matching
wall.

### Q at RENDER without a scalar decode

Since the SORT_RESULT VALUE's raw slot encodes
`raw = (2 · wall_index + 1) / 131072`, the quadratic Q is a pure
affine on `raw`:

```
wall_j       = raw · 65536 − 0.5
2 · wall_j   = raw · 131072 − 1
```

So the Q-projection W_Q for this attention can be written to read
the raw slot column directly and emit `[raw · 131072 − 1, 1]`.
The `wall_j²` constant is never computed — it falls out of
softmax.

The upstream `attend_most_recent_matching(is_SORT_RESULT_value)`
now returns a 1-wide raw slot (no decode to scalar needed). Its
output lands in a known residual column at RENDER positions at
layer ~6 (after the value-flag construction finishes).

### K at WALL

At WALL prefill positions, host-fed `wall_index` is a
near-integer scalar in `[0, max_walls − 1]` available at layer 0.

- `wall_k` channel: forward `wall_index` unchanged — 0 sublayers.
- `−wall_k²` channel: one `piecewise_linear` (the existing `square_signed` op works, or a trivial 2-breakpoint square for the 0..7 range) — 1 sublayer at layer 1.

Both K channels ready at layer 1–2.

### V unchanged

`value = Concatenate([ax, ay, bx, by, tex_id])` — same 5-wide
block as today, from host-fed WALL inputs available at layer 0.

## Expected depth impact

**Before:**
```
layer 0–13   wall_index_readback (attention hop + Linear decode)
layer 14–19  in_range → wall_j_onehot
layer 19–23  wall_geom_attention
→ wall geometry available at layer 23
```

**After:**
```
layer 0–6    wall_index_readback (attention hop only, no decode)
layer 7      Q-projection affine on raw (folded into W_Q)
layer 7–8    wall_geom_attention (quad-equality)
→ wall geometry available at layer 8
```

Estimated **~15 layers saved** at the wall_geom output point.

Downstream impact (from `docs/phase_c_depth_notes.md`-style
per-layer trace of RENDER):

- `render/precompute` currently spans 14-32. With wall geometry
  ready at layer 8 instead of 23, precompute can start at layer 8
  and finish earlier.
- `render/wall_height`, `render/tex_coord`, `render/tex_attention`
  each cascade earlier by similar amounts.
- `render/column_fill/tex_sample` (25 layers, currently spans
  48-72) could start earlier — but it's bounded by its own internal
  depth and the `tex_attention` it depends on. Expected shift is
  smaller.

Total compiled layer count is currently 75; RENDER ends at layer
73. The non-RENDER critical path ends at layer ~55 (thinking-wall
chain). If RENDER's tail moves from 73 down to ~58, **total
compiled layers could drop to ~58-60** — a ~15-layer reduction.

Actual win measured on landing.

## Critical files

1. `torchwright/doom/stages/render.py`
   - `_attend_wall_geometry`: swap from one-hot `attend_argmax_dot`
     to a 2-wide `attend_argmax_dot` with quadratic Q/K
     construction. K builds `−wall_index²` via `square_signed`.
     Q reads raw slot directly (attention input projection folds
     the affine).
   - `build_render` `render/wall_index_readback` block: narrow
     the readback's return to the 1-wide raw slot; remove the
     scalar decode. The subsequent `render/wall_index_onehot`
     block goes away entirely.
   - Delete `wall_j_onehot` variable and `render/wall_index_onehot`
     annotation.
2. `torchwright/doom/stages/wall.py`
   - `_compute_position_onehot` is no longer consumed by
     `wall_geom_attention`. Check whether it's used anywhere else
     (it's the K for the old attention). If unused, remove it;
     if consumed elsewhere (e.g. SORTED's legacy argmin — but
     Part 2 removed that), delete both the producer and the
     `position_onehot` field on `WallKVOutput`.
   - Add `wall_index_neg_sq` scalar to `WallKVOutput`: one
     `square_signed`-style op on host-fed `wall_index`, emitted as
     a 1-wide KV channel. Available at layer 1-2.
3. `torchwright/doom/game_graph.py`
   - `RenderKVInput.wall_position_onehot` field may become unused
     (see above). Replace with `wall_index_scalar` and
     `wall_index_neg_sq` 1-wide channels.
4. Tests:
   - `tests/doom/test_pipeline.py::TestPipeline::test_frame_matches_reference_*` — existing. Smoke-test is the walkthrough match; cardinal + oblique poses exercise every wall_index.
   - `tests/doom/test_render_graph_precision.py` — check if any test hardcodes the wall_j one-hot shape; update if so.

## Attention hardness sanity check

Quadratic attention over integer walls (score gap = 1, match_gain
= 20) gives:

```
softmax_weight(matching_wall) ≈ e^0 / (e^0 + 2 · e^−20 + 4 · e^−80 + …)
                              ≈ 1 / (1 + ~4e−9)
                              ≈ 1.0 (hardness > 0.999)
```

Compare to SORTED's quad attention at the same match_gain, where
the test suite's `assert_hardness_gt=0.99` passes cleanly.

For max_walls = 8, the farthest wall has score `−49`, which is
negligible after softmax. No TF32 precision concerns — the score
magnitudes are all small integers.

## Out-of-scope follow-ups

Not included in Part 1; mentioned here so future-you knows what
depth-cuts came up in the analysis:

- **`render/tex_attention` quadratic-ification.** The texture
  attention matches on `(tex_id, u_col_index)`. Keeping `tex_id`
  as an 8-wide one-hot is fine (layer-0 prefill), but the
  `u_col_index` side could use a quadratic match instead of an
  `in_range` one-hot conversion. Benefit is smaller since
  `u_col_index` is continuous, not integer.
- **One-hot-in-VALUE-embedding.** For small-cardinality VALUE
  emits (SORT_RESULT 0..7, BSP_RANK 0..7), replacing the scalar
  raw slot with an 8-wide one-hot in the VALUE embedding would
  make the one-hot layer-0-available at the next position's
  embedding leaf, eliminating even the 6-layer readback cost.
  Larger architectural change (touches the vocabulary and
  `emit_integer_value_embedding`); can land as a follow-up.
- **RENDER's internal attention cascade.** Four attentions chain
  inside RENDER: wall_index_readback → wall_geom_attention →
  tex_attention → column_fill. This Part 1 collapses the first two
  ~steps. Parallelizing the remaining hops or shortening
  `tex_coord`/`tex_sample` math is Phase C Part 2+ territory.

## Verification

```bash
# 1. Fast unit-layer smoke: compile the render graph alone.
make test-local FILE=tests/doom/test_render_graph_precision.py

# 2. Full doom test suite on Modal.
make test FILE=tests/doom/

# 3. End-to-end render match — the definitive smoke test.
make walkthrough ARGS="--frames 3"

# 4. Depth measurement — the whole point of the refactor.
make graph-stats

# 5. Noise pipeline. The only new op is the square_signed call at
#    WALL positions; check docs/op_noise_data.json for any
#    unexpected shift.
make measure-noise
```

Expected outcomes:

- All existing doom tests pass.
- Walkthrough matches reference (within existing tolerance — no
  numerical envelope change).
- `make graph-stats` shows `render/wall_geom_attention` span
  shrinking from ~0-23 down to ~0-8, and total compiled layers
  dropping from 75 to ~58-60.
- `make measure-noise` shows no drift on existing ops;
  `square_signed` (or whatever square op is used) already has a
  noise entry.
