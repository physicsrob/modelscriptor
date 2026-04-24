# Phase C, Part 2 — Int-slot VALUE embedding for small-cardinality integers

Add two columns to the VALUE-row embedding (call them `K` and
`K_NS`) so that small-cardinality integer identifiers (SORT_RESULT,
BSP_RANK, IS_RENDERABLE, HIT_*) can be read back from the KV cache as
the integer they actually are, with no decode Linear and no W_consumer
amplification.  The same two columns let the host's argmax against
`W_EMBED.T` peak at the correct VALUE_k via the quadratic-equality
identity Phase C Part 1 used for attention.

This refactor also fixes a latent bug in the existing
`emit_integer_value_embedding` / `_decode_payload_to_float` round-trip
(see *The bug being fixed* below).

## Prerequisites

- Phase B Part 3 landed: VALUE rows use the layout
  ``[E8_VALUE(8) | raw(1) | gray(16)]``.
- No dependency on Phase C Part 1 (the wall_geom quadratic attention
  still works after this lands; it just gets a correct integer scalar
  as input instead of the broken ~3.2e-4 float).

## Development speed

Skip: per-name int columns, RESOLVED_ANGLE migration, `W_OUTPUT`
decoupling, the three-primitive `read_raw / decode / read_value`
API split.  Smoke test = `make walkthrough` matches reference; depth
unchanged or reduced; previously failing baseline tests pass.

## The bug being fixed

`emit_integer_value_embedding(integer_value=k, max_int=max_int, name)`
picks `W_EMBED[VALUE_k]` directly via a one-hot row lookup.  For
`SORT_RESULT` with `wall_index=3`, that's `VALUE_3`, whose raw slot
holds `(2·3 + 1) / 131072 ≈ 5.34e-5`.

The decoder `_decode_payload_to_float` is calibrated for the
*continuous* quantization path, where `value=3` would be quantized to
`q = 3 · 65535 / 7 ≈ 28086` and emit `VALUE_28086` whose raw slot
holds `0.4286`.  The decode formula
``value = raw·65536·LSB − 0.5·LSB + lo`` (with `LSB = 7/65535`) maps
`raw=0.4286 → value≈3.0`, but maps `raw=5.34e-5 → value≈3.2e-4`.

Consequence: every `kv.readback.get_value_after_last("SORT_RESULT")`
call in the current codebase returns ~3.2e-4 instead of the intended
integer.  Downstream, `wall_index_clamped` is ~0 for every wall, and
``wall_j_onehot`` is ``[1, 0, 0, …]`` always.  RENDER's
`wall_geom_attention` always reads wall 0 — which happens to be the
only visible wall at angle=0 (the only `test_frame_matches_reference`
case that passes), and produces visible-but-wrong renders elsewhere.

The fix lands in `emit_integer_value_embedding` (writes the K/K_NS
columns directly) and in `get_value_after_last` (dispatches to a new
int reader for small-int names so callers don't change).

## What changes

### Embedding layout

`D_EMBED` grows from 25 → 27.  New columns:

```
cols [ 0 :  8] — E8 category code                       (unchanged)
cols [ 8 :  9] — raw slot                               (unchanged)
cols [ 9 : 25] — Gray-code payload                      (unchanged)
cols [25 : 26] — K     = k             for k ≤ MAX_INT_K else 0
cols [26 : 27] — K_NS  = −k²           for k ≤ MAX_INT_K else 0
```

`MAX_INT_K = 255`.  Cap chosen for two reasons:

1. **Headroom.** The motivating consumer (SORT_RESULT carrying
   wall_index) needs ≤ 7 today, but more walls are likely.  Cap 255
   covers any per-frame integer we'd plausibly want as a small-int
   without running into …
2. **Float32 precision.** At `k_target = 255`, the K/K_NS contribution
   to score(VALUE_k) is `2·255·k − k²`, which peaks at `k=255` with
   value 65025 and falls off by 1 at `k=254` (margin = 1, plus 2 from
   gray Hamming-1, total argmax margin = 3).  Score magnitude
   ~66600 + ulp 1.2e-7 = absolute precision ~0.008.  Margin/precision
   = 375×.  Comfortable.  The same scheme stays workable up to
   ~`k=1024` (margin/precision ≈ 8×) before getting tight.

For VALUE_k rows with `k > 255`, K and K_NS are zero — continuous
emits don't activate this mechanism, no impact on their argmax.

### Encoder — predicted embedding for integer / boolean emit

`emit_integer_value_embedding(integer_value=k_target, max_int, name)`
and `emit_boolean_value_embedding(bool_value, name)` keep the existing
`Linear(onehot, rows)` row lookup for the first 25 columns
(E8/raw/gray), then **override** the K/K_NS columns:

```
predicted[K]    = 2 · k_target
predicted[K_NS] = 1
```

Argmax score from the K/K_NS columns:

```
score(VALUE_k) += (2·k_target)·K[k] + 1·K_NS[k]
                = 2·k_target·k − k²
                = −(k − k_target)² + k_target²
```

The `+ k_target²` is constant across k.  Argmax peaks at `k = k_target`
with margin 1 to adjacent k.  Combined with the existing E8 (~1600)
and gray (margin 2 to adjacent k) contributions, total argmax margin
is 3 — same belt-and-suspenders shape as before, with K/K_NS now
providing the primary discrimination in addition to gray.

`encode_value_binary` (the continuous emit path) appends `[K=0, K_NS=0]`
to its output so the result is also 27-wide and dot products are
well-defined.

### Reader — `get_int_after_last(name)` and dispatcher

New method on `ThinkingReadback`:

```python
def get_int_after_last(self, name: str, *, assert_hardness_gt=0.99) -> Node:
    """Return the integer carried in the K column of the most recent
    name-VALUE position.  No decode Linear; output is the integer
    scalar (within softmax-leakage noise ε·MAX_INT_K)."""
```

Implementation: same `attend_most_recent_matching` as
`get_value_after_last`, but `value=K_column` (1-wide, extracted from
the embedding leaf) instead of `value=raw_slot`.

`get_value_after_last(name)` becomes a dispatcher:

- If `name in INT_IDENTIFIER_NAMES`: route to `get_int_after_last(name)`.
- Else: existing raw-slot + decode Linear path.

`INT_IDENTIFIER_NAMES = {"SORT_RESULT", "BSP_RANK", "IS_RENDERABLE",
"HIT_FULL", "HIT_X", "HIT_Y"}` — the identifiers whose VALUE_RANGE_BY_NAME
is integer-valued and within `[0, MAX_INT_K]`.  RESOLVED_ANGLE (0..255)
*could* qualify but is left on the existing path for now (precision
considerations + no immediate consumer paying the cost).

### Local-position decode (sorted.py)

`_decode_local_value_to_float(embedding, "SORT_RESULT")` in
`sorted.py` currently extracts the raw slot from the local position's
embedding and applies the dequantize Linear.  After the refactor it
extracts the K column directly and returns it as-is — no Linear.

(The vis_lo path in the same file decodes a continuous value; it
keeps using `_decode_value_payload_to_float` unchanged.)

## Expected impact

### Bug fixes

The 19 baseline `test_pipeline.py` failures (all of which trace to
`wall_index` being ~3.2e-4 instead of an integer) should all pass
after this lands.  These cover:

- `test_sort_order_matches_bsp[160]`
- `test_sort_visibility_matches_reference[*]`
- `test_walls_are_visible[*]`
- `test_render_wall_heights[*]`
- `test_frame_matches_reference_cardinal[64,128,192]`
- `test_frame_matches_reference_oblique[45,100,160,210]`
- `test_frame_matches_reference_off_center[1.0,-3.0,50]`

### Depth

Modest depth reduction expected.  RENDER's `wall_index_readback`
annotation block currently spans layers 0–13 (attention + decode
Linear).  After the refactor, the decode Linear goes away; the block
should compress to just the attention sublayer (~5 layers tighter on
that segment, but the chain may not be the binding critical-path
constraint).

Phase C Part 1's `wall_geom_attention` keeps its quadratic form — the
math is correct, the inputs were broken.  Now that `wall_index_render`
is a clean integer, the quad attention will actually concentrate on
the right WALL position.

### Noise

`get_int_after_last` for SORT_RESULT: K-column value range across
keys is `{0..7}` at SORT_RESULT VALUE positions, `0` elsewhere
(other names' VALUEs and non-VALUE positions both have K=0 in their
embedding rows for k ≤ 7).  Wait — that's only true if other names'
VALUE_k for `k > 7` have K=0 by virtue of the cap.  For other names
emitting VALUE_k where `k ≤ 7` (e.g., a VIS_LO continuous-quantized
value that happens to round to k=3), K=3 leaks into our reader's V.
Softmax leakage from those positions is gated by `is_value_of("SORT_RESULT")`
to ε ≈ 0 in float32 (match_gain 12000 with score gap 24000 underflows
to zero), so cross-name contamination is negligible.

W_consumer downstream: the wall_geom Q only needs to multiply by 2 to
get `2·wall_index`.  Output noise = `ε · 7 · 2 = 14ε`.  For ε ≈ 1e-9
that's 1.4e-8 — well below every tolerance budget in the renderer.
Compared to the (broken) decoded path: the decoded path's noise
analysis was irrelevant because the decoded value itself was wrong by
~10000×.

## Critical files

1. `torchwright/doom/embedding.py`
   - Bump `D_EMBED` 25 → 27.  Add `D_K = 1`, `D_K_NS = 1` constants.
   - Add `MAX_INT_K = 255` constant.
   - Update `_build_w_embed` to populate K and K_NS for k ≤ MAX_INT_K.
   - Update module docstring.

2. `torchwright/doom/thinking_readback.py`
   - Add `_K_SLOT_START = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD`
     (= 25), `_K_NS_SLOT_START = 26`.
   - Update `encode_value_binary` to append `[K=0, K_NS=0]`.
   - Update `emit_integer_value_embedding` and
     `emit_boolean_value_embedding` to override K, K_NS.
   - Add `INT_IDENTIFIER_NAMES` constant.
   - Add `get_int_after_last` method on `ThinkingReadback`.
   - Update `get_value_after_last` to dispatch by name.
   - Update module docstring.

3. `torchwright/doom/stages/sorted.py`
   - `_decode_local_value_to_float` for SORT_RESULT: read K column
     directly via `extract_from`, return it.

4. Tests:
   - New unit test in `tests/doom/test_embedding.py`: emit_integer
     produces an embedding whose argmax against W_EMBED.T picks
     VALUE_(integer_value), for every integer_value in [0, MAX_INT_K].
   - New unit test in `tests/doom/test_thinking_readback.py`:
     `get_int_after_last` returns the correct integer for every
     small-int name.
   - The existing 19 failing pipeline tests should pass.

## Out-of-scope follow-ups

Not in this part; mentioned so future-you knows what's deferred:

- **`get_value_after_last` API split** into three primitives
  (`read_raw_after_last` / `decode` / `read_value_after_last`).
  Cleaner factoring; lets consumers do their own affine without
  going through bundled decode + assert.  Touches every existing
  caller — separate pass.
- **RESOLVED_ANGLE migration to int_slot.**  Range 0..255 fits
  MAX_INT_K but the existing path works and has no current
  consumer pain.  Migrate when there's a reason.
- **W_OUTPUT decoupling from W_EMBED.T.**  Would let the int_slot be
  read-only-by-attention (not part of host argmax).  Architecturally
  cleaner but the cap-at-255 approach makes argmax work without it.

## Verification

```bash
# 1. Embedding layer changes don't break vocabulary tests.
make test FILE=tests/doom/test_embedding.py

# 2. Readback round-trips for both paths.
make test FILE=tests/doom/test_thinking_readback.py

# 3. Full doom pipeline — the 19 baseline failures should drop.
make test FILE=tests/doom/test_pipeline.py

# 4. Walkthrough is the smoke test.
make walkthrough ARGS="--frames 3"

# 5. Depth measurement.
make graph-stats
```

Expected outcomes:

- All `test_embedding.py` and `test_thinking_readback.py` tests pass,
  including new unit tests for the K/K_NS layout and int reader.
- `test_pipeline.py` failure count drops from 19 → ~0 (bugs fixed).
- Walkthrough matches reference at every frame.
- `make graph-stats` shows `render/wall_index_readback` shrinking from
  the current 0–13 span; total compiled layers stay flat or drop
  slightly.
