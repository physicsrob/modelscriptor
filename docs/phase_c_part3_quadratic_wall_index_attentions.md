# Phase C, Part 3 — Quadratic-equality for the remaining wall_index attentions

Three more attentions still match on wall_index via 8-wide one-hot
encoding.  This part swaps each to the same 2-wide quadratic-equality
form Phase C Part 1 introduced for `render/wall_geom_attention`:

- `thinking_wall/wall_geom_attention` (current span 0–14)
- `render/vis_hi_content_attention` (current span 0–23 — the latest
  attention in RENDER)
- `sort/vis_lo_attention` (current span 11–13)

## Prerequisites

- Phase C Part 1: `WallKVOutput` exposes `wall_index_at_wall` and
  `wall_index_neg_sq` at WALL positions (the quad K channels for
  attentions whose K side is at WALL).
- Phase C Part 2: `INT_IDENTIFIER_NAMES` includes SORT_RESULT, the
  K column of the embedding carries integer wall_index directly,
  and `get_value_after_last("SORT_RESULT")` returns a clean integer.

## Per-attention plan

### 1. `thinking_wall/wall_geom_attention`

Today: 8-wide one-hot match between `wall_j_onehot` (built from
`current_wall_index` via `in_range`) at thinking-identifier positions
and `wall_position_onehot` at WALL positions.

After: 2-wide quad-equality.  K already exists from Part 1
(`wall_index_at_wall`, `wall_index_neg_sq`); Q just needs
`current_wall_index` as a scalar.  thinking_wall already has
`current_wall_index` internally (it's what `wall_j_onehot` is built
from); we expose it locally without changing any external API.

**Zero coupling cost** — entirely internal to thinking_wall.

### 2. New thinking_wall exports for vis_hi / vis_lo

Both `render/vis_hi_content_attention` and `sort/vis_lo_attention`
read VIS_HI / VIS_LO VALUE positions in the thinking phase, keyed by
which wall produced that VALUE.  Today they consume thinking_wall's
8-wide `value_wall_index_onehot`.  For quad-equality they need:

- `value_wall_index_scalar`: the integer wall_index at thinking-VALUE
  positions, sentinel `-100` elsewhere.
- `value_wall_index_neg_sq`: `-wall_index²` at thinking-VALUE
  positions, sentinel `-1000` elsewhere.

Same sentinel pattern (`select(is_thinking_value, real, sentinel,
approximate=False)`) as `bsp_rank_scalar_for_sort` /
`bsp_rank_neg_sq_for_sort` in thinking_wall.py.

### 3. `render/vis_hi_content_attention`

Today: composite key `[is_vis_hi_value (1), value_wall_index_onehot
(8)]` → 9-wide.  Composite query
`[1, query_wall_onehot (8)]` → 9-wide.  Q-side `query_wall_onehot`
built from RENDER's wall_index via in_range (the
`render/wall_index_onehot` block).

After: composite key `[is_vis_hi_value, value_wall_index_scalar,
value_wall_index_neg_sq]` → 3-wide.  Composite query
`[1, 2·wall_index_render, 1]` → 3-wide.

The `render/wall_index_onehot` block was the only remaining consumer
of `wall_j_onehot` at RENDER after Part 1.  After this, that block is
deleted entirely.

### 4. `sort/vis_lo_attention`

Same shape as vis_hi: 3-wide composite K and Q.  Q-side wall_index
from `_decode_local_value_to_float` for SORT_RESULT (post-Part-2,
this is a clean integer from local K column extract).

The `sort/vis_lo_lookup/sort/vis_lo_query_compose` block's in_range
→ one-hot construction goes away — Q is just `[1, 2·wall_idx, 1]`.

## Score / sentinel sanity check

For all three: K's quad pattern is `[wall_idx, -wall_idx²]` where
`wall_idx ∈ [0, max_walls-1]` (= [0, 7]).  Q is
`[2·k_target, 1]`.  Score at matching key = `k_target²`; adjacent
keys differ by 1; sentinel keys (where the type bit is also off)
score `2·k_target·(-100) − 1000 = -200·k_target − 1000`, i.e. ≤
`-1000` for k_target=0 and ≤ `-2400` for k_target=7.

For the vis_hi/vis_lo cases the score also has the type-match
contribution: `is_X_value · 1` adds `+1` at matching positions and
`0` (or sentinel) elsewhere.  Net: matching wall + matching type
scores `k_target² + 1`; matching wall + wrong type scores `k_target²
+ 0`; non-matching wall + matching type scores `(k_target² − Δ²) +
1`; sentinel scores `≤ -1000`.  With `match_gain=12000` (the
existing content match-gain), all the renderable separations are
~e^12000 vs ~e^0 — saturated.

## Expected impact

`render/wall_index_onehot` block deleted (5 layers reclaimed in
RENDER).  Each migrated attention's K/Q width drops 9→3 (cheaper
per-head allocation).  vis_hi span moves earlier in RENDER —
`state_transitions` end (which depends on vis_hi for the next-state
calc) should compress.  Total layer count: TBD on landing; not the
primary motivation.

## Out-of-scope

- BSP_RANK / IS_RENDERABLE / HIT_* migrations to int-slot — they
  round-trip correctly via the continuous emit path and there's no
  hot-path consumer paying a decode cost (per Phase C Part 2 plan).
- The three-primitive readback API split.

## Verification

```bash
make test FILE=tests/doom/test_pipeline.py
make test FILE=tests/doom/test_thinking_readback.py
make test FILE=tests/doom/test_render_graph_precision.py
make graph-stats
```

Expected: pipeline failure count stays at 6 (the pre-existing
`vis_hi=0.0` stub failures and the angle-160 issues remain — not
addressed here); other suites unchanged or improved; depth modestly
reduced.
