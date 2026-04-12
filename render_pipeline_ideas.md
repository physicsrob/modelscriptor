# Render Pipeline Ideas: Walls-as-Tokens + Multi-Phase Architecture

Raw brainstorm dump from design session. Not fact-checked, not
committed to. Intended as a reference for future work.

---

## The north star

A compelling blog post about rendering DOOM in a standard transformer.
Attention should be the central mechanism, not a side concern. The
host must be dumb (slice-writes only). One compiled transformer should
ideally handle any level.

---

## Current architecture (what's on main)

- One graph, compiled to a HeadlessTransformer.
- Per frame: W * H/rp autoregressive steps (e.g., 160 * 10 = 1600).
- Each step runs the full graph: game logic + per-segment intersection
  for ALL N walls in parallel + reduce_min + wall_height_lookup +
  texture_column_lookup + textured_column_fill.
- Wall geometry baked into Linear weights at compile time.
- Texture data baked into piecewise_linear weights at compile time
  (post M1 rewrite using linear_bin_index + dynamic_extract).
- Per-step cost: total_ops = 328 + 203*N (measured).
- Peak residual width (peak_d): flat at ~451 for N<=8, then grows
  linearly. Hits d=2048 cap at N~32. Compiler adds extra layers to
  cope with pressure above the cap.
- Level-specific: changing the level requires recompilation.

### Key scaling numbers (from profile_walls.py)

| N walls | layers (d=2048) | layers (d=6144) | peak_d (true) |
|--------:|----------------:|----------------:|--------------:|
|       4 |              51 |               - |           451 |
|       8 |              57 |               - |           451 |
|      16 |              67 |               - |           795 |
|      32 |              90 |              65 |         1,471 |
|      64 |             165 |              77 |         4,849 |

---

## Core idea: walls as runtime tokens

Instead of baking wall geometry into weights, feed walls as input
tokens. The transformer's attention becomes the "gather" primitive
that selects which wall a given ray hits.

### Token layout (original single-hit proposal)

| Position | Content | Role |
|----------|---------|------|
| 0 | Player state + inputs | Seed for game logic |
| 1..N | Wall segments (x1,y1,x2,y2,tex_id) | Prefill: populate KV cache |
| N+1..end | Render queries (one per patch) | Attend to walls, emit pixels |

### Attention design for wall selection

- Q = logit_scale * (ray_cos, ray_sin, is_query)
- K = (wall_cos_mid, wall_sin_mid, is_wall * wall_bias - dist_scale * wall_dist)
- V = (wall_id, wall geometry params, texture_id)
- Q*K at query@wall = logit_scale * (cos(ray - wall_angle) + wall_bias - dist_scale * wall_dist)

The `wall_bias` term keeps wall logits strictly positive so they
dominate the zero-logit self-attention baseline from causal masking.
Without it, walls at distance ~1 tie with V=0 query-self rows and
the gathered output is halved. Tested and verified in
test_wall_attention.py.

Zero-K isolation: query rows have K=(0,0,0) because wall slots are
zero on query rows. So queries never attend to each other. Combined
with V=0 on query rows, any weight leaking onto queries contributes
nothing.

### Softmax sharpness (measured)

| N walls | logit_scale | correct-wall weight |
|--------:|------------:|--------------------:|
|       2 |          50 |              > 0.99 |
|       4 |         100 |             > 0.999 |
|       8 |         100 |              > 0.99 |
|      16 |         200 |             > 0.999 |
|      32 |         500 |              > 0.99 |

E8 spherical embeddings (from spherical_codes.py) would likely
eliminate the sharpness concern entirely for discrete ID lookups.
Angular similarity (cos/sin) is still needed for continuous
ray-wall alignment.

### Parametric single-wall intersection (measured)

When wall endpoints are runtime inputs (from attention V) instead
of baked constants, the intersection math (den, num_t, num_u) costs
more because products of two runtime scalars need signed_multiply
or piecewise_linear_2d instead of free Linears.

Optimized formulation using algebraic simplification:

    ex = bx - ax;  ey = by - ay
    dx = ax - px;  dy = py - ay
    den   = ey * cos - ex * sin     (2 pos*trig products via pwl_2d)
    num_t = ey * dx + ex * dy       (2 pos*pos products via pwl_2d)
    num_u = dx * sin + dy * cos     (2 pos*trig products via pwl_2d)

Key insight: collecting terms in (ax-px) and (py-ay) shares dx/dy
between num_t and num_u, cutting products from 10 to 6.

| Version | ops | layers | peak_d |
|---------|----:|-------:|-------:|
| Naive (10 x signed_multiply) | 233 | 10 | 62 |
| Algebraic + pwl_2d for pos*trig | 83 | 8 | 34 |
| + pwl_2d for pos*pos too | **51** | **6** | **34** |

Compare to baked per-wall marginal: 203 ops/wall. The parametric
version is ~25% of one wall's baked cost, and it's constant in N.

---

## The multi-hit problem

Original DOOM renders multiple visible surfaces per column (see over
a short wall to the far wall behind it). The single-hit
walls-as-tokens design only finds one wall per ray via attention.

### Why sorting matters

To render K visible surfaces per column in depth order, you need the
K closest walls that are on the ray. The baked design could add a
sorting network (bitonic sort, O(log^2 N) depth) on top of the
existing N-parallel distance computations. Walls-as-tokens can't
sort because it deliberately avoids materializing all N distances
on the residual stream.

### Iterative attention as "sort"

Key insight: autoregressive iteration converts depth cost to rollout
length cost. Instead of sorting N items in one deep graph, emit N
tokens where each one finds "the next closest wall" by attending to
all walls with a distance-floor suppression term.

---

## The multi-phase pipeline (the big idea)

Map DOOM's multi-phase rendering pipeline directly onto the
transformer's autoregressive rollout. Each phase is a block of
tokens that reads the previous phase's results from the KV cache
and writes refined data for the next phase.

### Phase layout

| Phase | Tokens | What each computes |
|-------|-------:|-----|
| 0. Texture columns | T = N_tex * tex_w | Nothing (raw data passthrough) |
| 1. Wall geometry | N | Attend to textures for binding; compute E8 key, derived params |
| 2. Sorted walls | N | Iterative min-find: emit k-th closest wall globally |
| 3. Column resolver | W | Per-column: find K visible walls from sorted list, compute projections |
| 4. Render patches | W * H/rp | Read own column's resolved hit list, texture lookup, column fill |

For T=256, N=32, W=160, H/rp=10: total = 2080 tokens per frame.
Current baked: 1600 tokens per frame.

### DOOM pipeline mapping

| DOOM phase | Transformer phase | Produces |
|------------|-------------------|----------|
| BSP traversal (front-to-back) | Sort phase | Walls in distance order |
| R_AddLine / R_StoreWallRange | Column resolver | Per-column visible-seg list |
| R_DrawColumn / R_RenderSegLoop | Render patches | Pixel output |
| R_DrawPlanes (visplanes) | Part of render or separate phase | Floor/ceiling fill |

### Phase 0: Texture columns

Positions 1..T. Each token's input carries one column of one
texture: tex_h * 3 floats of pixel data. K is an E8 code for
(texture_id, column_index). V is the pixel data.

Texture tokens come first in the causal chain so wall tokens can
attend backward to them if useful.

### Phase 1: Wall geometry

Positions T+1..T+N. Each token's input carries raw wall endpoints
(x1, y1, x2, y2, texture_id).

During prefill:
- Attend to position 0 to read player state (px, py, angle) via
  get_prev_value pattern (already used in game_graph.py for seed
  broadcast).
- Compute derived quantities: ex, ey, dx, dy, distance from player,
  angular midpoint (cos_mid, sin_mid), visible column range
  (col_start, col_end).
- Optionally attend to texture tokens to bind texture identity
  (carry texture E8 code in V for downstream use).
- Write K = (E8_wall_id, cos_mid, sin_mid, -dist, col_start,
  col_end, ...) and V = (wall geometry + derived params + tex info).

### Phase 2: Sorted walls

Positions T+N+1..T+2N. Each token emits the k-th closest wall
globally (by distance from player).

Sort position k:
1. Read d_{k-1} from position k-1 via get_prev_value attention.
   (First sort token reads 0 or -inf.)
2. Compute bucket index for d_{k-1}: linear_bin_index(d_{k-1}, 0,
   max_dist, n_buckets).
3. Build suppression mask: in_range(0, bucket_{k-1}+1, n_buckets)
   gives a one-hot-like mask over suppressed distance buckets.
4. Attention to wall tokens with:
   - Q = (angular-don't-care, 1, -big*supp_0, ..., -big*supp_{n_buckets-1})
   - K = (angular, -dist, bucket_0_indicator, ..., bucket_{n_buckets-1}_indicator)
   - Q*K = -dist + wall_bias + sum(-big * supp_i * bucket_i)
   - Walls in suppressed buckets get -big logit; walls above
     d_{k-1} compete by exact distance.
5. Emit: sorted wall's (wall_id, geometry, distance, col_start,
   col_end, tex_id, ...).

Subtlety: bucket granularity. Within a bucket, walls are tiebroken
by exact distance (the -dist term in Q*K). Finer buckets (64 bins)
reduce within-bucket ambiguity.

Subtlety: same-bucket walls. If d_{k-1} is mid-bucket, suppressing
that bucket kills ALL walls in it, including ones we haven't
selected yet. Fix: suppress only STRICTLY LOWER buckets, and
within the current bucket rely on exact-distance tiebreak. This
means the suppression mask is in_range(0, bucket_{k-1}, n_buckets)
(exclusive upper bound) rather than +1.

### Phase 3: Column resolver (the key insight)

Positions T+2N+1..T+2N+W. Each token resolves one screen column.

This phase exists because the multi-hit work (finding K visible
walls per column) is the SAME for every patch in the column. With
H/rp = 10 patches per column, computing the hit list once and
caching it saves 9 redundant multi-hit lookups per column.

Column resolver for column c:
1. N-parallel attention read: use N attention heads (or one wide
   head), each targeting a specific sort position via
   attend_to_offset(delta=-(k+1)) for k=0..N-1. This reads ALL
   sorted walls' params in one attention layer (wide but shallow).
2. For each of N sorted walls: check "does this wall cover column
   c?" via in_range(col_c, wall_col_start, wall_col_end). N
   parallel in_range checks.
3. Priority-select: take the first K walls (in sort order = distance
   order) that pass the on-ray test. A K-deep cascade of selects
   over N candidates.
4. For each of K selected walls: run parametric intersection at
   column c's exact ray angle to get (wall_top, wall_bot,
   tex_col_idx, distance). K * 51 ops.
5. Write to KV cache: K * (wall_top, wall_bot, tex_col_idx,
   texture_id, distance, ...).

Estimated cost: ~400 ops per resolver token.

Alternative to step 4: if the sort phase pre-computed the
intersection formula coefficients (ey, -ex, dx, dy, num_t per
wall), the resolver could evaluate them at the column's ray angle
via a few Linears + one division, potentially cheaper than the
full parametric intersection.

### Phase 4: Render patches

Positions T+2N+W+1..end. Each token renders one rp-tall patch of
one column.

Render token for (column c, patch p):
1. Attend to column c's resolver token (via E8-coded column ID).
   Read back K * (wall_top, wall_bot, tex_col_idx, texture_id,
   distance).
2. For each of K hits: attend to texture-column tokens with
   Q = E8(texture_id, tex_col_idx) to get tex_column_colors
   (tex_h * 3 floats). K texture lookups.
3. For each screen row in the patch: determine which of the K walls
   (if any) covers this row, in depth order. Priority cascade:
   first wall whose [wall_top, wall_bot] contains row wins.
4. For the winning wall at this row: use linear_bin_index +
   dynamic_extract to get the texture color from
   tex_column_colors.
5. Compose with ceiling/floor for rows not covered by any wall.
6. Emit rp * 3 pixel values.

Estimated cost: ~250 ops per render token.

### Cost estimate (rough)

For N=32, W=160, H/rp=10, K=3, T=256 tex columns:

Total tokens: 256 + 32 + 32 + 160 + 1600 = 2080.

Monolithic graph: all phases' ops run at every token. Per-token
cost is roughly max(phase ops) ~ 400-500. Total per-frame:
2080 * 450 ~ 936k ops.

Compare to current baked single-hit: 1600 * 6824 ~ 10.9M ops.

Roughly 10-12x reduction WITH multi-hit K=3 support.

---

## Textures as tokens

### Motivation

If walls are runtime data but textures are still baked into weights,
swapping to a new level still requires recompilation. For true level
independence, textures must also be runtime data.

### Design

One token per texture column. For 4 textures * 64 columns = 256
tokens. Each carries tex_h * 3 = 192 floats of pixel data in its V.
K is an E8 code for (texture_id, column_index).

### Cost comparison

Current (piecewise_linear with baked weights): 1 MLP sublayer per
texture lookup.

Textures-as-tokens: 6-10 MLP sublayers per texture lookup (attention
+ query construction).

More expensive per query. But:
- Level-independent (no recompile to change textures)
- Supports dynamic textures (animated walls, damage states)
- Scales to large atlases without growing model weights
- Completes the "everything is a token" narrative

### Token ordering

Texture columns BEFORE wall segments in the causal chain. This lets
wall tokens attend to their texture tokens during prefill if useful
(e.g., binding texture identity, pre-computing texture-aware
properties). Costs nothing even if unused.

### E8 for keying

Each texture column's K is an E8 code for its (texture_id,
column_index) pair. Lookup is sharp by construction regardless of
atlas size. No logit-scale tuning needed.

---

## Staircases and platforms

### Per-wall heights (staircases) -- straightforward

Each wall token's V includes (floor_height_world, ceiling_height_world).
The render phase reads these and projects to screen space:

    factor = perp_cos / distance
    wall_top    = screen_center - (ceiling_world - eye_y) * factor
    wall_bottom = screen_center - (floor_world   - eye_y) * factor

Two runtime multiplies (~6 sublayers). Constant in N, additive on
top of existing per-render-step cost.

Easier in walls-as-tokens than baked: the baked design would need
per-segment wall_height_lookup instances or a wall-index-dependent
lookup table.

### Platforms (see over/under) -- enabled by multi-hit

Seeing over a short wall to the far wall is exactly the K=2 multi-hit
case. The column resolver finds both walls; the render phase composites
them.

### Horizontal surfaces (floor/ceiling at varying heights)

Requires a "visplane" pass: for each column, after rendering walls,
fill floor/ceiling regions based on which sector the ray is over.
Neither design (baked or walls-as-tokens) currently has this.

Could be an additional phase: sector tokens, each carrying floor
height + floor texture. Render phase attends to sectors to get the
floor/ceiling for un-walled regions.

---

## Intermediate computation tokens (general principle)

The autoregressive rollout IS the computation pipeline. Each phase
of tokens refines the previous phase's output. The KV cache
accumulates increasingly-processed data.

Work done at an intermediate token is done ONCE, and the result is
available to ALL subsequent tokens. So if many render tokens need
the same computed quantity, computing it once and reading via
attention is cheaper than recomputing.

### What's worth pre-computing at intermediate tokens

Per-wall derived quantities (shared across all columns hitting that
wall):
- ex, ey, dx, dy
- distance from player
- angular range (col_start, col_end)
- pre-computed intersection formula coefficients
- texture identity binding

Per-column derived quantities (shared across all patches in that
column):
- K-hit list (which walls, in what order)
- per-hit wall_top, wall_bot, tex_col_idx
- pre-computed texture column data

### What's NOT worth pre-computing

Per-wall-per-column quantities (wall_top at column c for wall k):
only needed by one render token group (the patches of that column).
Not shared widely enough to justify an intermediate token.

Per-pixel quantities: never shared. Always compute at render time.

---

## Key risks and open questions

### Built and tested

- Single-hit wall selection via attention (test_wall_attention.py,
  16 tests, all pass including probe_graph)
- Parametric single-wall intersection (test_parametric_intersection.py,
  5 tests, 51 ops / 6 layers / peak_d=34)
- Cost scaling of baked design (profile_walls.py, linear in N,
  peak_d saturates at d=2048)

### Designed but not tested

- Sort phase: iterative attention with bucket-based distance
  suppression. Core mechanism is plausible; bucket granularity and
  same-bucket tiebreak need prototype testing.
- Column resolver: N-parallel attention heads reading specific sort
  positions via attend_to_offset. Novel for this project; needs
  compiler-level verification.
- Multi-hit column fill: priority cascade over K walls per screen
  row. Straightforward but hasn't been costed.

### Unknown

- Whether E8 codes compose well with continuous angular/distance
  terms in a single Q*K dot product (mixed discrete + continuous
  keys).
- Whether the N-head parallel attention at the resolver compiles
  without blowing up the attention sublayer's parameter count.
- Whether the monolithic-graph "all phases run at all tokens" cost
  model is accurate, or whether the compiler optimizes away
  gated-off branches somehow.
- Per-frame KV cache memory at 2080 tokens * d_model. At d=2048:
  2080 * 2048 * 2 (K+V) * 2 (fp16) * n_layers bytes. For 30
  layers: ~500 MB. Might be tight on smaller GPUs.

### The wall_bias pitfall

Hand-designed attention has a zero-logit baseline from causal
self-attention. If a wall's natural logit (cos_diff - dist_scale *
wall_dist) can reach zero, it ties with this baseline and the
gathered output is halved. Fix: route is_wall * wall_bias into K's
distance dimension. wall_bias must exceed max(dist_scale * wall_dist)
for all realistic distances. Tested at wall_bias=30 which handles
max_dist~20 comfortably.

This is a general principle for hand-designed attention: keep "real"
logits strictly above the zero-logit floor set by non-relevant
positions in the causal window.

---

## Blog narrative angles

### "DOOM's renderer IS a transformer pipeline"

DOOM's rendering algorithm (BSP traversal -> seg processing ->
drawsegs -> visplanes) maps directly onto a transformer's
autoregressive rollout (sort phase -> column resolver -> render
patches -> floor/ceiling fill). Each phase is a block of tokens.
The KV cache is the data structure that carries intermediate results
between phases, exactly like DOOM's internal arrays (drawsegs list,
visplane list, etc.).

### "Attention is the missing gather primitive"

The baked design uses reduce_min (a binary tree of selects) for wall
selection and piecewise_linear for texture lookup. Both are "find one
thing from many" operations. Attention does the same thing natively
via softmax, and it operates on RUNTIME data rather than baked
weights. This one mechanism replaces both reduce_min and
piecewise_linear-as-table-lookup, and it's the thing transformers
were designed to do.

### "Everything is a token"

Wall geometry, texture data, sorted intermediate results, per-column
hit lists — all represented as tokens in the autoregressive sequence.
The transformer becomes a rendering engine that takes a scene
description (as tokens) and produces pixels (as token outputs).
Level independence falls out for free: same compiled transformer,
different input tokens, different rendered level.

### "We traded depth for length"

The hardest part of multi-hit rendering (sorting walls by depth) is
an O(log^2 N) depth operation in a sorting network. By moving the
sort into the autoregressive rollout (one wall emitted per token),
we trade depth for sequence length: O(N) more tokens, but bounded
depth per token. The transformer's native sequential-generation
mechanism implements the sort.

---

## Immediate next steps (if pursuing)

1. Sort-phase prototype: 3-wall, 1-query test exercising iterative
   attention with distance-bucket suppression. Verify sorted output
   matches expected order. ~1-2 days.

2. Column-resolver prototype: 3 sorted walls + 1 resolver token.
   Verify N-parallel attention + on-ray filtering + priority select.
   ~1-2 days.

3. End-to-end mini: 3 walls + 3 columns + 3 patches. Full pipeline
   from wall tokens through sorted walls through resolver through
   rendered patches. Verify against reference renderer. ~3-5 days.

4. Integration: replace baked rendering pipeline in game_graph.py
   with multi-phase architecture. This is the big lift.
