# The DOOM Game Graph

This is a transformer that runs DOOM. Not a neural net trained to play
DOOM — a hand-designed computational graph whose forward pass *is* the
game engine. The graph compiles to a real transformer (attention heads,
MLP sublayers, linear layers) and executes on GPU via `model.step()`.

The host is deliberately dumb: it feeds tokens in and bitblits pixels
out. All game logic — player movement, collision detection, BSP
traversal, front-to-back wall sorting, perspective projection, and
texture-mapped column rendering — happens inside the transformer's
forward pass.

## Token Sequence

Each frame, the transformer processes this sequence:

```
TEX_COL×(num_tex × tex_w) → INPUT → BSP_NODE×M → WALL×N →
EOS → SORTED_WALL×N → (THINKING → RENDER×k)×N
```

These compile into four phases:

| Phase | Tokens | Mode | Typical count |
|-------|--------|------|---------------|
| -1 (Tex) | TEX_COL | Prefill | num_tex × tex_w (e.g. 512) |
| 0 (Prefill) | INPUT + BSP_NODE + WALL + EOS | Prefill | 1 + M + N + 1 (e.g. 58) |
| 1 (Sort) | SORTED_WALL | Autoregressive | N (e.g. 8) |
| 2 (Render) | THINKING + RENDER | Autoregressive | ~N + N×k (e.g. 250) |

Prefill tokens are processed in a single batched call (they don't
depend on each other's outputs). Autoregressive tokens are generated
one at a time — each token's output determines the next token's input.

## Token Types

Eight token types, identified by E8 spherical codes (8-dimensional
unit vectors from the E8 lattice):

```
TOKEN_INPUT      (0)    TOKEN_TEX_COL     (5)
TOKEN_WALL       (1)    TOKEN_THINKING    (6)
TOKEN_BSP_NODE   (7)    TOKEN_SORTED_WALL (3)
TOKEN_EOS        (2)    TOKEN_RENDER      (4)
```

The E8 codes serve as position encoding. Texture IDs also map to E8
vectors starting at index 8, so the same embedding space covers both
token type and texture identity.

## Stages

### TEX_COL — Texture Data (Phase -1)

One token per column of each texture. Each carries the raw RGB pixel
data for that column (`tex_h × 3` floats) plus an E8 code identifying
which texture it belongs to and a one-hot encoding of its column index
within that texture.

These tokens sit in the KV cache for the rest of the frame. When
RENDER tokens need texture pixels, they retrieve them via
`attend_argmax_dot` — a dot-product attention where the query encodes
(texture_id, column_index) and the best-matching TEX_COL position's
pixel data is returned as the value.

The stage's only computation is converting the host-fed column index
into a one-hot vector (`tc_onehot_01`) used as the attention key.

### INPUT — Player Controls (Phase 0)

A single token. Receives the player's current position, angle, and
six boolean movement flags (forward, backward, turn left, turn right,
strafe left, strafe right).

Computes:

- **New angle**: `(old_angle + turn_right×speed - turn_left×speed) mod 256`.
  Angles are integers 0..255 (a full circle in 256 steps).
- **Velocity**: `(dx, dy)` from the new angle and movement flags.
  Forward/backward move along the facing direction; strafing moves
  perpendicular. Uses `piecewise_linear` trig lookups over the
  256-entry trig table.
- **Trig values**: `(cos, sin)` of the new angle, needed by
  downstream stages for coordinate rotations.

All five derived values (vel_dx, vel_dy, move_cos, move_sin,
new_angle) are **broadcast to every position** via
`attend_mean_where`. Since exactly one position has `is_input=1`, the
"mean" is just that position's value — every subsequent token can read
the player's velocity and facing direction as plain node inputs.

### BSP_NODE — Spatial Classification (Phase 0)

M tokens (typically 48), one per splitting plane of the BSP tree.

Each BSP_NODE token carries a normalized plane `(nx, ny, d)` and
classifies the player as FRONT or BACK:

```
side_P = sign(nx × player_x + ny × player_y + d)
```

The result (1 for FRONT, 0 for BACK) is spread into slot `i` of an
M-wide vector via the token's `bsp_node_id_onehot`. An
`attend_mean_where` over all BSP_NODE positions gathers these into a
shared `side_P_vec` — a binary vector available at every position
telling which side of every BSP plane the player is on.

WALL tokens consume `side_P_vec` to compute their BSP rank (see
below).

### WALL — Geometry + Physics + Precomputation (Phase 0)

N tokens (typically 8), one per wall segment. The workhorse of the
prefill phase. Each WALL token performs four independent computations
on its wall segment:

**1. Collision detection.**
Three ray-segment intersection tests: the player's full velocity
vector `(vel_dx, vel_dy)`, an x-only ray `(vel_dx, 0)`, and a y-only
ray `(0, vel_dy)`. Each produces a hit/miss flag (±1). Uses
`piecewise_linear_2d` products to compute parametric intersection
coordinates and validity checks.

The axis-separated rays enable wall sliding in the EOS stage: if the
full ray hits but the x-only ray doesn't, the player can still move
on the x axis.

**2. BSP rank.**
`rank = dot(wall_bsp_coeffs, side_P_vec) + wall_bsp_const` — a dot
product of host-precomputed coefficients against the BSP side
decisions. Produces a clean integer in 0..N-1 that gives the
front-to-back ordering of walls from the player's current position.
The BSP tree structure guarantees these ranks are a permutation — no
ties — which is critical for the sort stage's argmin attention to
produce clean selections.

**3. Render precomputation.**
Rotates wall geometry into the player's angular frame, computing five
constants that the RENDER stage needs for per-column projection:

- `sort_den`: the denominator of the central-ray intersection
  (proportional to perpendicular distance).
- `C, D, E`: rotation-frame constants for per-column angle offset,
  texture u-coordinate, and perspective correction.
- `H_inv`: the wall height scaling factor (`H / |sort_num_t|`).

These are column-independent — they depend only on the wall's
geometry relative to the player, not on which screen column is being
rendered. Pre-computing them here avoids re-deriving them at every
RENDER token.

**4. Visibility column range.**
Projects the wall onto the screen to determine which columns
(`vis_lo`, `vis_hi`) it subtends. Involves rotating both wall
endpoints into the player's frame, clipping against the field-of-view
cone (two half-plane tests), and projecting the clipped endpoints via
`atan(cross/dot)` (approximated by `low_rank_2d` with rank 3). Gated
to 0 for non-renderable walls.

A wall is **renderable** if it is a real wall token, its central ray
is not parallel to the wall (`|sort_den| > 0.05`), and the
intersection is in front of the player (`sort_num_t × sign(sort_den)
> 0`).

**Payload packing.**
All per-wall data is concatenated into a single `sort_value` vector
(13 + max_walls wide) for the SORTED stage to retrieve via attention:

```
[ax, ay, bx, by, tex_id,           (5)  geometry
 sort_den, C, D, E, H_inv,         (5)  render precompute
 bsp_rank,                          (1)  sort score
 vis_lo, vis_hi,                    (2)  visibility columns
 position_onehot]                   (max_walls)  wall identity
```

**Indicators_above.**
A max_walls-wide thermometer vector where slot `c` is 1 iff
`bsp_rank >= c` AND the wall is renderable. This is the key-side
basis for the SORTED stage's threshold-based argmin attention.

### EOS — Collision Resolution + State Broadcast (Phase 0)

A single token marking the end of prefill. Two jobs:

**Collision resolution.**
Aggregates the per-WALL hit flags via `attend_mean_where` over WALL
positions. If the averaged flag for any ray exceeds a threshold
(0.05), at least one wall was hit. Applies axis-separated wall
sliding:

- X-axis: move if neither the full ray nor the x-only ray hit.
- Y-axis: move if neither the full ray nor the y-only ray hit.

```
resolved_x = select(can_move_x, player_x + vel_dx, player_x)
resolved_y = select(can_move_y, player_y + vel_dy, player_y)
```

**State broadcast.**
Copies the resolved `(x, y, angle)` to every position via
`attend_mean_where` over the single EOS position. This makes the
post-collision player state available to the SORTED and RENDER stages
without requiring the host to compute or feed it.

**Sort loop seeding.**
The EOS token's output includes a `sort_feedback` vector that
initializes the autoregressive sort loop. The key field is
`prev_bsp_rank = -1`, so the first SORTED token's threshold is 0 —
it picks any renderable wall (all `indicators_above[0]` values are 1
for renderable walls).

### SORTED_WALL — Front-to-Back Sort (Phase 1)

N tokens, autoregressive. Each picks the next-closest wall in BSP
rank order.

The attention primitive is `attend_argmin_above_integer`: given a
threshold (derived from `prev_bsp_rank` in the sort_feedback), it
finds the WALL position with the smallest BSP rank strictly exceeding
that threshold. The key-side `indicators_above` thermometer encodes
both the rank comparison and the renderability gate — non-renderable
walls have all-zero indicators and are invisible to the attention.

Each SORTED token:

1. Converts `prev_bsp_rank` to a threshold one-hot (clamped to
   `[0, max_walls-1]` for exhaustion safety).
2. Runs `attend_argmin_above_integer` over WALL positions, retrieving
   the packed payload of the winning wall.
3. Unpacks the payload into geometry, render precomputes, visibility
   columns, BSP rank, and wall identity one-hot.
4. Detects sort exhaustion: if `prev_bsp_rank >= sel_bsp_rank`, the
   attention averaged garbage (no valid keys above threshold).
   Sentinels the BSP rank to 99 so THINKING ignores exhausted picks.
5. Gates render data to zero at non-SORTED positions so THINKING's
   attention sums cleanly.

The output `sel_bsp_rank` feeds back as the next step's
`prev_bsp_rank`, advancing the threshold by exactly 1 each step.
After N steps, all renderable walls have been picked in front-to-back
order.

### THINKING — Wall Selection for Rendering (Phase 2)

One token per wall, interleaved with RENDER blocks. Selects the next
un-rendered wall and loads its render data into the feedback overlay.

Uses `attend_argmin_unmasked` over SORTED positions:

- **Score**: `sel_bsp_rank` (BSP rank of the wall selected at each
  SORTED step, sentineled to 99 at exhausted positions).
- **Mask**: `render_mask` (max_walls-wide, 1 for already-rendered
  walls — the attention ignores masked positions).
- **Position key**: `sel_onehot` (wall-index one-hot, aligning the
  mask with wall identity).
- **Value**: the wall's render data + visibility bounds + one-hot.

The attention picks the lowest-ranked un-rendered wall and returns
its `(sort_den, C, D, E, H_inv, tex_id, vis_lo, vis_hi, onehot)`.
These are packed into the `render_feedback` overlay, seeding the next
block of RENDER tokens with everything they need to render that wall.

### RENDER — Pixel Generation (Phase 2)

The pixel-producing token. Each paints a vertical chunk of one screen
column. A state machine with three transitions drives the
autoregressive loop.

**Per-token computation:**

1. **Active column.** If `render_is_new_wall`, reset to `fb_col_lo`
   (the wall's leftmost visible column). Otherwise, continue from
   `render_col`.

2. **Angle offset.** `angle_offset = (col × fov / W) - fov/2` — the
   horizontal angle of this column relative to the screen center, in
   trig-table steps (0..255 units).

3. **Wall height.** `tan(angle_offset)` via piecewise_linear, then:
   ```
   den_over_cos = sort_den - C × tan(offset)
   wall_height  = H_inv × |den_over_cos|
   ```
   The first line is the per-column horizontal projection factor. The
   second scales by the wall's precomputed height reciprocal. Uses
   `piecewise_linear_2d` with log-spaced breakpoints for `H_inv`
   (values span 0.01 to ~H/0.3).

4. **Texture column.** Determines which column of the wall's texture
   maps to this screen column:
   ```
   abs_nuc = |D + E × tan(offset)|
   tex_col = |{k : tex_w × abs_nuc >= k × abs_den}|
   ```
   This is a thermometer comparison — no division needed. Each
   threshold `k` is an exact `multiply_const` + `subtract` +
   `compare`, so the only approximation error comes from the upstream
   `piecewise_linear_2d` that produced `abs_nuc`.

5. **Texture fetch.** `attend_argmax_dot` retrieves the pixel data
   from the matching TEX_COL position in the KV cache. The query
   is `(tex_id_e8, scaled_col_onehot)` — the E8 code of the
   texture (looked up via piecewise_linear from the integer tex_id)
   concatenated with a scaled one-hot of the texture column. The
   TEX_COL key is `(texture_id_e8, scaled_tc_onehot_01)`. The
   dot product is largest at the matching (texture, column) pair.

6. **Chunk fill.** Paints `chunk_size` rows (default 20) of the
   column. For each row, computes the texture row index via
   `floor((y - wall_top) × tex_height / wall_height)`, extracts the
   RGB from the texture column via `dynamic_extract`, and composites
   over ceiling/floor colors using `in_range` masks.

7. **State transitions.** Three mutually exclusive cases:
   - **More chunks**: `active_start + chunk_size < wall_bottom` —
     stay on this column, advance `chunk_start` by `chunk_size`.
   - **Advance column**: no more chunks, `active_col + 1 <= vis_hi`
     — move to the next column, reset `chunk_start` to sentinel (-1).
   - **Advance wall**: no more chunks, no more columns — add
     `fb_onehot` to `render_mask`. If all walls masked, set
     `done = +1`. Output `token_type = E8_THINKING` to trigger wall
     selection for the next wall.

## Outputs

Every position produces two categories of output:

### Overlaid Outputs (Autoregressive Feedback)

Fed back as the next token's input via delta transfer — the output
lands at the same residual-stream columns as the corresponding input.

- **token_type** (8-wide): E8 code for the next token type. The
  transformer decides what comes next:
  - EOS → SORTED_WALL (start sort)
  - SORTED → SORTED_WALL (continue sort)
  - THINKING → RENDER (start rendering this wall)
  - RENDER → RENDER or THINKING (continue column/chunk or next wall)

- **sort_feedback** (8 + 5 + 3 + max_walls wide): Sort loop state.
  Layout: `[E8_SORTED_WALL, sel_wall_data(5), sel_bsp_rank(1),
  vis_lo(1), vis_hi(1), sel_onehot(max_walls)]`. The load-bearing
  field is `sel_bsp_rank` at offset 13 — the threshold for the next
  SORTED step's argmin.

- **render_feedback** (2 × max_walls + 11 wide): Render loop state.
  Layout: `[render_mask(max_walls), render_col(1),
  render_is_new_wall(1), render_chunk_start(1), sort_den(1), C(1),
  D(1), E(1), H_inv(1), tex_id(1), col_lo(1), col_hi(1),
  fb_onehot(max_walls)]`. Carries the current wall's identity,
  precomputed render parameters, visibility bounds, and the
  column/chunk state machine position.

### Overflow Outputs (Host Bitblits)

Placed after the input region in the residual stream. The host reads
these and writes them to the framebuffer:

- **pixels** (chunk_size × 3 wide): RGB values for the current chunk.
- **col** (1): Screen column index.
- **start** (1): Screen row where this chunk begins.
- **length** (1): Number of rows painted (0 for non-RENDER tokens).
- **done** (1): +1 when all walls are fully rendered, -1 otherwise.

The host's rendering loop is trivial: read `(pixels, col, start,
length)` from each RENDER token's output, bitblit the pixel strip to
the framebuffer at `(col, start)` with skip-fill compositing (don't
overwrite already-filled pixels), and stop when `done > 0`.

## How the Graph Becomes a Transformer

The computational graph compiles to a standard transformer via
`compile_headless`:

- **Attention heads** implement the five cross-position data flows:
  - `attend_mean_where`: broadcast (INPUT→all, BSP→all, EOS→all,
    WALL→EOS for collision aggregation).
  - `attend_argmin_above_integer`: threshold-based selection
    (WALL→SORTED for front-to-back sort).
  - `attend_argmin_unmasked`: masked selection (SORTED→THINKING for
    wall picking).
  - `attend_argmax_dot`: dot-product lookup (TEX_COL→RENDER for
    texture fetch).

- **MLP sublayers** implement nonlinear functions via
  `piecewise_linear` and `piecewise_linear_2d` approximations:
  trig lookups (cos, sin, tan over 256 entries), reciprocals
  (geometric-breakpoint grids for ~1% relative error), 2D products
  (coordinate rotations, perspective projection), and comparisons.

- **Linear layers** implement exact affine transforms: coordinate
  rotations, payload packing/unpacking, wall-top/bottom from
  wall-height, threshold one-hot construction.

Every cross-position dependency flows through attention. There is no
mechanism for data to travel between positions except through the
attention heads — the MLP and Linear layers operate position-wise.
This is what makes the graph a legitimate transformer, not just a
program that happens to run on GPU.

## Typical Dimensions

For the default configuration (8 walls, 320×240 screen, 8 textures of
64×64, 48 BSP nodes, chunk_size=20, d=2048):

| Parameter | Value |
|-----------|-------|
| d (residual stream width) | 2048 |
| d_head | 32 or 64 |
| Compiled layers | ~200-300 |
| Total parameters | ~80M (all deterministic, no training) |
| Prefill tokens | ~570 (512 TEX_COL + 58 game) |
| Sort tokens | 8 |
| Render tokens | ~250 (variable) |
| Wall-clock per frame (A100) | ~35ms |

## Key Design Decisions

**Why E8 spherical codes for token types?**
The 8-dimensional E8 lattice provides 240 unit vectors with large
pairwise distances. Using these as token-type identifiers means the
`equals_vector` check (a dot product against the known code) has
wide margin between match and non-match, making the comparison robust
to numerical noise in the residual stream.

**Why BSP-based sort instead of distance sort?**
BSP ranks are clean integers computed via a simple dot product —
no division, no distance calculation, no numerical ambiguity at
equal distances. The BSP tree guarantees a total order (no ties),
which is critical for the argmin attention to produce clean
one-hot-like selections. Distance-based sorting would require
comparing floating-point distances with potential ties at wall
intersections.

**Why a threshold-based sort instead of a mask?**
The sort_feedback carries one scalar (`prev_bsp_rank`) instead of a
max_walls-wide mask. This narrows the feedback vector and simplifies
the attention: instead of "find the smallest score among unmasked
positions," it's "find the smallest score above this threshold."
Exhaustion is detected by checking whether the selected rank exceeds
the threshold — no mask bookkeeping needed.

**Why THINKING tokens?**
A RENDER token doesn't know which wall it's rendering — that
information lives in the render_feedback overlay. THINKING exists to
select the next un-rendered wall and load its render data into the
feedback before the RENDER block begins. See
[design_feedback_elimination.md](design_feedback_elimination.md) for
a proposed alternative that encodes wall identity in the token type
and eliminates THINKING entirely.

**Why chunked rendering?**
A single RENDER token paints `chunk_size` rows (default 20). A wall
that is 60 rows tall at some column needs 3 RENDER tokens for that
column. This bounds the per-token output width at `chunk_size × 3`
RGB values. Without chunking, the worst case (a wall filling the
entire 240-row screen) would need a 720-wide output — eating residual
stream budget. Chunking trades token count for narrower outputs.

**Why is the host "dumb"?**
The host's only jobs are: (1) feed token inputs, (2) copy overlaid
outputs back as the next input, (3) bitblit pixel strips to the
framebuffer, (4) stop when `done > 0`. It performs no game logic,
no sorting, no rendering decisions. This constraint keeps the
transformer self-contained — the forward pass alone is the complete
game engine, and the host is a generic autoregressive inference loop
that could drive any graph, not just DOOM.
