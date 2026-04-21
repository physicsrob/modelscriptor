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
EOS → PLAYER_X → PLAYER_Y → PLAYER_ANGLE →
[SORTED_WALL → RENDER×k]×N  (interleaved sort + render)
```

These compile into four phases:

| Phase | Tokens | Mode | Typical count |
|-------|--------|------|---------------|
| -1 (Tex) | TEX_COL | Prefill | num_tex × tex_w (e.g. 512) |
| 0 (Prefill) | INPUT + BSP_NODE + WALL + EOS | Prefill | 1 + M + N + 1 (e.g. 58) |
| 0b (Player) | PLAYER_X + PLAYER_Y + PLAYER_ANGLE | Autoregressive | 3 |
| 1+2 (Sort+Render) | SORTED_WALL + RENDER (interleaved) | Autoregressive | N + dynamic (~258) |

Prefill tokens are processed in a single batched forward pass.
Dependencies between them (WALL reads INPUT's broadcasts, WALL reads
BSP's side decisions) are resolved internally through attention layers
within that pass — no autoregressive stepping needed. Autoregressive
tokens are generated one at a time — each token's output determines
the next token's input.

Sort and render are interleaved: a SORTED_WALL token picks the next
closest wall and sets up its identity as feedback, then RENDER tokens
paint that wall's columns.  When a RENDER token finishes the last
column, it emits ``token_type = E8_SORTED_WALL`` so the transformer
picks the next wall.  The host just copies overlaid outputs to the
next input — it never inspects token types or patches wall identity.

## Token Types

Ten token types, identified by E8 spherical codes (8-dimensional unit
vectors from the E8 lattice):

```
TOKEN_INPUT          (0)    TOKEN_TEX_COL        (5)
TOKEN_WALL           (1)    TOKEN_BSP_NODE       (7)
TOKEN_EOS            (2)    TOKEN_PLAYER_X     (240)
TOKEN_SORTED_WALL    (3)    TOKEN_PLAYER_Y     (241)
TOKEN_RENDER         (4)    TOKEN_PLAYER_ANGLE (242)
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

A single token. Receives the player's current angle and six boolean
movement flags (forward, backward, turn left, turn right, strafe
left, strafe right). Does not receive player position — that is
consumed by BSP, WALL, and EOS directly.

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
the player's velocity and facing direction via attention.

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

### WALL — Geometry + Physics (Phase 0)

N tokens (typically 8), one per wall segment. The workhorse of the
prefill phase. Each WALL token performs several computations on its
wall segment, then packs the results into a payload for the sort
stage:

**1. Collision detection.**
Three ray-segment intersection tests: the player's full velocity
vector `(vel_dx, vel_dy)`, an x-only ray `(vel_dx, 0)`, and a y-only
ray `(0, vel_dy)`. Each produces a hit/miss flag (±1). Uses
`piecewise_linear_2d` products to compute parametric intersection
coordinates and validity checks.

The axis-separated rays enable wall sliding in the EOS stage: if the
full velocity hits a wall but the x-only ray doesn't, the player can
still slide along the x axis (see EOS below).

**2. BSP rank.**
`rank = dot(wall_bsp_coeffs, side_P_vec) + wall_bsp_const` — a dot
product of host-precomputed coefficients against the BSP side
decisions. Produces a clean integer in 0..N-1 that gives the
front-to-back ordering of walls from the player's current position.
The BSP tree structure guarantees these ranks are a permutation — no
ties — which is critical for the sort stage's argmin attention to
produce clean selections.

**3. Visibility column range.**
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

**4. Payload packing.**
All per-wall data is concatenated into a single `sort_value` vector
(8 + max_walls wide) for the SORTED stage to retrieve via attention.
Layout (from `wall_payload.py`):

```
[0..5)    wall geometry       (ax, ay, bx, by, tex_id)
[5..6)    bsp_rank            (the sort score)
[6..8)    visibility columns  (vis_lo, vis_hi)
[8..8+N)  position_onehot     (wall identity one-hot)
```

**Indicators_above.**
A max_walls-wide thermometer vector where slot `c` is 1 iff
`bsp_rank >= c` AND the wall is renderable. This is the key-side
basis for the SORTED stage's threshold-based argmin attention.

### EOS — Collision Resolution (Phase 0)

A single token marking the end of prefill.

Aggregates the per-WALL hit flags via `attend_mean_where` over WALL
positions. If the averaged flag for any ray exceeds a threshold
(0.05), at least one wall was hit. Applies axis-separated wall
sliding — movement on an axis is blocked only if **both** the full
velocity ray and that axis's solo ray hit a wall:

- X-axis: blocked only if both the full ray and the x-only ray hit.
- Y-axis: blocked only if both the full ray and the y-only ray hit.

This is what enables wall sliding: if the player walks diagonally
into a wall, the full velocity ray hits, but the perpendicular
axis's solo ray may miss, allowing the player to slide along the
wall on that axis.

The resolved `(x, y)` is emitted as overflow outputs
(`eos_resolved_x`, `eos_resolved_y`). The new angle (unchanged by
collision — EOS only resolves position) is emitted separately as
`eos_new_angle`. The host reads all three and feeds them to the
PLAYER tokens in Phase 0b.

### PLAYER — State Broadcast (Phase 0b)

Three tokens, emitted after EOS and before SORTED:

- **PLAYER_X**: broadcasts the resolved x position to all positions.
- **PLAYER_Y**: broadcasts the resolved y position to all positions.
- **PLAYER_ANGLE**: looks up `cos(θ)` and `sin(θ)` via
  `piecewise_linear` trig tables and broadcasts both to all positions.

The host feeds the resolved player state (read from EOS overflow
outputs) as the `player_x`, `player_y`, `player_angle` inputs at
these three positions. The `attend_mean_where` broadcasts land in
the KV cache, so all downstream tokens (SORTED, RENDER) can read the
post-collision player position and trig values via attention.

### SORTED_WALL — Front-to-Back Sort

Interleaved with RENDER — one SORTED_WALL token runs before each
wall's RENDER tokens. Each picks the next-closest wall in BSP rank
order.

The attention primitive is `attend_argmin_above_integer`: given a
threshold one-hot (derived from the feedback-driven
`sort_position_index`), it finds the WALL position with the smallest
BSP rank strictly exceeding that threshold. The key-side
`indicators_above` thermometer encodes both the rank comparison and
the renderability gate — non-renderable walls have all-zero indicators
and are invisible to the attention.

Each SORTED token:

1. Converts `sort_position_index` to a threshold one-hot (clamped to
   `[0, max_walls-1]` for exhaustion safety).
2. Runs `attend_argmin_above_integer` over WALL positions, retrieving
   the packed payload of the winning wall.
3. Unpacks the payload into visibility columns, wall identity one-hot,
   texture ID, and BSP rank.
4. Detects sort exhaustion: if `sort_position_index > sel_bsp_rank`,
   the attention averaged garbage (no valid keys above threshold).
   Emits `sort_done = +1` so the host can stop early.
5. Outputs `sort_position_index + 1` so the next SORTED token picks
   the next wall in rank order.

The threshold advances by 1 each step: SORTED outputs
`position_index + 1`, RENDER forwards it unchanged.

**Feedback to RENDER.** At each SORTED position, the overlaid outputs
carry the selected wall's identity (`sel_onehot`, `vis_lo`, `vis_hi`,
`sel_tex_id`) plus `render_col = vis_lo` and `render_chunk_k = 0`.
The host copies these to the next token's input via standard feedback
— no caching, no conditional logic.  The SORTED token also outputs
`token_type = E8_RENDER` so the next token runs the RENDER stage.

### RENDER — Pixel Generation

The pixel-producing token.  Each paints a vertical chunk of one screen
column.  Wall identity comes from discrete overlaid inputs
(`render_wall_j_onehot`, `render_tex_id`, `render_vis_lo`,
`render_vis_hi`) set by the preceding SORTED_WALL token and forwarded
unchanged by RENDER tokens within the same wall.

**Per-token computation:**

1. **Wall geometry attention.** Attend to the WALL position matching
   `render_wall_j_onehot` and read raw geometry `(ax, ay, bx, by)`.

2. **Render precomputation.** From raw geometry plus player state
   (position and cos/sin from PLAYER broadcasts), compute the rotation
   products `(sort_den, C, D, E, sort_num_t)` and derive `H_inv`.
   These are the same quantities the WALL stage used to compute in
   earlier designs — now they're computed fresh at each RENDER token
   from the raw wall coordinates and player state.

3. **Active column.** The active column is `render_col`.  SORTED sets
   this to `vis_lo` for each new wall; the state machine advances it
   on column transitions.

4. **Angle offset.** `angle_offset = (col × fov / W) - fov/2` — the
   horizontal angle of this column relative to the screen center, in
   trig-table steps (0..255 units).

5. **Wall height.** `tan(angle_offset)` via piecewise_linear, then:
   ```
   den_over_cos = sort_den - C × tan(offset)
   wall_height  = H_inv × |den_over_cos|
   ```
   The first line is the per-column horizontal projection factor. The
   second scales by the wall's precomputed height reciprocal. Uses
   `piecewise_linear_2d` with log-spaced breakpoints for `H_inv`
   (values span 0.01 to ~H/0.3).

6. **Texture column.** Determines which column of the wall's texture
   maps to this screen column:
   ```
   abs_nuc = |D + E × tan(offset)|
   tex_col = |{k : tex_w × abs_nuc >= k × abs_den}|
   ```
   This is a thermometer comparison — no division needed. Each
   threshold `k` is an exact `multiply_const` + `subtract` +
   `compare`, so the only approximation error comes from the upstream
   `piecewise_linear_2d` that produced `abs_nuc`.

7. **Texture fetch.** `attend_argmax_dot` retrieves the pixel data
   from the matching TEX_COL position in the KV cache. The query
   is `(tex_id_e8, scaled_col_onehot)` — the E8 code of the
   texture (looked up via piecewise_linear from the integer tex_id)
   concatenated with a scaled one-hot of the texture column. The
   TEX_COL key is `(texture_id_e8, scaled_tc_onehot_01)`. The
   dot product is largest at the matching (texture, column) pair.

8. **Chunk fill.** Paints `chunk_size` rows (default 20) of the
   column. For each row, computes the texture row index via
   `floor((y - wall_top) × tex_height / wall_height)`, extracts the
   RGB from the texture column via `dynamic_extract`, and composites
   over ceiling/floor colors using `in_range` masks.

9. **State transitions.** Three mutually exclusive cases:
   - **More chunks**: `active_start + chunk_size < wall_bottom` —
     stay on this column, advance `chunk_k` by 1.
     Next token type: `E8_RENDER`.
   - **Advance column**: no more chunks, `active_col + 1 <= vis_hi`
     — move to the next column, reset `chunk_k` to 0.
     Next token type: `E8_RENDER`.
   - **Advance wall**: no more chunks, no more columns — add
     `render_wall_j_onehot` to `render_mask`.  If all walls masked,
     set `done = +1`.  Otherwise set next token type to
     `E8_SORTED_WALL` so the transformer picks the next wall.
     (`sort_position_index` is forwarded unchanged — SORTED
     increments it on its own turn.)

## Outputs

Every position produces two categories of output:

### Overlaid Outputs (Autoregressive Feedback)

Fed back as the next token's input via delta transfer — the output
lands at the same residual-stream columns as the corresponding input.
The host copies each overlaid field verbatim from output to input.

- **token_type** (8-wide): E8 code for the next token type.  SORTED
  outputs `E8_RENDER` (always followed by RENDER).  RENDER outputs
  `E8_SORTED_WALL` on wall transitions (not done), `E8_RENDER`
  otherwise.  The host copies this to the next input without
  inspecting it.

- **render_mask** (max_walls-wide): Walls fully rendered so far.
  RENDER updates this on wall transitions by adding the current
  wall's one-hot.  SORTED forwards it unchanged.

- **sort_position_index** (1): Which sorted wall we're on (0-indexed).
  SORTED outputs `position_index + 1`; RENDER forwards it unchanged.

- **render_col** (1): Current screen column. SORTED seeds this to
  `vis_lo`; RENDER advances it via the state machine.

- **render_chunk_k** (1): Chunk index within the current column
  (0, 1, 2, ...). Reset to 0 on column and wall transitions.

- **render_tex_id** (1): Texture ID of the wall being rendered.
  SORTED seeds this from the wall payload; RENDER forwards it.

- **render_vis_lo**, **render_vis_hi** (1 each): Screen-column
  visibility bounds. SORTED seeds from the wall payload; RENDER
  forwards.

- **render_wall_j_onehot** (max_walls-wide): One-hot identifying
  which wall is currently being rendered. SORTED seeds from the
  wall payload; RENDER forwards. Used by the wall geometry
  attention to read the correct WALL position's raw coordinates.

### Overflow Outputs (Host Reads)

Placed after the input region in the residual stream. The host reads
these directly:

- **pixels** (chunk_size × 3 wide): RGB values for the current chunk.
- **col** (1): Screen column index.
- **start** (1): Screen row where this chunk begins.
- **length** (1): Number of rows painted (0 for non-RENDER tokens).
- **done** (1): +1 when all walls are fully rendered, -1 otherwise.
- **advance_wall** (1): +1 when transitioning to the next wall, -1
  otherwise.  Informational only — the host does not need this because
  wall transitions are driven by the `token_type` overlaid output.
- **sort_done** (1): +1 when the sort has exhausted all renderable
  walls at this SORTED position, -1 otherwise.
- **eos_resolved_x**, **eos_resolved_y** (1 each): Collision-resolved
  player position. The host reads these from the EOS token's output
  and feeds them to the PLAYER tokens.
- **eos_new_angle** (1): Post-input new angle (from the INPUT stage,
  not computed by EOS). The host feeds this to PLAYER_ANGLE.

The host's rendering loop is trivial: step the transformer, read
`(pixels, col, start, length)` from overflow output, bitblit the
pixel strip to the framebuffer at `(col, start)` with skip-fill
compositing (don't overwrite already-filled pixels), copy overlaid
outputs to the next input, and stop when `done > 0` or
`sort_done > 0`.  The host never inspects token type, never caches
wall data, never patches inputs.

## How the Graph Becomes a Transformer

The computational graph compiles to a standard transformer via
`compile_game` (which calls `compile_headless` internally):

- **Attention heads** implement cross-position data flows:
  - `attend_mean_where`: broadcast (INPUT→all, BSP→all, PLAYER→all,
    WALL→EOS for collision aggregation).
  - `attend_argmin_above_integer`: threshold-based selection
    (WALL→SORTED for front-to-back sort).
  - `attend_argmax_dot`: dot-product lookup (TEX_COL→RENDER for
    texture fetch, WALL→RENDER for wall geometry).

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

For the default `compile_game` configuration (8 walls, 8 textures of
64×64, 48 BSP nodes, chunk_size=20, d=2048):

| Parameter | Value |
|-----------|-------|
| d (residual stream width) | 2048 |
| d_head | 32 or 64 |
| Prefill tokens | ~570 (512 TEX_COL + 58 game) |
| Player tokens | 3 |
| Sort tokens | 8 |
| Render tokens | variable (depends on screen size + wall visibility) |

Layer count and parameters depend on the graph configuration and
optimization passes. All parameters are deterministic — no training
is involved.

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

**Why a simple counter threshold instead of feeding back the selected rank?**
Each SORTED token's threshold is `sort_position_index` (0, 1, 2, ...),
incremented by 1 at each SORTED step. The token at step `i` picks the
wall with the `i`-th smallest BSP rank among renderable walls. The
alternative would be feeding back the selected wall's actual BSP rank
as the next threshold — but that would require the attention to
produce a clean integer rank under piecewise-linear approximation
noise, then use that noisy value as the next threshold. The counter
approach avoids compounding attention noise through the sort sequence.
Exhaustion (fewer renderable walls than sort positions) is detected by
comparing `sort_position_index` against the selected wall's BSP rank.

**Why PLAYER tokens instead of EOS broadcasting state?**
The EOS token resolves the collision and emits the new `(x, y, angle)`
as overflow outputs. Rather than broadcasting from EOS (which would
mix collision intermediates into the KV cache row that RENDER reads),
three dedicated PLAYER tokens broadcast one value each. The host reads
the resolved state from EOS overflow and feeds it as inputs to
PLAYER_X, PLAYER_Y, PLAYER_ANGLE. Each `attend_mean_where` broadcast
gives every downstream position clean access to the resolved player
state.

**Why does RENDER recompute wall geometry instead of reading precomputed values?**
Earlier designs precomputed `(sort_den, C, D, E, H_inv)` at WALL
positions and packed them into the sort payload for SORTED to forward
to RENDER via feedback. The current design has RENDER read raw
`(ax, ay, bx, by)` from WALL via a dot-product attention on
`render_wall_j_onehot`, then compute the rotation products fresh using
player state from the PLAYER broadcasts. This eliminated the
`render_feedback` vector and the THINKING token type that existed to
load wall data into that feedback. The trade-off is more compute per
RENDER token in exchange for a simpler, feedback-free data flow.

**Why chunked rendering?**
A single RENDER token paints `chunk_size` rows (default 20). A wall
that is 60 rows tall at some column needs 3 RENDER tokens for that
column. This bounds the per-token output width at `chunk_size × 3`
RGB values. Without chunking, the worst case (a wall filling the
entire screen height) would need a very wide output — eating residual
stream budget. Chunking trades token count for narrower outputs.

**Why is the host "dumb"?**
The host's only jobs are: (1) feed token inputs, (2) copy overlaid
outputs back as the next input, (3) bitblit pixel strips to the
framebuffer, (4) stop when `done > 0`. It performs no game logic,
no sorting, no rendering decisions. This constraint keeps the
transformer self-contained — the forward pass alone is the complete
game engine, and the host is a generic autoregressive inference loop
that could drive any graph, not just DOOM.
