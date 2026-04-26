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
TEX_COL×(num_tex × tex_w) →
INPUT → BSP_NODE×M → WALL×N → EOS →
PLAYER_X → PLAYER_Y → PLAYER_ANGLE →
[THINKING_WALL_n → (id → VALUE)×17]×N → (RESOLVED → VALUE)×3 →
[SORTED_WALL → SORT_RESULT → VALUE → RENDER×k]×N
```

These compile into five phases:

| Phase | Tokens | Mode | Typical count |
|-------|--------|------|---------------|
| -1 (Tex) | TEX_COL | Prefill | num_tex × tex_w (e.g. 512) |
| 0 (Prefill) | INPUT + BSP_NODE + WALL + EOS | Prefill | 1 + M + N + 1 (e.g. 58) |
| 1 (Player) | PLAYER_X + PLAYER_Y + PLAYER_ANGLE | Autoregressive | 3 |
| 2 (Thinking) | THINKING_WALL markers + per-wall identifier+VALUE pairs + RESOLVED identifier+VALUE pairs | Autoregressive | N×(1 + 17×2) + 3×2 (e.g. 286) |
| 3 (Sort+Render) | SORTED_WALL + SORT_RESULT id + VALUE + RENDER (interleaved) | Autoregressive | 3N + render columns (variable) |

Prefill tokens (phases -1 and 0) are processed in a single batched
forward pass. The two prefill broadcasts (INPUT → all positions
for velocity / trig / new angle, BSP → all positions for
`side_P_vec`) resolve internally through attention layers within
that pass — no autoregressive stepping needed. Autoregressive
tokens (phases 1+) are generated one at a time — each token's
output determines the next token's input.

The thinking phase is a sequential cascade: each per-wall block
emits 17 named scalar values (BSP rank, renderability, rotation
products, FOV-clip parameters, projected screen columns, collision
flags) as its own VALUE token. After all 8 walls have emitted, three
RESOLVED identifiers fire, each consuming the global hit aggregates
to produce post-collision player state. Sort and render then
interleave: a 3-position SORTED pipeline picks the next wall and
emits its index as a VALUE token, after which RENDER tokens paint
that wall's columns. When RENDER finishes the last column, it
emits the next SORTED_WALL marker as the next token, and the
transformer picks the next wall.

## Token Embedding (`W_EMBED`)

Every token is a single 16-bit integer ID. The host feeds
`token_ids` as a 1-wide input slot; the graph looks the ID up in
`W_EMBED` (a `(V, D_EMBED)` matrix, `V = 65576`, `D_EMBED = 49`)
to produce a 49-wide residual leaf at every position. On the
output side a 49-wide embedding emit lands in the residual stream
at every position, gets projected through `W_EMBED.T`, and is
argmaxed by the host wrapper to pick the next ID.

The 49 columns split into four blocks:

```
cols [ 0 :  8] — E8 category code           (per-category ±1 entries)
cols [ 8 :  9] — raw slot                   (Gray-code "x" coordinate for VALUE_k)
cols [ 9 : 25] — Gray-code payload          (16-wide ±1, Hamming-1 between adjacent k)
cols [25 : 26] — K slot                     (k for VALUE_k, k ≤ 255; else 0)
cols [26 : 47] — slot one-hot               (21-wide ±1, +1 at the row's own slot)
cols [47 : 48] — is_any_identifier          (±1)
cols [48 : 49] — is_value_category          (±1)
```

**E8 codes.** Each named category gets an 8-dimensional unit
vector from the E8 lattice (240 of them available). The codes have
large pairwise distances (self-dot 1600, cross-dot ≤ 800), which
makes `equals_vector` checks against a known code robust to
numerical noise. All 65,536 VALUE rows share one code, all 8
THINKING_WALL markers each get their own, every identifier gets
its own, and every prompt-position category gets its own.

**Raw slot + Gray code.** The 16-bit VALUE space is encoded
together: a continuous "x = (2k+1)/131072" raw position plus a
16-wide ±1 bit pattern that has Hamming distance 1 between
adjacent k. The producer (`encode_value_binary`) computes the bits
from triangle waves over x via two `piecewise_linear` sublayers
plus a packed `compare`-against-0.5 sublayer. The argmax against
`W_EMBED.T` resolves to the nearest VALUE_k.

**K slot.** For VALUE_k with k ≤ 255 (`MAX_INT_K`), the K column
holds the integer k literally. A consumer that wants to read back
an integer (today: SORT_RESULT carrying wall_index 0..7) does
`attend_most_recent_matching(value=K_column)` and the matched K
value *is* the integer — no decode Linear. This also bypasses a
calibration trap in the older raw-slot decode: a one-hot row
lookup put `raw = (2k+1)/131072` in the predicted embedding, which
the continuous-path decoder mapped to `≈ k / 9362` instead of `k`.

**Type-tag block (cols 26..49).** The slot one-hot is
`+1` at the row's own slot column for IDENTIFIER rows and `-1`
everywhere else (including non-identifier rows). `is_any_identifier`
is `+1` for the 21 identifier rows, `-1` elsewhere.
`is_value_category` is `+1` for all 65,536 VALUE rows, `-1`
elsewhere. Storing these as ±1 (not {0, 1}) means a per-position
direct extract IS the boolean — a `cond_gate` consumer doesn't
need a `bool_to_01` shim. Without this block the same flags would
take ~4–5 layers of `equals_vector` + `cond_gate` to construct
freshly at every consumer; with it they're a single `extract_from`
at depth 1.

## Identifiers (21 total)

The identifiers are the named scalars that flow through the
thinking phase and the sort phase. Each has its own vocab row and
its own column in the slot one-hot block.

**Per-wall (17):** `BSP_RANK`, `IS_RENDERABLE`, `CROSS_A`, `DOT_A`,
`CROSS_B`, `DOT_B`, `T_STAR_L`, `T_STAR_R`, `T_LO`, `T_HI`, `COL_A`,
`COL_B`, `VIS_LO`, `VIS_HI`, `HIT_FULL`, `HIT_X`, `HIT_Y`. Emitted
once per wall in this exact order.

**RESOLVED (3):** `RESOLVED_X`, `RESOLVED_Y`, `RESOLVED_ANGLE`.
Emitted once per frame after the last wall's `HIT_Y`.

**Sort phase (1):** `SORT_RESULT`. Emitted once per sort step at
the SORT_RESULT position, carrying the picked wall index.

Each identifier has a declared `(lo, hi)` float range in
`VALUE_RANGE_BY_NAME` (e.g. `BSP_RANK ∈ [0, 7]`,
`CROSS_A ∈ [-40, 40]`, `VIS_LO ∈ [-2, 122]`,
`SORT_RESULT ∈ [0, 7]`). The producer scales its float into
`[0, 65535]` via `quantize_to_range`; the host's argmax against
`W_EMBED.T` rounds to a uint16 token ID; the consumer scales back
via `dequantize_from_range`. The boundary contributes one
`(hi - lo) / (2·65535)` LSB of error per quantization round-trip.

## Token Types (E8 category codes)

Ten prompt-side and decode-side categories use distinct E8 codes;
the 21 identifiers and the 65,536 VALUE rows each get their own
codes too. The full assignment lives in `embedding._CATEGORY_INDEX`.

The prompt-position codes:

```
INPUT        BSP_NODE    WALL         EOS         TEX_COL
PLAYER_X     PLAYER_Y    PLAYER_ANGLE
SORTED_WALL  RENDER      DONE
```

Texture IDs map to E8 vectors at offset 8, so the same code space
covers token type and texture identity.

## Stages

### TEX_COL — Texture Data (Phase -1)

One token per column of each texture. Each carries the raw RGB
pixel data for that column (`tex_h × 3` floats), an E8 code
identifying which texture it belongs to, and a one-hot encoding of
its column index within that texture.

These tokens sit in the KV cache for the rest of the frame. When
RENDER tokens need texture pixels, they retrieve them via
`attend_argmax_dot` — a dot-product attention where the query
encodes (texture_id, column_index) and the matching TEX_COL
position's pixel data is returned as the value.

The stage's only computation is converting the host-fed column
index into a one-hot vector (`tc_onehot_01`) used as the attention
key.

### INPUT — Player Controls (Phase 0)

A single token. Receives the player's current angle and six
boolean movement flags (forward, backward, turn left, turn right,
strafe left, strafe right). Does not receive player position —
that goes in directly via `PLAYER_X`/`PLAYER_Y`.

Computes:

- **New angle**: `(old_angle + turn_right×speed - turn_left×speed) mod 256`.
  Angles are integers 0..255 (a full circle in 256 steps).
- **Velocity**: `(vel_dx, vel_dy)` from the new angle and movement
  flags. Forward/backward move along the facing direction;
  strafing moves perpendicular. Uses `piecewise_linear` lookups
  over a 256-entry trig table.
- **Trig values**: `(move_cos, move_sin)` of the new angle.

All five derived values are broadcast to every position via
`attend_mean_where`. Since exactly one position has `is_input=1`,
the "mean" is just that position's value — every subsequent token
can read the player's velocity and facing direction via attention.

### BSP_NODE — Spatial Classification (Phase 0)

M tokens (typically 48), one per splitting plane of the BSP tree.

Each token carries a normalized plane `(nx, ny, d)` and classifies
the player as FRONT or BACK:

```
side_P = sign(nx × player_x + ny × player_y + d)
```

The result (1 for FRONT, 0 for BACK) is spread into slot `i` of
an M-wide vector via the token's `bsp_node_id_onehot`. An
`attend_mean_where` over all BSP_NODE positions gathers these into
a shared `side_P_vec` — a binary vector available at every
position telling which side of every BSP plane the player is on.
The thinking-phase BSP_RANK identifier consumes `side_P_vec` to
compute each wall's BSP rank.

### WALL — Geometry Carrier (Phase 0)

N tokens (typically 8), one per wall segment. This stage is a thin
data carrier:

- Raw geometry (`ax`, `ay`, `bx`, `by`, `tex_id`,
  `wall_bsp_coeffs`, `wall_bsp_const`) is host-supplied at every
  WALL position and read directly by consumers via attention.
- The stage's only computation is `wall_index_neg_sq = -wall_index²`,
  the second channel of the quadratic-equality K used by every
  attention that picks a wall by index. The first channel
  (`wall_index` itself) is forwarded unchanged from the host
  input. One `square` MLP sublayer.

All collision detection, BSP rank computation, renderability
gating, FOV clipping, and screen-column projection that previous
designs did at the WALL token now happen inside the thinking
phase, with each result emitted as its own identifier-VALUE
pair.

### EOS — End of Prompt (Phase 0)

A single token marking the end of the prefill. EOS performs no
graph computation — it exists purely as a boundary marker so the
prefill batch has a known endpoint. The PLAYER tokens fire next.

### PLAYER_X / PLAYER_Y / PLAYER_ANGLE — State Broadcast (Phase 1)

Three tokens emitted before the thinking phase.

- **PLAYER_X**: broadcasts the pre-collision x position to all
  positions.
- **PLAYER_Y**: broadcasts the pre-collision y position to all
  positions.
- **PLAYER_ANGLE**: looks up `cos(θ)` and `sin(θ)` via
  `piecewise_linear` trig tables and broadcasts both.
  PLAYER_ANGLE also emits `THINKING_WALL_0` as its next-token
  prediction so the autoregressive loop steps directly into the
  thinking phase without a host nudge.

The host feeds the *pre-collision* `(x, y, angle)`; collision
resolution happens later in the thinking phase via the RESOLVED
identifiers, whose emitted values RENDER consumes via readback.

### THINKING_WALL — Per-Wall State Machine (Phase 2)

The bulk of the per-frame compute. For each wall index
`n ∈ [0, N)` the host emits a `THINKING_WALL_n` marker token, then
the transformer drives 17 alternating identifier-VALUE pairs:

```
THINKING_WALL_n
  BSP_RANK         VALUE(bsp_rank)        # integer 0..N-1
  IS_RENDERABLE    VALUE(is_renderable)   # 0 or 1
  CROSS_A          VALUE(cross_a)         # rotation product, [-40, 40]
  DOT_A            VALUE(dot_a)           #   "
  CROSS_B          VALUE(cross_b)
  DOT_B            VALUE(dot_b)
  T_STAR_L         VALUE(t_star_L)        # left-FOV clip parameter, [-2, 2]
  T_STAR_R         VALUE(t_star_R)        # right-FOV clip parameter
  T_LO             VALUE(t_lo)            # FOV clip min, [0, 1]
  T_HI             VALUE(t_hi)            # FOV clip max
  COL_A            VALUE(col_a)           # endpoint-A screen column, [-2, W+2]
  COL_B            VALUE(col_b)           #   "
  VIS_LO           VALUE(vis_lo)          # visible column min
  VIS_HI           VALUE(vis_hi)          # visible column max
  HIT_FULL         VALUE(hit_running_or)  # collision flag, running OR
  HIT_X            VALUE(hit_x_running_or)
  HIT_Y            VALUE(hit_y_running_or)
```

35 tokens per wall (1 marker + 17 identifier+VALUE pairs); with 8
walls that's 280 tokens. The transformer drives the marker→id
and id→VALUE next-token predictions; the host echoes them back
unchanged.

**Three primary cross-position reads at each identifier step:**

1. **Current wall identity.** `attend_most_recent_matching` fires
   on `THINKING_WALL_n` markers and returns the most recent
   marker's `n`. Determines which wall this position is computing
   for.
2. **Wall geometry.** A 2-wide quadratic-equality `attend_argmax_dot`
   keys on `(wall_index, -wall_index²)` against WALL prefill
   positions and reads the value block
   `(wall_ax, wall_ay, wall_bx, wall_by, wall_bsp_coeffs, wall_bsp_const)`.
   Same shape attention as in the SORTED stage; see "Quadratic-
   equality attention" below.
3. **Prior identifier values.** `ThinkingReadback.get_value_after_last(name)`
   runs `attend_most_recent_matching` against thinking-VALUE
   positions whose preceding identifier was `name`, returning the
   prior value's raw slot decoded back to a float. Lets a step
   read what an earlier identifier emitted (for example, `T_LO`
   reads `T_STAR_L` and `T_STAR_R` via this path instead of
   recomputing the FOV clip).

**Two classes of identifier computation:**

- **Base values** (BSP_RANK, IS_RENDERABLE, CROSS/DOT, HIT_*):
  derived from first principles using the attended wall geometry,
  the BSP `side_P_vec`, and the player pre-collision pose. Math
  is the same ray-segment intersection / atan projection / FOV
  clipping that older designs ran at the WALL token.
- **Derived values** (T_STAR_L/R, T_LO, T_HI, COL_A/B, VIS_LO,
  VIS_HI): read CROSS/DOT/T/COL from the KV cache via
  `ThinkingReadback` and apply the next layer of clip / project
  math. The derivation chain is split across slots so each step's
  own depth is small (~12 ops at the deepest slot).

**Running-OR HIT_* accumulators.** Each `HIT_FULL` thinking token
emits not just *this* wall's collision flag but the OR of it with
the previous `HIT_FULL` value read from the KV cache:

```
emitted = saturate(this_walls_flag + prev_HIT_FULL_value)
```

`saturate(x) = min(1, max(0, x))` clamps the OR result. By the
time wall N-1's `HIT_FULL` fires, the emitted value is the global
OR across all walls. Wall 0 reads zero from the empty cache, the
OR identity. Same structure for `HIT_X` and `HIT_Y`. The RESOLVED
identifiers later read these globals as scalars via a single
attention hop, with no cross-position aggregation needed.

**RESOLVED identifiers (frame boundary).** After the last wall's
`HIT_Y`, three tokens fire:

```
RESOLVED_X       VALUE(resolved_x)
RESOLVED_Y       VALUE(resolved_y)
RESOLVED_ANGLE   VALUE(resolved_angle)
```

Each reads the global aggregates and applies axis-separated wall
sliding:

```
any_hit_full = readback("HIT_FULL") > 0.5
any_hit_x    = readback("HIT_X")    > 0.5
any_hit_y    = readback("HIT_Y")    > 0.5

use_new_x = NOT(any_hit_full AND any_hit_x)   # block X only if both rays hit
use_new_y = NOT(any_hit_full AND any_hit_y)   # block Y only if both rays hit

resolved_x = select(use_new_x, player_x + vel_dx, player_x)
resolved_y = select(use_new_y, player_y + vel_dy, player_y)
```

`RESOLVED_ANGLE` is just the `INPUT` stage's post-turn angle —
collision doesn't rotate the player. RENDER reads
`resolved_x` / `resolved_y` later via the same readback machinery.

**Scale-find broadcast.** Independently of the per-wall identifier
cascade, THINKING_WALL also runs a one-shot scale-find pass: a
cross-WALL `attend_argmax_dot` reduction over per-wall
`max(|ax|, |ay|, |bx|, |by|)` produces `global_max_abs_coord`,
`log_inv_scale`, and `inv_scale` as broadcast scalars consumed by
RENDER's coord normalization. See "Coordinate Handling" below.

### SORTED — Front-to-Back Selection (Phase 3)

Each wall transition fires a 3-position pipeline:

```
SORTED_WALL marker → SORT_RESULT id → SORT_RESULT VALUE → RENDER...
```

The marker fires at the start of each new wall transition;
`wall_counter` (an autoregressive 1-wide overlay) tracks "how many
walls picked so far," starting at 0.

**SORT_RESULT id position — the quadratic-equality attention.** Goal:
pick the renderable wall whose `bsp_rank == wall_counter`.
Expand the squared distance:

```
score(key) = -(bsp_rank - N)² = -bsp_rank² + 2·N·bsp_rank - N²
```

The `-N²` term is query-only and falls out of softmax. The
remaining dot product splits into:

```
query at SORT_RESULT id  : [2N, 1]
key at BSP_RANK position : [bsp_rank, -bsp_rank²]   (renderable)
key elsewhere            : [-100, -1000]            (sentinel)
```

`match_gain = 20` over a unit score gap drives softmax
concentration above 0.999 on the matching wall. Sentinel keys
(non-renderable walls and non-BSP_RANK positions) score so far
below the matching wall that they fall out entirely. The
attention's V carries `[wall_index_onehot, bsp_rank_scalar]` — the
picked wall's identity one-hot (decoded to a scalar via a Linear
over `[0, 1, …, max_walls-1]`) plus its BSP rank (used post-hoc
for the exhaustion check).

**Exhaustion detection.** When `wall_counter` exceeds the number
of renderable walls, no wall's rank matches; softmax concentration
falls and the picked wall's rank disagrees with `wall_counter`.
The same attention also returns the picked rank; a post-hoc
`compare(picked_rank, wall_counter)` produces `sort_done`, which
the host reads from overflow and uses to terminate.

**SORT_RESULT VALUE position.** The picked wall_index is emitted
as a 16-bit VALUE token via `emit_integer_value_embedding`. The
predicted embedding carries the Gray code for `k = wall_index`;
the host's argmax against `W_EMBED.T` picks `VALUE_k`. When the
host feeds that token back, the embedding leaf at the SORT_RESULT
VALUE position is `W_EMBED[k]` — whose K column carries `k`
literally. A downstream attention reading the K column of the
most-recent SORT_RESULT VALUE position recovers `wall_index` as a
clean integer with no decode Linear.

The same position also reads `VIS_LO` for this wall via a 3-wide
content attention (composite key
`[is_vis_lo_value, value_wall_index_scalar, value_wall_index_neg_sq]`,
quadratic-equality on the wall index) and writes it to
`render_col` — RENDER's first column for the new wall.

### RENDER — Pixel Generation (Phase 3)

The pixel-producing token. Each paints `chunk_size` rows (default
20) of one screen column.

**Wall identity.** Read via
`readback.get_int_after_last("SORT_RESULT")`, which attends to the
most-recent SORT_RESULT VALUE position and returns the integer
wall_index from the K column. No overlay carry.

**Wall geometry.** A 2-wide quadratic-equality `attend_argmax_dot`
keys on `(wall_index, -wall_index²)` against WALL positions and
reads `(ax, ay, bx, by, tex_id)`. Same K shape as SORTED's
attention. Replaced an older 8-wide one-hot scheme that needed a
multi-layer `in_range` Q-prep cascade.

**vis_hi.** A 3-wide content attention against thinking VIS_HI
VALUE positions, keyed on
`(is_vis_hi_value, value_wall_index_scalar, value_wall_index_neg_sq)`.
Same quadratic-equality shape as the SORT_RESULT-VALUE vis_lo
read. (`vis_lo` itself was already written into `render_col` by
the SORT_RESULT VALUE step, so RENDER reads it directly from its
overlaid input.)

**Per-token pipeline:**

1. **Render precompute.** From raw geometry plus player
   `(resolved_x, resolved_y, cos, sin)` (post-collision position
   from the RESOLVED readbacks; cos/sin from the PLAYER_ANGLE
   broadcast), normalize the six coords (`sel_ax/ay/bx/by`,
   `player_x/y`) by the broadcast `log_inv_scale` (see "Coordinate
   Handling" above), then compute the rotation products `(sort_den,
   C, D, E, sort_num_t)` directly in normalized units. The eight
   `piecewise_linear_2d` multiplies use `NORM_DIFF_BP × TRIG_BP` (or
   `NORM_DIFF_BP × NORM_DIFF_BP` for the diff-diff cross product),
   giving denser cells near zero than the real-units `DIFF_BP` grid.
   The trig-product outputs scale linearly with `inv_scale`;
   `sort_num_t` (= `(b-a) × (a-p)`) scales with `inv_scale²` — the
   wall_height step (3) corrects with a log-domain combination, and
   the texture-column thermometer reads ratios that are
   scale-invariant.
2. **Active column.** The active column is `col` (an autoregressive
   overlay; SORT_RESULT VALUE seeds it to `vis_lo`).
3. **Wall height (log-decomp).** `tan(angle_offset)` via
   `piecewise_linear`, then:
   ```
   den_over_cos_norm = sort_den_norm - C_norm × tan(offset)
   log_wall_height   = log(H) + log(|den_over_cos_norm|)
                                + log_inv_scale
                                - log(|sort_num_t_norm|)
   wall_height       = clamp(exp(log_wall_height), 0, H)
   ```
   The math comes from `wall_height_real = H · |den_over_cos_real| /
   |sort_num_t_real|` after substituting the normalized values
   `|abs_den_norm| = inv_scale · |abs_den_real|` and
   `|abs_num_norm| = inv_scale² · |abs_num_real|`. Two parallel
   `log()` calls + a Linear sum + `exp()` keep every intermediate
   well-conditioned in float32 (each per-section log error ~3e-3
   relative); the older `reciprocal × multiply_2d` form would have
   multiplied two per-op piecewise errors and accumulated them at
   the divide-by-small-denominator regime.
4. **Texture column.** A thermometer comparison
   `tex_col = |{k : tex_w × |D + E·tan(offset)| ≥ k × abs_den}|`
   — no division, every threshold is a `multiply_const + subtract +
   compare`. The threshold compare pre-shifts by `0.5 · inv_scale`
   so the comparison stays at zero in normalized space.
5. **Texture fetch.** `attend_argmax_dot` retrieves pixels from
   the matching TEX_COL position. Query is
   `(tex_id_e8, scaled_col_onehot)`; key is
   `(texture_id_e8, scaled_tc_onehot_01)`.
6. **Chunk fill.** Paints `chunk_size` rows. For each row,
   computes the texture row index via
   `floor((y - wall_top) × tex_height / wall_height)`, extracts
   RGB via `dynamic_extract`, and composites over ceiling/floor
   colors using `in_range` masks.
7. **State transitions.** Three mutually exclusive cases:
   - **More chunks** (`active_start + chunk_size < wall_bottom`):
     stay on this column, advance `chunk_k` by 1. Next token type:
     `RENDER`.
   - **Advance column** (no more chunks, `col + 1 ≤ vis_hi`): move
     to the next column, reset `chunk_k` to 0. Next token type:
     `RENDER`.
   - **Advance wall** (no more chunks, no more columns): if
     `wall_counter ≥ max_walls`, set `done = +1`. Otherwise next
     token type: `SORTED_WALL`, so the transformer picks the next
     wall (and `wall_counter` increments at the SORTED_WALL → SORT_RESULT
     id step that follows).

**Headless mode.** Setting `render_pixels=False` at build time
skips steps 5 and 6 (the texture fetch and chunk fill) and emits
zero pixels. The state machine still runs identically. Used by
rollout tests that assert on tokens, not pixels.

## Coordinate Handling

The graph's piecewise-linear ops are defined over breakpoint grids
with bounded input ranges. Real DOOM maps put geometry thousands of
units from the WAD origin, and individual scenes can span a wide
range of scales. Two coordinate transformations land everything
inside the operating envelope before precision-sensitive ops fire:
a host-side translation and an in-graph per-frame scale
normalization.

### Host-side translation (`MapSubset.scene_origin`)

Real maps (E1M1 etc.) place geometry thousands of units from the
WAD origin; the player spawn might be at `(1056, -3616)`. The
graph's `max_coord` envelope (default 20, 100 for real-map
subsets) won't admit such inputs. Rather than widen `max_coord`
to ten thousand and lose breakpoint density everywhere, the host
carries a per-scene `scene_origin` tuple on `MapSubset` and
applies it as a translation at the host boundary:

* `step_frame` subtracts `scene_origin` from `wall_ax/ay/bx/by`,
  player `(px, py)`, and the BSP plane `d` (transformed via
  `d → d + nx·origin_x + ny·origin_y`) before each prefill /
  autoregressive feed.
* When reading `RESOLVED_X` / `RESOLVED_Y` back from overflow,
  `step_frame` adds `scene_origin` so the host's `GameState` stays
  in world coords.
* `load_map_subset` defaults `scene_origin` to the player spawn —
  a one-line heuristic that keeps the player and the closest walls
  inside the local envelope without per-frame recomputation.
* `build_scene_subset` (hand-authored scenes) defaults
  `scene_origin = (0, 0)`; its scenes are already centred near
  origin, so the shift is a no-op.

The graph never sees `scene_origin` — it's a host-side affine
constant baked at subset build time.

### In-graph scale normalization (`scale-find` + `normalize_coord`)

To keep precision-sensitive coord-coord multiplies inside a tight
piecewise-linear cell envelope (`NORM_DIFF_BP`, denser near zero
than `DIFF_BP`), the thinking phase runs a **scale-find pass**:

1. Each WALL position contributes its own `max(|ax|, |ay|, |bx|,
   |by|)` to a single cross-position `attend_argmax_dot` reduction
   whose query is a constant 1. The winning value is the global
   maximum coord magnitude across every wall in the scene.
2. `log_inv_scale = -log(global_max_abs_coord)` and `inv_scale =
   exp(log_inv_scale)` follow as 1-sublayer transformations.

The three scalars (`global_max_abs_coord`, `log_inv_scale`,
`inv_scale`) are broadcast across every position via the standard
KV-cache mechanism and live in `ThinkingWallOutput`. RENDER reads
`log_inv_scale` for `normalize_coord` calls and `inv_scale` for the
texture-column threshold shift.

`normalize_coord(coord, log_inv_scale)` decomposes `coord · inv_scale`
as `sign(coord) · exp(log|coord| + log_inv_scale)`. End-to-end
relative error is ~0.07 % over the operating envelope
(|coord| ∈ [0.1, 100], |coord · inv_scale| ≤ 1).

**Per-stage decision rule.** Normalization is applied per-stage,
not globally. A stage normalizes its coord inputs only if it has
an internal computation whose precision needs to be uniform across
scene scales (i.e. it multiplies coord by coord at a regime where
the piecewise-linear cell width matters relative to the result
magnitude). Stages without such a need stay in real units, because
normalization makes their inputs scale with `inv_scale^d` for some
degree `d > 0` — and once the inputs shrink below an op's
discrete-decision deadband, the op silently outputs the wrong
answer.

* **RENDER** normalizes (the wall_height log-decomp and the
  texture-column thermometer both need scale-invariant precision).
* **Thinking_wall** does *not* normalize. Its outputs are
  sign-bearing (HIT_*, IS_RENDERABLE), wire-format-clamped (CROSS,
  DOT, RESOLVED), or scale-invariant ratios (T_STAR, T_LO/HI,
  COL_A/B as `atan(cross/dot)·col_scale`). Normalizing
  `sort_num_t = (b-a) × (a-p)` would make it scale with
  `inv_scale²`, pushing the downstream `compare(adj_num_t, 0)`'s
  0.1-input-unit deadband to swallow nearly all valid geometry on
  large scenes.
* **SORTED** does *not* normalize. Its math operates on integer
  ranks for which `inv_scale` is meaningless.

The corollary: `compare(x, 0)` is not a scale-free sign operator —
it is a margin test with absolute tolerance `1/sharpness` in input
units. Sign extraction on a normalized degree-`d>0` quantity will
fail at scenes where `inv_scale^d` is small enough to push typical
inputs into the ramp deadband. See
`docs/numerical_noise_findings.md` for the worked example.

## Outputs

Every position produces two categories of output.

### Overlaid outputs (autoregressive feedback)

Land back at the matching input slot in the next step's residual
stream via delta transfer. The host copies each field verbatim.

| Field | Width | Producer | Notes |
|-------|-------|----------|-------|
| `render_col` | 1 | SORT_RESULT VALUE seeds to `vis_lo`; RENDER advances | Current screen column |
| `render_chunk_k` | 1 | SORT_RESULT VALUE resets to 0; RENDER increments | Chunk index in column |
| `wall_counter` | 1 | SORT_RESULT id increments; SORTED_WALL marker, SORT_RESULT VALUE, and RENDER all forward unchanged | Number of walls picked so far |

### Overflow outputs (host reads)

Placed after the input region in the residual stream. Read by
the host but not fed back as input.

| Field | Width | Notes |
|-------|-------|-------|
| `next_token_embedding` | 49 | Argmaxed against `W_EMBED.T` to pick the next token ID. Built by a flat `switch()` over six mutually-exclusive branches: SORT_RESULT id → factored VALUE(wall_index); SORT_RESULT VALUE → embed("RENDER"); thinking-active (excluding SORT_RESULT) → thinking_wall's cascade; SORTED_WALL marker → embed("SORT_RESULT"); RENDER → render's next-type; PLAYER_ANGLE → embed("THINKING_WALL_0"). |
| `pixels` | chunk_size × 3 | RGB strip for the current chunk |
| `col` | 1 | Screen column index of the current chunk |
| `start` | 1 | Screen row where the chunk begins |
| `length` | 1 | Number of rows in the chunk (0 for non-RENDER) |
| `done` | 1 | +1 when all walls fully rendered |
| `advance_wall` | 1 | +1 on wall transitions (informational) |
| `sort_done` | 1 | +1 at SORT_RESULT id when sort exhausted |
| `sort_wall_index` | 1 | Picked wall_index (host-visible, not fed back) |

The host's loop is trivial: step the transformer, read
`(pixels, col, start, length)` from overflow, bitblit the strip to
the framebuffer at `(col, start)` with skip-fill compositing,
copy overlaid outputs to the next input, argmax
`next_token_embedding` to pick the next token ID, and stop when
`done > 0` or `sort_done > 0`. The host never inspects token type,
never caches wall data, never patches inputs.

## Cross-Position Primitives

Every cross-position dependency in the graph compiles to a
specific attention shape:

- **`attend_mean_where`** — broadcasts. Used by INPUT (broadcast
  velocity / new angle / move trig), BSP (broadcast `side_P_vec`),
  and PLAYER_X / PLAYER_Y / PLAYER_ANGLE (broadcast pre-collision
  position and cos/sin).
- **`attend_argmax_dot`** — content-addressed lookup. Used by
  texture fetch (TEX_COL → RENDER), wall geometry attentions
  (WALL → thinking_wall and WALL → RENDER, both 2-wide
  quadratic-equality), and the SORT_RESULT id quadratic-equality
  attention (BSP_RANK thinking positions → SORT_RESULT id).
- **`attend_most_recent_matching`** — cache readback. Used by the
  thinking-phase prev-id channel, by `ThinkingReadback` for prior
  identifier values, by RENDER for `wall_index`, `resolved_x`,
  `resolved_y`, and by the various `vis_lo` / `vis_hi` content
  attentions.

The old `attend_argmin_above_integer` primitive is gone — every
"pick the smallest above threshold" pattern in the previous design
turned into a quadratic-equality dot product over the right
identity scalar.

**Quadratic-equality attention.** Given a scalar key `k_i` at each
position and a target `k_target` at the query, building K =
`[k_i, -k_i²]` and Q = `[2·k_target, 1]` makes the dot product
`-(k_i - k_target)² + k_target²`. The constant query-side term
falls out of softmax; the remaining score peaks at `k_i = k_target`
and falls off quadratically with distance. Used in five places:

- 2-wide pure form (`match_gain = 20`, unit score gap): the
  SORT_RESULT id BSP-rank match, and the two wall_index → WALL
  geometry matches (one fired from RENDER, one from
  thinking_wall).
- 3-wide composite form prefixed with a type-match bit
  (`match_gain = 12000`, score gap ≥ 2 between matching+matching-
  type and any other key): VIS_LO read at SORT_RESULT VALUE, VIS_HI
  read at RENDER. The type bit picks out thinking-VALUE positions
  whose preceding identifier was VIS_LO / VIS_HI; the quadratic
  pair picks the wall.

Both forms give softmax concentration above 0.99.

## Compilation

The graph compiles to a standard transformer via
`forward_compile` (called by `compile_game` / `compile_headless`):

- **Attention heads** implement the cross-position primitives
  above.
- **MLP sublayers** implement nonlinear functions via
  `piecewise_linear` and `piecewise_linear_2d` approximations:
  trig lookups (cos, sin, tan over 256 entries), reciprocals
  (geometric breakpoint grids for ~1% relative error), 2D
  products (rotations, perspective projection), comparisons, and
  the L1/L2 triangle-wave Gray-code encoder.
- **Linear layers** implement exact affine transforms: coordinate
  rotations, payload packing/unpacking, scaling, and threshold
  one-hot construction.

Every cross-position dependency flows through attention. There is
no mechanism for data to travel between positions except through
the attention heads — MLP and Linear layers operate position-wise.
The compile-time scheduler (CP-SAT, see `cpsat_scheduler.md`)
places every node into a layer subject to read-after-write and
column-allocation constraints.

## Typical Dimensions

For the default `compile_game` configuration (8 walls, 8 textures
of 64×64, 48 BSP nodes, chunk_size=20, d=2048):

| Parameter | Value |
|-----------|-------|
| `d` (residual stream width) | 2048 (3072 in the larger walkthrough config) |
| `d_head` | 32 or 64 |
| `D_EMBED` | 49 |
| `V` (vocab size) | 65576 |
| Prefill tokens | ~570 (512 TEX_COL + 58 game) |
| Player tokens | 3 |
| Thinking tokens | 286 (8 × 35 + 6) |
| Sort tokens (per frame) | 3 × 8 = 24 |
| Render tokens | variable (depends on screen size + wall visibility) |
| Compiled depth | varies with `d` and the scheduler; recent measurements land in the 50-70 layer range |

All parameters are deterministic — no training is involved.

## Key Design Decisions

**Why E8 spherical codes for category labels?**
The 8-dimensional E8 lattice provides 240 unit vectors with large
pairwise distances (self-dot 1600, cross-dot ≤ 800). Using these
as category identifiers makes the `equals_vector` check (a dot
product against the known code) have wide margin between match
and non-match, making the comparison robust to numerical noise in
the residual stream.

**Why a Gray code for VALUE rows?**
The 65,536 VALUE rows share a single E8 category code, so the
host's argmax against `W_EMBED.T` distinguishes them by the raw
slot and Gray-code payload alone. Adjacent VALUE_k rows have
Hamming distance 1 in the 16-wide Gray code (dot product 14
between adjacent rows vs. self-dot 16), giving a clean argmax
margin. The producer side computes the Gray code via two
`piecewise_linear` triangle-wave sublayers plus a packed
`compare` sublayer — three MLP sublayers total — and never has
to maintain a 65,536-row lookup table at the producer.

**Why a K column for small-cardinality integers?**
For `SORT_RESULT` (which carries `wall_index ∈ [0, 7]`), readback
needs the *integer* not just a float approximation of it. With
`K_column[VALUE_k] = k` for `k ≤ 255`, an attention with
`V = K_column` returns the integer literally — no decode Linear,
no calibration drift. The K column is also what the K-slot
readback uses to make wall_index round-trip cleanly through the
SORTED → RENDER hand-off.

**Why the type-tag block in the embedding?**
The slot one-hot, `is_any_identifier`, and `is_value_category`
columns let the readback chain skip the
`equals_vector` + `cond_gate` + `Concat` construction that
otherwise builds these flags at depth 4–5 fresh at each consumer.
Storing them as ±1 columns in `W_EMBED` makes them direct extracts
at depth 1, dropping the critical-path floor for every readback
attention.

**Why thinking tokens for per-wall computation?**
Earlier designs ran all per-wall math at the prefill WALL token
in a single deep computation (~62 ops), then packed the results
into a wide payload for downstream stages to retrieve via
attention. Splitting into autoregressive thinking tokens has
three wins: (1) the per-wall critical path drops to ~12 ops at the
deepest slot because each step reads its inputs from the KV cache
instead of recomputing from scratch; (2) cross-position
aggregation collapses to readbacks (e.g. global hit-aggregate via
running OR + single readback, replacing
`attend_mean_where` over WALL positions); (3) downstream
consumers (SORTED, RENDER) can read named scalar values via
content attention instead of unpacking a fixed payload.

**Why running-OR HIT_* accumulators?**
Cross-position aggregation via `attend_mean_where` followed by a
threshold compare produces a soft signal whose worst case bumps
into the per-op noise budget when many walls contribute. Folding
the OR into the autoregressive sequence — each `HIT_*` step emits
`saturate(this_flag + prev_value)` — keeps the running aggregate
in 0/1 territory at every step (saturation clamps drift) and
reduces the consumer's read to a single readback hop.

**Why quadratic-equality attention for integer-keyed lookups?**
Earlier "pick the smallest above threshold" mechanisms used the
`attend_argmin_above_integer` primitive over a thermometer-coded
key. That worked but required 8-wide one-hot Q (or
8-wide thermometer K) and was sensitive to softmax dilution when
many candidate keys had similar scores. The quadratic-equality
identity `-(k - k_target)² = -k² + 2·k_target·k - k_target²`
collapses the same lookup to a 2-wide K and Q with a clean
gap-of-1 between adjacent integer keys. Saves residual-stream
width at the consumer and gives sharper softmax concentration.

**Why SORT_RESULT as its own VALUE token?**
Earlier designs piped wall identity from SORTED into RENDER via
overlaid outputs (`render_wall_j_onehot`, `render_vis_lo`,
`render_vis_hi`, `render_tex_id`). Emitting wall_index as a
SORT_RESULT VALUE token instead lets RENDER read it via the same
KV-cache readback every other thinking-token consumer uses, with
no special-case overlay machinery and no per-RENDER-token forward
of stale wall identity.

**Why does RENDER recompute wall geometry instead of reading
precomputed values?**
The per-wall rotation products `(sort_den, C, D, E, H_inv)`
depend on player position, which only becomes resolved
(post-collision) at the RESOLVED step in the thinking phase. By
the time RENDER fires, recomputing from raw `(ax, ay, bx, by)` +
`(resolved_x, resolved_y, cos, sin)` is cheaper than threading
the products through the thinking phase as additional VALUE
tokens. Raw geometry is read via the 2-wide quadratic-equality
wall_index attention; player state is read via the existing
PLAYER broadcast and RESOLVED readbacks.

**Why a simple counter (`wall_counter`) instead of feeding back the selected rank?**
Each SORTED step's threshold is `wall_counter`, an integer that
increments by 1 at each SORT_RESULT id step. The SORT_RESULT id
position picks the wall whose `bsp_rank == wall_counter`. The
alternative — feeding the picked rank back as the next threshold
— would require the attention to produce a clean integer rank
under piecewise-linear approximation noise, then use that noisy
value as the next threshold. The counter approach avoids
compounding attention noise through the sort sequence.
Exhaustion is detected by comparing `wall_counter` against the
selected wall's BSP rank.

**Why is the host "dumb"?**
The host's only jobs are: (1) feed token inputs, (2) copy overlaid
outputs back as the next input, (3) bitblit pixel strips to the
framebuffer, (4) argmax `next_token_embedding` against
`W_EMBED.T` to pick the next token ID, (5) stop when `done > 0`.
It performs no game logic, no sorting, no rendering decisions.
This constraint keeps the transformer self-contained — the
forward pass alone is the complete game engine, and the host is
a generic autoregressive inference loop that could drive any
graph, not just DOOM.
