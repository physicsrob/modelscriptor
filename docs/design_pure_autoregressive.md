# Pure Autoregressive Game Graph

## Status

Design document. Supersedes
[design_feedback_elimination.md](design_feedback_elimination.md).

## Motivation

The current game graph carries two feedback vectors — `sort_feedback`
(16+ wide) and `render_feedback` (27+ wide) — that smuggle internal
state machine state from one token to the next via the input overlay.
This has no analog in how language models work. An LLM's context is its
previous outputs (the KV cache) and the current token's identity.
Hidden side channels are an implementation artifact, not an
architectural necessity.

This document proposes eliminating both feedback vectors so that every
token's computation is a pure function of:

1. Its **type and identity** (discrete integers and one-hots: which
   token am I, which wall, which column, which chunk).
2. **Host-provided ground truth** (player state, wall geometry,
   textures — on WAD-representing tokens only).
3. **Attention over previous tokens' outputs** (the KV cache).

No side channels. The result is a game graph that is structurally
identical to how a normal autoregressive transformer generates tokens.

## Current Architecture

```
EOS → SORTED×N → (THINKING → RENDER×k)×N
        ↑ sort_feedback    ↑ render_feedback (27 values)
        └─ prev_bsp_rank   └─ wall identity, precomputes,
           sel_onehot         mask, column, chunk position
```

THINKING exists because RENDER doesn't know which wall it's rendering.
The render_feedback vector carries that identity plus the selected
wall's precomputed render parameters plus the column/chunk state
machine position. The sort_feedback carries the previous BSP rank
threshold so each SORTED token can find the next wall.

## Proposed Architecture

### Token Sequence

```
Prefill (textures):  TEX_COL × (num_tex × tex_w)
Prefill (world):     INPUT + BSP_NODE×M + WALL×N + EOS
Auto (player):       PLAYER_X → PLAYER_Y → PLAYER_ANGLE
Auto (sort):         SORTED×N
Auto (render):       RENDER_{wall_j, col_c, chunk_k} × ...
```

The prefill phases are unchanged from today. The autoregressive phase
replaces the current `SORTED → (THINKING → RENDER×k)×N` with three
sub-phases: player state emission, sorting, and rendering. THINKING
tokens are eliminated entirely.

### Discrete player state tokens

After EOS resolves collisions, it emits three tokens that carry the
resolved player state as discrete values in their token types:

- **PLAYER_X_{x_int}**: quantized x position on a 64k grid (grid
  spacing ≈ 0.0003 on a ±10 coordinate range — well below the
  piecewise-linear approximation error).
- **PLAYER_Y_{y_int}**: quantized y position, same grid.
- **PLAYER_ANGLE_{θ}**: the discrete angle (integer 0..255, already
  discrete in the current design).

These tokens sit in the KV cache for the rest of the frame. The exact
mechanism for how downstream tokens use these values is a design
decision deferred to implementation — the MVP can start with float
conversion and broadcast (`attend_mean_where`), with room to optimize.
The key architectural property: discrete player state is available in
the KV cache before sorting or rendering begins.

### Position-derived sort threshold

Each SORTED token derives its threshold from its position in the
sequence rather than from the previous token's `prev_bsp_rank`
output. SORTED token at position i uses threshold i − 1:

- SORTED[0]: threshold −1 → picks any renderable wall (rank 0)
- SORTED[1]: threshold 0 → picks rank 1
- SORTED[i]: threshold i − 1 → picks rank i

This eliminates the `sort_feedback` vector entirely. The `prev_bsp_rank`
field is replaced by a position-derived constant. Each SORTED token's
computation depends only on the WALL positions in the KV cache and its
own position — no dependency on other SORTED tokens' outputs.

The host emits N SORTED tokens sequentially, each carrying its position
index. No feedback copying needed.

### Richly-typed RENDER tokens

Each RENDER token encodes its full identity as discrete values:

```
token_type = [E8_RENDER, wall_j_onehot, col_c, chunk_k, ...]
```

- **wall_j**: which wall (one-hot, max_walls wide). Determines which
  WALL position to attend to for geometry.
- **col_c**: which screen column (integer). Determines the per-column
  angle offset — and critically, `tan(angle_offset)` becomes a
  **compile-time constant** (see below).
- **chunk_k**: which vertical chunk within this column (integer).
  Determines `active_start = wall_top + chunk_k × chunk_size`.

Additional discrete values may be carried through the token type
by the host (the host copies the full token_type vector from output
to input):

- **tex_id**: integer texture index (0..num_tex).
- **vis_lo, vis_hi**: integer column bounds for this wall's visible
  range.

The state machine decides what comes next. When chunk_k reaches the
last chunk for the current column, the output token type advances to
the next column or the next wall:

```
RENDER_{wall_3, col_15, chunk_2}
  → RENDER_{wall_3, col_15, chunk_3}    (more chunks)
  → RENDER_{wall_3, col_16, chunk_0}    (advance column)
  → RENDER_{wall_4, col_lo_4, chunk_0}  (advance wall)
```

When the last wall's last column's last chunk finishes, the token
emits `done = +1`.

### What happens to render_feedback

Every field is eliminated:

| Field | Current width | Disposition |
|-------|--------------|-------------|
| render_mask | max_walls | Gone — wall ordering is in the token sequence |
| fb_sort_den, fb_C, fb_D, fb_E, fb_H_inv | 5 | Gone — RENDER computes from raw geometry |
| fb_tex_id | 1 | Gone — discrete value in token type |
| fb_col_lo, fb_col_hi | 2 | Gone — discrete values in token type |
| fb_onehot | max_walls | Gone — wall identity in token type |
| render_col | 1 | Gone — col_c in token type |
| render_chunk_start | 1 | Gone — derived from chunk_k in token type |
| render_is_new_wall | 1 | Gone — derivable from token type transition |

### What happens to sort_feedback

The sort_feedback vector is eliminated entirely. Its load-bearing
field (`prev_bsp_rank` at offset 13) is replaced by the
position-derived threshold. The wall data fields (geometry, visibility
bounds, one-hot) that currently travel through the sort_feedback
into THINKING are no longer needed — RENDER attends directly to
WALL and SORTED positions.

### RENDER computes precomputes from raw geometry

In the current design, RENDER receives wall precomputes (sort_den, C,
D, E, H_inv) via the render_feedback overlay — values that were
computed at the WALL stage during prefill and carried through SORTED
and THINKING.

In the new design, RENDER computes these values itself:

1. Attends to WALL_j for raw geometry (ax, ay, bx, by) — available in
   the KV cache from layer 0 (host-fed at WALL positions during
   prefill).
2. Reads cos(θ), sin(θ) from PLAYER_ANGLE's KV cache — available from
   layer ~4 (computed during PLAYER_ANGLE's autoregressive pass).
3. Computes the rotation products (ey·cos, ex·sin, fx·sin, gy·cos,
   etc.) via `piecewise_linear_2d`.
4. Derives sort_den, C, D, E, sort_num_t from the products.

This is the same math the WALL stage currently performs, but starting
~10 layers earlier because RENDER reads trig values from PLAYER_ANGLE
(available at layer ~4) rather than from the INPUT broadcast (available
at layer ~14 during prefill). The WALL stage's precompute computation
(C, D, E, H_inv) can be removed from the WALL stage entirely — WALL
only needs sort_den and sort_num_t for BSP rank and renderability.

### Key optimizations

**tan(angle_offset) is a compile-time constant.** With col_c in the
token type, `angle_offset = col_c × fov / W − fov/2` is a
compile-time constant per column. Therefore `tan(angle_offset)` is
also a compile-time constant. This eliminates three expensive
operations from RENDER's critical path:

- The `piecewise_linear` tan lookup (~4 ops) — replaced by a literal.
- The `piecewise_linear_2d` for `C · tan(offset)` in `den_over_cos`
  — becomes `multiply_const` (one linear op).
- The `piecewise_linear_2d` for `E · tan(offset)` in `tex_coord`
  — becomes `multiply_const` (one linear op).

This saves ~10 ops on RENDER's critical path and replaces approximate
2D products with exact linear operations.

**Integer wall_height via thermometer_floor_div.** Instead of computing
`H_inv = H / |sort_num_t|` (which requires a reciprocal chain of ~5
ops) and then `wall_height = H_inv × |den_over_cos|` (another
`piecewise_linear_2d`), we compute wall_height directly as an integer:

```
wall_height_int = floor(H × |den_over_cos| / |sort_num_t|)
```

Using `thermometer_floor_div` with max_value=H (screen height). This
is ~4 ops of depth (H parallel comparisons) and eliminates both the
reciprocal and the H_inv multiplication. The result is exact to ±0.5
pixels — invisible at screen resolution.

This eliminates H_inv entirely from the graph. The WALL stage no longer
needs to compute or carry it, and the sort_value payload shrinks.

## Depth Analysis

### Current

The compiled transformer is 70 layers. RENDER spans 67 of those
layers and is the depth driver. The critical path through the full
graph (41 real ops through wall/visibility → sort → output) is shorter,
but RENDER's per-token computation chain determines the compiled depth.

### Proposed

RENDER remains the critical path driver. Tracing the chain:

```
Layer ~1:  attend WALL_j → raw geometry (ax, ay, bx, by)
           attend PLAYER_X/Y → player position
Layer ~4:  cos(θ), sin(θ) from PLAYER_ANGLE KV cache
Layer ~5:  edge vectors (ex, ey), differences (fx, gy)
Layer ~9:  rotation products via piecewise_linear_2d (all parallel)
Layer ~11: sort_den, C, D, E, sort_num_t
           den_over_cos = sort_den − C·tan_c  (linear op)
Layer ~13: H×|den_over_cos| and |sort_num_t|
Layer ~17: wall_height_int via thermometer  ┐ parallel
           tex_coord via thermometer        ┘
Layer ~19: texture attention (attend_argmax_dot to TEX_COL)
Layer ~42: column_fill / tex_sample (23 ops — the dominant block)
Layer ~45: state transitions + output token type
```

**Estimated critical depth: ~45 layers** (down from 70). Conservative
estimate with compiler packing overhead: ~50.

The column_fill / tex_sample block (23 ops) accounts for half the
critical path. It is unchanged from the current design and is the
primary target for future depth reduction.

## What This Costs

- **token_type widens** from 8 to 8 + max_walls + 3–5 dimensions
  (wall one-hot, col, chunk, and optionally tex_id, vis_lo, vis_hi).
- **3 extra autoregressive tokens** per frame (PLAYER_X, PLAYER_Y,
  PLAYER_ANGLE). Negligible versus the ~250 RENDER tokens.
- **RENDER is deeper per-token** (computes precomputes that were
  previously free from feedback). But the total compiled depth is
  lower because RENDER starts computing ~10 layers earlier.
- **WALL stage changes**: C, D, E, H_inv computation removed (only
  needed sort_den, sort_num_t for BSP rank / renderability). The
  sort_value payload shrinks.
- **SORTED stage changes**: threshold from position, not feedback.
  sort_feedback output removed.
- **Output assembly simplifies**: no feedback packing, no THINKING
  dispatch. Just token_type + overflow outputs (pixels, col, start,
  length, done).

## What This Gains

- **THINKING tokens eliminated** — N fewer tokens per frame.
- **render_feedback eliminated** — 27+ input dimensions removed.
- **sort_feedback eliminated** — 16+ input dimensions removed.
- **~25 fewer compiled layers** (estimated 45 vs current 70).
- **Simpler output assembly** — no thinking_render_fb construction,
  no feedback packing/unpacking, no render_mask bookkeeping.
- **Architectural clarity** — every token's inputs are self-describing:
  its discrete type, the world state, and what previous tokens emitted.
- **Pure autoregressive structure** — structurally identical to how a
  language model generates tokens. Context is the KV cache. State is
  the output sequence. There is nothing else.

## The Dumb-Host Constraint

The host remains a dumb token feeder and pixel bitblitter:

1. Feed token inputs (type + WAD data for prefill tokens).
2. Copy the output token_type vector back as the next input's
   token_type — this now carries wall index, column, chunk, and
   other discrete values, but the host doesn't interpret them.
3. Bitblit pixel strips to the framebuffer.
4. Stop when `done > 0`.

The host does not decide which wall to render, what the next column
is, or how many chunks a column needs. The transformer's output
token_type encodes all of this. The host copies it verbatim.

The one new host responsibility: reading the quantized player position
from EOS's output to populate the PLAYER token types. This is the same
level of complexity as the current host loop's feedback copying (reading
a field from the output and writing it to the next input).

## Implementation Phases

### MVP

1. Position-derived sort threshold (eliminate sort_feedback).
2. Wall identity in RENDER token type (eliminate THINKING).
3. Column and chunk in RENDER token type (eliminate render_col,
   render_chunk_start, render_is_new_wall from feedback).
4. RENDER computes precomputes from raw geometry + PLAYER token
   state (eliminate fb_sort_den, fb_C, fb_D, fb_E, fb_H_inv,
   fb_tex_id, fb_col_lo, fb_col_hi, fb_onehot — the rest of
   render_feedback).
5. PLAYER_X, PLAYER_Y, PLAYER_ANGLE tokens after EOS.
6. tan(angle_offset) as compile-time constant.
7. Integer wall_height via thermometer_floor_div.

Steps 1–4 eliminate all feedback. Steps 5–7 are the depth
optimizations that make the elimination depth-neutral or better.
These could be implemented together or staged, but the depth
optimizations should not be deferred long — without them, feedback
elimination roughly doubles RENDER's depth.

### Future Optimizations

**COL_SETUP tokens with discrete wall_height.** A per-column token
emitted before each column's RENDER chunks. It computes wall_height,
tex_coord, and performs the texture fetch, then outputs the integer
wall_height h in the next token's type. RENDER tokens carrying h
can start tex_row computation at layer 0 and column_fill as soon as
texture data arrives from COL_SETUP's KV cache. Estimated savings:
~5–12 layers depending on column_fill restructuring. Cost: ~120
extra autoregressive tokens per frame.

**Optimized PLAYER token mechanisms.** The MVP uses float conversion
and broadcast. The discrete position and angle values may enable
more efficient formulations — the rotation products decompose into
wall-and-angle-only terms plus position-and-angle-only terms, and
the position-and-angle terms are shared across all walls. Exploring
whether this decomposition can reduce depth or eliminate
piecewise_linear_2d operations is an open question.

**column_fill depth reduction.** At 23 ops, column_fill / tex_sample
is half of RENDER's critical path. The current `tex_sample_batch_size=8`
processes 20 rows in batches. Increasing parallelism (at the cost of
residual stream width) or restructuring the dynamic_extract could
reduce this block.
