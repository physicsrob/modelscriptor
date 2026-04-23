# The DOOM Game Graph (Post Phase A)

This is a transformer that runs DOOM. Not a neural net trained to play
DOOM — a hand-designed computational graph whose forward pass *is* the
game engine. The graph compiles to a real transformer (attention heads,
MLP sublayers, linear layers) and executes on GPU via `model.step()`.

The host is deliberately dumb: it feeds tokens in, reads integers out,
and bitblits pixels to the framebuffer. All game logic — player
movement, collision detection, BSP traversal, front-to-back wall
sorting, perspective projection, and texture-mapped column rendering —
happens inside the transformer's autoregressive generation.

## Token Sequence

Each frame, the transformer processes this sequence:

```
TEX_COL×(num_tex × tex_w) → INPUT → BSP_NODE×M →
[WALL 0] [AX] v [AY] v ... [BSP_CONST] v →
[WALL 1] [AX] v ... → ... → [WALL N] [AX] v ... →
EOS → PLAYER_X → PLAYER_Y → PLAYER_ANGLE →
[THINKING_WALL_0] [BSP_RANK] v [IS_RENDERABLE] v ... [HIT_Y] v →
[THINKING_WALL_1] ... → ... → [THINKING_WALL_N] ... →
[RESOLVED_X] v → [RESOLVED_Y] v → [RESOLVED_ANGLE] v →
[SORTED_0] wall_idx → RENDER×k →
[SORTED_1] wall_idx → RENDER×k →
... → DONE
```

These compile into the following phases:

| Phase | Tokens | Mode | Typical count |
|-------|--------|------|---------------|
| Tex | TEX_COL | Prefill | num_tex × tex_w (e.g. 512) |
| Prefill | INPUT + BSP_NODE + WALL data + EOS | Prefill | 1 + M + N×109 + 1 (e.g. ~922) |
| Player | PLAYER_X/Y/ANGLE | Prefill | 3 |
| Thinking | THINKING_WALL + RESOLVED | Autoregressive | ~222 |
| Sort+Render | SORTED + RENDER (interleaved) | Autoregressive | N + dynamic (~258) |

Prefill tokens are processed in a single batched forward pass.
Dependencies between them (BSP reads player state, wall data tokens
inherit wall identity from markers) are resolved internally through
attention layers within that pass.

Autoregressive tokens are generated one at a time. The host runs a
single loop: feed the previous output, read the new output, interpret
it. The host does not distinguish thinking from rendering — it
ignores thinking tokens and blits pixels for RENDER tokens.

## Token Types

Token types are identified by E8 spherical codes (8-dimensional unit
vectors from the E8 lattice) for prefill tokens and by vocabulary
entries (16-bit integers from reserved ranges) for autoregressive
tokens.

Prefill token types (E8 codes):

```
TOKEN_INPUT          (0)    TOKEN_TEX_COL        (5)
TOKEN_WALL_MARKER    (1)    TOKEN_BSP_NODE       (7)
TOKEN_WALL_DATA      (6)    TOKEN_PLAYER_X     (240)
TOKEN_EOS            (2)    TOKEN_PLAYER_Y     (241)
TOKEN_RENDER         (4)    TOKEN_PLAYER_ANGLE (242)
```

Autoregressive token types (vocabulary ranges):

| Range | Token type |
|-------|-----------|
| 0-65535 (general) | Thinking value tokens (quantized 16-bit integers) |
| Reserved entries | THINKING_WALL_0..7, identifier tokens (BSP_RANK, IS_RENDERABLE, CROSS_A, DOT_A, CROSS_B, DOT_B, T_LO, T_HI, VIS_LO, VIS_HI, HIT_FULL, HIT_X, HIT_Y, RESOLVED_X, RESOLVED_Y, RESOLVED_ANGLE) |
| Reserved entries | SORTED_0..7, DONE |
| (RENDER tokens use existing E8-based type system) |

## Positional Encoding

Every token carries a sequential position index (0, 1, 2, 3, ...)
as positional encoding. The host assigns it trivially from the
token's position in the sequence.

Used for recency bias in "most recent marker" attention patterns.
No groups, no offsets, no squared terms.

## Stages

### TEX_COL — Texture Data (Prefill)

Unchanged from the current architecture. One token per column of
each texture. Each carries the raw RGB pixel data for that column
(`tex_h × 3` floats) plus an E8 code identifying which texture it
belongs to and a one-hot encoding of its column index.

RENDER tokens retrieve texture pixels via `attend_argmax_dot`.

### INPUT — Player Controls (Prefill)

Unchanged. A single token. Receives current angle and six movement
flags. Computes new angle, velocity `(dx, dy)`, and trig values
`(cos, sin)`. Broadcasts all five via `attend_mean_where`.

### BSP_NODE — Spatial Classification (Prefill)

Unchanged. M tokens (typically 48). Each classifies the player
against a BSP splitting plane:

```
side_P = sign(nx × player_x + ny × player_y + d)
```

Results are gathered into a shared `side_P_vec` via
`attend_mean_where`. This binary vector is available to the
thinking phase via the KV cache.

### WALL Data — Scene Geometry (Prefill)

Wall geometry is tokenized as identifier-value pairs. Each wall is
a marker token followed by 54 pairs (5 geometry + 49 BSP
coefficients):

```
[WALL 0] [AX] 12345 [AY] 54321 [BX] ... [BY] ... [TEX_ID] 2
         [BSP_C0] v [BSP_C1] v ... [BSP_C47] v [BSP_CONST] v
```

Each identifier (AX, AY, etc.) is a distinct vocabulary entry.
Each value is a quantized 16-bit integer. The marker token's
value encodes the wall identity. That's 1 marker + 108 data
tokens = 109 tokens per wall, 872 tokens for 8 walls.

During prefill, each wall data token inherits its wall identity
from the most recent WALL marker via recency-based attention. The
marker's first MLP layer converts its token value to a wall_index
one-hot (8-wide, via `map_to_table`). Data tokens attend to the
marker and store the one-hot in the KV cache alongside their field
type.

This makes each data token addressable by content: later tokens
can query for (field_type=AX, wall_index=3) and find the right
token. One attention head, zero MLP layers for the lookup.

### EOS — End of Prompt (Prefill)

A single token marking the end of the prompt. No computation.

In the pre-Phase-A architecture, EOS performed collision
resolution. Post Phase A, collision resolution moves to thinking
tokens (see RESOLVED below).

### PLAYER — Pre-Collision State Broadcast (Prefill)

Three tokens (PLAYER_X, PLAYER_Y, PLAYER_ANGLE), carrying the
pre-collision player state from the host. Each broadcasts its
value via `attend_mean_where`.

RENDER reads cos/sin from PLAYER_ANGLE (collision does not change
angle, so pre-collision and post-collision angle are identical).
RENDER reads resolved x/y from thinking tokens instead.

### THINKING_WALL — Wall Computation (Thinking, Autoregressive)

The core of Phase A. Each wall's computation is decomposed into
13 identifier-value pairs, preceded by a THINKING_WALL marker.
Walls are processed sequentially (wall 0 completes before wall 1
starts).

```
[THINKING_WALL_0]
  [BSP_RANK] 5
  [IS_RENDERABLE] 1
  [CROSS_A] 12345
  [DOT_A] 54321
  [CROSS_B] 11111
  [DOT_B] 22222
  [T_LO] 33333
  [T_HI] 44444
  [VIS_LO] 15
  [VIS_HI] 45
  [HIT_FULL] 1
  [HIT_X] 0
  [HIT_Y] 1
```

Each identifier is a distinct vocabulary entry. Each value is a
quantized 16-bit integer. Each indented line is TWO autoregressive
steps: the identifier token and the value token. The computation
at each value token:

**BSP_RANK.** Reads `side_P_vec` from the BSP_NODE broadcast in
the KV cache (computed during prefill, unchanged) and the wall's
BSP coefficients (from the prompt, via content match on
field_type + wall_index). Computes
`rank = dot(coeffs, sides) + const`. Emits integer rank (0-7).

**IS_RENDERABLE.** Reads wall geometry from the prompt and player
state from the KV cache. Checks `|sort_den| > ε` and
`num_t × sign(den) > 0`. Emits 0 or 1.

**CROSS_A, DOT_A, CROSS_B, DOT_B.** Reads wall endpoints
`(ax, ay, bx, by)` from the prompt and player `(cos, sin)` from
the KV cache. Rotates each endpoint into the player's frame via
`piecewise_linear_2d` products. Each token computes one component.

**T_LO, T_HI.** Reads the rotated endpoints (CROSS_A etc. from
earlier thinking tokens, via "most recent of type" attention).
Computes FOV clipping: `t* = f_A / (f_A - f_B)` for each FOV
boundary plane, then aggregates via max/min. This is the deepest
thinking token at ~6-7 MLP layers (the reciprocal + multiply
chain).

**VIS_LO, VIS_HI.** Reads the clip parameters (T_LO, T_HI) and
the rotated endpoints. Projects the clipped endpoints to screen
columns via `low_rank_2d` (rank-3 SVD approximation of atan).
Gated by IS_RENDERABLE.

**HIT_FULL, HIT_X, HIT_Y.** Reads wall geometry from the prompt
and player velocity from the INPUT broadcast. Computes ray-segment
intersection for the full velocity ray, x-only ray, and y-only
ray. Each emits 0 or 1.

### Thinking State Machine

The transformer alternates between emitting identifiers and
computing values. Without overlaid state, it determines what to
do from the KV cache:

**Is this step an identifier or a value?** The token reads its own
input (the previous step's output). If the input is in the
identifier vocabulary range, this step is a value token — compute
and emit. If the input is in the value range, this step is an
identifier token — emit the next identifier. This is a compare on
the input value (~1 MLP layer).

**Which identifier comes next?** The identifier token attends to
"most recent identifier" in the KV cache. From the previous
identifier, it determines the next one in the fixed sequence
(BSP_RANK → IS_RENDERABLE → CROSS_A → ... → HIT_Y). This is a
`map_to_table` lookup (~1 MLP layer). After the last value in a
wall (HIT_Y), the next step emits either THINKING_WALL_{N+1}
(if more walls remain) or RESOLVED_X (if all walls are done).

**Wall transition.** The transformer attends to "most recent
THINKING_WALL marker" and reads the wall index. If
wall_index == max_walls - 1, this was the last wall — emit
RESOLVED_X. Otherwise emit THINKING_WALL_{N+1}. This is a
compare + select (~1-2 MLP layers).

### RESOLVED — Collision Resolution (Thinking, Autoregressive)

Three identifier-value pairs after all walls' thinking tokens:

```
[RESOLVED_X] 12345
[RESOLVED_Y] 54321
[RESOLVED_ANGLE] 128
```

Reads all 8 walls' collision flags (HIT_FULL, HIT_X, HIT_Y) from
the thinking tokens in the KV cache. Reads player position from
the prompt and velocity from INPUT. Applies axis-separated wall
sliding (same algorithm as the pre-Phase-A EOS stage):

- X-axis blocked if both full ray and x-only ray hit any wall.
- Y-axis blocked if both full ray and y-only ray hit any wall.

Emits the resolved position and angle as quantized integers.
RENDER reads these from the KV cache for the post-collision
player position.

### SORTED — Front-to-Back Wall Selection (Decode, Autoregressive)

Interleaved with RENDER. Each SORTED token picks the next-closest
wall and emits its index as a separate value token:

```
[SORTED_0] 3 [RENDER] [RENDER] ... [RENDER]
[SORTED_1] 0 [RENDER] [RENDER] ... [RENDER]
```

SORTED_N is a vocabulary entry encoding the sort step counter N.
The token extracts N from its own input value.

Each SORTED token:

1. Gathers bsp_rank and is_renderable for all 8 walls from
   thinking tokens (16 attention heads, content-based matching).
2. Computes `indicators_above` internally (~2 MLP layers).
3. Runs `attend_argmin_above_integer` with threshold N to pick
   the renderable wall with rank equal to N.
4. Emits the picked wall's index as a value token.

**RENDER → SORTED transition.** When the last RENDER token's state
machine decides to advance walls, it derives the current sort
counter by attending to "most recent SORTED" in the KV cache,
extracting N from the vocabulary entry (via `map_to_table`),
incrementing, and emitting SORTED_{N+1}.

### RENDER — Pixel Generation (Decode, Autoregressive)

Mostly unchanged from the pre-Phase-A architecture. Each RENDER
token paints a vertical chunk of one screen column (`chunk_size`
rows, default 20).

Two changes:

1. **Wall identity.** RENDER attends to "most recent wall index
   value following a SORTED token" in the KV cache, instead of
   reading `render_wall_index` from overlaid input.

2. **Resolved player position.** RENDER attends to RESOLVED_X and
   RESOLVED_Y thinking tokens by content matching, instead of
   reading from PLAYER broadcasts. For cos/sin: RENDER still reads
   from PLAYER_ANGLE in the prefill (collision doesn't change
   angle).

Everything else is unchanged: wall geometry attention, render
precomputation, texture fetch, chunk fill with `dynamic_extract`,
compositing, and the state machine (advancing col/chunk_k/wall
via overlaid outputs).

## How Thinking Tokens Find Data

Three attention mechanisms, all using the same principle: content
matching with recency bias.

### 1. Finding the current wall marker

Every thinking token attends to the most recent THINKING_WALL
marker in the KV cache. Scoring: (type match bonus) +
(position_idx). THINKING_WALL markers outscore non-markers; among
markers, the most recent one (highest position_idx) wins.

The marker's KV cache entry includes its wall_index one-hot
(computed from its token value by `map_to_table`, 1 MLP layer).
The thinking token reads the one-hot and knows its wall identity.

### 2. Finding previous phases' results

A thinking token attends to "most recent token of type X" — for
example, the T_LO value token attends to the most recent
CROSS_A_VALUE token. Since walls are processed sequentially, the
most recent value of each type is always the current wall's.

Same scoring as #1: type match bonus + position_idx recency.

### 3. Finding wall data in the prompt

During prefill, each wall data token inherited a wall_index
one-hot from its WALL marker (see WALL Data section above). The
data token's KV cache entry encodes (field_type, wall_index).

The thinking token queries by content: (field_type=AX,
target_wall_one_hot). Peaks at the matching data token. The
thinking token gets target_wall_one_hot from mechanism #1.

## Outputs

### Thinking Phase Outputs

The host ignores thinking tokens. They are the transformer talking
to itself — intermediate geometry results stored in the KV cache
for the rendering phase to consume.

### SORTED Outputs

Each SORTED_N token emits a wall index as a value token. The host
ignores this.

### RENDER Outputs (Overflow, unchanged)

- **pixels** (chunk_size × 3 wide): RGB values for the current chunk.
- **col** (1): Screen column index.
- **start** (1): Screen row where this chunk begins.
- **length** (1): Number of rows painted (0 for non-RENDER tokens).
- **done** (1): +1 when all walls are fully rendered.
- **sort_done** (1): +1 when sort has exhausted all renderable walls.

### RENDER Overlaid Outputs (Internal, unchanged)

RENDER tokens still use overlaid state internally for their state
machine:

- **token_type** (8-wide): E8 code for the next token type.
- **render_col** (1): Current screen column.
- **render_chunk_k** (1): Chunk index within the current column.
- **wall_counter** (1): How many walls have been rendered.
- **render_wall_index** (1): Unused post Phase A (wall identity now
  comes from the SORTED value token in the KV cache).

## Host Protocol

One autoregressive loop handles everything:

```python
# Prefill (batched)
past = prefill(tex_cols + input + bsp_nodes + wall_data + eos + player)

# Single autoregressive loop — host doesn't know about phases
while not done:
    output, past = model.step(prev_token, past)
    token = decode_output(output)

    if is_render_token(token):
        read (pixels, col, start, length) from overflow
        blit pixels to framebuffer at (col, start)
    elif token == DONE:
        done = True
    else:
        pass  # thinking, SORTED, wall index — host ignores

    prev_token = build_input(token)
```

The host does not distinguish thinking from rendering. It does not
know when one phase ends and another begins. The transformer
decides when to stop thinking and emit SORTED_0.

## Quantization

Thinking tokens emit 16-bit integers. Continuous intermediate
values are linearly mapped from their float range.

| Value | Float range | 16-bit resolution |
|-------|------------|-------------------|
| bsp_rank | 0-7 integer | Exact |
| is_renderable | 0 or 1 | Exact |
| cross/dot (a, b) | [-40, 40] | 0.0012 |
| t_lo, t_hi | [0, 1] | 0.000015 |
| vis_lo, vis_hi | [-2, 122] | 0.0019 |
| hit_full/x/y | 0 or 1 | Exact |
| resolved_x/y | [-20, 20] | 0.0006 |
| resolved_angle | 0-255 integer | Exact |

Accumulated quantization error through the full chain is estimated
at ~0.003 screen columns (3 quantization boundaries × ~0.001 per
boundary). Well within the ±0.5 column rendering tolerance.
Should be validated empirically during testing.

## How the Graph Becomes a Transformer

The computational graph compiles to a standard transformer via
`compile_game` (which calls `compile_headless` internally):

- **Attention heads** implement cross-position data flows:
  - `attend_mean_where`: broadcast (INPUT→all, BSP→all, PLAYER→all).
  - `attend_argmin_above_integer`: threshold-based selection
    (thinking tokens→SORTED for front-to-back sort).
  - `attend_argmax_dot`: dot-product lookup (TEX_COL→RENDER for
    texture fetch, WALL→RENDER for wall geometry).
  - Content matching with recency: thinking tokens reading from
    markers, previous phases, and prompt data.

- **MLP sublayers** implement nonlinear functions via
  `piecewise_linear` and `piecewise_linear_2d` approximations.

- **Linear layers** implement exact affine transforms.

The deepest single autoregressive step is the RENDER precompute
at ~15 MLP layers (unchanged). Thinking tokens are all shallower
(deepest is T_LO/T_HI at ~6-7 layers). The overall compiled
graph depth is estimated at ~25-30 layers, down from ~70.

## Key Design Decisions

**Why thinking tokens?**
The WALL stage's computation is ~17 sequential MLP layers — the
primary driver of the current ~70-layer compiled depth. By
decomposing it into autoregressive steps (each ~1-7 layers deep),
the compiled graph only needs to be deep enough for the deepest
single step. Trading autoregressive steps for model depth.

**Why identifier-value pairs?**
Each thinking value gets a named identifier token before it —
like key-value pairs in structured text. The token stream is
self-documenting: reading the output shows exactly what the
transformer computed and in what order. The identifier also tells
the transformer what to compute next (the value token reads the
preceding identifier to determine its role).

**Why sequential wall ordering?**
Walls are processed one at a time (wall 0's full thinking
sequence completes before wall 1 starts). This enables the "most
recent of type X" attention pattern — the most recent CROSS_A
value is always the current wall's, without needing to match on
wall identity. Simpler attention, no wall-identity tracking
needed for intra-wall references.

**Why recency-based attention instead of position arithmetic?**
Thinking tokens find data via content matching with recency bias:
attend to the most recent token of a given type. This avoids
positional encoding schemes with fixed layouts, group sizes, or
offset arithmetic. The only positional input is a sequential
counter (0, 1, 2, ...) used for the recency tiebreak. No
constraints on where tokens appear in the sequence.

**Why does SORTED emit wall index as a separate value token?**
Without overlaid state for the wall identity, SORTED's result
(which wall to render next) must be communicated through the
token stream. The SORTED_N command token triggers the sort
computation; the following value token carries the result. RENDER
reads it via "most recent value after SORTED" attention.

**Why does EOS no longer do collision resolution?**
EOS is an end-of-prompt marker. Collision resolution needs
collision flags, which are now computed during the thinking phase
(after EOS). The resolution moves to RESOLVED_X/Y/ANGLE thinking
tokens, which read collision flags from the KV cache and emit the
post-collision player state.

**Why are BSP_NODE tokens unchanged?**
BSP_NODE computation is shallow (~2 layers) and already works.
The `side_P_vec` broadcast is already in the KV cache for the
thinking tokens to read. Converting BSP to thinking tokens would
add complexity without reducing the compiled depth. Future phases
may convert it for consistency with the byte-token vision.

**Why no BSP_SIDE thinking tokens?**
BSP_NODE is unchanged in the prefill, so `side_P_vec` is already
computed and available in the KV cache via `attend_mean_where`.
The WALL_RANK thinking token reads it directly. BSP_SIDE thinking
tokens would be redundant.

**Why does RENDER still use overlaid state?**
RENDER's internal state machine (col, chunk_k advancement) is
unchanged in this phase. Eliminating overlaid state from RENDER
is part of the future per-pixel output phase, not Phase A. The
boundary is clean: thinking tokens and SORTED use the new
single-integer format; RENDER uses the existing overlaid format.
