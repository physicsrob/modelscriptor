# Design: Byte-token DOOM renderer

The DOOM renderer becomes a sequence-to-sequence model with a 16-bit
vocabulary (0-65535). The input sequence is a structured scene
description. The output sequence is a stream of thinking tokens and
draw commands. Every token — input and output — is a single integer.
The host feeds integers and interprets the output as terminal-style
draw commands. The architecture is identical to a byte-level
autoregressive language model.

## Vocabulary

Every token is a 16-bit integer (0-65535). The input projection maps
the integer to a d-wide continuous vector via a linear scaling, then
the first MLP layer handles nonlinear type discrimination (via
`in_range` or `map_to_table`). This is analogous to an embedding
layer — the linear projection plays the role of the embedding matrix,
and the MLP layer handles the discrete type lookup that a true
embedding table would do in one step.

Output tokens are integers from reserved vocabulary ranges:

| Range     | Meaning              |
|-----------|----------------------|
| 0-255     | Palette byte (pixel) |
| 256-375   | COL command (256 + col_index) |
| 376-475   | ROW command (376 + row_index) |
| 476       | SORTED               |
| 477       | DONE                 |
| 478-733   | Thinking: vis_lo/vis_hi (478 + col_value) |
| 734+      | Thinking: intermediate geometry values |

The host decodes output integers by range. The transformer's state
machine outputs into the appropriate range via `add_const` or
`select`.

Wall coordinates, BSP plane coefficients, and player position are
quantized to 16-bit integers by the host before feeding. 65536 levels
over [-20, 20] gives ~0.0006-unit resolution — better than original
DOOM's fixed-point. Palette indices (0-255) use a subset of the
vocabulary.

## Input sequence (the "prompt")

The prompt is a structured description of the scene. Markers provide
structure (like XML tags); data tokens carry single integer values
(like characters between tags). A token's meaning comes from its
position in the sequence, not from self-describing metadata.

```
--- Textures (33,280 tokens) ---

[COL tex=0 col=0]  p p p p p ... p        1 marker + 64 palette bytes
[COL tex=0 col=1]  p p p p p ... p
  ...
[COL tex=7 col=63] p p p p p ... p

--- Walls (440 tokens) ---

[WALL 0]  ax ay bx by tex_id  c₀ c₁ ... c₄₇ const    1 marker + 54 values
[WALL 1]  ax ay bx by tex_id  c₀ c₁ ... c₄₇ const
  ...
[WALL 7]  ax ay bx by tex_id  c₀ c₁ ... c₄₇ const

--- BSP tree (192 tokens) ---

[BSP 0]  nx ny d                           1 marker + 3 values
[BSP 1]  nx ny d
  ...
[BSP 47] nx ny d

--- Player state + controls (12 tokens) ---

[PLAYER]  x y angle
[INPUT]   fwd back turn_l turn_r strafe_l strafe_r

--- End of prompt ---

[EOS]
```

Approximate token counts for the default configuration (8 textures,
64x64, 8 walls, 48 BSP nodes):

| Section   | Markers | Data tokens | Total  |
|-----------|---------|-------------|--------|
| Textures  | 512     | 32,768      | 33,280 |
| Walls     | 8       | 432         | 440    |
| BSP nodes | 48      | 144         | 192    |
| Player    | 2       | 9           | 11     |
| EOS       | 1       | 0           | 1      |
| **Total** |         |             | **~33,924** |

## Output sequence (the "response")

The output has two phases: a thinking phase (geometry computation
emitted as discrete tokens) and a drawing phase (pixel output).

### Phase 1: Thinking tokens (chain-of-thought geometry)

After reading the prompt, the transformer "thinks through" the
geometry — computing wall visibility, BSP rank, and collision
results, emitting each intermediate result as a discrete token. This
is chain-of-thought reasoning applied to rendering.

The wall computation decomposes into fine-grained steps. Each value
gets its own identifier-value token pair — like key-value pairs in
structured text. The identifier names the value; the following token
carries the computed result as a quantized 16-bit integer. Walls are
processed sequentially (wall 0 completes before wall 1 starts).

```
[BSP_SIDE] s₀                     BSP plane side decisions
[BSP_SIDE] s₁                     (one pair per node)
  ...
[BSP_SIDE] s₄₇

[THINKING_WALL_0]                  wall marker (sets wall identity)
  [BSP_RANK] 5                    front-to-back sort key
  [IS_RENDERABLE] 1               visibility flag
  [CROSS_A] 12345                 rotated endpoint A (cross component)
  [DOT_A] 54321                   rotated endpoint A (dot component)
  [CROSS_B] 11111                 rotated endpoint B (cross component)
  [DOT_B] 22222                   rotated endpoint B (dot component)
  [T_LO] 33333                   FOV clip lower bound
  [T_HI] 44444                   FOV clip upper bound
  [VIS_LO] 15                    visible screen column range (low)
  [VIS_HI] 45                    visible screen column range (high)
  [HIT_FULL] 1                   collision: full velocity ray
  [HIT_X] 0                      collision: x-only ray
  [HIT_Y] 1                      collision: y-only ray

[THINKING_WALL_1]
  [BSP_RANK] 3
  ... (same 13 identifier-value pairs)

  ... (walls 2-7)

[RESOLVED_X] 12345                post-collision player position
[RESOLVED_Y] 54321
[RESOLVED_ANGLE] 128
```

Each identifier-value pair is two autoregressive steps. The
identifier tells the transformer what to compute; the value token
does the computation and emits the quantized result. The deepest
single computation is FOV clipping (T_LO/T_HI) at ~6 MLP layers.

Thinking tokens find their context via content matching with recency
bias (see "How the transformer finds data" below). Each value token
attends to "most recent THINKING_WALL marker" to learn its wall
identity, and to "most recent [CROSS_A] value" (etc.) for previous
phases' results. No fixed layout or positional arithmetic needed.

**Quantization at token boundaries.** Intermediate values (rotated
coordinates, clip parameters) are quantized to 16-bit integers.
Accumulated error across the full chain: ~0.003 screen columns, well
within the ±0.5 column rendering tolerance.

Token count for thinking: 96 BSP_SIDE pairs + 8 × (1 marker +
13 pairs × 2) + 3 resolved pairs ≈ **~318 thinking tokens**.

### Phase 2: Drawing (pixel output)

After the thinking phase, the transformer generates draw commands — a
terminal protocol. The host interprets these like ANSI escape codes:
cursor positioning followed by pixel data.

```
[SORTED]                        pick next wall (sort stage)
[COL 10]                        set cursor X = 10, precompute column geometry
[ROW 5]                         set cursor Y = 5
p p p p p p p p p p p           palette bytes (pixels at cursor, Y increments)
[COL 11]                        next column
[ROW 3]
p p p p p p p p p p p p p
  ...
[SORTED]                        next wall
[COL 20]
  ...
[DONE]                          frame complete
```

Each `p` is a single integer — a DOOM palette index (0-255). The
host writes each palette byte to the framebuffer at the current
cursor position and advances the cursor. COL and ROW reposition the
cursor. SORTED triggers the sort stage to select the next wall. DONE
ends the frame.

For a 120×100 screen, the drawing phase is roughly **12,000-25,000
tokens** depending on wall overlap (front-to-back compositing may
re-visit columns covered by earlier walls; the host skips
already-filled pixels via conditional writes). A typical box room
scene with 4 non-overlapping walls: ~12,000 tokens. A complex scene
with overlapping walls: up to ~25,000.

## Positional encoding

One integer per token: the sequential position index (0, 1, 2,
3, ...). The host assigns it trivially from the token's position
in the sequence. This serves the same role as positional embeddings
in a standard transformer.

Used for recency bias in "most recent marker/value" attention
patterns (see below). The attention scores each candidate by
type_match_bonus + position_idx. Among type-matching positions,
the highest position_idx (most recent) wins.

No groups, no offsets, no squared terms, no fixed layout
constraints.

## How the transformer finds data

Three mechanisms, all based on content matching with recency bias.

### 1. Most recent marker (for thinking tokens)

A thinking token needs to know which wall it's computing for. It
attends to the most recent THINKING_WALL marker in the KV cache.
The attention scores each position by (type match bonus) +
(position_idx). All THINKING_WALL markers outscore non-markers.
Among markers, the most recent one wins.

The marker's KV cache entry includes a wall_index one-hot (8-wide,
computed from the marker's token value via map_to_table, 1 MLP
layer). The thinking token reads this and knows its wall identity.

Same pattern for finding previous phases' results: "most recent
token of type CROSS_A_VALUE" gives the current wall's CROSS_A,
since walls are processed sequentially.

### 2. Content matching (for prompt data)

A thinking token needs wall geometry (ax, ay, etc.) from the
prompt. During prefill, each wall data token inherited a
wall_index one-hot from its nearest WALL marker (via mechanism #1
above — the same recency pattern works during prefill). The data
token's KV cache entry encodes (field_type, wall_index_one_hot).

The thinking token queries by content: "find the token with
field_type=AX and wall_index=3." The dot product peaks at the
matching data token. One attention head, returns the value.

### 3. Texture pixel lookup (future phases)

The texture pipeline is unchanged in Phase A. Future phases will
convert TEX_COL tokens to per-pixel tokens and use a two-step
lookup: first find the column marker (content match on texture_id
+ column_index), then find the pixel (position-based match using
the marker's position as a reference). The exact mechanism for the
position-based step is TBD — it may use the same recency +
content approach or a position-arithmetic scheme, depending on what
Phase A validates.

## Per-pixel computation model

With thinking tokens, all heavy geometry is pre-computed and
available as discrete input tokens. The per-pixel computation is
lightweight:

1. **Attend to column start** — read cached wall_top, wall_bottom,
   tex_col_idx, wall identity from the [COL] token. (1 attention)
2. **Texture pixel attention** — fetch the palette index for this
   pixel's texture coordinate, using the two-step position-based
   lookup. (1-2 attentions)
3. **Ceiling/floor/wall decision** — compare this pixel's Y against
   wall_top and wall_bottom. (1 MLP layer)
4. **Select color** — choose wall palette index, ceiling palette
   index, or floor palette index. (1 MLP layer)
5. **State machine** — decide next token: another pixel (advance Y),
   COL (column done), SORTED (wall done), or DONE (frame done).
   Compares Y against screen_bottom, col against vis_hi,
   wall_counter against max_walls. (2-3 MLP layers)
6. **Output** — one integer (palette index or command token).

Per-pixel depth: **~5-6 MLP layers**. Column-start depth:
**~15 MLP layers** (rotation products, wall height, texture column
index — computed once, cached for all pixels in the column).

### Layer count

The compiled graph's layer count is set by the deepest single
autoregressive step.

| Token type | Reads from | Own computation | Total depth |
|------------|-----------|-----------------|-------------|
| BSP_SIDE | BSP data at L0 | compare (1 layer) | ~2 |
| WALL_RANK | BSP_SIDE + coefficients at L0 | dot + compare (2 layers) | ~3 |
| WALL_CLIP | WALL_ROTATE at L0 | reciprocal + multiply chain (6 layers) | ~7 |
| WALL_VIS | WALL_CLIP at L0 | low_rank_2d + selects (5 layers) | ~6 |
| SORTED | WALL_RANK + WALL_VIS at L0 | argmin (3 layers) | ~4 |
| COL | wall geometry at L0 | rotation + wall height + tex col (15 layers) | ~16 |
| Pixel | COL + texture at L0 | compare + select + state machine (5 layers) | ~6 |

The deepest step is the COL token at ~16 layers. **Total compiled
graph: ~20 layers** (16 + scheduling overhead). Down from ~70.

The thinking tokens eliminate the cross-token-type serialization that
drives the current layer count. In the current architecture, WALL
visibility (~17 layers) must complete before SORTED can start,
forcing the compiled graph to ~70 layers. With thinking tokens, each
phase is its own autoregressive step, and the compiled graph only
needs to be deep enough for the deepest single step.

### Ceiling and floor pixels

Each pixel token decides its own color. If the pixel's Y is above
the wall, it outputs the ceiling palette index. If below, the floor
palette index. If between wall_top and wall_bottom, it outputs the
texture's palette index. This is a compare + select, about 2 MLP
layers.

This eliminates the current known dumb-host violation where the host
decides ceiling/floor color based on `y < center_y`. All rendering
logic moves into the transformer where it belongs.

## Host protocol

The host is a token feeder and a terminal emulator:

**Prefill.** Feed ~34K integer tokens (the scene prompt). The host
computes positional encoding for each token (block_id, local_pos, and
their squared terms). The host quantizes continuous scene values
(wall coordinates, BSP planes, player position) to 16-bit integers.

**Decode.** Autoregressive loop, identical to LLM inference:
1. Feed the previous output token as the next input.
2. Read the output token (one integer).
3. Interpret the output by range:
   - **0-255** (palette byte): write the corresponding RGB (from the
     fixed DOOM palette lookup table) to
     framebuffer[cursor_x, cursor_y]. Advance cursor Y.
   - **256-375** (COL N): set cursor X to N − 256.
   - **376-475** (ROW N): set cursor Y to N − 376.
   - **476** (SORTED): no host action (the transformer is selecting
     the next wall internally).
   - **477** (DONE): frame complete.
   - **478+** (thinking token): no host action. The transformer is
     computing intermediate geometry results for its own use.

The host does zero rendering computation. Palette-to-RGB conversion
is display-side formatting (a fixed 256-entry lookup table), the same
category as a terminal emulator mapping character codes to glyphs.
Thinking tokens are the transformer talking to itself — the host
passes them through unchanged.

## Impact summary

| Metric | Current | Proposed |
|--------|---------|----------|
| Compiled layers | ~70 | ~20 |
| d_head | 128 | 32 (requires compact col encoding) |
| Heads per layer | 16 | 64 |
| d_input | ~340 | ~10 (1 value + position) |
| Prefill tokens | ~570 | ~34,000 |
| Thinking tokens/frame | 0 | ~100 |
| Draw tokens/frame | ~260 | ~12,000-25,000 |
| Output per token | 60 floats (20 RGB pixels) | 1 integer |
| Ceiling/floor fill | Host (violation) | Transformer (fixed) |
| I/O data type | Mixed float/int | All 16-bit integers |

The transformer is dramatically smaller (~20 layers vs ~70) but
generates many more tokens per frame (~12K-25K vs ~260). Each
autoregressive step is much cheaper (shallow model, sparse
attention). Wall-clock time per frame depends on the balance between
fewer-layers-per-step and more-steps-per-frame. Estimated 5-10×
slower per frame than the current architecture. A 10-frame
walkthrough GIF takes minutes instead of seconds — fine for a blog
post.

## What stays the same

The core rendering algorithms are unchanged:

- BSP-based front-to-back wall ordering (SORTED stage)
- Ray-segment collision detection (WALL stage, now via thinking tokens)
- Perspective projection and FOV clipping (WALL stage, via thinking tokens)
- Rotation products and wall height computation (COL token precompute)
- Texture coordinate calculation (thermometer floor division)

These are the same piecewise-linear-approximation computations in the
same graph structure. The thinking tokens decompose the long chains
into shorter phases, but the math within each phase is identical.

## The narrative

Input: a 34,000-token structured prompt describing a DOOM level —
textures, walls, a BSP tree, and a player's position. Each token is
a 16-bit integer from a 65K vocabulary, the same size as a typical
LLM's vocabulary.

Output: the transformer first "thinks" — working through wall
visibility, clipping, and depth sorting, emitting intermediate
results as tokens. Then it "draws" — a stream of cursor positions
and palette bytes that paint a frame of DOOM, like a program printing
to a terminal.

The host feeds tokens and interprets the output. The same
autoregressive loop that drives any chat model. The same data types.
The same architecture. The transformer even shows its work, like
chain-of-thought reasoning. The only difference is what the tokens
mean.
