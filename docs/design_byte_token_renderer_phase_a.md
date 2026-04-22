# Phase A: Wall thinking tokens

Move the WALL stage's computation from the prefill into
autoregressive "thinking tokens" — discrete intermediate results
the transformer emits before it starts rendering. The RENDER stage
and texture pipeline are unchanged.

This is the first shippable phase of the byte-token renderer design
(see `design_byte_token_renderer.md` for the full vision).

## What changes

The WALL stage currently computes four things at each WALL position
during prefill: collision detection, BSP rank, renderability, and
visibility columns. These are ~17 sequential MLP layers — the
deepest single-stage computation in the graph and the primary driver
of the ~70-layer compiled depth.

This phase moves all four computations into thinking tokens: small
autoregressive steps where each step computes one value, quantizes
it to a 16-bit integer, and emits it. The next step reads the
previous result at layer 0 (as an input token) rather than at
layer 17+ (from deep in the KV cache). Each step is only a few
MLP layers deep.

The collision resolution currently performed by EOS also moves to
thinking tokens (it reads collision flags which are now thinking
token outputs). EOS becomes a pure end-of-prompt marker.

SORTED changes its output format: instead of setting render_wall_index
via overlaid state, it emits the wall index as a separate value token.

RENDER changes minimally: it reads wall_index from the SORTED value
token in the KV cache (instead of from overlaid input), and reads
the resolved player position from thinking tokens (instead of from
PLAYER broadcasts).

## What stays the same

- TEX_COL tokens and the entire texture pipeline
- BSP_NODE tokens and computation (side_P_vec broadcast)
- RENDER stage internals (chunk_fill, texture sampling, compositing,
  state machine, overlaid col/chunk_k)
- The compiler core (residual allocation, scheduling, weight writing)
- PLAYER_X/Y/ANGLE tokens carry pre-collision state in the prefill
  (RENDER reads cos/sin from these; collision doesn't change angle)

## Token sequence

### Prefill (batched)

```
TEX_COL × (num_tex × tex_w)         texture columns (unchanged)
INPUT                                 movement controls + angle
BSP_NODE × max_bsp_nodes             splitting planes (unchanged)
[WALL 0] [AX] v [AY] v [BX] v [BY] v [TEX_ID] v [BSP_C0] v ... [BSP_C47] v [BSP_CONST] v
[WALL 1] [AX] v [AY] v ...
  ...
[WALL 7] [AX] v [AY] v ...
EOS                                   end of prompt (no computation)
PLAYER_X  PLAYER_Y  PLAYER_ANGLE      pre-collision state
```

Wall data is tokenized as identifier-value pairs: one marker per
wall, followed by 54 pairs (5 geometry + 49 BSP coefficients).
Each identifier is a distinct vocabulary entry (AX, AY, BX, BY,
TEX_ID, BSP_C0 through BSP_C47, BSP_CONST). Each value `v` is a
quantized 16-bit integer.

That's 1 marker + 108 data tokens = 109 tokens per wall, 872
tokens for 8 walls.

During prefill, each wall data token inherits its wall identity
from the most recent WALL marker via attention (recency-based,
using the sequential position index). The marker's first MLP layer
converts its token value to a wall_index one-hot (8-wide, via
map_to_table). Data tokens attend to the marker and store the
one-hot in the KV cache alongside their field type. This makes
each data token addressable by content: (field_type, wall_index).

### Thinking phase (autoregressive)

One autoregressive loop handles both thinking and rendering. The
host doesn't distinguish phases — it feeds tokens and interprets
the output. Thinking tokens are values the host ignores.

BSP_SIDE tokens are not needed — BSP_NODE is unchanged in the
prefill and `side_P_vec` is already available in the KV cache via
the existing `attend_mean_where` broadcast. The thinking phase
starts directly with wall computation.

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

[THINKING_WALL_1]
  [BSP_RANK] 3
  ... (same 13 identifier-value pairs)

  ... (walls 2-7)

[RESOLVED_X] 12345
[RESOLVED_Y] 54321
[RESOLVED_ANGLE] 128
```

Each indented line is TWO autoregressive steps: the identifier
token (a vocabulary entry naming the value) and the value token
(a quantized 16-bit integer). The identifier tells the transformer
what to compute next. The value token does the computation and
emits the result.

Total thinking tokens: 8 × (1 marker + 13 identifier-value pairs
× 2) + 3 resolved pairs × 2 = **222 tokens**.

### Thinking state machine

The transformer alternates between emitting identifiers and
computing values. Without overlaid state, it determines what to
do from the KV cache:

**Is this step an identifier or a value?** The token reads its own
input (the previous step's output). If the input is in the
identifier vocabulary range, this step is a value token — compute
and emit. If the input is in the value range, this step is an
identifier token — emit the next identifier in the sequence. This
is a compare on the input value (~1 MLP layer).

**Which identifier comes next?** The identifier token attends to
"most recent identifier" in the KV cache. From the previous
identifier, it determines the next one in the fixed sequence
(BSP_RANK → IS_RENDERABLE → CROSS_A → DOT_A → ... → HIT_Y).
This is a map_to_table lookup (~1 MLP layer). After the last
identifier in a wall (HIT_Y's value), the next step emits either
THINKING_WALL_{N+1} (if more walls remain) or RESOLVED_X (if
all walls are done).

**How does the transition from the last wall to RESOLVED work?**
The transformer attends to "most recent THINKING_WALL marker" and
reads the wall index. If wall_index == max_walls - 1, this was
the last wall — emit RESOLVED_X. Otherwise emit the next
THINKING_WALL marker (with incremented wall index). This is a
compare + select (~1-2 MLP layers).

### Decode (autoregressive, mostly unchanged)

```
[SORTED_0] 3 [RENDER] [RENDER] ... [RENDER]
[SORTED_1] 0 [RENDER] [RENDER] ... [RENDER]
  ...
[DONE]
```

SORTED_N is a vocabulary entry encoding the sort step counter.
The wall index (3, 0, etc.) is a separate value token.
RENDER tokens use existing overlaid state (col, chunk_k)
internally.

**RENDER → SORTED transition.** When the last RENDER token's state
machine decides to advance walls, it needs to emit SORTED_{N+1}.
It derives N by attending to "most recent SORTED" in the KV cache,
extracting the counter from the vocabulary entry (via
map_to_table, ~1 MLP layer), incrementing, and encoding the result
as SORTED_{N+1}.

## How thinking tokens find data

Three attention mechanisms, all using the same principle: content
matching with recency bias.

### 1. Finding the current wall marker

Every thinking token needs to know which wall it's computing for.
It attends to the most recent THINKING_WALL marker in the KV cache.

The attention scores each position by: (type match bonus) +
(position_idx recency). All THINKING_WALL markers outscore
non-markers due to the type match bonus. Among markers, the most
recent one (highest position_idx) wins.

This requires a sequential position index as positional encoding —
a single integer per token (0, 1, 2, ...), fed by the host. No
groups, offsets, or squared terms.

The marker's value in the KV cache includes its wall_index one-hot
(computed from its token value by map_to_table, 1 MLP layer). The
thinking token reads the one-hot and now knows its wall identity.

### 2. Finding previous phases' results

The [T_LO] value token needs [CROSS_A]'s value for the same wall.
Same mechanism: attend to "most recent token of type CROSS_A_VALUE."
Since walls are processed sequentially, the most recent CROSS_A
value is always the current wall's.

### 3. Finding wall data in the prompt

The [CROSS_A] value token needs wall geometry (ax, ay) from the
prompt. This is content-based matching on two criteria: field type
AND wall identity.

During prefill, each wall data token inherited a wall_index one-hot
from its marker (see "Prefill" section above). The data token's KV
cache entry encodes (field_type, wall_index_one_hot).

The thinking token queries for (field_type=AX, target_wall_one_hot).
The dot product peaks at the matching data token. One attention head,
returns the quantized ax value.

The thinking token gets target_wall_one_hot from mechanism #1
(attending to the THINKING_WALL marker). It converts the wall_index
to a one-hot via map_to_table (1 MLP layer).

## How SORTED and RENDER consume thinking results

### SORTED

SORTED_N gathers bsp_rank and is_renderable for all 8 walls via
content-based attention (16 heads: 8 for bsp_rank × 8 walls, 8
for is_renderable). It computes indicators_above internally (~2
MLP layers), then runs attend_argmin_above_integer (unchanged
attention pattern). It outputs the picked wall's index as a
separate value token.

The counter N comes from the SORTED_N vocabulary entry — the
token extracts it from its own input value.

### RENDER

Two changes from the current RENDER:

1. **Wall identity**: RENDER attends to "most recent wall index
   value following a SORTED token" in the KV cache (instead of
   reading render_wall_index from overlaid input).

2. **Resolved player position**: RENDER attends to [RESOLVED_X],
   [RESOLVED_Y] thinking tokens by content matching (instead of
   reading from PLAYER broadcasts). For cos/sin: RENDER still
   reads from PLAYER_ANGLE in the prefill (collision doesn't
   change angle, so pre-collision angle = post-collision angle).

Everything else about RENDER is unchanged: chunk_fill, texture
sampling, compositing, state machine, overlaid col/chunk_k.

## Quantization

Each thinking token's value is a 16-bit integer (0-65535).
Continuous intermediate values are linearly mapped from their
float range to this integer range.

| Value | Float range | Resolution at 16-bit |
|-------|------------|---------------------|
| bsp_rank | 0-7 (integer) | Exact, no quantization |
| is_renderable | 0 or 1 | Exact |
| cross_a, dot_a, cross_b, dot_b | [-40, 40] | 0.0012 |
| t_lo, t_hi | [0, 1] | 0.000015 |
| vis_lo, vis_hi | [-2, 122] | 0.0019 |
| hit_full, hit_x, hit_y | 0 or 1 | Exact |
| resolved_x, resolved_y | [-20, 20] | 0.0006 |
| resolved_angle | 0-255 (integer) | Exact |

The mapping (range, scale) is fixed per value type and baked into
the compiled weights. The producing token's output layer quantizes
(maps float to nearest integer). The consuming token's input
projection dequantizes (scales integer back to float range). Both
are free Linears.

Accumulated quantization error across the full chain (rotation →
clip → projection) is estimated at ~0.003 screen columns based on
3 quantization boundaries × ~0.001 per boundary. This is well
within the ±0.5 column rendering tolerance but should be validated
empirically during testing.

## Positional encoding

One integer per token: the sequential position index (0, 1, 2,
3, ...). The host assigns it trivially from the token's position
in the sequence.

Used only for recency bias in "most recent marker/value" attention
patterns. Not used for position arithmetic, group identification,
or layout-dependent computation.

No groups, no offsets, no squared terms, no fixed layout
constraints.

## Host protocol

One autoregressive loop handles everything. The host doesn't know
or care whether the transformer is thinking or rendering:

```
prev = first_autoregressive_token
while not done:
    output, past = model.step(prev, past)
    token = decode_output(output)

    if is_render_token(token):
        blit pixels to framebuffer
    elif token == DONE:
        done = True
    else:
        pass  # thinking, SORTED, wall index — ignore

    prev = build_input(token)
```

The transition from thinking to rendering is invisible to the
host. The transformer decides when to stop thinking and emit
[SORTED_0].

## Expected impact

The deepest single-phase thinking token is WALL_CLIP at ~6-7 MLP
layers. The deepest decode token is the RENDER precompute (COL
equivalent) at ~15 layers, unchanged from today.

The compiled graph depth is determined by the deepest single
autoregressive step, which remains the RENDER precompute at ~15
layers. The thinking tokens don't increase this — they're all
shallower.

However, the total compiled depth should decrease because the
WALL computation no longer serializes with RENDER via the KV
cache. Currently RENDER waits for WALL results at layer ~23
before it can start. With thinking tokens, RENDER reads wall
results at layer 0 (they're input tokens, not deep KV entries).

Estimated compiled depth: **~25-30 layers** (down from ~70).

Cost: ~222 additional autoregressive steps for the thinking phase.
Acceptable.

## Testing strategy

The thinking token outputs can be validated against the current
WALL stage's continuous outputs. For each wall, the current
system produces (bsp_rank, is_renderable, vis_lo, vis_hi,
collision flags) as continuous values. The thinking tokens produce
the same values as quantized integers. The test: compile the
current graph, run a frame, capture WALL outputs. Then compile
the thinking-token graph, run the same frame, capture thinking
outputs. Dequantize and compare within the quantization tolerance.

The final rendered frame should match within the existing test
tolerances — the thinking tokens are computing the same math,
just in smaller steps with quantization at the boundaries.
