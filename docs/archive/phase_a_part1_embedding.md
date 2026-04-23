# Phase A, Part 1 — Token architecture foundation

The first part of Phase A is about the *representation*, not any new
DOOM behavior. After this part lands, DOOM works end-to-end with
M4-equivalent HIT_FULL/HIT_X/HIT_Y thinking-token behavior, but
carried by a clean embedding/unembedding architecture. Nothing new
is computed; everything is re-routed.

## Strong opinions this part commits to

### Tokens are discrete IDs

Every autoregressive step consumes exactly one token ID and emits
exactly one token ID. The host loop is `id_out = decode(transformer(embed(id_in), bypasses))`. There is no second semantic field
riding alongside `id` (no `thinking_value` overlaid alongside a type
code). If a token needs to carry a number, that number *is* the ID.

### Embedding is a matrix lookup

Reuse the existing `torchwright/graph/embedding.py` `Embedding` class,
generalized so it accepts `d_embed != 8` and a caller-supplied table.
The DOOM vocabulary's embedding table is a hand-designed constant:

```
d_embed = 72
V ≈ 65_600
W_embed: (V, 72)
```

Embed is `W_embed[id]`; a single row of the matrix per ID. The
existing `Embedding` Node plus `compile.py`'s `Gather → MatMul` wiring
materialize this lookup in the compiled graph.

### Deembed is a dot product

At the output, `logits = output_slice @ W_unembed.T`, then argmax →
next ID. Weights are tied: `W_unembed = W_embed`. The existing
`Unembedding` helper (and the token-transformer export path that
implements MatMul(embed_table.T)) is the vanilla LM-head the
architecture commits to.

### The VALUE part of the embedding uses a 4+4+4+4 factored one-hot

Value IDs 0–65535 decompose into four 4-bit factors
`(h3, h2, h1, h0)`, each encoded as a 16-wide one-hot in the
embedding. That's 64 columns of VALUE encoding plus 8 columns of
category code, giving `d_embed = 72`.

Argmax at deembed decomposes into four independent 16-way argmaxes
(sharp, no host-side rounding). Emit requires four sequential
thermometer extractions (~4 sublayers of depth at the emit position).

### There is one shared `E8_VALUE` category code

Every VALUE token — wherever it shows up in the sequence — carries
the same type code in the embedding's first 8 columns. This category
is not tied to thinking-token emission specifically; any future
position type that emits a 16-bit quantized integer uses it. Prompt
positions, thinking-token positions, output positions — same
`E8_VALUE` code. The meaning of a value is determined by context
(preceding identifier), not by a value-type-specific code.

### Non-pure I/O is an explicit bypass

Positions like TEX_COL, WALL, BSP_NODE, PLAYER_X/Y/ANGLE, INPUT,
RENDER carry rich position-specific data that doesn't fit into a
single token ID. That data is a *bypass*: raw float columns in the
residual stream, fed by the host alongside the token ID and never
routed through the embedding matrix.

Phase I's bypass surface (declarable explicitly at graph-construction
time):

- **Prompt bypasses**: `tex_pixels`, `texture_id_e8`, `tex_col_input`,
  `wall_ax/ay/bx/by`, `wall_tex_id`, `wall_index`, `wall_bsp_coeffs`,
  `wall_bsp_const`, `bsp_plane_nx/ny/d`, `bsp_node_id_onehot`,
  input flags (`input_forward`, etc.), `player_x`, `player_y`,
  `player_angle`.
- **Autoregressive bypass inputs (overlaid)**: `render_col`,
  `render_chunk_k`, `wall_counter`. Retained as-is; their elimination
  is a future phase per CLAUDE.md.
- **Autoregressive bypass outputs (overflow)**: `pixels`, `col`,
  `start`, `length`, `done`, `advance_wall`, `sort_done`,
  `sort_vis_hi`, `sort_wall_index`.

### Embedding and bypasses are sibling residual leaves

They coexist in the residual stream at disjoint columns; there is no
graph-level "join" op. Stages reference whichever leaf they need —
`is_wall = equals_vector(embedding, embed_lookup("WALL"))` for the
categorical check, direct references to bypass leaves for the
position-specific payload. Transformer layers between the embed and
the unembed see a single residual and don't know or care which
columns came from where.

## Scope

- Generalize `Embedding` / `Unembedding` to support `d_embed > 8` and
  caller-supplied tables.
- Build `torchwright/doom/embedding.py`: vocabulary, hand-designed
  `W_embed` with the 4+4+4+4 layout, helpers for `embed_lookup(name)`
  / `vocab_id(name)` and for `value_id(int)`.
- Introduce a bypass declaration convention (exact mechanism TBD —
  see Open Questions).
- Swap DOOM's compile path so it accepts `(token_ids, bypass_row)`
  input instead of the current all-raw input row, and emits logits
  from a dot-product LM head.
- Rewire every `equals_vector(token_type, E8_X)` detector in the
  existing stages to `equals_vector(embedding, embed_lookup("X"))`.
- Port `thinking_wall.py`'s M4 state machine to sit on the embedding
  carrier. The 10 Phase-II value slots stay stubbed (return the
  `VALUE` category with ID = 0); HIT_FULL/X/Y continue to compute
  real values. RESOLVED tokens are NOT introduced in this part (they
  arrive in Part 4).

## Not in scope

- Extending the state machine to 16 identifiers (Part 2).
- The 10 new value computations (Part 3).
- RESOLVED_X/Y/ANGLE, EOS gutting, RENDER resolved-position migration
  (Part 4).
- Eliminating RENDER's `col`/`chunk_k`/`wall_counter` overlaid state
  (future phase, out of Phase A).
- `WALL` prefill as identifier-value pairs (future phase, out of
  Phase A).
- Fixing M4's SORTED softmax dilution / affine-bounds regressions
  (Part 5).

## Acceptance criteria

Part 1 is done when all of the following hold:

1. Existing non-DOOM examples (`fibonacci`, `adder`, `adder_v2`,
   `calculator`, `calculator_v2`, `caesar_cipher`, `sort_digits_v4`,
   `token_balance`, `binary_increment`) pass their tests unchanged.
   Backwards compatibility of the generalized `Embedding` class.
2. DOOM `compile_game` produces a working compiled module through the
   new path.
3. `make walkthrough ARGS="--scene box --frames 3"` renders correctly
   and matches reference within existing tolerances.
4. `tests/doom/test_thinking_wall.py`'s HIT_FULL/HIT_X/HIT_Y
   dual-path assertions pass through the new embedding carrier. (If
   its step-offset constants are stale from M4, they are updated to
   match the new token sequence — but the numerical behavior is
   unchanged from M4.)
5. Every non-pure input and output in the DOOM graph is declared via
   the bypass mechanism; the `phase_a_plan.md` index (written later)
   can enumerate them.
6. The symbols `thinking_value`, `out_thinking_value`,
   `E8_THINKING_VALUE`, and the `inputs["thinking_value"]` raw input
   slot are removed from the codebase. `grep -r thinking_value
   torchwright tests examples` returns no hits referring to the old
   overlaid field.

## Open questions

These are deliberately left for the implementation session to
resolve. None of them should require revisiting a strong opinion
above.

### Bypass declaration API

Three candidates (new `create_bypass_input`, flag on `create_input`,
documentation-only convention). Pick whichever feels cleanest during
implementation; the important thing is that every bypass is auditable
in one place. Non-load-bearing.

### Argmax placement

The ONNX compiled module can emit either `next_token_ids: int64[n]`
(argmax inside the graph) or `logits: float[n, V]` (argmax host-side).
Real LLMs emit logits for flexibility; the existing
`compile_headless` pipeline emits raw residual slices. Either is
acceptable for DOOM — the host doesn't sample, it argmaxes, so
in-graph or out-of-graph argmax is semantically equivalent.

### Compile-path extension vs. new path

Two concrete options:

- Extend `compile_headless` to recognize `Embedding` as another leaf
  source, emit a parallel `Gather(W_embed, token_ids) → MatMul(embed_proj)`
  alongside the existing raw-input `MatMul(input_proj)`. Accepts two
  input tensors. Minimal compiler surgery.
- Adopt the existing token-transformer export path and extend *it* to
  support bypass inputs/outputs. More structural, matches real-LLM
  conventions.

Either is fine. The shape of the final ONNX graph differs but the
graph-side semantics don't.

### Vocabulary ID allocation details

We have ranges:

- `0..65535` VALUE
- `65536..65543` `THINKING_WALL[0..7]`
- `65544..65559` per-wall identifiers + RESOLVED identifiers
- `65560..65562` `SORTED_WALL`, `RENDER`, `DONE`
- `65563..`    prompt-position tokens (INPUT, BSP_NODE, WALL, EOS,
  TEX_COL, PLAYER_X/Y/ANGLE)

Exact numbers within each range can be assigned during implementation.

### Whether the existing design doc needs updating

`docs/design_byte_token_renderer_phase_a.md` describes the
post-Phase-A state at the token-sequence level (which hasn't
changed). Its "thinking value" language is slightly stale under the
embedding-matrix framing but still technically accurate (the value
*is* the token). Leave as-is for now and revisit when writing the
master plan. If inconsistent at read time, the plan + parts are
authoritative.

### Residual width budget

`d_embed = 72`. With `d_model = 2048`, that's 3.5% of the residual
stream. Bypass columns are on top of this (pixel rows, BSP coeffs,
etc. already consume substantial width today). Need to confirm
during implementation that the residual doesn't get tight at any
point; allocator should report if pressure is a concern.

## High-level task list (not exhaustive)

1. Generalize `torchwright/graph/embedding.py` — accept `d_embed` and
   `table` arguments; keep backwards compat for the 8-dim examples.
2. Build `torchwright/doom/embedding.py`: vocab list, hand-constructed
   `W_embed` with the 4+4+4+4 layout, category-code table for
   non-VALUE tokens, `embed_lookup(name)` / `value_id(int)` helpers.
3. Choose and implement the bypass declaration API.
4. Wire DOOM's compile path: token_ids input → embedding leaf → residual
   projection; bypass row → existing input-projection mechanism.
   Output side: gather embedding slice → MatMul(W_embed.T) → logits.
5. Rewire every stage's `equals_vector(token_type, E8_X)` → via
   embedding lookup. Detector list in game_graph's
   `_detect_token_types` rebuilt from the new vocab.
6. Delete `thinking_value` overlaid field and all references.
7. Port `thinking_wall.py` to the new carrier (M4 behavior preserved,
   10 new slots stubbed).
8. Run existing examples + `make test` + `make walkthrough` to
   validate.
