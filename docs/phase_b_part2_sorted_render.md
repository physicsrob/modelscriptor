# Phase B, Part 2 — SORTED restructure, RENDER migration

SORTED picks walls via a content-based attention over thinking-token
BSP_RANK values, emits the picked `wall_index` as a VALUE token
(replacing overlay state), and RENDER reads `wall_index` plus
`vis_lo` / `vis_hi` via attention instead of overlay.

## Prerequisites

- `main` at commit `bf6ce48` or later (Phase A Part 4).
- The `BSP_RANK` thinking token emits `bsp_rank` as a 16-bit VALUE
  per Phase A Part 3. Part 2 adds auxiliary scalar channels to its
  emit (see §1 below).
- The existing `VIS_LO` / `VIS_HI` thinking tokens emit correct
  values per Phase A Part 3. RENDER's new reads query them by
  content match.
- No dependency on Part 1. Parts 1 and 2 can land in either order
  or concurrently.

## Development speed

Skip: formal output-assembly regression suite, host-protocol
documentation updates beyond what's load-bearing, dilution
diagnostics for the old SORTED code (moot after rewrite). Smoke
test = `make walkthrough` matches reference with wall identity +
vis bounds now flowing through attention.

## What's landing

Three interlocked changes. SORTED's mechanism rewrites; its output
format changes; RENDER's consumption pattern changes. All three
ship together because intermediate states don't render correctly.

### 1. SORTED's new attention — quadratic-equality over thinking tokens

Today SORTED uses `attend_argmin_above_integer` over prefill WALL
positions. Each WALL position carries an 8-wide `indicators_above`
thermometer (rank-versus-threshold × renderability); the attention
picks the smallest rank above the current threshold.

Part 2 replaces this with a direct equality attention. The SORTED
token at ordinal `N` wants the wall whose `bsp_rank == N` (among
renderable walls). The attention score peaks at equality via a
quadratic decomposition:

```
score(wall_w) = -(bsp_rank_w - N)²
              = -bsp_rank_w² + 2·N·bsp_rank_w - N²
```

The `-N²` term is query-only, constant across keys, falls out of
softmax. The `2·N·bsp_rank_w` term is a standard `query · key` dot
product. The `-bsp_rank_w²` term is key-only — a bias attached to
the key side.

**Implementation:**

- At each BSP_RANK thinking position, emit two scalars on the
  value-side KV in addition to the 16-bit VALUE embedding:
  - `bsp_rank_scalar` — the integer rank 0..7 for renderable walls;
    `-1000` (sentinel) for non-renderable. The sentinel pushes the
    quadratic score far negative so non-renderable walls drop out
    of the softmax.
  - `bsp_rank_neg_sq` — `-bsp_rank_scalar²`. For renderable walls
    this is in `[-49, 0]`. For non-renderable walls it's
    `-1_000_000`, which also drops the attention weight.
- At the SORTED position, the query for the quadratic attention is
  `[2N, 1]`. The key at each BSP_RANK thinking position is
  `[bsp_rank_scalar, bsp_rank_neg_sq]`. The dot product equals
  `-(bsp_rank - N)² + N²`. After softmax the `+N²` offset cancels.
- The attention's value-side carries the wall_index one-hot (already
  in each thinking token's residual from Phase A Part 2). The
  attention returns the picked wall's 8-wide one-hot.

**Softmax gain.** The score gap between a matching wall and its
nearest non-matching neighbour is `-(1)² = -1`. At gap −1 a gain
of ~20 yields softmax weight > 0.999 on the matching wall. Tune
empirically during implementation; start at 20.

**Exhaustion detection.** When `N` exceeds the number of renderable
walls, no wall's rank matches `N`. Softmax spreads across walls.
Detect post-hoc: the same attention also returns the picked wall's
`bsp_rank_scalar` (value-side carries both the one-hot and the
scalar rank). If the returned rank differs from `N`, emit
`sort_done`. Equivalent to today's exhaustion check — the signal
source is different (scalar comparison vs. threshold overflow) but
the semantic is the same.

### 2. SORTED emits wall_index as a VALUE token

Today SORTED seeds `render_wall_j_onehot`, `render_vis_lo`,
`render_vis_hi`, `render_tex_id` into the overlay. These are
forwarded by subsequent RENDER tokens within the same wall via the
host's overlay-copy loop.

After Part 2, SORTED emits the picked wall_index as a standard
16-bit VALUE token (0..7) via the factored 4+4+4+4 one-hot payload
— the same embedding shape every other VALUE token uses. The
SORTED marker is followed by a VALUE step that emits `wall_index`.
From the transformer's perspective, SORTED behaves like the
identifier-value pattern used throughout the thinking phase.

**Distinct identifier category.** The SORTED VALUE is tagged with
a new identifier flag (`SORT_RESULT`) so downstream consumers can
distinguish it from other VALUE tokens. `SORT_RESULT` is added to
the vocabulary as a new identifier entry.

**Host loop.** The host stops reading `render_wall_j_onehot`,
`render_vis_lo`, `render_vis_hi`, `render_tex_id` from the
overlay. It continues to copy the remaining overlay fields
(`render_col`, `render_chunk_k`, `render_mask`, `token_type`,
`sort_position_index`).

### 3. RENDER reads wall identity via attention

Today RENDER reads `render_wall_j_onehot` from overlay. After
Part 2:

- RENDER does `attend_most_recent_matching(is_SORT_RESULT_value)`
  — Part 3's existing primitive, keyed on the new `SORT_RESULT`
  identifier flag. Returns the scalar wall_index (0..7).
- `map_to_table` converts the scalar to an 8-wide one-hot.
- The one-hot feeds RENDER's existing WALL-geometry attention
  (`attend_argmax_dot` against WALL prefill positions) unchanged.

**Stability across RENDER tokens.** The first RENDER token after a
SORTED pair reads the just-emitted SORT_RESULT. Subsequent RENDER
tokens painting additional columns/chunks for the same wall keep
reading the same SORT_RESULT (the next SORTED marker hasn't fired).
Wall identity is stable without any overlay carry.

**Wall transition.** When RENDER's state machine decides "done with
this wall," it emits the next SORTED marker as its output token
(via the embedding-driven next-token path). The next SORTED fires,
emits its wall_index VALUE, subsequent RENDER reads pick up the
new wall identity automatically.

### 4. RENDER reads vis_lo, vis_hi via content attention

Today `render_vis_lo`, `render_vis_hi` flow through overlay from
SORTED. After Part 2:

- RENDER queries `(identifier=VIS_LO, wall_index=W)` against
  thinking-token positions. The `VIS_LO` thinking token for wall W
  carries both the identifier flag and the wall_index one-hot in
  its KV (from Phase A Part 2's identifier tagging and marker
  broadcast). The attention peaks at the matching position and
  returns the scalar `vis_lo`.
- Same for `vis_hi`.
- Two attention heads per RENDER token.

`W` comes from §3: the wall_index RENDER just resolved from the
SORT_RESULT VALUE. Both reads happen at layer 0 of RENDER's step
(the thinking tokens finished writing before any RENDER fires).

### 5. tex_id routing

Today `render_tex_id` flows through overlay. After Part 2, RENDER
reads `tex_id` directly from prefill WALL by wall_index content
match — WALL's KV carries `tex_id` as part of raw geometry, and
the existing `attend_argmax_dot` on the wall_index one-hot already
retrieves it. No new attention needed; the field is just extracted
alongside `ax`, `ay`, `bx`, `by`.

### 6. Overlay cleanup

Remove from the overlaid output region:

- `render_wall_j_onehot` (`max_walls`-wide)
- `render_vis_lo` (1 scalar)
- `render_vis_hi` (1 scalar)
- `render_tex_id` (1 scalar)

Update `game_graph._assemble_output` to stop including them. The
overlay region narrows by `max_walls + 3` columns.

Retained overlaid outputs (still load-bearing for RENDER's state
machine): `render_col`, `render_chunk_k`, `render_mask`,
`token_type`, `sort_position_index`. These are Phase C concerns.

## Scope

- At BSP_RANK thinking token: add `bsp_rank_scalar` and
  `bsp_rank_neg_sq` scalar channels on the value-side KV. Gate by
  `is_renderable`: non-renderable walls get `bsp_rank_scalar =
  -1000`, `bsp_rank_neg_sq = -1_000_000`.
- Add `SORT_RESULT` identifier to the vocabulary.
- Restructure SORTED marker + VALUE:
  - The SORTED marker is the existing `SORTED_0..7` vocabulary
    entry (unchanged).
  - The SORT_RESULT identifier + VALUE step follows the marker:
    the identifier step emits `SORT_RESULT` token ID; the VALUE
    step emits the picked `wall_index` as the 16-bit VALUE via the
    standard factored embedding.
- Rewrite SORTED's attention in `stages/sorted.py`:
  quadratic-equality attention over BSP_RANK thinking positions.
  Return wall_index one-hot + picked rank (for exhaustion check).
- In `stages/render.py`: replace the `render_wall_j_onehot` overlay
  read with `attend_most_recent_matching(is_SORT_RESULT_value)` +
  `map_to_table` to reconstruct the one-hot.
- Add two content-attention reads at RENDER for `vis_lo`,
  `vis_hi` against thinking-token positions.
- Drop `render_wall_j_onehot`, `render_vis_lo`, `render_vis_hi`,
  `render_tex_id` from overlaid outputs. Update
  `game_graph._assemble_output`, the output-assembly code, and any
  consumers (tests, trace fields) that referenced them.
- Update the host's step loop in `compile.py`: stop copying the
  removed overlay fields.
- Wall transition: RENDER emits the next SORTED marker's token ID
  (not the E8 overlay code) when advancing walls. This is a
  straightforward embedding output change — similar to how Part 4
  changed PLAYER_ANGLE to emit THINKING_WALL_0.

## Not in scope

- Any changes to thinking-slot structure beyond adding scalar
  channels to BSP_RANK. Part 1 owns the T_STAR_L/R, COL_A/B split
  and running accumulators.
- Prefill WALL gutting (Part 3).
- Removing prefill WALL's `indicators_above` / `sort_value` payload
  production. After Part 2, nothing reads these outputs; Part 3
  deletes them.
- RESOLVED migration (Part 3).
- Final depth measurement (Part 4).
- SORTED softmax dilution diagnostic from M4 / Phase A — the
  dilution was an artefact of the old `attend_argmin_above_integer`
  primitive operating over a long thinking-phase sequence. The new
  quadratic attention is a different primitive with different
  characteristics; if dilution still appears, diagnose then, but
  the expectation is it's moot.

## Smoke test

`make walkthrough ARGS="--scene box --frames 10"` matches reference.
The walkthrough exercises both SORTED's wall selection and
RENDER's wall-identity + vis bounds reads. If both pass, Part 2 is
correct end-to-end.

A `--scene multi` walkthrough is informative too (exercises
non-trivial wall orderings) but one scene's reference match is
sufficient smoke for Part 2. Part 4 runs both.

## Interaction with Part 1

Both parts edit `stages/thinking_wall.py`:

- Part 1 adds new slot emits and restructures the factor cascade.
- Part 2 adds scalar channels to the BSP_RANK thinking token's
  emit (on the value-side KV for SORTED's quadratic attention).

The edits touch different functions and different thinking slots;
merge conflicts unlikely. Neither part depends on the other.
Either order of landing works.

Part 2 also edits `stages/sorted.py`, `stages/render.py`,
`game_graph.py`, and `compile.py` — Part 1 doesn't touch these.

## Open questions

- **Softmax match-gain for the quadratic attention.** Start at 20;
  tune if concentration falls below 0.999 on any scene. The score
  gap at rank±1 is 1, so gain directly sets concentration.
- **Exhaustion-signal wiring.** Today `sort_done` is an overflow
  output. The new post-hoc rank-vs-N check produces the same
  signal. Confirm it slots into RENDER's existing state machine
  cleanly.
- **Overlay removal ordering.** If Part 2 lands before Part 3,
  prefill WALL still produces `indicators_above` and the
  `sort_value` payload; nothing reads them. Dead code until Part 3
  deletes it. Acceptable intermediate state.

## High-level task list

1. Emit `bsp_rank_scalar` and `bsp_rank_neg_sq` at BSP_RANK thinking
   positions, gated by `is_renderable`.
2. Add `SORT_RESULT` identifier to vocabulary and prev-id table.
3. Rewrite SORTED's attention: quadratic-equality dot product over
   BSP_RANK thinking positions. Capture wall_index one-hot + picked
   scalar rank.
4. Emit the picked wall_index as a VALUE token at the SORT_RESULT
   position.
5. RENDER: replace overlay read of `render_wall_j_onehot` with
   `attend_most_recent_matching(is_SORT_RESULT_value)` +
   `map_to_table`.
6. RENDER: add content-attention reads for `vis_lo`, `vis_hi` on
   thinking positions by `wall_index`.
7. Update wall-transition state machine: emit next SORTED marker's
   token ID when advancing walls.
8. Drop `render_wall_j_onehot`, `render_vis_lo`, `render_vis_hi`,
   `render_tex_id` from overlaid outputs.
9. Update host loop (`compile.py`) to stop copying removed overlay
   fields.
10. `make walkthrough` — reference match.
