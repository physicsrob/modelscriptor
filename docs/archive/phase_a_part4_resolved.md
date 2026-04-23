# Phase A, Part 4 — RESOLVED + EOS gutting + RENDER migration

This is the part where collision resolution moves out of the prefill
and into the thinking phase, and where the rendered frame starts
depending on thinking-token values.

Three changes land together in one atomic commit:

1. **RESOLVED math** — the three stub slots (RESOLVED_X, RESOLVED_Y,
   RESOLVED_ANGLE) get real implementations. The identifier steps
   aggregate WALL-stage collision flags and emit post-collision
   player state via the thinking-token mechanism.
2. **EOS gutting** — the collision-resolution computation in
   `stages/eos.py` is deleted. EOS stays in the prefill as a
   semantic end-of-prompt marker with no graph work.
3. **RENDER migration** — RENDER reads post-collision player_x/y
   from the RESOLVED value positions via the Part 3 helper, instead
   of from the PLAYER broadcast.

Partial states of this part don't render correctly. Ship atomically.

## Strong opinions this part commits to

### RESOLVED aggregates collision flags from the prefill WALL stage

Approach (A) from the review: the RESOLVED_X_ID and RESOLVED_Y_ID
identifier steps use `attend_mean_where(is_wall, ...)` to aggregate
`hit_full/hit_x/hit_y` across the 8 WALL prefill positions. The math
is identical to the existing EOS collision-resolution code —
axis-separated wall sliding with `(use_new_x OR use_new_y)` gating
the velocity application.

Approach (B) — aggregating HIT_* thinking-token values via
content-matched attention across all 8 walls — is deferred. It
becomes the path once the prefill WALL stage is deleted, which is
post-Phase-A. The `thinking_readback` helper (Part 3) returns
"most recent," not "aggregate"; routing RESOLVED through it would
require a second cross-step mechanism, so Phase A stays on approach (A).

### PLAYER_X / PLAYER_Y semantics flip to pre-collision

In Parts 1–3, EOS computes collision and the host reads
`eos_resolved_x/y` from overflow, feeds those resolved values back
into PLAYER_X / PLAYER_Y prompt tokens. PLAYER broadcast is
post-collision.

In Part 4, EOS is gutted. Host feeds frame-start (pre-collision)
player position to PLAYER_X / PLAYER_Y prompt tokens. PLAYER
broadcast is now pre-collision.

This flip is the most visible architectural delta from Parts 1–3.
Every consumer of the PLAYER broadcast must be re-audited:

- RENDER previously read post-collision from the broadcast.
  Post-Part-4, RENDER reads post-collision from the RESOLVED
  thinking tokens via the helper. The broadcast is no longer used
  for position.
- Thinking-token identifier steps (HIT_*, RESOLVED_X/Y) previously
  read pre-collision from a bypass input. Post-Part-4, they read
  pre-collision from the broadcast.

cos/sin continue to come from PLAYER_ANGLE's broadcast — collision
doesn't change angle.

### The pre-collision player bypass is eliminated

Today thinking positions have bypass input slots for `player_x` /
`player_y` that the host fills with pre-collision values. This
bypass exists because PLAYER broadcast is post-collision pre-Part-4.

After point 2's flip, the broadcast *is* pre-collision. The bypass
is redundant. It's removed in the same Part 4 commit:

- Delete the bypass slot from `inputs["player_x"]` / `inputs["player_y"]`'s
  non-prompt-position use.
- Change thinking-stage KV inputs to read `player_x` / `player_y`
  from the broadcast (via `attend_mean_where(is_player_x, ...)`),
  matching the existing pattern.

PLAYER_X / PLAYER_Y prompt tokens still exist and still need
raw-input bypasses (how else does the host supply the position to
the prompt?), but the *thinking-position* bypass goes away.

### RENDER reads RESOLVED via the Part 3 helper

```python
resolved_x = readback.get_value_after_last("RESOLVED_X")
resolved_y = readback.get_value_after_last("RESOLVED_Y")
```

These return Nodes holding dequantized floats in their design-doc
ranges (`[-20, 20]`). RENDER's `_compute_precomputes` uses these
wherever it currently references `kv.player_x` / `kv.player_y`. No
new attention mechanism — the helper built for Part 3's derived
values is reused.

### Host reads resolved state from the emitted token stream

The compiled module does not emit a new overflow field for resolved
state. Host runs argmax on the output at known step offsets:

```
resolved_x_step     = max_walls * 27 + 1
resolved_y_step     = max_walls * 27 + 3
resolved_angle_step = max_walls * 27 + 5
```

Dequantizes each argmax'd token ID to its design-doc range, updates
`game_state` for the next frame. The existing trace fields
(`FrameTrace.eos_resolved_x/y/angle`) are populated from these
dequantized values — the semantics are preserved, the source
changes.

### EOS marker stays; computation goes

EOS remains in the prefill as an end-of-prompt marker. This is
semantically important — it's what separates the fixed-length
prompt from the autoregressive decode phase. The E8 code, the
position in the sequence, and the marker role all stay.

What goes:

- The collision-resolution math in `stages/eos.py` (all of `_resolve_collision`).
- `EosKVInput` / `EosToken` dataclasses.
- The EOS overflow outputs (`eos_resolved_x`, `eos_resolved_y`,
  `eos_new_angle`) from the graph.
- The EOS overflow reads in `compile.py` / `step_frame`.

`stages/eos.py` becomes a thin no-op file (or is deleted, with the
marker handled entirely by the token sequence — implementation
choice, non-load-bearing).

### The THINKING_WALL_0 host-injection hack is removed

Today the host synthesizes `THINKING_WALL_0` as the first
autoregressive token after PLAYER_ANGLE. This hack predates the
embedding-carrier design and shouldn't be perpetuated.

In Part 4, the state machine drives the transition end-to-end:
PLAYER_ANGLE's step emits `THINKING_WALL_0` as its output token.
The autoregressive loop then unrolls from there with no host
intervention.

Mechanically:

- PLAYER_ANGLE's stage output (next-token-id) is set to the
  embedding of `THINKING_WALL_0`. (Today it's the embedding of
  whatever arbitrary default; the host overrides by injection.)
- The host loop no longer synthesizes the first thinking token;
  it just argmaxes the PLAYER_ANGLE step's output and reads back
  `THINKING_WALL_0` naturally.

This is small in code but cleans up a cross-boundary hack.

### Part 4 ships atomic

The three changes (RESOLVED math, EOS gutting, RENDER migration)
are interlocked:

- Landing RESOLVED without migrating RENDER → RENDER reads stale
  post-collision from PLAYER broadcast (which is now pre-collision
  because EOS is gutted). Wrong output.
- Landing RENDER migration without RESOLVED math → RENDER reads
  stub RESOLVED values. Wrong output.
- Landing EOS gutting without RESOLVED + RENDER → host has no
  resolved state at all. Broken frame.

Ship all three in one commit. Do not attempt a partial PR.

## Scope

- Implement RESOLVED_X_ID / RESOLVED_Y_ID / RESOLVED_ANGLE_ID
  identifier-step computations (the 3 Part-2 stubs become real).
  Port from `stages/eos.py:_resolve_collision` for the X/Y math;
  RESOLVED_ANGLE is a passthrough of INPUT's `new_angle`.
- Replace `stages/eos.py`'s collision-resolution computation with
  a no-op (or delete the stage module if nothing else references
  its exports).
- Remove `EosKVInput`, `EosToken`, and their wiring in
  `game_graph.py`.
- Remove `eos_resolved_x`, `eos_resolved_y`, `eos_new_angle` from
  the overflow outputs.
- Update `compile.py` / `step_frame`:
  - Host feeds pre-collision game_state to PLAYER_X / PLAYER_Y.
  - Host does not inject THINKING_WALL_0.
  - Host reads resolved state from argmax output at step offsets.
  - Remove `eos_rx_out_s` / `eos_ry_out_s` / `eos_angle_out_s` reads.
- Change thinking-stage KV inputs to read pre-collision player from
  PLAYER broadcast (not bypass).
- Remove the pre-collision player_x / player_y bypass slots.
- PLAYER_ANGLE step's next-token-id output is `THINKING_WALL_0`'s
  embedding.
- RENDER's `_compute_precomputes` uses `readback.get_value_after_last("RESOLVED_X")`
  / `RESOLVED_Y` for post-collision position. cos/sin unchanged.
- Integration test: a collision-scenario walkthrough (player walks
  into a wall) produces a rendered frame matching reference within
  existing tolerances.

## Not in scope

- Deleting the EOS marker (explicitly kept as end-of-prompt marker).
- Deleting the prefill WALL stage. Future phase — RESOLVED still
  aggregates from prefill WALL in Phase A.
- The "aggregate values across walls" helper that would let RESOLVED
  use HIT_* thinking tokens instead of prefill WALL collision flags.
- RENDER's `col` / `chunk_k` / `wall_counter` overlaid state. Per
  CLAUDE.md, the per-pixel-output phase that eliminates these is
  post-Phase-A.
- The M4 SORTED softmax dilution regression and M4 affine-bounds
  looseness. Those go in Part 5.

## Acceptance criteria

1. RESOLVED_X / RESOLVED_Y / RESOLVED_ANGLE dual-path tests pass
   within design-doc quantization tolerance. Reference computation
   matches what the old EOS stage produced.
2. `stages/eos.py` contains no collision-resolution math.
3. `grep -r "eos_resolved_x\|EosKVInput" torchwright tests` returns
   zero hits.
4. `grep -r "E8_THINKING_WALL\[0\]" compile.py` returns zero hits
   (the host no longer injects THINKING_WALL_0).
5. `grep -r "player_x.*bypass" torchwright` — the pre-collision
   bypass slot is gone. PLAYER_X / PLAYER_Y prompt-token inputs
   retain their bypass (that's how the host supplies the prompt
   value), but no thinking-position bypass remains.
6. `make walkthrough ARGS="--scene box --frames 10"` matches
   reference within existing tolerances.
7. A collision-scenario integration test passes: scene with player
   pressed against a wall, one frame of forward movement, rendered
   output shows the player's sliding position (x or y unchanged on
   the blocked axis).
8. `test_pipeline.py`'s collision tolerance assertions (currently
   tied to `trace.eos_resolved_x` via the trace field) pass against
   the new RESOLVED path.
9. `make test` passes.

## Open questions

### Trace field renaming

`FrameTrace.eos_resolved_x` / `eos_resolved_y` / `eos_new_angle` are
now legacy names (EOS doesn't produce these anymore). Renaming to
`resolved_x` / `resolved_y` / `new_angle` is cosmetic. Touches
`trace.py` and every test that inspects the trace. Optional cleanup
— do it if the scope feels small, defer if not.

### Collision-scenario integration test

We need at least one scene that exercises wall-sliding on both
axes. Options:
- An existing test scene from `test_pipeline.py`.
- A new hand-authored box-room + corner-approach scene.
- A WAD-map scene that naturally produces collisions.

Implementation picks one; the doc doesn't need to pre-specify.

### Interaction with M4 SORTED dilution regression

Part 4 changes the thinking-phase length (from ~56 Part-1/2 steps
to ~222 Part-2/3/4 steps). The M4 SORTED dilution symptom was
triggered by the 56 extra steps from M4's addition of 3 HIT_*
pairs; the much larger expansion here may worsen it. Part 5
addresses the regression; Part 4 just measures.

If the regression becomes severe enough to block Part 4's
integration test (e.g., the collision scene times out on GPU), a
mitigation inside Part 4 may be unavoidable. Handle during
implementation.

## High-level task list (not exhaustive)

1. Implement RESOLVED_X_ID / RESOLVED_Y_ID identifier-step math.
   Port from `stages/eos.py:_resolve_collision`. Aggregate WALL
   collision flags via `attend_mean_where`, combine with
   pre-collision player + vel_dx/dy for resolved position.
2. Implement RESOLVED_ANGLE_ID as a passthrough of `input_out.new_angle`.
3. Gut `stages/eos.py` — remove `_resolve_collision`, `EosKVInput`,
   `EosToken`, `build_eos`'s implementation. Leave a stub or delete.
4. Remove EOS references from `game_graph.py`:
   `EosKVInput` import, `build_eos` call, EOS overflow output wiring.
5. Update `game_graph._assemble_output` to drop `eos_resolved_x/y`
   and `eos_new_angle` overflow fields.
6. Update `compile.py`:
   - Host feeds pre-collision game_state to PLAYER_X / PLAYER_Y.
   - Remove EOS overflow reads.
   - Remove THINKING_WALL_0 host injection; read first thinking
     token from PLAYER_ANGLE step's argmax.
   - Read resolved state from argmax output at
     `max_walls * 27 + {1, 3, 5}` step offsets; dequantize.
7. PLAYER stage: set next-token-id output at PLAYER_ANGLE positions
   to `embed_lookup("THINKING_WALL_0")`.
8. Thinking-stage KV inputs: read pre-collision player_x/y from
   PLAYER broadcast; remove the pre-collision bypass slot.
9. RENDER: replace `kv.player_x` / `kv.player_y` references in
   `_compute_precomputes` with `readback.get_value_after_last(...)`.
10. Write collision-scenario integration test.
11. Run `make test` and `make walkthrough`; confirm rendered frame
    matches reference with the collision path re-routed.
12. Measure depth with `make graph-stats`; confirm RESOLVED
    identifier steps ≤ 15 layers.
