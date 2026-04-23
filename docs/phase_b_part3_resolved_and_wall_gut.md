# Phase B, Part 3 — RESOLVED migration, prefill WALL gutted

RESOLVED reads the running-accumulator HIT_* values from thinking
tokens (the Part 1 mechanism) and applies axis-separated sliding.
With SORTED (Part 2) already reading from thinking tokens, prefill
WALL has no remaining computational consumers and is reduced to a
raw-data carrier.

## Prerequisites

- Part 1 landed: `HIT_FULL`, `HIT_X`, `HIT_Y` thinking tokens emit
  running OR values (each wall's emitted value is
  `own_flag OR aggregate_through_previous_walls`). Wall 7's
  emitted HIT_* values are the global OR across all 8 walls.
- Part 2 landed: SORTED reads `bsp_rank` / `is_renderable` from
  BSP_RANK thinking positions via quadratic-equality attention.
  RENDER reads wall identity and `vis_lo` / `vis_hi` via content
  attention. Prefill WALL's computation has zero remaining
  downstream consumers.
- The existing RESOLVED identifier steps (`RESOLVED_X`,
  `RESOLVED_Y`, `RESOLVED_ANGLE`) are wired into the thinking
  state machine (Phase A Part 4) and compute via
  `attend_mean_where` over prefill WALL collision flags.

## Development speed

Skip: expanded regression tests beyond existing collision
walkthroughs, detailed WALL-deletion validation, trace-field
cleanup. The existing Part 4 collision scenario already exercises
RESOLVED's math; if it matches reference after Part 3's rewrite,
the migration is correct. Smoke test = walkthrough + collision
scenario match reference.

## What's landing

### 1. RESOLVED reads running accumulators

Today `_compute_resolved` in `stages/thinking_wall.py` aggregates
per-wall collision flags across prefill WALL positions:

```python
resolve_attn = attend_mean_where(
    pos_encoding, validity=is_wall,
    value=Concatenate([hit_full_01, hit_x_01, hit_y_01]),
)
avg_hf = extract_from(resolve_attn, 3, 0, 1, "tw_avg_hf")
avg_hx = extract_from(resolve_attn, 3, 1, 1, "tw_avg_hx")
avg_hy = extract_from(resolve_attn, 3, 2, 1, "tw_avg_hy")

any_hit_full = compare(avg_hf, 0.05)
any_hit_x    = compare(avg_hx, 0.05)
any_hit_y    = compare(avg_hy, 0.05)
```

Then applies axis-separated sliding:

```python
use_new_x = NOT(any_hit_full AND any_hit_x)
use_new_y = NOT(any_hit_full AND any_hit_y)
resolved_x = select(use_new_x, player_x + vel_dx, player_x)
resolved_y = select(use_new_y, player_y + vel_dy, player_y)
```

After Part 1, each HIT_* thinking token emits the running OR
through previous walls. By the time any RESOLVED identifier step
fires (after wall 7's last HIT_Y), the global OR for each ray is
already sitting in the KV cache as the most recent HIT_FULL /
HIT_X / HIT_Y value.

Part 3 replaces the cross-position aggregation with direct reads:

```python
any_hit_full = readback.get_value_after_last("HIT_FULL")  # global OR
any_hit_x    = readback.get_value_after_last("HIT_X")
any_hit_y    = readback.get_value_after_last("HIT_Y")
```

The sliding math is unchanged. The attention pattern simplifies
from three `attend_mean_where` gathers + three threshold compares
to three `attend_most_recent_matching` reads (Part 3 readback
primitive, one attention each, layer 0 reads).

RESOLVED's own depth drops from ~8 ops to ~3. Every following
RENDER token that reads `resolved_x` / `resolved_y` via the
existing readback benefits from RESOLVED finishing earlier.

**How the running accumulator works (reminder).** Each HIT_FULL
thinking token for wall `W` emits:

```
emitted_value = hit_full_W OR running_aggregate_through_W_minus_1
```

The right-hand side is read via `attend_most_recent_matching(
is_HIT_FULL_value)` — returns zero if no earlier HIT_FULL exists
(wall 0's case). The OR is implemented as `saturate(a + b)` over
0/1 inputs. By the last wall's HIT_FULL step, the emitted value is
`hit_full_0 OR hit_full_1 OR ... OR hit_full_7` — the global
aggregate. `get_value_after_last` at the RESOLVED position returns
this global value.

### 2. Prefill WALL gutted to a data carrier

After Parts 1 and 2 land, the only reads from prefill WALL
positions are raw geometry: `ax`, `ay`, `bx`, `by`, `tex_id`, and
the BSP coefficients. These are read by thinking tokens (via
content attention on wall_index) for rotation products and by
RENDER (via the same attention) for its own geometry precompute.

Everything else that prefill WALL computes is now unread:

- **Collision detection** (three ray-segment intersection tests
  producing `hit_full`, `hit_x`, `hit_y`). RESOLVED reads from
  thinking accumulators after Part 3.
- **BSP rank computation** (`dot(wall_bsp_coeffs, side_P_vec) +
  wall_bsp_const`). SORTED reads `bsp_rank` from BSP_RANK thinking
  tokens after Part 2.
- **Renderability flag** (`|sort_den| > ε AND num_t × sign(den) >
  0`). SORTED reads gated BSP_RANK scalars after Part 2.
- **Visibility columns** (rotate endpoints → FOV clip via
  reciprocal + `multiply_2d` → atan projection via `low_rank_2d`
  → column-fold select). RENDER reads `vis_lo` / `vis_hi` from
  thinking tokens after Part 2.
- **`indicators_above` thermometer construction.** Was the
  key-side payload for the old `attend_argmin_above_integer`.
  Unused after Part 2's quadratic-equality attention replaces
  that primitive.
- **`sort_value` payload packing.** Was the value-side payload
  for SORTED to forward to RENDER. Unused after Part 2.

Part 3 deletes all of the above from `stages/wall.py`. The stage
retains:

- Raw geometry in residual: `ax`, `ay`, `bx`, `by`, `tex_id`,
  `wall_bsp_coeffs`, `wall_bsp_const` (host-supplied bypasses).
- Wall_index one-hot in residual (constructed via `map_to_table`
  from the wall marker's value — load-bearing for content
  addressing from thinking tokens and RENDER).

The stage's critical path drops from ~62 ops to ~1–2 ops (just the
wall_index one-hot construction).

### 3. Overflow / overlay cleanup

Remove any WALL-side overflow outputs that existed solely to
support the computed fields. Any output-assembly code referencing
deleted fields is updated. Tests that referenced WALL's computed
outputs (if any remain) are deleted or adjusted.

## Scope

- Rewrite `_compute_resolved` in `stages/thinking_wall.py` to read
  running-OR HIT_* values via `readback.get_value_after_last`.
  Apply the unchanged sliding math.
- Delete collision-detection code from `stages/wall.py`
  (`_compute_collision_flags` and associated helpers).
- Delete the visibility-columns chain from `stages/wall.py`
  (`_compute_visibility_columns`, `_plane_clip_contribs`,
  `_endpoint_to_column`, and associated helpers).
- Delete `bsp_rank` computation, `is_renderable` computation,
  `indicators_above` construction, and `sort_value` payload
  packing from `stages/wall.py`.
- Keep raw geometry storage and the `wall_index` one-hot
  construction at WALL positions.
- Update `game_graph.py` to reflect the narrowed WALL outputs
  (the `wall_out` dataclass / return signature narrows).
- Update `game_graph._assemble_output` (and any other output
  assembly) to stop referencing deleted WALL fields.
- Remove any WALL-related overflow outputs that are no longer
  consumed.

## Not in scope

- WALL-as-identifier-value-pairs prefill refactor. WALL stays as
  N single-token-per-wall prompts; restructuring each wall into
  109 tokens is Phase C.
- Deleting the WALL stage module entirely. It remains as a minimal
  data-carrier stage producing the wall_index one-hot and storing
  raw geometry.
- Trace-field renaming (`FrameTrace.eos_resolved_x` etc. are still
  legacy names from the pre-Phase-A EOS stage). Cosmetic; defer.
- Affine-bounds looseness fix from Phase A Part 5's carryover. The
  looseness source was the geometry-attention outputs in
  thinking_wall that read from prefill WALL's visibility columns
  — after Part 3, those reads route to raw geometry directly, so
  the looseness source is gone. If `test_affine_bounds` still
  regresses after Part 3, diagnose then.
- Final depth measurement and completion note (Part 4).

## Smoke test

- `make walkthrough ARGS="--scene box --frames 10"` matches
  reference.
- `make walkthrough ARGS="--scene multi --frames 10"` matches
  reference.
- The collision-scenario integration test from Phase A Part 4
  passes (player walks into a wall; resolved position shows wall
  sliding on one axis).

If all three pass, the RESOLVED migration is correct and prefill
WALL's deletion hasn't removed any load-bearing computation.

## Open questions

- **Numerical tolerance through the running-accumulator chain.**
  Each HIT_* thinking token applies OR via a Linear + saturate on
  0/1 inputs. Eight applications through the wall sequence stay
  bounded because saturation clamps at 1. But the
  `attend_most_recent_matching` attention softmax has its own
  noise floor; accumulation over 8 walls may soften the 0/1
  boundary slightly. Verify via the existing collision walkthrough
  — if `resolved_x` / `resolved_y` diverge from reference near
  the collision boundary, bump the readback match-gain.
- **stages/wall.py's shape after gutting.** The remaining
  `wall_index` one-hot construction is trivial (~1–2 ops). Two
  options: keep the file as a minimal data-carrier stage, or
  delete the file and inline the remaining logic into
  `game_graph.py`. Decide during implementation based on what
  reads cleaner. Neither affects correctness.

## High-level task list

1. Rewrite `_compute_resolved`: read running-OR HIT_* values via
   `readback.get_value_after_last`; apply sliding math.
2. Delete collision-detection code from `stages/wall.py`.
3. Delete visibility-columns chain from `stages/wall.py`.
4. Delete `bsp_rank`, `is_renderable`, `indicators_above`,
   `sort_value` from `stages/wall.py`.
5. Update `game_graph.py`: narrow WALL outputs dataclass, drop
   references to deleted fields.
6. Update output-assembly code: drop any remaining deleted-WALL
   references.
7. Remove WALL-related overflow outputs that are no longer
   consumed.
8. Run box + multi walkthroughs and the collision scenario;
   confirm reference match.
