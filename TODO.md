# TODO

## fp16 inference for Flash Attention

The render decode loop runs at ~35ms/step (GPU-bound). The bottleneck is SDPA
over long sequences (~800 positions): each step attends over an 800×800-ish
matrix using PyTorch's materialized ("math") softmax kernel.

**Why the math kernel is used:** PyTorch's SDPA dispatcher only enables Flash
Attention for fp16/bf16 inputs. All model weights and activations are currently
fp32, so every SDPA call falls back to the materialized kernel regardless of
whether an `attn_mask` is provided.

**The opportunity:** Casting weights and activations to fp16 at inference time
(e.g., `.half()` on `CompiledHeadless._net`) would let SDPA dispatch to Flash
Attention on A100. Flash Attention is O(N) in memory and substantially faster
for long sequences — potentially 3–5× faster SDPA, which could bring decode
from ~35ms to ~15–20ms per step.

The dynamic decode path (`forward_cached`, `is_causal=False`, no explicit mask)
already satisfies all Flash Attention preconditions except dtype. The static
KV cache path (added in this branch) uses a float `attn_mask` and would need
to switch to a boolean mask or be dropped in favour of the dynamic path.

**Risk:** Weights are compiled in fp32 and encode precise numerical thresholds
(comparison results, attention argmin/argmax patterns). fp16 has ~3 significant
decimal digits vs ~7 for fp32. It's unknown whether the compiled logic survives
the precision loss without correctness failures. A test that compares fp32 vs
fp16 `step_frame` outputs would answer this.

**Suggested experiment:**
1. `module._net.to(torch.float16)` after `compile_game()`
2. Cast inputs to fp16 in `_build_res_stream`
3. Run `make walkthrough ARGS="--frames 5"` and compare to fp32 reference
4. If pixel errors are within tolerance, fp16 is viable

Files: `torchwright/compiler/export.py` (`_build_res_stream`, `CompiledHeadless`),
`torchwright/compiler/components/attn.py` (weights), `torchwright/doom/compile.py`

## Move ceiling/floor into the transformer

The host currently decides ceiling vs floor per pixel (`ceil if y < center_y
else floor_c`). This is computation on the host side. The transformer should
emit ceiling/floor pixels itself — either as part of each render chunk (fill
non-wall rows within the chunk) or as a separate post-render pass.

Files: `torchwright/doom/compile.py` (step_frame render loop),
`torchwright/doom/game_graph.py` (render output)

## Clip column iteration to screen bounds

The state machine iterates col_lo..col_hi, which can include negative columns
and columns >= W (e.g., cols=[-2, 122) with W=120). The host safely skips
out-of-bounds columns, but each is still a full autoregressive step (~70ms).
At 120x100 with 4 full-screen walls, ~16 steps are wasted per frame.

Fix: clamp col_lo/col_hi to [0, W) in the graph, or have the state machine
skip out-of-bounds columns via the feedback loop.

Files: `torchwright/doom/game_graph.py` (state machine col iteration)

## Make graph-side done_flag work when N < max_walls

The graph checks `mask_sum > max_walls - 0.5`, which only fires when all
max_walls slots are masked. When the scene has fewer walls, the host
terminates via bit-count instead. The graph's done_flag computation runs on
every RENDER token but never triggers — wasted graph nodes.

Options:
- Feed N as a graph input so the threshold is dynamic
- Detect sentinel wall data (score=99) after all real walls are masked
- Accept the current host-side fix and remove the graph-side done_flag entirely

Files: `torchwright/doom/game_graph.py` (state_transitions),
`torchwright/doom/compile.py` (host-side termination)

---

# Known test flakiness & intermittent failures

Three distinct failure modes in the full `make test` run that looked
like flakiness on the surface.  Modes A and B are fixed; Mode C is
partially localized but not yet root-caused.

## Mode A — `test_fuse_chain_of_three` (order-dependent bug) — FIXED

`fuse_consecutive_linears` collected `(L1,L2)` and `(L2,L3)` candidates
via `for node in all_nodes` (set iteration).  Certain global_node_id
offsets (e.g., 5, where slot `8 % 8 = 0` puts L3 before L2 in CPython
set order) caused `(L2,L3)` to be processed first, leaving L3 depending
on L1 — which a second `while True` pass then fused again, yielding
`total=3` instead of the expected `2`.

Fix: sort the fusion candidate list by `l1.node_id` before the mutation
loop, so upstream pairs are always fused first.

Regression test: `test_fuse_chain_ordering_regression` in
`tests/graph/test_optimize.py` loops through offsets `[0, 5, 101, 997]`
and asserts `total == 2` at each.

Files: `torchwright/graph/optimize.py:87-91`,
`tests/graph/test_optimize.py` (new regression test).

## Mode B — pixel-precision cluster (counter-shift drift) — FIXED

`Node.__hash__` returns `node_id`, so set/dict iteration order in the
compiler's `graph_analysis.py` and `residual_assignment.py` depends on
the global counter.  Earlier tests that allocate nodes shift the
counter, which shifts residual-column assignments, which drifts
compiled pixel values by a few percent — enough to fail the tightened
0.45 tolerance in `test_game_graph.py::test_renders_*`.

Fix: autouse fixture `reset_node_id_counter` in `tests/conftest.py`
resets `torchwright.graph.node.global_node_id = 0` before every test,
so node IDs are stable regardless of suite ordering.

Files: `tests/conftest.py` (new autouse fixture).

## Mode C — sort concentration failure at angle-192 — FIXED (2026-04-14)

Root cause: `_compute_bsp_rank` packed two unrelated concerns — BSP rank
and renderability — into a single score via a sentinel-plus-
`wall_index*0.1`-tiebreak encoding.  The 0.1-wide budget for tiebreak
survived reference eval but compiled to a 0.053 gap (~47% loss), which
gave the softmax only 4.25 logit of separation at sort[2] — enough to
blend 1.4% of west's geometry into north's and produce the observed
`[4.86, 5.00, -5.00, 4.86]` drift.

### Fix

Surface renderability as a first-class `is_renderable` ±1 validity
signal from the WALL stage and swap SORTED's argmin over to a new
`attend_argmin_valid_unmasked` primitive.  The sort score is now a
clean integer BSP rank (1.0 spacing, no sentinel, no tiebreak); the
validity path contributes `_QUERY_GAIN · _VALIDITY_LARGE = 80000` to
every valid key, so valid vs. invalid separation is ~160000 logit
units — Mode C's concentration failure is now impossible by
construction.

Asserts tightened at the same time:
- `assert_distinct_across(sort_score, is_wall, margin=0.8)` (was 0.5)
- `assert_score_gap_at_least(checked_score, is_wall, margin=0.5)` (was 0.05)

Both are comfortably satisfied by integer-rank spacing.

Files: `torchwright/ops/attention_ops.py` (+
`attend_argmin_valid_unmasked`), `torchwright/doom/stages/wall.py`
(`_compute_bsp_rank` returns `(rank, is_renderable)`;
`WallOutputs.is_renderable`), `torchwright/doom/stages/sorted.py`
(`SortedInputs.is_renderable`, primitive swap, tighter asserts),
`torchwright/doom/game_graph.py` (wiring), plus tests.

Commit: see `feat: surface wall renderability as a first-class SORTED
validity signal`.

### Verification at angle-192

- `test_inspect_sorted_argmin_attention_weights_at_sort2` now runs the
  capture at sort[0] (the only non-vacuous step — sort[1..3] degenerate
  into masked-valid re-picks of south).  Weight on south ≈ 1.0;
  valid/invalid logit gap is orders of magnitude above the design
  separation needed.
- `step_frame` debug print at angle-192 shows sort[0..3] all select
  south cleanly: `wall=[-5.00,-5.00,5.00,-5.00]`.  No blend.
- `test_angle_192_validity_excludes_invalid_clean_pick` (renamed from
  `test_angle_192_sentinel_ties_clean_pick`) passes with the new
  validity-first API.

### Known residual (not Mode C)

`test_bsp_rank_integration.py::test_renders_in_all_four_directions[192]`
still fails with max pixel error 0.500 (same magnitude as before the
fix).  The sort is now clean, so this is no longer a wall-selection
issue — it's a separate render-precision drift at angle 192 that the
0.35 threshold (originally sized against wrong-wall errors of 0.35-0.7)
catches.  Also tracked as part of the wider render-precision cluster
under `test_game_graph.py::test_renders_from_angle[192]`,
`test_renders_oblique_angle[20/100/160/210]`,
`test_renders_off_center_oblique[(3,2)@20/(1,-3)@50]` — all 0.4-0.6
pixel errors that predate this fix.

### End-of-sort wastefulness (follow-up)

With `N_renderable < max_walls`, after all renderable walls are picked
the masked-valid fallback re-picks the last wall at every subsequent
sort step.  RENDER overdraws; the host's `filled[y, c]` pixel dedup
hides the waste.  A principled fix is a compiled "done" signal from
SORTED that lets downstream stages short-circuit.  Flagged as a TODO
near the `attend_argmin_valid_unmasked` call site in `sorted.py`.
