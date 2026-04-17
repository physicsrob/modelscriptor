# Phase E xfail: post-mortem

**Test:** `tests/doom/test_game_graph.py::TestGameGraph::test_renders_off_center_oblique[3.0-2.0-20]`
**Xfail landed:** commit `c2d5a7a` (Phase E)
**Investigation:** commit TBD (this doc)

## Status

- **Proximate cause:** Precisely characterized below.
- **Root cause:** Partially characterized. The specific residual-column aliasing (or equivalent structural fault) has not been named — doing so requires compiler-internals inspection beyond what the Phase-6 investigation deliverables cover.
- **Fix:** Not landed. The xfail remains, with an honest reason (updated from "precision loss near the edge of FOV") and a pointer to this doc.

## What happens at scene (px=3, py=2, angle=20)

At sort[0] (the first SORTED token after EOS), the SORTED stage's attention returns a value whose `sel_bsp_rank` residual column reads **-1171.875** instead of a clean integer in `[0, max_walls-1]`. `sort_done` correctly detects the nonsense (`-1 - -1171.875 = +1170.875 > -0.5`), fires, and the SORTED stage's sentinel replaces the output with 99. Downstream THINKING then picks the 99-marked cached position (or blends), producing garbage wall data that RENDER draws incorrectly.

All observations below use the diagnostic harness from plan 1; the investigation script lives at `scripts/investigate_phase_e.py`.

### Attention softmax at sort[0]

The SORTED stage's `attend_argmin_above_integer` concentrates with weight `1.000000` on **position 86 — SORTED[0] itself**, not on any of the four WALL positions. Per-head-representative logits at the query row:

| Position | Label               | Logit      | Weight    |
|---------:|---------------------|-----------:|----------:|
| 86       | SORTED[0] (self)    | +800.0000  | 1.000000  |
| 85       | EOS                 | +748.5167  | 0.000000  |
| 84       | WALL[3] (south)     | +636.8276  | 0.000000  |
| 81       | WALL[0] (east)      | +613.7040  | 0.000000  |
| 82       | WALL[1] (west)      | +555.9678  | 0.000000  |
| 83       | WALL[2] (north)     | +554.3101  | 0.000000  |

### What the primitive math says these should be

`attend_argmin_above_integer`'s logit formula at position `i` under the threshold-0 query is:

```
logit_i = _QUERY_GAIN · (−score_i + tiebreak · pos_scalar_i)
       + _ABOVE_BONUS · indicators_above_i[0]
```

With `_QUERY_GAIN = 8`, `_ABOVE_BONUS = 1000`, and the graph-level intended values:

* **Renderable WALL at rank 0:** `score=0`, `indicators_above[0]=1` → `logit ≈ 0 + 1000 = +1000`.
* **Non-WALL position (SORTED[0], EOS, BSP_NODE, TEX_COL):** `score=0` (zero `wall_bsp_coeffs`/`wall_bsp_const`), `indicators_above[0]=0` (gated by `is_renderable=false` at non-WALL) → `logit ≈ 0 + 0 = ~0`.

### What the compiled module actually produces

The WALL-position logits are **≈ 400 below design** (+555 to +636 instead of +1000). SORTED[0]'s logit is **≈ 800 above design** (+800 instead of ~0).

For the primitive math to produce these, one of the following must hold at non-WALL positions:

1. **`score ≈ -100`** at non-WALL positions. That would give `_QUERY_GAIN · 100 = +800` in column 0.
2. **Or `indicators_above[0] ≈ 0.8`** at non-WALL positions. That would give `_ABOVE_BONUS · 0.8 = +800` in the indicator columns.

Either way, **something in the compiled residual stream at non-WALL positions carries a non-zero value in the score or indicator rows** — graph-level, both should be exactly zero.

At WALL positions, the ≈ 400-unit deficit corresponds to either `indicators_above ≈ 0.6` (instead of 1) or a compensating `score ≈ +50`. Graph-level, both are wrong: `indicators_above[0] = I(bsp_rank ≥ 0 AND is_renderable)` should be exactly 1 for every renderable wall at threshold 0.

### Magnitude rules out ordinary noise

The observed errors are 1–2 orders of magnitude outside any documented per-op noise budget (`docs/numerical_noise.md`):

* `compare_near_thresh_05` worst abs error ≤ 0.005 — not 0.4.
* `piecewise_linear_2d doom_diff_trig` worst abs error 7.78 on a reference of -22.22 (~35% relative) — far below the 100-scale absolute errors we'd need to explain the SORTED[0] logit.
* Per-op compose through the BSP-rank chain at non-WALL positions evaluates `0 + 0 + … + 0 = 0`; no precision op can turn a chain of exact zeros into ≈ -100.

**Conclusion:** the error is not drift inside a single op. It's a structural contamination in the compiled residual stream at specific positions — most naturally explained by column aliasing where some other node's value lands in bsp_rank's columns at the attention read layer.

## Why the original xfail reason was wrong

Commit `c2d5a7a`'s xfail reason read:

> Phase E regression: at this off-center + oblique pose, the SORTED `attend_argmin_above_integer` softmax fails to concentrate on step 0 — all `indicators_above` slots evaluate to near-zero (likely due to compile-side precision loss in the per-wall is_renderable gate at geometry that lands near the attention-edge-of-view) …

Two of three substantive claims are wrong:

1. ✅ **"softmax fails to concentrate on step 0"** — corroborated. It concentrates on the **self** position (SORTED[0]), which is equally wrong.
2. ❌ **"all `indicators_above` slots evaluate to near-zero"** — not corroborated. `indicators_above` is evaluated across many positions and the compiled values aren't shown; the claim doesn't map to a measurable quantity.
3. ❌ **"compile-side precision loss in the per-wall is_renderable gate at geometry that lands near the attention-edge-of-view"** — **not supported by any evidence**. This is the ℓ∞ exemplar of "dressing up don't-know as know" the project doctrine exists to prevent. The observed logit magnitudes are far too large for precision drift in the validity gate; they're consistent with residual contamination at non-WALL positions whose cause is unknown.

## Partial mitigations that *looked* like fixes but weren't

Commit `a979f69` ("flat sum_nodes, 100→70 layers") was claimed in `tests/debug/test_probe_phase_e_trace.py` to have "collaterally fixed the underlying precision loss so the sentinel no longer surfaces on main." That claim is also wrong:

* The probe test only checks for `sel_bsp_rank = -_ABOVE_BONUS` (a single specific sentinel value).
* Post-a979f69 (HEAD of this branch), the raw `sel_bsp_rank` at sort[0] for this scene is **-1171.875**, not -1000. The probe test passes only because it checks for exactly -1000 and the post-select sentinel (99) hides the underlying bogus value.
* The actual symptom (wrong attention pick + value contamination) **is still present**.

`a979f69` reduced compiled-graph depth, which perturbed the compiler's residual-column allocation. That may have changed *where* the aliasing lands, but did not fix the underlying structural fault. Other scenes happen not to hit it; (3, 2, 20) does.

The pattern — an unrelated optimization "fixing" a bug by reshuffling residual allocation — is itself worth naming as a project risk. Such fixes are coincidence, not resolution.

## What would fix this properly

A real fix requires one of:

1. **Compiler-internals investigation.** Instrument the residual allocator to log, at each layer: which nodes share columns, and for each Attn layer, which nodes' columns are read by the key/value matrices. For this scene at the SORTED attention's layer, identify which node's columns land on `sort_value`'s `bsp_rank` slot and on the `indicators_above` slots. That's the aliasing pair. Then either:
   * Change the allocator to forbid that pair (general fix), or
   * Change the graph to break the pair (point fix).

2. **Pin residual state explicitly.** If the compiler supported pinned allocations for specific nodes, we could ensure `bsp_rank` and `indicators_above` get non-aliased columns at the SORTED attention layer. Today there's no such mechanism.

3. **Remove the residual-sensitivity of the above-integer primitive.** If `attend_argmin_above_integer`'s key_in Concatenate stayed un-fused (the current Linear fusion absorbs the Concatenate, which may be where the aliasing enters), the allocator would see the 3-slot structure explicitly.

Options 1 and 2 are substantial compiler work. Option 3 requires a compiler-fusion policy change (or an opt-out), also non-trivial.

## Doctrine violations this investigation surfaced

* **Never defer numerical problems.** c2d5a7a shipped a plausible reason rather than a measured one. Any 1–2-orders-of-magnitude logit error that a future agent might ever claim is "precision loss" should be challenged with this exact calculation: does the primitive's math close at the observed numbers? Here it didn't even close to an order of magnitude.
* **Xfail hygiene.** The xfail reason had three claims, two unfounded. Today's doctrine would require "unknown; investigating; linked to `docs/postmortems/phase_e_xfail.md`."
* **Foundation rule.** Phase E built on a residual-aliasing sensitivity that was already known-known per `plan-e.md` (5 prior failed attempts). We shipped anyway. The doctrine's lesson: when a plan already lists unresolved open questions about compiler behavior, those questions are the work.

## Tooling used

* `scripts/investigate_phase_e.py` — purpose-built investigation script. It:
  1. Rebuilds the game graph with the same compile knobs as `compile_game` (d=2048, d_head=32).
  2. Walks the graph to find `sel_bsp_rank_effective`, steps backward through the `select()` to find `raw_sel_bsp_rank` and `sort_done`.
  3. Probes all three nodes via `probe_layer_diff` at sort[0] during an autoregressive decode step.
  4. Probes the SORTED attention's softmax via `probe_attention` with per-position labels.
* `torchwright/debug/probe.py` — harness from plan 1.
* `docs/numerical_noise.md` and `op_noise_data.json` — noise budgets (plan 3), used to rule out per-op precision loss as an explanation.
* Compiler assertions from plan 2 did not fire for this scene (their scope is residual liveness, not cross-node column non-overlap).

## Regression test

The existing `tests/debug/test_probe_phase_e_trace.py` locks in "no `-_ABOVE_BONUS` (== -1000) in `sel_bsp_rank`." That's a too-narrow check — the current post-a979f69 value is -1171.875, not -1000, and that test passes vacuously. A correct regression test should assert either:

1. Raw `sel_bsp_rank` at sort[0] for (3, 2, 20) is within `[0, max_walls - 1]` (a clean integer), or
2. The SORTED attention's softmax at sort[0] concentrates on a WALL position, not SORTED[0].

Both are direct observables from the investigation script. Leaving the test narrow would recreate the Phase D failure mode. Updating it is tracked as a follow-up alongside any future fix.
