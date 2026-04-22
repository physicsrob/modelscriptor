# Overlay target-column reserve fix: the right fix, wider-scope cause still open

**Test:** `tests/doom/test_pipeline.py::TestPipeline::test_sort_order_matches_bsp[210]`
**Fix landed:** commit `e404eec` ("reserve overlay target columns from reuse by intermediate allocations")
**Status:** Fix is correct and eliminates both the runtime failure and the `debug=True` self-consistency violation I traced. But the chain from the specific invariant violation I pinned down (the `advance_wall` overlay's delta overwriting column 160 while `Linear 2870` is still live there) to the observed runtime symptom (`wall_counter` output drifting `1.0000 → 1.0078 → 0` at step 92) is not closed — for every read path I could follow through layer 54, the per-overlay outputs still land correctly at their target_cols. Either a *second* aliasing that the first-failure-only check hid, or a render-position compute path I didn't probe, is doing the actual runtime damage. The fix happens to cure both because it prevents every overlay target_col from overlapping any live node's allocation.

## What I did pin down

The compiler-invariant violation is real, and it's at the delta layer:

- Under the reverted fix, `Linear 2870` (60-wide `pixels` output) has `residual_map.get_indices(Linear 2870)` including **column 160** (as its slot 25).
- The `advance_wall` overlay declares `target_cols = [160]`.
- At layer 54 (the appended delta-transfer layer), head 0 emits the combined-form delta head for `advance_wall`: slot 0 writes `+residual[53]` (source), slot 1 writes `-residual[160]` (subtract), both at O column 160. Post-sublayer `residual[160] = Linear 2900`'s value (= `-1.0` at prefill positions, since `advance_wall = select(is_render, render_out.advance_wall, neg_one)` and `is_render=−1` off-render).
- Through prefill, `Linear 2870`'s residual slot for index 25 lives at col 160 and held `0.0` end-to-end (prefill has no render frames). At layer 54 that `0.0` is overwritten to `-1.0`. Self-consistency catches it: Linear 2870's value at col 160 is different at layer 39 (when it was computed) vs. layer 54 (after the overwrite) — `max_abs_diff = 1.0`.

Neither the delta math itself nor `Linear 2870`'s own delta-transfer reads produce wrong *outputs* from this overwrite alone: all reads in the sublayer are pre-sublayer, so they still see `Linear 2870[25]` at col 160 before the overwrite lands, and each overlay's declared target_cols end up with its own intended value. **The runtime bug must therefore be caused by something else I haven't traced**, most likely another overlay/node aliasing that the first-failure-only check never surfaced. Candidates for that investigation are listed in §4.

## Evidence

### 1. debug=True self-consistency check pinpoints node + layer range

Running prefill through `module.step(debug=True, debug_atol=1e-7)` with the fix reverted fires the check with a magnitude-1.0 diff, not fp noise:

```
Residual-stream self-consistency failure (atol=1e-07):
  node:      output (id=2870, type=Linear, width=60)
  first:     layer_39_mlp_out_state
  later:     layer_54_mlp_out_state
  worst col: residual[160] (node index 25)
  first val: [0.0, 0.0, …, 0.0]        # 86 positions, all 0.0
  later val: [-1.0, -1.0, …, -1.0]     # 86 positions, all -1.0
  max_abs_diff: 1.0
```

Re-reading the `later val`: that's the value at **column 160** across 86 positions. Not all 60 of `Linear 2870`'s cols flip — only col 160 does.

### 2. Per-layer trajectory pinpoints the exact layer of the write

`scripts/probe_per_layer_trajectory.py` walks every captured sublayer state and prints residual[160]'s value at each one. The transition happens at `layer_54_attn_skip_out_state`:

```
layer_39_mlp_out_state    col[160]=+0.0000  …  <-- Linear 2870 computed here
layer_40..53_*            col[160]=+0.0000
layer_54_attn_skip_out_state  col[160]=-1.0000  <-- CHANGED
layer_54_mlp_out_state    col[160]=-1.0000
```

Layer 54 is the appended **delta-transfer layer** (`forward_compile.py` appends it after the main loop). So the write is emitted by the delta-transfer machinery itself.

### 3. Layer 54 attention inspection identifies the exact head

`scripts/probe_layer_54.py` reads `net.layers[54].attn.attn` and finds every head whose O projection has a non-zero entry at residual column 160. Exactly one head contributes:

```
head 0, slot 0: O coef = +1.000000
  Q  [pos 0]: 100.0    (hardness=100 queries on pos_encoding col 0)
  K  [pos 0]: 1.0
  V  [col 53]: 1.0
head 0, slot 1: O coef = -1.000000
  Q  [pos 1]: 100.0
  K  [pos 1]: 1.0
  V  [col 160]: 1.0
```

This is the exact signature of the combined-head form of `delta_transfer` in `_write_delta_transfer`: source (+1) on slot 0, subtract (-1) on slot 1, current-position attention via pos_encoding with hardness=100. For the `advance_wall` overlay:

- `target_cols = [160]` (col 160 in the overflow region is the residual home of `advance_wall` output)
- `source_cols = [53]` (where `Linear 2900` — the `advance_wall` `select(...)` leaf — was scheduled)
- `subtract_cols = target_cols = [160]`

So the write is exactly `residual[160] += residual[53] - residual[160]`, i.e. `residual[160] := Linear 2900` (the advance_wall output). At prefill positions (all non-render), `Linear 2900` evaluates `select(is_render, render_out.advance_wall, neg_one)` to `-1`, which matches the observed `-1.0` at 86 positions.

### 4. The mechanism — confirmed for prefill, unconfirmed for the render-step drift

`Linear 2870` is the `pixels` output (60 wide). Its own delta op has `source_cols = residual_map.get_indices(Linear 2870) = [41, 66, 118, …, 160, …, 225]`. Index 25 in that list is col 160. So `Linear 2870`'s delta head reads `residual[160]` pre-sublayer and writes it to `target_cols[25] = col 192` (pixel index 25 in the overflow `pixels` region).

In the *same* attention sublayer, `Linear 2900`'s delta head also reads pre-sublayer, so it sees `Linear 2870`'s slot-25 value at col 160, not the post-sublayer `-1.0`. Both reads complete before any writes land. So within a single forward pass, each overlay's declared target_cols end up holding that overlay's intended value: `Linear 2870`'s delta copies the intended pixel value out to col 192; `Linear 2900`'s delta writes the intended `-1.0` to col 160. On paper both output slots look fine.

The `residual[160]` post-sublayer value is simply `Linear 2900`'s output — the *overlay* being written uses its slot correctly. What's destroyed is `Linear 2870`'s slot 25 in the residual stream, which is *not* read by anything after layer 54 (it's already been copied to col 192 by `Linear 2870`'s own delta, and nothing else reads col 160 downstream of the delta layer).

**This is where my understanding runs out.** The self-consistency check legitimately fires — the compiler said `Linear 2870` owns col 160 through layer 54 but col 160's value changed. That is a real invariant violation. But the runtime drift I originally captured (`wc` going `1.0000 → 1.0078 → 0` at step 92) has to be caused by *some* value inside the compile being wrong, and this specific col-160 overwrite on its own does not explain that — the chain from "col 160 held `-1.0` instead of `Linear 2870[25]` at the moment layer 54's delta wrote" to "step 92's `wall_counter` output drifts by 0.008" is not traced.

Plausible routes worth investigating:

- **More than one node suffers the same aliasing.** The self-consistency check raises on the first failure, so other nodes with overlapping cols are invisible. A version that collects every violation instead of raising on the first would say whether `Linear 2839` (`wall_counter` output, at col 45) or any of its select-chain intermediates also overlap an overlay target_col.
- **An op during layers 40–53 writes to col 160 at RENDER positions.** At prefill my trajectory probe shows col 160 stable at 0 through layers 39–53. At a RENDER position, some conditional compute gated on `is_render=+1` might additively touch col 160, corrupting `Linear 2870[25]` *before* layer 54 runs. I did not run the trajectory probe at step 92 because the `render/wall_vis_attention` hardness Assert terminates prefill before I can reach it.
- **KV-cache contamination at layer 54.** Reads at the delta layer are current-position via `pos_encoding` with hardness 100, so past-position weights are ~e^-100 — vanishingly small in theory, but if the O matrix has structural non-sparseness across head dimensions, a small fraction of a corrupted past col 160 could reach future positions' Q/K/V at layer 54. Non-obvious; wants a concrete weight-shape check.

### 5. Where my earlier "delta math self-corrects" claim was wrong

My claim: "`post[target] = pre[target] + source - subtract` with `subtract_cols == target_cols` gives `post[target] = source`, regardless of what intermediate owns target_cols." That arithmetic is correct. What's wrong is the conclusion I drew from it.

Correct statement: the overlay *being written* gets its correct value at its target_cols. What I missed is that the live node sharing the target col has its residual-stream value there destroyed by the write. Whether that destruction propagates to wrong runtime output depends on whether anything downstream reads `residual[col]` assuming it still holds the live node's value. Every route I've traced so far in the delta-layer math does its read *before* the corresponding write, so per-overlay output values at their declared target_cols come out correct. The runtime drift at step 92 therefore points at a second layer of the problem I haven't yet characterised — see §4's list.

## Why the fix works

Reserving the overflow target cols `[160..229]` at compile start forces the allocator to pick a different home for `Linear 2870` — `[38, 64..74, 235..421]`, disjoint from all overlay target_cols. The delta at col 160 then only touches `Linear 2900`'s declared target, no collateral damage.

The fix is load-bearing via the allocator, but for a good reason: the invariant "no live node's residual cols may overlap an overlay's target_cols" is a real compiler-invariant requirement that the scheduler wasn't previously enforcing. The reserve machinery is the mechanism that enforces it.

## What this means for doctrine

- **D1 (compiler-bug protocol):** honored correctly. The fix addresses a real compiler-invariant violation in the allocator/scheduler interaction, not a user-graph workaround.
- **D2 (never defer numerical problems):** partially served. The violation at col 160 is bit-exact (a magnitude-1.0 write, not fp noise); no op noise budget is implicated *at the point the check fires*. The 0.008 runtime drift at step 92 is still a numerical-scale symptom whose mechanism I haven't traced from the identified violation, so D2's discipline of "the bit-level reason" isn't yet fully satisfied for the runtime symptom.
- **D3 (understanding rule):** partial. One-sentence compression of the pinned violation: *layer 54's `advance_wall` delta head overwrites `residual[160]` while `Linear 2870` still claims col 160 as its slot 25*. One-sentence compression of the step-92 runtime drift: not available.
- **D6 (reproducer):** `scripts/probe_debug_true.py` + `probe_per_layer_trajectory.py` + `probe_layer_54.py` form a complete reproducer chain for the invariant violation. `probe_sort_divergence.py` captures the downstream runtime symptom. The missing link between them — a unit-level probe that shows corrupted data *propagating* — is an open item.

## Open follow-ups

1. **Should the scheduler's `_get_effective_consumers` be tightened to never let output leaves become "dead" before the delta layer runs?** Today it treats a leaf of the terminal Concatenate as dead once the Concatenate is added to `computed_nodes`. That's what lets the overlap form. The reserve machinery shortcuts the problem by pre-reserving the offending columns; a longer-term fix might be to extend "live until delta" to every output leaf.

2. **The `render/wall_vis_attention` softmax-hardness assert fires even with the fix applied.** The self-consistency check passes with the fix, but a downstream Assert surfaces at prefill position 1 with `max_weight=0.5`. That is independent of this postmortem and should be investigated separately; it shows the `debug=True` pipeline now has at least one unrelated latent issue visible.

3. **Scope sweep.** The fix prevents this class of overlap across all overlays, not just `advance_wall`. But other allocation patterns at different `d`, `d_head`, `d_hidden`, or with different `SchedulingPolicy` choices could still expose analogous issues in completely different places (e.g., output leaves overlapping pinned inputs, or cross-position KV reads reaching into columns that got re-assigned mid-compile without the allocator realising). The reserve machinery only protects overlay target_cols.

## Reproducers

- `scripts/probe_sort_divergence.py` — per-position overlaid-output trace at angle=210, set `TW_DEBUG_SORT=1`. Shows the `~0.008` drift at step 92 and the subsequent collapse.
- `scripts/probe_debug_true.py` — runs prefill with `debug=True, debug_atol=1e-7`. Self-consistency check fires with the fix reverted, passes with the fix applied.
- `scripts/probe_per_layer_trajectory.py` — walks every captured sublayer state and prints residual values at a node's target cols. Identifies the exact layer (54) where `residual[160]` transitions from 0 → -1.
- `scripts/probe_layer_54.py` — inspects `net.layers[54].attn.attn` and identifies the exact head (`head 0, slots 0/1`) and its Q/K/V/O footprint, matching it to `delta_transfer` for the `advance_wall` overlay.
- `scripts/probe_step_92_debug.py` — attempts `debug=True` at step 92 specifically; currently short-circuited by the unrelated `render/wall_vis_attention` hardness Assert firing during prefill. Finishing this probe is the next step toward closing the gap between the pinned-down violation and the runtime symptom.
