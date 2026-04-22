# Overlay target-column reserve fix: root cause is elsewhere

**Test:** `tests/doom/test_pipeline.py::TestPipeline::test_sort_order_matches_bsp[210]`
**Fix landed:** commit `e404eec` ("reserve overlay target columns from reuse by intermediate allocations")
**Status:** Fix resolves the symptom, but the stated root cause ("delta silently corrupted whatever lived there") does not match what actually happens end-to-end in the compiled forward pass. The underlying failure mode at runtime is identified (overlaid outputs drift by ~0.008 at step 92, E8 `token_type` degrades, subsequent positions collapse to zeros), but the specific op responsible for the drift is not yet pinned down.

## Commit claim vs. what is actually going on

The commit explains the failure as:

> The delta-transfer layer at end of compile writes unconditionally to
> each overlay's target columns. The allocator was free to reassign
> those columns to intermediate nodes during the main compile loop
> (once their original owner became dead), so the delta write silently
> corrupted whatever lived there at end-of-compile.

Tracing the delta-transfer math against the actual allocations
`compile_game` produced at angle=210 does not reproduce this story:

- The delta op is `post[target] = pre[target] + source - subtract`,
  and `subtract_cols == target_cols` unconditionally. The
  pre-sublayer value at `target_cols` cancels out, leaving
  `post[target] = pre[source]`. This holds regardless of what
  intermediate happens to own `target_cols` going into the delta
  sublayer.
- All reads in the attention sublayer come from the pre-sublayer
  residual, so parallel writes to `source_cols` by other overlays'
  deltas do not corrupt this overlay's read.
- Cross-overlay aliasing of source/target cols, and self-aliasing
  (Linear 2870's residual cols overlapping its own target cols) were
  both present in the failing compile. Every case still produces the
  correct output at each target_col by that math.

Yet the test really does fail without the fix, and really does pass
with it. The fix is doing something — just not what the commit says.

## What actually happens at angle=210

With the fix reverted and `TW_DEBUG_SORT=1` enabled (see
`scripts/probe_sort_divergence.py`), the first four frames at
`px=0, py=0, angle=210` produce:

```
[s] step=  90 wc=+1.0000 wi=+0.0010 rc=+56.0494 rck=-0.0000 aw=-1.00 sd=-1.00 ln= 0 done=-1.00 tt=[-10.00 +10.00 -10.00 -10.00 -10.00 -30.00 +10.00 -10.00]
[s] step=  91 wc=+1.0000 wi=+0.0010 rc=+57.0494 rck=-0.0000 aw=-1.00 sd=-1.00 ln=12 done=-1.00 tt=[-10.00 +10.00 -10.00 -10.00 -10.00 -30.00 +10.00 -10.00]
[s] step=  92 wc=+1.0078 wi=+0.0078 rc=+58.3044 rck=+0.0207 aw=-1.00 sd=-1.00 ln=12 done=-1.00 tt=[ -9.97 +10.03  -9.97  -9.97  -9.97 -29.97 +10.03  -9.97]
[s] step=  93 wc=+0.0000 wi=+0.0000 rc=+0.0000  rck=-0.0000 aw=-1.00 sd=-1.00 ln= 0 done=-1.00 tt=[ +0.00  +0.00  +0.00  +0.00  +0.00  +0.00  +0.00  -0.00]
```

- Step 90 is the first SORTED token. `token_type` output is clean
  E8_RENDER (`±10` with `-30` on the label component).
- Step 91 is the first RENDER. Passthrough: every overlaid output
  equals its input to machine precision.
- **Step 92 is the second RENDER. All overlaid outputs have drifted
  by ~0.0078, and the E8 `token_type` is off by 0.03 from the clean
  encoding.**
- **Step 93 reads Step 92's drifted output as its input. The drifted
  E8 no longer matches any token_type constant exactly; the
  transformer's attention heads degrade to all zeros. Every
  subsequent position produces zeros, and the SORT never advances
  past wall 0.**

With the fix applied, the same trace is exact at every step: `wc`
stays at `+1.0000`, `rc` increments by exactly 1.0 per render step,
and the E8 `token_type` remains at exactly `±10/-30`. No drift at
all.

So: the failure is a single-shot numerical corruption at step 92,
not a delta-transfer aliasing issue. The delta layer never
misbehaves on either side of the fix — it sees clean values going
in and writes clean values going out. The drift happens inside the
main compile path.

## Why the fix still works

The fix's effective change is not "stop the delta from corrupting"
but **"shift the allocator onto a different code path"**.

Without the fix, Linear 2870 (the 60-wide `pixels` output) is
allocated at a scattered set of columns that includes
`[160..166, 167..170, 174, 175, 194, 198, 202, 206, 208..225]` —
overlapping both its own overlay's target_cols (`[167..226]`) and
seven other overlays' target_cols (`[160..166]`).

Reserving `[160..229]` at compile start forces Linear 2870 onto
`[38, 64..74, 235..421]` — no overlap with any overlay target_col.
This in turn cascades through the scheduler's placement decisions
for every other node, producing a numerically different compiled
transformer.

On that different compiled transformer, the second-RENDER
passthrough is exact. That is the fix.

## Why this matters

1. **The commit's one-sentence reason is wrong.** Per D2/D3, the
   stated explanation does not survive a close reading of the delta
   math. Anyone who reads the commit and tries to reason about when
   this class of bug could recur will reach incorrect conclusions.

2. **The fix is a load-bearing coincidence.** It prevents this
   specific allocator decision, but the underlying fragility — some
   op whose compiled output drifts by ~0.03 under specific
   allocation conditions — is still in the graph. A different scene,
   angle, `d`, `chunk_size`, or any future scheduler tweak could
   reintroduce the same failure mode under a different allocation
   coincidence, and the pin/reserve would no longer be the thing
   that "fixes" it.

3. **The drift is too large for ordinary softmax leak.** With
   `attention_hardness=100`, leakage on other positions is ~e^-100,
   which cannot account for 0.03-scale drift on a pure passthrough
   (`next_wall_counter = token.wall_counter` at RENDER). Something
   more structural is going on — a missing cancel, an additive
   write to a col that wasn't zeroed, a leak via a specific
   attention head's V/O scatter, or similar.

## What is still unknown

- **Which op is introducing the 0.03 drift.** Candidates worth
  probing first: the `select(is_render, …)` chain that computes
  `out_token_type` and `out_wall_counter`; the `equals_vector`
  op that derives `is_render`, `is_sorted` from `token_type`; any
  `compute_linear` whose source_cols intersect Linear 2870's
  residual cols at the relevant layer.
- **Whether it is an allocator-bookkeeping bug** (a col that got
  reused without cancel, so an additive write sums on top of a
  stale value) or **a legitimate numerical-sensitivity issue**
  (piecewise-linear op accumulating error beyond its declared
  budget).
- **How many other scene/angle/d combinations are silently riding
  on the same coincidence.** The test only parametrises on angle;
  the scheduler's allocation decisions depend on `d`, `d_head`,
  `d_hidden`, `chunk_size`, etc., so the same fragility could
  already be hiding behind other test configurations.

## Direct evidence from `debug=True` self-consistency check

Running the DOOM prefill through `module.step(..., debug=True, debug_atol=1e-7)`
with the fix reverted fires the residual-stream self-consistency
check immediately:

```
Residual-stream self-consistency failure (atol=1e-07):
  node:   output (id=2870, type=Linear, width=60)
  first:  layer_39_mlp_out_state
  later:  layer_54_mlp_out_state
  cols:   [41, 66, 118, 119, 120, 121, 122, 123, 124, 130, 133, 135,
           140, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
           159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
           174, 175, 194, 198, 202, 206, 208, 209, 210, 211, 212, 213,
           214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225]
  worst col: residual[160] (node index 25)
  first val: [0.0, 0.0, …, 0.0]        # 86 positions, all 0.0
  later val: [-1.0, -1.0, …, -1.0]     # 86 positions, all -1.0
  max_abs_diff: 1.0
```

With the fix applied, the self-consistency check passes cleanly; a
different (pre-existing) `render/wall_vis_attention` softmax-hardness
assert fires instead, but that is an independent issue and the
consistency check has already run to completion by that point.

This is not fp noise. It is a magnitude-1.0 overwrite of every one of
Linear 2870's 60 residual columns at every position, happening
between layer 39 and layer 54 in a window where the node is still
registered in `residual_assignment`. Some op in that window writes
–1.0 additively into those columns. The 60-wide width, the –1.0
constant value, and the per-position uniformity point at one of the
`neg_one` / `zero_pixels` literals or `select(…, neg_one)` fallbacks
in `torchwright/doom/game_graph.py::_assemble_output` — but the
specific op producing the write has not been identified.

Column 160 in particular is an overlay target column (overlay
`Linear 2900`'s `target_cols = [160]`). Without the fix, that column
is simultaneously owned by Linear 2870 in the allocator. That
overlap is exactly the pattern the fix's reserve machinery prevents,
and once reserved, Linear 2870 is forced onto a disjoint allocation
(`[38, 64..74, 235..421]`) that the corrupting op does not reach.

So the fix does address a real compiler-invariant violation — just
not the one the commit message describes. The delta transfer layer
was never where the value changed; the change happens 15 layers
earlier, during normal compile scheduling, inside a cell the
scheduler told the rest of the system was still live for Linear 2870.

## Follow-ups

1. **Run `probe_compiled` at the angle=210 first-RENDER position
   with the fix reverted.** The oracle is the recursive graph
   evaluation; the first node whose compiled value diverges beyond
   `atol` pins the drifting op.

2. **Update the commit message / add a cross-reference here.** The
   current xfail-grade reason on the commit ("delta silently
   corrupted") is in the form flagged by D5 as unacceptable. This
   postmortem stands in until the specific op is pinned down.

3. **Keep the pin/reserve as defense-in-depth, but decouple it
   from the correctness claim.** Reserving overlay target_cols is
   cheap (~3.5% of the 2048-wide residual stream at DOOM settings)
   and prevents one class of scheduler pathology. But the
   downstream passthrough should produce integer wall_counter /
   exact E8 encoding regardless of where Linear 2870 lands. That
   is the real correctness property, and the real fix should
   restore it.

4. **Once the drifting op is identified**, add a reproducer at the
   smallest possible layer (op-level or a minimal 2–3-node graph)
   per D6, not at the full-DOOM integration layer where it
   currently lives.

## Reproducer

`scripts/probe_sort_divergence.py` compiles the full game at
`d=2048`, runs one frame at `(0, 0, 210)`, and with
`TW_DEBUG_SORT=1` prints every overlaid output per position so the
drift is visible frame-by-frame. The fix can be toggled by applying
or reverting commit `e404eec`; each configuration's trace is
deterministic on its own.
