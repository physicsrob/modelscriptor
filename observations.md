# Observations: DOOM as a Transformer

Notes on interesting things we've learned while compiling DOOM into
transformer weights.  Intended as raw material for blog writing.

---

## Texture mapping requires O(tex_height) MLP sublayers, and you can't beat it

Wall texture mapping needs to assign a runtime texture color to each
screen row based on which texture band that row falls in.  This is a
"select from N runtime values based on a runtime condition" operation.

In a transformer MLP sublayer (Linear → ReLU → Linear), the output is a
*linear* function of the hidden units.  You can compute a step function
of the condition in the hidden layer, and you can pass the texture color
through the hidden layer, but you can't *multiply* them — that would
require a product of two hidden-unit outputs, and the output projection
is linear.

So each texture band requires two MLP sublayers: one to compute the band
mask (`in_range`), one to select the color (`broadcast_select`).  For
a 64-row texture, that's 128 MLP sublayers in the naive approach.

**The constant is improvable, not the scaling.**  By processing bands in
groups (e.g., 8 at a time), computing all masks and selects within a
group in parallel, summing the group's results, then freeing the
intermediate columns before the next group, we bring the cost down to
~2 layers per group.  For 64 rows in groups of 8: ~16 MLP sublayers
instead of 128.  The scheduler can pack independent `in_range` and
`broadcast_select` calls into the same physical MLP sublayer as long as
their combined neuron count fits within d_model.

This is a fundamental property of the architecture: **selecting from K
runtime values costs O(K) MLP sublayers**.  Compile-time values are free
(baked into weights), but runtime values — like texture pixels that
came out of a `map_to_table` lookup — must be routed through the MLP
bottleneck.  No binary decomposition or hierarchical scheme avoids this,
because the selection requires multiplying a runtime indicator by a
runtime value, which takes a full MLP sublayer.

The practical consequence: texture height is the scarce resource.  Width
is cheap (handled by a single `map_to_table` lookup keyed by the u
coordinate), but each row of vertical resolution costs ~2 MLP sublayers
divided by the group packing factor.  DOOM's 64-tall textures are
feasible (~16-22 layers with grouping) but 128-tall textures would eat
half the layer budget.

**Debugging footnote:** we initially suspected that chaining
`broadcast_select` calls (each using the previous output as
false_value) caused the internal `big_offset` constant to accumulate.
It doesn't — `broadcast_select` composes correctly.  The actual bug
was passing inputs to the `HeadlessTransformerModule` in the wrong
order (it expects alphabetical by input name, not declaration order).
Scrambled wall coordinates produced garbage that happened to look like
offset leakage.  Lesson: always check input ordering first.

## Less than 1% of the network does useful work

The compiled DOOM transformer uses far less than 1% of its parameters.
This isn't a bug — it's a structural consequence of using a dense
transformer as a substrate for sparse computation.

**Three levels of waste, each multiplicative:**

1. **Layer scheduling sparsity.**  Each MLP sublayer has d slots (e.g.,
   1024), but a `compare` op uses 2.  A `signed_multiply` creates a
   4-deep sequential chain where each intermediate layer might only fill
   a handful of slots.  The scheduler packs independent ops into the
   same layer when data dependencies allow, but most layers end up
   5-15% utilized on slots alone.

2. **Weight sparsity within used slots.**  Even a "used" MLP slot has a
   weight column of width d in linear1 and a weight row of width d in
   linear2.  But a typical op reads from 1-4 residual stream positions
   and writes to 1-4 positions.  So within each "used" slot, ~99% of
   the weight entries are structural zeros.  The matrix is dense; the
   computation is sparse.

3. **Attention overhead.**  Each attention head costs 4·d·d_head
   parameters (65K at d=1024, d_head=16).  These heads do cheap work:
   copying a scalar, adding two values, cancelling a dead intermediate.
   A 1-wide Linear node burns an entire head to route a single value.

Concretely at d=1024: each layer has ~6.3M parameters.  A typical layer
uses 5-15 attention heads (~0.5M allocated, far fewer non-zero) and
10-50 MLP slots (~100K allocated, most weights zero).  Total non-zero
weight entries per layer might be in the low thousands out of millions.

**The reason d must be large is not computation but memory.**  The
residual stream is the transformer's only working memory — every
intermediate value that's simultaneously alive needs its own column(s).
During the per-segment intersection phase with 22 wall segments, peak
liveness easily hits 500+ columns.  So d is sized for storage, but you
pay d² in parameters per layer.

This suggests a blog-worthy framing: **the transformer is a massively
parallel computer where we're running a single-threaded program.**  The
architecture offers d² capacity per layer anticipating that every neuron
might need to read from every position.  Our DOOM renderer uses a tiny
subgraph of that connectivity — each op reads 1-4 inputs and writes 1
output — but pays the full dense-matrix tax.

## Multiplication is the expensive primitive, and it's everywhere

The `signed_multiply` operation is the dominant cost driver in the
rendering pipeline, and it appears in almost every stage:

- Shared products: px·sin(θ), py·cos(θ)
- Distance: num_t · (1/den)
- Fish-eye correction: dist · cos(perp_angle)
- Texture u-normalization: adj_num_u · (1/abs_den)
- Collision detection: old_y·dx, old_x·dy

Each `signed_multiply` compiles to ~4 MLP sublayers via the polarization
identity: a·b = (|a+b|² − |a−b|²) / 4.  This requires two `abs` ops
(1 MLP each, using ReLU(x) + ReLU(−x)) and two `square` ops (1 MLP
each, via piecewise-linear approximation).  The intermediate values also
consume residual stream columns while alive, adding storage pressure.

This is interesting because multiplication feels like a "basic" op, but
in a ReLU network it's genuinely hard.  An MLP sublayer computes
piecewise-linear functions of its input.  Multiplication is bilinear —
you can approximate x² with a piecewise-linear function (that's what
`square` does), but you can't compute it exactly in one layer.  The
polarization identity converts one bilinear operation into two
univariate nonlinearities (squaring), which is optimal for this
architecture.

**Hypothetical improvement:** a native 2D piecewise-linear lookup
(`piecewise_linear_2d`) that takes two scalar inputs and produces one
output in a single MLP sublayer.  This would tabulate a·b directly on a
grid, cutting each multiply from 4 layers to 1.  The cost would be
grid_size² neurons per multiply, so a 40×40 grid uses 1,600 MLP
slots — feasible at d≥2048 but tight at d=1024.  Whether the 4× depth
reduction is worth the width cost depends on whether the pipeline is
depth-bound or width-bound.  (For the rendering pipeline, it's almost
certainly depth-bound.)

## The residual stream is a register file, and register pressure is the real constraint

The compiler's scheduler has an explicit "pressure" mode (triggered when
free columns drop below 25% of d) where it reprioritises operations
that free columns over operations on the critical path.  This is
exactly analogous to register pressure in a CPU compiler.

The residual stream is a fixed-width register file of d columns.  Every
intermediate value that's simultaneously alive occupies columns.  When
the file fills up, the scheduler must insert "cancel" operations (zero
out dead values to reclaim columns) using attention heads, even if those
heads could have been doing useful computation.  In the worst case, the
compiler stalls: it can't schedule a new operation because there's
nowhere to put the result, and it can't free columns because nothing is
dead yet.

The per-segment intersection phase is where this bites hardest.  For N
segments processed in parallel, peak liveness is roughly:
- Shared values: ~10 columns (trig, positions, products)
- Per segment: ~6-10 columns (num_t, num_u, adj versions, validity
  flags, distance, maybe tex_meta)
- Accumulating: distances and values awaiting min-reduction

With 22 segments: 10 + 22×8 + 22 = ~208 columns at peak.  That's 20%
of d=1024, which is manageable.  But at d=512 it would be 40%, and the
scheduler would spend layers on cancellation housekeeping instead of
forward progress.

**The architectural analogy:** classical compilers face the same tension
between instruction-level parallelism (do many things at once → need
many live registers) and register pressure (too many live values →
spill to memory).  CPU compilers resolve this with register spilling to
the stack.  Transformers have no stack — the residual stream is all
there is.  So the compiler's only option when pressure is high is to
serialise: process fewer segments in parallel, freeing columns between
batches.  This trades depth (more layers) for width (fewer live
columns), exactly like a CPU compiler trading instructions for spills.

## Division and reciprocal are cheap; multiplication is not

A counterintuitive cost ranking:

| Operation | MLP sublayers | Why |
|-----------|-----------|-----|
| 1/x (reciprocal) | 1 | Piecewise-linear in 1 variable — just a lookup table |
| x mod n | ~1 | floor(x/n) is piecewise-linear, then x - n·floor(x/n) is free |
| x × y | ~4 | Bilinear; requires polarization identity through abs + square |
| cos(θ), sin(θ) | 1 | Discrete angle space → piecewise-linear lookup |

Division (a/b) costs ~5 layers: 1 for reciprocal(b), then 4 for
a × (1/b).  But we avoid most divisions by precomputing 1/den as a
function of ray angle alone (the `_build_angle_lookup` table), turning
a runtime division into a 1-layer lookup.  The renderer exploits this
aggressively — all per-segment signed_inv_den, abs_den, and sign_den
values are precomputed in a single `piecewise_linear_nd` call.

The lesson: **univariate nonlinear functions are cheap (1 MLP sublayer via
piecewise-linear approximation), bivariate ones are expensive (require
decomposition into univariate parts).**  Any time you can restructure a
bivariate computation into a univariate lookup + free linear ops, you
save layers.  The angle-lookup optimisation saves ~5 layers × N
segments on the critical path — the single biggest optimisation in the
renderer.

## Runtime gather went from "impossible" to "2 MLP sublayers"

The observation above ("Texture mapping requires O(tex_height) MLP
sublayers, and you can't beat it") was written under one implicit
assumption: that the IR could not express "read element `k` from a
runtime vector where `k` is a runtime integer".  Every runtime
selection had to be phrased as a scatter — loop over all K candidates,
emit a per-candidate mask via `in_range`, fan out with
`broadcast_select`, sum the results — which scales as O(K) MLP
sublayers because each candidate mask is independent work.

Then `dynamic_extract` was added, and the ceiling moved.  Its recipe
is short: one `in_range(idx, idx + 1, n_entries)` to build a one-hot
mask, one `broadcast_select` to zero every slot except the selected
one, one free `Linear` to collapse `n_entries * d_fill` down to
`d_fill`.  Total cost: **2 MLP sublayers plus one free Linear**,
regardless of `n_entries`.  The textured fill rewrite turns the
per-tex-row scatter loop inside-out into a per-screen-row gather:

    for each of rows_per_patch screen rows:
        tex_idx = linear_bin_index(y, wall_top, wall_bottom, tex_height)
        color   = dynamic_extract(tex_column_colors, tex_idx, tex_h, 3)

Same math, opposite iteration direction, dramatically fewer ops — and
fewer failure modes (the scatter approach was quietly wrong when
`wall_height < tex_height` because the per-band masks overlapped at
round-off scale).

**The framing to take away**: "K things, pick one" is not a fundamental
cost ceiling; it's a question of whether the IR has a primitive that
inverts the scatter into a gather.  Once you have one — either a
dedicated op like `dynamic_extract` or the transformer's native
attention — the O(K) cost collapses to O(1).  Any other place in a
compiled graph that's running a scatter loop over runtime candidates
is probably a rewrite away from the same collapse.  The first
observation in this file is no longer a ceiling, just a characteristic
of the scatter *implementation* of that kind of select.

## The "passing tests" that were hiding a real correctness bug

When the textured fill was rewritten, the new implementation failed
four previously-green regression tests.  The apparent cause was the
new rewrite.  The actual cause, discovered after debugging, was that
**the old implementation was silently wrong** and the old tests only
passed because their tolerances absorbed the error.

The bug lived in `map_to_table`, used in the texture column lookup
stage with 2-D keys `(texture_id, col_idx)` mapping to `tex_height * 3`
RGB values per column.  `map_to_table` scores each key-value unit by
dot product against the input and fires every unit whose score exceeds
`key @ key - 1`.  For the input `(texture_id=0, col=7)` and the key set
`{(0, 0), (0, 1), ..., (0, 7)}`, *every* unit fires with some non-zero
magnitude and the final `Linear` sum returns a weighted combination of
several texture columns — not the exact one requested.  The
`tex_column_colors` node was producing values like `[0.37, 1.97, …]`
even for the default 8×8 atlas where the answer should be one exact
column.

The old regression test used `atol=0.2` and `max_boundary_pixels=48` as
its tolerance.  Both were generous enough to absorb the blended
lookups.  When the new fill was written with tighter precision, the
blend became visible and the tests failed — misleadingly pointing at
the new code rather than the old bug.  Fix: replace `map_to_table` at
this site with `piecewise_linear` keyed on a flat integer
`texture_id * tex_w + col`, which returns an exact texture column at
integer inputs and interpolates cleanly between adjacent columns.

**The lesson worth keeping:**  precision-sensitive tests whose pass
criterion happens to include the worst-case error of the
implementation being tested can hide real correctness bugs for a long
time.  The old test "was green" but was only green because the
generous tolerance covered the leaky lookup.  Any time you tighten
tolerance on a passing test and it starts failing, the first hypothesis
should be "the old test was papering over an error," not "the new code
regressed."

## Scene complexity couples to model width, quietly

The baked rendering design unrolls per-segment intersection `N` times
in the graph: each wall gets its own `_segment_intersection` Linear
nodes with compile-time constants, its own `_segment_distance` MLP
pipeline, its own contribution to the `reduce_min` tree, and its own
row in the per-angle lookup.  Most of this is structurally parallel —
all `N` segments can run in the same layers if residual width
permits — so the cost story depends on both N and `d`.

A compile-cost sweep at the current default `d=2048`:

| N walls | layers | peak_d |
|---:|---:|---:|
|  4 |  51 |  451 |
|  8 |  57 |  451 |
| 16 |  67 |  795 |
| 32 |  90 | 2048 (capped) |
| 64 | 165 | 2048 (capped) |

And the same walls at `d=6144` (large enough that no config caps):

| N walls | layers | peak_d (true) |
|---:|---:|---:|
| 32 | 65 | 1471 |
| 64 | 77 | 4849 |

Two things jump out:

1. **`peak_d` is flat up to N=8, then grows sharply.**  At small scene
   size the per-wall work doesn't dominate the residual stream — fixed
   stages (game logic, `_shared_products`, texture column lookup,
   column fill) hold the floor at ~450 columns.  Per-wall work starts
   contributing to peak residual width around N=16 and takes over
   completely by N=32.
2. **At `d=2048`, the compiler hits the cap at N=32 and starts adding
   layers to compensate.**  65 layers at `d=6144` vs 90 at `d=2048`:
   the extra 25 layers exist purely to sequentialize work that couldn't
   fit in the narrower residual.  At N=64, it's 77 vs 165 — more than
   double, entirely due to residual pressure.

**The hidden scaling wall:** at realistic DOOM-scene wall counts
(`N ≈ 32-64`), the current design silently demands `d ≥ 5000` to avoid
pathological layer counts, even though smaller scenes at `d=2048`
compile and run fine.  Nobody ever writes down "this architecture
requires `d_model` to scale with scene complexity" because it doesn't
*look* like a requirement until you plot it — per-step graph ops grow
smoothly, the model still compiles, the tests still pass — and yet
the per-frame decode cost at large N is dominated by extra layers
paying rent for residual pressure that wasn't there at small N.

This is the single most compelling reason to reach for runtime wall
data: moving wall geometry out of the residual stream and into the KV
cache via attention caps `peak_d` at the single-wall cost
(`~34-50` columns for the parametric intersection prototype),
decoupling `d_model` from scene complexity entirely.

## Attention is a runtime gather, and "zero-K" is a load-bearing design pattern

Even before `dynamic_extract` landed, the transformer had a perfectly
good runtime gather primitive: *attention*.  `softmax(Q·K)` over a
prefix of N key vectors selects the one whose key best aligns with the
query, and the value read gets routed through the standard attention
value projection.  For "pick which of N walls this ray hits", attention
is a more natural fit than `reduce_min` — `reduce_min` is a binary
tree of `select` ops with depth `log₂(N)` and some fiddly
comparator logic; one attention head does the whole thing in a single
softmax over all `N` walls at once, using Q = ray direction, K = wall
midpoint direction, V = wall parameters.

The wall-attention prototype (`tests/graph/test_wall_attention.py`)
hand-builds exactly this.  It packs wall and query tokens into the
same input sequence with role flags (`is_wall`, `is_query`), hand-
designs Q/K/V matrices for `Q·K = logit_scale · (cos(ray − wall_angle)
+ wall_bias − dist_scale · wall_dist)`, and verifies the attention
output concentrates on the correct wall sharply enough (>0.99 weight
with `logit_scale=200`) to read back that wall's parameters cleanly
for N up to 32 walls on a circle.

Two patterns from this work are worth writing down:

**Zero-K cross-role isolation.**  In the same rollout you have wall
positions and query positions.  Causal masking alone would let each
query attend to *prior queries* — their K vectors would participate
in the softmax and their V vectors would leak into the gathered
result.  The fix is elegant: project K from `(wall_cos, wall_sin,
-dist)` slots that are *exactly zero at query positions*.  A query
row's K is therefore `(0, 0, 0)`, its Q·K against any other key is
zero, and the softmax assigns it essentially zero weight.  Project V
from the same wall slots, so even if some weight leaks onto a query
row its V is also zero and contributes nothing to the gather.  Causal
masking handles wall → query direction; zero K handles query → query
direction; zero V handles the leakage floor.  Three independent
mechanisms compose cleanly, none of them requiring runtime branching
or role-conditional ops.

**The zero-logit baseline pitfall.**  There's a subtle failure mode
that only bites when you hand-design attention (training would learn
around it).  Causal self-attention at position `k` includes `k` in
its attention set, and `k` has K=0 (it's a query row), so its logit is
exactly 0.  If your real wall logits can equal zero — for example,
because `cos_diff = dist_scale · wall_dist` for some geometry — the
correct wall *ties* with the V=0 self-row and softmax splits mass
50/50 between them.  The gathered output is half the correct answer.

The fix is to bias every wall logit strictly positive by routing
`is_wall · wall_bias` (with `wall_bias` large enough to dominate
realistic distance terms) into K's distance dimension.  At query
positions the bias is zero (since `is_wall=0` there); at wall
positions it adds a constant offset that keeps wall logits above the
zero-logit self baseline regardless of `cos_diff` or `wall_dist`.

**What to remember:** hand-designing attention requires thinking about
the logit distribution as an explicit object.  Soft mechanisms that
training handles implicitly — keeping logit magnitudes apart from
competing zero baselines, ensuring softmax sharpness is adequate for
the scene density, handling ties between "correct" and "unused"
positions — all become explicit constraints on the Q/K matrix design.
The payoff is that the same attention primitive that replaces
`reduce_min` for wall selection can be reused anywhere else in the
graph where you want a runtime gather over a variable-length prefix.

## Algebraic simplification beat primitive substitution, by a lot

One concrete optimisation anecdote worth recording.  When building the
parametric single-wall intersection prototype (walls-as-tokens
derisking), the naive translation of `_segment_intersection` with
runtime wall endpoints cost 233 ops / 10 layers in the compiled graph.
It had ten `signed_multiply` calls — one per runtime product — because
every term in the formulas:

    den   = ey · cos - ex · sin
    num_t = ax · ey - ay · ex + ex · py - ey · px
    num_u = ax · sin - ay · cos + py · cos - px · sin

needs a product of two runtime scalars once the wall coordinates
aren't compile-time constants.

The obvious optimisation was to replace `signed_multiply` (~3 MLP
sublayers each, via the polarization identity) with
`piecewise_linear_2d` (1 MLP sublayer each, a 2-D grid lookup).  Doing
that alone would have cut 10 products × 2 layers = 20 layers worth of
work.  Fine.

But it turns out the much bigger win was **algebraic simplification**.
Collecting terms:

    num_t = ey · (ax − px) + ex · (py − ay)
    num_u = (ax − px) · sin + (py − ay) · cos

Both formulas now share the intermediates `dx = ax − px` and
`dy = py − ay` — both free Linears.  The product count drops from 10
to 6: two for `den`, two for `num_t` (pos·pos), two for `num_u`
(pos·trig).  Same math, same precision, fewer operations.

Cost trajectory of the three optimisation steps, measured on the
compiled graph:

| Change | ops | layers | peak_d |
|---|---:|---:|---:|
| Naive, 10 × `signed_multiply` | 233 | 10 | 62 |
| Algebraic simplification (10 → 6 products) + `piecewise_linear_2d` for pos·trig | 83 | 8 | 34 |
| + `piecewise_linear_2d` for the two pos·pos products too | **51** | **6** | **34** |

The algebraic simplification step alone cut ops by **150** — from 233
to 83 — and most of that came from the 10 → 6 product reduction, not
the primitive swap.  Switching `signed_multiply` to
`piecewise_linear_2d` got us the rest of the way, from 83 to 51.

**The lesson**: in a transformer-graph IR where each primitive has a
known cost, the first place to look for optimisations is *not* at the
primitive level but at the algebra.  Sharing intermediates between
outputs — finding a common subexpression like `(ax − px)` that serves
multiple formulas — is free, since free Linears are actually free and
`Concatenate` over shared residual-stream columns costs nothing.
Primitive substitution is usually easier to think about but yields
less.  In the parametric intersection, the algebraic rewrite was worth
roughly 2× the primitive swap.

---

# Observations: attention-based sorting

Notes from building the `sort_digits_v1` through `sort_digits_v4`
examples and the `torchwright/ops/attention_ops.py` primitive library.

## The bilinear ceiling: why attention can't express "next above prev"

A vanilla attention head computes `softmax(Q·K^T) · V`.  The logit
`Q[j] · K[i]` is bilinear in the query-side and key-side features.
Step functions (e.g., "is `score_i > threshold_j`?") mix both sides
nonlinearly and can't be produced by any polynomial in
`(query_features, key_features)`.

We verified this by trying every polynomial logit shape we could
construct — `-(s - prev - δ)²`, `α·s + β·(s - prev)²`, and many
others.  For every one, the softmax argmax either picks `prev` itself
(the item we already emitted) or the global max/min, never "smallest
`s` strictly above `prev`."

**The workaround: move the step function into the residual stream.**
Three places it can live:

| Location | How | Width cost | Used by |
|----------|-----|------------|---------|
| Key-side precomputed | `I(score > d)` for each fixed `d` | N columns per threshold | V1 |
| Query-side mask vector | Running bitmask, Q·K computes `mask[pos_i]` | N columns per input slot | V4 |
| Inside attention (non-vanilla) | ReLU between Q·K and softmax | Zero extra width | Not built |

## Unroll, don't self-reference

The compiler (`torchwright/compiler/forward/`) uses Kahn's topological
sort on the Node DAG.  A self-referential node (value at position `p`
depends on own value at `p-1` via `attend_to_offset`) creates a cycle.
The compiler doesn't detect this cleanly — it silently produces an
incomplete topological order, spins through `max_layers` trying to
schedule the remaining nodes, and raises a generic "did not converge"
error.

**Fix: unroll into distinct Nodes**, exactly as `prefix_ops.py` does
for its Hillis-Steele stages.  V4's mask update is:
```python
mask_k = create_literal_value(torch.zeros(max_input))
for k in range(max_out):
    selection_onehot = attend_argmin_unmasked(..., mask_vector=mask_k, ...)
    mask_k = elementwise_max(mask_k, selection_onehot)  # new Node each step
```

**Width cost of unrolling:** each step's state lives in the residual
stream simultaneously.  V4 with 10 steps × (10-slot mask + 8-dim
embed + overhead) fits in D_MODEL = 512.

## The `output_sequence` aliasing bug

`output_sequence` gates slot `k` via `attend_to_offset(is_trigger,
delta_pos=-k)`.  For `k` near the period of the fastest sine component
(≈ 2π ≈ 6.28), the attention aliases — it picks the trigger position
instead of `P - k`, firing the wrong slot.  Confirmed empirically:

```
k= 0: -1.000 -1.000 +1.000   ← correct
k= 5: -1.000 -1.000 +1.000   ← ALIASED
k= 6: -1.000 -1.000 +1.000   ← ALIASED
k=11: -1.000 -1.000 +1.000   ← ALIASED (≈ 2× period)
```

**Fix: `_emit_by_slot_index`.**  Compute `steps_since = pos_scalar −
trigger_pos_scalar` (scalar arithmetic, no aliasing) and gate with
`compare(steps_since, k ± 0.5)`.  All sort variants use this helper.

## The `digit_to_scaled_scalar` precision trap

`digit_to_scaled_scalar` → `map_to_table` with
`embedding_step_sharpness = 1.0`.  Embedding norms are ~40 (self-dot
~1600), so 0.04% softmax leakage (weight 0.9996 on the winner)
reduces the dot product by ~0.6, eating more than half the 1-unit
tolerance.  Chaining through `get_prev_value` and a second
`map_to_table` compounds the error until the scalar drifts.

**Fix (V1):** run a second attention with `value = digit_scalar` so
the scalar comes out of the softmax directly as a weighted average of
per-position scalars, staying within 1e-3 of the integer answer.

**Takeaway:** avoid chaining `map_to_table` on softmax-averaged
vectors.  Use scalar-valued attention outputs instead.

## Score envelope and the causal-mask sentinel

`Attn.compute` masks future positions to -1000.  With the original
`attention_hardness = 100` query scaling, a score of 5 produces a
logit of `100 × 8 × (-5) = -4000`, below the sentinel — making the
softmax pick masked future positions.

**Fix:** the new primitives use `_QUERY_GAIN = 8` (extracted from the
slowest cosine, which is ≈ 1 for all realistic positions).  Logits
stay in `[-960, +960]` for `|score| ≤ 120`, safely above -1000.
Softmax decisiveness per unit score: `exp(8) ≈ 3000` — adequate for
distinct integer scores.

## Primitives catalog

All in `torchwright/ops/attention_ops.py`.  Each compiles to one
vanilla attention head.

| Primitive | Intent | Used by |
|-----------|--------|---------|
| `attend_argmin` | Argmin of score in causal window | — |
| `attend_argmax` | Argmax of score | — |
| `attend_argmin_where` | Argmin restricted by a per-key validity mask | V1, V4 |
| `attend_argmax_where` | Argmax restricted by validity | — |
| `attend_argmin_above_integer` | Argmin above a per-query threshold via indicator basis | V1 |
| `attend_argmin_unmasked` | Argmin excluding positions set in a per-query mask vector | V4 |

## Design-space comparison

| Variant | Attention's role | Handles duplicates | D_MODEL | Max digits |
|---------|------------------|--------------------|---------|------------|
| V1 | Discovers next digit above threshold (indicator basis) | No | 384 | 10 |
| V2 | Only counting via prefix_sum; MLP decides & emits | Yes | 1024 | 5 |
| V4 | Masked argmin each step (attention does the selection) | Yes | 512 | 10 |

**When to use which pattern:**
- Pure "attention does the selection" with duplicates → V4 (unrolled mask).
- Distinct-only, clearest mental model → V1 (indicator basis).
- Need to understand what MLP-only looks like → V2.

## Perpendicular distance is precomputable; u-coordinate might be too

For a flat wall segment, the perpendicular distance from the player to
the wall line is a single scalar independent of ray direction.  The
current renderer recomputes the full parametric intersection (6
`piecewise_linear_2d` products) at every RENDER token, then derives
wall height from the ray distance plus fish-eye correction.  But
`dist_r * perp_cos` equals the perpendicular distance for any ray
hitting the same wall line --- so `wall_height = H / perp_dist` can be
computed from a single precomputed scalar, skipping the per-column
distance pipeline entirely.  See `distance_optimization_plan.md`.

The u-coordinate (texture column selection) also has a potential
shortcut: the sort phase already computes `col_a` and `col_b` (the
screen columns where each wall endpoint projects) for the visibility
mask.  The u-parameter could be approximated as
`(col_idx - col_a) / (col_b - col_a)`, eliminating 4 more
`piecewise_linear_2d` products per RENDER token.  This screen-space
interpolation is not perspective-correct for walls at oblique angles,
but for DOOM's vertical-wall geometry the error should be small.
Worth evaluating empirically as a "fast mode" tradeoff.

## Follow-ups

- **Delay-1 self-reference compiler support.**  Would let V4 express
  its mask as one node instead of `MAX_OUT` unrolled copies.
- **ReLU-in-logit extension to `Attn`.**  Non-vanilla but would
  eliminate the indicator basis / mask workarounds entirely.
- **`output_sequence` fix.**  Replace `attend_to_offset` gating with
  scalar-arithmetic `_emit_by_slot_index` project-wide.
- **V3 (two-attention hierarchical).**  Planned but not yet built.

---

## Visibility matching forces d_head >= screen width

The DOOM render attention selects which sorted wall to draw at each
screen column.  The selection score is a dot product between a column
one-hot (from the RENDER token) and a visibility mask (from each
SORTED_WALL token):

```
score = VIS_GAIN * sum_c(col_onehot[c] * vis_mask[c]) + ...
```

Since `col_onehot` is a one-hot vector, this just extracts `vis_mask[col]`
-- but in the attention Q·K framework, that extraction is a bilinear dot
product that requires **W dimensions** in Q/K space, one per screen column.

With the sorted flag and distance tiebreak, the Q/K width is `W + 1`.
For the default walkthrough resolution (W=120), that's `d_qk = 121`,
requiring `d_head >= 128` (next power of 2).  At W=160, `d_head >= 256`.

This is the dominant constraint on d_head in the DOOM graph.  All other
attention heads (sort, collision) use d_head < 32.  The render attention
alone forces d_head to scale with screen width.

**Cost implication:** each attention head has 4 parameter matrices of
size `d × d_head`.  With d=2048 and d_head=128, that's ~1M params per
head.  The render attention uses one head for Q/K and one or more for
V/O (which only needs 5 dimensions for wall data passthrough, split
across heads via the V/O splitting mechanism).

**Possible alternatives (not yet explored):**
- Hash the column index into a smaller space (e.g., locality-sensitive
  hashing) — reduces d_qk at the cost of collision risk.
- Decompose the visibility match into multiple lower-dimensional
  attention heads (e.g., 4 heads of width 32 covering column groups).
- Move visibility selection out of attention entirely — use MLP-based
  dynamic extraction instead, at the cost of more transformer layers.
