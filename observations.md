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
