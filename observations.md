# Observations: DOOM as a Transformer

Notes on interesting things we've learned while compiling DOOM into
transformer weights.  Intended as raw material for blog writing.

---

## Texture mapping requires O(tex_height) FFN layers, and you can't beat it

Wall texture mapping needs to assign a runtime texture color to each
screen row based on which texture band that row falls in.  This is a
"select from N runtime values based on a runtime condition" operation.

In a transformer FFN layer (Linear → ReLU → Linear), the output is a
*linear* function of the hidden units.  You can compute a step function
of the condition in the hidden layer, and you can pass the texture color
through the hidden layer, but you can't *multiply* them — that would
require a product of two hidden-unit outputs, and the output projection
is linear.

So each texture band requires two ReLU layers: one to compute the band
mask (`in_range`), one to select the color (`broadcast_select`).  For
a 64-row texture, that's 128 FFN layers in the naive approach.

**The constant is improvable, not the scaling.**  By processing bands in
groups (e.g., 8 at a time), computing all masks and selects within a
group in parallel, summing the group's results, then freeing the
intermediate columns before the next group, we bring the cost down to
~2 layers per group.  For 64 rows in groups of 8: ~16 FFN layers
instead of 128.  The scheduler can pack independent `in_range` and
`broadcast_select` calls into the same physical FFN layer as long as
their combined ReLU unit count fits within d_model.

This is a fundamental property of the architecture: **selecting from K
runtime values costs O(K) ReLU layers**.  Compile-time values are free
(baked into weights), but runtime values — like texture pixels that
came out of a `map_to_table` lookup — must be routed through the ReLU
bottleneck.  No binary decomposition or hierarchical scheme avoids this,
because the selection requires multiplying a runtime indicator by a
runtime value, which takes a full ReLU layer.

The practical consequence: texture height is the scarce resource.  Width
is cheap (handled by a single `map_to_table` lookup keyed by the u
coordinate), but each row of vertical resolution costs ~2 FFN layers
divided by the group packing factor.  DOOM's 64-tall textures are
feasible (~16-22 layers with grouping) but 128-tall textures would eat
half the layer budget.
