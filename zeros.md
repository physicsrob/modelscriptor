# Zero Node Accumulation in the 3-Digit Adder

## Observation

When compiling the 3-digit adder (`examples/adder.py`, `max_digits=3`) with
`d=256, d_head=16`, the compiler creates increasingly many zero-valued Constant
nodes at each layer:

| Layer | Non-leaf nodes | Zero nodes | Other leaf | Total |
|-------|---------------|------------|------------|-------|
| -0    | 5             | 1          | 2          | 8     |
| -1    | 5             | 7          | 1          | 13    |
| -2    | 8             | 12         | 2          | 22    |
| -3    | 9             | 23         | 2          | 34    |
| -4    | 9             | 31         | 3          | 43    |
| -5    | 9             | 34         | 4          | 47    |

The non-leaf count stabilizes at 9 by layer -3. The zero count grows to 34 and
dominates the node set.

## Where the zeros come from

Every zero is freshly created — zero carry-over between layers is 0 at every
layer we checked. They are created by `SkipLayerComponent.get_strategies()`
(`modelscriptor/compiler/components/skip.py:50`), which calls
`create_constant(torch.zeros(len(node)))` each time it generates strategies for
a node. Each skip-through in each sublayer creates a new zero.

## Two consequences

### 1. Residual stream exhaustion

At layer -5, the zero nodes' dimensions sum to:

    9×1 + 19×8 + 6×16 = 257 dimensions

This exceeds `d=256`. The feature assignment solver cannot fit all nodes into
the residual stream, so compilation cannot succeed regardless of strategy search
improvements.

### 2. Beam search bloat

The strategy combination beam search (`get_combined_strategies`) processes all
nodes including zeros. At layer -5, it runs over 47 nodes when only 9 are
non-leaf. Zeros have trivial strategies but still participate in the
combinatorial search, multiplying the number of constraint solver calls.

## Stuck non-leaf nodes

Several non-leaf nodes appear at every layer from -3 onward without being
compiled:

- `Add(id=149, d=16)` — present at layers -3 through -8
- `Attn(id=53, d=8)` and `Attn(id=59, d=8)` — present at layers -5 through -8

These nodes are repeatedly passed through skip connections (creating new zeros
each time) but never compiled by any sublayer's computation path. The
compilation makes no progress on them.
