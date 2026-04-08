# Raycasting Approaches: Analysis

Three candidate approaches for implementing raycasting in a compiled transformer.
The question is which maps best to torchwright's computation model.

## Background: Torchwright's Cost Model

Two resources matter:

- **Depth** (layers): each `linear_relu_linear` is one FFN sublayer. Sequential
  dependencies stack layers. This is the primary bottleneck — more layers means
  a bigger transformer.
- **Width** (d_model columns): values live in the residual stream. Independent
  operations at the same depth share a layer. The compiler packs multiple
  independent FFN chains into a single layer as long as their intermediate
  dimensions fit (see `scheduler.py:291-301`).

Key implication: **N independent multiplications cost the same depth as 1
multiplication** — they just use more width. Parallelism across width is free
in terms of layer count.

---

## Approach 1: Sequential DDA (`unrolled_loop`)

Classic DDA: step the ray through the grid one cell at a time, stop on wall hit.

### Algorithm

```
initialize ray position, direction, side distances
for each step (up to max_iters):
    if side_dist_x < side_dist_y:
        advance in x, update side_dist_x
    else:
        advance in y, update side_dist_y
    if map[cell] is wall:
        freeze all state, record distance
```

### Why It Seems Natural

This is how Doom actually does it. DDA is the textbook raycasting algorithm.
`unrolled_loop` fits it directly.

### Cost Analysis

State variables: ~8 (ray_x, ray_y, side_dist_x, side_dist_y, map_x, map_y,
hit, wall_dist, side).

Per iteration:
- `done_fn`: 1 FFN layer (compare hit flag)
- `step_fn`: ~5-8 FFN layers (compare side distances, conditional advance,
  map lookup)
- `select` per state variable: 8 FFN layers
- `select` for done flag: 1 FFN layer
- **Total per iteration: ~15-18 FFN layers**

Max iterations: diagonal of grid = ~11 for 8×8, ~90 for 64×64.

**Total depth for 8×8: ~165-200 layers.**
**Total depth for 64×64: ~1350-1600 layers.**

Width: low (~50-80 columns for state variables).

### Sequential Dependency

The core issue: which direction to step (X vs Y) depends on comparing
side_dist_x vs side_dist_y, which accumulate differently depending on all
prior steps. This creates a true sequential dependency chain through every
iteration.

Even though `unrolled_loop` always creates all iteration nodes (no real early
termination in the graph), every iteration depends on the previous iteration's
output. The compiler cannot parallelize across iterations.

---

## Approach 2: Parallel DDA (enumerate all crossings, find first hit)

### Key Insight

The sequence of grid crossings along a ray is **deterministic given the ray
direction**. It does NOT depend on wall hits. Wall hits only determine where to
STOP — they don't affect where the crossings ARE.

A ray through a grid crosses two types of grid lines:
- **Vertical crossings**: where the ray crosses x = integer
- **Horizontal crossings**: where the ray crosses y = integer

These are two independent arithmetic sequences:
- Vertical crossing i is at ray distance: `first_x + i * delta_x`
- Horizontal crossing j is at ray distance: `first_y + j * delta_y`

where `delta_x = |1 / ray_dir_x|` and `delta_y = |1 / ray_dir_y|`.

For each crossing, the map cell entered is independently computable:
- At vertical crossing i: `map_x = start_x + i * step_x`, and
  `map_y = floor(player_y + distance * ray_dir_y)`
- At horizontal crossing j: `map_y = start_y + j * step_y`, and
  `map_x = floor(player_x + distance * ray_dir_x)`

No crossing's computation depends on any other crossing.

### Algorithm

```
for each crossing (all in parallel):
    compute distance along ray         (arithmetic)
    compute the map cell entered        (fixed-point multiply + floor)
    look up map[cell] → is_wall, color  (table lookup)
    if not wall: set distance = BIG     (select)

find minimum distance among all crossings  (parallel reduction)
return (min_distance, wall_color, side)
```

### Cost Analysis

For 8×8 grid: max 7 vertical + 7 horizontal = 14 crossings.
For 64×64 grid: max 63 + 63 = 126 crossings.

Per-crossing computation (all crossings in parallel = same layers):
- Distance: `first + i * delta` — 1 fixed-point multiply + 1 add ≈ 4 FFN layers
- Cross-coordinate: `player + distance * direction` — 1 fixed-point multiply ≈ 3 FFN layers
- Floor: `thermometer_floor_div` ≈ 1 FFN layer
- Map lookup: `scalar_lookup` or `map_to_table` ≈ 1 FFN layer
- Mask non-walls: `select(is_wall, distance, BIG)` ≈ 1 FFN layer
- **Total: ~10 FFN layers (independent of crossing count)**

Min-reduction:
- Each stage: `compare` (1 FFN) + `select` on distance+metadata (3-4 FFN) ≈ 5 layers
- Stages: `ceil(log2(14))` = 4 for 8×8, `ceil(log2(126))` = 7 for 64×64
- **Total: ~20 layers for 8×8, ~35 layers for 64×64**

**Total depth for 8×8: ~30 layers.**
**Total depth for 64×64: ~45 layers.**

Width per crossing: ~8-10 columns (distance, world coords, map coords, flags).
- 8×8: 14 crossings × 10 = 140 columns
- 64×64: 126 crossings × 10 = 1260 columns

### Edge Cases

**Ray parallel to an axis**: when `ray_dir_x ≈ 0`, `delta_x → ∞`, so all
vertical crossing distances → ∞ and are never selected. Handled naturally if
the trig table stores `delta_x` and `delta_y` directly (precomputed per angle),
with large sentinel values for near-axis angles.

**All crossings computed regardless**: even if the nearest wall is 1 cell away,
all 14 (or 126) crossings are evaluated. But since they share the same layers,
this costs no extra depth — only width.

---

## Approach 3: Parallel Segment Intersection

Treat the map as a collection of wall segments (line segments). Test the ray
against every segment simultaneously.

### Algorithm

```
for each wall segment (all in parallel):
    compute ray-segment intersection    (2D cross products)
    check if intersection is valid      (bounds + direction checks)
    record distance                     (or numerator/denominator pair)

find nearest valid intersection         (parallel reduction)
```

### Cost Analysis

For 8×8 grid: ~50-80 wall segments depending on layout.
For E1M1: potentially 500+ segments.

Per-segment intersection:
- 2D cross products: 2 multiplies + subtract ≈ 7 FFN layers
- Bounds checking: 2-4 comparisons ≈ 4 FFN layers
- Direction check: 1 comparison ≈ 1 FFN layer
- **Total: ~12 FFN layers**

Min-reduction (avoiding division via cross-multiply comparison):
- Each stage: 2 multiplies + compare + select ≈ 10 FFN layers
- Stages: `ceil(log2(80))` ≈ 7 for 8×8
- **Total: ~70 layers for reduction**

Final distance computation (1 division for the winner): ~5 FFN layers.

**Total depth for 8×8: ~87 layers.**
**Total depth for E1M1: ~130+ layers.**

Width: 80 segments × 15 columns = 1200 for 8×8. Scales badly.

### Advantages

- Exact geometry — no grid discretization
- Could handle non-axis-aligned walls (relevant for E1M1's diagonal walls)
- No special handling for axis-parallel rays

### Disadvantages

- Much wider than parallel DDA
- Reduction is more expensive (cross-multiply comparisons cost ~2× more than
  scalar comparisons)
- Scales with map complexity (segment count), not grid size
- E1M1 width requirements may be prohibitive

---

## Comparison

| | Sequential DDA | Parallel DDA | Parallel Segment |
|---|---|---|---|
| **Depth (8×8)** | ~180 layers | **~30 layers** | ~87 layers |
| **Depth (64×64/E1M1)** | ~1500 layers | **~45 layers** | ~130 layers |
| **Width (8×8)** | **~60 cols** | ~140 cols | ~1200 cols |
| **Width (64×64/E1M1)** | **~60 cols** | ~1260 cols | ~7500+ cols |
| **New primitives** | unrolled_loop, trig, fixed-point | trig, fixed-point, parallel min | trig, fixed-point, parallel min, cross product |
| **Diagonal walls** | No (grid-aligned only) | No (grid-aligned only) | Yes |
| **Conceptual simplicity** | High | Medium | Medium |

## Assessment

**Parallel segment intersection is the chosen approach.**

The initial analysis favored parallel DDA for its depth advantage (~30 vs ~87
layers). However, a deeper analysis (accounting for the fact that most
per-segment math is linear in the runtime inputs and therefore free) narrowed
the gap significantly: **~30 layers (DDA) vs ~45 layers (segment)**.

At comparable cost, the segment-based approach wins because:

1. **It handles Doom's actual geometry.** E1M1 has diagonal walls, variable-
   height sectors, and non-grid geometry. DDA would need fundamental extensions
   or replacement. Segment intersection handles all of this natively.

2. **It's closer to how Doom works.** Doom uses BSP-ordered segment rendering,
   not grid-based raycasting (that's Wolfenstein 3D).

3. **It scales with the right knob.** BSP culling can be layered on to reduce
   effective segment count for E1M1, without changing the core rendering logic.

4. **No wasted work.** Building DDA for the intermediate goal then switching to
   segments for E1M1 means implementing the renderer twice.

### Why the cost is closer than it first appears

The initial segment estimate (~87 layers) assumed expensive per-segment
intersection math. But most of it is free:

- `num_t = (A-P) × (B-A)`: segment endpoints (A, B) are constants baked into
  weights. This is linear in P → **free** (Linear node, no FFN cost).
- `den = D × (B-A)`: linear in D → **free**.
- The bilinear term `(A-P) × D` requires multiplying runtime values, but the
  expensive products (`Px*Dy`, `Py*Dx`) are **shared across all segments** —
  computed once (~3 FFN layers).

So per-segment cost is just validity checks and masking (~4-5 FFN layers,
parallelized across all segments in the same layers).

The main remaining cost is the min-reduction. Two options:
- Cross-multiply comparison (avoids division): ~6 layers/stage × 5 stages = 30
- Per-segment division then scalar comparison: ~4 layers/segment + 3 layers/stage × 5 = ~19

Both are manageable. The right choice depends on width constraints.

**`unrolled_loop` should still be built.** It's useful for game logic (enemy AI,
physics simulation, iterative algorithms) even though the renderer doesn't need
it.

## Open Questions

- **Parallel min-reduction**: needs to be built as a new primitive. Takes a list
  of (value, metadata) pairs and returns the pair with minimum value. This is
  straightforward — log2(N) stages of compare + select.
- **Fixed-point precision**: how many fractional bits are needed for the
  intersection math? Affects the max_value parameter for multiply_integers.
- **Min-reduction variant**: cross-multiply comparison (no division, wider) vs
  per-segment division (narrower reduction, but per-segment division cost).
  Need to prototype both.
- **BSP culling for E1M1**: how to bake BSP tree traversal into the graph to
  reduce segment count from ~300 to ~50-100 per column.
