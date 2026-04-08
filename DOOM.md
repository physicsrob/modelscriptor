# DOOM as a Transformer

## Vision

Implement DOOM's E1M1 as a transformer — not by training, but by **compiling
the game logic directly into transformer weights** using torchwright. The
transformer *is* the game engine. This is the flagship demo for torchwright.

## How It Works

One forward pass of the transformer renders one frame of gameplay.

### Input (Residual Stream at Position 0)

The first position acts as a BOS token. It encodes:

- **Player input**: the full DOOM input set (move forward/back, strafe
  left/right, turn left/right, shoot, use/open)
- **Current game state**: player x, y, angle/heading, and whatever additional
  state the game logic requires (health, ammo, door states, enemy positions,
  etc.)

### Computation (Transformer Layers)

The transformer layers implement the game logic:

- **State update**: process player input to compute new position, handle
  collision detection, update enemy AI, door/elevator mechanics, combat
- **Raycasting**: for each screen column, cast a ray from the player's position
  and angle to determine wall distance, texture, lighting

The map geometry of E1M1 is **baked into the weights** (via lookup tables,
conditionals, etc.). Each compiled transformer is specific to one map.

### Output (Residual Stream After Final Layer)

No embedding or unembedding layers. The residual stream directly contains:

- **Rendered pixel columns**: raw RGB values for each column of the frame
- **Updated game state**: new x, y, angle, enemy positions, etc. — everything
  needed to feed back in as input to the next frame

Each autoregressive position after the BOS corresponds to one screen column,
rendered left to right (column 0, 1, 2, ...). Each position's output includes
the RGB pixel values for that column (one value per pixel row per channel).

### Multi-Frame Rollout

Soft goal: the architecture should support rolling out multiple frames in a
single autoregressive sequence. Frame N's final state becomes frame N+1's BOS.

## Parameterization

The torchwright graph definition should be parameterizable so a single codebase
can compile transformers at different fidelity levels:

- **Resolution**: number of columns (screen width) and rows (screen height)
- **Feature set**: walls only, walls + doors, walls + doors + enemies, etc.
- **Map complexity**: full E1M1 vs. a single room subset

This produces a spectrum from "tiny transformer playable in real time" to
"massive transformer that renders full E1M1 faithfully."

## Scope

### North Star (Ideal Goal)

Full E1M1 playable in a compiled transformer:

- Complete DOOM input set
- Full map geometry baked into weights
- Enemies, items, doors, elevators, combat
- Raw RGB pixel output per column
- State carried forward between frames

### Acceptable Reductions

- Subset of E1M1 geometry
- Reduced enemy/item set
- Simplified lighting
- Lower resolution

### Non-Goals (For Now)

- Real-time performance (proof of principle first; can require a supercomputer)
- Multiple maps (E1M1 only)
- Sound
- Network multiplayer
- Training or learning of any kind — this is pure compilation

## Rendering Approach: Parallel Segment Intersection

Doom uses BSP-based segment rendering, not grid-based raycasting (that's
Wolfenstein). We follow Doom's approach: the map is a set of wall segments,
and rendering tests a ray against all segments to find the nearest hit.

See [raycasting_approaches.md](raycasting_approaches.md) for the full analysis
of DDA vs segment-based vs BSP. Summary: parallel segment intersection is
comparably difficult to parallel DDA for the intermediate goal (~45 vs ~30
layers), but carries forward to E1M1's geometry (diagonal walls, variable-height
sectors) without algorithmic rewrite. Since the cost is comparable, we go with
the approach that scales.

### How It Works (Per Screen Column)

Each autoregressive position computes one screen column:

1. **Ray direction**: from player angle + column offset, look up trig values
   (sin, cos) from a 256-entry table baked into weights.
2. **Shared products**: compute `Px * sin(angle)` and `Py * cos(angle)` once.
   These are reused across all segment tests.
3. **Per-segment intersection** (all segments in parallel):
   - `num_t = (A-P) × (B-A)` — linear in P → **free** (no FFN cost)
   - `den = D × (B-A)` — linear in D → **free**
   - Validity checks (t > 0, 0 ≤ u ≤ 1) via sign comparisons
   - Mask invalid intersections
4. **Min-reduction**: find nearest valid intersection via parallel comparison
   tree (log2(N) stages).
5. **Wall height**: `screen_height / perpendicular_distance` for the winner.
6. **Column fill**: given wall height and color, produce ceiling/wall/floor
   pixels.

### Key Property

Most per-segment math is linear in the runtime inputs (P and D) with constant
coefficients (segment endpoints baked into weights). This means the per-segment
cost is dominated by validity checks and the min-reduction, not the intersection
math itself.

### Scaling to E1M1

For the intermediate goal (~30 segments), brute-force parallel testing is fine.
For E1M1 (~300+ segments), BSP culling can be layered on: the BSP tree is baked
into the graph as conditional logic, pruning invisible segments before parallel
testing. This trades ~30 layers of depth for a large width reduction (300 →
50-100 effective segments).

## Primary Intermediate Goal: Minimal Segment-Based Renderer

Before tackling full Doom, the first major milestone is a **standalone
segment-based renderer compiled into a transformer** — the rendering core of
Doom, isolated from all game logic.

### What It Does

Given a player position and viewing angle, render a first-person view of a
small map defined by wall segments. No movement, no enemies, no doors — just:
"you are here, looking this way, here is what you see."

### Specification

- **Map**: small set of wall segments (~20-30), defining a few rooms. Walls are
  solid colors, no textures. Can include diagonal walls.
- **Resolution**: parameterized, baseline ~32 columns × 40 rows
- **Input** (position 0 of residual stream):
  - Player x, y as fixed-point integers
  - Player angle as a discrete index into a trig table (0–255)
- **Output** (residual stream after final layer):
  - One autoregressive position per screen column
  - Each position contains raw RGB values for every pixel row in that column
- **Rendering algorithm**: per column, test ray against all wall segments in
  parallel, find nearest hit, compute projected wall height, fill pixels

### New Primitives Required

Each of these is a reusable op that carries forward to the full Doom goal:

1. **Trig lookup** — `sin(angle)` and `cos(angle)` via `scalar_lookup` with
   precomputed tables. Angle is a discrete index (0–255 for 256 headings).

2. **Fixed-point multiply** — extend `multiply_integers` to handle values with
   fractional bits. Needed for: the shared products `Px*Dy` and `Py*Dx`, and
   the final wall height computation. Core primitive for all spatial math.

3. **Modulo** — `a % b = a - floor(a/b) * b`. Needed for: angle wrapping.
   Synthesizable from existing `thermometer_floor_div` + `multiply_integers`.

4. **Parallel min-reduction** — given N (value, metadata) pairs, return the
   pair with minimum value. Implemented as a log2(N)-stage comparison tree
   using `compare` + `select`. General-purpose primitive for any "find nearest"
   operation.

5. **Raw numeric output** — a new output pattern where the residual stream
   contains raw scalar values (pixel intensities) rather than token embeddings.
   No unembedding layer. This is a departure from all existing examples.

6. **Column fill** — given a projected wall height and a wall color, produce the
   pixel values for one screen column: ceiling color above the wall, wall color
   in the middle, floor color below.

### Why This Is the Right Intermediate Goal

- **Rendering is the hardest novel subproblem** in Doom. Game logic (movement,
  collision, AI) is complex but compositionally built from arithmetic and
  conditionals that torchwright already handles. Rendering requires spatial
  math, trig, and a pipeline that doesn't exist yet.
- **Every new primitive carries forward**. Trig, fixed-point math, parallel min,
  raw output — all are needed for the full Doom goal. Nothing is throwaway.
- **The rendering algorithm scales directly to E1M1.** Unlike grid-based DDA,
  segment intersection handles Doom's actual geometry (diagonal walls, variable
  sectors) without algorithmic changes. BSP culling can be layered on for
  performance.
- **It's self-contained and testable**. Verify correctness by comparing the
  transformer's output against a reference software renderer pixel by pixel.
- **It's visually compelling**. Even a 32×40 flat-shaded renderer produces
  recognizable 3D-looking output from a compiled transformer.

## Open Questions

- **Coordinate representation**: discrete vs. fixed-point vs. scaled integers?
  Depends on what arithmetic ops can support at the needed precision.
- **State size**: what is the minimal state vector for E1M1? How much of the
  game state fits in the residual stream?
- **Column output encoding**: exact layout of RGB values in the residual stream
  per position. One row per residual-stream dimension, or packed differently?
- **Transformer size**: how many layers and what d_model for various fidelity
  levels? Won't know until we start building.
- **Min-reduction cost**: cross-multiply comparison (avoids division, ~6
  layers/stage) vs per-segment division then scalar comparison (~4 layers
  per segment but simple ~3 layers/stage reduction). Need to prototype both.
- **BSP culling design**: for E1M1 scale, how to bake BSP traversal into the
  graph to reduce effective segment count per column.
