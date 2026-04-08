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
comparably difficult to parallel DDA for the intermediate goal, and carries
forward to E1M1's geometry (diagonal walls, variable-height sectors) without
algorithmic rewrite. BSP is used for sector identification (floor/ceiling
heights, lighting), not for rendering culling — see **BSP Strategy** below.

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
For E1M1 (~300+ segments), brute-force parallel testing **still works**. The key
property — segment endpoints baked into weights make intersection math free — means
all 300 segments cost the same depth as 30. Width grows (peak ~1500 columns at
d_model=2048), but the min-reduction only adds `ceil(log2(300/30))` ≈ 3 extra
stages (~9 layers). See **BSP Strategy** and **Layer Budget** below for the full
analysis.

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

See **Phase 1** below for the full breakdown. Summary:

- **Already exist:** `reduce_min` (parallel min-reduction), `map_to_table`
  (trig lookup), `reciprocal`, `signed_multiply`
- **Need to build:** fixed-point multiply (extend `multiply_integers` for
  fractional bits), modulo (angle wrapping)
- **Infrastructure:** raw numeric I/O (no embedding/unembedding), reference
  software renderer for verification
- **Composition (not new ops):** column fill (compare + select per row), trig
  lookup (map_to_table with precomputed sin/cos table)

### Why This Is the Right Intermediate Goal

- **Rendering is the hardest novel subproblem** in Doom. Game logic (movement,
  collision, AI) is complex but compositionally built from arithmetic and
  conditionals that torchwright already handles. Rendering requires spatial
  math, trig, and a pipeline that doesn't exist yet.
- **Every new primitive carries forward**. Trig, fixed-point math, parallel min,
  raw output — all are needed for the full Doom goal. Nothing is throwaway.
- **The rendering algorithm scales directly to E1M1.** Unlike grid-based DDA,
  segment intersection handles Doom's actual geometry (diagonal walls, variable
  sectors) without algorithmic changes. Scaling from 30 to 300 segments adds
  width but only ~9 layers of depth (3 extra reduction stages).
- **It's self-contained and testable**. Verify correctness by comparing the
  transformer's output against a reference software renderer pixel by pixel.
- **It's visually compelling**. Even a 32×40 flat-shaded renderer produces
  recognizable 3D-looking output from a compiled transformer.

## BSP Strategy

Doom uses a BSP (Binary Space Partitioning) tree for rendering. The BSP tree
partitions the map into convex subsectors and provides a front-to-back
traversal order from any player position. Doom uses this to skip
already-occluded geometry.

### Why BSP Doesn't Help Rendering in a Fixed Graph

BSP's value in Doom is **skipping work**: traverse front-to-back, track which
screen columns are filled, stop drawing once everything is occluded. This
reduces ~300 segments to ~20-50 actually drawn per column.

In a fixed computation graph, **you cannot skip work**. Every node is always
evaluated. The graph's width is set at construction time. Runtime conditions
can mask results (set distance = BIG) but cannot prevent computation. This
removes BSP's primary motivation.

Three alternative BSP-for-rendering approaches were analyzed:

**BSP-driven PVS routing** — use BSP to identify the player's subsector, look
up a precomputed Potentially Visible Set, and only test those segments. This
genuinely reduces width (300 → 80 segments), but segment endpoints are now
**runtime values** loaded from a table instead of **constants** baked into
weights. This means intersection math requires `multiply_integers` (~3 FFN
layers) instead of being free. Net result: spend ~8 layers (BSP traversal +
table lookup + runtime multiplies) to save ~6 layers (shorter reduction). A
net loss of ~2 layers, plus PVS correctness risk in open areas.

**Parallel BSP evaluation with masking** — evaluate all ~200 BSP node
decisions in parallel, compute per-subsector visibility, mask invisible
segments. This adds ~4 layers for BSP but saves nothing: all 300 segments are
still computed and still enter the min-reduction. Masking just ensures masked
segments lose comparisons — they don't go away.

**BSP-ordered sequential traversal** — use `unrolled_loop` to test segments
in BSP front-to-back order, freezing on first hit. This trades width for
depth: instead of 300 parallel segments, test sequentially. With ~300
iterations unrolled at ~12 layers each, this is ~3600 layers. Completely
unviable.

### Recommendation: BSP for Sector Identification

BSP is genuinely useful for a different purpose: **classifying the player's
position into a subsector/sector**. This gives floor height, ceiling height,
light level, and sector type — all needed for rendering beyond flat-shaded
single-height walls.

Each BSP node has a splitting line. The test "which side of the line is the
player on?" is linear in (Px, Py) — a single Linear node (free) followed by
one `compare` (1 FFN layer). All ~200 BSP node tests run in parallel in **1
FFN layer**. Then per-subsector leaf identification uses `bool_all_true` over
~12 ancestor decisions in **2 FFN layers**. Sector property lookup via
`map_to_table` adds **1 FFN layer**.

**Total BSP cost: ~4 layers, ~360 peak columns** (freed before rendering
starts). This is cheap, authentic to Doom, and enables real features.

The rendering pipeline remains brute-force parallel segment intersection. The
blog post narrative:

> The BSP tree's 200 splitting planes are baked into the transformer's early
> layers, classifying the player's position into one of E1M1's 150 subsectors
> in 4 layers. The transformer then tests all 300 wall segments simultaneously
> — BSP was invented because CPUs can only test one segment at a time. A
> transformer doesn't have that limitation.


## Layer Budget

Constraints: **≤120 layers, d_model ≤ 8192**. Layer counts below are based on
the cost of the underlying primitives: `compare` = 1 FFN layer, `select` = 2
FFN layers (cond_add_vector + ReLU gate), `reduce_min` stage = 3 FFN layers
(compare + 2× parallel select), `multiply_integers` = 3 FFN layers.

### Per-Column Rendering Pipeline

| Phase | Layers | Notes |
|-------|--------|-------|
| **Ray setup** | | |
| Trig lookup (sin/cos from angle) | 1 | `map_to_table`, 256 entries |
| Shared products (Px·sinθ, Py·cosθ, etc.) | 3 | `multiply_integers`, 4 products parallel |
| **Segment intersection (300 segs)** | | |
| Validity checks (t>0, 0≤u≤1, den≠0) | 5 | compare + select, all segments parallel |
| Mask invalid intersections | 2 | select → distance = BIG |
| **Min-reduction** | **27** | **9 stages × 3 layers/stage** |
| **Post-intersection rendering** | | |
| Wall height (1/distance × scale) | 3 | `reciprocal` via `piecewise_linear` |
| Sector properties lookup | 1 | `map_to_table` on segment's sector ID |
| Upper/lower wall bounds (variable heights) | 4 | arithmetic from sector floor/ceil heights |
| Column fill (ceiling/wall/floor per row) | 5 | compares for row classification + selects |
| Texture column computation | 3 | `multiply_integers` for u × tex_width |
| Texture pixel lookup | 2 | split lookup: by column then by row |
| Distance-based lighting | 4 | `piecewise_linear` dimming + multiply RGB |
| Floor/ceiling rendering | 10 | reciprocal per row + flat lookup + lighting |
| **Per-frame (shared across columns)** | | |
| BSP sector identification | 4 | 200 split tests + leaf ID + sector lookup |
| Game state (movement, collision) | 12 | input processing, position update, collision |
| | | |
| **TOTAL** | **~86** | |

### What Dominates

The min-reduction is the single largest component at 27 layers (31% of total).
It's determined by segment count: `ceil(log2(N))` stages × 3 layers/stage.
Reducing segment count from 300 to 80 saves only 6 layers (27→21), which is
why BSP-based culling isn't worth the overhead it adds.

### Fidelity Knobs

If needed, layer count can be reduced via fidelity reductions rather than
BSP-based culling:

- **Flat-colored floors/ceilings** instead of textured: saves ~8 layers
- **Solid-color walls** instead of textured: saves ~5 layers
- **Simplified lighting** (sector-only, no distance dimming): saves ~3 layers
- **Fewer segments** (subset of E1M1 geometry): saves ~3 layers per halving

### Intermediate Goal Budget

The minimal segment-based renderer (flat-shaded, ~30 segments, no game logic):

| Phase | Layers |
|-------|--------|
| Trig lookup | 1 |
| Shared products | 3 |
| Validity checks + masking | 7 |
| Min-reduction (30 segs, 5 stages) | 15 |
| Wall height | 3 |
| Column fill | 5 |
| **TOTAL** | **~34** |


## Phased Plan

Nine phases from here to E1M1. Each phase has a concrete, testable deliverable.

### Phase 1: Rendering Primitives

Build the ops and infrastructure that don't exist yet. No compiled transformer
— just unit-tested primitives ready to compose.

**New ops:**

- **Fixed-point multiply** — extend `multiply_integers` (or use
  `signed_multiply` with scale factors) for values with fractional bits. Core
  primitive for all spatial math.
- **Modulo** — `a % b = a - floor(a/b) * b` via `thermometer_floor_div` +
  `multiply_integers`. Needed for angle wrapping (0–255).

**Existing ops** that turn out to already cover some of the "New Primitives
Required" list:

- `reduce_min` — parallel min-reduction already exists
- `map_to_table` — trig lookup is just `map_to_table` with a precomputed
  256-entry sin/cos table
- `reciprocal` — already exists via `piecewise_linear`
- `signed_multiply` — already exists

**Infrastructure:**

- **Raw numeric I/O** — compile pipeline must handle graphs with no
  `Embedding`/`Unembedding`. Input is raw scalars via `create_input`; output
  is raw scalars read directly from the residual stream. May require changes
  to `CompiledTransformerModule`.
- **Reference software renderer** — a simple Python software renderer that
  takes (Px, Py, angle, segments) and produces pixel output. Used for
  pixel-exact verification in all subsequent phases.

**Deliverable:** all new ops unit-tested; compile pipeline handles raw I/O;
reference renderer produces correct images.


### Phase 2: Flat-Shaded Static Renderer

Compose Phase 1 primitives into a complete rendering pipeline compiled into a
transformer. First visible output.

**What it does:** given (Px, Py, angle), render a first-person view of a small
map. Solid-color walls, uniform floor/ceiling colors.

**Map:** ~20-30 wall segments defining a few connected rooms. Includes at
least one diagonal wall to validate non-axis-aligned geometry.

**Resolution:** parameterized, baseline 32 columns × 40 rows.

**Pipeline (per column):**

1. Trig lookup → sin/cos of ray angle
2. Shared products (Px·sin, Py·cos, etc.) via `multiply_integers`
3. Per-segment intersection — linear in (P, D) with constant coefficients → free
4. Validity checks + masking → `compare` + `select`
5. `reduce_min` → nearest valid intersection
6. Wall height → `reciprocal(distance)`
7. Column fill → ceiling/wall/floor pixels via `compare` + `select` per row

**Deliverable:** compiled transformer that takes static (Px, Py, angle) and
outputs a 32×40 RGB image. **Pixel-exact match** against the reference
software renderer at multiple test positions and angles.

**Estimate:** ~34 layers, d_model ~512.


### Phase 3: Player Movement + Collision

Add game state. The transformer becomes playable.

**What's new:**

- **Input processing** — player inputs (forward, back, strafe, turn) encoded
  in position 0 alongside game state.
- **Position update** — new_x = x + speed·cos(angle), etc. Uses the same
  fixed-point multiply and trig ops.
- **Collision detection** — test the movement vector against wall segments to
  prevent walking through walls. This is another segment intersection pass
  (same math, different ray).
- **State carry** — the output residual stream includes updated (x, y, angle)
  alongside rendered pixels. This state feeds back as input to the next frame.
- **Angle wrapping** — modulo to keep angle in 0–255 range.

**Map:** same as Phase 2 (or slightly expanded).

**Deliverable:** a playable first-person walkthrough. Player moves with
keyboard inputs, can't walk through walls, view updates each frame. State
persists across frames via autoregressive rollout.

**Estimate:** ~50 layers (rendering + collision + state update), d_model ~512.


### Phase 4: BSP + Variable-Height Sectors

Add Doom-style architecture: sectors with different floor and ceiling heights.
Requires BSP for sector identification.

**What's new:**

- **BSP decision tree** — all ~N BSP split-line tests in parallel (1 layer),
  leaf identification via `bool_all_true` (2 layers), sector property lookup
  via `map_to_table` (1 layer). See **BSP Strategy** above.
- **Sector properties** — floor height, ceiling height, light level per
  sector. Baked into a lookup table indexed by sector ID.
- **Multi-section column fill** — a wall between two sectors of different
  heights shows upper wall, main wall, lower wall. Column fill becomes
  multi-section: compute wall_top/wall_bottom for each section, classify each
  row, select the right color.
- **Segment metadata** — each segment carries front/back sector IDs through
  the min-reduction so the winner's sector properties are available for column
  fill.

**Map:** a purpose-built test map (~40-50 segments) with stairs, platforms,
and height variation. Multiple sectors at different heights connected by
steps and drops.

**Deliverable:** compiled transformer correctly renders variable-height
architecture. Stairs look like stairs. The BSP tree is baked into the weights
and correctly identifies the player's sector.

**Estimate:** ~60 layers, d_model ~1024.


### Phase 5: Textures + Lighting

Visual fidelity. After this phase, the renderer looks recognizably Doom-like.

**What's new:**

- **Wall textures** — compute texture coordinates (u from hit position along
  wall, v from vertical position within wall). Look up texture pixels via
  split `map_to_table` (by column then by row). Texture data baked into
  weights.
- **Floor/ceiling textures** — for each row above/below the wall, compute
  world-space (x, y) of that floor/ceiling point via `reciprocal` of row
  distance, then look up flat texture.
- **Lighting** — combine sector light level (from BSP) with distance-based
  dimming. Apply as a multiply on RGB values.

**Map:** same test map as Phase 4, now with textures and lighting applied.

**Deliverable:** textured walls, floors, and ceilings with lighting. Side-by-
side comparison with a reference Doom-style renderer.

**Estimate:** ~80 layers, d_model ~1024-2048.

**Key risk:** texture data is large. Doom textures are 64×64 = 4096 pixels.
A single `map_to_table` with 4096 entries needs d_int = 4096. Split lookup
(by column then by row, 2 layers) is the likely approach. Total texture
parameter count could be significant. Texture encoding strategy should be
validated early in this phase.


### Phase 6: E1M1 Geometry

Scale from test maps to the real E1M1 map.

**What's new:**

- **WAD data import** — tooling to extract E1M1's segments, sectors, BSP tree,
  and textures from the Doom WAD file (or a cleaned-up derivative). Converts
  to the format expected by the graph constructor.
- **Scale to ~300 segments** — brute-force parallel intersection at full
  scale. Min-reduction grows from 5 stages to 9 stages (+12 layers).
  Width grows to peak ~1500 columns.
- **Full BSP tree** — ~200 BSP nodes, ~150 subsectors. Same 4-layer
  structure, just wider.
- **All E1M1 sector properties** — every sector's floor height, ceiling
  height, light level, and textures.

**Map:** E1M1.

**Deliverable:** walk through E1M1 in a compiled transformer with full visual
fidelity (textures, lighting, variable heights). Compare against a reference
Doom renderer or actual Doom screenshots.

**Estimate:** ~92 layers, d_model ~2048.

This is the first major visual milestone: **it looks like Doom E1M1.**


### Phase 7: Doors, Elevators, Switches

Interactive environment mechanics.

**What's new:**

- **Mutable sector state** — door and elevator positions stored in the game
  state vector. Sector floor/ceiling heights become state-dependent (base
  height + offset from game state) rather than static constants.
- **Door logic** — when the player presses "use" near a door segment, the
  door's sector ceiling begins moving up. After a delay, it moves back down.
  Implemented with `unrolled_loop` or frame-by-frame state updates.
- **Elevator logic** — similar to doors but for floor height. Trigger by
  walking onto the elevator sector.
- **Switch/trigger detection** — proximity test (player near a linedef with a
  special type) using segment distance checks.

**Map:** E1M1 with functional doors and elevators.

**Deliverable:** doors open and close, elevators move, switches trigger
actions. Player can navigate through E1M1's door-blocked areas.

**Estimate:** ~98 layers, d_model ~2048. Main cost is mutable state in the
game state vector and the conditional sector-height logic.


### Phase 8: Sprite Rendering + Items

Render 2D sprites (items, decorations) composited into the 3D scene.

**What's new:**

- **Sprite projection** — for each sprite, compute screen-space position and
  scale from player-relative coordinates. This is similar to wall projection:
  translate to player space, compute angle, project.
- **Per-column sprite test** — for each column, determine which sprites (if
  any) overlap that column and are closer than the wall. This is a second
  parallel reduction (similar to segment intersection) over all active sprites.
- **Depth compositing** — compare sprite distance vs wall distance per column.
  If the sprite is closer, its pixel replaces the wall/floor pixel.
- **Sprite pixel lookup** — `map_to_table` indexed by (sprite_type,
  sprite_column, sprite_row) to get pixel color. Transparent pixels
  (color key) fall through to the wall behind.
- **Item pickup** — proximity test between player and item positions. When
  close enough, the item's state flag is set to "collected" and it stops
  rendering. Update health/ammo/keys in game state.

**Map:** E1M1 with items (health packs, ammo, armor, keys) and decorations
(barrels, pillars, lamps) placed at their canonical positions.

**Deliverable:** items and decorations are visible as sprites in the 3D view.
Player can walk over items to collect them. Keys are required to open
keyed doors (from Phase 7).

**Estimate:** ~106 layers, d_model ~2048-4096. Sprite rendering adds a second
parallel reduction pass (~15-20 layers) and sprite pixel lookups.


### Phase 9: Enemies + Combat → Full E1M1

The final phase. Add enemies and combat to complete the E1M1 experience.

**What's new:**

- **Enemy state** — per-enemy: position, health, AI state (idle, chasing,
  attacking, pain, dead), animation frame. Stored in the game state vector.
- **Enemy AI** — simplified state machine. Idle enemies activate on line of
  sight or sound. Chasing enemies move toward the player (pathfinding
  simplified to direct movement with collision). Attacking enemies fire
  projectiles or hitscan attacks at intervals.
- **Enemy rendering** — enemies are sprites (Phase 8 infrastructure).
  Animation frame selected by AI state + frame counter.
- **Player combat** — hitscan weapon (pistol/shotgun): on "shoot" input,
  check if the player's crosshair ray intersects an enemy. Apply damage.
- **Projectile handling** — if enemies use projectile attacks, projectiles are
  entities with position + velocity, advanced each frame, with
  player-collision checks.
- **Damage + death** — player takes damage, enemy takes damage. Health
  reaches 0 → death state (enemy falls, player game-over).

**Map:** full E1M1 with all enemies, items, and secrets at canonical
placements.

**Deliverable:** **full E1M1 playable in a compiled transformer.** Walk
through the level, fight enemies, collect items, open doors, find secrets,
reach the exit. Every frame is one forward pass.

**Estimate:** ~115 layers, d_model ~4096. Enemy AI and combat add ~10-15
layers (state machine logic, line-of-sight checks, damage computation).
Width grows with number of active entities (enemies + projectiles + items).


### Phase Summary

| Phase | Deliverable | New capability | ~Layers | ~d_model |
|-------|-------------|----------------|---------|----------|
| 1 | Unit-tested ops + raw I/O | Fixed-point, modulo, reference renderer | — | — |
| 2 | Static flat-shaded renderer | Segment intersection + column fill | 34 | 512 |
| 3 | Playable walkthrough | Movement, collision, state carry | 50 | 512 |
| 4 | Variable-height architecture | BSP sector ID, multi-height sectors | 60 | 1024 |
| 5 | Textured + lit renderer | Textures, floor/ceiling, lighting | 80 | 2048 |
| 6 | E1M1 visuals | Full map, WAD import, 300 segments | 92 | 2048 |
| 7 | Interactive E1M1 | Doors, elevators, switches | 98 | 2048 |
| 8 | Items + decorations | Sprite rendering, pickups | 106 | 4096 |
| 9 | **Full E1M1** | **Enemies, combat, AI** | **115** | **4096** |


## Open Questions

- **Coordinate representation**: discrete vs. fixed-point vs. scaled integers?
  Depends on what arithmetic ops can support at the needed precision.
- **State size**: what is the minimal state vector for E1M1? How much of the
  game state fits in the residual stream?
- **Column output encoding**: exact layout of RGB values in the residual stream
  per position. One row per residual-stream dimension, or packed differently?
- **Texture encoding**: Doom textures are 64×64 or 128×128 pixels. A single
  `map_to_table` with 4096+ entries needs d_int=4096, which may exceed d_model.
  Split lookup (by column then by row) needs 2 layers but requires an
  intermediate representation. Texture data dominates parameter count.
- **Floor/ceiling rendering**: Doom renders floors/ceilings as horizontal
  textured spans. Per-column, each row above/below the wall needs a reciprocal
  (row→world distance) + sector lookup + texture lookup. Estimated at ~10
  layers but this is the least certain estimate in the budget. Flat-colored
  floors/ceilings save ~8 layers if the budget is tight.
- **Collision detection sharing**: game-state collision detection is essentially
  a second segment intersection pass. Can the reduction tree be shared or
  reused, or does collision need a smaller dedicated segment set?
