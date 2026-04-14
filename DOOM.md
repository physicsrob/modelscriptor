# DOOM as a Transformer

## Vision

Implement DOOM's E1M1 as a transformer — not by training, but by **compiling
DOOM's rendering algorithm directly into transformer architecture** using
torchwright. WAD assets (walls, textures, BSP tree) flow through the
transformer as input tokens. The transformer *is* the game engine. This is
the flagship demo for torchwright.

## Architectural Analogues

DOOM's 1993 architecture maps naturally onto transformer architecture.
These aren't superficial parallels — they share the same abstract structure.

| DOOM | Transformer | Why they fit |
|------|-------------|--------------|
| Screen column | Autoregressive position | Independent units, processed sequentially |
| Wall segment (seg) | Token | Atomic primitive with properties |
| BSP plane test | Attention comparison | Binary spatial classification |
| BSP tree structure | Sparse attention pattern | Each seg attends to its ~log(N) ancestors |
| Sector properties | Shared context via attention | Many segs reference one sector |
| Front-to-back draw | Autoregressive output | Ordered sequential generation |
| Column clipper | Attention mask | Track what's visible/occluded |
| Texture column | Attention-based lookup | Retrieve pixels by column index |

The deepest analogy: **DOOM's column-by-column rendering IS autoregressive**.
DOOM was designed around the observation that each screen column is
independent; the transformer's autoregressive structure processes independent
positions. They're the same abstraction, 30 years apart.


## How It Works

One forward pass renders one frame. All game assets flow through the
transformer as tokens — nothing is baked into weights.

### Token Sequence (Per Frame)

```
TEX_COL × (num_tex × tex_w) → INPUT → BSP_NODE × M → WALL × N → EOS
    → SORTED_WALL × N → THINKING/RENDER × (dynamic)
```

Seven token types drive the pipeline:

| Token | Purpose |
|-------|---------|
| **TEX_COL** | Texture column pixel data. Each token carries one column of one texture. RENDER tokens retrieve pixels via attention. |
| **INPUT** | Player state (x, y, angle) + movement controls. Attention distributes velocity and trig values to downstream tokens. |
| **BSP_NODE** | BSP splitting plane (nx, ny, d). Each node computes `side_P = sign(nx·px + ny·py + d)` — "which side of this plane is the player on?" |
| **WALL** | Wall segment geometry + BSP coefficients for sort. Computes collision flags and BSP rank. |
| **EOS** | End of prefill. Outputs resolved player state (after collision) and seeds the sort phase. |
| **SORTED_WALL** | Autoregressive sort by BSP rank. Each step finds the next wall in BSP traversal order via `attend_argmin_unmasked`. |
| **THINKING** | Wall selection for current screen region. Hoists wall-attention out of per-chunk RENDER tokens. |
| **RENDER** | Pixel output. Each token renders a chunk of rows for the current column. |

### Data Flow

1. **Prefill** (batched): Load texture columns, player state, BSP structure, and wall geometry
2. **Sort** (autoregressive): Order walls front-to-back by BSP rank
3. **Render** (autoregressive): For each screen column, select visible wall and render pixels

All cross-position data flows through attention. The host is a dumb token
feeder and pixel bitblitter — it copies autoregressive outputs back as
inputs and composites pixels to the framebuffer.

### WAD-to-Transformer Pipeline

```
┌──────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│   WAD File   │────▶│   load_map_subset()   │────▶│  Segments +          │
│  (doom1.wad) │     │   (x, y, max_walls)   │     │  Textures +          │
└──────────────┘     └───────────────────────┘     │  BSP nodes + coeffs  │
                                                    └──────────┬───────────┘
                                                               │
                     ┌─────────────────────────┐              │
                     │     Transformer         │◀─────────────┘
                     │     (~65 layers)        │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Frame + State         │
                     └─────────────────────────┘
```

A single Python function loads the WAD, selects walls by distance to player,
extracts the minimal BSP subtree covering them, and precomputes BSP
coefficients per wall. The transformer remains a dumb feeder + bitblitter.


## BSP Integration

BSP is DOOM's signature innovation. Rather than computing distances and
sorting (what a modern engine might do), DOOM uses the BSP tree's pre-computed
spatial structure to determine rendering order via O(log N) binary decisions.

### BSP in the Transformer

The BSP tree flows through as **BSP_NODE tokens** — not baked into weights,
not flattened into walls. Each node is a first-class token with a distinct
architectural role.

### BSP Rank: Linear Algebra from Tree Structure

For each wall W at subsector S, its position in BSP traversal order is:

```
rank(W) = Σ over ancestors i of: (side_P[i] ≠ side_W[i]) × sibling_size[i]
```

The intuition: at each ancestor, if the player is on the same side as W, we
visit W's subtree first (add 0). Otherwise we visit the sibling subtree
first (add its size).

This collapses into a **sparse dot product**:

```
rank(W) = dot(coeffs_W, side_P_vec) + const_W
```

Where:
- `side_P_vec` is the M-dim vector of BSP decisions (computed at runtime)
- `coeffs_W` is a precomputed M-dim sparse vector (non-zero at ancestors)
- `const_W` is a precomputed scalar

**The tree structure — traversal order, sibling sizes, path membership —
collapses into linear coefficients.** What was recursive tree traversal in
1993 becomes a dot product in the transformer.

### Computation in the Transformer

| Step | Operation | Layers |
|------|-----------|--------|
| 1 | Each BSP_NODE computes `side_P = sign(nx·px + ny·py + d)` | 1 |
| 2 | Each WALL gathers `side_P_vec` via attention to BSP_NODEs | 1 |
| 3 | Each WALL computes `rank = dot(coeffs_W, side_P_vec) + const_W` | 1 |
| 4 | Sort phase uses rank via existing `attend_argmin_unmasked` | unchanged |

### What the Host Precomputes

`load_map_subset` does BSP work on the CPU:

1. Parse WAD lumps (NODES, SSECTORS, SEGS, VERTEXES, LINEDEFS, SIDEDEFS)
2. Select ~32 closest segs to player position
3. Find their subsectors, extract minimal covering BSP subtree
4. Renumber nodes 0..M-1
5. For each seg: trace ancestors, compute `coeffs_W` and `const_W` from
   sibling subtree sizes and path sides

### Tradeoff vs Distance-Based Sort

| | Distance sort (current) | BSP rank |
|---|---|---|
| Sort key computation | ~21 layers (intersection + score) | ~3 layers (dot product) |
| Per-wall input | 5 floats | ~46 floats (5 + M+1 coeffs) |
| Host complexity | Simple | BSP parsing + precomputation |
| Algorithm authenticity | Generic | DOOM's actual algorithm |

The BSP version moves ~18 layers of transformer computation into host-side
precomputation — a good trade when the host is Python and the transformer
is compiled.


## Current State

### What's Working

The renderer is fully functional for textured, playable walkthroughs with
distance-based sorting:

- **Textured walls** from real DOOM WAD textures (downscaled to 8×8)
- **Player movement** with full DOOM input set (forward/back, strafe, turn)
- **Collision detection** via velocity-ray intersection
- **Front-to-back wall ordering** via autoregressive sort (distance-based)
- **Multi-room scenes** with diagonal walls (tested up to 22 segments)
- **Walkthrough generation** (`make walkthrough` produces GIF output)

### Architecture Stats (120×100, 8 walls, 4 textures)

```
Layers:           65
Allocated params: ~1.2B
Non-zero params:  ~8M (0.7% density)
d_model:          2048
d_head:           128
```

### Layer Breakdown (current, distance-based sort)

| Component | Layers | Notes |
|-----------|--------|-------|
| render (total) | 43 | tex_sample dominates at 23 layers |
| sort/visibility | 17 | Col_lo/col_hi tangent computation |
| wall/collision | 17 | Per-wall collision detection |
| wall/intersection | 10 | Central ray intersection (for sort) |
| wall/sort_score | 11 | Distance → sort key |
| input/game_logic | 12 | Angle update, velocity computation |
| eos/collision_resolve | 12 | Resolve collision across all walls |
| sort/attention | 9 | Argmin attention for sort |
| thinking/wall_attention | 7 | Wall selection for render |

### Test Scenes

- **box_room**: 4-wall square room (10×10 units)
- **multi_room**: 22-segment scene with two rooms, corridor, diagonal walls

### Known Limitations

1. **No WAD map loading** — only hand-authored scenes
2. **Ceiling/floor fill** — currently done by host, not transformer
3. **Out-of-bounds columns** — wasted steps for columns outside [0, W)
4. **Fixed wall height** — no variable-height sectors yet
5. **Distance-based sort** — not using BSP structure


## Scope

### North Star (Ideal Goal)

Full E1M1 playable in a compiled transformer:

- Walk through the complete level geometry
- Doors, elevators, switches functional
- Items collectible, enemies fightable
- Raw RGB pixel output, state carried between frames

### Acceptable Reductions

- Subset of visible geometry per frame (≤32 walls)
- Reduced enemy/item count
- Simplified lighting (sector-based, no distance dimming)
- Lower resolution (120×100 baseline)

### Non-Goals (For Now)

- Real-time performance (proof of principle first)
- Multiple maps (E1M1 only)
- Sound, network multiplayer
- Training or learning — this is pure compilation


## Phased Plan

Six phases from current state to E1M1 playable. Each phase has a concrete
deliverable.

### Phase 1: WAD Loading + BSP Integration

Parse E1M1 geometry from the WAD. Replace distance-based sort with BSP rank
sort. First render of real E1M1 geometry.

**New in `wad.py`:**

- Parse VERTEXES lump → vertex coordinates
- Parse LINEDEFS lump → wall endpoints + sidedef references
- Parse SIDEDEFS lump → texture names + sector references
- Parse SEGS lump → wall fragments + subsector assignment
- Parse SSECTORS lump → BSP leaves
- Parse NODES lump → BSP tree structure

**New function: `load_map_subset`:**

```python
def load_map_subset(
    wad_path: str,
    map_name: str,                # "E1M1"
    x: float, y: float,           # Player position
    max_walls: int = 32,
    max_textures: int = 8,
    tex_size: int = 8,
) -> Tuple[List[Segment], List[np.ndarray], List[BspNode]]:
    """Load walls, textures, and BSP subset near (x, y) from a DOOM map.
    
    - Selects closest max_walls segs to player
    - Extracts minimal BSP subtree covering their subsectors
    - Precomputes BSP coefficients per seg
    - Returns ready-to-feed tokens
    """
```

**New in transformer:**

- BSP_NODE token type with plane coefficients `(nx, ny, d)`
- WALL token input expanded with `coeffs_W` (M-dim) + `const_W` (scalar)
- BSP_NODE computes `side_P = sign(nx·px + ny·py + d)` (1 layer)
- WALL gathers `side_P_vec` via attention to BSP_NODEs (1 layer)
- WALL computes `rank = dot(coeffs_W, side_P_vec) + const_W` (1 layer)
- Replace distance-based sort key with BSP rank in SORTED_WALL phase

**Removed:**

- `wall/intersection` for sort (collision still needs it)
- `wall/sort_score` (replaced by BSP rank)

**Map:** E1M1 (WAD), rendered as ~32-wall subset near player.

**Deliverable:** Walk through E1M1 geometry with BSP-based wall ordering.
Walls are fed dynamically from WAD based on player position. BSP tree flows
through the transformer as tokens.

**Estimate:** Net layer change ~0 (remove ~21, add ~3, keep visibility at 17).
Input size grows per WALL (adds M+1 coefficients).


### Phase 2: Variable-Height Sectors

Add DOOM-style architecture with different floor and ceiling heights per
sector. Requires sector data — decide architecture (SECTOR tokens vs
flattened into WALL/BSP_NODE).

**What's new:**

- **Sector data** (architecture TBD based on Phase 1 learnings):
  - Option A: SECTOR tokens, WALL/BSP_LEAF reference by ID
  - Option B: Sector props flattened into WALL tokens
  - Option C: Sector props on BSP leaf nodes
- **Player sector identification** via BSP leaf traversal
- **Multi-section column fill**: upper wall, main wall, lower wall for
  varying sector heights
- **Height clipping** for walls spanning sector boundaries

**Map:** E1M1 region with stairs or height variation.

**Deliverable:** Stairs and platforms render correctly. Walking up/down
stairs changes view appropriately. Player's current sector properties
(floor/ceiling height) available for rendering.

**Estimate:** +5-10 layers for multi-section column fill and sector lookup.


### Phase 3: Floor/Ceiling Rendering in Transformer

Move floor and ceiling fill into the transformer (currently done by host).

**What's new:**

- **Per-row distance**: reciprocal of row offset from horizon
- **Flat-colored floors/ceilings**: use sector colors from Phase 2 data
- **Proper clipping**: floor/ceiling only where no wall

**Deliverable:** Complete frames output by transformer, no host-side fill.
Ceiling and floor colors match sector properties.

**Estimate:** +5-8 layers for per-row reciprocal and fill logic.


### Phase 4: Lighting

Add sector-based and distance-based lighting.

**What's new:**

- **Sector light level** from SECTORS lump (0-255 range)
- **Distance dimming**: distant walls darker
- **RGB multiply**: apply light level to texture colors

**Deliverable:** Dark sectors are dark, distant walls fade. Matches DOOM's
lighting model.

**Estimate:** +3-5 layers for light computation and multiply.


### Phase 5: E1M1 Walkthrough

First major visual milestone: full E1M1 walkthrough with textures, lighting,
and variable heights.

**What's new:**

- Tune `max_walls` for complex E1M1 areas (may need 32-48)
- More textures from WAD
- Full map traversal testing (enter, walk through rooms, navigate layout)
- Performance tuning for larger N

**Deliverable:** GIF walkthrough of E1M1 — enter the level, walk through
rooms, navigate the layout. Visually recognizable as E1M1.

**Estimate:** May need to tune sort/render efficiency for larger N.


### Phase 6: Doors and Elevators

Interactive environment mechanics.

**What's new:**

- **Mutable sector state**: door/elevator positions in game state
- **Sector height offsets**: base height + state offset
- **Use input**: player presses "use" near a door to open it
- **Timed state**: doors close after delay

**Deliverable:** Doors open and close. Elevators move. Can navigate E1M1's
door-blocked areas.

**Estimate:** +5-10 layers for state transitions and conditional heights.


### Future Phases (Deferred)

These phases are documented but not immediately planned:

**Sprites + Items**: Render items/decorations as 2D sprites composited into
the 3D scene. Item pickup updates game state.

**Enemies + Combat**: Enemy state machine, AI movement, hitscan weapons,
damage system. Full E1M1 gameplay.


## Phase Summary

| Phase | Deliverable | Status | ~Layers |
|-------|-------------|--------|---------|
| — | Textured playable renderer (distance sort) | **Done** | 65 |
| 1 | WAD loading + BSP_NODE tokens + BSP sort | Not started | ~65 |
| 2 | Variable-height sectors | Not started | +10 |
| 3 | Floor/ceiling in transformer | Not started | +8 |
| 4 | Sector + distance lighting | Not started | +5 |
| 5 | E1M1 walkthrough | Not started | ~88 |
| 6 | Doors + elevators | Not started | +10 |
| — | Sprites + items | Deferred | — |
| — | Enemies + combat | Deferred | — |


## Technical Notes

### Subset Selection Strategy

`load_map_subset` selects walls by distance from player position. Simple
Euclidean distance from player to wall midpoint (or closest point on wall).
No angle filtering — the transformer's BSP-based sort handles visibility.

For E1M1's ~732 segs, naive sorting is fast enough on the host. If needed,
spatial indexing (grid, quadtree) can speed selection.

### BSP Subset Extraction

Given selected segs and their subsectors, find the minimal BSP subtree:
1. Collect unique subsector IDs
2. Walk each subsector's ancestor path in the full BSP
3. Union all ancestors — these are the nodes in the subset
4. Renumber 0..M-1 for the transformer

For 32 segs from ~25 subsectors with depth ~8, expect ~30-40 BSP_NODE tokens.

### Texture Management

Each frame uses at most `max_textures` distinct textures (limited by TEX_COL
prefill size and attention capacity). The subset selector:

1. Selects walls by distance
2. Collects unique texture IDs from selected walls
3. If > max_textures, prioritizes textures on closer walls
4. Loads and downscales selected textures from WAD

### Layer Budget

Target: ≤120 layers at d_model ≤ 4096.

Current: 65 layers at d_model 2048 for 8 walls.

Phase 1 is roughly layer-neutral (removes distance sort layers, adds BSP
rank layers). Subsequent phases have ~45 layers of headroom for variable
heights, lighting, doors, and beyond.

### Sequence Length Scaling

The sort phase is O(N) autoregressive steps. At ~70ms/step on A100:

| Walls | Sort steps | Sort time |
|-------|------------|-----------|
| 8 | 8 | ~0.6s |
| 16 | 16 | ~1.1s |
| 32 | 32 | ~2.2s |
| 48 | 48 | ~3.4s |

Prefill grows with N + M (walls + BSP nodes): ~75 tokens for a 32-wall
subset vs ~42 currently.


## Open Questions

- **Optimal max_walls**: What's the minimum N that covers typical E1M1
  views? Need to test with real map data.

- **Sector architecture (Phase 2)**: SECTOR tokens vs flattened vs on BSP
  leaves. Decide based on Phase 1 learnings about BSP_NODE patterns.

- **Door state encoding**: How many simultaneously-animating doors does
  E1M1 require? Affects game state vector size.

- **Texture atlas size**: E1M1 uses ~40 unique textures. With max_textures=8
  per frame, need priority logic for which to load.

- **BSP rank encoding**: `side_P` as ±1 or 0/1? Formula assumes 0/1 for
  clean XOR; map `compare` output accordingly.
