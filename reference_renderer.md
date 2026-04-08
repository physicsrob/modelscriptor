# Reference Software Renderer

A simple Python renderer that produces pixel-exact output for verifying the
compiled transformer at each phase. The reference renderer and the torchwright
graph must implement **identical math** — same fixed-point representation,
same trig tables, same intersection formula. Any divergence is a bug in one
or the other.

## Role

This is a test oracle, not a game engine. It needs to be:

- **Correct** — mathematically identical to the transformer's computation
- **Readable** — simple enough to audit by hand for small cases
- **Slow is fine** — no performance requirements whatsoever

It does NOT need to:

- Run in real time
- Handle input or game state (the caller provides player position and angle)
- Be extensible or well-architected


## Interface

```python
def render_frame(
    player_x: float,
    player_y: float,
    player_angle: int,       # 0–255, index into trig table
    segments: List[Segment],
    config: RenderConfig,
) -> np.ndarray:
    """
    Returns an RGB image as a numpy array of shape
    (config.screen_height, config.screen_width, 3).

    Pixel values are floats in the same scale/range as the transformer's
    residual stream output.
    """
```

```python
@dataclass
class Segment:
    ax: float       # endpoint A
    ay: float
    bx: float       # endpoint B
    by: float
    color: tuple    # (r, g, b)

@dataclass
class RenderConfig:
    screen_width: int    # number of columns (autoregressive positions)
    screen_height: int   # number of pixel rows per column
    fov_columns: int     # how many trig table entries the FOV spans
    trig_table: np.ndarray  # precomputed sin/cos, shape (256, 2)
    ceiling_color: tuple
    floor_color: tuple
```

## Algorithm

For each screen column `col` in `0..screen_width-1`:

### 1. Ray direction

Compute the ray angle index:

```
ray_angle = (player_angle + col - screen_width // 2) % 256
```

(The exact column-to-angle mapping may change — the key requirement is that
this matches the transformer's mapping exactly.)

Look up `sin(ray_angle)` and `cos(ray_angle)` from `trig_table`. These are
the **same values** the transformer uses (baked into its `map_to_table`
weights).

### 2. Shared products

Compute `Px * sin(ray_angle)`, `Py * cos(ray_angle)`, etc. — the same
products the transformer computes via `multiply_integers`. Use the same
precision/rounding if applicable.

### 3. Per-segment intersection

For each segment, compute:

```
# Direction vector
dx = cos(ray_angle)
dy = sin(ray_angle)

# Segment vector
ex = bx - ax
ey = by - ay

# Cross products
den   = dx * ey - dy * ex
num_t = (ax - px) * ey - (ay - py) * ex
num_u = (ax - px) * dy - (ay - py) * dx
```

Intersection is valid when:

- `den != 0` (ray not parallel to segment)
- `t > 0` (intersection is in front of player)
- `0 <= u <= 1` (intersection is within segment bounds)

where `t = num_t / den` and `u = num_u / den`. The sign checks can be done
on the numerators and denominator without division (to match the
transformer's comparison approach).

Record the valid intersection with the smallest `t`.

### 4. Wall height

```
# Correct for fisheye
perp_distance = t * cos(ray_angle - player_angle)

wall_height = screen_height / perp_distance  # or some scaled version
```

The exact wall height formula (scaling factor, clamping) must match the
transformer.

### 5. Column fill

```
wall_top  = screen_height / 2 - wall_height / 2
wall_bottom = screen_height / 2 + wall_height / 2

for row in 0..screen_height-1:
    if row < wall_top:
        pixel = ceiling_color
    elif row < wall_bottom:
        pixel = segment_color
    else:
        pixel = floor_color
```

## Precision Contract

The reference renderer must match the transformer **exactly**, not
approximately. This means:

- **Trig values**: looked up from the same table, not computed with
  `math.sin`. The trig table is a shared artifact passed to both the renderer
  and the graph constructor.
- **Multiplication**: if the transformer uses fixed-point with a specific
  scale factor, the reference renderer must too. Whether this means the
  reference renderer operates in floats and the values happen to align, or
  it explicitly uses integer arithmetic with the same rounding, depends on
  the coordinate representation (see open question below).
- **Comparisons**: the transformer's `compare` has a smooth step
  (step_sharpness=10), meaning values within ~0.1 of the threshold are
  ambiguous. The reference renderer should use hard thresholds. Test cases
  should avoid positions where a ray grazes a segment endpoint within this
  margin.

## Testing Strategy

Provide a set of test cases as `(player_x, player_y, player_angle)` tuples
paired with a segment list. These should include:

- **Head-on wall**: player facing a wall straight ahead
- **Angled wall**: player facing a diagonal segment
- **Corner view**: two walls meeting at a corner
- **Through doorway**: player looking through a gap between segments into a
  room behind
- **Parallel ray**: ray nearly parallel to a segment (edge case for `den ≈ 0`)
- **Behind player**: segments entirely behind the player (should not render)
- **Multiple depths**: two walls at different distances in the same column

Each test case should be verified by hand (or by diagram) for at least one
column before trusting the renderer as an oracle.

## Open Questions

- **Coordinate representation**: the reference renderer operates in floats.
  If the transformer uses fixed-point integers, we need a conversion layer
  (or the reference renderer operates in the same fixed-point). This decision
  affects how "pixel-exact match" works — do we compare float outputs within
  epsilon, or do we compare exact integer values? Depends on the coordinate
  representation decision.
- **Column-to-angle mapping**: the exact formula for which trig table entry
  each column uses. `(player_angle + col - screen_width/2) % 256` is a
  starting point but the FOV scaling may differ.
- **Wall height formula**: the exact scaling constant and whether/how wall
  height is clamped when the player is very close to a wall.
- **Pixel value range**: are RGB values 0–255 integers? 0.0–1.0 floats?
  Some other range that's convenient for the residual stream?

## Phased Extensions

The reference renderer grows alongside the transformer:

- **Phase 2**: flat-shaded walls, uniform ceiling/floor — the spec above
- **Phase 4**: variable-height sectors — `Segment` gains front/back sector
  IDs, column fill uses sector floor/ceiling heights
- **Phase 5**: textured walls and floors — texture lookup added to column
  fill, lighting applied
- **Phase 6**: E1M1 geometry — same renderer, just more segments and sectors
- **Phase 8**: sprites — sprite projection and depth compositing added
