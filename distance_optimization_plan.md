# Distance Optimization Plan

Precompute perpendicular distance at sort time to eliminate redundant
per-column intersection work during rendering.

## Observation

For a flat wall segment, the **perpendicular distance** from the player
to the wall's infinite line is a single scalar:

    perp_dist = |cross(P - A, B - A)| / |B - A|

This value is independent of ray direction.  In the current DOOM
renderer, `wall_height = H / (dist_r * perp_cos)` where `dist_r` is
the per-column ray distance and `perp_cos` is the fish-eye correction.
The product `dist_r * perp_cos` equals the perpendicular distance for
any ray hitting the same infinite wall line.  So wall height can be
computed directly from `perp_dist` without any per-column intersection:

    wall_height = H / perp_dist

No fish-eye correction needed --- we're starting from the perpendicular
distance, not the ray distance.

## Current per-RENDER-token cost (game_graph.py lines 736--785)

The parametric intersection at each RENDER token currently performs:

| Step | What | MLP sublayers | Why |
|------|------|---:|-----|
| Products | 6x `piecewise_linear_2d` (ey*cos, ex*sin, ey*dx, ex*dy, dx*sin, dy*cos) | 6 | 2 for den, 2 for num_t, 2 for num_u |
| den processing | sign/abs/inv_abs_den | ~3 | Reciprocal + sign normalization |
| Distance | adj_num_t, dist = adj_num_t * inv_den, validity | ~5 | signed_multiply + 5 compare + bool_all_true |
| Wall height | `_wall_height_lookup(dist_r, perp_cos)` | ~5 | perp_dist multiply + reciprocal + scale |
| U normalization | `_u_norm_lookup(adj_num_u, abs_den)` | ~5 | linear_bin_index |
| **Total** | | **~24** | |

Of these, wall height (5 sublayers) depends only on the final distance
scalar, which can be replaced by the precomputed perpendicular distance.
The distance computation itself (products for num_t + distance steps =
2 + 5 = 7 sublayers) also becomes unnecessary.

## What stays: u-coordinate

The texture column index `tex_col_idx` is derived from `(adj_num_u,
abs_den)`, which requires the per-column intersection.  Specifically:

- `den = ey * cos - ex * sin` (2 products: ey*cos, ex*sin)
- `num_u = dx * sin + dy * cos` (2 products: dx*sin, dy*cos)
- `sign_den, abs_den` from den (compare + abs)
- `adj_num_u = select(sign_den, num_u, -num_u)`
- `tex_col_idx = linear_bin_index(adj_num_u, 0, abs_den, tex_w)`

This requires 4 of the 6 `piecewise_linear_2d` products plus the den
processing and u normalization.  It cannot be precomputed because it
depends on the per-column `ray_cos / ray_sin`.

## Savings

| Step | Current | After optimization | Savings |
|------|---:|---:|---:|
| Products | 6 `piecewise_linear_2d` | 4 `piecewise_linear_2d` | 2 sublayers |
| den processing | ~3 | ~3 (still needed for u) | 0 |
| Distance | ~5 | 0 (skipped) | 5 sublayers |
| Wall height | ~5 (dist*perp_cos + reciprocal) | ~2 (reciprocal of perp_dist only) | 3 sublayers |
| U normalization | ~5 | ~5 (unchanged) | 0 |
| **Total** | **~24** | **~14** | **~10 sublayers** |

Saving ~10 MLP sublayers per RENDER token.  With W render tokens per
frame, this is 10 * W fewer layers of computation per frame.

## Implementation plan

### Step 1: Compute perp_dist at SORTED_WALL time

In the WALL token phase (game_graph.py ~344--404), add:

```python
# Wall edge vector (already computed as w_ex, w_ey)
# Player-to-wall-A vector (already computed as w_fx = ax - px, w_gy = py - ay)
# cross(P-A, B-A) = w_fx * w_ey - (-w_gy) * w_ex = w_fx * w_ey + w_gy * w_ex
# |B-A| = sqrt(w_ex^2 + w_ey^2)
# perp_dist = |cross| / |B-A|
```

The cross product requires 2 `piecewise_linear_2d` products
(w_fx * w_ey, w_gy * w_ex), then a subtract and abs.  The wall length
|B-A| requires `square_signed(w_ex)` + `square_signed(w_ey)` + sum +
sqrt (via `piecewise_linear`).  The division requires a reciprocal +
signed_multiply.

Total cost: ~6--8 MLP sublayers at the WALL token.  This runs once per
wall during prefill, not per render column, so it adds to prefill
depth but saves 10 * W sublayer-evaluations during rendering.

### Step 2: Pack perp_dist into the sort value

Extend `wall_value_for_sort` to include `perp_dist`:

```python
wall_value_for_sort = Concatenate([
    wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,
    w_dx, w_dy, center_ray_dist, perp_dist,       # +1 value
    position_onehot,
])
d_sort_val = 9 + max_walls                         # was 8
```

### Step 3: Pass perp_dist through render attention

Extend the render attention value passthrough from 5 to 6 values:

```python
# Value: [ax, ay, bx, by, tex_id, perp_dist]
v_matrix = torch.zeros(len(render_attn_in), d_head_render)
for i in range(6):                                  # was 5
    v_matrix[s_wall_data + i, W + 1 + i] = 1.0

o_matrix = torch.zeros(d_head_render, 6)            # was 5
for i in range(6):
    o_matrix[W + 1 + i, i] = 1.0
```

d_head_render becomes W + 1 + 6 = W + 7 (was W + 6).

### Step 4: Use perp_dist for wall height

Replace the current wall height computation:

```python
# Current: dist_r → _wall_height_lookup(dist_r, perp_cos)
# New: perp_dist directly
r_perp_dist = _extract_from(render_attn, 6, 5, 1, "r_perp_dist")
inv_perp_dist = reciprocal(r_perp_dist, min_value=0.5, max_value=2.0*max_coord)
wall_height = multiply_const(inv_perp_dist, float(H))
wall_top = ...  # center - half_height
wall_bottom = ...  # center + half_height
```

### Step 5: Drop num_t computation

Remove the 2 `piecewise_linear_2d` products for num_t (ey*dx, ex*dy)
and the adj_num_t / dist_r / validity pipeline.  Keep den and num_u
computation for the u-coordinate.

The validity checks (is_den_nz, is_t_pos, is_u_ge0, is_u_le_den)
currently gate dist_r to BIG_DISTANCE for invalid intersections.
With precomputed perp_dist, the distance is always valid (the wall
was selected by the visibility mask attention).  The u-coordinate
clamping in linear_bin_index handles edge cases.

### Step 6: Adjust the RENDER pipeline

The render section (game_graph.py ~736--785) becomes:

```python
# --- Parametric intersection (u-coordinate only) ---
ex = subtract(r_wall_bx, r_wall_ax)
ey = subtract(r_wall_by, r_wall_ay)
dx_r = subtract(r_wall_ax, player_x)
dy_r = subtract(player_y, r_wall_ay)

# den (needed for u normalization)
ey_cos = piecewise_linear_2d(ey, ray_cos, ...)
ex_sin = piecewise_linear_2d(ex, ray_sin, ...)
den = subtract(ey_cos, ex_sin)

# num_u (needed for u normalization)
dx_sin = piecewise_linear_2d(dx_r, ray_sin, ...)
dy_cos = piecewise_linear_2d(dy_r, ray_cos, ...)
num_u = add(dx_sin, dy_cos)

# den processing (for u norm)
sign_den = compare(den, 0.0)
abs_den = abs(den)
adj_num_u = select(sign_den, num_u, negate(num_u))

# Wall height from precomputed perp_dist (no fish-eye needed)
inv_perp_dist = reciprocal(r_perp_dist, ...)
wall_height = multiply_const(inv_perp_dist, float(H))
# ... wall_top, wall_bottom as before

# Texture column (unchanged)
tex_col_idx = _u_norm_lookup(adj_num_u, abs_den, tex_w, max_coord)
```

## Further opportunity: u-coordinate from screen-space interpolation

**Not recommended for initial implementation (sacrifices accuracy).**

If the sort phase also precomputed `col_a` and `col_b` (the screen
column where each wall endpoint projects), the u-coordinate could be
approximated without any parametric intersection:

    u = (col_idx - col_a) / (col_b - col_a)

This would eliminate all 4 remaining `piecewise_linear_2d` products
and the den processing, saving an additional ~8 sublayers per RENDER
token.  But the screen-space interpolation is not perspective-correct:
`u` does not vary linearly with screen column for walls at oblique
angles.  For DOOM's vertical-wall geometry the error is small but
non-zero.

The endpoint projections are already computed at sort time
(game_graph.py ~546--631: `col_a`, `col_b` for the visibility mask).
Passing them through the render attention value (2 more floats) and
using them for u interpolation would be straightforward mechanically.

**Recommendation:** note this as a future "fast mode" option.  The
accuracy tradeoff should be evaluated empirically (render a test scene,
diff against reference) before committing to it.

## Testing

1. Compile a game graph with the optimization enabled
2. Run `step_frame` on `box_room_textured` and `multi_room_textured`
3. Compare rendered frames pixel-by-pixel against the reference renderer
4. Wall heights should match exactly (perpendicular distance is exact
   for flat walls)
5. Texture columns should be unchanged (u pipeline is unmodified)
