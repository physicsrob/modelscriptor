"""Build a torchwright graph that renders a flat-shaded first-person view.

The graph takes (perp_cos, player_x, player_y, ray_angle) per position and
outputs H*3 RGB values (one screen column).  Segment geometry is baked into
the weights as constants.
"""

from typing import List, Tuple

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.ops.arithmetic_ops import (
    abs,
    compare,
    multiply_const,
    negate,
    reciprocal,
    reduce_min,
    signed_multiply,
    subtract,
)
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.map_select import broadcast_select, in_range, select
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


# ---------------------------------------------------------------------------
# Stage 1: Trig lookup
# ---------------------------------------------------------------------------

def trig_lookup(ray_angle: Node) -> Tuple[Node, Node]:
    """Look up (cos, sin) via piecewise-linear interpolation over 256 entries.

    Exact at integer angle indices.  Uses two separate piecewise_linear
    ops (one for cos, one for sin) since map_to_table doesn't discriminate
    well for scalar integer keys.
    """
    from torchwright.ops.arithmetic_ops import piecewise_linear

    table = generate_trig_table()  # (256, 2): col0=cos, col1=sin
    breakpoints = list(range(256))
    cos_values = [float(table[i, 0]) for i in range(256)]
    sin_values = [float(table[i, 1]) for i in range(256)]

    ray_cos = piecewise_linear(ray_angle, breakpoints, cos_values, name="trig_cos")
    ray_sin = piecewise_linear(ray_angle, breakpoints, sin_values, name="trig_sin")
    return ray_cos, ray_sin


# ---------------------------------------------------------------------------
# Stage 3: Per-segment intersection (Linear nodes — zero FFN cost)
# ---------------------------------------------------------------------------

def _segment_intersection(
    px: Node,
    py: Node,
    ray_cos: Node,
    ray_sin: Node,
    px_sin: Node,
    py_cos: Node,
    seg: Segment,
) -> Tuple[Node, Node, Node]:
    """Compute (den, num_t, num_u) for one segment.

    All outputs are scalar Linear nodes — zero FFN layers.

    den   = cos * ey - sin * ex
    num_t = (ax*ey - ay*ex) + ex*py - ey*px
    num_u = (ax*sin - ay*cos) + (py_cos - px_sin)
    """
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay

    # den = cos * ey - sin * ex  (linear in cos, sin)
    cos_sin = Concatenate([ray_cos, ray_sin])
    den_matrix = torch.tensor([[ey], [-ex]])
    den = Linear(cos_sin, den_matrix, name="den")

    # num_t = (ax*ey - ay*ex) + ex*py - ey*px  (linear in px, py)
    const_t = seg.ax * ey - seg.ay * ex
    px_py = Concatenate([px, py])
    num_t_matrix = torch.tensor([[-ey], [ex]])
    num_t_bias = torch.tensor([const_t])
    num_t = Linear(px_py, num_t_matrix, num_t_bias, name="num_t")

    # num_u = (ax*sin - ay*cos) + (py_cos - px_sin)
    # = ax*sin - ay*cos + py*cos - px*sin
    # Linear in (ray_cos, ray_sin, px_sin, py_cos)
    inputs_u = Concatenate([ray_cos, ray_sin, px_sin, py_cos])
    num_u_matrix = torch.tensor([[-seg.ay], [seg.ax], [-1.0], [1.0]])
    num_u = Linear(inputs_u, num_u_matrix, name="num_u")

    return den, num_t, num_u


# ---------------------------------------------------------------------------
# Stage 4: Validity + distance
# ---------------------------------------------------------------------------

BIG_DISTANCE = 1000.0


def _segment_distance(
    den: Node,
    num_t: Node,
    num_u: Node,
    max_coord: float,
) -> Node:
    """Compute masked distance for one segment intersection.

    Returns BIG_DISTANCE if the intersection is invalid.

    Uses sign normalization: if den < 0, negate both num_t and den so we
    can treat den as positive for all subsequent checks.
    """
    # Bounds — keep tight to avoid precision issues from overly large
    # piecewise_linear tables.
    # den is bounded by segment_span (≤ 2*max_coord) * trig (≤ 1)
    max_den = 2.0 * max_coord
    # num_t is bounded by max_coord * segment_span ≤ 2*max_coord^2
    max_num_t = 2.0 * max_coord * max_coord
    # Max distance we care about
    max_dist = BIG_DISTANCE
    epsilon = 0.05

    # Step 1: Check den sign and normalize
    is_den_pos = compare(den, 0.0)

    # Normalize: make den effectively positive
    abs_den = abs(den)
    adj_num_t = select(is_den_pos, num_t, negate(num_t))
    adj_num_u = select(is_den_pos, num_u, negate(num_u))

    # Step 2: Validity checks (all assume den > 0 after normalization)
    is_den_nonzero = compare(abs_den, epsilon)
    is_t_pos = compare(adj_num_t, epsilon)
    is_u_ge_0 = compare(adj_num_u, -epsilon)
    u_minus_den = subtract(abs_den, adj_num_u)
    is_u_le_den = compare(u_minus_den, -epsilon)

    # Step 3: Compute distance t = num_t / den
    inv_den = reciprocal(abs_den, min_value=epsilon, max_value=max_den, step=0.5)
    dist = signed_multiply(
        adj_num_t, inv_den,
        max_abs1=max_num_t, max_abs2=1.0 / epsilon,
        step=1.0,
        max_abs_output=max_dist,
    )

    # Step 4: Mask invalid intersections to BIG_DISTANCE
    big = LiteralValue(torch.tensor([BIG_DISTANCE]), name="big_dist")
    dist = select(is_den_nonzero, dist, big)
    dist = select(is_t_pos, dist, big)
    dist = select(is_u_ge_0, dist, big)
    dist = select(is_u_le_den, dist, big)

    return dist


# ---------------------------------------------------------------------------
# Stage 7: Column fill
# ---------------------------------------------------------------------------

def _column_fill(
    wall_top: Node,
    wall_bottom: Node,
    wall_color: Node,
    config: RenderConfig,
) -> Node:
    """Fill a screen column with ceiling, wall, and floor colors.

    Uses vectorized in_range + broadcast_select to avoid per-row node
    fan-out that deadlocks the compiler scheduler.

    Two-pass approach:
      1. Base layer: floor color where row >= wall_bottom, ceiling elsewhere.
      2. Overlay: wall color in [wall_top, wall_bottom).

    Cost: 4 FFN layers (2 in_range + 2 broadcast_select).
    """
    H = config.screen_height
    ceil_node = LiteralValue(torch.tensor(list(config.ceiling_color)), name="ceiling")
    floor_node = LiteralValue(torch.tensor(list(config.floor_color)), name="floor")
    h_node = LiteralValue(torch.tensor([float(H)]), name="h_bound")

    # Step 1: Base column — floor below wall_bottom, ceiling above
    floor_masks = in_range(wall_bottom, h_node, H)
    base = broadcast_select(floor_masks, floor_node, ceil_node, H, 3)

    # Step 2: Overlay wall color in [wall_top, wall_bottom)
    wall_masks = in_range(wall_top, wall_bottom, H)
    return broadcast_select(wall_masks, wall_color, base, H, 3)


# ---------------------------------------------------------------------------
# Full graph
# ---------------------------------------------------------------------------

def build_renderer_graph(
    segments: List[Segment],
    config: RenderConfig,
    max_coord: float = 20.0,
) -> Tuple[Node, "PosEncoding"]:
    """Build the complete per-column rendering graph.

    Inputs (per position, alphabetical order for headless module):
        perp_cos:  cos(ray_angle - player_angle) for fish-eye correction
        player_x:  world x coordinate
        player_y:  world y coordinate
        ray_angle: integer 0-255, index into trig table

    Output: H*3 floats — RGB for each row of this screen column.

    Args:
        segments: Wall segments with constant geometry baked into weights.
        config: Render configuration (screen size, colors).
        max_coord: Upper bound on coordinate magnitudes.

    Returns:
        (output_node, pos_encoding) tuple for compilation.
    """
    pos_encoding = create_pos_encoding()

    # Inputs (names chosen so alphabetical order = perp_cos, player_x, player_y, ray_angle)
    perp_cos = create_input("perp_cos", 1)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    ray_angle = create_input("ray_angle", 1)

    # Stage 1: Trig lookup
    ray_cos, ray_sin = trig_lookup(ray_angle)

    # Stage 2: Shared products (needed for num_u across all segments)
    px_sin = signed_multiply(
        player_x, ray_sin,
        max_abs1=max_coord, max_abs2=1.0, step=0.1,
    )
    py_cos = signed_multiply(
        player_y, ray_cos,
        max_abs1=max_coord, max_abs2=1.0, step=0.1,
    )

    # Stages 3-4: Per-segment intersection + validity + distance
    distances = []
    colors = []
    for seg in segments:
        den, num_t, num_u = _segment_intersection(
            player_x, player_y, ray_cos, ray_sin, px_sin, py_cos, seg,
        )
        dist = _segment_distance(den, num_t, num_u, max_coord)
        distances.append(dist)
        colors.append(LiteralValue(torch.tensor(list(seg.color)), name="seg_color"))

    # Stage 5: Min-reduction — find nearest segment
    if len(segments) == 0:
        closest_dist = LiteralValue(torch.tensor([BIG_DISTANCE]), name="no_hit")
        wall_color = LiteralValue(torch.tensor([0.0, 0.0, 0.0]), name="no_color")
    elif len(segments) == 1:
        closest_dist = distances[0]
        wall_color = colors[0]
    else:
        closest_dist, wall_color = reduce_min(distances, colors)

    # Stage 6: Wall height via fish-eye corrected distance.
    # max_dist bounds the longest distance we need precision for — beyond
    # this, walls are so far away they render as 0-1 pixels. Using tight
    # bounds avoids enormous piecewise_linear tables that choke the scheduler.
    max_dist = 2.0 * max_coord

    # Clamp closest_dist into [0.5, max_dist] before the fish-eye multiply,
    # so the signed_multiply inputs stay within their declared ranges.
    # Distances >= max_dist produce walls of ~0 pixels anyway.
    min_dist_node = LiteralValue(torch.tensor([0.5]), name="min_dist")
    max_dist_node = LiteralValue(torch.tensor([max_dist]), name="max_dist")
    is_above_min = compare(closest_dist, 0.5)
    clamped_dist = select(is_above_min, closest_dist, min_dist_node)
    is_below_max = compare(clamped_dist, max_dist - 0.5)
    # is_below_max is true when clamped_dist > max_dist-0.5, meaning TOO BIG
    # We want to keep clamped_dist when it's <= max_dist
    # Invert: compare returns 1.0 when input > thresh
    clamped_dist = select(is_below_max, max_dist_node, clamped_dist)

    perp_dist = signed_multiply(
        clamped_dist, perp_cos,
        max_abs1=max_dist, max_abs2=1.0, step=1.0,
    )
    # Clamp perp_dist away from zero
    eps_node = LiteralValue(torch.tensor([0.5]), name="min_perp")
    is_too_small = compare(perp_dist, 0.5)
    safe_perp_dist = select(is_too_small, perp_dist, eps_node)

    inv_perp = reciprocal(safe_perp_dist, min_value=0.5, max_value=max_dist, step=1.0)
    wall_height = multiply_const(inv_perp, float(config.screen_height))

    # Wall bounds: center ± half wall height.
    # Pass continuous bounds directly to in_range, which uses a +0.5 offset
    # to classify each row. This may differ from the reference renderer's
    # int() truncation by at most 1 pixel at wall boundaries.
    H = config.screen_height
    center = float(H) / 2.0
    half_height = multiply_const(wall_height, 0.5)
    wall_top = Linear(
        half_height,
        torch.tensor([[-1.0]]),
        torch.tensor([center]),
        name="wall_top",
    )
    wall_bottom = Linear(
        half_height,
        torch.tensor([[1.0]]),
        torch.tensor([center]),
        name="wall_bottom",
    )

    # Handle "no hit" case: if closest_dist >= BIG, show only ceiling/floor
    # wall_top will be near center and wall_bottom near center (height ≈ 0)
    # This happens naturally since 1/BIG ≈ 0 → wall_height ≈ 0

    # Stage 7: Column fill
    output = _column_fill(wall_top, wall_bottom, wall_color, config)

    return output, pos_encoding
