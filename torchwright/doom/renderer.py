"""Build a torchwright graph that renders a flat-shaded first-person view.

The graph takes (perp_cos, player_x, player_y, ray_angle) per position and
outputs H*3 RGB values (one screen column).  Segment geometry is baked into
the weights as constants.
"""

from typing import List, Tuple

import builtins

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.ops.arithmetic_ops import (
    abs,
    clamp,
    compare,
    multiply_const,
    negate,
    piecewise_linear_nd,
    reciprocal,
    reduce_min,
    signed_multiply,
    subtract,
)
from torchwright.ops.logic_ops import bool_all_true
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
    cos_sin: Node,
    px_py: Node,
    trig_and_products: Node,
    seg: Segment,
) -> Tuple[Node, Node, Node]:
    """Compute (den, num_t, num_u) for one segment.

    All outputs are scalar Linear nodes — zero FFN layers.

    Args:
        cos_sin: Concatenate([ray_cos, ray_sin]), shared across segments.
        px_py: Concatenate([px, py]), shared across segments.
        trig_and_products: Concatenate([ray_cos, ray_sin, px_sin, py_cos]),
            shared across segments.
        seg: Wall segment with constant endpoints.

    Returns:
        (den, num_t, num_u) scalar nodes.
    """
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay

    # den = cos * ey - sin * ex  (linear in cos, sin)
    den = Linear(cos_sin, torch.tensor([[ey], [-ex]]), name="den")

    # num_t = (ax*ey - ay*ex) + ex*py - ey*px  (linear in px, py)
    const_t = seg.ax * ey - seg.ay * ex
    num_t = Linear(px_py, torch.tensor([[-ey], [ex]]),
                   torch.tensor([const_t]), name="num_t")

    # num_u = (ax*sin - ay*cos) + (py_cos - px_sin)
    num_u = Linear(trig_and_products,
                   torch.tensor([[-seg.ay], [seg.ax], [-1.0], [1.0]]),
                   name="num_u")

    return den, num_t, num_u


# ---------------------------------------------------------------------------
# Stage 4: Validity + distance
# ---------------------------------------------------------------------------

BIG_DISTANCE = 1000.0


def _segment_distance(
    num_t: Node,
    num_u: Node,
    signed_inv_den: Node,
    abs_den: Node,
    sign_den: Node,
    max_coord: float,
) -> Node:
    """Compute masked distance for one segment intersection.

    Uses precomputed per-angle values (signed_inv_den, abs_den, sign_den)
    from a lookup table, eliminating runtime reciprocal and sign checks.

    Args:
        num_t: Intersection numerator (linear in px, py).
        num_u: Segment parameter numerator (linear in trig + shared products).
        signed_inv_den: Precomputed 1/den preserving sign (from angle lookup).
        abs_den: Precomputed |den| (from angle lookup).
        sign_den: Precomputed sign(den) as ±1 (from angle lookup).
        max_coord: Upper bound on coordinate magnitudes.

    Returns:
        Scalar distance node (BIG_DISTANCE if invalid).
    """
    max_num_t = 2.0 * max_coord * max_coord
    max_inv_den = 100.0  # 1/epsilon where epsilon=0.01 in the lookup
    epsilon = 0.05

    # Sign-normalize num_t and num_u using precomputed sign_den.
    # select(sign_den, x, -x) makes both behave as if den > 0.
    adj_num_t = select(sign_den, num_t, negate(num_t))
    adj_num_u = select(sign_den, num_u, negate(num_u))

    # Validity checks
    is_den_nonzero = compare(abs_den, epsilon)
    is_t_pos = compare(adj_num_t, epsilon)
    is_u_ge_0 = compare(adj_num_u, -epsilon)
    u_minus_den = subtract(abs_den, adj_num_u)
    is_u_le_den = compare(u_minus_den, -epsilon)

    # Distance: t = num_t * signed_inv_den (= num_t / den)
    # Use adj_num_t * abs(signed_inv_den) = adj_num_t * (1/abs_den)
    # signed_inv_den has the right sign built in, but we already
    # sign-normalized num_t, so use the absolute reciprocal.
    abs_inv_den = select(sign_den, signed_inv_den, negate(signed_inv_den))
    dist = signed_multiply(
        adj_num_t, abs_inv_den,
        max_abs1=max_num_t, max_abs2=max_inv_den,
        step=1.0,
        max_abs_output=BIG_DISTANCE,
    )

    # Mask invalid intersections
    is_valid = bool_all_true([is_den_nonzero, is_t_pos, is_u_ge_0, is_u_le_den])
    big = LiteralValue(torch.tensor([BIG_DISTANCE]), name="big_dist")
    dist = select(is_valid, dist, big)

    return dist


def _build_angle_lookup(
    ray_angle: Node,
    segments: List[Segment],
) -> List[Tuple[Node, Node, Node]]:
    """Precompute per-segment den-related values as a function of ray angle.

    For each segment, den = cos(angle)*ey - sin(angle)*ex is fully determined
    by the ray angle and constant segment endpoints. Precomputing 1/den, |den|,
    and sign(den) for all 256 angles eliminates runtime reciprocal and sign
    normalization (saves ~5 ReLU layers on the per-segment critical path).

    Returns a list of (signed_inv_den, abs_den, sign_den) node tuples,
    one per segment.  All segments share a single piecewise_linear_nd lookup
    (1 FFN layer total, regardless of segment count).
    """
    trig_table = generate_trig_table()
    n_segs = len(segments)
    epsilon = 0.01  # threshold for "den is zero"

    # Precompute the 3 values per segment per angle
    # Output layout: [signed_inv_den_0, abs_den_0, sign_den_0,
    #                 signed_inv_den_1, abs_den_1, sign_den_1, ...]
    d_out = 3 * n_segs
    breakpoints = list(range(256))
    values = []
    for angle_idx in range(256):
        cos_a = float(trig_table[angle_idx, 0])
        sin_a = float(trig_table[angle_idx, 1])
        row = []
        for seg in segments:
            ex = seg.bx - seg.ax
            ey = seg.by - seg.ay
            den = cos_a * ey - sin_a * ex
            if builtins.abs(den) < epsilon:
                row.extend([0.0, 0.0, 1.0])  # invalid: inv=0, abs=0, sign=+1
            else:
                row.extend([1.0 / den, builtins.abs(den), 1.0 if den > 0 else -1.0])
        values.append(row)

    # Single lookup: 1 FFN layer, outputs 3*N values
    all_data = piecewise_linear_nd(
        ray_angle, breakpoints, values, name="angle_lookup",
    )

    # Split into per-segment (signed_inv_den, abs_den, sign_den) triples
    result = []
    for i in range(n_segs):
        offset = i * 3
        signed_inv_den = Linear(
            all_data,
            _extract_matrix(d_out, offset),
            name=f"signed_inv_den_{i}",
        )
        abs_den_node = Linear(
            all_data,
            _extract_matrix(d_out, offset + 1),
            name=f"abs_den_{i}",
        )
        sign_den = Linear(
            all_data,
            _extract_matrix(d_out, offset + 2),
            name=f"sign_den_{i}",
        )
        result.append((signed_inv_den, abs_den_node, sign_den))

    return result


def _extract_matrix(d_in: int, idx: int) -> torch.Tensor:
    """Create a (d_in, 1) matrix that extracts column idx."""
    m = torch.zeros(d_in, 1)
    m[idx, 0] = 1.0
    return m


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

def build_rendering_pipeline(
    player_x: Node,
    player_y: Node,
    ray_angle: Node,
    perp_cos: Node,
    segments: List[Segment],
    config: RenderConfig,
    max_coord: float = 20.0,
) -> Node:
    """Build the rendering pipeline from graph nodes.

    This is the core rendering graph.  It can be called with InputNodes
    (Phase 2 standalone renderer) or with computed nodes (Phase 3+ game
    graph where position and angle are derived from game logic).

    Args:
        player_x, player_y: World-coordinate nodes (scalar).
        ray_angle: Integer 0-255 angle node (scalar).
        perp_cos: Fish-eye correction cos(ray_angle - player_angle) (scalar).
        segments: Wall segments with constant geometry baked into weights.
        config: Render configuration (screen size, colors).
        max_coord: Upper bound on coordinate magnitudes.

    Returns:
        Output node: H*3 floats (RGB for each row of this screen column).
    """
    # Stage 1: Trig lookup
    ray_cos, ray_sin = trig_lookup(ray_angle)

    # Stage 2: Shared products (needed for num_u across all segments)
    px_sin = signed_multiply(
        player_x, ray_sin,
        max_abs1=max_coord, max_abs2=1.0, step=0.25,
    )
    py_cos = signed_multiply(
        player_y, ray_cos,
        max_abs1=max_coord, max_abs2=1.0, step=0.25,
    )

    # Shared Concatenates — created once, reused by all segments
    cos_sin = Concatenate([ray_cos, ray_sin])
    px_py = Concatenate([player_x, player_y])
    trig_and_products = Concatenate([ray_cos, ray_sin, px_sin, py_cos])

    # Per-angle lookup: precompute signed_inv_den, abs_den, sign_den for
    # all segments in a single piecewise_linear_nd (1 FFN layer).
    # Eliminates runtime reciprocal + sign normalization per segment.
    if len(segments) > 0:
        angle_data = _build_angle_lookup(ray_angle, segments)

    # Stages 3-4: Per-segment intersection + validity + distance
    distances = []
    colors = []
    for i, seg in enumerate(segments):
        _den, num_t, num_u = _segment_intersection(
            cos_sin, px_py, trig_and_products, seg,
        )
        signed_inv_den, abs_den_node, sign_den = angle_data[i]
        dist = _segment_distance(
            num_t, num_u, signed_inv_den, abs_den_node, sign_den, max_coord,
        )
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
    max_dist = 2.0 * max_coord
    clamped_dist = clamp(closest_dist, 0.5, max_dist)

    perp_dist = signed_multiply(
        clamped_dist, perp_cos,
        max_abs1=max_dist, max_abs2=1.0, step=1.0,
    )
    safe_perp_dist = clamp(perp_dist, 0.5, max_dist)

    inv_perp = reciprocal(safe_perp_dist, min_value=0.5, max_value=max_dist, step=1.0)
    wall_height = multiply_const(inv_perp, float(config.screen_height))

    # Wall bounds: center ± half wall height.
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

    # Stage 7: Column fill
    return _column_fill(wall_top, wall_bottom, wall_color, config)


def build_renderer_graph(
    segments: List[Segment],
    config: RenderConfig,
    max_coord: float = 20.0,
) -> Tuple[Node, "PosEncoding"]:
    """Build the complete per-column rendering graph (Phase 2 standalone).

    Creates InputNodes and calls build_rendering_pipeline.  For Phase 3+,
    use build_rendering_pipeline directly with computed nodes.

    Inputs (per position, alphabetical order for headless module):
        perp_cos:  cos(ray_angle - player_angle) for fish-eye correction
        player_x:  world x coordinate
        player_y:  world y coordinate
        ray_angle: integer 0-255, index into trig table

    Output: H*3 floats — RGB for each row of this screen column.

    Returns:
        (output_node, pos_encoding) tuple for compilation.
    """
    pos_encoding = create_pos_encoding()

    perp_cos = create_input("perp_cos", 1)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    ray_angle = create_input("ray_angle", 1)

    output = build_rendering_pipeline(
        player_x, player_y, ray_angle, perp_cos,
        segments, config, max_coord,
    )

    return output, pos_encoding
