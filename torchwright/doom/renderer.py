"""Build torchwright graphs that render flat-shaded or textured first-person views.

The graph takes per-position inputs and outputs H*3 RGB values (one screen
column).  Segment geometry is baked into the weights as constants.
"""

from typing import List, Optional, Tuple

import builtins

import numpy as np
import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    clamp,
    compare,
    multiply_const,
    negate,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    reduce_min,
    signed_multiply,
    subtract,
    sum_nodes,
    thermometer_floor_div,
)
from torchwright.ops.logic_ops import bool_all_true
from torchwright.ops.map_select import broadcast_select, in_range, map_to_table, select
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

    ray_cos = piecewise_linear(
        ray_angle, breakpoints, lambda i: float(table[int(i), 0]), name="trig_cos"
    )
    ray_sin = piecewise_linear(
        ray_angle, breakpoints, lambda i: float(table[int(i), 1]), name="trig_sin"
    )
    return ray_cos, ray_sin


# ---------------------------------------------------------------------------
# Wall height from distance (2D lookup)
# ---------------------------------------------------------------------------

# Distance breakpoints: dense near 0 where 1/x is steep, sparse far away.
_DIST_BREAKPOINTS = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    13.0,
    16.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
]

# perp_cos breakpoints: covers typical FOV range.
_COS_BREAKPOINTS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def _wall_height_lookup(
    closest_dist: Node,
    perp_cos: Node,
    config: RenderConfig,
    max_coord: float,
) -> Tuple[Node, Node, Node]:
    """Compute wall_top, wall_bottom, wall_height via a single 2D lookup.

    Replaces the 6-MLP chain (clamp → signed_multiply → clamp →
    reciprocal → multiply_const → Linear) with 1 MLP sublayer.

    Args:
        closest_dist: Scalar distance to nearest wall.
        perp_cos: Fish-eye correction cosine (scalar).
        config: Render configuration.
        max_coord: Upper bound on coordinate magnitudes.

    Returns:
        (wall_top, wall_bottom, wall_height) scalar nodes.
    """
    H = config.screen_height
    max_dist = 2.0 * max_coord

    # Clamp distance before the 2D lookup so BIG_DISTANCE (no-hit)
    # maps to a tiny wall height instead of extrapolating wildly.
    clamped_dist = clamp(closest_dist, 0.5, max_dist)

    def _wall_h(d, c):
        d_safe = builtins.max(0.5, builtins.min(max_dist, d))
        perp = builtins.max(0.5, builtins.min(max_dist, d_safe * c))
        return float(H) / perp

    wall_height = piecewise_linear_2d(
        clamped_dist,
        perp_cos,
        _DIST_BREAKPOINTS,
        _COS_BREAKPOINTS,
        _wall_h,
        name="wall_height_2d",
    )

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
    return wall_top, wall_bottom, wall_height


# ---------------------------------------------------------------------------
# Shared products (2D lookup)
# ---------------------------------------------------------------------------

# Position breakpoints: dense near 0, sparser at extremes.
_POS_BREAKPOINTS = [
    -20.0,
    -15.0,
    -10.0,
    -7.0,
    -5.0,
    -3.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
]

# Trig value breakpoints: full [-1, 1] range.
_TRIG_BREAKPOINTS = [
    -1.0,
    -0.9,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
]


def _shared_products(
    player_x: Node,
    player_y: Node,
    ray_cos: Node,
    ray_sin: Node,
) -> Tuple[Node, Node]:
    """Compute px*sin(θ) and py*cos(θ) via 2D lookups (1 MLP sublayer each).

    Replaces two ``signed_multiply`` calls (~3 MLP sublayers each) with two
    ``piecewise_linear_2d`` lookups (1 MLP sublayer each, parallel).
    """
    px_sin = piecewise_linear_2d(
        player_x,
        ray_sin,
        _POS_BREAKPOINTS,
        _TRIG_BREAKPOINTS,
        lambda x, s: x * s,
        name="px_sin_2d",
    )
    py_cos = piecewise_linear_2d(
        player_y,
        ray_cos,
        _POS_BREAKPOINTS,
        _TRIG_BREAKPOINTS,
        lambda y, c: y * c,
        name="py_cos_2d",
    )
    return px_sin, py_cos


# ---------------------------------------------------------------------------
# u-normalization (2D lookup)
# ---------------------------------------------------------------------------

# adj_num_u breakpoints: ranges from 0 to ~max_coord (sign-normalized).
_U_BREAKPOINTS = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
    30.0,
    40.0,
]

# abs_den breakpoints: from small (glancing angles) to large.
_DEN_BREAKPOINTS = [
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    15.0,
    25.0,
    40.0,
]


def _u_norm_lookup(
    winning_adj_num_u: Node,
    winning_abs_den: Node,
    tex_w: int,
    max_coord: float,
) -> Node:
    """Compute texture column index via a single 2D lookup.

    Replaces the 7-MLP chain (clamp → reciprocal → signed_multiply →
    clamp → multiply_const → thermometer_floor_div) with 1 MLP sublayer.

    Returns a scalar node containing floor(clamp(u/den, 0, 1) * tex_w),
    i.e. an integer in [0, tex_w-1].
    """

    def _tex_col(u, d):
        d_safe = builtins.max(0.01, builtins.min(2.0 * max_coord, d))
        u_norm = builtins.max(0.0, builtins.min(1.0 - 1e-4, u / d_safe))
        return builtins.min(tex_w - 1, builtins.max(0, int(u_norm * tex_w)))

    return piecewise_linear_2d(
        winning_adj_num_u,
        winning_abs_den,
        _U_BREAKPOINTS,
        _DEN_BREAKPOINTS,
        lambda u, d: float(_tex_col(u, d)),
        name="tex_col_2d",
    )


# ---------------------------------------------------------------------------
# Stage 3: Per-segment intersection (Linear nodes — zero MLP cost)
# ---------------------------------------------------------------------------


def _segment_intersection(
    cos_sin: Node,
    px_py: Node,
    trig_and_products: Node,
    seg: Segment,
) -> Tuple[Node, Node, Node]:
    """Compute (den, num_t, num_u) for one segment.

    All outputs are scalar Linear nodes — zero MLP sublayers.

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
    num_t = Linear(
        px_py, torch.tensor([[-ey], [ex]]), torch.tensor([const_t]), name="num_t"
    )

    # num_u = (ax*sin - ay*cos) + (py_cos - px_sin)
    num_u = Linear(
        trig_and_products,
        torch.tensor([[-seg.ay], [seg.ax], [-1.0], [1.0]]),
        name="num_u",
    )

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
        adj_num_t,
        abs_inv_den,
        max_abs1=max_num_t,
        max_abs2=max_inv_den,
        step=1.0,
        max_abs_output=BIG_DISTANCE,
    )

    # Mask invalid intersections
    is_valid = bool_all_true([is_den_nonzero, is_t_pos, is_u_ge_0, is_u_le_den])
    big = LiteralValue(torch.tensor([BIG_DISTANCE]), name="big_dist")
    dist = select(is_valid, dist, big)

    return dist


def _segment_distance_and_texinfo(
    num_t: Node,
    num_u: Node,
    signed_inv_den: Node,
    abs_den: Node,
    sign_den: Node,
    max_coord: float,
    texture_id: int,
) -> Tuple[Node, Node]:
    """Like _segment_distance but also returns texture metadata.

    Returns ``(dist, tex_meta)`` where tex_meta is a width-3 node
    containing ``[texture_id, adj_num_u, abs_den]`` (masked to defaults
    when the intersection is invalid).
    """
    max_num_t = 2.0 * max_coord * max_coord
    max_inv_den = 100.0
    epsilon = 0.05

    adj_num_t = select(sign_den, num_t, negate(num_t))
    adj_num_u = select(sign_den, num_u, negate(num_u))

    is_den_nonzero = compare(abs_den, epsilon)
    is_t_pos = compare(adj_num_t, epsilon)
    is_u_ge_0 = compare(adj_num_u, -epsilon)
    u_minus_den = subtract(abs_den, adj_num_u)
    is_u_le_den = compare(u_minus_den, -epsilon)

    abs_inv_den = select(sign_den, signed_inv_den, negate(signed_inv_den))
    dist = signed_multiply(
        adj_num_t,
        abs_inv_den,
        max_abs1=max_num_t,
        max_abs2=max_inv_den,
        step=1.0,
        max_abs_output=BIG_DISTANCE,
    )

    is_valid = bool_all_true([is_den_nonzero, is_t_pos, is_u_ge_0, is_u_le_den])
    big = LiteralValue(torch.tensor([BIG_DISTANCE]), name="big_dist")
    dist = select(is_valid, dist, big)

    # Texture metadata: carry through reduce_min
    tex_id_node = LiteralValue(
        torch.tensor([float(texture_id)]),
        name=f"tex_id_{texture_id}",
    )
    valid_meta = Concatenate([tex_id_node, adj_num_u, abs_den])
    invalid_meta = LiteralValue(torch.tensor([0.0, 0.0, 1.0]), name="invalid_meta")
    tex_meta = select(is_valid, valid_meta, invalid_meta)

    return dist, tex_meta


def _build_angle_lookup(
    ray_angle: Node,
    segments: List[Segment],
) -> List[Tuple[Node, Node, Node]]:
    """Precompute per-segment den-related values as a function of ray angle.

    For each segment, den = cos(angle)*ey - sin(angle)*ex is fully determined
    by the ray angle and constant segment endpoints. Precomputing 1/den, |den|,
    and sign(den) for all 256 angles eliminates runtime reciprocal and sign
    normalization (saves ~5 MLP sublayers on the per-segment critical path).

    Returns a list of (signed_inv_den, abs_den, sign_den) node tuples,
    one per segment.  All segments share a single piecewise_linear lookup
    (1 MLP sublayer total, regardless of segment count).
    """
    trig_table = generate_trig_table()
    n_segs = len(segments)
    epsilon = 0.01  # threshold for "den is zero"

    # Precompute the 3 values per segment per angle
    # Output layout: [signed_inv_den_0, abs_den_0, sign_den_0,
    #                 signed_inv_den_1, abs_den_1, sign_den_1, ...]
    d_out = 3 * n_segs
    breakpoints = list(range(256))

    def _angle_row(angle_idx):
        cos_a = float(trig_table[int(angle_idx), 0])
        sin_a = float(trig_table[int(angle_idx), 1])
        row = []
        for seg in segments:
            ex = seg.bx - seg.ax
            ey = seg.by - seg.ay
            den = cos_a * ey - sin_a * ex
            if builtins.abs(den) < epsilon:
                row.extend([0.0, 0.0, 1.0])
            else:
                row.extend([1.0 / den, builtins.abs(den), 1.0 if den > 0 else -1.0])
        return row

    # Single lookup: 1 MLP sublayer, outputs 3*N values
    all_data = piecewise_linear(
        ray_angle,
        breakpoints,
        _angle_row,
        name="angle_lookup",
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
    patch_row_start: Optional[Node] = None,
    rows_per_patch: Optional[int] = None,
) -> Node:
    """Fill a screen-column patch with ceiling, wall, and floor colors.

    When unsharded (``patch_row_start=0``, ``rows_per_patch=H``) this
    produces an entire column; with a smaller ``rows_per_patch`` it
    produces just the ``rows_per_patch``-row slice starting at
    ``patch_row_start``. Boundaries are shifted into patch-relative
    coordinates via ``subtract``.

    Cost: 4 MLP sublayers (2 in_range + 2 broadcast_select).
    """
    H = config.screen_height
    if rows_per_patch is None:
        rows_per_patch = H
    assert H % rows_per_patch == 0, (
        f"screen_height {H} must be divisible by rows_per_patch {rows_per_patch}"
    )
    if patch_row_start is None:
        patch_row_start = LiteralValue(torch.tensor([0.0]), name="patch_row_start_0")

    ceil_node = LiteralValue(torch.tensor(list(config.ceiling_color)), name="ceiling")
    floor_node = LiteralValue(torch.tensor(list(config.floor_color)), name="floor")

    # Shift wall_top / wall_bottom into patch-relative coordinates.
    wall_top_local = subtract(wall_top, patch_row_start)
    wall_bottom_local = subtract(wall_bottom, patch_row_start)
    # The effective "H" for the base-layer floor test is the row count
    # remaining above the patch, i.e. H - patch_row_start.
    h_local = Linear(
        patch_row_start,
        torch.tensor([[-1.0]]),
        torch.tensor([float(H)]),
        name="h_local_bound",
    )

    # Step 1: Base column — floor below wall_bottom, ceiling above
    floor_masks = in_range(wall_bottom_local, h_local, rows_per_patch)
    base = broadcast_select(floor_masks, floor_node, ceil_node, rows_per_patch, 3)

    # Step 2: Overlay wall color in [wall_top, wall_bottom)
    wall_masks = in_range(wall_top_local, wall_bottom_local, rows_per_patch)
    return broadcast_select(wall_masks, wall_color, base, rows_per_patch, 3)


# ---------------------------------------------------------------------------
# Textured column fill
# ---------------------------------------------------------------------------


def _textured_column_fill(
    wall_top: Node,
    wall_bottom: Node,
    wall_height: Node,
    tex_column_colors: Node,
    tex_height: int,
    config: RenderConfig,
    sum_fanout: int = 4,
    patch_row_start: Optional[Node] = None,
    rows_per_patch: Optional[int] = None,
) -> Node:
    """Fill a screen column with textured wall and solid floor/ceiling.

    Divides the wall into *tex_height* horizontal bands.  Each band gets
    the corresponding texture row color via an ``in_range`` mask +
    ``broadcast_select``, and all bands are summed into the final
    textured wall.

    The per-band fill tensors are each ``H*3`` wide, so at tex_size=64 /
    H=200 a flat sum would hold ``64 * 600 = 38,400`` residual-stream
    columns simultaneously.  ``sum_fanout`` caps how many of those bands
    live on the stream at once by chaining the sum through a running
    accumulator inside :func:`sum_nodes`; this trades a few extra
    Linears for a dramatically lower peak width.

    Args:
        wall_top: Scalar node -- top of wall in screen rows.
        wall_bottom: Scalar node -- bottom of wall in screen rows.
        wall_height: Scalar node -- wall_bottom - wall_top.
        tex_column_colors: Node of width ``tex_height * 3`` -- RGB for
            each texture row in this column.
        tex_height: Number of texture rows.
        config: Render configuration.
        sum_fanout: Max operands live per reduction step in the final
            band sum.  ``4`` keeps peak per-column at ~``sum_fanout *
            H*3`` plus the running accumulator and is a sweet spot for
            tex_size 8..64 at H in [80, 200].

    Returns:
        Node of width H*3 (one RGB per screen row).
    """
    H = config.screen_height
    if rows_per_patch is None:
        rows_per_patch = H
    assert H % rows_per_patch == 0, (
        f"screen_height {H} must be divisible by rows_per_patch {rows_per_patch}"
    )
    if patch_row_start is None:
        patch_row_start = LiteralValue(torch.tensor([0.0]), name="patch_row_start_0")

    ceil_node = LiteralValue(torch.tensor(list(config.ceiling_color)), name="ceiling")
    floor_node = LiteralValue(torch.tensor(list(config.floor_color)), name="floor")

    wall_top_local = subtract(wall_top, patch_row_start)
    wall_bottom_local = subtract(wall_bottom, patch_row_start)
    h_local = Linear(
        patch_row_start,
        torch.tensor([[-1.0]]),
        torch.tensor([float(H)]),
        name="h_local_bound",
    )

    # Base column: floor below wall_bottom, ceiling above
    floor_masks = in_range(wall_bottom_local, h_local, rows_per_patch)
    base = broadcast_select(
        floor_masks, floor_node, ceil_node, rows_per_patch, 3,
    )

    # Band boundaries: boundary_k = wall_top + k * wall_height / tex_height,
    # shifted into patch-relative coordinates via a -1.0 coefficient on
    # patch_row_start. Still free (Linear).
    wt_wh_prs = Concatenate([wall_top, wall_height, patch_row_start])
    zeros_patch = LiteralValue(
        torch.zeros(rows_per_patch * 3), name="zeros_tex"
    )

    band_fills: List[Node] = []
    for k in range(tex_height):
        lo_k = float(k) / tex_height
        hi_k = float(k + 1) / tex_height
        boundary_lo = Linear(
            wt_wh_prs,
            torch.tensor([[1.0], [lo_k], [-1.0]]),
            name=f"band_lo_{k}",
        )
        boundary_hi = Linear(
            wt_wh_prs,
            torch.tensor([[1.0], [hi_k], [-1.0]]),
            name=f"band_hi_{k}",
        )
        band_mask = in_range(boundary_lo, boundary_hi, rows_per_patch)

        extract = torch.zeros(tex_height * 3, 3)
        for c in range(3):
            extract[k * 3 + c, c] = 1.0
        row_color = Linear(tex_column_colors, extract, name=f"tex_row_{k}")

        band_fills.append(
            broadcast_select(band_mask, row_color, zeros_patch, rows_per_patch, 3)
        )

    textured_wall = sum_nodes(band_fills, max_fanout=sum_fanout)

    # Composite: wall region gets textured_wall, rest keeps base
    wall_masks = in_range(wall_top_local, wall_bottom_local, rows_per_patch)
    return broadcast_select(
        wall_masks, textured_wall, base, rows_per_patch, 3,
    )


# ---------------------------------------------------------------------------
# Textured rendering pipeline
# ---------------------------------------------------------------------------


def build_textured_rendering_pipeline(
    player_x: Node,
    player_y: Node,
    ray_angle: Node,
    perp_cos: Node,
    segments: List[Segment],
    config: RenderConfig,
    textures: List[np.ndarray],
    max_coord: float = 20.0,
    patch_row_start: Optional[Node] = None,
    rows_per_patch: Optional[int] = None,
) -> Node:
    """Build a textured rendering pipeline.

    Same structure as :func:`build_rendering_pipeline` but carries
    texture metadata (texture_id, adj_num_u, abs_den) through the
    min-reduction instead of a solid color.  After finding the nearest
    segment, normalises the u coordinate, looks up the texture column,
    and fills the wall with per-row texture colors.

    All textures must have the same dimensions (tex_width x tex_height).

    Args:
        player_x, player_y: World-coordinate nodes (scalar).
        ray_angle: Integer 0-255 angle node (scalar).
        perp_cos: Fish-eye correction node (scalar).
        segments: Wall segments with ``texture_id`` set.
        config: Render configuration.
        textures: List of (tex_width, tex_height, 3) arrays — the atlas.
        max_coord: Upper bound on coordinate magnitudes.

    Returns:
        Output node: H*3 floats (RGB for each row of this screen column).
    """
    tex_w = textures[0].shape[0]
    tex_h = textures[0].shape[1]

    # --- Stages 1-3: identical to solid-color pipeline ---

    ray_cos, ray_sin = trig_lookup(ray_angle)
    px_sin, py_cos = _shared_products(player_x, player_y, ray_cos, ray_sin)

    cos_sin = Concatenate([ray_cos, ray_sin])
    px_py = Concatenate([player_x, player_y])
    trig_and_products = Concatenate([ray_cos, ray_sin, px_sin, py_cos])

    if len(segments) > 0:
        angle_data = _build_angle_lookup(ray_angle, segments)

    # --- Stage 4: Per-segment intersection + texture metadata ---
    distances = []
    tex_metas = []
    for i, seg in enumerate(segments):
        _den, num_t, num_u = _segment_intersection(
            cos_sin,
            px_py,
            trig_and_products,
            seg,
        )
        signed_inv_den, abs_den_node, sign_den = angle_data[i]
        tid = seg.texture_id if seg.texture_id >= 0 else 0
        dist, tex_meta = _segment_distance_and_texinfo(
            num_t,
            num_u,
            signed_inv_den,
            abs_den_node,
            sign_den,
            max_coord,
            tid,
        )
        distances.append(dist)
        tex_metas.append(tex_meta)

    # --- Stage 5: Min-reduction ---
    if len(segments) == 0:
        closest_dist = LiteralValue(torch.tensor([BIG_DISTANCE]), name="no_hit")
        winning_meta = LiteralValue(torch.tensor([0.0, 0.0, 1.0]), name="no_meta")
    elif len(segments) == 1:
        closest_dist = distances[0]
        winning_meta = tex_metas[0]
    else:
        closest_dist, winning_meta = reduce_min(distances, tex_metas)

    # Extract winning metadata
    d_meta = 3
    winning_tex_id = Linear(
        winning_meta,
        _extract_matrix(d_meta, 0),
        name="win_tex_id",
    )
    winning_adj_num_u = Linear(
        winning_meta,
        _extract_matrix(d_meta, 1),
        name="win_adj_num_u",
    )
    winning_abs_den = Linear(
        winning_meta,
        _extract_matrix(d_meta, 2),
        name="win_abs_den",
    )

    # --- Stage 6: Wall height (2D lookup) ---
    wall_top, wall_bottom, wall_height = _wall_height_lookup(
        closest_dist,
        perp_cos,
        config,
        max_coord,
    )

    # --- Stage 7: u normalization (2D lookup) ---
    tex_col_idx = _u_norm_lookup(winning_adj_num_u, winning_abs_den, tex_w, max_coord)

    # --- Stage 8: Texture column lookup ---
    # Build map_to_table keyed by (texture_id, col_idx) → tex_height*3 values
    tex_key_to_value = {}
    for tid, tex in enumerate(textures):
        tw = tex.shape[0]
        for col in range(tw):
            # key = [texture_id, column_index]
            key = torch.tensor([float(tid), float(col)])
            # value = all row colors for this column, flattened
            val = torch.tensor(tex[col].flatten(), dtype=torch.float32)
            tex_key_to_value[key] = val

    default_val = torch.zeros(tex_h * 3)
    tex_lookup_input = Concatenate([winning_tex_id, tex_col_idx])
    tex_column_colors = map_to_table(tex_lookup_input, tex_key_to_value, default_val)

    # --- Stage 9: Textured column fill ---
    return _textured_column_fill(
        wall_top,
        wall_bottom,
        wall_height,
        tex_column_colors,
        tex_h,
        config,
        patch_row_start=patch_row_start,
        rows_per_patch=rows_per_patch,
    )


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
    patch_row_start: Optional[Node] = None,
    rows_per_patch: Optional[int] = None,
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

    # Stage 2: Shared products (2D lookups — 1 MLP sublayer each, parallel)
    px_sin, py_cos = _shared_products(player_x, player_y, ray_cos, ray_sin)

    # Shared Concatenates — created once, reused by all segments
    cos_sin = Concatenate([ray_cos, ray_sin])
    px_py = Concatenate([player_x, player_y])
    trig_and_products = Concatenate([ray_cos, ray_sin, px_sin, py_cos])

    # Per-angle lookup: precompute signed_inv_den, abs_den, sign_den for
    # all segments in a single piecewise_linear (1 MLP sublayer).
    # Eliminates runtime reciprocal + sign normalization per segment.
    if len(segments) > 0:
        angle_data = _build_angle_lookup(ray_angle, segments)

    # Stages 3-4: Per-segment intersection + validity + distance
    distances = []
    colors = []
    for i, seg in enumerate(segments):
        _den, num_t, num_u = _segment_intersection(
            cos_sin,
            px_py,
            trig_and_products,
            seg,
        )
        signed_inv_den, abs_den_node, sign_den = angle_data[i]
        dist = _segment_distance(
            num_t,
            num_u,
            signed_inv_den,
            abs_den_node,
            sign_den,
            max_coord,
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

    # Stage 6: Wall height (2D lookup)
    wall_top, wall_bottom, wall_height = _wall_height_lookup(
        closest_dist,
        perp_cos,
        config,
        max_coord,
    )

    # Stage 7: Column fill
    return _column_fill(
        wall_top,
        wall_bottom,
        wall_color,
        config,
        patch_row_start=patch_row_start,
        rows_per_patch=rows_per_patch,
    )
