"""Build torchwright graphs that render flat-shaded or textured first-person views.

The graph takes per-position inputs and outputs H*3 RGB values (one screen
column).  Segment geometry is baked into the weights as constants.
"""

from typing import List, Optional, Tuple

import builtins

import numpy as np
import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.scheduling_hints import sequential_scope
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    clamp,
    compare,
    floor_int,
    linear_bin_index,
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
from torchwright.ops.map_select import (
    broadcast_select,
    dynamic_extract,
    in_range,
    map_to_table,
    select,
)
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
# Wall height from distance
# ---------------------------------------------------------------------------


def _wall_height_lookup(
    closest_dist: Node,
    perp_cos: Node,
    config: RenderConfig,
    max_coord: float,
) -> Tuple[Node, Node, Node]:
    """Compute wall_top, wall_bottom, wall_height from the closest hit.

    The math is ``wall_height = H / (closest_dist * perp_cos)``,
    clamped so the no-hit case (``closest_dist = BIG_DISTANCE``)
    degenerates to a tiny wall instead of extrapolating wildly.

    Earlier versions used a single ``piecewise_linear_2d`` over
    ``(closest_dist, perp_cos)`` with sparse breakpoints, collapsing
    the whole "multiply, clamp, reciprocal, scale" pipeline into one
    MLP sublayer.  But the function being approximated is ``H/(d*c)``,
    which has steep gradients near small ``d*c``, and the bilinear
    interpolant on the sparse grid (~20 × 8 vertices) gave a few
    percent error in ``wall_height`` for typical wall hits — enough
    to push the wall band 1-2 screen rows off the reference's
    location and visibly thin out the wall in the rendered output.

    The reformulation here is the explicit decomposition:

    1. ``perp_dist = signed_multiply(closest_dist, perp_cos)``
       — both inputs are positive scalars, output bounded above
       by ``max_dist`` and below by ``min_dist`` after clamping.
    2. ``inv_perp_dist = 1/perp_dist`` via ``piecewise_linear`` with
       **geometric** breakpoints (constant relative error per
       segment, ~1% across the whole range with 48 breakpoints).
    3. ``wall_height = H * inv_perp_dist`` — free ``Linear``.
    4. ``wall_top``/``wall_bottom`` — free ``Linear`` from
       ``wall_height`` (vertically centred wall).

    Cost: ~5 MLP sublayers (signed_multiply ~3 + clamp 1 + reciprocal
    1) instead of 1 — but called once per screen column, so the
    total graph depth grows by ~4 sublayers, not by a factor.  See
    the matching note on :func:`_u_norm_lookup` for the same reasoning.

    Args:
        closest_dist: Scalar distance to nearest wall.
        perp_cos: Fish-eye correction cosine (scalar) in roughly
            ``[0.65, 1.0]``.
        config: Render configuration.
        max_coord: Upper bound on coordinate magnitudes.

    Returns:
        (wall_top, wall_bottom, wall_height) scalar nodes.
    """
    H = config.screen_height
    max_dist = 2.0 * max_coord
    min_dist = 0.5

    # 1. perp_dist = closest_dist * perp_cos.  Both positive but use
    #    signed_multiply since we don't have an unsigned version with
    #    matching API.  Output bounded by max_dist (closest_dist is
    #    already roughly ≤ max_dist for hits, BIG_DISTANCE for no-hit;
    #    perp_cos ≤ 1.0 always).
    perp_dist = signed_multiply(
        closest_dist,
        perp_cos,
        max_abs1=BIG_DISTANCE,
        max_abs2=1.0,
        max_abs_output=BIG_DISTANCE,
        step=0.5,
    )

    # 2. Clamp perp_dist to [min_dist, max_dist] so the reciprocal
    #    lookup stays in its valid range and the no-hit case
    #    (closest_dist == BIG_DISTANCE) collapses to wall_height =
    #    H / max_dist (a small wall, not 0).
    clamped_perp_dist = clamp(perp_dist, min_dist, max_dist)

    # 3. inv_perp_dist via geometric-breakpoint piecewise_linear.
    #    Geometric spacing gives constant relative error per segment;
    #    48 breakpoints over [0.5, 40] gives ~1% relative error on
    #    the reciprocal across the whole range.
    n_bp = 48
    ratio = (max_dist / min_dist) ** (1.0 / (n_bp - 1))
    bps: List[float] = [min_dist * (ratio ** k) for k in range(n_bp)]
    bps[0] = min_dist
    bps[-1] = max_dist
    inv_perp_dist = piecewise_linear(
        clamped_perp_dist,
        bps,
        lambda d: 1.0 / d,
        name="wall_height_inv_perp_dist",
    )

    # 4. wall_height = H * inv_perp_dist.  Free Linear.
    wall_height = multiply_const(inv_perp_dist, float(H))

    # 5. wall_top, wall_bottom from wall_height (free Linears).  The
    #    wall is always vertically centred at H/2 — see the note in
    #    _textured_column_fill for why this assumption is currently
    #    baked in upstream.
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
    """Compute the integer texture column index ``floor(num_u * tex_w / abs_den)``.

    Earlier versions of this op used :func:`piecewise_linear_2d` over
    ``(num_u, abs_den)`` with sparse breakpoints, which collapses the
    whole "divide and floor" pipeline into a single MLP sublayer.  But
    the function being approximated is a ``tex_w``-step staircase
    over a 2D domain, and bilinear interpolation between sparse grid
    vertices cannot resolve every step — for typical wall hits the
    output landed 1–2 columns off the true value, which manifested
    visually as a consistent texture-mapping offset across the
    rendered wall (the "wrong starting point" the demo gif showed).

    The reformulation uses :func:`linear_bin_index`, which is the
    purpose-built primitive for "continuous coordinate → integer bin
    over a runtime range":

    * ``num_u`` is the per-segment u-coordinate before normalisation.
    * ``abs_den`` is the magnitude of the ray-segment denominator
      (the per-angle scale that ``num_u`` should be divided by).
    * Bin range is ``[0, abs_den]``, divided into ``tex_w`` bins.
    * Output is ``floor((num_u - 0) * tex_w / (abs_den - 0))``
      clamped to ``[0, tex_w - 1]`` — exactly the texture column index.

    Cost: ~5 MLP sublayers (1 reciprocal + ~3 signed_multiply + 1
    floor_int) instead of 1 — but called once per screen column, so
    the total graph depth grows by ~4 sublayers, not by a factor.

    Args:
        winning_adj_num_u: Scalar node — the winning segment's u
            numerator (sign-normalised so it's non-negative).
        winning_abs_den: Scalar node — the winning segment's
            ``|den|`` value (always positive, bounded above by the
            wall length × ray-direction extreme).
        tex_w: Texture width in pixels (compile-time).
        max_coord: World coordinate bound — used to set the
            reciprocal lookup's range.

    Returns:
        Scalar node carrying an integer in ``[0, tex_w - 1]``.
    """
    from torchwright.graph.misc import LiteralValue

    zero = LiteralValue(torch.tensor([0.0]), name="u_norm_zero")
    return linear_bin_index(
        winning_adj_num_u,
        zero,
        winning_abs_den,
        n_bins=tex_w,
        # |abs_den| for box-room geometry sits in roughly
        # [0.5, 2 * max_coord]: smaller than 0.5 means an extremely
        # glancing ray (those are unlikely to be the winning segment
        # via reduce_min anyway), and the upper bound is the longest
        # wall edge.  Tight bounds → better signed_multiply precision
        # AND fewer neurons.
        min_range=0.5,
        max_range=2.0 * max_coord,
        n_reciprocal_breakpoints=48,
        mul_step=0.25,
        name="u_norm_bin",
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
    max_coord: float = 20.0,
    patch_row_start: Optional[Node] = None,
    rows_per_patch: Optional[int] = None,
    tex_sample_batch_size: int = 8,
) -> Node:
    """Fill a screen column with textured wall and solid floor/ceiling.

    For each screen row ``y`` in the patch, compute which texture row
    it samples — ``tex_row(y) = floor((y - wall_top) /
    (wall_bottom - wall_top) * tex_height)`` — then look up the
    corresponding RGB from ``tex_column_colors``.  This mirrors the
    reference ``render_column`` exactly.

    This replaces an earlier band-sum-over-tex-rows formulation whose
    per-band ``in_range`` masks underflowed or overlapped whenever
    ``wall_height < tex_height`` (far walls, big textures), producing
    catastrophic pixel values because many bands "won" the same screen
    row and their RGB contributions summed.  The per-screen-row
    formulation is O(rows_per_patch) in depth and correct at any
    ``wall_height / tex_height`` ratio.

    The heavy lifting is split between two primitives:

    * :func:`torchwright.ops.linear_bin_index` takes the runtime
      ``(y, wall_top, wall_bottom)`` tuple and returns an integer
      texture row index in ``[0, tex_height - 1]``.
    * :func:`torchwright.ops.dynamic_extract` reads the 3-wide RGB
      slice from ``tex_column_colors`` at that index.

    Both are unit-tested in ``tests/ops/test_resampling_primitives.py``
    — correctness of this fill reduces to correctness of those two
    ops plus the outer composite mask.

    Args:
        wall_top: Scalar node — top of wall in full-frame screen rows.
        wall_bottom: Scalar node — bottom of wall in full-frame rows.
        wall_height: Scalar node — ``wall_bottom - wall_top``.  Present
            for API compatibility with callers; the fill derives range
            internally via ``wall_top`` and ``wall_bottom``.
        tex_column_colors: Node of width ``tex_height * 3`` — RGB for
            each texture row in this column.
        tex_height: Number of texture rows (compile-time).
        config: Render configuration.
        max_coord: World-coordinate magnitude bound, used upstream by
            :func:`_wall_height_lookup` to set the ``wall_height``
            range.  The fill uses it to size the reciprocal lookup
            inside ``linear_bin_index``.
        patch_row_start: Scalar node — the y-offset (in full-frame
            coordinates) of the first row in the current patch.
            Defaults to 0 for unsharded renders.
        rows_per_patch: Height of the patch.  Defaults to the full
            screen height.

    Returns:
        Node of width ``rows_per_patch * 3`` — one RGB per screen row
        in the patch.
    """
    H = config.screen_height
    if rows_per_patch is None:
        rows_per_patch = H
    if patch_row_start is None:
        patch_row_start = LiteralValue(torch.tensor([0.0]), name="patch_row_start_0")

    # --- Base (non-wall) column: floor below wall_bottom, ceiling above ---
    with annotate("base"):
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
        floor_masks = in_range(wall_bottom_local, h_local, rows_per_patch)
        base = broadcast_select(
            floor_masks, floor_node, ceil_node, rows_per_patch, 3,
        )

    # --- Textured wall: per-screen-row texture sampling ---
    #
    # Each screen row y needs: floor((y - wall_top) * tex_height / wall_height)
    #
    # Expanding y = patch_row_start + y_idx + 0.5:
    #
    #   bin_f = (patch_row_start - wall_top) * tex_height * inv_range
    #         + (y_idx + 0.5)               * tex_height * inv_range
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #           base_product (SHARED multiply)  per-row offset (FREE Linear)
    #
    # This factors the expensive signed_multiply out of the per-row loop:
    # one shared multiply, then each row is a free Linear + clamp + floor
    # + dynamic_extract (4 MLP sublayers instead of 8).
    max_dist = 2.0 * max_coord
    min_wall_height = max(float(H) / max_dist, 0.5)
    max_wall_height = 2.0 * float(H)

    # Shared: inv_range and per_row_step
    range_ = subtract(wall_bottom, wall_top)
    clamped_range = clamp(range_, min_wall_height, max_wall_height)
    inv_range = reciprocal(
        clamped_range, min_value=min_wall_height, max_value=max_wall_height,
    )
    per_row_step = multiply_const(inv_range, float(tex_height))

    # Shared: base_product = (patch_row_start - wall_top) * per_row_step
    base_delta = subtract(patch_row_start, wall_top)
    max_abs_base_delta = max_wall_height
    clamped_base_delta = clamp(base_delta, -max_abs_base_delta, max_abs_base_delta)
    max_abs_step = float(tex_height) / min_wall_height
    base_product = signed_multiply(
        clamped_base_delta,
        per_row_step,
        max_abs1=max_abs_base_delta,
        max_abs2=max_abs_step,
        step=0.5,
    )

    def _build_tex_row(y_idx: int) -> Node:
        with annotate("tex_sample"):
            # bin_f = base_product + (y_idx + 0.5) * per_row_step
            # This is a free Linear: 1 × base_product + (y_idx+0.5) × per_row_step
            bin_f = Linear(
                Concatenate([base_product, per_row_step]),
                torch.tensor([[1.0], [float(y_idx) + 0.5]]),
                name=f"bin_f_row_{y_idx}",
            )
            clamped_bin_f = clamp(bin_f, 0.0, float(tex_height) - 0.5)
            tex_row_idx = floor_int(
                clamped_bin_f, min_value=0, max_value=tex_height - 1,
            )
            row_rgb = dynamic_extract(
                tex_column_colors, tex_row_idx, tex_height, 3,
            )
            return row_rgb

    # Each iteration pins ~192 cols (via the masked intermediate inside
    # dynamic_extract).  Without gating, the scheduler admits ~7 rows
    # concurrently and stalls on a residual-pressure plateau (see
    # optimization_guide §7).  ``sequential_scope`` wires scheduling
    # deps so at most ``tex_sample_batch_size`` rows are in flight at
    # once, keeping peak residual usage within budget.
    row_rgbs: List[Node] = sequential_scope(
        [lambda y_idx=y_idx: _build_tex_row(y_idx)
         for y_idx in range(rows_per_patch)],
        batch_size=tex_sample_batch_size,
    )

    textured_wall = Concatenate(row_rgbs)

    # --- Composite: wall region gets textured_wall, rest keeps base ---
    with annotate("composite"):
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
    tex_sample_batch_size: int = 8,
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
    #
    # Use :func:`piecewise_linear` with vector output over a flat
    # ``(texture_id * tex_w + col)`` key space.  The previous
    # implementation used :func:`map_to_table` with two-dimensional
    # ``(texture_id, column_index)`` keys, which was catastrophically
    # wrong for non-trivial textures: ``map_to_table``'s linear
    # dot-product scoring fires on every key whose
    # dot-product-with-input exceeds ``key @ key - 1`` — so for input
    # ``(0, 7)`` and keys ``(0, 0), (0, 1), ... (0, 7)`` every unit
    # fires with varying magnitudes and the Linear sum becomes a
    # weighted combination of several texture columns instead of just
    # one (surfaced as pixel values in ``[0.37, 1.97]`` instead of
    # the expected ``[0, 1]``).
    #
    # ``piecewise_linear`` with integer breakpoints and vector output
    # is the right shape: at any integer ``flat_key`` it returns the
    # exact corresponding texture column's RGB, and it costs one MLP
    # sublayer with roughly one hidden neuron per breakpoint
    # regardless of the output width.  For non-integer ``flat_key``
    # (possible under upstream arithmetic wiggle) it linearly
    # interpolates between the two adjacent columns — far better than
    # ``map_to_table``'s sum-everything behaviour.
    num_tex = len(textures)
    n_keys = num_tex * tex_w

    flat_key = add(
        multiply_const(winning_tex_id, float(tex_w)),
        tex_col_idx,
    )

    def _tex_column_values(flat_idx: float) -> List[float]:
        """Vector-valued lookup: ``(tid, col)`` → flattened
        ``tex_h * 3``-wide RGB for that texture column."""
        k = int(round(flat_idx))
        if 0 <= k < n_keys:
            tid = k // tex_w
            col = k % tex_w
            return [float(v) for v in textures[tid][col].flatten()]
        return [0.0] * (tex_h * 3)

    tex_column_colors = piecewise_linear(
        flat_key,
        breakpoints=[float(k) for k in range(n_keys)],
        fn=_tex_column_values,
        name="texture_column_lookup",
    )

    # --- Stage 9: Textured column fill ---
    return _textured_column_fill(
        wall_top,
        wall_bottom,
        wall_height,
        tex_column_colors,
        tex_h,
        config,
        max_coord=max_coord,
        patch_row_start=patch_row_start,
        rows_per_patch=rows_per_patch,
        tex_sample_batch_size=tex_sample_batch_size,
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
