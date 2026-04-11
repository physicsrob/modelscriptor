"""Build a torchwright graph that implements game logic + rendering.

One forward pass = one frame of gameplay: process inputs, update state
(movement + collision), then render the scene from the new position.

All positions compute game logic redundantly (same inputs → same result),
then each position renders its own screen column.
"""

from typing import List, Optional, Tuple

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    compare,
    mod_const,
    multiply_const,
    negate,
    signed_multiply,
    subtract,
    thermometer_floor_div,
)
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true, bool_any_true
from torchwright.ops.map_select import select
from torchwright.reference_renderer.types import RenderConfig, Segment

from torchwright.doom.renderer import (
    build_rendering_pipeline,
    build_textured_rendering_pipeline,
    trig_lookup,
)


# ---------------------------------------------------------------------------
# Angle update
# ---------------------------------------------------------------------------


def _compute_new_angle(
    old_angle: Node,
    input_turn_left: Node,
    input_turn_right: Node,
    turn_speed: int,
) -> Node:
    """Compute new facing angle from turn inputs.

    new_angle = (old_angle + turn_right*speed - turn_left*speed) % 256

    Adds 256 before mod to guarantee a non-negative input.
    """
    turn_r = multiply_const(input_turn_right, float(turn_speed))
    turn_l = multiply_const(input_turn_left, float(turn_speed))
    turn_delta = subtract(turn_r, turn_l)
    raw_angle = add(old_angle, turn_delta)
    shifted = add_const(raw_angle, 256.0)
    return mod_const(shifted, 256, 512 + turn_speed)


# ---------------------------------------------------------------------------
# Velocity from inputs
# ---------------------------------------------------------------------------


def _compute_velocity(
    new_angle: Node,
    input_forward: Node,
    input_backward: Node,
    input_strafe_left: Node,
    input_strafe_right: Node,
    move_speed: float,
) -> Tuple[Node, Node]:
    """Compute velocity (dx, dy) from player inputs and facing angle.

    Gates each direction component by its boolean input.
    Forward/backward use (cos, sin), strafe uses (sin, -cos)/(−sin, cos).
    """
    move_cos, move_sin = trig_lookup(new_angle)

    speed_cos = multiply_const(move_cos, move_speed)
    speed_sin = multiply_const(move_sin, move_speed)
    neg_speed_cos = negate(speed_cos)
    neg_speed_sin = negate(speed_sin)

    zero = LiteralValue(torch.tensor([0.0]), name="zero_vel")

    is_fwd = compare(input_forward, 0.5)
    is_bwd = compare(input_backward, 0.5)
    is_sl = compare(input_strafe_left, 0.5)
    is_sr = compare(input_strafe_right, 0.5)

    # dx: fwd→+cos, bwd→−cos, sl→+sin, sr→−sin
    dx = add(
        add(select(is_fwd, speed_cos, zero), select(is_bwd, neg_speed_cos, zero)),
        add(select(is_sl, speed_sin, zero), select(is_sr, neg_speed_sin, zero)),
    )

    # dy: fwd→+sin, bwd→−sin, sl→−cos, sr→+cos
    dy = add(
        add(select(is_fwd, speed_sin, zero), select(is_bwd, neg_speed_sin, zero)),
        add(select(is_sl, neg_speed_cos, zero), select(is_sr, speed_cos, zero)),
    )

    return dx, dy


# ---------------------------------------------------------------------------
# Collision detection (ray-segment intersection)
# ---------------------------------------------------------------------------


def _collision_intersection(
    old_x: Node,
    old_y: Node,
    dx: Node,
    dy: Node,
    oldy_dx: Node,
    oldx_dy: Node,
    segments: List[Segment],
    max_coord: float,
    max_vel: float,
) -> Node:
    """Test whether a movement ray crosses any wall segment.

    The ray goes from (old_x, old_y) with direction (dx, dy).
    Returns a boolean node: 1.0 if any segment is hit, -1.0 otherwise.

    Per-segment (den, num_t, num_u) are all Linear nodes (free).
    Shared products oldy_dx = old_y*dx and oldx_dy = old_x*dy are
    precomputed by the caller.
    """
    if not segments:
        return LiteralValue(torch.tensor([-1.0]), name="no_segs_no_hit")

    epsilon = 0.05

    # Precompute concatenations for per-segment Linear nodes
    dx_dy = Concatenate([dx, dy])
    old_xy = Concatenate([old_x, old_y])

    valid_flags = []
    for seg in segments:
        ex = seg.bx - seg.ax
        ey = seg.by - seg.ay

        # den = dx * ey - dy * ex  (linear in dx, dy)
        den = Linear(dx_dy, torch.tensor([[ey], [-ex]]), name="coll_den")

        # num_t = (ax - old_x)*ey - (ay - old_y)*ex  (linear in old_x, old_y)
        const_t = seg.ax * ey - seg.ay * ex
        num_t = Linear(old_xy, torch.tensor([[-ey], [ex]]),
                       torch.tensor([const_t]), name="coll_num_t")

        # num_u = (ax*dy - ay*dx) + (old_y*dx - old_x*dy)
        # First term is Linear in (dx, dy); second term is oldy_dx - oldx_dy
        # Combined: Linear in Concatenate([dy, dx, oldy_dx, oldx_dy])
        coll_products = Concatenate([dy, dx, oldy_dx, oldx_dy])
        num_u = Linear(
            coll_products,
            torch.tensor([[seg.ax], [-seg.ay], [1.0], [-1.0]]),
            name="coll_num_u",
        )

        # Sign normalization
        sign_den = compare(den, 0.0)
        adj_num_t = select(sign_den, num_t, negate(num_t))
        adj_num_u = select(sign_den, num_u, negate(num_u))
        abs_den = select(sign_den, den, negate(den))

        # Validity: den≠0, t>0, t≤1 (= abs_den - adj_num_t >= 0), u≥0, u≤1
        is_den_ok = compare(abs_den, epsilon)
        is_t_pos = compare(adj_num_t, epsilon)
        t_margin = subtract(abs_den, adj_num_t)
        is_t_le_den = compare(t_margin, -epsilon)
        is_u_ge_0 = compare(adj_num_u, -epsilon)
        u_margin = subtract(abs_den, adj_num_u)
        is_u_le_den = compare(u_margin, -epsilon)

        is_valid = bool_all_true(
            [is_den_ok, is_t_pos, is_t_le_den, is_u_ge_0, is_u_le_den]
        )
        valid_flags.append(is_valid)

    return bool_any_true(valid_flags)


def _resolve_collision_graph(
    old_x: Node,
    old_y: Node,
    dx: Node,
    dy: Node,
    segments: List[Segment],
    max_coord: float,
    move_speed: float,
) -> Tuple[Node, Node]:
    """Resolve movement collision with wall sliding.

    Tests three rays (full, x-only, y-only) and picks the best
    resolved position.
    """
    new_x = add(old_x, dx)
    new_y = add(old_y, dy)

    if not segments:
        return new_x, new_y

    max_vel = move_speed * 2.0  # diagonal move upper bound

    # Shared products (computed once, reused by all three rays)
    oldy_dx = signed_multiply(
        old_y, dx, max_abs1=max_coord, max_abs2=max_vel, step=0.1,
    )
    oldx_dy = signed_multiply(
        old_x, dy, max_abs1=max_coord, max_abs2=max_vel, step=0.1,
    )

    zero = LiteralValue(torch.tensor([0.0]), name="zero_coll")

    # Full ray: direction (dx, dy)
    hits_full = _collision_intersection(
        old_x, old_y, dx, dy, oldy_dx, oldx_dy,
        segments, max_coord, max_vel,
    )

    # X-only ray: direction (dx, 0)
    # oldy_dx reused, oldx_dy becomes old_x * 0 = 0
    hits_x = _collision_intersection(
        old_x, old_y, dx, zero, oldy_dx, zero,
        segments, max_coord, max_vel,
    )

    # Y-only ray: direction (0, dy)
    # oldy_dx becomes old_y * 0 = 0, oldx_dy reused
    hits_y = _collision_intersection(
        old_x, old_y, zero, dy, zero, oldx_dy,
        segments, max_coord, max_vel,
    )

    # Resolution: use new component if full move clear OR axis-only clear
    not_hits_full = negate(hits_full)
    not_hits_x = negate(hits_x)
    not_hits_y = negate(hits_y)

    use_new_x = bool_any_true([not_hits_full, not_hits_x])
    use_new_y = bool_any_true([not_hits_full, not_hits_y])

    resolved_x = select(use_new_x, new_x, old_x)
    resolved_y = select(use_new_y, new_y, old_y)

    return resolved_x, resolved_y


# ---------------------------------------------------------------------------
# Ray angle
# ---------------------------------------------------------------------------


def _compute_ray_angle(new_angle: Node, angle_offset: Node) -> Node:
    """Compute per-column ray angle: (new_angle + angle_offset) % 256."""
    raw = add(new_angle, angle_offset)
    shifted = add_const(raw, 256.0)
    return mod_const(shifted, 256, 512 + 64)  # max: 255 + 256 + fov/2


# ---------------------------------------------------------------------------
# Full game + rendering graph
# ---------------------------------------------------------------------------


def build_game_graph(
    segments: List[Segment],
    config: RenderConfig,
    max_coord: float = 20.0,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    textures=None,
    rows_per_patch: Optional[int] = None,
) -> Tuple[Node, "PosEncoding"]:
    """Build the complete game logic + rendering graph.

    One forward pass processes player inputs, updates position with
    collision detection, and renders the scene.  The host writes seed
    state and player inputs at position 0 only; rows 1..W-1 of the input
    tensor are zero.  The graph broadcasts position 0's inputs across
    all positions via ``get_prev_value`` and derives per-column
    ``angle_offset`` and ``perp_cos`` from the position encoding.

    Inputs (per position, alphabetical order).  Only row 0 is populated
    by the host:

        input_backward:     0.0 or 1.0
        input_forward:      0.0 or 1.0
        input_strafe_left:  0.0 or 1.0
        input_strafe_right: 0.0 or 1.0
        input_turn_left:    0.0 or 1.0
        input_turn_right:   0.0 or 1.0
        seed_angle:         seed facing direction (0-255)
        seed_x:             seed x position
        seed_y:             seed y position

    Output per position: H*3 pixel values + 3 state values
    (new_x, new_y, new_angle).  The state values are identical at every
    position because the broadcast pattern feeds identical inputs into
    the game-logic subgraph everywhere.

    Returns:
        (output_node, pos_encoding) tuple for compilation.
    """
    pos_encoding = create_pos_encoding()

    # Create inputs (alphabetical order — matches HeadlessTransformerModule)
    input_backward = create_input("input_backward", 1)
    input_forward = create_input("input_forward", 1)
    input_strafe_left = create_input("input_strafe_left", 1)
    input_strafe_right = create_input("input_strafe_right", 1)
    input_turn_left = create_input("input_turn_left", 1)
    input_turn_right = create_input("input_turn_right", 1)
    # Per-step feedback: the host threads the previous step's emitted
    # (col_idx, patch_idx_in_col) back in as these inputs so the graph
    # can compute the new values via a local delta instead of
    # back-solving the position from the positional encoding.  At
    # step 0 the values are ignored — is_pos_0 forces the outputs to
    # (0, 0) regardless.
    prev_col_idx = create_input("prev_col_idx", 1)
    prev_patch_idx_in_col = create_input("prev_patch_idx_in_col", 1)
    seed_angle = create_input("seed_angle", 1)
    seed_x = create_input("seed_x", 1)
    seed_y = create_input("seed_y", 1)

    # === Position-0 seed broadcast ===
    # is_pos_0 is +1 at position 0 and -1 elsewhere.  get_position_scalar
    # returns exactly 0.0 at pos 0 (sin(0)=0) and ~k at pos k>0, so the
    # 0.5 threshold is unambiguous at position 0 itself.
    position_scalar = pos_encoding.get_position_scalar()
    is_pos_0 = compare(
        position_scalar, 0.5, true_level=-1.0, false_level=1.0,
    )

    # Broadcast pos-0 values to all positions in a single attention head:
    # pack the 9 scalars into one width-9 node and call get_prev_value
    # once (d_head=16 has room for 9).
    seeds_packed = Concatenate([
        input_backward,
        input_forward,
        input_strafe_left,
        input_strafe_right,
        input_turn_left,
        input_turn_right,
        seed_angle,
        seed_x,
        seed_y,
    ])
    broadcast = pos_encoding.get_prev_value(seeds_packed, is_pos_0)

    def _extract(idx: int, name: str) -> Node:
        m = torch.zeros(9, 1)
        m[idx, 0] = 1.0
        return Linear(broadcast, m, name=name)

    eff_input_backward = _extract(0, "eff_input_backward")
    eff_input_forward = _extract(1, "eff_input_forward")
    eff_input_strafe_left = _extract(2, "eff_input_strafe_left")
    eff_input_strafe_right = _extract(3, "eff_input_strafe_right")
    eff_input_turn_left = _extract(4, "eff_input_turn_left")
    eff_input_turn_right = _extract(5, "eff_input_turn_right")
    eff_seed_angle = _extract(6, "eff_seed_angle")
    eff_seed_x = _extract(7, "eff_seed_x")
    eff_seed_y = _extract(8, "eff_seed_y")

    # === Graph-derived col_idx + patch_row_start (delta from host input) ===
    # Each position renders a single rows_per_patch-tall patch of a single
    # screen column.  Positions iterate raster-order: (col_idx, patch_idx)
    # = divmod(position, shards_per_col).  Rather than back-solve that
    # from a positional-encoding scalar (which is only accurate to ~310
    # positions before the sin approximation drifts), the host threads
    # the previous step's emitted (col_idx, patch_idx_in_col) back in as
    # inputs and the graph computes the new values via a one-step
    # increment with a wrap on the shards boundary:
    #
    #     patch_new = (prev_patch + 1) mod shards_per_col
    #     wrap      = floor_div(prev_patch + 1, shards_per_col)    {0, 1}
    #     col_new   = prev_col + wrap
    #
    # At position 0 the delta would produce garbage (there is no prior
    # step), so we select-override both outputs to 0 using the existing
    # is_pos_0 flag — which is driven by position_scalar but only needs
    # a threshold at 0.5 (position_scalar(0) = 0 exactly, so the compare
    # is trivially safe at any sequence length).
    W = config.screen_width
    H = config.screen_height
    fov = config.fov_columns
    rp = rows_per_patch if rows_per_patch is not None else H
    assert H % rp == 0, (
        f"screen_height {H} must be divisible by rows_per_patch {rp}"
    )
    shards_per_col = H // rp

    patch_plus_one = add_const(prev_patch_idx_in_col, 1.0)
    patch_new_from_delta = mod_const(
        patch_plus_one, shards_per_col, max_value=shards_per_col,
    )
    wrap = thermometer_floor_div(
        patch_plus_one, shards_per_col, max_value=shards_per_col,
    )
    col_new_from_delta = add(prev_col_idx, wrap)

    zero_col_patch = LiteralValue(torch.tensor([0.0]), name="zero_col_patch")
    col_idx = select(is_pos_0, zero_col_patch, col_new_from_delta)
    patch_idx_in_col = select(is_pos_0, zero_col_patch, patch_new_from_delta)
    patch_row_start = multiply_const(patch_idx_in_col, float(rp))

    # Integer-exact: matches the host-side
    #     angle_offset = (col - W//2) * fov_columns // W
    # when (W//2)*fov_columns % W == 0 (satisfied by the test fixtures).
    col_times_fov = multiply_const(col_idx, float(fov))
    ao_unsigned = thermometer_floor_div(
        col_times_fov, W, max_value=fov * W,
    )
    angle_offset = add_const(ao_unsigned, float(-(fov // 2)))

    # perp_cos = cos(angle_offset mod 256), reusing the existing trig lookup.
    perp_shifted = add_const(angle_offset, 256.0)
    perp_angle = mod_const(perp_shifted, 256, max_value=256 + fov)
    perp_cos, _perp_sin = trig_lookup(perp_angle)

    # === Game Logic Phase (consumes effective_* from broadcast) ===

    # 1. Angle update
    new_angle = _compute_new_angle(
        eff_seed_angle,
        eff_input_turn_left,
        eff_input_turn_right,
        turn_speed,
    )

    # 2. Velocity from inputs
    dx, dy = _compute_velocity(
        new_angle,
        eff_input_forward,
        eff_input_backward,
        eff_input_strafe_left,
        eff_input_strafe_right,
        move_speed,
    )

    # 3. Collision detection + position resolution
    resolved_x, resolved_y = _resolve_collision_graph(
        eff_seed_x, eff_seed_y, dx, dy, segments, max_coord, move_speed,
    )

    # === Rendering Phase ===

    # 4. Per-column ray angle: (new_angle + angle_offset) mod 256
    ray_angle = _compute_ray_angle(new_angle, angle_offset)

    # 5. Render using resolved position
    if textures is not None:
        pixels = build_textured_rendering_pipeline(
            resolved_x, resolved_y, ray_angle, perp_cos,
            segments, config, textures, max_coord,
            patch_row_start=patch_row_start,
            rows_per_patch=rp,
        )
    else:
        pixels = build_rendering_pipeline(
            resolved_x, resolved_y, ray_angle, perp_cos,
            segments, config, max_coord,
            patch_row_start=patch_row_start,
            rows_per_patch=rp,
        )

    # 6. Output: pixels + self-identifying destination + updated state.
    # col_idx and patch_row_start travel with the patch so the host is
    # a dumb stitcher that just reads them and indexes the frame buffer.
    output = Concatenate([
        pixels,
        col_idx,
        patch_row_start,
        resolved_x,
        resolved_y,
        new_angle,
    ])

    return output, pos_encoding
