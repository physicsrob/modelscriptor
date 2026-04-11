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

    One forward pass computes one (col, patch_idx_in_col) patch of one
    frame.  The graph is **fully position-independent**: every input row
    carries the state the graph needs to render that position, and the
    graph emits the "next" state for the host to thread back in on the
    following step.  There is no cross-position attention, no
    ``get_prev_value`` seed broadcast, and no consultation of
    ``position_scalar``.

    Inputs (per position, alphabetical order):

        cur_col_idx:         screen column being rendered this step
        cur_patch_idx_in_col: patch index within the column this step
        input_backward:      0.0 or 1.0      (real at step 0; 0 elsewhere)
        input_forward:       0.0 or 1.0      (real at step 0; 0 elsewhere)
        input_strafe_left:   0.0 or 1.0      (real at step 0; 0 elsewhere)
        input_strafe_right:  0.0 or 1.0      (real at step 0; 0 elsewhere)
        input_turn_left:     0.0 or 1.0      (real at step 0; 0 elsewhere)
        input_turn_right:    0.0 or 1.0      (real at step 0; 0 elsewhere)
        seed_angle:          current player angle (the seed at step 0; the
                             carried-forward post-update value after that)
        seed_x:              current player x (likewise)
        seed_y:              current player y (likewise)

    Game logic runs at every step on ``(seed_{x,y,angle}, input_*)``.  At
    step 0 the inputs are real player inputs and the game-logic update
    fires for real.  At steps ≥ 1 the inputs are all zero and the
    game-logic chain degenerates to a passthrough: ``new_angle = old``,
    ``dx = dy = 0``, ``resolved_x = seed_x``, ``resolved_y = seed_y``.
    So the host just has to carry ``(new_x, new_y, new_angle)`` forward
    from step 0's output.

    Output per position (see Concatenate at the bottom of the function):

        H*3 pixel values         -- the patch rendered from resolved_{x,y,angle}
        next_col_idx             -- col for step t+1 (feedback)
        next_patch_idx_in_col    -- patch for step t+1 (feedback)
        resolved_x, resolved_y   -- post-update player position (feedback)
        new_angle                -- post-update player angle (feedback)

    Returns:
        (output_node, pos_encoding) tuple for compilation.
    """
    pos_encoding = create_pos_encoding()

    # Create inputs (alphabetical order — matches HeadlessTransformerModule)
    cur_col_idx = create_input("cur_col_idx", 1)
    cur_patch_idx_in_col = create_input("cur_patch_idx_in_col", 1)
    input_backward = create_input("input_backward", 1)
    input_forward = create_input("input_forward", 1)
    input_strafe_left = create_input("input_strafe_left", 1)
    input_strafe_right = create_input("input_strafe_right", 1)
    input_turn_left = create_input("input_turn_left", 1)
    input_turn_right = create_input("input_turn_right", 1)
    seed_angle = create_input("seed_angle", 1)
    seed_x = create_input("seed_x", 1)
    seed_y = create_input("seed_y", 1)

    # === Sharding layout ===
    W = config.screen_width
    H = config.screen_height
    fov = config.fov_columns
    rp = rows_per_patch if rows_per_patch is not None else H
    assert H % rp == 0, (
        f"screen_height {H} must be divisible by rows_per_patch {rp}"
    )
    shards_per_col = H // rp

    # === Autoregressive delta: predict next (col, patch) for feedback ===
    #
    #     next_patch = (cur_patch + 1) mod shards_per_col
    #     wrap       = floor_div(cur_patch + 1, shards_per_col)   {0, 1}
    #     next_col   = cur_col + wrap
    #
    # The host writes (cur_col=0, cur_patch=0) at step 0 and threads this
    # step's (next_col, next_patch) into the next step's (cur_col,
    # cur_patch) slots.  No position-0 special case — the graph treats
    # every step uniformly.
    patch_plus_one = add_const(cur_patch_idx_in_col, 1.0)
    next_patch_idx_in_col = mod_const(
        patch_plus_one, shards_per_col, max_value=shards_per_col,
    )
    wrap = thermometer_floor_div(
        patch_plus_one, shards_per_col, max_value=shards_per_col,
    )
    next_col_idx = add(cur_col_idx, wrap)

    # This step's patch_row_start feeds the renderer (not the feedback
    # output — which uses next_patch_idx_in_col as its raw index).
    cur_patch_row_start = multiply_const(cur_patch_idx_in_col, float(rp))

    # Integer-exact: matches the host-side
    #     angle_offset = (col - W//2) * fov_columns // W
    # when (W//2)*fov_columns % W == 0 (satisfied by the test fixtures).
    col_times_fov = multiply_const(cur_col_idx, float(fov))
    ao_unsigned = thermometer_floor_div(
        col_times_fov, W, max_value=fov * W,
    )
    angle_offset = add_const(ao_unsigned, float(-(fov // 2)))

    # perp_cos = cos(angle_offset mod 256), reusing the existing trig lookup.
    perp_shifted = add_const(angle_offset, 256.0)
    perp_angle = mod_const(perp_shifted, 256, max_value=256 + fov)
    perp_cos, _perp_sin = trig_lookup(perp_angle)

    # === Game Logic Phase ===
    #
    # Runs at every step on the current-state inputs.  At step 0 the
    # player inputs are real and the update fires; at steps ≥ 1 they are
    # all zero and every op degenerates to a passthrough, so
    # resolved_{x,y,angle} simply equals the carried-forward seed_*.

    # 1. Angle update
    new_angle = _compute_new_angle(
        seed_angle,
        input_turn_left,
        input_turn_right,
        turn_speed,
    )

    # 2. Velocity from inputs
    dx, dy = _compute_velocity(
        new_angle,
        input_forward,
        input_backward,
        input_strafe_left,
        input_strafe_right,
        move_speed,
    )

    # 3. Collision detection + position resolution
    resolved_x, resolved_y = _resolve_collision_graph(
        seed_x, seed_y, dx, dy, segments, max_coord, move_speed,
    )

    # === Rendering Phase ===

    # 4. Per-column ray angle: (new_angle + angle_offset) mod 256
    ray_angle = _compute_ray_angle(new_angle, angle_offset)

    # 5. Render using resolved position
    if textures is not None:
        pixels = build_textured_rendering_pipeline(
            resolved_x, resolved_y, ray_angle, perp_cos,
            segments, config, textures, max_coord,
            patch_row_start=cur_patch_row_start,
            rows_per_patch=rp,
        )
    else:
        pixels = build_rendering_pipeline(
            resolved_x, resolved_y, ray_angle, perp_cos,
            segments, config, max_coord,
            patch_row_start=cur_patch_row_start,
            rows_per_patch=rp,
        )

    # 6. Output: pixels + next (col, patch) feedback + carried state.
    # The host pastes at the input-side (cur_col, cur_patch * rp) — which
    # it already knows, since it just wrote them into the input row.
    output = Concatenate([
        pixels,
        next_col_idx,
        next_patch_idx_in_col,
        resolved_x,
        resolved_y,
        new_angle,
    ])

    return output, pos_encoding
