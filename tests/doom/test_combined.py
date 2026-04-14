"""Combined walls-as-tokens graph: sort + parametric render end-to-end.

Proves the full architecture in one test: prefill walls, sort them by
distance, then render columns by attending to the sorted walls with
angular-similarity attention and running the parametric render pipeline.

Token sequence:  START → WALL×N → EOS → SORTED_WALL×N → RENDER×(W×H/rp)

No game logic (collision, movement) in this prototype — player state
passes through from START unchanged.  The focus is: does the sort →
render composition produce correct frames?
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import reference_eval
from torchwright.graph import Attn, Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    add_scaled_nodes,
    bool_to_01,
    compare,
    multiply_const,
    negate,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    signed_multiply,
    square_signed,
    subtract,
)
from torchwright.ops.arithmetic_ops import max as elementwise_max
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true, cond_gate, equals_vector
from torchwright.ops.map_select import in_range, select

from torchwright.doom.renderer import (
    _textured_column_fill,
    _u_norm_lookup,
    _wall_height_lookup,
    trig_lookup,
)
from torchwright.reference_renderer.render import render_column
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

TOKEN_START = 0
TOKEN_WALL = 1
TOKEN_EOS = 2
TOKEN_SORTED_WALL = 3
TOKEN_RENDER = 4

E8_START = index_to_vector(TOKEN_START)
E8_WALL = index_to_vector(TOKEN_WALL)
E8_EOS = index_to_vector(TOKEN_EOS)
E8_SORTED_WALL = index_to_vector(TOKEN_SORTED_WALL)
E8_RENDER = index_to_vector(TOKEN_RENDER)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_COORD = 20.0
BIG_DISTANCE = 1000.0

_DIFF_BP = [
    -40, -30, -20, -15, -10, -7, -5, -3, -2, -1, -0.5,
    0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40,
]
_TRIG_BP = [-1, -0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 1]
_SQRT_BP = [0, 0.25, 1, 2, 4, 9, 16, 25, 36, 49, 64, 100, 225, 400, 900, 1600, 3200]


# ---------------------------------------------------------------------------
# Combined graph
# ---------------------------------------------------------------------------


def build_combined_graph(
    config: RenderConfig,
    textures: List[np.ndarray],
    max_walls: int = 8,
    max_coord: float = MAX_COORD,
) -> Tuple[Node, PosEncoding]:
    """Build a graph that handles all five token types.

    Returns (output_node, pos_encoding).
    """
    H = config.screen_height
    W = config.screen_width
    fov = config.fov_columns
    tex_w, tex_h = textures[0].shape[0], textures[0].shape[1]

    pos_encoding = create_pos_encoding()

    # --- Inputs ---
    token_type = create_input("token_type", 8)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    player_angle = create_input("player_angle", 1)
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    wall_tex_id = create_input("wall_tex_id", 1)
    sort_mask = create_input("sort_mask", max_walls)
    col_idx = create_input("col_idx", 1)
    patch_idx = create_input("patch_idx", 1)

    # --- Token type detection ---
    is_start = equals_vector(token_type, E8_START)
    is_wall = equals_vector(token_type, E8_WALL)
    is_sorted = equals_vector(token_type, E8_SORTED_WALL)
    is_render = equals_vector(token_type, E8_RENDER)

    # --- Player state: fed by host at every position (no get_prev_value) ---
    # The host already knows (px, py, angle) and feeds them directly,
    # just like the baked game_graph's "Option A" pattern.  This avoids
    # a get_prev_value attention layer on the critical path.

    # =====================================================================
    # WALL POSITIONS: compute distance score + angular key + wall value
    # =====================================================================

    # Wall midpoint distance from player (player_{x,y} fed at every position)
    mid_x = multiply_const(add(wall_ax, wall_bx), 0.5)
    mid_y = multiply_const(add(wall_ay, wall_by), 0.5)
    w_dx = subtract(mid_x, player_x)
    w_dy = subtract(mid_y, player_y)
    dx_sq = square_signed(w_dx, max_abs=40.0, step=1.0)
    dy_sq = square_signed(w_dy, max_abs=40.0, step=1.0)
    dist_sq = add(dx_sq, dy_sq)
    wall_dist = piecewise_linear(dist_sq, _SQRT_BP,
                                  lambda x: math.sqrt(max(0, x)), name="wall_dist")

    # Sentinel score for non-wall positions
    sentinel = create_literal_value(torch.tensor([99.0]), name="sentinel")
    sort_score = select(is_wall, wall_dist, sentinel)

    # Wall index: fed by host (0, 1, 2, ... at WALL positions, 0 elsewhere).
    # No prefix_sum needed — the host knows the order it feeds walls.
    wall_index = create_input("wall_index", 1)
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    position_onehot = add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh)

    # Pack wall value for sort: geometry + angular info + onehot
    wall_value_for_sort = Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,
        w_dx, w_dy, wall_dist,
        position_onehot,
    ])
    # Value layout: [0:5] = wall_data, [5:7] = (dx, dy), [7] = dist,
    # [8:8+max_walls] = onehot.  Total: 8 + max_walls.

    # =====================================================================
    # SORTED_WALL POSITIONS: attend_argmin_unmasked to find next wall
    # =====================================================================

    selected_sort = attend_argmin_unmasked(
        pos_encoding=pos_encoding,
        score=sort_score,
        mask_vector=sort_mask,
        position_onehot=position_onehot,
        value=wall_value_for_sort,
    )
    # selected_sort: width = 8 + max_walls
    # Extract sub-fields from selected_sort
    d_sort_val = 8 + max_walls
    def _extract_from(node, d_total, start, width, name):
        m = torch.zeros(d_total, width)
        for i in range(width):
            m[start + i, i] = 1.0
        return Linear(node, m, name=name)

    sel_wall_data = _extract_from(selected_sort, d_sort_val, 0, 5, "sel_wall_data")
    sel_dx = _extract_from(selected_sort, d_sort_val, 5, 1, "sel_dx")
    sel_dy = _extract_from(selected_sort, d_sort_val, 6, 1, "sel_dy")
    sel_dist = _extract_from(selected_sort, d_sort_val, 7, 1, "sel_dist")
    sel_onehot = _extract_from(selected_sort, d_sort_val, 8, max_walls, "sel_onehot")

    # Gate sorted values: zero at non-sorted positions so render
    # attention's K is (0,0,0) there.
    gated_dx = cond_gate(is_sorted, sel_dx)
    gated_dy = cond_gate(is_sorted, sel_dy)
    gated_dist = cond_gate(is_sorted, sel_dist)
    gated_wall_data = cond_gate(is_sorted, sel_wall_data)

    # =====================================================================
    # RENDER POSITIONS: attend to sorted walls, run parametric render
    # =====================================================================

    # Compute ray angle from col_idx + player_angle
    col_times_fov = multiply_const(col_idx, float(fov))
    # angle_offset = col * fov / W - fov/2. Use integer-exact formula.
    ao_raw = piecewise_linear(
        col_times_fov,
        [float(i) for i in range(0, fov * W + 1, max(1, W))],
        lambda x: float(int(x) // W),
        name="ao_raw",
    )
    angle_offset = add_const(ao_raw, float(-(fov // 2)))
    ray_angle_raw = add(player_angle, angle_offset)
    ray_angle_shifted = add_const(ray_angle_raw, 256.0)
    from torchwright.ops.arithmetic_ops import mod_const
    ray_angle = mod_const(ray_angle_shifted, 256, 512 + fov)

    # Trig for this ray
    ray_cos, ray_sin = trig_lookup(ray_angle)

    # Perp cos for fish-eye correction
    perp_shifted = add_const(angle_offset, 256.0)
    perp_angle = mod_const(perp_shifted, 256, 256 + fov)
    perp_cos, _perp_sin = trig_lookup(perp_angle)

    # --- Render attention: find the closest sorted wall for this ray ---
    # Custom Attn node with angular-similarity Q·K.
    # Q at render positions: (ray_cos, ray_sin, 1) * gain
    # K at sorted positions: (dx, dy, wall_bias - dist_scale * dist)
    # K at other positions: (0, 0, 0) via gating above
    RENDER_GAIN = 80.0
    WALL_BIAS = 30.0
    DIST_SCALE = 1.0

    # Gate ray_cos/ray_sin: zero at non-render positions
    gated_ray_cos = cond_gate(is_render, ray_cos)
    gated_ray_sin = cond_gate(is_render, ray_sin)
    is_render_01 = bool_to_01(is_render)
    is_sorted_01 = bool_to_01(is_sorted)

    # Build the render attention input: combine query-side and key-side features
    render_attn_in = Concatenate([
        gated_ray_cos,           # query side: ray direction
        gated_ray_sin,
        is_render_01,            # query side: bias term
        gated_dx,                # key side: wall angular position
        gated_dy,
        gated_dist,              # key side: wall distance
        is_sorted_01,            # key side: wall bias flag
        gated_wall_data,         # value side: wall geometry
    ])

    d_attn_in = len(render_attn_in)
    s_ray_cos = 0
    s_ray_sin = 1
    s_is_render = 2
    s_dx = 3
    s_dy = 4
    s_dist = 5
    s_is_sorted = 6
    s_wall_data = 7  # 5 wide

    d_head_render = 3 + 5  # 3 for Q·K dims, 5 for value pass-through

    q_matrix = torch.zeros(d_attn_in, d_head_render)
    q_matrix[s_ray_cos, 0] = RENDER_GAIN
    q_matrix[s_ray_sin, 1] = RENDER_GAIN
    q_matrix[s_is_render, 2] = RENDER_GAIN

    k_matrix = torch.zeros(d_attn_in, d_head_render)
    k_matrix[s_dx, 0] = 1.0
    k_matrix[s_dy, 1] = 1.0
    k_matrix[s_is_sorted, 2] = WALL_BIAS
    k_matrix[s_dist, 2] = -DIST_SCALE

    v_matrix = torch.zeros(d_attn_in, d_head_render)
    for i in range(5):
        v_matrix[s_wall_data + i, 3 + i] = 1.0

    o_matrix = torch.zeros(d_head_render, 5)
    for i in range(5):
        o_matrix[3 + i, i] = 1.0

    render_attn = Attn(
        query_in=render_attn_in,
        key_in=render_attn_in,
        value_in=render_attn_in,
        query_matrix=q_matrix,
        key_matrix=k_matrix,
        value_matrix=v_matrix,
        output_matrix=o_matrix,
    )
    # render_attn output: 5 values = (wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id)

    # Extract attended wall params
    r_wall_ax = _extract_from(render_attn, 5, 0, 1, "r_ax")
    r_wall_ay = _extract_from(render_attn, 5, 1, 1, "r_ay")
    r_wall_bx = _extract_from(render_attn, 5, 2, 1, "r_bx")
    r_wall_by = _extract_from(render_attn, 5, 3, 1, "r_by")
    r_wall_tex = _extract_from(render_attn, 5, 4, 1, "r_tex")

    # --- Parametric intersection ---
    ex = subtract(r_wall_bx, r_wall_ax)
    ey = subtract(r_wall_by, r_wall_ay)
    dx_r = subtract(r_wall_ax, player_x)
    dy_r = subtract(player_y, r_wall_ay)

    ey_cos = piecewise_linear_2d(ey, ray_cos, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_ey_cos")
    ex_sin = piecewise_linear_2d(ex, ray_sin, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_ex_sin")
    den = subtract(ey_cos, ex_sin)

    ey_dx = piecewise_linear_2d(ey, dx_r, _DIFF_BP, _DIFF_BP, lambda a,b: a*b, name="r_ey_dx")
    ex_dy = piecewise_linear_2d(ex, dy_r, _DIFF_BP, _DIFF_BP, lambda a,b: a*b, name="r_ex_dy")
    num_t = add(ey_dx, ex_dy)

    dx_sin = piecewise_linear_2d(dx_r, ray_sin, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_dx_sin")
    dy_cos = piecewise_linear_2d(dy_r, ray_cos, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_dy_cos")
    num_u = add(dx_sin, dy_cos)

    # Den → angle data
    sign_den = compare(den, 0.0)
    abs_den = abs(den)
    inv_abs_den = reciprocal(abs_den, min_value=0.01, max_value=2.0*max_coord)
    signed_inv_den = select(sign_den, inv_abs_den, negate(inv_abs_den))

    # Distance
    adj_num_t = select(sign_den, num_t, negate(num_t))
    adj_num_u = select(sign_den, num_u, negate(num_u))
    is_den_nz = compare(abs_den, 0.05)
    is_t_pos = compare(adj_num_t, 0.05)
    is_u_ge0 = compare(adj_num_u, -0.05)
    u_minus_den = subtract(abs_den, adj_num_u)
    is_u_le_den = compare(u_minus_den, -0.05)
    abs_inv = select(sign_den, signed_inv_den, negate(signed_inv_den))
    dist_r = signed_multiply(adj_num_t, abs_inv, max_abs1=800, max_abs2=100,
                              step=1.0, max_abs_output=BIG_DISTANCE)
    is_valid = bool_all_true([is_den_nz, is_t_pos, is_u_ge0, is_u_le_den])
    big = create_literal_value(torch.tensor([BIG_DISTANCE]), name="big")
    dist_r = select(is_valid, dist_r, big)

    # Wall height
    wall_top, wall_bottom, wall_height = _wall_height_lookup(dist_r, perp_cos, config, max_coord)

    # Texture column
    tex_col_idx = _u_norm_lookup(adj_num_u, abs_den, tex_w, max_coord)
    num_tex = len(textures)
    n_keys = num_tex * tex_w
    flat_key = add(multiply_const(r_wall_tex, float(tex_w)), tex_col_idx)

    def _tex_col_vals(flat_idx):
        k = int(round(flat_idx))
        if 0 <= k < n_keys:
            tid = k // tex_w
            col = k % tex_w
            return [float(v) for v in textures[tid][col].flatten()]
        return [0.0] * (tex_h * 3)

    tex_column_colors = piecewise_linear(
        flat_key, [float(k) for k in range(n_keys)],
        _tex_col_vals, name="tex_col_lookup",
    )

    # Column fill
    patch_row_start = multiply_const(patch_idx, float(H))  # rp=H for now (full column)
    pixels = _textured_column_fill(
        wall_top, wall_bottom, wall_height,
        tex_column_colors, tex_h, config, max_coord=max_coord,
    )

    # --- Output: gated by token type ---
    # At SORTED_WALL positions: emit sort output (for host mask update)
    sort_output = Concatenate([
        create_literal_value(E8_SORTED_WALL, name="sort_type"),
        sel_wall_data,
        sel_onehot,
    ])
    # At RENDER positions: emit pixels
    render_output = Concatenate([
        create_literal_value(E8_RENDER, name="render_type"),
        pixels,
    ])
    # Pad to same width
    d_sort_out = 8 + 5 + max_walls
    d_render_out = 8 + H * 3
    d_out = max(d_sort_out, d_render_out)

    sort_padded = Concatenate([sort_output, create_literal_value(
        torch.zeros(d_out - d_sort_out), name="sort_pad")])
    render_padded = Concatenate([render_output, create_literal_value(
        torch.zeros(d_out - d_render_out), name="render_pad")])

    output = select(is_render, render_padded, sort_padded)

    return output, pos_encoding


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _build_row(compiled, max_walls, **kwargs):
    """Build a (1, d_input) row for module.step()."""
    defaults = {
        "token_type": torch.zeros(8),
        "player_x": torch.zeros(1),
        "player_y": torch.zeros(1),
        "player_angle": torch.zeros(1),
        "wall_ax": torch.zeros(1),
        "wall_ay": torch.zeros(1),
        "wall_bx": torch.zeros(1),
        "wall_by": torch.zeros(1),
        "wall_tex_id": torch.zeros(1),
        "wall_index": torch.zeros(1),
        "sort_mask": torch.zeros(max_walls),
        "col_idx": torch.zeros(1),
        "patch_idx": torch.zeros(1),
    }
    defaults.update(kwargs)
    d_input = max(s + w for _, s, w in compiled._input_specs)
    row = torch.zeros(1, d_input)
    for name, start, width in compiled._input_specs:
        v = defaults[name]
        if isinstance(v, (int, float)):
            v = torch.tensor([v])
        if v.dim() == 1:
            v = v.unsqueeze(0)
        row[:, start:start + width] = v
    return row


# ---------------------------------------------------------------------------
# Test: render box room via the combined graph
# ---------------------------------------------------------------------------


def test_combined_renders_box_room():
    """Compile the combined graph, run the full rollout (prefill walls,
    sort, render), and compare the rendered frame against the reference
    renderer for the box room scene.
    """
    config = RenderConfig(
        screen_width=16, screen_height=20, fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2), floor_color=(0.4, 0.4, 0.4),
    )
    textures = default_texture_atlas()
    max_walls = 8
    H = config.screen_height
    W = config.screen_width

    # Box room: 4 walls
    h = 5.0
    walls = [
        {"ax": h, "ay": -h, "bx": h, "by": h, "tex_id": 0.0},    # east
        {"ax": -h, "ay": -h, "bx": -h, "by": h, "tex_id": 1.0},   # west
        {"ax": -h, "ay": h, "bx": h, "by": h, "tex_id": 2.0},     # north
        {"ax": -h, "ay": -h, "bx": h, "by": -h, "tex_id": 3.0},   # south
    ]
    segs = [Segment(ax=w["ax"], ay=w["ay"], bx=w["bx"], by=w["by"],
                     color=(0.8, 0.2, 0.1), texture_id=int(w["tex_id"]))
            for w in walls]
    N = len(walls)

    px, py, player_angle = 0.0, 0.0, 0.0

    output_node, pos_encoding = build_combined_graph(config, textures, max_walls)
    compiled = compile_headless(
        output_node, pos_encoding,
        d=2048, d_head=32, max_layers=400, verbose=False,
    )

    past = compiled.empty_past()
    step = 0

    # Prefill: START
    row = _build_row(compiled, max_walls,
                     token_type=E8_START,
                     player_x=torch.tensor([px]),
                     player_y=torch.tensor([py]),
                     player_angle=torch.tensor([player_angle]))
    with torch.no_grad():
        out, past = compiled.step(row, past, past_len=step)
    step += 1

    # Prefill: WALL × N (host feeds player state + wall_index at every position)
    for i, w in enumerate(walls):
        row = _build_row(compiled, max_walls,
                         token_type=E8_WALL,
                         player_x=torch.tensor([px]),
                         player_y=torch.tensor([py]),
                         player_angle=torch.tensor([player_angle]),
                         wall_ax=torch.tensor([w["ax"]]),
                         wall_ay=torch.tensor([w["ay"]]),
                         wall_bx=torch.tensor([w["bx"]]),
                         wall_by=torch.tensor([w["by"]]),
                         wall_tex_id=torch.tensor([w["tex_id"]]),
                         wall_index=torch.tensor([float(i)]))
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1

    # Prefill: EOS
    row = _build_row(compiled, max_walls, token_type=E8_EOS)
    with torch.no_grad():
        out, past = compiled.step(row, past, past_len=step)
    step += 1

    # Sort: N steps
    onehot_sl = slice(8 + 5, 8 + 5 + max_walls)
    mask = np.zeros(max_walls)
    for k in range(N):
        row = _build_row(compiled, max_walls,
                         token_type=E8_SORTED_WALL,
                         player_x=torch.tensor([px]),
                         player_y=torch.tensor([py]),
                         player_angle=torch.tensor([player_angle]),
                         sort_mask=torch.tensor(mask, dtype=torch.float32))
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1
        onehot = out[0, onehot_sl].detach().cpu().numpy()
        mask = np.maximum(mask, np.round(onehot))

    # Render: W columns (full column, no patch sharding)
    frame = np.zeros((H, W, 3), dtype=np.float32)
    pixel_sl = slice(8, 8 + H * 3)
    for col in range(W):
        row = _build_row(compiled, max_walls,
                         token_type=E8_RENDER,
                         player_x=torch.tensor([px]),
                         player_y=torch.tensor([py]),
                         player_angle=torch.tensor([player_angle]),
                         col_idx=torch.tensor([float(col)]),
                         patch_idx=torch.tensor([0.0]))
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1
        pixels = out[0, pixel_sl].detach().cpu().numpy().reshape(H, 3)
        frame[:, col, :] = pixels

    # Compare against reference renderer
    ref_frame = np.zeros((H, W, 3), dtype=np.float64)
    for col in range(W):
        ref_frame[:, col, :] = render_column(
            col, px, py, int(player_angle), segs, config, textures=textures,
        )

    max_err = np.abs(frame - ref_frame).max()
    mean_err = np.abs(frame - ref_frame).mean()

    # Verify frame isn't blank
    assert frame.max() > 0.1, "frame appears blank"

    # The error budget is generous because we're going through ~10 stages
    # of piecewise_linear approximation plus an attention-based wall
    # selection (which can soft-blend at column boundaries).
    assert max_err < 0.5, (
        f"max pixel error {max_err:.3f} exceeds 0.5 (mean {mean_err:.3f})"
    )
    print(f"\nCombined v2 frame: max_err={max_err:.3f}, mean_err={mean_err:.3f}")
