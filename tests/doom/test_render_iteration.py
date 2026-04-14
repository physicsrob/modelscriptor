"""End-to-end test: wall sort → render-mask column iteration (± chunking).

Combines the wall sort prototype with the range_printer pattern to validate
the full pipeline: sort walls front-to-back, then iterate columns [col_lo,
col_hi) for each wall using attend_argmin_unmasked with a render_mask.

Two graph variants:
  build_sort_render_graph          — 2-level loop (wall × column)
  build_sort_render_chunked_graph  — 3-level loop (wall × column × chunk)

The graph has three token types:
  WALL         — prefill with wall geometry
  SORTED_WALL  — autoregressive sort (host-fed sort_mask)
  RENDER       — autoregressive column iteration (host-fed render_mask)

col_lo/col_hi (and vis_top/vis_bottom for chunked) are host-fed to WALL
positions.  The graph doesn't do atan2 projection or wall-height math —
those are tested separately.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    add_scaled_nodes,
    compare,
    multiply_const,
    piecewise_linear,
    square_signed,
    subtract,
)
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.prefix_ops import prefix_sum
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true, bool_not, equals_vector
from torchwright.ops.map_select import in_range, select

from torchwright.reference_renderer.render import intersect_ray_segment
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN_WALL = 0
TOKEN_SORTED_WALL = 1
TOKEN_RENDER = 2

E8_WALL = index_to_vector(TOKEN_WALL)
E8_SORTED_WALL = index_to_vector(TOKEN_SORTED_WALL)
E8_RENDER = index_to_vector(TOKEN_RENDER)

D_TOKEN_TYPE = 8
_SENTINEL_SCORE = 99.0
_RENDER_SENTINEL = 99.0
_SQUARE_MAX_ABS = 40.0
_SQUARE_STEP = 2.0
_SQRT_BREAKPOINTS = [0, 1, 4, 9, 16, 25, 36, 49, 64, 100, 200, 400, 800, 1600]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _extract_from(node, d_total, start, width, name):
    m = torch.zeros(d_total, width)
    for i in range(width):
        m[start + i, i] = 1.0
    return Linear(node, m, name=name)


def build_sort_render_graph(
    max_walls: int,
) -> Tuple[Node, PosEncoding]:
    """Build a graph with sort + render iteration phases.

    Returns ``(output_node, pos_encoding)``.
    """
    pos_encoding = create_pos_encoding()

    # --- Inputs ---
    token_type = create_input("token_type", D_TOKEN_TYPE)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    wall_tex_id = create_input("wall_tex_id", 1)
    sort_mask = create_input("sort_mask", max_walls)
    # Host-fed column range for SORTED_WALL positions
    wall_col_lo = create_input("wall_col_lo", 1)
    wall_col_hi = create_input("wall_col_hi", 1)
    # Render feedback
    render_mask = create_input("render_mask", max_walls)
    render_col = create_input("render_col", 1)
    render_is_new_wall = create_input("render_is_new_wall", 1)

    # --- Token type detection ---
    is_wall = equals_vector(token_type, E8_WALL)
    is_sorted = equals_vector(token_type, E8_SORTED_WALL)
    is_render = equals_vector(token_type, E8_RENDER)

    # =================================================================
    # SORT PHASE (same as wall_sort_prototype)
    # =================================================================

    # Wall midpoint distance
    mid_x = multiply_const(add(wall_ax, wall_bx), 0.5)
    mid_y = multiply_const(add(wall_ay, wall_by), 0.5)
    dx = subtract(mid_x, player_x)
    dy = subtract(mid_y, player_y)
    dx_sq = square_signed(dx, max_abs=_SQUARE_MAX_ABS, step=_SQUARE_STEP)
    dy_sq = square_signed(dy, max_abs=_SQUARE_MAX_ABS, step=_SQUARE_STEP)
    dist_sq = add(dx_sq, dy_sq)
    score_raw = piecewise_linear(
        dist_sq, _SQRT_BREAKPOINTS,
        lambda x: math.sqrt(max(0.0, x)), name="dist_score",
    )
    sentinel = create_literal_value(torch.tensor([_SENTINEL_SCORE]), name="sentinel")
    sort_score = select(is_wall, score_raw, sentinel)

    # Wall index via prefix_sum
    is_wall_01 = multiply_const(add_const(is_wall, 1.0), 0.5)
    n_stages = max(5, math.ceil(math.log2(2 * max_walls + 3)))
    wall_count = prefix_sum(pos_encoding, is_wall_01, n_stages=n_stages)
    wall_index = add_const(wall_count, -1.0)

    # Position one-hot {0, 1}
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    position_onehot = add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh)

    # Sort value: wall_data + col_lo + col_hi + position_onehot
    sort_value = Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,  # 0-4
        wall_col_lo, wall_col_hi,                           # 5-6
        position_onehot,                                    # 7+
    ])
    d_sort_val = 7 + max_walls

    # Sort attention
    selected_sort = attend_argmin_unmasked(
        pos_encoding=pos_encoding,
        score=sort_score,
        mask_vector=sort_mask,
        position_onehot=position_onehot,
        value=sort_value,
    )

    sel_wall_data = _extract_from(selected_sort, d_sort_val, 0, 5, "sel_wall_data")
    sel_col_lo = _extract_from(selected_sort, d_sort_val, 5, 1, "sel_col_lo")
    sel_col_hi = _extract_from(selected_sort, d_sort_val, 6, 1, "sel_col_hi")
    sel_onehot = _extract_from(selected_sort, d_sort_val, 7, max_walls, "sel_onehot")

    # Sort rank: number of walls already sorted
    sort_rank = Linear(sort_mask, torch.ones(max_walls, 1), name="sort_rank")

    # Sort rank one-hot {0, 1}
    sort_rank_p1 = add_const(sort_rank, 1.0)
    sr_onehot_bool = in_range(sort_rank, sort_rank_p1, max_walls)
    sort_rank_onehot = add_scaled_nodes(0.5, sr_onehot_bool, 0.5, ones_oh)

    # =================================================================
    # RENDER PHASE (range_printer pattern)
    # =================================================================

    # Render attention: select first unrendered sorted wall.
    # Score at SORTED_WALL positions = sort_rank (0, 1, 2, ...).
    # Score at other positions = sentinel.
    render_sentinel = create_literal_value(
        torch.tensor([_RENDER_SENTINEL]), name="render_sentinel",
    )
    render_score = select(is_sorted, sort_rank, render_sentinel)

    # Position one-hot and value at SORTED_WALL positions
    render_position_onehot = select(
        is_sorted, sort_rank_onehot,
        create_literal_value(torch.zeros(max_walls), name="z_mw"),
    )
    render_value = Concatenate([sel_col_lo, sel_col_hi, sort_rank_onehot])
    d_render_val = 2 + max_walls

    selected_render = attend_argmin_unmasked(
        pos_encoding=pos_encoding,
        score=render_score,
        mask_vector=render_mask,
        position_onehot=render_position_onehot,
        value=render_value,
    )

    r_col_lo = _extract_from(selected_render, d_render_val, 0, 1, "r_col_lo")
    r_col_hi = _extract_from(selected_render, d_render_val, 1, 1, "r_col_hi")
    r_onehot = _extract_from(selected_render, d_render_val, 2, max_walls, "r_onehot")

    # State machine
    active_col = select(render_is_new_wall, r_col_lo, render_col)
    next_col_val = add_const(active_col, 1.0)
    remaining = subtract(r_col_hi, next_col_val)
    inner_continues = compare(remaining, 0.5)
    inner_done = bool_not(inner_continues)

    mask_with_new = add(render_mask, r_onehot)
    next_render_mask = select(inner_done, mask_with_new, render_mask)

    mask_sum = Linear(mask_with_new, torch.ones(max_walls, 1), name="render_mask_sum")
    all_done = compare(mask_sum, max_walls - 0.5)
    done_flag = bool_all_true([inner_done, all_done])

    next_is_new_wall = inner_done
    zero_col = create_literal_value(torch.tensor([0.0]), name="zero_col")
    next_col_output = select(inner_continues, next_col_val, zero_col)

    # =================================================================
    # Output: gated by token type
    # =================================================================

    # SORTED_WALL output: type + wall_data + sel_onehot
    sort_output = Concatenate([
        create_literal_value(E8_SORTED_WALL, name="sort_type"),
        sel_wall_data,
        sel_onehot,
    ])
    d_sort_out = D_TOKEN_TYPE + 5 + max_walls

    # RENDER output: active_col + done + feedback
    render_output = Concatenate([
        create_literal_value(E8_RENDER, name="render_type"),
        active_col,
        done_flag,
        next_render_mask,
        next_col_output,
        next_is_new_wall,
    ])
    d_render_out = D_TOKEN_TYPE + 1 + 1 + max_walls + 1 + 1

    # Pad and select
    d_out = max(d_sort_out, d_render_out)

    def _pad(node, cur_width):
        if cur_width >= d_out:
            return node
        return Concatenate([node, create_literal_value(
            torch.zeros(d_out - cur_width), name="pad")])

    sort_padded = _pad(sort_output, d_sort_out)
    render_padded = _pad(render_output, d_render_out)
    default_padded = _pad(
        create_literal_value(torch.zeros(D_TOKEN_TYPE), name="default_type"),
        D_TOKEN_TYPE,
    )

    inner = select(is_sorted, sort_padded, default_padded)
    output = select(is_render, render_padded, inner)

    return output, pos_encoding


# ---------------------------------------------------------------------------
# Chunked graph: 3-level loop (wall × column × chunk)
# ---------------------------------------------------------------------------


def build_sort_render_chunked_graph(
    max_walls: int,
    chunk_size: int = 5,
) -> Tuple[Node, PosEncoding]:
    """Build a graph with sort + chunked render iteration.

    Like :func:`build_sort_render_graph` but each (wall, column) pair
    emits ``ceil(visible_height / chunk_size)`` RENDER tokens.  Each
    token outputs ``(col, chunk_start, chunk_length)``.

    Wall visibility (vis_top, vis_bottom) is host-fed per WALL token.

    Returns ``(output_node, pos_encoding)``.
    """
    pos_encoding = create_pos_encoding()

    # --- Inputs ---
    token_type = create_input("token_type", D_TOKEN_TYPE)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    wall_tex_id = create_input("wall_tex_id", 1)
    sort_mask = create_input("sort_mask", max_walls)
    wall_col_lo = create_input("wall_col_lo", 1)
    wall_col_hi = create_input("wall_col_hi", 1)
    wall_vis_top = create_input("wall_vis_top", 1)
    wall_vis_bottom = create_input("wall_vis_bottom", 1)
    # Render feedback (4 fields)
    render_mask = create_input("render_mask", max_walls)
    render_col = create_input("render_col", 1)
    render_is_new_wall = create_input("render_is_new_wall", 1)
    render_chunk_start = create_input("render_chunk_start", 1)  # -1 = new column

    # --- Token type detection ---
    is_wall = equals_vector(token_type, E8_WALL)
    is_sorted = equals_vector(token_type, E8_SORTED_WALL)
    is_render = equals_vector(token_type, E8_RENDER)

    # =================================================================
    # SORT PHASE
    # =================================================================

    mid_x = multiply_const(add(wall_ax, wall_bx), 0.5)
    mid_y = multiply_const(add(wall_ay, wall_by), 0.5)
    dx = subtract(mid_x, player_x)
    dy = subtract(mid_y, player_y)
    dx_sq = square_signed(dx, max_abs=_SQUARE_MAX_ABS, step=_SQUARE_STEP)
    dy_sq = square_signed(dy, max_abs=_SQUARE_MAX_ABS, step=_SQUARE_STEP)
    dist_sq = add(dx_sq, dy_sq)
    score_raw = piecewise_linear(
        dist_sq, _SQRT_BREAKPOINTS,
        lambda x: math.sqrt(max(0.0, x)), name="dist_score",
    )
    sentinel = create_literal_value(torch.tensor([_SENTINEL_SCORE]), name="sentinel")
    sort_score = select(is_wall, score_raw, sentinel)

    is_wall_01 = multiply_const(add_const(is_wall, 1.0), 0.5)
    n_stages = max(5, math.ceil(math.log2(2 * max_walls + 3)))
    wall_count = prefix_sum(pos_encoding, is_wall_01, n_stages=n_stages)
    wall_index = add_const(wall_count, -1.0)

    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    position_onehot = add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh)

    # Sort value: includes vis_top, vis_bottom for render phase
    sort_value = Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,  # 0-4
        wall_col_lo, wall_col_hi,                           # 5-6
        wall_vis_top, wall_vis_bottom,                      # 7-8
        position_onehot,                                    # 9+
    ])
    d_sort_val = 9 + max_walls

    selected_sort = attend_argmin_unmasked(
        pos_encoding=pos_encoding,
        score=sort_score,
        mask_vector=sort_mask,
        position_onehot=position_onehot,
        value=sort_value,
    )

    sel_wall_data = _extract_from(selected_sort, d_sort_val, 0, 5, "sel_wall_data")
    sel_col_lo = _extract_from(selected_sort, d_sort_val, 5, 1, "sel_col_lo")
    sel_col_hi = _extract_from(selected_sort, d_sort_val, 6, 1, "sel_col_hi")
    sel_vis_top = _extract_from(selected_sort, d_sort_val, 7, 1, "sel_vis_top")
    sel_vis_bottom = _extract_from(selected_sort, d_sort_val, 8, 1, "sel_vis_bottom")
    sel_onehot = _extract_from(selected_sort, d_sort_val, 9, max_walls, "sel_onehot")

    sort_rank = Linear(sort_mask, torch.ones(max_walls, 1), name="sort_rank")
    sort_rank_p1 = add_const(sort_rank, 1.0)
    sr_onehot_bool = in_range(sort_rank, sort_rank_p1, max_walls)
    sort_rank_onehot = add_scaled_nodes(0.5, sr_onehot_bool, 0.5, ones_oh)

    # =================================================================
    # RENDER PHASE (3-level: wall × column × chunk)
    # =================================================================

    render_sentinel = create_literal_value(
        torch.tensor([_RENDER_SENTINEL]), name="render_sentinel",
    )
    render_score = select(is_sorted, sort_rank, render_sentinel)

    render_position_onehot = select(
        is_sorted, sort_rank_onehot,
        create_literal_value(torch.zeros(max_walls), name="z_mw"),
    )
    render_value = Concatenate([
        sel_col_lo, sel_col_hi, sel_vis_top, sel_vis_bottom, sort_rank_onehot,
    ])
    d_render_val = 4 + max_walls

    selected_render = attend_argmin_unmasked(
        pos_encoding=pos_encoding,
        score=render_score,
        mask_vector=render_mask,
        position_onehot=render_position_onehot,
        value=render_value,
    )

    r_col_lo = _extract_from(selected_render, d_render_val, 0, 1, "r_col_lo")
    r_col_hi = _extract_from(selected_render, d_render_val, 1, 1, "r_col_hi")
    r_vis_top = _extract_from(selected_render, d_render_val, 2, 1, "r_vis_top")
    r_vis_bottom = _extract_from(selected_render, d_render_val, 3, 1, "r_vis_bottom")
    r_onehot = _extract_from(selected_render, d_render_val, 4, max_walls, "r_onehot")

    # --- Determine active column ---
    active_col = select(render_is_new_wall, r_col_lo, render_col)

    # --- Determine active chunk start ---
    # chunk_start < -0.5 → sentinel → new column → use vis_top
    neg_half = create_literal_value(torch.tensor([-0.5]), name="neg_half")
    is_new_col = compare(subtract(neg_half, render_chunk_start), 0.0)  # True when -0.5 > chunk_start
    active_start = select(is_new_col, r_vis_top, render_chunk_start)
    active_bottom = r_vis_bottom  # always from attention (same wall)

    # --- Chunk length ---
    from torchwright.ops.arithmetic_ops import clamp
    raw_length = subtract(active_bottom, active_start)
    chunk_length = clamp(raw_length, 0.0, float(chunk_size))

    # --- 3-way state transition ---
    next_chunk_start_val = add_const(active_start, float(chunk_size))
    has_more_chunks = compare(subtract(active_bottom, next_chunk_start_val), 0.5)

    col_p1 = add_const(active_col, 1.0)
    not_more_chunks = bool_not(has_more_chunks)
    has_more_cols = compare(subtract(r_col_hi, col_p1), 0.5)
    advance_col = bool_all_true([not_more_chunks, has_more_cols])
    advance_wall = bool_all_true([not_more_chunks, bool_not(has_more_cols)])

    # Mask update
    mask_with_new = add(render_mask, r_onehot)
    next_render_mask = select(advance_wall, mask_with_new, render_mask)

    # Done detection
    mask_sum = Linear(mask_with_new, torch.ones(max_walls, 1), name="render_mask_sum")
    all_walls_done = compare(mask_sum, max_walls - 0.5)
    done_flag = bool_all_true([advance_wall, all_walls_done])

    # Next col
    zero_col = create_literal_value(torch.tensor([0.0]), name="zero_col")
    next_col_output = select(
        has_more_chunks, active_col,
        select(advance_col, col_p1, zero_col),
    )

    # Next chunk_start: advance if more chunks, sentinel otherwise
    chunk_sentinel = create_literal_value(torch.tensor([-1.0]), name="chunk_sentinel")
    next_chunk = select(has_more_chunks, next_chunk_start_val, chunk_sentinel)

    # Next is_new_wall
    pos_one = create_literal_value(torch.tensor([1.0]), name="pos_one")
    neg_one = create_literal_value(torch.tensor([-1.0]), name="neg_one")
    next_is_new_wall = select(advance_wall, pos_one, neg_one)

    # =================================================================
    # Output
    # =================================================================

    sort_output = Concatenate([
        create_literal_value(E8_SORTED_WALL, name="sort_type"),
        sel_wall_data,
        sel_onehot,
    ])
    d_sort_out = D_TOKEN_TYPE + 5 + max_walls

    render_output = Concatenate([
        create_literal_value(E8_RENDER, name="render_type"),
        active_col,         # col being rendered
        active_start,       # chunk start_y
        chunk_length,       # chunk length
        done_flag,          # done?
        next_render_mask,   # feedback: mask
        next_col_output,    # feedback: col
        next_is_new_wall,   # feedback: is_new_wall
        next_chunk,         # feedback: chunk_start
    ])
    d_render_out = D_TOKEN_TYPE + 4 + max_walls + 3

    d_out = max(d_sort_out, d_render_out)

    def _pad(node, cur_width):
        if cur_width >= d_out:
            return node
        return Concatenate([node, create_literal_value(
            torch.zeros(d_out - cur_width), name="pad")])

    sort_padded = _pad(sort_output, d_sort_out)
    render_padded = _pad(render_output, d_render_out)
    default_padded = _pad(
        create_literal_value(torch.zeros(D_TOKEN_TYPE), name="default_type"),
        D_TOKEN_TYPE,
    )

    inner = select(is_sorted, sort_padded, default_padded)
    output = select(is_render, render_padded, inner)

    return output, pos_encoding


# ---------------------------------------------------------------------------
# Reference renderer utilities
# ---------------------------------------------------------------------------

TRIG = generate_trig_table()


def _vis_cols(seg, px, py, angle, config):
    """Compute the contiguous column range where a wall is visible."""
    W, fov = config.screen_width, config.fov_columns
    cols = []
    for col in range(W):
        ao = (col - W // 2) * fov // W
        ra = (angle + ao) % 256
        if intersect_ray_segment(px, py, TRIG[ra, 0], TRIG[ra, 1], seg):
            cols.append(col)
    if not cols:
        return None
    return min(cols), max(cols) + 1  # [lo, hi)


def _config(W=32, H=24, fov=8):
    return RenderConfig(
        screen_width=W, screen_height=H, fov_columns=fov,
        trig_table=TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _build_step_row(compiled, token_type_vec, max_walls, **kwargs):
    mask_sort = kwargs.get("sort_mask", np.zeros(max_walls))
    mask_render = kwargs.get("render_mask", np.zeros(max_walls))
    vals = {
        "player_x": torch.tensor([[kwargs.get("px", 0.0)]]),
        "player_y": torch.tensor([[kwargs.get("py", 0.0)]]),
        "render_chunk_start": torch.tensor([[kwargs.get("render_chunk_start", -1.0)]]),
        "render_col": torch.tensor([[kwargs.get("render_col", 0.0)]]),
        "render_is_new_wall": torch.tensor([[kwargs.get("render_is_new_wall", -1.0)]]),
        "render_mask": torch.tensor(mask_render, dtype=torch.float32).unsqueeze(0),
        "sort_mask": torch.tensor(mask_sort, dtype=torch.float32).unsqueeze(0),
        "token_type": token_type_vec.unsqueeze(0),
        "wall_ax": torch.tensor([[kwargs.get("ax", 0.0)]]),
        "wall_ay": torch.tensor([[kwargs.get("ay", 0.0)]]),
        "wall_bx": torch.tensor([[kwargs.get("bx", 0.0)]]),
        "wall_by": torch.tensor([[kwargs.get("by", 0.0)]]),
        "wall_col_hi": torch.tensor([[kwargs.get("col_hi", 0.0)]]),
        "wall_col_lo": torch.tensor([[kwargs.get("col_lo", 0.0)]]),
        "wall_tex_id": torch.tensor([[kwargs.get("tex_id", 0.0)]]),
        "wall_vis_bottom": torch.tensor([[kwargs.get("vis_bottom", 0.0)]]),
        "wall_vis_top": torch.tensor([[kwargs.get("vis_top", 0.0)]]),
    }
    d_input = max(s + w for _, s, w in compiled._input_specs)
    row = torch.zeros(1, d_input)
    for name, start, width in compiled._input_specs:
        if name in vals:
            row[:, start:start + width] = vals[name]
    return row


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

# SORTED_WALL: [type(8) | wall_data(5) | onehot(max_walls)]
_SORT_ONEHOT_OFFSET = D_TOKEN_TYPE + 5

# RENDER: [type(8) | active_col(1) | done(1) | mask(N) | col(1) | is_new(1)]
_RENDER_COL = D_TOKEN_TYPE
_RENDER_DONE = D_TOKEN_TYPE + 1
_RENDER_FB_OFFSET = D_TOKEN_TYPE + 2  # mask starts here


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _four_walls_with_ranges():
    """Four walls in front of the player at (0, 0) facing angle 0 (east).

    Returns (walls, config, col_ranges) where col_ranges[i] = (lo, hi)
    for each wall in *input* order.
    """
    config = _config(W=32, H=24, fov=8)
    walls = [
        {"ax": 3.0, "ay": -3.0, "bx": 3.0, "by": 3.0, "tex_id": 0.0},   # dist ~3
        {"ax": 6.0, "ay": -6.0, "bx": 6.0, "by": 6.0, "tex_id": 1.0},   # dist ~6
        {"ax": 10.0, "ay": -10.0, "bx": 10.0, "by": 10.0, "tex_id": 2.0}, # dist ~10
        {"ax": 15.0, "ay": -15.0, "bx": 15.0, "by": 15.0, "tex_id": 3.0}, # dist ~15
    ]
    col_ranges = {}
    for i, w in enumerate(walls):
        seg = Segment(ax=w["ax"], ay=w["ay"], bx=w["bx"], by=w["by"],
                      color=(1, 0, 0))
        rng = _vis_cols(seg, 0, 0, 0, config)
        if rng is not None:
            col_ranges[i] = rng
    return walls, config, col_ranges


def _expected_sort_order(px, py, walls):
    """Return wall indices sorted by midpoint distance."""
    dists = []
    for i, w in enumerate(walls):
        mx = (w["ax"] + w["bx"]) / 2
        my = (w["ay"] + w["by"]) / 2
        d = math.sqrt((mx - px) ** 2 + (my - py) ** 2)
        dists.append((d, i))
    dists.sort()
    return [idx for _, idx in dists]


# ---------------------------------------------------------------------------
# Test: autoregressive sort → render iteration
# ---------------------------------------------------------------------------


def test_sort_then_render():
    """Full autoregressive rollout: sort 4 walls, then iterate columns."""
    walls, config, col_ranges = _four_walls_with_ranges()
    max_walls = len(walls)
    N = len(walls)
    sort_order = _expected_sort_order(0, 0, walls)

    output_node, pos_encoding = build_sort_render_graph(max_walls)
    compiled = compile_headless(
        output_node, pos_encoding,
        d=512, d_head=32, max_layers=200, verbose=False,
    )

    past = compiled.empty_past()
    step = 0

    # --- Prefill: WALL tokens (with host-computed col_lo/col_hi) ---
    for i, w in enumerate(walls):
        lo, hi = col_ranges.get(i, (0, 0))
        row = _build_step_row(
            compiled, E8_WALL, max_walls,
            px=0.0, py=0.0,
            ax=w["ax"], ay=w["ay"], bx=w["bx"], by=w["by"],
            tex_id=w["tex_id"],
            col_lo=float(lo), col_hi=float(hi),
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1

    # --- Sort phase: N steps ---
    sort_mask = np.zeros(max_walls)
    sorted_wall_indices = []
    sorted_col_ranges = []

    for k in range(N):
        row = _build_step_row(
            compiled, E8_SORTED_WALL, max_walls,
            px=0.0, py=0.0, sort_mask=sort_mask,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1

        # Read sort output
        onehot_sl = slice(_SORT_ONEHOT_OFFSET, _SORT_ONEHOT_OFFSET + max_walls)
        onehot = out[0, onehot_sl].detach().cpu().numpy()
        selected_idx = int(np.argmax(np.round(onehot)))
        sorted_wall_indices.append(selected_idx)

        # Look up col range for the selected wall
        if selected_idx in col_ranges:
            sorted_col_ranges.append(col_ranges[selected_idx])
        else:
            sorted_col_ranges.append((0, 0))

        sort_mask = np.maximum(sort_mask, np.round(onehot))

    # Verify sort order
    assert sorted_wall_indices == sort_order, (
        f"Sort order: {sorted_wall_indices}, expected {sort_order}"
    )

    # --- Render iteration phase ---
    # Build expected column sequence
    expected_cols = []
    for lo, hi in sorted_col_ranges:
        expected_cols.extend(range(lo, hi))

    render_mask = np.zeros(max_walls)
    is_new = 1.0
    col_val = 0.0
    emitted = []

    fb_offset = _RENDER_FB_OFFSET
    max_steps = len(expected_cols) + 10

    for k in range(max_steps):
        # Feed col_lo/col_hi at the SORTED_WALL positions (already in KV).
        # At RENDER positions, the graph reads them via attention.
        row = _build_step_row(
            compiled, E8_RENDER, max_walls,
            px=0.0, py=0.0,
            render_mask=render_mask,
            render_col=col_val,
            render_is_new_wall=is_new,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1

        active_col = out[0, _RENDER_COL].item()
        done = out[0, _RENDER_DONE].item()
        emitted.append(round(active_col))

        # Read feedback
        fb = out[0, fb_offset:fb_offset + max_walls + 2].detach().cpu().numpy()
        render_mask = np.round(fb[:max_walls]).clip(0, 1)
        col_val = float(fb[max_walls])
        is_new = float(fb[max_walls + 1])

        if done > 0.0:
            break

    assert emitted == expected_cols, (
        f"Emitted {emitted}, expected {expected_cols}"
    )


# ---------------------------------------------------------------------------
# Test: autoregressive sort → chunked render iteration (3-level loop)
# ---------------------------------------------------------------------------

# Chunked RENDER output layout:
# [type(8) | col(1) | start_y(1) | length(1) | done(1) | mask(N) | col(1) | new_wall(1) | chunk_start(1)]
_C_COL = D_TOKEN_TYPE
_C_START_Y = D_TOKEN_TYPE + 1
_C_LENGTH = D_TOKEN_TYPE + 2
_C_DONE = D_TOKEN_TYPE + 3
_C_FB_OFFSET = D_TOKEN_TYPE + 4  # mask starts here


def test_sort_then_render_chunked():
    """Full autoregressive rollout with chunking: sort 3 walls, then
    iterate columns with multiple chunks per tall wall column."""
    chunk_size = 5
    config = _config(W=32, H=24, fov=8)

    # 3 walls at different distances, with host-computed vis extents
    walls = [
        {"ax": 3.0, "ay": -3.0, "bx": 3.0, "by": 3.0, "tex_id": 0.0},
        {"ax": 6.0, "ay": -6.0, "bx": 6.0, "by": 6.0, "tex_id": 1.0},
        {"ax": 10.0, "ay": -10.0, "bx": 10.0, "by": 10.0, "tex_id": 2.0},
    ]
    # Pre-computed wall visibility (vis_top, vis_bottom) — host-fed
    H = config.screen_height
    center = H / 2.0
    wall_vis = [
        (8.0, 16.0),   # wall 0: height=8, 2 chunks (5+3)
        (10.0, 14.0),  # wall 1: height=4, 1 chunk
        (11.0, 13.0),  # wall 2: height=2, 1 chunk
    ]
    # Column ranges from reference renderer
    col_ranges = {}
    for i, w in enumerate(walls):
        seg = Segment(ax=w["ax"], ay=w["ay"], bx=w["bx"], by=w["by"],
                      color=(1, 0, 0))
        rng = _vis_cols(seg, 0, 0, 0, config)
        if rng is not None:
            col_ranges[i] = rng

    max_walls = len(walls)
    N = len(walls)
    sort_order = _expected_sort_order(0, 0, walls)

    output_node, pos_encoding = build_sort_render_chunked_graph(max_walls, chunk_size)
    compiled = compile_headless(
        output_node, pos_encoding,
        d=512, d_head=32, max_layers=200, verbose=False,
    )

    past = compiled.empty_past()
    step = 0

    # --- Prefill: WALL tokens ---
    for i, w in enumerate(walls):
        lo, hi = col_ranges.get(i, (0, 0))
        vt, vb = wall_vis[i]
        row = _build_step_row(
            compiled, E8_WALL, max_walls,
            px=0.0, py=0.0,
            ax=w["ax"], ay=w["ay"], bx=w["bx"], by=w["by"],
            tex_id=w["tex_id"],
            col_lo=float(lo), col_hi=float(hi),
            vis_top=vt, vis_bottom=vb,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1

    # --- Sort phase ---
    sort_mask = np.zeros(max_walls)
    sorted_wall_indices = []
    sorted_col_ranges = []
    sorted_vis = []

    for k in range(N):
        row = _build_step_row(
            compiled, E8_SORTED_WALL, max_walls,
            px=0.0, py=0.0, sort_mask=sort_mask,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1

        onehot_sl = slice(_SORT_ONEHOT_OFFSET, _SORT_ONEHOT_OFFSET + max_walls)
        onehot = out[0, onehot_sl].detach().cpu().numpy()
        selected_idx = int(np.argmax(np.round(onehot)))
        sorted_wall_indices.append(selected_idx)
        sorted_col_ranges.append(col_ranges.get(selected_idx, (0, 0)))
        sorted_vis.append(wall_vis[selected_idx])
        sort_mask = np.maximum(sort_mask, np.round(onehot))

    assert sorted_wall_indices == sort_order

    # --- Build expected output sequence ---
    expected = []  # list of (col, start_y, length)
    for wall_k in range(N):
        lo, hi = sorted_col_ranges[wall_k]
        vt, vb = sorted_vis[wall_k]
        vis_height = vb - vt
        for col in range(lo, hi):
            y = vt
            while y < vb:
                length = min(chunk_size, vb - y)
                expected.append((col, y, length))
                y += chunk_size

    # --- Chunked render iteration ---
    render_mask = np.zeros(max_walls)
    is_new = 1.0
    col_val = 0.0
    chunk_start_val = -1.0
    emitted = []

    fb_off = _C_FB_OFFSET
    max_steps = len(expected) + 20

    for k in range(max_steps):
        row = _build_step_row(
            compiled, E8_RENDER, max_walls,
            px=0.0, py=0.0,
            render_mask=render_mask,
            render_col=col_val,
            render_is_new_wall=is_new,
            render_chunk_start=chunk_start_val,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step)
        step += 1

        got_col = out[0, _C_COL].item()
        got_start = out[0, _C_START_Y].item()
        got_len = out[0, _C_LENGTH].item()
        done = out[0, _C_DONE].item()
        emitted.append((round(got_col), round(got_start), round(got_len)))

        # Read feedback: [mask(N), col(1), is_new_wall(1), chunk_start(1)]
        fb = out[0, fb_off:fb_off + max_walls + 3].detach().cpu().numpy()
        render_mask = np.round(fb[:max_walls]).clip(0, 1)
        col_val = float(fb[max_walls])
        is_new = float(fb[max_walls + 1])
        chunk_start_val = float(fb[max_walls + 2])

        if done > 0.0:
            break

    assert emitted == expected, (
        f"Mismatch at step {len(emitted)}.\n"
        f"Got:      {emitted[:10]}{'...' if len(emitted)>10 else ''}\n"
        f"Expected: {expected[:10]}{'...' if len(expected)>10 else ''}\n"
        f"Total got={len(emitted)}, expected={len(expected)}"
    )
