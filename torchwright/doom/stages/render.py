"""RENDER stage: per-column wall selection + height + texture + pixel fill.

At each RENDER token (one per screen column, per patch row band) the
graph:

1. Attends over SORTED positions to pick the nearest **visible** wall
   for this column (visibility mask dot-producted against a column
   one-hot).  The wall's packed render precomputation + tex_id come
   back as the attention value.
2. Computes wall height from the precomputed ``H_inv`` and the
   per-column horizontal angle offset.
3. Computes the texture u-coordinate from the precomputed (D, E)
   rotated-frame offsets.
4. Attends over TEX_COL positions to fetch the ``tex_h * 3`` pixel
   column matching this wall's (tex_id, column).
5. Calls ``_textured_column_fill`` to paint the final patch-row pixels.
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    bool_to_01,
    clamp,
    floor_int,
    multiply_const,
    piecewise_linear,
    piecewise_linear_2d,
    subtract,
    thermometer_floor_div,
)
from torchwright.ops.attention_ops import attend_argmax_dot
from torchwright.ops.logic_ops import cond_gate
from torchwright.ops.map_select import in_range
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import DIFF_BP, TEX_E8_OFFSET
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.renderer import _textured_column_fill


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class RenderInputs:
    # Host-fed at RENDER positions.
    col_idx: Node             # which screen column this token paints
    patch_idx: Node           # which vertical patch band
    texture_id_e8: Node       # host-fed (at TEX_COL positions)
    tex_pixels: Node          # host-fed (at TEX_COL positions)

    # Token-type flags.
    is_render: Node
    is_sorted: Node
    is_tex_col: Node

    # Outputs of the SORTED stage (per-SORTED-position Nodes, read via attention).
    gated_render_data: Node
    gated_vis_mask: Node

    # Output of the TEX_COL stage (per-TEX_COL-position one-hot).
    tc_onehot_01: Node

    pos_encoding: PosEncoding


@dataclass
class RenderOutputs:
    pixels: Node              # rp * 3 floats, fills this column's pixel strip


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_render(
    inputs: RenderInputs,
    config: RenderConfig,
    textures: List[np.ndarray],
    rows_per_patch: int,
    max_coord: float,
) -> RenderOutputs:
    H = config.screen_height
    W = config.screen_width
    fov = config.fov_columns
    tex_w, tex_h = textures[0].shape[0], textures[0].shape[1]

    with annotate("render/wall_attention"):
        angle_offset, col_onehot_01 = _compute_column_features(
            inputs.col_idx, W=W, fov=fov,
        )
        r_sort_den, r_C, r_D, r_E, r_H_inv, r_wall_tex = _attend_to_wall(
            inputs.pos_encoding,
            is_render=inputs.is_render,
            is_sorted=inputs.is_sorted,
            col_onehot_01=col_onehot_01,
            gated_vis_mask=inputs.gated_vis_mask,
            gated_render_data=inputs.gated_render_data,
        )

    with annotate("render/wall_height"):
        tan_o, tan_val_bp = _compute_angle_offset_tan(angle_offset, fov=fov)
        den_over_cos, abs_den_over_cos = _compute_den_over_cos(
            r_sort_den, r_C, tan_o, tan_val_bp,
        )
        wall_top, wall_bottom, wall_height = _compute_wall_height(
            r_H_inv, abs_den_over_cos, H=H, max_coord=max_coord,
        )

    with annotate("render/tex_coord"):
        tex_col_idx = _compute_texture_column(
            r_D, r_E, tan_o, tan_val_bp,
            abs_den_over_cos, max_coord=max_coord, tex_w=tex_w,
        )

    with annotate("render/tex_attention"):
        tex_column_colors = _attend_to_texture_column(
            inputs.pos_encoding,
            is_render=inputs.is_render,
            is_tex_col=inputs.is_tex_col,
            r_wall_tex=r_wall_tex,
            tex_col_idx=tex_col_idx,
            tc_onehot_01=inputs.tc_onehot_01,
            texture_id_e8=inputs.texture_id_e8,
            tex_pixels=inputs.tex_pixels,
            num_tex=len(textures),
            tex_w=tex_w,
        )

    with annotate("render/column_fill"):
        patch_row_start = multiply_const(inputs.patch_idx, float(rows_per_patch))
        pixels = _textured_column_fill(
            wall_top, wall_bottom, wall_height,
            tex_column_colors, tex_h, config, max_coord=max_coord,
            patch_row_start=patch_row_start, rows_per_patch=rows_per_patch,
        )

    return RenderOutputs(pixels=pixels)


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _compute_column_features(col_idx: Node, *, W: int, fov: int):
    """Per-column horizontal angle offset + column one-hot for wall attention."""
    col_times_fov = multiply_const(col_idx, float(fov))
    ao_raw = thermometer_floor_div(col_times_fov, W, fov * (W - 1))
    angle_offset = add_const(ao_raw, float(-(fov // 2)))

    col_p1 = add_const(col_idx, 1.0)
    col_onehot_01 = bool_to_01(in_range(col_idx, col_p1, W))
    return angle_offset, col_onehot_01


def _attend_to_wall(
    pos_encoding: PosEncoding,
    *,
    is_render: Node,
    is_sorted: Node,
    col_onehot_01: Node,
    gated_vis_mask: Node,
    gated_render_data: Node,
):
    """Argmax-dot attention to pick this column's visible wall.

    Query: column one-hot (only nonzero at RENDER positions).
    Key:   wall visibility mask + SORT_BIAS on is_sorted so ties break
           toward SORTED positions.
    """
    VIS_GAIN = 500.0
    SORT_BIAS = 100.0
    render_attn = attend_argmax_dot(
        pos_encoding,
        query_vector=Concatenate([
            cond_gate(is_render, col_onehot_01),
            bool_to_01(is_render),
        ]),
        key_vector=Concatenate([
            gated_vis_mask,
            multiply_const(bool_to_01(is_sorted), SORT_BIAS),
        ]),
        value=gated_render_data,
        match_gain=VIS_GAIN,
    )
    r_sort_den = extract_from(render_attn, 6, 0, 1, "r_sort_den")
    r_C = extract_from(render_attn, 6, 1, 1, "r_C")
    r_D = extract_from(render_attn, 6, 2, 1, "r_D")
    r_E = extract_from(render_attn, 6, 3, 1, "r_E")
    r_H_inv = extract_from(render_attn, 6, 4, 1, "r_H_inv")
    r_wall_tex = extract_from(render_attn, 6, 5, 1, "r_tex")
    return r_sort_den, r_C, r_D, r_E, r_H_inv, r_wall_tex


def _compute_angle_offset_tan(angle_offset: Node, *, fov: int):
    """tan(angle_offset) lookup, plus its breakpoints for the 2D product grids."""
    half_fov = fov // 2
    tan_bp = [float(i) for i in range(-half_fov, half_fov + 1)]
    tan_o = piecewise_linear(
        angle_offset, tan_bp,
        lambda x: math.tan(x * 2.0 * math.pi / 256.0),
        name="tan_offset",
    )
    max_tan = math.tan(half_fov * 2.0 * math.pi / 256.0) * 1.1
    tan_val_bp = [-max_tan + i * (2 * max_tan / 10) for i in range(11)]
    return tan_o, tan_val_bp


def _compute_den_over_cos(
    r_sort_den: Node,
    r_C: Node,
    tan_o: Node,
    tan_val_bp,
):
    """den/cos = sort_den - C*tan(offset) — the horizontal projection factor."""
    C_tan = piecewise_linear_2d(
        r_C, tan_o, DIFF_BP, tan_val_bp,
        lambda a, b: a * b, name="C_tan_o",
    )
    den_over_cos = subtract(r_sort_den, C_tan)
    abs_den_over_cos = abs(den_over_cos)
    return den_over_cos, abs_den_over_cos


def _compute_wall_height(
    r_H_inv: Node,
    abs_den_over_cos: Node,
    *,
    H: int,
    max_coord: float,
):
    """Wall height = H_inv_num_t * |den/cos|, clamped; vertical span centered on H/2."""
    max_h_inv = float(H) / 0.3
    h_inv_n = 16
    h_inv_ratio = (max_h_inv / 0.01) ** (1.0 / (h_inv_n - 1))
    height_inv_bp = [0.01 * (h_inv_ratio ** k) for k in range(h_inv_n)]
    height_inv_bp[0] = 0.0
    height_inv_bp[-1] = max_h_inv

    doc_max = 2.5 * max_coord
    doc_bp = [doc_max * i / 15 for i in range(16)]

    wall_height_raw = piecewise_linear_2d(
        r_H_inv, abs_den_over_cos,
        height_inv_bp, doc_bp,
        lambda a, b: a * b, name="wall_height_raw",
    )
    wall_height = clamp(wall_height_raw, 0.0, float(H))

    center = float(H) / 2.0
    half_height = multiply_const(wall_height, 0.5)
    wall_top = Linear(
        half_height, torch.tensor([[-1.0]]),
        torch.tensor([center]), name="wall_top",
    )
    wall_bottom = Linear(
        half_height, torch.tensor([[1.0]]),
        torch.tensor([center]), name="wall_bottom",
    )
    return wall_top, wall_bottom, wall_height


def _compute_texture_column(
    r_D: Node,
    r_E: Node,
    tan_o: Node,
    tan_val_bp,
    abs_den_over_cos: Node,
    *,
    max_coord: float,
    tex_w: int,
) -> Node:
    """Texture u-coordinate via (D + E*tan(offset)) / (den/cos), mapped to column index."""
    E_tan = piecewise_linear_2d(
        r_E, tan_o, DIFF_BP, tan_val_bp,
        lambda a, b: a * b, name="E_tan_o",
    )
    num_u_over_cos = add(r_D, E_tan)
    abs_nuc = abs(num_u_over_cos)

    doc_max = 2.5 * max_coord
    doc_bp = [doc_max * i / 15 for i in range(16)]

    u_raw = piecewise_linear_2d(
        abs_nuc, abs_den_over_cos,
        doc_bp, doc_bp,
        lambda n, d: n / d if d > 0.01 else 0.0,
        name="u_ratio",
    )
    tex_col_float = multiply_const(u_raw, float(tex_w))
    tex_col_clamped = clamp(tex_col_float, 0.0, float(tex_w) - 0.5)
    return floor_int(tex_col_clamped, 0, tex_w - 1)


def _attend_to_texture_column(
    pos_encoding: PosEncoding,
    *,
    is_render: Node,
    is_tex_col: Node,
    r_wall_tex: Node,
    tex_col_idx: Node,
    tc_onehot_01: Node,
    texture_id_e8: Node,
    tex_pixels: Node,
    num_tex: int,
    tex_w: int,
) -> Node:
    """Argmax-dot attention from RENDER → TEX_COL matching (tex_id, col) keys."""
    tex_e8_query = piecewise_linear(
        r_wall_tex,
        [float(i) for i in range(num_tex)],
        lambda tid: [float(v) for v in
                     index_to_vector(int(round(tid)) + TEX_E8_OFFSET)],
        name="tex_id_to_e8",
    )
    tex_col_p1 = add_const(tex_col_idx, 1.0)
    rc_onehot_01 = bool_to_01(in_range(tex_col_idx, tex_col_p1, tex_w))

    COL_SCALE = 10.0
    TEX_MATCH_GAIN = 1000.0
    scaled_rc = multiply_const(rc_onehot_01, COL_SCALE)
    scaled_tc = multiply_const(tc_onehot_01, COL_SCALE)
    return attend_argmax_dot(
        pos_encoding,
        query_vector=cond_gate(
            is_render, Concatenate([tex_e8_query, scaled_rc])),
        key_vector=cond_gate(
            is_tex_col, Concatenate([texture_id_e8, scaled_tc])),
        value=cond_gate(is_tex_col, tex_pixels),
        match_gain=TEX_MATCH_GAIN,
    )
