"""SORTED stage: autoregressive argmin sort + per-wall visibility mask.

At each SORTED_WALL token the graph:

1. Runs ``attend_argmin_unmasked`` over the WALL positions to pick the
   nearest wall not yet picked (the host feeds the running ``prev_mask``
   back as input each step).
2. Unpacks the WALL payload into geometry + render precomputation +
   position one-hot, updates the running mask, and gates the render
   block to zero at non-SORTED positions for downstream attention.
3. Computes a **visibility mask** — the range of screen columns that
   the selected wall subtends from the resolved player pose.  Wall
   endpoints are rotated into the player's angular frame; tangent of
   the angle maps through ``piecewise_linear`` to a screen column.
"""

import math
from dataclasses import dataclass

import torch

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    clamp,
    compare,
    negate,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    signed_multiply,
    subtract,
)
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import cond_gate
from torchwright.ops.map_select import in_range, select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import ATAN_BP, DIFF_BP, TRIG_BP
from torchwright.doom.renderer import trig_lookup
from torchwright.doom.wall_payload import (
    extract_geometry_field,
    unpack_wall_payload,
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class SortedInputs:
    # WALL-stage outputs (per-WALL-position Nodes read via attention).
    sort_score: Node
    position_onehot: Node
    sort_value: Node

    # Host-fed running mask of already-picked walls.
    prev_mask: Node

    # Resolved player state broadcast from EOS.
    eos_px: Node
    eos_py: Node
    eos_angle: Node

    # Token-type flag (1.0 at SORTED_WALL positions).
    is_sorted: Node

    pos_encoding: PosEncoding


@dataclass
class SortedOutputs:
    # Per-SORTED payload pieces emitted by the orchestrator.
    sel_wall_data: Node       # ax, ay, bx, by, tex_id (5-wide)
    sel_onehot: Node          # per-position position_onehot of picked wall
    updated_mask: Node        # prev_mask + sel_onehot, fed back by host

    # Consumed by the RENDER stage via attend_argmax_dot.
    gated_render_data: Node   # [sort_den, C, D, E, H_inv, tex_id] gated to 0 off-SORTED
    gated_vis_mask: Node      # per-column boolean mask gated to 0 off-SORTED


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_sorted(
    inputs: SortedInputs,
    config: RenderConfig,
    max_walls: int,
    max_coord: float,
) -> SortedOutputs:
    with annotate("sort/attention"):
        (
            sel_wall_data,
            sel_render,
            sel_onehot,
            sel_tex_id,
            updated_mask,
            gated_render_data,
        ) = _argmin_and_unpack(inputs, max_walls)

    with annotate("sort/visibility"):
        gated_vis_mask = _compute_visibility_mask(
            sel_wall_data,
            inputs.eos_px, inputs.eos_py, inputs.eos_angle,
            inputs.is_sorted,
            config=config,
            max_coord=max_coord,
        )

    return SortedOutputs(
        sel_wall_data=sel_wall_data,
        sel_onehot=sel_onehot,
        updated_mask=updated_mask,
        gated_render_data=gated_render_data,
        gated_vis_mask=gated_vis_mask,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _argmin_and_unpack(inputs: SortedInputs, max_walls: int):
    """Pick the nearest unmasked wall and split its payload for downstream use."""
    selected_sort = attend_argmin_unmasked(
        pos_encoding=inputs.pos_encoding,
        score=inputs.sort_score,
        mask_vector=inputs.prev_mask,
        position_onehot=inputs.position_onehot,
        value=inputs.sort_value,
    )
    unpacked = unpack_wall_payload(selected_sort, max_walls)
    sel_wall_data = unpacked.wall_data
    sel_render = unpacked.render_data
    sel_onehot = unpacked.onehot
    sel_tex_id = extract_geometry_field(sel_wall_data, "tex_id")
    updated_mask = add(inputs.prev_mask, sel_onehot)

    # Gate sorted values so non-SORTED positions contribute 0 to the RENDER
    # attention that reads these fields.
    gated_render_data = cond_gate(
        inputs.is_sorted, Concatenate([sel_render, sel_tex_id])
    )
    return sel_wall_data, sel_render, sel_onehot, sel_tex_id, updated_mask, gated_render_data


def _compute_visibility_mask(
    sel_wall_data: Node,
    eos_px: Node,
    eos_py: Node,
    eos_angle: Node,
    is_sorted: Node,
    *,
    config: RenderConfig,
    max_coord: float,
) -> Node:
    """Column range subtended by the selected wall from the player's pose.

    Wall endpoints A, B are translated into the player's frame, then
    ``tan(angle) = cross / dot`` is mapped to a screen column via the
    camera's FOV.  The (lo, hi) columns are clamped and turned into a
    boolean in_range mask over ``[0, W)``.
    """
    W = config.screen_width
    fov = config.fov_columns

    sel_ax = extract_geometry_field(sel_wall_data, "ax")
    sel_ay = extract_geometry_field(sel_wall_data, "ay")
    sel_bx = extract_geometry_field(sel_wall_data, "bx")
    sel_by = extract_geometry_field(sel_wall_data, "by")

    dax = subtract(sel_ax, eos_px)
    day = subtract(sel_ay, eos_py)
    dbx = subtract(sel_bx, eos_px)
    dby = subtract(sel_by, eos_py)

    sort_cos, sort_sin = trig_lookup(eos_angle)

    cross_a, dot_a = _rotate_into_player_frame(sort_cos, sort_sin, dax, day, "a")
    cross_b, dot_b = _rotate_into_player_frame(sort_cos, sort_sin, dbx, dby, "b")

    tan_a = _tan_from_cross_dot(cross_a, dot_a, max_coord, suffix="")
    tan_b = _tan_from_cross_dot(cross_b, dot_b, max_coord, suffix="_b")

    col_a = _tan_to_screen_column(tan_a, W=W, fov=fov, name="col_a")
    col_b = _tan_to_screen_column(tan_b, W=W, fov=fov, name="col_b")

    col_a_c = clamp(col_a, -2.0, float(W + 2))
    col_b_c = clamp(col_b, -2.0, float(W + 2))

    a_lt_b = compare(subtract(col_b_c, col_a_c), 0.0)
    vis_lo = select(a_lt_b, col_a_c, col_b_c)
    vis_hi = select(a_lt_b, col_b_c, col_a_c)

    vis_mask = in_range(vis_lo, vis_hi, W)
    return cond_gate(is_sorted, vis_mask)


def _rotate_into_player_frame(cos_p: Node, sin_p: Node, dx: Node, dy: Node, suffix: str):
    """Return (cross, dot) of a 2D point with the player's facing vector.

    ``cross = cos*dy - sin*dx``  (left-of-facing magnitude)
    ``dot   = cos*dx + sin*dy``  (forward projection)
    """
    cross = subtract(
        piecewise_linear_2d(cos_p, dy, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"cos_dy_{suffix}"),
        piecewise_linear_2d(sin_p, dx, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"sin_dx_{suffix}"),
    )
    dot = add(
        piecewise_linear_2d(cos_p, dx, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"cos_dx_{suffix}"),
        piecewise_linear_2d(sin_p, dy, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"sin_dy_{suffix}"),
    )
    return cross, dot


def _tan_from_cross_dot(cross: Node, dot: Node, max_coord: float, suffix: str) -> Node:
    """Compute signed ``cross / max(|dot|, 0.1)`` via reciprocal + signed_multiply.

    Clamping ``|dot|`` away from zero avoids divide-by-zero when the
    point is nearly perpendicular to the facing direction.
    """
    dot_sign = compare(dot, 0.0)
    dot_abs = abs(dot)
    dot_clamped = select(
        compare(dot_abs, 0.1),
        dot_abs,
        create_literal_value(torch.tensor([0.1]), name=f"dot_min{suffix}"),
    )
    inv_dot = reciprocal(dot_clamped, min_value=0.1, max_value=2.0 * max_coord)
    signed_inv = select(dot_sign, inv_dot, negate(inv_dot))
    return signed_multiply(
        cross, signed_inv,
        max_abs1=max_coord, max_abs2=1.0 / 0.1,
        step=0.5, max_abs_output=20.0,
    )


def _tan_to_screen_column(tan: Node, *, W: int, fov: int, name: str) -> Node:
    """Map a ``tan(angle)`` to a floating-point screen column via atan + FOV scaling."""
    fov_rad = float(fov) * math.pi / 128.0
    col_from_tan_scale = float(W) / fov_rad
    return piecewise_linear(
        tan, ATAN_BP,
        lambda t: math.atan(t) * col_from_tan_scale + W / 2.0,
        name=name,
    )
