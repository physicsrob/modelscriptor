"""THINKING stage: select the next wall to render, seed RENDER iteration.

At each THINKING token the graph runs ``attend_argmin_unmasked`` over
SORTED positions.  The score is ``sel_bsp_rank`` (the BSP rank of the
wall selected at that SORTED step); the mask is ``render_mask`` (walls
already fully rendered).  The position key is ``sel_onehot`` (the
wall-index one-hot), so both mask spaces align in wall-index space.

The attention's **value** is the full wall payload plus visibility
column bounds plus the per-wall one-hot needed to advance the render
mask:

    value = [sort_den, C, D, E, H_inv, tex_id, vis_lo, vis_hi,
             sel_onehot]   # 8 + max_walls wide

Each field is emitted as its own Node so the orchestrator can assemble
the render_feedback vector without extra extracts.

Output contract: the selected wall's fields + one-hot, already gated
by the attention's softmax (values at non-SORTED positions are zero).
"""

from dataclasses import dataclass

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.asserts import assert_01, assert_integer, assert_onehot
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import cond_gate
from torchwright.ops.map_select import select

from torchwright.doom.graph_utils import extract_from

import torch


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class ThinkingInputs:
    # SORTED-stage outputs (per-SORTED-position Nodes read via attention).
    sel_bsp_rank: Node         # score; sentinel replaces at non-SORTED
    sel_onehot: Node           # wall-index position key for argmin
    gated_render_data: Node    # 6-wide wall render data gated to 0 off-SORTED
    vis_lo: Node               # col_lo (float)
    vis_hi: Node               # col_hi (float)

    # Host-fed running mask of walls already fully rendered.
    render_mask: Node          # max_walls-wide

    # Token-type flags.
    is_sorted: Node
    # is_thinking only matters at orchestrator output time, not here

    pos_encoding: PosEncoding


@dataclass
class ThinkingOutputs:
    """Selected wall's fields, ready for the render_feedback seed.

    Names prefixed with ``t_`` so they're distinguishable from the
    per-position wall-data fields during orchestrator assembly.
    """

    t_sort_den: Node
    t_C: Node
    t_D: Node
    t_E: Node
    t_H_inv: Node
    t_tex_id: Node
    t_col_lo: Node
    t_col_hi: Node
    t_onehot: Node             # max_walls-wide


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_thinking(inputs: ThinkingInputs, max_walls: int) -> ThinkingOutputs:
    with annotate("thinking/wall_attention"):
        # Score: sel_bsp_rank at SORTED positions, sentinel elsewhere so
        # the argmin only considers SORTED candidates.
        render_sentinel = create_literal_value(
            torch.tensor([99.0]), name="render_sentinel",
        )
        render_score = assert_integer(
            select(inputs.is_sorted, inputs.sel_bsp_rank, render_sentinel)
        )

        # Position key: sel_onehot (wall-index) at SORTED, zeros elsewhere.
        z_mw = create_literal_value(
            torch.zeros(max_walls), name="z_mw_thinking",
        )
        render_position_onehot = assert_onehot(select(
            inputs.is_sorted, inputs.sel_onehot, z_mw,
        ))

        # Value: render data + col bounds + one-hot for downstream mask update.
        render_value = Concatenate([
            inputs.gated_render_data,  # 6
            inputs.vis_lo,             # 1
            inputs.vis_hi,             # 1
            inputs.sel_onehot,         # max_walls
        ])
        render_value_gated = cond_gate(inputs.is_sorted, render_value)

        selected_render = attend_argmin_unmasked(
            pos_encoding=inputs.pos_encoding,
            score=render_score,
            mask_vector=assert_01(inputs.render_mask),
            position_onehot=render_position_onehot,
            value=render_value_gated,
        )

        d_rv = 8 + max_walls
        t_sort_den = extract_from(selected_render, d_rv, 0, 1, "t_sort_den")
        t_C        = extract_from(selected_render, d_rv, 1, 1, "t_C")
        t_D        = extract_from(selected_render, d_rv, 2, 1, "t_D")
        t_E        = extract_from(selected_render, d_rv, 3, 1, "t_E")
        t_H_inv    = extract_from(selected_render, d_rv, 4, 1, "t_H_inv")
        t_tex_id   = extract_from(selected_render, d_rv, 5, 1, "t_tex_id")
        t_col_lo   = extract_from(selected_render, d_rv, 6, 1, "t_col_lo")
        t_col_hi   = extract_from(selected_render, d_rv, 7, 1, "t_col_hi")
        t_onehot   = extract_from(selected_render, d_rv, 8, max_walls, "t_onehot")

    return ThinkingOutputs(
        t_sort_den=t_sort_den,
        t_C=t_C,
        t_D=t_D,
        t_E=t_E,
        t_H_inv=t_H_inv,
        t_tex_id=t_tex_id,
        t_col_lo=t_col_lo,
        t_col_hi=t_col_hi,
        t_onehot=t_onehot,
    )
