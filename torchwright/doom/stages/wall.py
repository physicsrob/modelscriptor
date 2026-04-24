"""WALL stage: produces the per-wall one-hot used for content attention.

After Phase B Part 3 the prefill WALL stage is a thin data carrier.
Raw wall geometry (``wall_ax/ay/bx/by``, ``wall_tex_id``,
``wall_bsp_coeffs``, ``wall_bsp_const``) is supplied by the host at
each WALL position and read directly from ``inputs[...]`` by
``thinking_wall`` and ``render``.  The single responsibility of the
``build_wall`` stage is to materialize the ``wall_index`` into a
one-hot (+ 0.5 bias) so downstream content-attention queries can land
cleanly on the matching wall via dot-product.

The per-wall collision flags, BSP rank, renderability flag, visibility
columns, ``indicators_above`` thermometer, and packed sort-value
payload are all gone:

* Running-OR HIT_FULL/HIT_X/HIT_Y thinking tokens (Phase B Part 1)
  carry the global collision aggregate; RESOLVED reads them via
  readback (Phase B Part 3).
* BSP_RANK and IS_RENDERABLE identifier tokens (Phase A Part 3)
  carry the rank and renderability; SORTED's quadratic-equality
  attention reads them from the thinking KV (Phase B Part 2).
* VIS_LO / VIS_HI identifier tokens (Phase A Part 3) carry the
  visibility columns; RENDER reads them via content attention on
  wall_index (Phase B Part 2).
"""

from dataclasses import dataclass

import torch

from torchwright.graph import Node, annotate
from torchwright.graph.asserts import assert_onehot
from torchwright.ops.arithmetic_ops import add_const, add_scaled_nodes
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.map_select import in_range


@dataclass
class WallKVOutput:
    position_onehot: Node  # per-wall one-hot + 0.5 bias — used by
    # thinking_wall (wall_geom attention key) and
    # RENDER (WALL-geometry attention key).


def build_wall(
    *,
    wall_index: Node,
    max_walls: int,
) -> WallKVOutput:
    with annotate("wall/onehot"):
        position_onehot = _compute_position_onehot(wall_index, max_walls)
    return WallKVOutput(position_onehot=position_onehot)


def _compute_position_onehot(wall_index: Node, max_walls: int) -> Node:
    """One-hot(wall_index) biased by +0.5 so SORTED attention can pick by dot product."""
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    return assert_onehot(add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh))
