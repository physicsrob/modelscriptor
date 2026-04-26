"""WALL stage: emit per-wall K channels for the quadratic-equality
content attentions.

The prefill WALL stage is a thin data carrier.  Raw wall geometry
(``wall_ax/ay/bx/by``, ``wall_tex_id``, ``wall_bsp_coeffs``,
``wall_bsp_const``) is supplied by the host at each WALL position
and read directly from ``inputs[...]`` by ``thinking_wall`` and
``render``.  Per-wall collision flags, BSP rank, renderability,
visibility columns, and the packed sort-value payload all live
elsewhere now:

* Running-OR HIT_FULL/HIT_X/HIT_Y thinking tokens carry the global
  collision aggregate; RESOLVED reads them via readback.
* BSP_RANK and IS_RENDERABLE identifier tokens carry the rank and
  renderability; SORTED's quadratic-equality attention reads them
  from the thinking KV.
* VIS_LO / VIS_HI identifier tokens carry the visibility columns;
  RENDER reads them via content attention on wall_index.

The stage emits one channel: ``wall_index_neg_sq`` (=
``-wall_index²``), the second K channel of every quadratic-equality
wall_index attention in the graph (``render/wall_geom_attention``
and ``thinking_wall/wall_geom_attention``).  A sentinel pattern at
non-WALL positions is applied at each consumer.  ``wall_index``
itself is the host-fed scalar at WALL positions, used directly by
consumers as the first quad K channel.
"""

from dataclasses import dataclass

from torchwright.graph import Node, annotate
from torchwright.ops.arithmetic_ops import multiply_const, square


@dataclass
class WallKVOutput:
    wall_index_neg_sq: Node  # 1-wide ``-wall_index²`` channel — the second K
    # channel of every quadratic-equality wall_index
    # attention.  ``wall_index`` itself is forwarded
    # unchanged from the host (layer 0) as the first
    # K channel.  Computed via one ``square`` sublayer
    # on the host-fed scalar in ``[0, max_walls-1]``
    # with integer breakpoints (exact at every wall).


def build_wall(
    *,
    wall_index: Node,
    max_walls: int,
) -> WallKVOutput:
    with annotate("wall/wall_index_neg_sq"):
        # K's second channel for every quadratic-equality wall_index
        # match.  ``wall_index`` lands on integers in
        # ``[0, max_walls-1]``; ``square`` on a unit grid is exact at
        # those points and contributes one MLP sublayer.  The
        # ``multiply_const(-1)`` is fold-only.
        wall_index_sq = square(
            wall_index, max_value=float(max_walls - 1), step=1.0
        )
        wall_index_neg_sq = multiply_const(wall_index_sq, -1.0)
    return WallKVOutput(wall_index_neg_sq=wall_index_neg_sq)
