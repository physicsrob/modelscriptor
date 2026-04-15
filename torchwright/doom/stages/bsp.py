"""BSP stage: classify player against each splitting plane, broadcast sides.

Each BSP_NODE token carries a normalized splitting plane ``(nx, ny, d)``
such that ``nx*px + ny*py + d > 0`` iff the player is on the FRONT side.
The host pre-normalizes the plane so that ``|nx|, |ny| ≤ 1``.

At BSP_NODE[i], the graph computes that plane's side_P value (1 for
FRONT, 0 for BACK) and spreads it into slot ``i`` of the output vector
via ``bsp_node_id_onehot``.  An ``attend_mean_where`` averages across
all M BSP_NODE positions, and multiplying by M recovers the dense
per-slot 0/1 side decisions.

Output
------
``side_P_vec``: ``max_bsp_nodes``-wide, visible at every position.
``side_P_vec[i] ∈ {0, 1}`` — the side decision for BSP node ``i``.
WALL tokens dot-product this with their precomputed coefficients to
derive a BSP rank (see ``stages/wall.py``).
"""

from dataclasses import dataclass

from torchwright.graph import Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add, compare, multiply_2d, multiply_const,
)
from torchwright.ops.attention_ops import attend_mean_where
from torchwright.ops.logic_ops import cond_gate


@dataclass
class BspInputs:
    """Host-fed BSP values + token-type gate."""

    player_x: Node
    player_y: Node
    bsp_plane_nx: Node
    bsp_plane_ny: Node
    bsp_plane_d: Node
    bsp_node_id_onehot: Node  # max_bsp_nodes-wide
    is_bsp_node: Node
    pos_encoding: PosEncoding


@dataclass
class BspOutputs:
    """Per-position, per-slot BSP side decisions.

    ``side_P_vec`` is ``max_bsp_nodes``-wide with values in ``{0, 1}``
    (with small interpolation noise).  Visible at every position because
    it's produced by an attend_mean_where broadcast.
    """

    side_P_vec: Node


def build_bsp(
    inputs: BspInputs,
    *,
    max_coord: float,
    max_bsp_nodes: int,
) -> BspOutputs:
    """Compute the per-slot side_P vector broadcast.

    Parameters
    ----------
    max_coord:
        Upper bound on ``|player_x|, |player_y|``; sets the dynamic range
        for the ``multiply_2d`` products.
    max_bsp_nodes:
        Width of the ``bsp_node_id_onehot`` slot vector.
    """
    with annotate("bsp/side_p"):
        bsp_nx_px = multiply_2d(
            inputs.bsp_plane_nx, inputs.player_x,
            max_abs1=1.0, max_abs2=max_coord,
            step1=0.1, step2=1.0,
            name="bsp_nx_px",
        )
        bsp_ny_py = multiply_2d(
            inputs.bsp_plane_ny, inputs.player_y,
            max_abs1=1.0, max_abs2=max_coord,
            step1=0.1, step2=1.0,
            name="bsp_ny_py",
        )
        bsp_raw = add(add(bsp_nx_px, bsp_ny_py), inputs.bsp_plane_d)
        # ±1 bool: +1 if raw > 0 (FRONT), -1 if raw ≤ 0 (BACK).
        side_P_bool = compare(bsp_raw, 0.0)
        # At BSP_NODE[i]: emit onehot_i when side=FRONT, zero otherwise.
        # Other token types get a garbage value that attend_mean_where
        # will ignore (validity=is_bsp_node filters to BSP_NODE positions).
        side_P_spread = cond_gate(side_P_bool, inputs.bsp_node_id_onehot)

    with annotate("bsp/broadcast"):
        side_P_mean = attend_mean_where(
            inputs.pos_encoding,
            validity=inputs.is_bsp_node,
            value=side_P_spread,
        )
        # Undo the mean's division by M to recover per-slot 0/1 values.
        side_P_vec = multiply_const(side_P_mean, float(max_bsp_nodes))

    return BspOutputs(side_P_vec=side_P_vec)
