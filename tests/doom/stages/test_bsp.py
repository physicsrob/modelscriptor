"""Unit tests for the BSP stage (torchwright.doom.stages.bsp).

At BSP_NODE[i], the stage emits the player's side_P decision (+1/0)
into slot ``i`` of a broadcast vector.  ``attend_mean_where`` averages
over all M BSP_NODE positions; multiplying by M recovers the dense
0/1 side vector visible at every position.

These tests feed multi-position sequences where some positions are
BSP_NODE tokens (with hand-crafted planes) and others are receivers.
Expected side_P values are derived directly from ``nx*px + ny*py + d``.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright.doom.stages.bsp import BspInputs, build_bsp

_MAX_COORD = 20.0
_MAX_BSP_NODES = 4


@pytest.fixture(scope="module")
def bsp_module():
    """Compile just build_bsp's side_P_vec output.

    All five host-fed inputs (player_x, player_y, plane_nx, plane_ny,
    plane_d, bsp_node_id_onehot, is_bsp_node) are exposed as
    ``create_input`` so the tests can set them per-position.
    """
    pos = create_pos_encoding()
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    bsp_plane_nx = create_input("bsp_plane_nx", 1)
    bsp_plane_ny = create_input("bsp_plane_ny", 1)
    bsp_plane_d = create_input("bsp_plane_d", 1)
    bsp_node_id_onehot = create_input("bsp_node_id_onehot", _MAX_BSP_NODES)
    is_bsp_node = create_input("is_bsp_node", 1)

    out = build_bsp(
        BspInputs(
            player_x=player_x,
            player_y=player_y,
            bsp_plane_nx=bsp_plane_nx,
            bsp_plane_ny=bsp_plane_ny,
            bsp_plane_d=bsp_plane_d,
            bsp_node_id_onehot=bsp_node_id_onehot,
            is_bsp_node=is_bsp_node,
            pos_encoding=pos,
        ),
        max_coord=_MAX_COORD,
        max_bsp_nodes=_MAX_BSP_NODES,
    )
    return compile_headless(
        out.side_P_vec,
        pos,
        d=1024,
        d_head=16,
        max_layers=40,
        verbose=False,
    )


def _pack(module, rows: list[dict]) -> torch.Tensor:
    d_input = max(s + w for _, s, w in module._input_specs)
    T = len(rows)
    t = torch.zeros(T, d_input, dtype=torch.float32)
    for i, row in enumerate(rows):
        for name, start, width in module._input_specs:
            t[i, start : start + width] = torch.tensor(
                row[name],
                dtype=torch.float32,
            ).reshape(width)
    return t


def _bsp_row(i: int, nx: float, ny: float, d: float, px: float, py: float) -> dict:
    """One BSP_NODE row with onehot slot i and plane (nx, ny, d).

    player_x/y must be set here because the plane classification
    (multiply_2d(plane_nx, player_x, ...)) reads them at the BSP_NODE
    position, not at the receiver.
    """
    oh = [1.0 if k == i else 0.0 for k in range(_MAX_BSP_NODES)]
    return {
        "is_bsp_node": 1.0,  # +1 = valid at this BSP_NODE position
        "bsp_plane_nx": nx,
        "bsp_plane_ny": ny,
        "bsp_plane_d": d,
        "bsp_node_id_onehot": oh,
        "player_x": px,
        "player_y": py,
    }


def _receiver_row(px: float, py: float) -> dict:
    """A non-BSP_NODE receiver position.  Not used for the broadcast math."""
    return {
        "is_bsp_node": -1.0,  # -1 = not a BSP_NODE (matches equals_vector convention)
        "bsp_plane_nx": 0.0,
        "bsp_plane_ny": 0.0,
        "bsp_plane_d": 0.0,
        "bsp_node_id_onehot": [0.0] * _MAX_BSP_NODES,
        "player_x": px,
        "player_y": py,
    }


# ---------------------------------------------------------------------------
# side_P decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "px,py,planes,expected_sides",
    [
        # One plane x=0 (nx=1, ny=0, d=0).  Player at (3, 0) → FRONT → side_P[0]=1.
        (3.0, 0.0, [(1.0, 0.0, 0.0)], [1.0, 0.0, 0.0, 0.0]),
        # Same plane, player at (-3, 0) → BACK → side_P[0]=0.
        (-3.0, 0.0, [(1.0, 0.0, 0.0)], [0.0, 0.0, 0.0, 0.0]),
        # Two planes: x=0 and y=0.  Player at (3, 2) → FRONT of both.
        (3.0, 2.0, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [1.0, 1.0, 0.0, 0.0]),
        # Three planes, player on various sides.
        # plane 0: x-5 → raw = 1*3 + 0*2 + (-5) = -2 → BACK → 0
        # plane 1: y+5 → raw = 0 + 1*2 + 5 = 7 → FRONT → 1
        # plane 2: x+y → raw = 3 + 2 + 0 = 5 → FRONT → 1
        (
            3.0,
            2.0,
            [(1.0, 0.0, -5.0), (0.0, 1.0, 5.0), (1.0, 1.0, 0.0)],
            [0.0, 1.0, 1.0, 0.0],
        ),
    ],
)
def test_side_P_vec_matches_plane_classification(
    bsp_module,
    px,
    py,
    planes,
    expected_sides,
):
    """side_P_vec[i] must equal 1 iff nx_i*px + ny_i*py + d_i > 0.

    The recovery ``side_P_vec = attend_mean_where(...) * max_bsp_nodes``
    assumes the host always supplies exactly ``max_bsp_nodes`` BSP_NODE
    tokens — padding with nx=ny=d=0 (which always classifies as BACK)
    for unused slots.  We follow that convention here.
    """
    assert len(planes) <= _MAX_BSP_NODES, "test setup exceeds max_bsp_nodes"

    rows = [
        _bsp_row(i, nx, ny, d, px=px, py=py) for i, (nx, ny, d) in enumerate(planes)
    ]
    # Pad remaining slots with planes that classify unambiguously as BACK
    # (nx=1, ny=0, d=-99 → raw = px - 99 ≤ -79 across the whole map).
    # Using raw=0 planes is fragile because PL2D multiply fuzz can flip
    # the sign of the near-zero compare.
    for i in range(len(planes), _MAX_BSP_NODES):
        rows.append(_bsp_row(i, 1.0, 0.0, -99.0, px=px, py=py))
    rows.append(_receiver_row(px, py))

    inputs = _pack(bsp_module, rows)
    with torch.no_grad():
        out = bsp_module(inputs)
    receiver_idx = len(rows) - 1
    side_P_at_receiver = out[receiver_idx].tolist()
    for i, expected in enumerate(expected_sides):
        actual = side_P_at_receiver[i]
        assert abs(actual - expected) < 0.2, (
            f"slot {i}: side_P={actual:+.2f}, expected {expected:+.1f} "
            f"(player=({px},{py}), planes={planes})"
        )
