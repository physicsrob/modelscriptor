"""Phase 1 reference — pure-Python BSP rank computation.

Computes the canonical BSP rank of every wall segment in a `MapSubset`
given a `GameState`. The sandbox renderer must produce the same
answers through its autoregressive design; this function is the ground
truth the phase test asserts against.

No framework, no PWL, no autoregression — straight Python over the
loaded data.
"""

from __future__ import annotations

from doom_sandbox.types import GameState, MapSubset


def expected_bsp_ranks(scene: MapSubset, state: GameState) -> list[int]:
    """Compute the BSP rank of every wall segment.

    For each segment `s`, the rank is

        rank(s) = round(seg_bsp_coeffs[s] · side_P_vec + seg_bsp_consts[s])

    where `side_P_vec[i] = 1` if the player is on the FRONT side of
    BSP plane `i` (`nx_i · player_x + ny_i · player_y + d_i > 0`),
    else `0`.

    The computation is invariant under `scene.scene_origin` — shifting
    the player position and BSP `d`-coefficients by the same offset
    leaves `side_P` unchanged — so this function operates directly on
    the raw stored values without applying the shift.

    Parameters
    ----------
    scene : MapSubset
        The scene fixture: segments, BSP planes, and precomputed rank
        coefficients.
    state : GameState
        The player's state. Only `state.x` and `state.y` are used.

    Returns
    -------
    list[int]
        BSP rank per segment, in ascending segment-index order. Length
        is `len(scene.segments)`.
    """
    side_P_vec = [
        1 if (node.nx * state.x + node.ny * state.y + node.d) > 0 else 0
        for node in scene.bsp_nodes
    ]

    ranks: list[int] = []
    for s in range(len(scene.segments)):
        coeffs = scene.seg_bsp_coeffs[s]
        const = scene.seg_bsp_consts[s]
        rank_float = sum(c * sp for c, sp in zip(coeffs, side_P_vec)) + const
        ranks.append(round(rank_float))
    return ranks
