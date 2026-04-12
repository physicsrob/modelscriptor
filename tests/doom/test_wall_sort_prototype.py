"""Wall-sort phase prototype — autoregressive selection sort of wall segments.

De-risks the multi-phase walls-as-tokens renderer architecture by
testing ONLY the sort phase: prefill wall tokens, then autoregressively
emit them in distance-sorted order using ``attend_argmin_unmasked``.

Token protocol
--------------

Every input row carries an 8-column E8 spherical-code token type, plus
type-specific payload fields.  Five token types:

    START (E8 index 0)   — player state (position 0)
    WALL (E8 index 1)    — wall geometry (positions 1..N)
    EOS (E8 index 2)     — end of prefill
    SORTED_WALL (E8 index 3) — autoregressive sort output
    START_RENDER (E8 index 4) — sort complete (optional)

Input row layout (total width = 16 + max_walls):

    token_type    (8)   — E8 spherical code
    player_x      (1)   — START only
    player_y      (1)   — START only
    player_angle  (1)   — START only (unused by sort, reserved)
    wall_ax       (1)   — WALL only
    wall_ay       (1)   — WALL only
    wall_bx       (1)   — WALL only
    wall_by       (1)   — WALL only
    wall_tex_id   (1)   — WALL only
    sort_mask     (N)   — SORTED_WALL only, {0,1} mask of emitted walls

Output row layout (total width = 13 + max_walls):

    token_type    (8)   — E8[SORTED_WALL]
    wall_ax       (1)   — selected wall's ax
    wall_ay       (1)
    wall_bx       (1)
    wall_by       (1)
    wall_tex_id   (1)
    position_onehot (N) — one-hot of selected wall's index

Autoregressive protocol (after EOS):

    Host feeds SORTED_WALL token with current mask → graph returns
    closest unmasked wall's data + one-hot → host updates mask by
    OR-ing in the one-hot → repeat N times.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    add_scaled_nodes,
    multiply_const,
    piecewise_linear,
    square_signed,
    subtract,
)
from torchwright.ops.arithmetic_ops import max as elementwise_max
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.map_select import in_range, select
from torchwright.ops.prefix_ops import prefix_sum


# ---------------------------------------------------------------------------
# Token type enum (E8 indices)
# ---------------------------------------------------------------------------

TOKEN_START = 0
TOKEN_WALL = 1
TOKEN_EOS = 2
TOKEN_SORTED_WALL = 3
TOKEN_START_RENDER = 4

E8_START = index_to_vector(TOKEN_START)
E8_WALL = index_to_vector(TOKEN_WALL)
E8_EOS = index_to_vector(TOKEN_EOS)
E8_SORTED_WALL = index_to_vector(TOKEN_SORTED_WALL)
E8_START_RENDER = index_to_vector(TOKEN_START_RENDER)


# ---------------------------------------------------------------------------
# Input slot layout
# ---------------------------------------------------------------------------

# Slot widths.
D_TOKEN_TYPE = 8
D_PLAYER = 3        # player_x, player_y, player_angle
D_WALL = 5          # wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id

# Input names in alphabetical order (what the compiler expects).
INPUT_NAMES = (
    "player_angle",   # 1
    "player_x",       # 1
    "player_y",       # 1
    "sort_mask",      # max_walls
    "token_type",     # 8
    "wall_ax",        # 1
    "wall_ay",        # 1
    "wall_bx",        # 1
    "wall_by",        # 1
    "wall_tex_id",    # 1
)

# Score sentinel for non-wall positions.  Must be high enough that the
# argmin never picks a non-wall position, but under 100 to fit the
# attend_argmin_unmasked envelope (_MAX_SCORE_UNMASKED_ABS = 100).
_SENTINEL_SCORE = 99.0

# Square breakpoint step for dx², dy².  max_abs=40 / step=1 = 40 breakpoints.
_SQUARE_MAX_ABS = 40.0
_SQUARE_STEP = 1.0

# Sqrt breakpoints for converting dist_sq → distance as the score.
# Distance (not dist_sq) gives much better softmax separation:
# with _QUERY_GAIN=8, a 1-unit distance difference → logit gap of 8 →
# exp(8)≈3000× ratio.  Max score ≈ 56 for max_coord=20, under 100.
_SQRT_BREAKPOINTS = [
    0.0, 0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0,
    64.0, 100.0, 225.0, 400.0, 900.0, 1600.0, 3200.0,
]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_wall_sort_graph(
    max_walls: int,
) -> Tuple[Node, PosEncoding]:
    """Build the wall-sort graph.

    The graph runs at every position.  At WALL positions, it computes
    a distance-based score and a position one-hot.  At SORTED_WALL
    positions, it uses ``attend_argmin_unmasked`` with a host-fed mask
    to find the next closest wall and emits its data.

    Returns ``(output_node, pos_encoding)``.
    """
    pos_encoding = create_pos_encoding()

    # --- Inputs ---
    token_type = create_input("token_type", D_TOKEN_TYPE)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    player_angle = create_input("player_angle", 1)
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    wall_tex_id = create_input("wall_tex_id", 1)
    sort_mask = create_input("sort_mask", max_walls)

    # --- Token type detection ---
    is_start = equals_vector(token_type, E8_START)    # {-1, +1}
    is_wall = equals_vector(token_type, E8_WALL)

    # --- Broadcast player state from START token ---
    # get_prev_value reads (player_x, player_y) from the position
    # where is_start = +1 (position 0).  Every subsequent position
    # receives the same values via causal attention.
    packed_player = Concatenate([player_x, player_y])
    broadcast_player = pos_encoding.get_prev_value(packed_player, is_start)
    # Extract px, py from the 2-wide broadcast result.
    px = Linear(broadcast_player, torch.tensor([[1.0], [0.0]]), name="px")
    py = Linear(broadcast_player, torch.tensor([[0.0], [1.0]]), name="py")

    # --- Wall midpoint distance from player ---
    mid_x = multiply_const(add(wall_ax, wall_bx), 0.5)
    mid_y = multiply_const(add(wall_ay, wall_by), 0.5)
    dx = subtract(mid_x, px)
    dy = subtract(mid_y, py)
    dx_sq = square_signed(dx, max_abs=_SQUARE_MAX_ABS, step=_SQUARE_STEP)
    dy_sq = square_signed(dy, max_abs=_SQUARE_MAX_ABS, step=_SQUARE_STEP)
    dist_sq = add(dx_sq, dy_sq)
    # Use distance (sqrt) as score, not dist_sq.  Distance gives much
    # better softmax separation: for walls at distances 1 and 2.83,
    # dist_sq scores are 1 and 8 (gap 7/64 ≈ 0.1 after scaling),
    # while distance scores are 1 and 2.83 (gap 1.83, × _QUERY_GAIN=8
    # → logit gap 14.6 → exp ratio 2.2M).
    import math as _math
    score_raw = piecewise_linear(
        dist_sq,
        _SQRT_BREAKPOINTS,
        lambda x: _math.sqrt(max(0.0, x)),
        name="dist_score",
    )

    # Sentinel for non-wall positions: the argmin never picks them.
    sentinel = create_literal_value(
        torch.tensor([_SENTINEL_SCORE]), name="sentinel",
    )
    score = select(is_wall, score_raw, sentinel)

    # --- Wall index + position one-hot ---
    # Convert is_wall from {-1, +1} to {0, 1} for prefix_sum.
    is_wall_01 = multiply_const(add_const(is_wall, 1.0), 0.5)

    # prefix_sum: inclusive cumulative sum.  n_stages must satisfy
    # 2^n_stages >= max sequence length.  Max seq = 1 + max_walls + 1
    # + max_walls + 1 = 2*max_walls + 3.  For max_walls=8: 19, need 5.
    n_stages = max(5, math.ceil(math.log2(2 * max_walls + 3)))
    wall_count = prefix_sum(pos_encoding, is_wall_01, n_stages=n_stages)
    wall_index = add_const(wall_count, -1.0)  # 0-indexed

    # One-hot from wall_index.  in_range returns {-1, +1}; convert to {0, 1}.
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_vec = create_literal_value(torch.ones(max_walls), name="ones_oh")
    position_onehot = add_scaled_nodes(0.5, onehot_bool, 0.5, ones_vec)

    # --- Pack wall data for value readout ---
    wall_value = Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,
        position_onehot,
    ])

    # --- Sort: attend_argmin_unmasked ---
    selected = attend_argmin_unmasked(
        pos_encoding=pos_encoding,
        score=score,
        mask_vector=sort_mask,
        position_onehot=position_onehot,
        value=wall_value,
    )

    # --- Output: token_type + selected wall data + selected onehot ---
    sorted_type = create_literal_value(E8_SORTED_WALL, name="sorted_type")
    output = Concatenate([sorted_type, selected])

    return output, pos_encoding


# ---------------------------------------------------------------------------
# Input row builders
# ---------------------------------------------------------------------------


def _d_input(max_walls: int) -> int:
    return D_TOKEN_TYPE + D_PLAYER + D_WALL + max_walls


def _input_dict_for_batch(
    n_pos: int,
    max_walls: int,
) -> Dict[str, torch.Tensor]:
    """Create a zeroed input dict for ``n_pos`` positions."""
    return {
        "token_type": torch.zeros(n_pos, D_TOKEN_TYPE),
        "player_x": torch.zeros(n_pos, 1),
        "player_y": torch.zeros(n_pos, 1),
        "player_angle": torch.zeros(n_pos, 1),
        "wall_ax": torch.zeros(n_pos, 1),
        "wall_ay": torch.zeros(n_pos, 1),
        "wall_bx": torch.zeros(n_pos, 1),
        "wall_by": torch.zeros(n_pos, 1),
        "wall_tex_id": torch.zeros(n_pos, 1),
        "sort_mask": torch.zeros(n_pos, max_walls),
    }


def _set_start(inputs: dict, pos: int, px: float, py: float, angle: float):
    """Fill a START token at position ``pos``."""
    inputs["token_type"][pos] = E8_START
    inputs["player_x"][pos, 0] = px
    inputs["player_y"][pos, 0] = py
    inputs["player_angle"][pos, 0] = angle


def _set_wall(
    inputs: dict, pos: int,
    ax: float, ay: float, bx: float, by: float, tex_id: float,
):
    """Fill a WALL token at position ``pos``."""
    inputs["token_type"][pos] = E8_WALL
    inputs["wall_ax"][pos, 0] = ax
    inputs["wall_ay"][pos, 0] = ay
    inputs["wall_bx"][pos, 0] = bx
    inputs["wall_by"][pos, 0] = by
    inputs["wall_tex_id"][pos, 0] = tex_id


def _set_eos(inputs: dict, pos: int):
    inputs["token_type"][pos] = E8_EOS


def _set_sort(inputs: dict, pos: int, mask: np.ndarray):
    inputs["token_type"][pos] = E8_SORTED_WALL
    inputs["sort_mask"][pos] = torch.tensor(mask, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Reference: numpy distance sort
# ---------------------------------------------------------------------------


def _wall_dists(px: float, py: float, walls: List[dict]) -> np.ndarray:
    """Compute midpoint distances (squared, scaled) for sorting."""
    dists = []
    for w in walls:
        mid_x = (w["ax"] + w["bx"]) / 2.0
        mid_y = (w["ay"] + w["by"]) / 2.0
        dx = mid_x - px
        dy = mid_y - py
        dists.append(dx * dx + dy * dy)
    return np.array(dists)


def _expected_sort_order(px: float, py: float, walls: List[dict]) -> List[int]:
    """Return wall indices in ascending distance order."""
    dists = _wall_dists(px, py, walls)
    return list(np.argsort(dists))


# ---------------------------------------------------------------------------
# Build full input batch for reference_eval (non-autoregressive)
# ---------------------------------------------------------------------------


def _build_full_batch(
    px: float, py: float, angle: float,
    walls: List[dict],
    max_walls: int,
) -> Tuple[Dict[str, torch.Tensor], int, List[int]]:
    """Build the full input batch for ``reference_eval``.

    Pre-computes the correct sort masks at each sort position (using
    the numpy-computed sort order) so the batch can be evaluated in
    one shot without autoregressive feedback.

    Returns ``(inputs, n_pos, expected_sort_order)``.
    """
    N = len(walls)
    assert N <= max_walls
    # Sequence: START, WALL*N, EOS, SORTED_WALL*N
    n_pos = 1 + N + 1 + N
    inputs = _input_dict_for_batch(n_pos, max_walls)

    # Position 0: START
    _set_start(inputs, 0, px, py, angle)

    # Positions 1..N: WALL
    for i, w in enumerate(walls):
        _set_wall(inputs, 1 + i, w["ax"], w["ay"], w["bx"], w["by"], w["tex_id"])

    # Position N+1: EOS
    _set_eos(inputs, 1 + N)

    # Positions N+2..2N+1: SORTED_WALL with pre-computed masks
    sort_order = _expected_sort_order(px, py, walls)
    mask = np.zeros(max_walls)
    for k, wall_idx in enumerate(sort_order):
        pos = 2 + N + k
        _set_sort(inputs, pos, mask)
        # Update mask for next step: set the bit for this wall
        mask[wall_idx] = 1.0

    return inputs, n_pos, sort_order


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Output layout: [token_type (8), wall_data (5), position_onehot (max_walls)]
_OUT_TOKEN_TYPE = slice(0, D_TOKEN_TYPE)
_OUT_WALL_AX = D_TOKEN_TYPE
_OUT_WALL_AY = D_TOKEN_TYPE + 1
_OUT_WALL_BX = D_TOKEN_TYPE + 2
_OUT_WALL_BY = D_TOKEN_TYPE + 3
_OUT_WALL_TEX = D_TOKEN_TYPE + 4


def _out_onehot_slice(max_walls: int) -> slice:
    return slice(D_TOKEN_TYPE + D_WALL, D_TOKEN_TYPE + D_WALL + max_walls)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _four_walls():
    """Four walls at known distinct distances from (0, 0)."""
    return [
        {"ax": 3.0, "ay": 0.0, "bx": 5.0, "by": 0.0, "tex_id": 0.0},   # mid=(4,0), dist²=16
        {"ax": -1.0, "ay": -1.0, "bx": 1.0, "by": -1.0, "tex_id": 1.0}, # mid=(0,-1), dist²=1
        {"ax": 6.0, "ay": 6.0, "bx": 8.0, "by": 6.0, "tex_id": 2.0},   # mid=(7,6), dist²=85
        {"ax": 1.0, "ay": 2.0, "bx": 3.0, "by": 2.0, "tex_id": 3.0},   # mid=(2,2), dist²=8
    ]


def _eight_walls_well_separated():
    """Eight walls with well-separated distances from the origin.

    All midpoints lie on the X axis so distances are exact and easy to
    verify.  Each pair is separated by ≥ 2 units of distance, giving
    a logit gap of ``8 × 2 = 16`` → ``exp(16) ≈ 8.9M×`` ratio.
    """
    # Place midpoints at x = 1, 3, 6, 10, 15, 21, 28, 36 (all on x-axis).
    # Distances from origin ARE the x-coordinates.
    # Gaps: 2, 3, 4, 5, 6, 7, 8 — all ≥ 2.
    mids = [1, 3, 6, 10, 15, 21, 28, 36]
    walls = []
    for i, mx in enumerate(mids):
        walls.append({
            "ax": float(mx - 1), "ay": 0.0,
            "bx": float(mx + 1), "by": 0.0,
            "tex_id": float(i),
        })
    return walls


# ---------------------------------------------------------------------------
# Test 1: oracle matches numpy reference (4 walls)
# ---------------------------------------------------------------------------


def test_oracle_sort_4_walls():
    """Build the full sequence for 4 walls, run ``reference_eval``,
    and verify the sort output at each SORTED_WALL position matches
    the numpy-computed distance order.
    """
    max_walls = 8
    output_node, pos_encoding = build_wall_sort_graph(max_walls)

    px, py, angle = 0.0, 0.0, 0.0
    walls = _four_walls()
    inputs, n_pos, sort_order = _build_full_batch(px, py, angle, walls, max_walls)

    cache = reference_eval(output_node, inputs, n_pos)
    out = cache[output_node]  # (n_pos, d_output)

    N = len(walls)
    for k in range(N):
        sort_pos = 2 + N + k
        expected_wall_idx = sort_order[k]
        expected_wall = walls[expected_wall_idx]

        got_ax = out[sort_pos, _OUT_WALL_AX].item()
        got_ay = out[sort_pos, _OUT_WALL_AY].item()
        got_bx = out[sort_pos, _OUT_WALL_BX].item()
        got_by = out[sort_pos, _OUT_WALL_BY].item()
        got_tex = out[sort_pos, _OUT_WALL_TEX].item()

        assert abs(got_ax - expected_wall["ax"]) < 0.5, (
            f"sort step {k}: ax={got_ax}, expected {expected_wall['ax']} "
            f"(wall {expected_wall_idx})"
        )
        assert abs(got_ay - expected_wall["ay"]) < 0.5, (
            f"sort step {k}: ay={got_ay}, expected {expected_wall['ay']}"
        )
        assert abs(got_bx - expected_wall["bx"]) < 0.5, (
            f"sort step {k}: bx={got_bx}, expected {expected_wall['bx']}"
        )
        assert abs(got_by - expected_wall["by"]) < 0.5, (
            f"sort step {k}: by={got_by}, expected {expected_wall['by']}"
        )
        assert abs(got_tex - expected_wall["tex_id"]) < 0.5, (
            f"sort step {k}: tex_id={got_tex}, expected {expected_wall['tex_id']}"
        )


# ---------------------------------------------------------------------------
# Test 2: oracle matches numpy reference (8 random walls)
# ---------------------------------------------------------------------------


def test_oracle_sort_8_walls_random():
    """Same as test 1 but with 8 randomly-placed walls."""
    max_walls = 8
    output_node, pos_encoding = build_wall_sort_graph(max_walls)

    px, py, angle = 0.0, 0.0, 0.0
    walls = _eight_walls_well_separated()
    inputs, n_pos, sort_order = _build_full_batch(px, py, angle, walls, max_walls)

    cache = reference_eval(output_node, inputs, n_pos)
    out = cache[output_node]

    N = len(walls)
    for k in range(N):
        sort_pos = 2 + N + k
        expected_wall_idx = sort_order[k]
        expected_wall = walls[expected_wall_idx]
        got_ax = out[sort_pos, _OUT_WALL_AX].item()
        got_ay = out[sort_pos, _OUT_WALL_AY].item()
        assert abs(got_ax - expected_wall["ax"]) < 0.5, (
            f"sort step {k}: ax mismatch "
            f"(got {got_ax}, expected {expected_wall['ax']}, wall {expected_wall_idx})"
        )
        assert abs(got_ay - expected_wall["ay"]) < 0.5, (
            f"sort step {k}: ay mismatch"
        )


# ---------------------------------------------------------------------------
# Test 3: wall data is preserved (not blended by softmax)
# ---------------------------------------------------------------------------


def test_wall_data_preserved():
    """Verify that sorted outputs contain exact copies of wall data,
    not softmax-blended averages.  Uses walls with highly distinct
    coordinates so any blending would be obvious.
    """
    max_walls = 8
    output_node, pos_encoding = build_wall_sort_graph(max_walls)

    # Walls with distinct coordinates but within the score range
    # (all within max_coord=20 so scores stay under 99).
    walls = [
        {"ax": 14.0, "ay": 14.0, "bx": 16.0, "by": 14.0, "tex_id": 10.0},  # mid=(15,14), dist=20.5
        {"ax": 0.5, "ay": 0.5, "bx": 1.5, "by": 0.5, "tex_id": 20.0},      # mid=(1,0.5), dist=1.12
        {"ax": -8.0, "ay": -6.0, "bx": -6.0, "by": -6.0, "tex_id": 30.0},  # mid=(-7,-6), dist=9.22
    ]
    # Sort: wall 1 (dist 1.12), wall 2 (dist 9.22), wall 0 (dist 20.5)

    px, py, angle = 0.0, 0.0, 0.0
    inputs, n_pos, sort_order = _build_full_batch(px, py, angle, walls, max_walls)

    cache = reference_eval(output_node, inputs, n_pos)
    out = cache[output_node]

    N = len(walls)
    for k in range(N):
        sort_pos = 2 + N + k
        expected_wall = walls[sort_order[k]]
        got_ax = out[sort_pos, _OUT_WALL_AX].item()
        # At these scales, even 1% blending would give an error > 1.
        # We allow 0.5 for the piecewise-linear approximation of square.
        assert abs(got_ax - expected_wall["ax"]) < 1.0, (
            f"sort step {k}: ax={got_ax:.2f}, expected {expected_wall['ax']:.2f} "
            f"— possible softmax blending?"
        )


# ---------------------------------------------------------------------------
# Test 4: probe compiled vs oracle
# ---------------------------------------------------------------------------


def test_probe_compiled_vs_oracle():
    """Compile the sort graph and verify every materialised node's
    compiled value matches the oracle.
    """
    max_walls = 4
    output_node, pos_encoding = build_wall_sort_graph(max_walls)

    px, py, angle = 0.0, 0.0, 0.0
    walls = _four_walls()[:4]
    inputs, n_pos, _ = _build_full_batch(px, py, angle, walls, max_walls)

    report = probe_graph(
        output_node,
        pos_encoding=pos_encoding,
        input_values=inputs,
        n_pos=n_pos,
        d=1024,
        d_head=32,
        verbose=False,
        atol=1.0,
    )
    assert report.first_divergent is None, (
        f"probe reported divergence:\n{report.format_short()}"
    )


# ---------------------------------------------------------------------------
# Test 5: autoregressive rollout (compiled, host-driven sort)
# ---------------------------------------------------------------------------


def _build_step_row(
    compiled,
    token_type_vec: torch.Tensor,
    px: float = 0.0,
    py: float = 0.0,
    angle: float = 0.0,
    ax: float = 0.0,
    ay: float = 0.0,
    bx: float = 0.0,
    by: float = 0.0,
    tex_id: float = 0.0,
    sort_mask: np.ndarray = None,
    max_walls: int = 8,
) -> torch.Tensor:
    """Build a (1, d_input) row for ``module.step()`` using the
    compiled module's input_specs to place each field at the right
    column offset.
    """
    vals = {
        "token_type": token_type_vec.unsqueeze(0),             # (1, 8)
        "player_x": torch.tensor([[px]]),
        "player_y": torch.tensor([[py]]),
        "player_angle": torch.tensor([[angle]]),
        "wall_ax": torch.tensor([[ax]]),
        "wall_ay": torch.tensor([[ay]]),
        "wall_bx": torch.tensor([[bx]]),
        "wall_by": torch.tensor([[by]]),
        "wall_tex_id": torch.tensor([[tex_id]]),
        "sort_mask": torch.tensor(
            sort_mask if sort_mask is not None else np.zeros(max_walls),
            dtype=torch.float32,
        ).unsqueeze(0),
    }
    d_input = max(start + width for _, start, width in compiled._input_specs)
    row = torch.zeros(1, d_input)
    for name, start, width in compiled._input_specs:
        row[:, start: start + width] = vals[name]
    return row


def test_autoregressive_sort_4_walls():
    """Compile the graph and run the full autoregressive rollout:
    prefill START + 4 WALL + EOS, then N sort steps with host-driven
    mask feedback.  Verify the emitted walls match the expected
    distance order.
    """
    max_walls = 8
    output_node, pos_encoding = build_wall_sort_graph(max_walls)

    compiled = compile_headless(
        output_node, pos_encoding,
        d=1024, d_head=32, max_layers=200,
        verbose=False,
    )

    px, py, angle = 0.0, 0.0, 0.0
    walls = _four_walls()
    N = len(walls)
    expected_order = _expected_sort_order(px, py, walls)

    past = compiled.empty_past()
    step_idx = 0

    # Prefill: START
    row = _build_step_row(compiled, E8_START, px=px, py=py, angle=angle, max_walls=max_walls)
    with torch.no_grad():
        out, past = compiled.step(row, past, past_len=step_idx)
    step_idx += 1

    # Prefill: WALL * N
    for w in walls:
        row = _build_step_row(
            compiled, E8_WALL,
            ax=w["ax"], ay=w["ay"], bx=w["bx"], by=w["by"],
            tex_id=w["tex_id"], max_walls=max_walls,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step_idx)
        step_idx += 1

    # Prefill: EOS
    row = _build_step_row(compiled, E8_EOS, max_walls=max_walls)
    with torch.no_grad():
        out, past = compiled.step(row, past, past_len=step_idx)
    step_idx += 1

    # Autoregressive sort: N steps
    mask = np.zeros(max_walls)
    sorted_walls = []
    onehot_slice = _out_onehot_slice(max_walls)

    for k in range(N):
        row = _build_step_row(
            compiled, E8_SORTED_WALL, sort_mask=mask, max_walls=max_walls,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step_idx)
        step_idx += 1

        got_ax = out[0, _OUT_WALL_AX].item()
        got_ay = out[0, _OUT_WALL_AY].item()
        got_bx = out[0, _OUT_WALL_BX].item()
        got_by = out[0, _OUT_WALL_BY].item()
        got_tex = out[0, _OUT_WALL_TEX].item()
        sorted_walls.append({
            "ax": got_ax, "ay": got_ay, "bx": got_bx, "by": got_by,
            "tex_id": got_tex,
        })

        # Update mask from onehot
        onehot = out[0, onehot_slice].detach().cpu().numpy()
        mask = np.maximum(mask, np.round(onehot))

    # Verify sort order
    for k, expected_idx in enumerate(expected_order):
        expected_wall = walls[expected_idx]
        got = sorted_walls[k]
        assert abs(got["ax"] - expected_wall["ax"]) < 1.0, (
            f"autoregressive sort step {k}: ax={got['ax']:.2f}, "
            f"expected {expected_wall['ax']:.2f} (wall {expected_idx})"
        )
        assert abs(got["tex_id"] - expected_wall["tex_id"]) < 0.5, (
            f"autoregressive sort step {k}: tex_id={got['tex_id']:.2f}, "
            f"expected {expected_wall['tex_id']:.2f}"
        )
