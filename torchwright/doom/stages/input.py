"""INPUT stage: game logic + control broadcast.

At INPUT positions the graph:

1. Updates player-facing angle from turn inputs.
2. Derives (dx, dy) velocity from movement inputs and the new angle.
3. Looks up (cos, sin) of the new angle for use by downstream stages.

It then broadcasts those derived values to **every** position via
``attend_mean_where`` so WALL, EOS, SORTED, and RENDER can consume them
as plain ``Node`` inputs.  The broadcast outputs are the stage's public
contract — the pre-broadcast values are intermediate and not exposed.
"""

from dataclasses import dataclass

import torch

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    compare,
    mod_const,
    multiply_const,
    negate,
    subtract,
)
from torchwright.ops.attention_ops import attend_mean_where
from torchwright.ops.map_select import select

from torchwright.doom.graph_utils import extract_from
from torchwright.doom.renderer import trig_lookup

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class InputInputs:
    """Host-fed nodes and graph-level dependencies consumed by the INPUT stage."""

    player_angle: Node
    input_turn_left: Node
    input_turn_right: Node
    input_forward: Node
    input_backward: Node
    input_strafe_left: Node
    input_strafe_right: Node
    is_input: Node  # per-position flag (1.0 at INPUT positions)
    pos_encoding: PosEncoding  # shared across all stages


@dataclass
class InputOutputs:
    """Post-broadcast values available at every position for downstream stages."""

    vel_dx: Node
    vel_dy: Node
    move_cos: Node
    move_sin: Node
    new_angle: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_input(
    inputs: InputInputs,
    turn_speed: int,
    move_speed: float,
) -> InputOutputs:
    """Build the INPUT stage subgraph."""
    with annotate("input"):
        with annotate("game_logic"):
            new_angle = _compute_new_angle(
                inputs.player_angle,
                inputs.input_turn_left,
                inputs.input_turn_right,
                turn_speed,
            )
            vel_dx, vel_dy = _compute_velocity(
                new_angle,
                inputs.input_forward,
                inputs.input_backward,
                inputs.input_strafe_left,
                inputs.input_strafe_right,
                move_speed,
            )
            move_cos, move_sin = trig_lookup(new_angle)

        with annotate("attention"):
            return _broadcast_to_all_positions(
                inputs.pos_encoding,
                inputs.is_input,
                vel_dx,
                vel_dy,
                move_cos,
                move_sin,
                new_angle,
            )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _compute_new_angle(
    old_angle: Node,
    input_turn_left: Node,
    input_turn_right: Node,
    turn_speed: int,
) -> Node:
    """new_angle = (old_angle + turn_right*speed - turn_left*speed) mod 256"""
    turn_r = multiply_const(input_turn_right, float(turn_speed))
    turn_l = multiply_const(input_turn_left, float(turn_speed))
    turn_delta = subtract(turn_r, turn_l)
    raw_angle = add(old_angle, turn_delta)
    shifted = add_const(raw_angle, 256.0)
    return mod_const(shifted, 256, 512 + turn_speed)


def _compute_velocity(
    new_angle: Node,
    input_forward: Node,
    input_backward: Node,
    input_strafe_left: Node,
    input_strafe_right: Node,
    move_speed: float,
):
    """Compute (dx, dy) from player inputs and facing angle."""
    move_cos, move_sin = trig_lookup(new_angle)
    speed_cos = multiply_const(move_cos, move_speed)
    speed_sin = multiply_const(move_sin, move_speed)
    neg_speed_cos = negate(speed_cos)
    neg_speed_sin = negate(speed_sin)
    zero = LiteralValue(torch.tensor([0.0]), name="zero_vel")
    is_fwd = compare(input_forward, 0.5)
    is_bwd = compare(input_backward, 0.5)
    is_sl = compare(input_strafe_left, 0.5)
    is_sr = compare(input_strafe_right, 0.5)
    dx = add(
        add(select(is_fwd, speed_cos, zero), select(is_bwd, neg_speed_cos, zero)),
        add(select(is_sl, speed_sin, zero), select(is_sr, neg_speed_sin, zero)),
    )
    dy = add(
        add(select(is_fwd, speed_sin, zero), select(is_bwd, neg_speed_sin, zero)),
        add(select(is_sl, neg_speed_cos, zero), select(is_sr, speed_cos, zero)),
    )
    return dx, dy


def _broadcast_to_all_positions(
    pos_encoding: PosEncoding,
    is_input: Node,
    vel_dx: Node,
    vel_dy: Node,
    move_cos: Node,
    move_sin: Node,
    new_angle: Node,
) -> InputOutputs:
    """Use attend_mean_where over is_input to copy INPUT-position values everywhere.

    Since exactly one position has is_input=1, the "mean" is just that value.
    """
    ctrl_attn = attend_mean_where(
        pos_encoding,
        validity=is_input,
        value=Concatenate([vel_dx, vel_dy, move_cos, move_sin, new_angle]),
    )
    return InputOutputs(
        vel_dx=extract_from(ctrl_attn, 5, 0, 1, "a_vdx"),
        vel_dy=extract_from(ctrl_attn, 5, 1, 1, "a_vdy"),
        move_cos=extract_from(ctrl_attn, 5, 2, 1, "a_mcos"),
        move_sin=extract_from(ctrl_attn, 5, 3, 1, "a_msin"),
        new_angle=extract_from(ctrl_attn, 5, 4, 1, "a_angle"),
    )
