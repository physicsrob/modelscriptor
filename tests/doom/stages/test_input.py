"""Unit tests for the INPUT stage (torchwright.doom.stages.input).

These tests compile a subgraph rooted at ``build_input``'s outputs and
verify that at an INPUT token:

* ``new_angle`` matches the reference turn logic.
* ``vel_dx, vel_dy`` match the reference velocity computation.

The stage broadcasts its outputs to every position via
``attend_mean_where``; a two-token sequence (INPUT + one non-INPUT
receiver) is used so the broadcast actually does something observable.
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer.trig import generate_trig_table

from torchwright.doom.graph_constants import E8_INPUT, E8_WALL
from torchwright.doom.stages.input import InputInputs, build_input


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_MOVE_SPEED = 0.3
_TURN_SPEED = 4

_TRIG_TABLE = generate_trig_table()


@pytest.fixture(scope="module")
def input_module():
    """Compile build_input's broadcast outputs.

    The stage uses ``attend_mean_where(validity=is_input)``, so we need
    a ``pos_encoding`` and a runtime sequence.  The module is a standard
    headless transformer; we feed 2-position batches where position 0
    is the INPUT token and position 1 is the "receiver".
    """
    pos = create_pos_encoding()
    player_angle = create_input("player_angle", 1)
    input_forward = create_input("input_forward", 1)
    input_backward = create_input("input_backward", 1)
    input_turn_left = create_input("input_turn_left", 1)
    input_turn_right = create_input("input_turn_right", 1)
    input_strafe_left = create_input("input_strafe_left", 1)
    input_strafe_right = create_input("input_strafe_right", 1)
    is_input = create_input("is_input", 1)

    out = build_input(
        InputInputs(
            player_angle=player_angle,
            input_turn_left=input_turn_left,
            input_turn_right=input_turn_right,
            input_forward=input_forward,
            input_backward=input_backward,
            input_strafe_left=input_strafe_left,
            input_strafe_right=input_strafe_right,
            is_input=is_input,
            pos_encoding=pos,
        ),
        turn_speed=_TURN_SPEED,
        move_speed=_MOVE_SPEED,
    )
    output = Concatenate([
        out.new_angle, out.vel_dx, out.vel_dy, out.move_cos, out.move_sin,
    ])
    return compile_headless(
        output, pos, d=1024, d_head=16, max_layers=40, verbose=False,
    )


def _pack(module, rows: list[dict]) -> torch.Tensor:
    """Pack ``rows`` (one dict per token) into the module's input tensor."""
    d_input = max(s + w for _, s, w in module._input_specs)
    T = len(rows)
    t = torch.zeros(T, d_input, dtype=torch.float32)
    for i, row in enumerate(rows):
        for name, start, width in module._input_specs:
            t[i, start:start + width] = torch.tensor(
                row[name], dtype=torch.float32,
            ).reshape(width)
    return t


def _reference_velocity(angle_idx: int, inputs: dict) -> tuple[float, float]:
    cos_a = float(_TRIG_TABLE[angle_idx, 0])
    sin_a = float(_TRIG_TABLE[angle_idx, 1])
    dx = 0.0
    dy = 0.0
    if inputs.get("forward"):
        dx += _MOVE_SPEED * cos_a
        dy += _MOVE_SPEED * sin_a
    if inputs.get("backward"):
        dx -= _MOVE_SPEED * cos_a
        dy -= _MOVE_SPEED * sin_a
    if inputs.get("strafe_left"):
        dx += _MOVE_SPEED * sin_a
        dy -= _MOVE_SPEED * cos_a
    if inputs.get("strafe_right"):
        dx -= _MOVE_SPEED * sin_a
        dy += _MOVE_SPEED * cos_a
    return dx, dy


def _reference_new_angle(old_angle: int, turn_left: bool, turn_right: bool) -> int:
    a = old_angle
    if turn_right:
        a = (a + _TURN_SPEED) % 256
    if turn_left:
        a = (a - _TURN_SPEED) % 256
    return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle,controls,exp_angle_delta", [
    (0, {"turn_right": True}, +_TURN_SPEED),
    (10, {"turn_left": True}, -_TURN_SPEED),
    (0, {}, 0),
    (250, {"turn_right": True}, +_TURN_SPEED),   # wrap-around near 256
    (3, {"turn_left": True}, -_TURN_SPEED),      # wrap-around near 0
])
def test_new_angle_matches_reference(input_module, angle, controls, exp_angle_delta):
    """new_angle = (old + turn_right*speed - turn_left*speed) mod 256."""
    row_input = {
        "player_angle": float(angle),
        "input_forward": 0.0,
        "input_backward": 0.0,
        "input_turn_left": 1.0 if controls.get("turn_left") else 0.0,
        "input_turn_right": 1.0 if controls.get("turn_right") else 0.0,
        "input_strafe_left": 0.0,
        "input_strafe_right": 0.0,
        "is_input": 1.0,
    }
    # Position 1 receives the broadcast; its control values are irrelevant.
    row_receiver = {**row_input, "is_input": 0.0}
    inputs = _pack(input_module, [row_input, row_receiver])
    with torch.no_grad():
        out = input_module(inputs)
    # Check the broadcast output at the receiver position.
    new_angle_at_receiver = out[1, 0].item()
    expected = _reference_new_angle(
        angle,
        bool(controls.get("turn_left")),
        bool(controls.get("turn_right")),
    )
    assert abs(new_angle_at_receiver - expected) < 1.0, (
        f"angle={angle} controls={controls}: new_angle={new_angle_at_receiver:.2f}, "
        f"expected {expected}"
    )


@pytest.mark.parametrize("angle,controls", [
    (0, {"forward": True}),      # east
    (64, {"forward": True}),     # north
    (128, {"forward": True}),    # west
    (0, {"backward": True}),
    (64, {"strafe_right": True}),
    (0, {"strafe_left": True}),
    (0, {"forward": True, "strafe_right": True}),   # diagonal
])
def test_velocity_matches_reference(input_module, angle, controls):
    """vel_dx, vel_dy at broadcast-receiver must match reference trig-driven velocity."""
    row_input = {
        "player_angle": float(angle),
        "input_forward": 1.0 if controls.get("forward") else 0.0,
        "input_backward": 1.0 if controls.get("backward") else 0.0,
        "input_turn_left": 0.0,
        "input_turn_right": 0.0,
        "input_strafe_left": 1.0 if controls.get("strafe_left") else 0.0,
        "input_strafe_right": 1.0 if controls.get("strafe_right") else 0.0,
        "is_input": 1.0,
    }
    row_receiver = {**row_input, "is_input": 0.0}
    inputs = _pack(input_module, [row_input, row_receiver])
    with torch.no_grad():
        out = input_module(inputs)
    vel_dx = out[1, 1].item()
    vel_dy = out[1, 2].item()
    ref_dx, ref_dy = _reference_velocity(angle, controls)
    # Allow a little slack for trig-table lookup + piecewise interp.
    assert abs(vel_dx - ref_dx) < 0.05, (
        f"angle={angle} controls={controls}: vel_dx={vel_dx:+.3f}, expected {ref_dx:+.3f}"
    )
    assert abs(vel_dy - ref_dy) < 0.05, (
        f"angle={angle} controls={controls}: vel_dy={vel_dy:+.3f}, expected {ref_dy:+.3f}"
    )
