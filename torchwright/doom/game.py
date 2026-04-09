"""Game state and update logic for the DOOM game loop."""

from dataclasses import dataclass

import numpy as np
from typing import List

from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer.collision import resolve_collision_ray
from torchwright.reference_renderer.types import Segment


@dataclass
class GameState:
    """Mutable game state carried across frames."""

    x: float
    y: float
    angle: int  # 0-255 discrete angle index
    move_speed: float = 0.3
    turn_speed: int = 4


def update_state(
    state: GameState,
    inputs: PlayerInput,
    segments: List[Segment],
    trig_table: np.ndarray,
    collision_fn=None,
) -> GameState:
    """Compute the next frame's game state from inputs.

    Processes turn inputs (angle change), then movement inputs
    (position change with collision resolution).

    Args:
        state: Current game state.
        inputs: Player input flags for this frame.
        segments: Wall segments for collision detection.
        trig_table: (256, 2) trig table -- col 0 = cos, col 1 = sin.
        collision_fn: Callable(old_x, old_y, new_x, new_y, segments) -> (x, y).
            Defaults to resolve_collision_ray (matches the compiled transformer).

    Returns:
        New GameState with updated position and angle.
    """
    if collision_fn is None:
        collision_fn = resolve_collision_ray

    # Turn
    new_angle = state.angle
    if inputs.turn_left:
        new_angle = (state.angle - state.turn_speed) % 256
    if inputs.turn_right:
        new_angle = (state.angle + state.turn_speed) % 256

    # Movement direction from angle
    cos_a = float(trig_table[new_angle, 0])
    sin_a = float(trig_table[new_angle, 1])

    dx, dy = 0.0, 0.0
    if inputs.forward:
        dx += state.move_speed * cos_a
        dy += state.move_speed * sin_a
    if inputs.backward:
        dx -= state.move_speed * cos_a
        dy -= state.move_speed * sin_a
    if inputs.strafe_left:
        dx += state.move_speed * sin_a
        dy -= state.move_speed * cos_a
    if inputs.strafe_right:
        dx -= state.move_speed * sin_a
        dy += state.move_speed * cos_a

    # Collision resolution
    new_x, new_y = collision_fn(
        state.x, state.y,
        state.x + dx, state.y + dy,
        segments,
    )

    return GameState(
        x=new_x, y=new_y, angle=new_angle,
        move_speed=state.move_speed, turn_speed=state.turn_speed,
    )
