"""GameState — per-frame player state."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GameState(BaseModel):
    """Per-frame player state.

    Attributes
    ----------
    x, y : float
        Player position in scene coordinates (post-`scene_origin` shift
        is the host's job; this field carries the value the agent's
        prefill should encode).
    angle : int
        Discrete angle index in `[0, 256)`. Maps to a 256-entry
        cos/sin lookup table — angle 0 faces +x, angle 64 faces +y, etc.
    move_speed : float
        Distance the player moves per forward/backward step.
    turn_speed : int
        Turn rate, in angle-index units per turn step.
    """

    model_config = ConfigDict(extra="ignore")

    x: float
    y: float
    angle: int = Field(ge=0, lt=256)
    move_speed: float = 0.3
    turn_speed: int = 4
