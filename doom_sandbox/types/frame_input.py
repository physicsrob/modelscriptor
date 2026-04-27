"""FrameInput — per-frame movement input flags."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrameInput(BaseModel):
    """Per-frame movement input — what the player pressed this tick.

    Distinct from `GameState`, which is *state* (current position,
    angle). `FrameInput` is *input* (what the player asked for this
    frame). The pattern is
    `(GameState_N, FrameInput_N) → GameState_{N+1}`.

    All flags default to False — the no-input frame.
    """

    model_config = ConfigDict(extra="ignore")

    forward:      bool = False
    backward:     bool = False
    turn_left:    bool = False
    turn_right:   bool = False
    strafe_left:  bool = False
    strafe_right: bool = False
