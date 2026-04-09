"""Player input representation for the DOOM game loop."""

from dataclasses import dataclass


@dataclass
class PlayerInput:
    """Boolean flags for player actions in a single frame."""

    forward: bool = False
    backward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    turn_left: bool = False
    turn_right: bool = False
