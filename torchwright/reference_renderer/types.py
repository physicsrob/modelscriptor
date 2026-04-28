from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Segment:
    """A sector-boundary wall segment.

    Every seg carries the front sector's floor and ceiling.  When
    ``back_floor`` and ``back_ceiling`` are also set the seg is
    two-sided — a sector boundary that may render an opaque *upper*
    wall (between the lower of the two ceilings and the higher), an
    opaque *lower* wall (between the higher of the two floors and the
    lower), and a transparent middle gap that the ray sees through to
    whatever's behind.

    ``texture_id`` is the middle / single-sided texture.
    ``upper_texture_id`` and ``lower_texture_id`` apply only to two-
    sided segs.  All texture ids index the renderer's texture atlas;
    ``-1`` falls back to ``color``.
    """

    ax: float
    ay: float
    bx: float
    by: float
    color: Tuple[float, float, float]
    front_floor: float
    front_ceiling: float
    texture_id: int = -1
    back_floor: Optional[float] = None
    back_ceiling: Optional[float] = None
    upper_texture_id: int = -1
    lower_texture_id: int = -1


@dataclass
class RenderConfig:
    """Configuration for the reference renderer.

    ``player_eye_z`` is the player's eye height in renderer units.  It
    drives vertical projection of sector floors and ceilings onto
    screen rows: world z above the eye line tilts up, below tilts
    down.
    """

    screen_width: int
    screen_height: int
    fov_columns: int
    trig_table: np.ndarray  # shape (256, 2), col 0 = cos, col 1 = sin
    ceiling_color: Tuple[float, float, float]
    floor_color: Tuple[float, float, float]
    player_eye_z: float = 0.0
