from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Segment:
    """A wall segment defined by two endpoints and a color.

    When *texture_id* is non-negative the segment is textured: the
    renderer looks up pixels from the texture atlas instead of using
    the solid *color*.  A value of -1 (the default) means solid color.
    """

    ax: float
    ay: float
    bx: float
    by: float
    color: Tuple[float, float, float]
    texture_id: int = -1


@dataclass
class RenderConfig:
    """Configuration for the reference renderer."""

    screen_width: int
    screen_height: int
    fov_columns: int
    trig_table: np.ndarray  # shape (256, 2), col 0 = cos, col 1 = sin
    ceiling_color: Tuple[float, float, float]
    floor_color: Tuple[float, float, float]
