from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Segment:
    """A wall segment defined by two endpoints and a color."""

    ax: float
    ay: float
    bx: float
    by: float
    color: Tuple[float, float, float]


@dataclass
class RenderConfig:
    """Configuration for the reference renderer."""

    screen_width: int
    screen_height: int
    fov_columns: int
    trig_table: np.ndarray  # shape (256, 2), col 0 = cos, col 1 = sin
    ceiling_color: Tuple[float, float, float]
    floor_color: Tuple[float, float, float]
