"""Dataclasses for capturing pipeline execution traces.

A FrameTrace records intermediate values at each stage boundary during
step_frame execution.  When something breaks, the first failing boundary
tells you exactly which stage diverged.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SortStepTrace:
    """One step of the front-to-back sort loop."""

    position_index: int
    wall_j_onehot: np.ndarray  # (max_walls,) one-hot identifying selected wall
    selected_wall_index: int  # argmax of wall_j_onehot
    vis_lo: float
    vis_hi: float
    tex_id: float
    sort_done: bool


@dataclass
class RenderStepTrace:
    """One step of the render loop."""

    col: int
    start: int
    length: int
    pixels: np.ndarray  # (chunk_size, 3)
    done: bool
    wall_index: int  # which sorted wall is being rendered


@dataclass
class FrameTrace:
    """Full trace of one step_frame call."""

    eos_resolved_x: float = 0.0
    eos_resolved_y: float = 0.0
    eos_new_angle: float = 0.0
    sort_steps: List[SortStepTrace] = field(default_factory=list)
    n_renderable: int = 0
    render_steps: List[RenderStepTrace] = field(default_factory=list)
