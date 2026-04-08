from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from torchwright.reference_renderer.types import RenderConfig, Segment


def intersect_ray_segment(
    px: float,
    py: float,
    ray_cos: float,
    ray_sin: float,
    seg: Segment,
) -> Optional[float]:
    """Compute ray-segment intersection distance.

    Returns the ray parameter t (distance along ray) if the ray hits
    the segment, or None if it misses.  Validity is checked via sign
    tests on the numerators and denominator — no division until we
    know the hit is valid.

    Args:
        px, py: Player (ray origin) position.
        ray_cos, ray_sin: Ray direction from trig table.
        seg: Wall segment to test.

    Returns:
        t > 0 if the ray hits the segment, else None.
    """
    dx = ray_cos
    dy = ray_sin
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay
    fx = seg.ax - px
    fy = seg.ay - py

    den = dx * ey - dy * ex
    num_t = fx * ey - fy * ex
    num_u = fx * dy - fy * dx

    if den == 0.0:
        return None  # ray parallel to segment

    if den > 0.0:
        if num_t <= 0.0:
            return None  # intersection behind player
        if num_u < 0.0 or num_u > den:
            return None  # intersection outside segment
    else:
        if num_t >= 0.0:
            return None
        if num_u > 0.0 or num_u < den:
            return None

    return num_t / den


def render_column(
    col: int,
    player_x: float,
    player_y: float,
    player_angle: int,
    segments: List[Segment],
    config: RenderConfig,
) -> np.ndarray:
    """Render a single screen column.

    Returns an array of shape (screen_height, 3) with RGB values.
    """
    h = config.screen_height
    column = np.empty((h, 3), dtype=np.float64)

    # Default fill: ceiling above center, floor below
    center = h // 2
    column[:center] = config.ceiling_color
    column[center:] = config.floor_color

    # Ray direction — map column to angle using fov_columns for angular span
    col_offset = col - config.screen_width // 2
    ray_angle = (player_angle + col_offset * config.fov_columns // config.screen_width) % 256
    ray_cos = config.trig_table[ray_angle, 0]
    ray_sin = config.trig_table[ray_angle, 1]

    # Find nearest intersection
    best_t: Optional[float] = None
    best_seg: Optional[Segment] = None

    for seg in segments:
        t = intersect_ray_segment(player_x, player_y, ray_cos, ray_sin, seg)
        if t is not None and (best_t is None or t < best_t):
            best_t = t
            best_seg = seg

    if best_t is None or best_seg is None:
        return column

    # Fish-eye correction
    angle_diff = (ray_angle - player_angle) % 256
    perp_cos = config.trig_table[angle_diff, 0]
    perp_distance = best_t * perp_cos

    if perp_distance <= 0.0:
        return column

    # Wall height and vertical span
    wall_height = h / perp_distance
    half_wall = wall_height / 2.0
    center_f = h / 2.0
    wall_top = max(0, int(center_f - half_wall))
    wall_bottom = min(h, int(center_f + half_wall))

    if wall_top < wall_bottom:
        column[:wall_top] = config.ceiling_color
        column[wall_top:wall_bottom] = best_seg.color
        column[wall_bottom:] = config.floor_color

    return column


def save_png(frame: np.ndarray, path: str, scale: float = 255.0) -> None:
    """Save a rendered frame as a PNG file.

    Args:
        frame: (H, W, 3) float array from render_frame.
        path: Output file path.
        scale: Multiplier to convert float colors to 0-255 range.
               Default 255.0 assumes colors are in [0.0, 1.0].
    """
    pixels = np.clip(frame * scale, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def render_frame(
    player_x: float,
    player_y: float,
    player_angle: int,
    segments: List[Segment],
    config: RenderConfig,
) -> np.ndarray:
    """Render a complete frame.

    Returns an RGB image as a numpy array of shape
    (config.screen_height, config.screen_width, 3).
    """
    frame = np.empty((config.screen_height, config.screen_width, 3), dtype=np.float64)

    for col in range(config.screen_width):
        frame[:, col, :] = render_column(
            col, player_x, player_y, player_angle, segments, config
        )

    return frame
