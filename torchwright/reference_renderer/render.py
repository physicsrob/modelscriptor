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
) -> Optional[Tuple[float, float]]:
    """Compute ray-segment intersection distance and u parameter.

    Returns ``(t, u)`` where *t* is the distance along the ray and *u*
    is the fractional position along the segment (0 at A, 1 at B).
    Returns ``None`` if the ray misses.

    Args:
        px, py: Player (ray origin) position.
        ray_cos, ray_sin: Ray direction from trig table.
        seg: Wall segment to test.

    Returns:
        ``(t, u)`` if the ray hits the segment, else ``None``.
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

    return num_t / den, num_u / den


def render_column(
    col: int,
    player_x: float,
    player_y: float,
    player_angle: int,
    segments: List[Segment],
    config: RenderConfig,
    textures: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Render a single screen column.

    Args:
        col: Screen column index.
        player_x, player_y: Player world coordinates.
        player_angle: Player facing direction (0-255).
        segments: Wall segments.
        config: Render configuration.
        textures: Optional texture atlas — list of (W, H, 3) arrays
            indexed by ``Segment.texture_id``.

    Returns:
        Array of shape (screen_height, 3) with RGB values.
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
    best_u: float = 0.0
    best_seg: Optional[Segment] = None

    for seg in segments:
        hit = intersect_ray_segment(player_x, player_y, ray_cos, ray_sin, seg)
        if hit is not None:
            t, u = hit
            if best_t is None or t < best_t:
                best_t = t
                best_u = u
                best_seg = seg

    if best_t is None or best_seg is None:
        return column

    # Fish-eye correction
    angle_diff = (ray_angle - player_angle) % 256
    perp_cos = config.trig_table[angle_diff, 0]
    perp_distance = best_t * perp_cos

    if perp_distance <= 0.0:
        return column

    # Wall height and vertical span (center-sampling rasterization).
    #
    # Row ``i`` is part of the wall iff its centre ``i + 0.5`` falls in
    # the open-on-the-right interval ``[wall_top_f, wall_bottom_f)``.
    # Equivalently the integer wall row range is
    # ``[ceil(wall_top_f - 0.5), ceil(wall_bottom_f - 0.5))``.  The
    # texture row for screen row ``i`` is then computed at the same
    # centre ``i + 0.5`` against the **fractional** wall edges, so the
    # texture sampling is geometrically meaningful instead of being
    # quantised to whichever ``int()``-rounded wall_top happens to land
    # at.  This matches what the compiled graph does (its
    # ``in_range(wall_top, wall_bottom, ...)`` mask uses the same
    # ``i + 0.5`` centres, and the textured fill samples
    # ``linear_bin_index(y_abs + 0.5, wall_top, wall_bottom, tex_h)``).
    import math as _math

    wall_height_f = h / perp_distance
    center_f = h / 2.0
    wall_top_f = center_f - wall_height_f / 2.0
    wall_bottom_f = center_f + wall_height_f / 2.0

    wall_top = max(0, _math.ceil(wall_top_f - 0.5))
    wall_bottom = min(h, _math.ceil(wall_bottom_f - 0.5))

    if wall_top < wall_bottom:
        column[:wall_top] = config.ceiling_color
        column[wall_bottom:] = config.floor_color

        if textures is not None and best_seg.texture_id >= 0:
            tex = textures[best_seg.texture_id]
            tw, th = tex.shape[0], tex.shape[1]
            tex_col = min(int(best_u * tw), tw - 1)
            for row in range(wall_top, wall_bottom):
                v = ((row + 0.5) - wall_top_f) / wall_height_f
                tex_row = min(int(v * th), th - 1)
                column[row] = tex[tex_col, tex_row]
        else:
            column[wall_top:wall_bottom] = best_seg.color

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
    textures: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Render a complete frame.

    Args:
        player_x, player_y: Player world coordinates.
        player_angle: Player facing direction (0-255).
        segments: Wall segments.
        config: Render configuration.
        textures: Optional texture atlas for wall textures.

    Returns:
        An RGB image as a numpy array of shape
        (config.screen_height, config.screen_width, 3).
    """
    frame = np.empty((config.screen_height, config.screen_width, 3), dtype=np.float64)

    for col in range(config.screen_width):
        frame[:, col, :] = render_column(
            col, player_x, player_y, player_angle, segments, config,
            textures=textures,
        )

    return frame
