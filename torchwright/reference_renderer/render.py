import math
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Per-wall projection
# ---------------------------------------------------------------------------


def _ray_angle_for_column(col: int, player_angle: int, config: RenderConfig) -> int:
    """Map a screen column index to the discrete ray angle (0-255)."""
    col_offset = col - config.screen_width // 2
    return (player_angle + col_offset * config.fov_columns // config.screen_width) % 256


@dataclass
class WallProjection:
    """Result of projecting one wall segment from the player's viewpoint.

    Produced by :func:`project_wall`.  Pass to :func:`render_wall_column`
    to get per-column rendering results.
    """

    seg: Segment
    vis_lo: int  # first screen column that hits this wall (inclusive)
    vis_hi: int  # last screen column that hits this wall (inclusive)


def project_wall(
    player_x: float,
    player_y: float,
    player_angle: int,
    seg: Segment,
    config: RenderConfig,
) -> Optional[WallProjection]:
    """Project a wall segment onto the screen and compute its visible column range.

    Iterates every screen column, casts the column's ray against *seg*,
    and records which columns produce a hit.  Returns ``None`` if no
    column hits the wall.

    Args:
        player_x, player_y: Player world coordinates.
        player_angle: Player facing direction (0-255).
        seg: Wall segment to project.
        config: Render configuration (screen size, FOV, trig table).

    Returns:
        A :class:`WallProjection` with the visible column range, or
        ``None`` if the wall is not visible from this viewpoint.
    """
    lo = None
    hi = None
    for col in range(config.screen_width):
        ray_angle = _ray_angle_for_column(col, player_angle, config)
        ray_cos = config.trig_table[ray_angle, 0]
        ray_sin = config.trig_table[ray_angle, 1]
        hit = intersect_ray_segment(player_x, player_y, ray_cos, ray_sin, seg)
        if hit is not None:
            if lo is None:
                lo = col
            hi = col
    if lo is None or hi is None:
        return None
    return WallProjection(seg=seg, vis_lo=lo, vis_hi=hi)


# ---------------------------------------------------------------------------
# Per-wall-per-column rendering
# ---------------------------------------------------------------------------


@dataclass
class WallColumnResult:
    """Per-column rendering result for a single wall.

    Produced by :func:`render_wall_column`.
    """

    distance: float  # perpendicular distance (after fish-eye correction)
    wall_height: float  # wall height in screen rows (float, before clamping)
    wall_top: int  # first wall row (inclusive, clamped to screen)
    wall_bottom: int  # one past last wall row (exclusive, clamped to screen)
    tex_col: int  # texture column index (-1 if untextured)
    pixels: np.ndarray  # (screen_height, 3) full column with ceiling/wall/floor


def render_wall_column(
    col: int,
    wall_proj: WallProjection,
    player_x: float,
    player_y: float,
    player_angle: int,
    config: RenderConfig,
    textures: Optional[List[np.ndarray]] = None,
) -> Optional[WallColumnResult]:
    """Render one screen column of a specific wall.

    Casts the column's ray against the wall, computes perpendicular
    distance, wall height, texture column, and fills a pixel strip
    with ceiling, wall, and floor colors.

    Args:
        col: Screen column index.
        wall_proj: Wall projection from :func:`project_wall`.
        player_x, player_y: Player world coordinates.
        player_angle: Player facing direction (0-255).
        config: Render configuration.
        textures: Optional texture atlas.

    Returns:
        A :class:`WallColumnResult`, or ``None`` if the ray doesn't
        hit the wall at this column or the wall is behind the player.
    """
    h = config.screen_height
    seg = wall_proj.seg

    ray_angle = _ray_angle_for_column(col, player_angle, config)
    ray_cos = config.trig_table[ray_angle, 0]
    ray_sin = config.trig_table[ray_angle, 1]

    hit = intersect_ray_segment(player_x, player_y, ray_cos, ray_sin, seg)
    if hit is None:
        return None

    t, u = hit

    angle_diff = (ray_angle - player_angle) % 256
    perp_cos = config.trig_table[angle_diff, 0]
    perp_distance = t * perp_cos

    if perp_distance <= 0.0:
        return None

    wall_height_f = h / perp_distance
    center_f = h / 2.0
    wall_top_f = center_f - wall_height_f / 2.0
    wall_bottom_f = center_f + wall_height_f / 2.0

    wall_top = max(0, math.ceil(wall_top_f - 0.5))
    wall_bottom = min(h, math.ceil(wall_bottom_f - 0.5))

    # Texture column
    tex_col = -1
    if textures is not None and seg.texture_id >= 0:
        tw = textures[seg.texture_id].shape[0]
        tex_col = min(int(u * tw), tw - 1)

    # Build pixel column
    column = np.empty((h, 3), dtype=np.float64)
    center = h // 2
    column[:center] = config.ceiling_color
    column[center:] = config.floor_color

    if wall_top < wall_bottom:
        column[:wall_top] = config.ceiling_color
        column[wall_bottom:] = config.floor_color

        if textures is not None and seg.texture_id >= 0:
            tex = textures[seg.texture_id]
            tw, th = tex.shape[0], tex.shape[1]
            tc = min(int(u * tw), tw - 1)
            for row in range(wall_top, wall_bottom):
                v = ((row + 0.5) - wall_top_f) / wall_height_f
                tex_row = min(int(v * th), th - 1)
                column[row] = tex[tc, tex_row]
        else:
            column[wall_top:wall_bottom] = seg.color

    return WallColumnResult(
        distance=perp_distance,
        wall_height=wall_height_f,
        wall_top=wall_top,
        wall_bottom=wall_bottom,
        tex_col=tex_col,
        pixels=column,
    )


# ---------------------------------------------------------------------------
# Column and frame rendering
# ---------------------------------------------------------------------------


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

    Projects every wall segment, finds the nearest one visible at this
    column, and returns the pixel strip.  Uses :func:`project_wall` and
    :func:`render_wall_column` internally.

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
    default = np.empty((h, 3), dtype=np.float64)
    center = h // 2
    default[:center] = config.ceiling_color
    default[center:] = config.floor_color

    best_result: Optional[WallColumnResult] = None

    for seg in segments:
        proj = project_wall(player_x, player_y, player_angle, seg, config)
        if proj is None:
            continue
        if not (proj.vis_lo <= col <= proj.vis_hi):
            continue
        result = render_wall_column(
            col, proj, player_x, player_y, player_angle, config, textures
        )
        if result is None:
            continue
        if best_result is None or result.distance < best_result.distance:
            best_result = result

    if best_result is not None:
        return best_result.pixels
    return default


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
            col,
            player_x,
            player_y,
            player_angle,
            segments,
            config,
            textures=textures,
        )

    return frame
