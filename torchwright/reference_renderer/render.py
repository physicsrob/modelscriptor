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
    """Map a screen column index to the nearest discrete ray angle (0-255).

    Used by the legacy ``project_wall`` / ``render_wall_column`` helpers
    that take an integer angle and look up cos/sin from the trig
    table.  ``render_column`` instead uses
    :func:`_ray_angle_offset_radians` and ``math.cos`` / ``math.sin``
    so each screen column gets a unique, sub-BAM-precision direction
    (320 columns at HFOV=64 BAM units is 0.2 BAM/col — integer
    truncation here would group 5 adjacent columns onto the same
    ray, producing visible 5-pixel horizontal stripes on textured
    walls).
    """
    col_offset = col - config.screen_width // 2
    return (player_angle + col_offset * config.fov_columns // config.screen_width) % 256


def _ray_angle_offset_radians(col: int, config: RenderConfig) -> float:
    """Return this column's ray angle offset from the player heading, in radians.

    Linear-angle distribution across the FOV.  Screen RIGHT (col >
    centre) maps to player's RIGHT — i.e., CW rotation from heading
    = negative angle offset in math convention (where positive
    angle = CCW).  Float-precision so each column has a unique
    angle.
    """
    col_offset = col - config.screen_width // 2
    fov_radians = config.fov_columns * 2.0 * math.pi / 256.0
    return -col_offset * fov_radians / config.screen_width


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


def _round_y(y: float) -> int:
    """Convert a continuous screen-y to an integer row index (round-half-up)."""
    return math.ceil(y - 0.5)


def _paint_textured_strip(
    column: np.ndarray,
    row_top: int,
    row_bottom: int,
    proj_top: float,
    proj_bottom: float,
    tex: Optional[np.ndarray],
    tex_col: int,
    fallback_color: Tuple[float, float, float],
    world_height: float,
    tex_v_offset: float,
) -> None:
    """Paint a vertical strip into ``column`` rows ``[row_top, row_bottom)``.

    ``proj_top`` / ``proj_bottom`` are the *unclipped* projected screen-y
    coordinates of the strip's full extent — used to compute per-row
    v-coordinates so that a clipped wall samples the texture
    consistently with an unclipped one.  ``world_height`` is the wall
    fragment's vertical extent in world units.  ``tex_v_offset`` is the
    DOOM ``texturemid`` offset minus the fragment top: it shifts the
    sampled texture row, implementing DOOM's per-class pegging:

    - One-sided default (top-pegged): ``tex_v_offset = 0``.
    - Upper wall default: ``tex_v_offset = tex_height - upper_world_height``
      (this anchors the texture so its bottom aligns with the back
      sector's ceiling — the bottom of the upper wall fragment).
    - Lower wall default: ``tex_v_offset = 0`` (texture top aligns with
      back floor — the top of the lower wall fragment).

    DOOM convention: 1 texture pixel = 1 world unit, so
    ``tex_row = int(v * world_height + tex_v_offset) mod tex_height``.
    """
    if row_top >= row_bottom:
        return
    if tex is None:
        column[row_top:row_bottom] = fallback_color
        return
    th = tex.shape[1]
    span = proj_bottom - proj_top
    if span <= 0:
        column[row_top:row_bottom] = fallback_color
        return
    for row in range(row_top, row_bottom):
        v = ((row + 0.5) - proj_top) / span
        if v < 0.0:
            v = 0.0
        elif v >= 1.0:
            v = 1.0 - 1e-9
        tex_row = int(v * world_height + tex_v_offset) % th
        column[row] = tex[tex_col, tex_row]


def _texture_for(
    textures: Optional[List[np.ndarray]],
    tex_id: int,
) -> Optional[np.ndarray]:
    if textures is None or tex_id < 0 or tex_id >= len(textures):
        return None
    return textures[tex_id]


def _tex_col_world(u: float, seg: Segment, tex: Optional[np.ndarray]) -> int:
    """Tile by world distance along the seg (DOOM convention).

    ``u`` is the fractional position along the seg from a → b (0..1).
    Multiplied by the seg's world length and modded by the texture's
    pixel width gives the texel column.  Assumes 1 texture pixel = 1
    world unit, which is DOOM's authoring convention.
    """
    if tex is None:
        return 0
    tw = tex.shape[0]
    seg_world_length = math.hypot(seg.bx - seg.ax, seg.by - seg.ay)
    return int(u * seg_world_length) % tw


def render_column(
    col: int,
    player_x: float,
    player_y: float,
    player_angle: int,
    segments: List[Segment],
    config: RenderConfig,
    textures: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Render a single screen column following DOOM's projection.

    Per-column algorithm — same shape as DOOM's R_RenderSegLoop:

    1. Cast a ray at the column's view angle.
    2. Gather every hit, back-face culled (player must be on the seg's
       FRONT side; the seg's a → b direction defines front as "right").
    3. Sort hits front-to-back by perpendicular distance.
    4. Walk hits, maintaining ``ceilingclip`` (last row painted from
       top) and ``floorclip`` (first row painted from bottom).  Each
       hit may paint into the open range ``(ceilingclip, floorclip)``
       and narrow it.
    5. A one-sided seg paints from front_ceiling to front_floor and
       closes the column.  A two-sided seg paints an upper fragment
       (where back_ceiling < front_ceiling) and / or a lower fragment
       (where back_floor > front_floor), then leaves the middle gap
       transparent so the ray continues to the next hit.

    Vertical projection uses a single focal length derived from the
    horizontal FOV — ``f = (W/2) / tan(HFOV/2)`` — so pixels are
    aspect-square and texture pixels render with consistent
    horizontal/vertical scale.  Texture mapping follows DOOM's
    1-pixel-per-world-unit convention with default per-class pegging.

    Args:
        col: Screen column index.
        player_x, player_y: Player position in raw world units.
        player_angle: Player facing direction (0-255).
        segments: Wall segments (every seg must be sector-aware —
            ``front_floor`` and ``front_ceiling`` set).
        config: Render configuration (incl. ``player_eye_z``).
        textures: Texture atlas — list of (W, H, 3) arrays indexed by
            ``Segment.texture_id`` / ``upper_texture_id`` /
            ``lower_texture_id``.

    Returns:
        Array of shape (screen_height, 3) with RGB values.
    """
    h = config.screen_height
    h_half = h / 2.0
    eye_z = config.player_eye_z
    center = h // 2

    # DOOM projection: same focal length for both axes.  HFOV in
    # angle units (0..256 = 360°): half-FOV in radians = fov_columns *
    # π / 256.  Then f = (W/2) / tan(half-FOV).
    focal_length = (config.screen_width / 2.0) / math.tan(
        config.fov_columns * math.pi / 256.0
    )

    # Cast this column's ray.  Float-precision angle: the column's
    # offset from the player heading in radians, then absolute ray
    # angle = player_angle (BAM, integer) + offset.  Using math.cos /
    # math.sin directly avoids the 5-pixel-stripe aliasing that
    # integer-truncated trig-table lookup produces at sub-BAM angular
    # resolutions (e.g. 320 cols × 64 BAM HFOV = 0.2 BAM/col).
    angle_offset_rad = _ray_angle_offset_radians(col, config)
    player_angle_rad = player_angle * 2.0 * math.pi / 256.0
    ray_angle_rad = player_angle_rad + angle_offset_rad
    cos_a = math.cos(ray_angle_rad)
    sin_a = math.sin(ray_angle_rad)
    perp_cos = math.cos(angle_offset_rad)

    hits: List[Tuple[float, float, Segment]] = []
    for seg in segments:
        # Back-face cull: WADs store two segs per two-sided linedef
        # (one per facing).  The player only sees the one whose FRONT
        # is on the player's side of the seg's line.  By DOOM
        # convention FRONT is on the right when traversing the seg
        # from a → b, i.e. ``cross = dy*(px-ax) - dx*(py-ay) > 0``.
        dx = seg.bx - seg.ax
        dy = seg.by - seg.ay
        cross = dy * (player_x - seg.ax) - dx * (player_y - seg.ay)
        if cross <= 0.0:
            continue
        result = intersect_ray_segment(player_x, player_y, cos_a, sin_a, seg)
        if result is None:
            continue
        t, u = result
        if t <= 0.0:
            continue
        perp = t * perp_cos
        if perp <= 0.0:
            continue
        hits.append((perp, u, seg))
    hits.sort(key=lambda x: x[0])

    column = np.empty((h, 3), dtype=np.float64)
    column[:center] = config.ceiling_color
    column[center:] = config.floor_color

    ceilingclip = -1  # last row painted from the top
    floorclip = h  # first row painted from the bottom

    def project_z(world_z: float, perp: float) -> float:
        return h_half - (world_z - eye_z) * focal_length / perp

    for perp, u, seg in hits:
        if ceilingclip + 1 >= floorclip:
            break

        # Project the front sector's ceiling/floor onto screen.
        fy_top = project_z(seg.front_ceiling, perp)
        fy_bot = project_z(seg.front_floor, perp)
        front_world_height = seg.front_ceiling - seg.front_floor

        if seg.back_floor is None:
            # One-sided wall: opaque, top-pegged at front_ceiling.
            wall_top = max(_round_y(fy_top), ceilingclip + 1)
            wall_bot = min(_round_y(fy_bot), floorclip)
            tex = _texture_for(textures, seg.texture_id)
            tex_col = _tex_col_world(u, seg, tex)
            _paint_textured_strip(
                column,
                wall_top,
                wall_bot,
                fy_top,
                fy_bot,
                tex,
                tex_col,
                seg.color,
                world_height=front_world_height,
                tex_v_offset=0.0,
            )
            ceilingclip = floorclip - 1
            break

        # Two-sided seg: paint upper / lower fragments, leave middle gap.

        # Upper wall (back ceiling lower than front ceiling — soffit/lintel).
        # DOOM default pegging: bottom of texture aligns with back
        # ceiling, so the texture appears to "hang down" from the
        # opening's top edge.  Equivalent v-offset:
        # ``tex_h - upper_world_height``.
        if seg.back_ceiling < seg.front_ceiling:
            bc_y = project_z(seg.back_ceiling, perp)
            upper_top = max(_round_y(fy_top), ceilingclip + 1)
            upper_bot = min(_round_y(bc_y), floorclip)
            tex = _texture_for(textures, seg.upper_texture_id)
            tex_col = _tex_col_world(u, seg, tex)
            upper_world_h = seg.front_ceiling - seg.back_ceiling
            tex_v_offset = (tex.shape[1] - upper_world_h) if tex is not None else 0.0
            _paint_textured_strip(
                column,
                upper_top,
                upper_bot,
                fy_top,
                bc_y,
                tex,
                tex_col,
                seg.color,
                world_height=upper_world_h,
                tex_v_offset=tex_v_offset,
            )
            new_top = _round_y(bc_y)
        else:
            new_top = _round_y(fy_top)
        if new_top - 1 > ceilingclip:
            ceilingclip = new_top - 1

        # Lower wall (back floor higher than front floor — step up).
        # DOOM default pegging: texture top aligns with back floor —
        # tex_v_offset = 0.
        if seg.back_floor > seg.front_floor:
            bf_y = project_z(seg.back_floor, perp)
            lower_top = max(_round_y(bf_y), ceilingclip + 1)
            lower_bot = min(_round_y(fy_bot), floorclip)
            tex = _texture_for(textures, seg.lower_texture_id)
            tex_col = _tex_col_world(u, seg, tex)
            _paint_textured_strip(
                column,
                lower_top,
                lower_bot,
                bf_y,
                fy_bot,
                tex,
                tex_col,
                seg.color,
                world_height=seg.back_floor - seg.front_floor,
                tex_v_offset=0.0,
            )
            new_bot = _round_y(bf_y)
        else:
            new_bot = _round_y(fy_bot)
        if new_bot < floorclip:
            floorclip = new_bot
        # Continue to next hit through the gap.

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
            col,
            player_x,
            player_y,
            player_angle,
            segments,
            config,
            textures=textures,
        )

    return frame
