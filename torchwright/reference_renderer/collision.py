"""Collision detection for player movement against wall segments."""

from typing import List, Tuple

from torchwright.reference_renderer.types import Segment


def point_segment_distance(px: float, py: float, seg: Segment) -> float:
    """Minimum distance from point (px, py) to the line segment.

    Projects the point onto the segment line, clamps to [0, 1], and
    returns the Euclidean distance to the nearest point on the segment.
    """
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay
    seg_len_sq = ex * ex + ey * ey

    if seg_len_sq == 0.0:
        # Degenerate segment (zero length)
        dx = px - seg.ax
        dy = py - seg.ay
        return (dx * dx + dy * dy) ** 0.5

    # Parameter t: projection of point onto segment line
    t = ((px - seg.ax) * ex + (py - seg.ay) * ey) / seg_len_sq
    t = max(0.0, min(1.0, t))

    # Nearest point on segment
    nearest_x = seg.ax + t * ex
    nearest_y = seg.ay + t * ey

    dx = px - nearest_x
    dy = py - nearest_y
    return (dx * dx + dy * dy) ** 0.5


def _point_clear(
    px: float, py: float, segments: List[Segment], margin: float,
) -> bool:
    """Return True if (px, py) is at least *margin* away from all segments."""
    for seg in segments:
        if point_segment_distance(px, py, seg) < margin:
            return False
    return True


def resolve_collision(
    old_x: float,
    old_y: float,
    new_x: float,
    new_y: float,
    segments: List[Segment],
    margin: float = 0.2,
) -> Tuple[float, float]:
    """Resolve player movement against wall segments with wall sliding.

    Tries the full move first. If blocked, tries X-only and Y-only
    independently. This gives natural wall-sliding behavior.

    Args:
        old_x, old_y: Current player position (assumed collision-free).
        new_x, new_y: Desired new position.
        segments: Wall segments to collide against.
        margin: Minimum distance the player must maintain from walls.

    Returns:
        (resolved_x, resolved_y) -- the actual position after collision.
    """
    # Try the full move
    if _point_clear(new_x, new_y, segments, margin):
        return new_x, new_y

    # Try X-only move
    resolved_x = new_x if _point_clear(new_x, old_y, segments, margin) else old_x

    # Try Y-only move
    resolved_y = new_y if _point_clear(old_x, new_y, segments, margin) else old_y

    return resolved_x, resolved_y


# ---------------------------------------------------------------------------
# Ray-based collision (matches the compiled transformer implementation)
# ---------------------------------------------------------------------------


def _ray_hits_segment(
    ox: float, oy: float, dx: float, dy: float, seg: Segment,
) -> bool:
    """Test if a movement ray crosses a wall segment.

    The ray goes from (ox, oy) with direction (dx, dy).  A hit requires
    0 < t < 1 (intersection within the movement) and 0 <= u <= 1
    (intersection on the segment).
    """
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay

    den = dx * ey - dy * ex
    if den == 0.0:
        return False

    fx = seg.ax - ox
    fy = seg.ay - oy
    num_t = fx * ey - fy * ex
    num_u = fx * dy - fy * dx

    # t in (0, 1]: hit within the movement step (inclusive at endpoint
    # to prevent the player from landing exactly on a wall).
    # u in [0, 1]: hit is on the wall segment.
    if den > 0.0:
        if num_t <= 0.0 or num_t > den:
            return False
        if num_u < 0.0 or num_u > den:
            return False
    else:
        if num_t >= 0.0 or num_t < den:
            return False
        if num_u > 0.0 or num_u < den:
            return False

    return True


def ray_hits_any_segment(
    ox: float, oy: float, dx: float, dy: float, segments: List[Segment],
) -> bool:
    """Return True if the movement ray (ox,oy)→(ox+dx,oy+dy) crosses any segment."""
    for seg in segments:
        if _ray_hits_segment(ox, oy, dx, dy, seg):
            return True
    return False


def resolve_collision_ray(
    old_x: float,
    old_y: float,
    new_x: float,
    new_y: float,
    segments: List[Segment],
) -> Tuple[float, float]:
    """Resolve collision using ray-segment intersection with wall sliding.

    This matches the transformer's compiled collision detection: test
    the movement vector against all wall segments.  If the full move
    crosses a wall, try axis-separated movement for wall sliding.

    Args:
        old_x, old_y: Current player position.
        new_x, new_y: Desired new position.
        segments: Wall segments to collide against.

    Returns:
        (resolved_x, resolved_y) after collision resolution.
    """
    dx = new_x - old_x
    dy = new_y - old_y

    if not ray_hits_any_segment(old_x, old_y, dx, dy, segments):
        return new_x, new_y

    # Wall sliding: try each axis independently
    resolved_x = new_x if not ray_hits_any_segment(old_x, old_y, dx, 0.0, segments) else old_x
    resolved_y = new_y if not ray_hits_any_segment(old_x, old_y, 0.0, dy, segments) else old_y

    return resolved_x, resolved_y
