"""Pure-Python reference for the rollout assertions.

Recomputes the contract VALUEs the compiled graph should emit:

* per-wall ``BSP_RANK``, ``IS_RENDERABLE``, ``VIS_LO``, ``VIS_HI``,
  ``HIT_FULL``, ``HIT_X``, ``HIT_Y``;
* frame-level ``RESOLVED_X/Y/ANGLE``;
* the renderable ``sort_order`` (BSP-ascending list of renderable wall
  indices) and the per-sort-position ``RENDER`` token count.

The references intentionally use the same predicates the graph runs
(central-ray ``|sort_den| > 0.05`` for ``IS_RENDERABLE``, running-OR
collision flags) rather than ``project_wall``'s broader per-column
test.  See the rollout plan's "Risks / open" notes for why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import MapSubset
from torchwright.reference_renderer.render import (
    _ray_angle_for_column,
    intersect_ray_segment,
    project_wall,
)
from torchwright.reference_renderer.types import RenderConfig, Segment


# Match the thinking-wall stage's ``_compute_bsp_and_renderable`` central-ray
# gate (``|sort_den| > 0.05``) — broader than ``project_wall``'s per-column
# test.
_REND_DEN_EPS = 0.05
# Hit-validity epsilon mirrors the thinking-wall stage's ``_validity``
# helper (which mirrors collision math in the reference renderer).
_HIT_EPS = 0.05


def _player_velocity(
    angle: int,
    inputs: PlayerInput,
    trig_table: np.ndarray,
    move_speed: float,
    turn_speed: int,
) -> Tuple[float, float]:
    """Compute (vel_x, vel_y) the way the INPUT stage broadcasts."""
    new_angle = angle
    if inputs.turn_left:
        new_angle -= turn_speed
    if inputs.turn_right:
        new_angle += turn_speed
    new_angle = new_angle % 256

    cos_t = float(trig_table[new_angle, 0])
    sin_t = float(trig_table[new_angle, 1])

    fwd = (1.0 if inputs.forward else 0.0) - (1.0 if inputs.backward else 0.0)
    strafe = (1.0 if inputs.strafe_right else 0.0) - (
        1.0 if inputs.strafe_left else 0.0
    )

    vel_x = move_speed * (fwd * cos_t - strafe * sin_t)
    vel_y = move_speed * (fwd * sin_t + strafe * cos_t)
    return vel_x, vel_y


def _ref_collision_validity(
    den: float, num_t: float, num_u: float, eps: float = _HIT_EPS
) -> bool:
    """Ray-segment intersection validity, mirroring thinking-wall's
    ``_validity`` (epsilon-margined parametric checks)."""
    sign_den = 1.0 if den > 0 else -1.0
    adj_t = num_t * sign_den
    adj_u = num_u * sign_den
    abs_den = abs(den)
    return (
        abs_den > eps
        and adj_t > eps
        and (abs_den - adj_t) > -eps
        and adj_u > -eps
        and (abs_den - adj_u) > -eps
    )


def _ref_hits(
    seg: Segment, px: float, py: float, vx: float, vy: float
) -> Tuple[int, int, int]:
    """Per-wall (hit_full, hit_x, hit_y) triple — local hit math, before
    the running-OR accumulation across walls."""
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay
    dax = seg.ax - px
    day = seg.ay - py

    p_dx_ey = vx * ey
    p_dy_ex = vy * ex
    p_dax_ey = dax * ey
    p_day_ex = day * ex
    p_dax_dy = dax * vy
    p_day_dx = day * vx

    num_t = p_dax_ey - p_day_ex

    den_full = p_dx_ey - p_dy_ex
    num_u_full = p_dax_dy - p_day_dx
    hit_full = _ref_collision_validity(den_full, num_t, num_u_full)

    den_x = p_dx_ey
    num_u_x = -p_day_dx
    hit_x = _ref_collision_validity(den_x, num_t, num_u_x)

    den_y = -p_dy_ex
    num_u_y = p_dax_dy
    hit_y = _ref_collision_validity(den_y, num_t, num_u_y)

    return int(hit_full), int(hit_x), int(hit_y)


def _ref_bsp_ranks(subset: MapSubset, px: float, py: float) -> np.ndarray:
    """Reproduce the wall stage's BSP-rank dot product."""
    n_cols = subset.seg_bsp_coeffs.shape[1]
    side_P_vec = np.zeros(n_cols)
    for i, node in enumerate(subset.bsp_nodes):
        val = node.nx * px + node.ny * py + node.d
        side_P_vec[i] = 1.0 if val > 0 else 0.0
    return subset.seg_bsp_coeffs @ side_P_vec + subset.seg_bsp_consts


def _central_ray_renderable(
    seg: Segment, px: float, py: float, angle: int, trig_table: np.ndarray
) -> bool:
    """Graph-side renderability: ``|sort_den| > 0.05`` for the
    central-ray cast at ``angle``, plus ``num_t * sign(den) > 0``.

    Matches ``thinking_wall._compute_bsp_and_renderable``.  Broader than
    ``project_wall`` (which scans every column).  This is the predicate
    the graph encodes into ``IS_RENDERABLE``."""
    cos_a = float(trig_table[angle % 256, 0])
    sin_a = float(trig_table[angle % 256, 1])

    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay
    fx = seg.ax - px
    gy = py - seg.ay

    sort_den = ey * cos_a - ex * sin_a
    sort_num_t = ey * fx + ex * gy

    if abs(sort_den) <= _REND_DEN_EPS:
        return False
    sign_den = 1.0 if sort_den > 0 else -1.0
    adj_num_t = sort_num_t * sign_den
    return adj_num_t > 0.0


@dataclass
class WallRef:
    """Per-wall reference values that the rollout asserts against."""

    bsp_rank: int
    is_renderable: bool  # central-ray check (graph IS_RENDERABLE token)
    is_projected: bool  # project_wall returned non-None (per-column rays)
    vis_lo: float  # only meaningful when is_projected
    vis_hi: float
    hit_full: bool  # running-OR across walls 0..i
    hit_x: bool
    hit_y: bool


@dataclass
class RenderTokenExpectation:
    """Expected per-RENDER overflow values for one (wall, col, chunk) triple.

    Mirrors the float arithmetic in ``render._chunk_fill``:

        vis_top    = clamp(wall_top_f,    0, H)
        vis_bottom = clamp(wall_bottom_f, 0, H)
        active_start = vis_top + chunk_k * chunk_size
        chunk_length = clamp(vis_bottom - active_start, 0, chunk_size)

    where ``wall_top_f = H/2 - H/(2 * perp_distance)``.  ``start`` and
    ``length`` are int-rounded to match the host's ``int(round(...))``
    extraction in ``step_frame``.
    """

    col: int
    start: int
    length: int
    chunk_k: int


@dataclass
class Reference:
    """Full reference snapshot for one scenario.

    ``sort_order`` is the *graph-expected* sort sequence.  The graph
    iterates ``wall_counter`` from 0, picking the renderable wall
    whose ``bsp_rank`` matches.  When a rank is missing among
    renderable walls the iteration exhausts (sort_done fires).  So
    ``sort_order`` is the prefix of renderable walls whose ranks are
    consecutive starting from 0.

    ``render_count_per_sort_slot`` mirrors that prefix: one entry per
    ``sort_order`` element, set to the wall's ``vis_hi - vis_lo + 1``
    when that wall is project-visible (else 0 — the slot still emits
    RENDERs but the count comparison is skipped via ``is_projected``).

    ``render_tokens_per_sort_slot`` is the per-slot list of expected
    ``RenderTokenExpectation`` entries, one per (col, chunk_k) pair the
    chunk loop should visit for that wall — computed from the graph's
    chunk-fill arithmetic, not from the reference renderer's
    ``render_wall_column`` (which uses a different rounding rule).
    Slots whose wall is not project-visible get an empty list.
    """

    walls: List[WallRef]
    resolved_x: float
    resolved_y: float
    resolved_angle: float
    sort_order: List[int]
    render_count_per_sort_slot: List[int]
    render_tokens_per_sort_slot: List[List[RenderTokenExpectation]]


def _render_tokens_for_wall(
    seg: Segment,
    px: float,
    py: float,
    angle: int,
    config: RenderConfig,
    chunk_size: int,
) -> List[RenderTokenExpectation]:
    """Expected RENDER overflow tuples for one wall, one full sort slot.

    Iterates every screen column; for each column where the column ray
    hits the wall, emits one entry per chunk_k produced by the graph's
    chunk loop.  The loop stops when the graph's ``has_more_chunks``
    predicate flips false:

        has_more_chunks ⇔ wall_bottom_clamped - (active_start + chunk_size) > 0.5

    Equivalently, after emitting chunk K the loop stops if
    ``vis_bottom - vis_top - (K+1)*chunk_size <= 0.5``.

    Columns with no hit (or perp_distance ≤ 0) emit no entries — those
    columns don't reach the chunk loop.
    """
    H = config.screen_height
    out: List[RenderTokenExpectation] = []
    for col in range(config.screen_width):
        ray_angle = _ray_angle_for_column(col, angle, config)
        ray_cos = float(config.trig_table[ray_angle, 0])
        ray_sin = float(config.trig_table[ray_angle, 1])
        hit = intersect_ray_segment(px, py, ray_cos, ray_sin, seg)
        if hit is None:
            continue
        t, _u = hit
        angle_diff = (ray_angle - angle) % 256
        perp_cos = float(config.trig_table[angle_diff, 0])
        perp_distance = t * perp_cos
        if perp_distance <= 0.0:
            continue
        wall_top_f = H / 2.0 - H / (2.0 * perp_distance)
        wall_bottom_f = H / 2.0 + H / (2.0 * perp_distance)
        vis_top = max(0.0, min(float(H), wall_top_f))
        vis_bottom = max(0.0, min(float(H), wall_bottom_f))
        k = 0
        while True:
            active_start = vis_top + k * chunk_size
            length = max(0.0, min(float(chunk_size), vis_bottom - active_start))
            out.append(
                RenderTokenExpectation(
                    col=col,
                    start=int(round(active_start)),
                    length=int(round(length)),
                    chunk_k=k,
                )
            )
            # Graph's has_more_chunks predicate: > 0.5 margin guards
            # against rounding flicker at the chunk boundary.
            if vis_bottom - (active_start + chunk_size) <= 0.5:
                break
            k += 1
    return out


def compute_reference(
    *,
    px: float,
    py: float,
    angle: int,
    inputs: PlayerInput,
    subset: MapSubset,
    config: RenderConfig,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    chunk_size: int = 20,
) -> Reference:
    """Build a :class:`Reference` for the given scenario."""
    trig = config.trig_table
    segs = subset.segments

    bsp_ranks = _ref_bsp_ranks(subset, px, py)
    vx, vy = _player_velocity(angle, inputs, trig, move_speed, turn_speed)

    walls: List[WallRef] = []
    running_hf = 0
    running_hx = 0
    running_hy = 0
    for i, seg in enumerate(segs):
        local_hf, local_hx, local_hy = _ref_hits(seg, px, py, vx, vy)
        running_hf |= local_hf
        running_hx |= local_hx
        running_hy |= local_hy

        is_rend = _central_ray_renderable(seg, px, py, angle, trig)

        # vis_lo/vis_hi from project_wall (per-column ray test).  The
        # graph and the reference can disagree by ±2 columns at glancing
        # angles; the rollout test uses ``abs=2`` tolerance on this
        # comparison.  When project_wall returns None the comparison is
        # skipped — the graph's central-ray gate is broader than the
        # per-column gate, so the graph can compute vis_lo/vis_hi from
        # the wall plane intersection even when no actual screen column
        # hits the wall.
        proj = project_wall(px, py, angle, seg, config)
        is_projected = proj is not None
        if is_projected:
            vis_lo = float(proj.vis_lo)
            vis_hi = float(proj.vis_hi)
        else:
            vis_lo = -2.0
            vis_hi = -2.0

        walls.append(
            WallRef(
                bsp_rank=int(round(bsp_ranks[i])),
                is_renderable=is_rend,
                is_projected=is_projected,
                vis_lo=vis_lo,
                vis_hi=vis_hi,
                hit_full=bool(running_hf),
                hit_x=bool(running_hx),
                hit_y=bool(running_hy),
            )
        )

    # Resolved state via the same ``update_state`` the existing pipeline
    # tests already use.
    state = GameState(x=px, y=py, angle=angle, move_speed=move_speed, turn_speed=turn_speed)
    new_state = update_state(state, inputs, segs, trig)

    # Graph-expected sort order: iterate wall_counter from 0, pick
    # the renderable wall with that bsp_rank.  When a rank is missing
    # among renderable walls, the SORTED quadratic-equality attention
    # picks the closest existing rank (a previously-emitted wall) and
    # ``sort_done`` fires.  So we take the prefix of consecutive ranks.
    rank_to_wall: dict = {}
    for i, w in enumerate(walls):
        if w.is_renderable:
            rank_to_wall.setdefault(w.bsp_rank, i)
    sort_order: List[int] = []
    n = 0
    while n in rank_to_wall:
        sort_order.append(rank_to_wall[n])
        n += 1

    # RENDER token count per sort slot.  At chunk_size >= H the chunk
    # loop emits exactly one RENDER per visible column, so the count
    # collapses to ``vis_hi - vis_lo + 1`` for project-visible walls.
    # For sort-order walls without a project_wall projection, set the
    # count to 0 — the slot still emits some RENDERs (driven by the
    # graph's FOV-clipped vis_lo/vis_hi) but the test skips the count
    # comparison via ``is_projected``.
    # Render-time projection uses POST-step state — the SORTED + RENDER
    # phase reads the resolved player position from PLAYER_X/Y/ANGLE,
    # not the pre-step input.  Pre- and post-step differ by one
    # ``move_speed`` step under ``forward=True`` etc., which shifts
    # perp_distance enough to flip a chunk-loop boundary by ~2 rows.
    render_px = float(new_state.x)
    render_py = float(new_state.y)
    render_angle = int(new_state.angle) % 256

    render_count_per_sort_slot: List[int] = []
    render_tokens_per_sort_slot: List[List[RenderTokenExpectation]] = []
    for wall_i in sort_order:
        ref = walls[wall_i]
        if ref.is_projected:
            render_count_per_sort_slot.append(int(round(ref.vis_hi - ref.vis_lo + 1)))
            render_tokens_per_sort_slot.append(
                _render_tokens_for_wall(
                    segs[wall_i],
                    render_px,
                    render_py,
                    render_angle,
                    config,
                    chunk_size,
                )
            )
        else:
            render_count_per_sort_slot.append(0)
            render_tokens_per_sort_slot.append([])

    return Reference(
        walls=walls,
        resolved_x=float(new_state.x),
        resolved_y=float(new_state.y),
        resolved_angle=float(new_state.angle),
        sort_order=sort_order,
        render_count_per_sort_slot=render_count_per_sort_slot,
        render_tokens_per_sort_slot=render_tokens_per_sort_slot,
    )
