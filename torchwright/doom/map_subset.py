"""WAD map subset loader for the DOOM-in-transformer project.

The transformer has a fixed input size — at most ``max_walls`` SEGS and
``max_bsp_nodes`` BSP tree nodes per frame.  This module picks the
closest segs to the player, extracts the minimal BSP subtree covering
their subsectors, and precomputes the per-seg coefficients that turn
BSP traversal order into a sparse dot product over runtime player-side
decisions.

The key identity — derived in ``DOOM.md`` — is:

    rank(W) = dot(coeffs_W, side_P_vec) + const_W

where ``side_P_vec[i] ∈ {0, 1}`` is the runtime "which side of node i's
splitting plane is the player on?" and ``coeffs_W`` / ``const_W`` are
baked from the BSP structure and W's path through it.

Conventions used throughout this module:

- Plane equation: ``nx*x + ny*y + d``.  Encoding: ``nx=dy``, ``ny=-dx``,
  ``d = dx*py_node - dy*px_node`` (matches DOOM's original side
  classification).
- ``side_P[i] = 1`` iff player is on the FRONT side of node i's plane
  (``raw > 0``); ``side_P[i] = 0`` iff on the BACK side.
- ``side_W[i]`` is the side of node i that leads to W's subsector
  (0 = front child, 1 = back child).
- In BSP traversal, the "near" subtree is visited first: front if
  player on front, back if on back.  Segs in the "far" subtree come
  later.  Sum the sizes of near subtrees visited before W's subtree
  to get W's rank.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from torchwright.doom.wad import (
    SUBSECTOR_FLAG,
    BspNode,
    MapData,
    WADReader,
    sector_color,
    _assign_tex_id,
    _pick_seg_texture,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BspNodeSubset:
    """A BSP node's splitting plane expressed as an implicit line.

    The plane equation is ``nx*x + ny*y + d == 0``, and for any point
    ``(x, y)``:

        side_P = 1 iff nx*x + ny*y + d > 0 (player on FRONT side)
        side_P = 0 iff nx*x + ny*y + d <= 0 (BACK side)
    """

    nx: float
    ny: float
    d: float


@dataclass
class MapSubset:
    """Transformer-ready slice of a DOOM map from a player position.

    All segments, textures, and BSP nodes are capped to the limits
    passed to :func:`load_map_subset`.  Segments are ordered from
    closest to farthest (distance from the player).

    ``seg_bsp_coeffs`` has shape ``(N, max_bsp_nodes)`` — rows match
    ``segments``, columns match ``bsp_nodes``.  Unused columns (when
    fewer than ``max_bsp_nodes`` real nodes are present) are filled
    with zeros, so they contribute nothing to rank.
    """

    segments: List["Segment"]
    textures: List[np.ndarray]
    tex_name_to_id: Dict[str, int]
    bsp_nodes: List[BspNodeSubset]
    seg_bsp_coeffs: np.ndarray
    seg_bsp_consts: np.ndarray
    # Original seg indices in md.segs, in the same order as ``segments``.
    # Useful for cross-checking against the reference BSP traversal.
    original_seg_indices: List[int]


# ---------------------------------------------------------------------------
# Plane math
# ---------------------------------------------------------------------------


def _make_plane(node: BspNode) -> BspNodeSubset:
    """Convert DOOM's (px, py, dx, dy) representation to (nx, ny, d).

    Derivation: DOOM classifies a point on the FRONT side when
    ``dx*(y - py) < dy*(x - px)``.  Rearranging:

        dy*x - dx*y + (dx*py - dy*px) > 0   (front)

    So (nx, ny, d) = (dy, -dx, dx*py - dy*px) with the rule that
    ``nx*x + ny*y + d > 0 ⇒ FRONT``.
    """
    return BspNodeSubset(
        nx=float(node.dy),
        ny=float(-node.dx),
        d=float(node.dx) * float(node.py) - float(node.dy) * float(node.px),
    )


def side_P(plane: BspNodeSubset, px: float, py: float) -> int:
    """Classify a point against a BSP plane.

    Returns 1 if on the FRONT side (``raw > 0``), else 0.
    """
    raw = plane.nx * px + plane.ny * py + plane.d
    return 1 if raw > 0 else 0


# ---------------------------------------------------------------------------
# BSP subtree walk
# ---------------------------------------------------------------------------


def _decode_child(child_ref: int) -> Tuple[bool, int]:
    """Decode a BSP child reference. Returns (is_subsector, index)."""
    if child_ref & SUBSECTOR_FLAG:
        return True, child_ref & ~SUBSECTOR_FLAG
    return False, child_ref


def _walk_paths(
    md: MapData, root_node_idx: int
) -> Dict[int, List[Tuple[int, int]]]:
    """Return, for every subsector, its path from the BSP root.

    A path is a list of ``(node_idx, side)`` pairs where ``side`` is
    ``0`` when the traversal descended into the front child, ``1`` into
    the back child.  Paths are ordered root-first.
    """
    paths: Dict[int, List[Tuple[int, int]]] = {}

    def visit(is_ss: bool, idx: int, path: List[Tuple[int, int]]) -> None:
        if is_ss:
            paths[idx] = list(path)
            return
        node = md.nodes[idx]
        path.append((idx, 0))
        visit(*_decode_child(node.front_child), path)
        path.pop()
        path.append((idx, 1))
        visit(*_decode_child(node.back_child), path)
        path.pop()

    visit(False, root_node_idx, [])
    return paths


def _count_selected_in_subtree(
    md: MapData,
    selected_seg_indices: Set[int],
    root_node_idx: int,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Walk the BSP tree and count selected segs per subtree.

    Returns three dicts:
    - ``ss_count[ss_idx]``: number of selected segs in subsector ss_idx
    - ``front_count[node_idx]``: selected segs in node's front subtree
    - ``back_count[node_idx]``: selected segs in node's back subtree
    """
    ss_count: Dict[int, int] = {}
    front_count: Dict[int, int] = {}
    back_count: Dict[int, int] = {}

    def count(is_ss: bool, idx: int) -> int:
        if is_ss:
            if idx in ss_count:
                return ss_count[idx]
            ss = md.subsectors[idx]
            c = sum(
                1 for s in range(ss.first_seg, ss.first_seg + ss.seg_count)
                if s in selected_seg_indices
            )
            ss_count[idx] = c
            return c
        node = md.nodes[idx]
        f = count(*_decode_child(node.front_child))
        b = count(*_decode_child(node.back_child))
        front_count[idx] = f
        back_count[idx] = b
        return f + b

    count(False, root_node_idx)
    return ss_count, front_count, back_count


def _build_seg_to_subsector(md: MapData) -> Dict[int, int]:
    """Map each seg index to its owning subsector index."""
    mapping: Dict[int, int] = {}
    for ss_idx, ss in enumerate(md.subsectors):
        for seg_idx in range(ss.first_seg, ss.first_seg + ss.seg_count):
            mapping[seg_idx] = ss_idx
    return mapping


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def _seg_midpoint(md: MapData, seg_idx: int) -> Tuple[float, float]:
    seg = md.segs[seg_idx]
    v1 = md.vertices[seg.v1]
    v2 = md.vertices[seg.v2]
    return (v1.x + v2.x) * 0.5, (v1.y + v2.y) * 0.5


def _select_closest_segs(
    md: MapData, px: float, py: float, max_walls: int,
) -> List[int]:
    """Return original seg indices of up to ``max_walls`` closest segs."""
    ranked: List[Tuple[float, int]] = []
    for seg_idx in range(len(md.segs)):
        seg = md.segs[seg_idx]
        # Skip segs with invalid vertex refs (defensive)
        if seg.v1 >= len(md.vertices) or seg.v2 >= len(md.vertices):
            continue
        # Skip segs with invalid linedef refs
        if seg.linedef >= len(md.linedefs):
            continue
        # Skip segs whose sidedef is missing (e.g. back side of a
        # one-sided linedef)
        ld = md.linedefs[seg.linedef]
        sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        if sd_idx < 0 or sd_idx >= len(md.sidedefs):
            continue
        mx, my = _seg_midpoint(md, seg_idx)
        dist2 = (mx - px) ** 2 + (my - py) ** 2
        ranked.append((dist2, seg_idx))
    # Stable sort by distance, tie-break on seg index.
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [idx for _, idx in ranked[:max_walls]]


# ---------------------------------------------------------------------------
# Coefficient precomputation
# ---------------------------------------------------------------------------


def _compute_coefficients(
    md: MapData,
    selected_seg_indices: List[int],
    seg_to_ss: Dict[int, int],
    paths: Dict[int, List[Tuple[int, int]]],
    front_count: Dict[int, int],
    back_count: Dict[int, int],
    old_to_new_node: Dict[int, int],
    max_bsp_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (N, M) coefficient matrix and the (N,) constant vector.

    For each selected seg W at path [(node_i, side_W_i)]:

    - If ``side_W_i = 0`` (W in front subtree of node_i):
        coeffs[W, i] = -back_count[node_i]
        const[W]   += +back_count[node_i]
      Reason: contribution is ``(1 - side_P) * back_count`` when W is
      in the front subtree, i.e. segs behind (on the back side) come
      first iff player on back side.
    - If ``side_W_i = 1`` (W in back subtree of node_i):
        coeffs[W, i] = +front_count[node_i]
        const[W]   += 0
      Reason: contribution is ``side_P * front_count`` — front segs
      come first iff player on front side.
    """
    N = len(selected_seg_indices)
    coeffs = np.zeros((N, max_bsp_nodes), dtype=np.float64)
    consts = np.zeros(N, dtype=np.float64)
    for row, W_idx in enumerate(selected_seg_indices):
        ss_idx = seg_to_ss[W_idx]
        path = paths[ss_idx]
        for (old_node_idx, side_W) in path:
            new_id = old_to_new_node.get(old_node_idx)
            if new_id is None:
                # Ancestor not in subset — shouldn't happen, but guard
                # against it anyway.
                continue
            if side_W == 0:
                bc = back_count.get(old_node_idx, 0)
                coeffs[row, new_id] = -float(bc)
                consts[row] += float(bc)
            else:
                fc = front_count.get(old_node_idx, 0)
                coeffs[row, new_id] = float(fc)
    return coeffs, consts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_map_subset(
    wad_path: str,
    map_name: str,
    px: float,
    py: float,
    max_walls: int = 32,
    max_textures: int = 8,
    max_bsp_nodes: int = 48,
    tex_size: int = 8,
) -> MapSubset:
    """Load a transformer-ready subset of a DOOM map near ``(px, py)``.

    Selects the ``max_walls`` closest SEGS by Euclidean distance from
    their midpoint to ``(px, py)``.  Extracts the minimal BSP subtree
    whose leaves cover the selected segs' subsectors.  Precomputes the
    per-seg coefficients needed by the transformer's BSP rank sort.

    Raises ``ValueError`` if the required BSP subtree exceeds
    ``max_bsp_nodes`` — callers should increase the cap or reduce
    ``max_walls``.
    """
    from torchwright.reference_renderer.textures import downscale_texture
    from torchwright.reference_renderer.types import Segment

    wad = WADReader(wad_path)
    md = wad.get_map(map_name)

    if len(md.nodes) == 0:
        raise ValueError(
            f"map {map_name!r} has no BSP nodes — subset loader requires "
            "a non-trivial BSP"
        )

    # --- 1. Select N closest segs ---
    selected_orig: List[int] = _select_closest_segs(md, px, py, max_walls)
    selected_set: Set[int] = set(selected_orig)
    if not selected_orig:
        raise ValueError(
            f"no valid segs found near ({px}, {py}) in {map_name!r}"
        )

    # --- 2. Map segs to subsectors + collect selected subsectors ---
    seg_to_ss = _build_seg_to_subsector(md)
    selected_subsectors: Set[int] = {seg_to_ss[s] for s in selected_orig}

    # --- 3. Walk the BSP tree ---
    root_idx = len(md.nodes) - 1   # DOOM convention: root is last node
    paths = _walk_paths(md, root_idx)

    # --- 4. Extract minimal subtree ---
    subset_node_ids: Set[int] = set()
    for ss in selected_subsectors:
        for (node_idx, _side) in paths[ss]:
            subset_node_ids.add(node_idx)
    if len(subset_node_ids) > max_bsp_nodes:
        raise ValueError(
            f"BSP subtree has {len(subset_node_ids)} nodes but "
            f"max_bsp_nodes={max_bsp_nodes}; reduce max_walls or "
            "increase max_bsp_nodes"
        )
    # Renumber nodes, preserving original order (lower BSP indices first
    # means leaves come before root, which is a stable, readable order).
    sorted_old_ids = sorted(subset_node_ids)
    old_to_new_node: Dict[int, int] = {
        old: new for new, old in enumerate(sorted_old_ids)
    }

    bsp_nodes: List[BspNodeSubset] = [
        _make_plane(md.nodes[old_id]) for old_id in sorted_old_ids
    ]

    # --- 5. Count selected segs in each subtree ---
    _ss_count, front_count, back_count = _count_selected_in_subtree(
        md, selected_set, root_idx,
    )

    # --- 6. Compute coefficients per seg ---
    coeffs, consts = _compute_coefficients(
        md, selected_orig, seg_to_ss, paths,
        front_count, back_count,
        old_to_new_node, max_bsp_nodes,
    )

    # --- 7. Convert selected segs to Segment objects ---
    segments: List[Segment] = []
    name_to_id: Dict[str, int] = {}
    seen_tex_names: List[str] = []  # preserve first-sight order
    for W_idx in selected_orig:
        seg = md.segs[W_idx]
        ld = md.linedefs[seg.linedef]
        sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        sd = md.sidedefs[sd_idx]
        tex_name = _pick_seg_texture(sd)
        tex_id = _assign_tex_id(tex_name, name_to_id)
        if tex_name not in ("-", "") and tex_name not in seen_tex_names:
            seen_tex_names.append(tex_name)
        v1 = md.vertices[seg.v1]
        v2 = md.vertices[seg.v2]
        color = sector_color(sd.sector)
        segments.append(Segment(
            ax=float(v1.x), ay=float(v1.y),
            bx=float(v2.x), by=float(v2.y),
            color=color, texture_id=tex_id,
        ))

    # --- 8. Load textures, capped at max_textures ---
    # Prioritize by appearance order (closer segs saw them first).
    kept_names: List[str] = []
    textures: List[np.ndarray] = []
    for name in seen_tex_names:
        if len(kept_names) >= max_textures:
            break
        tex = wad.get_texture(name)
        if tex is None:
            continue
        kept_names.append(name)
        textures.append(downscale_texture(tex, tex_size, tex_size))

    # Remap texture ids so that only kept textures get valid ids; the
    # rest become -1 (so the renderer falls back to solid color).
    new_name_to_id: Dict[str, int] = {
        name: i for i, name in enumerate(kept_names)
    }
    remapped: List[Segment] = []
    for seg in segments:
        if seg.texture_id < 0:
            remapped.append(seg)
            continue
        old_name = _reverse_lookup(name_to_id, seg.texture_id)
        new_id = new_name_to_id.get(old_name, -1)
        remapped.append(Segment(
            ax=seg.ax, ay=seg.ay, bx=seg.bx, by=seg.by,
            color=seg.color, texture_id=new_id,
        ))
    segments = remapped

    return MapSubset(
        segments=segments,
        textures=textures,
        tex_name_to_id=new_name_to_id,
        bsp_nodes=bsp_nodes,
        seg_bsp_coeffs=coeffs,
        seg_bsp_consts=consts,
        original_seg_indices=list(selected_orig),
    )


def _reverse_lookup(d: Dict[str, int], value: int) -> str:
    """Find a key by value (used for small dicts during texture remap)."""
    for k, v in d.items():
        if v == value:
            return k
    return "-"


# ---------------------------------------------------------------------------
# Trivial subsets (for test fixtures)
# ---------------------------------------------------------------------------


def make_trivial_subset(
    segments: List["Segment"],
    textures: List[np.ndarray],
    max_bsp_nodes: int = 48,
    tex_name_to_id: Optional[Dict[str, int]] = None,
) -> MapSubset:
    """Build a :class:`MapSubset` with **no BSP structure**.

    All BSP coefficients are zero; per-seg constants are set to the
    seg index, producing a deterministic sort order matching the
    input order — regardless of player position.

    This is **not BSP-correct** — it gives wrong occlusion order when
    walls overlap on screen.  But for test scenes that rely on
    non-overlapping walls (box rooms, simple multi-room), it
    reproduces the previous distance-based sort behavior well enough
    to keep regression tests stable while the graph transitions to
    BSP-based ordering.
    """
    N = len(segments)
    coeffs = np.zeros((N, max_bsp_nodes), dtype=np.float64)
    # const_W = seg index → stable sort preserves input order
    consts = np.arange(N, dtype=np.float64)
    return MapSubset(
        segments=list(segments),
        textures=list(textures),
        tex_name_to_id=dict(tex_name_to_id or {}),
        bsp_nodes=[],
        seg_bsp_coeffs=coeffs,
        seg_bsp_consts=consts,
        original_seg_indices=list(range(N)),
    )


def subset_from_walls(
    walls: List[dict],
    textures: List[np.ndarray],
    max_bsp_nodes: int = 48,
) -> MapSubset:
    """Convert a list of wall dicts (old API) into a trivial MapSubset.

    Accepts the ``[{"ax","ay","bx","by","tex_id"}, ...]`` shape used
    by existing test fixtures and scene helpers.  The resulting
    subset has no BSP structure — see :func:`make_trivial_subset`.
    """
    from torchwright.reference_renderer.types import Segment

    segments = [
        Segment(
            ax=float(w["ax"]), ay=float(w["ay"]),
            bx=float(w["bx"]), by=float(w["by"]),
            color=(0.5, 0.5, 0.5),
            texture_id=int(w.get("tex_id", -1)),
        )
        for w in walls
    ]
    return make_trivial_subset(segments, textures, max_bsp_nodes=max_bsp_nodes)


def _central_ray_sort_scores(
    walls: List[dict],
    player_x: float,
    player_y: float,
    player_angle: int,
    trig_table: np.ndarray,
    sentinel: float = 99.0,
) -> np.ndarray:
    """Compute the pre-BSP distance-based sort scores per wall.

    Reproduces the exact math previously in ``wall/sort_score`` in
    ``game_graph.py``: for each wall the central viewing ray's
    signed intersection distance (``t = num_t / den`` with positive
    validity checks) is used as the sort score.  Invalid or behind-
    player intersections get the sentinel (large) value.
    """
    cos_a = float(trig_table[player_angle % 256, 0])
    sin_a = float(trig_table[player_angle % 256, 1])
    scores = np.full(len(walls), sentinel, dtype=np.float64)
    for i, w in enumerate(walls):
        ax = float(w["ax"]); ay = float(w["ay"])
        bx = float(w["bx"]); by = float(w["by"])
        ex, ey = bx - ax, by - ay
        fx = ax - player_x
        gy = player_y - ay
        den = ey * cos_a - ex * sin_a
        num_t = ey * fx + ex * gy
        if abs(den) < 0.05:
            continue
        # Mirror the graph's sign-adjustment so positive adj_num_t
        # means "in front" irrespective of ``den``'s sign.
        if den > 0:
            adj_num_t, abs_den = num_t, den
        else:
            adj_num_t, abs_den = -num_t, -den
        if adj_num_t <= 0.0:
            continue
        scores[i] = min(adj_num_t / abs_den, 1000.0)
    return scores


def subset_from_walls_with_sort(
    walls: List[dict],
    textures: List[np.ndarray],
    player_x: float,
    player_y: float,
    player_angle: int,
    trig_table: np.ndarray,
    max_bsp_nodes: int = 48,
) -> MapSubset:
    """Build a trivial MapSubset whose ranks match the old distance sort.

    Coefficients are zero (no BSP structure), but each wall's constant
    is set to the central-ray sort score.  This preserves the sort
    order produced by the old ``wall/sort_score`` block, so legacy
    callers (tests, walkthrough, play) that pass only a wall list plus
    player state still produce the same pixel output after BSP
    integration.

    Use this instead of :func:`subset_from_walls` when the caller has
    access to the player's state and wants BSP-integration-era sort
    correctness without building a real BSP tree.
    """
    from torchwright.reference_renderer.types import Segment

    N = len(walls)
    segments = [
        Segment(
            ax=float(w["ax"]), ay=float(w["ay"]),
            bx=float(w["bx"]), by=float(w["by"]),
            color=(0.5, 0.5, 0.5),
            texture_id=int(w.get("tex_id", -1)),
        )
        for w in walls
    ]
    consts = _central_ray_sort_scores(
        walls, player_x, player_y, player_angle, trig_table,
    )
    coeffs = np.zeros((N, max_bsp_nodes), dtype=np.float64)
    return MapSubset(
        segments=segments,
        textures=list(textures),
        tex_name_to_id={},
        bsp_nodes=[],
        seg_bsp_coeffs=coeffs,
        seg_bsp_consts=consts,
        original_seg_indices=list(range(N)),
    )


# ---------------------------------------------------------------------------
# Helpers for tests (reference BSP traversal)
# ---------------------------------------------------------------------------


def bsp_traversal_order(
    md: MapData,
    px: float,
    py: float,
    selected_seg_indices: Optional[Set[int]] = None,
) -> List[int]:
    """Reference BSP front-to-back traversal for testing.

    Walks the full BSP and returns the seg indices in the order DOOM
    would render them.  If ``selected_seg_indices`` is provided, the
    output is filtered to just those segs (preserving order).

    This is the ground-truth comparator for ``rank`` math — if the
    precomputed coefficients are right, sorting selected segs by rank
    should reproduce the filtered traversal.
    """
    out: List[int] = []

    def visit(is_ss: bool, idx: int) -> None:
        if is_ss:
            ss = md.subsectors[idx]
            for seg_idx in range(ss.first_seg, ss.first_seg + ss.seg_count):
                if selected_seg_indices is None or seg_idx in selected_seg_indices:
                    out.append(seg_idx)
            return
        node = md.nodes[idx]
        plane = _make_plane(node)
        if side_P(plane, px, py) == 1:
            # Player on FRONT → visit front first
            visit(*_decode_child(node.front_child))
            visit(*_decode_child(node.back_child))
        else:
            visit(*_decode_child(node.back_child))
            visit(*_decode_child(node.front_child))

    visit(False, len(md.nodes) - 1)
    return out
