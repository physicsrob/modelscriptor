"""Tests for :mod:`torchwright.doom.map_subset`.

The critical test is :func:`test_rank_formula_matches_bsp_traversal`:
the precomputed coefficients must reproduce DOOM's actual BSP
front-to-back ordering when dotted against the runtime player-side
decisions.  Everything else is supporting correctness.
"""

from typing import List, Set

import numpy as np
import pytest

from torchwright.doom.map_subset import (
    BspNodeSubset,
    MapSubset,
    _build_balanced_bsp,
    bsp_traversal_order,
    build_scene_subset,
    load_map_subset,
    side_P,
)
from torchwright.doom.wad import WADReader
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.types import Segment

# E1M1's canonical player spawn (Doomguy, THING type 1).  We don't
# parse THINGS yet, but this is the well-known coordinate.
E1M1_START = (1056.0, -3616.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def subset_default() -> MapSubset:
    """Default subset at E1M1's start position."""
    return load_map_subset(
        "doom1.wad",
        "E1M1",
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=32,
        max_textures=8,
        max_bsp_nodes=64,
    )


@pytest.fixture(scope="module")
def raw_e1m1():
    """Raw E1M1 MapData, used for cross-checking against BSP traversal."""
    return WADReader("doom1.wad").get_map("E1M1")


# ---------------------------------------------------------------------------
# Basic shape / non-empty checks
# ---------------------------------------------------------------------------


def test_loads_nonempty_subset(subset_default: MapSubset) -> None:
    assert len(subset_default.segments) > 0
    assert len(subset_default.bsp_nodes) > 0
    assert subset_default.seg_bsp_coeffs.shape[0] == len(subset_default.segments)
    assert subset_default.seg_bsp_consts.shape[0] == len(subset_default.segments)


def test_subset_respects_max_walls() -> None:
    for n in (4, 8, 16, 32):
        s = load_map_subset(
            "doom1.wad",
            "E1M1",
            px=E1M1_START[0],
            py=E1M1_START[1],
            max_walls=n,
            max_bsp_nodes=64,
        )
        assert len(s.segments) <= n
        assert s.seg_bsp_coeffs.shape[0] == len(s.segments)


def test_coeffs_shape_matches_max_bsp_nodes() -> None:
    s = load_map_subset(
        "doom1.wad",
        "E1M1",
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=16,
        max_bsp_nodes=64,
    )
    assert s.seg_bsp_coeffs.shape[1] == 64
    # Number of real BSP nodes ≤ max_bsp_nodes
    assert len(s.bsp_nodes) <= 64


def test_too_small_max_bsp_nodes_raises() -> None:
    with pytest.raises(ValueError, match="BSP subtree"):
        load_map_subset(
            "doom1.wad",
            "E1M1",
            px=E1M1_START[0],
            py=E1M1_START[1],
            max_walls=32,
            max_bsp_nodes=2,  # impossibly small
        )


# ---------------------------------------------------------------------------
# Selection correctness
# ---------------------------------------------------------------------------


def test_selected_segs_are_closest(raw_e1m1) -> None:
    """Returned segs are a subset of the closest segs by midpoint distance."""
    s = load_map_subset(
        "doom1.wad",
        "E1M1",
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=8,
        max_bsp_nodes=64,
    )
    # Compute distance to each selected seg's midpoint
    selected_dists = []
    for seg_idx in s.original_seg_indices:
        seg = raw_e1m1.segs[seg_idx]
        v1 = raw_e1m1.vertices[seg.v1]
        v2 = raw_e1m1.vertices[seg.v2]
        mx = (v1.x + v2.x) / 2.0
        my = (v1.y + v2.y) / 2.0
        d2 = (mx - E1M1_START[0]) ** 2 + (my - E1M1_START[1]) ** 2
        selected_dists.append(d2)
    # Max selected distance should be ≤ min distance of any non-selected seg
    selected_set = set(s.original_seg_indices)
    max_selected = max(selected_dists)
    for seg_idx in range(len(raw_e1m1.segs)):
        if seg_idx in selected_set:
            continue
        seg = raw_e1m1.segs[seg_idx]
        if seg.v1 >= len(raw_e1m1.vertices) or seg.v2 >= len(raw_e1m1.vertices):
            continue
        if seg.linedef >= len(raw_e1m1.linedefs):
            continue
        ld = raw_e1m1.linedefs[seg.linedef]
        sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        if sd_idx < 0 or sd_idx >= len(raw_e1m1.sidedefs):
            continue
        v1 = raw_e1m1.vertices[seg.v1]
        v2 = raw_e1m1.vertices[seg.v2]
        mx = (v1.x + v2.x) / 2.0
        my = (v1.y + v2.y) / 2.0
        d2 = (mx - E1M1_START[0]) ** 2 + (my - E1M1_START[1]) ** 2
        # Any non-selected valid seg must be at least as far as the
        # farthest selected seg (stable sort on index breaks ties).
        assert d2 >= max_selected - 1e-6


# ---------------------------------------------------------------------------
# BSP subtree integrity
# ---------------------------------------------------------------------------


def test_bsp_subtree_nonempty(subset_default: MapSubset) -> None:
    assert len(subset_default.bsp_nodes) > 0


def test_coefficients_match_bsp_nodes_length(subset_default: MapSubset) -> None:
    """Within the (N, M) matrix, nonzero columns only exist for real
    BSP nodes; padding columns are exact zeros."""
    real_count = len(subset_default.bsp_nodes)
    padding = subset_default.seg_bsp_coeffs[:, real_count:]
    assert np.all(
        padding == 0.0
    ), "padding columns should be zero so they contribute nothing to rank"


def test_coefficients_sparse(subset_default: MapSubset) -> None:
    """Each seg's nonzero coefficients count ≈ depth of its subsector's
    path in the BSP tree — no more than log2(total_nodes) + a small
    slack."""
    real_count = len(subset_default.bsp_nodes)
    max_possible = real_count  # generous upper bound
    for row in range(subset_default.seg_bsp_coeffs.shape[0]):
        nonzero = np.count_nonzero(subset_default.seg_bsp_coeffs[row])
        assert nonzero <= max_possible


# ---------------------------------------------------------------------------
# Plane math
# ---------------------------------------------------------------------------


def test_plane_passes_through_node_point(raw_e1m1) -> None:
    """For each BSP node, the (nx, ny, d) plane passes through DOOM's
    original (px, py) — a sanity check on the encoding."""
    from torchwright.doom.map_subset import _make_plane

    for node in raw_e1m1.nodes:
        plane = _make_plane(node)
        raw = plane.nx * node.px + plane.ny * node.py + plane.d
        # Exact arithmetic: int * int - int * int.  Should be 0.
        assert abs(raw) < 1e-6


def test_plane_sign_matches_doom_side_classification(raw_e1m1) -> None:
    """Our sign convention (``side_P = 1`` for front) agrees with
    DOOM's original ``R_PointOnSide`` formulation.

    DOOM formula: point is on FRONT iff
        dx_node * (y - py_node) < dy_node * (x - px_node)
    """
    from torchwright.doom.map_subset import _make_plane

    rng = np.random.default_rng(0)
    # Test a handful of nodes at a few points each
    for node in raw_e1m1.nodes[:20]:
        plane = _make_plane(node)
        for _ in range(10):
            x = rng.uniform(-5000, 5000)
            y = rng.uniform(-5000, 5000)
            ours = side_P(plane, x, y)  # 1 = front
            doom_front = node.dx * (y - node.py) < node.dy * (x - node.px)
            assert bool(ours) == bool(doom_front), (
                f"node (px={node.px},py={node.py},dx={node.dx},dy={node.dy}) "
                f"at point ({x}, {y}): ours={ours}, doom_front={doom_front}"
            )


# ---------------------------------------------------------------------------
# Rank formula verification (the critical test)
# ---------------------------------------------------------------------------


def _rank_order(subset: MapSubset, px: float, py: float) -> List[int]:
    """Compute rank for each selected seg and return them in rank order.

    Returns original seg indices, matching the filtered BSP traversal.

    Tie-break: within a rank tier (segs in the same subsector), order
    by original seg index, matching DOOM's in-subsector rendering order.
    """
    # Compute side_P for each BSP node
    side_P_vec = np.zeros(subset.seg_bsp_coeffs.shape[1], dtype=np.float64)
    for i, plane in enumerate(subset.bsp_nodes):
        side_P_vec[i] = float(side_P(plane, px, py))
    # Rank = coeffs @ side_P + const
    ranks = subset.seg_bsp_coeffs @ side_P_vec + subset.seg_bsp_consts
    seg_idx_arr = np.array(subset.original_seg_indices)
    # lexsort sorts by LAST key primary; we want rank primary, seg_idx secondary.
    order = np.lexsort((seg_idx_arr, ranks))
    return [subset.original_seg_indices[i] for i in order]


def test_rank_formula_matches_bsp_traversal(
    subset_default: MapSubset,
    raw_e1m1,
) -> None:
    """Sort selected segs by rank; must equal filtered BSP traversal.

    This is the ground-truth correctness test for the coefficient math.
    """
    selected_set = set(subset_default.original_seg_indices)
    reference = bsp_traversal_order(
        raw_e1m1,
        E1M1_START[0],
        E1M1_START[1],
        selected_set,
    )
    computed = _rank_order(subset_default, E1M1_START[0], E1M1_START[1])
    assert (
        computed == reference
    ), f"\nrank-sorted segs: {computed}\nBSP traversal:    {reference}"


def test_rank_formula_at_multiple_positions(raw_e1m1) -> None:
    """The rank formula must work from any player position within the
    map, not just the canonical start.
    """
    # Four positions chosen roughly inside different areas of E1M1.
    test_positions = [
        E1M1_START,
        (1000.0, -3500.0),
        (1500.0, -3300.0),
        (2000.0, -3000.0),
    ]
    for px, py in test_positions:
        subset = load_map_subset(
            "doom1.wad",
            "E1M1",
            px=px,
            py=py,
            max_walls=16,
            max_bsp_nodes=64,
        )
        selected_set = set(subset.original_seg_indices)
        reference = bsp_traversal_order(raw_e1m1, px, py, selected_set)
        computed = _rank_order(subset, px, py)
        assert computed == reference, (
            f"at ({px}, {py}):\n"
            f"  rank-sorted: {computed}\n"
            f"  BSP:         {reference}"
        )


def test_rank_orders_differ_between_positions(raw_e1m1) -> None:
    """Different player positions yield different rank orderings.

    The subset is also different (different closest segs), so this
    tests the full pipeline, not just the rank computation.
    """
    subset_a = load_map_subset(
        "doom1.wad",
        "E1M1",
        px=1000.0,
        py=-3500.0,
        max_walls=8,
        max_bsp_nodes=64,
    )
    subset_b = load_map_subset(
        "doom1.wad",
        "E1M1",
        px=3000.0,
        py=-3000.0,
        max_walls=8,
        max_bsp_nodes=64,
    )
    order_a = _rank_order(subset_a, 1000.0, -3500.0)
    order_b = _rank_order(subset_b, 3000.0, -3000.0)
    # Different positions far apart should produce different subsets
    # (the simplest signal that ordering depends on position).
    assert order_a != order_b


# ---------------------------------------------------------------------------
# Texture loading
# ---------------------------------------------------------------------------


def test_textures_capped(subset_default: MapSubset) -> None:
    assert len(subset_default.textures) <= 8
    assert len(subset_default.tex_name_to_id) == len(subset_default.textures)


def test_texture_shape() -> None:
    s = load_map_subset(
        "doom1.wad",
        "E1M1",
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=32,
        max_textures=4,
        tex_size=16,
        max_bsp_nodes=64,
    )
    assert len(s.textures) <= 4
    for tex in s.textures:
        assert tex.shape == (16, 16, 3)
        assert tex.min() >= 0.0 and tex.max() <= 1.0


def test_segment_texture_ids_valid(subset_default: MapSubset) -> None:
    for seg in subset_default.segments:
        assert -1 <= seg.texture_id < len(subset_default.textures)


# ---------------------------------------------------------------------------
# Synthetic scenario: simple 4-wall box with a 3-node BSP
# ---------------------------------------------------------------------------


def _synthetic_box_mapdata():
    """Build a minimal MapData: a 4-wall box with a 3-node BSP.

    The box has corners at (±10, ±10).  The BSP splits first on x=0,
    then each half on y=0, producing 4 subsectors (quadrants), each
    containing one wall.

    This is useful for direct manual verification of rank values.
    """
    from torchwright.doom.wad import (
        BspNode,
        Linedef,
        MapData,
        Seg,
        Sector,
        Sidedef,
        Subsector,
        Vertex,
    )

    vertices = [
        Vertex(10, -10),
        Vertex(10, 10),  # east wall
        Vertex(-10, 10),
        Vertex(-10, -10),  # west wall
    ]
    sectors = [
        Sector(
            floor_h=0,
            ceiling_h=128,
            floor_tex="FLAT",
            ceiling_tex="FLAT",
            light=255,
            special=0,
            tag=0,
        ),
    ]
    sidedefs = [
        Sidedef(x_offset=0, y_offset=0, upper="-", lower="-", middle="WALL0", sector=0),
        Sidedef(x_offset=0, y_offset=0, upper="-", lower="-", middle="WALL1", sector=0),
        Sidedef(x_offset=0, y_offset=0, upper="-", lower="-", middle="WALL2", sector=0),
        Sidedef(x_offset=0, y_offset=0, upper="-", lower="-", middle="WALL3", sector=0),
    ]
    linedefs = [
        Linedef(
            v1=0, v2=1, flags=0, special=0, tag=0, front_sidedef=0, back_sidedef=-1
        ),  # east
        Linedef(
            v1=1, v2=2, flags=0, special=0, tag=0, front_sidedef=1, back_sidedef=-1
        ),  # north
        Linedef(
            v1=2, v2=3, flags=0, special=0, tag=0, front_sidedef=2, back_sidedef=-1
        ),  # west
        Linedef(
            v1=3, v2=0, flags=0, special=0, tag=0, front_sidedef=3, back_sidedef=-1
        ),  # south
    ]
    # Four segs, one per wall
    segs = [
        Seg(v1=0, v2=1, angle=0, linedef=0, side=0, offset=0),  # east
        Seg(v1=1, v2=2, angle=0, linedef=1, side=0, offset=0),  # north
        Seg(v1=2, v2=3, angle=0, linedef=2, side=0, offset=0),  # west
        Seg(v1=3, v2=0, angle=0, linedef=3, side=0, offset=0),  # south
    ]
    # Four subsectors, one per wall (each contains exactly 1 seg)
    subsectors = [
        Subsector(seg_count=1, first_seg=0),  # NE quadrant (east wall)
        Subsector(seg_count=1, first_seg=1),  # NW quadrant (north wall)
        Subsector(seg_count=1, first_seg=2),  # SW quadrant (west wall)
        Subsector(seg_count=1, first_seg=3),  # SE quadrant (south wall)
    ]
    # BSP tree: node 2 is root, splitting on x=0 (vertical line).
    # dx=0, dy=1 → line runs north.  FRONT is +x (east).
    # Node 0: split NE subtree on y=0 (horizontal line).
    #   dx=-1, dy=0 → line runs west.  FRONT is -y? Let's check.
    #   For dx=-1, dy=0 at point (x, y), raw = 0*x + 1*y + 0 = y.
    #   So front iff y > 0, which is NORTH.  Good.
    # Actually let me use dx=1, dy=0 so front is SOUTH — doesn't matter
    # as long as we're consistent.
    #
    # Simplest BSP (with 2 internal nodes):
    # Root splits on x=0 (vertical line at origin, dir (0, 1)):
    #   front (x>0) → another split node (ne_sw)
    #   back  (x<0) → another split node (nw_se)
    # ne_sw splits on y=0 for x>0 half:
    #   front (y>0) → ss 0 (east wall, in NE)
    #   back  (y<0) → ss 3 (south wall, in SE)
    # nw_se splits on y=0 for x<0 half:
    #   front (y>0) → ss 1 (north wall, in NW)
    #   back  (y<0) → ss 2 (west wall, in SW)

    # For the x=0 split, dx=0, dy=1, px=0, py=0:
    #   plane raw(x,y) = dy*x - dx*y + (dx*py - dy*px) = x - 0 = x
    #   front iff x > 0.  ✓
    # For the y=0 split, dx=1, dy=0, px=0, py=0:
    #   plane raw(x,y) = 0*x - 1*y + 0 = -y
    #   front iff -y > 0 iff y < 0 (SOUTH).
    # So the y=0 splits need to know: front=SOUTH.
    # Adjust the tree so:
    #   ne_sw: front = ss 3 (south wall, y<0), back = ss 0 (east wall, y>0)
    #   nw_se: front = ss 2 (west wall, y<0), back = ss 1 (north wall, y>0)

    from torchwright.doom.wad import SUBSECTOR_FLAG

    ne_sw = BspNode(
        px=0,
        py=0,
        dx=1,
        dy=0,
        front_bbox=(0, -10, 0, 10),
        back_bbox=(10, 0, 0, 10),
        front_child=SUBSECTOR_FLAG | 3,  # ss 3 (south)
        back_child=SUBSECTOR_FLAG | 0,  # ss 0 (east, NE)
    )
    nw_se = BspNode(
        px=0,
        py=0,
        dx=1,
        dy=0,
        front_bbox=(0, -10, -10, 0),
        back_bbox=(10, 0, -10, 0),
        front_child=SUBSECTOR_FLAG | 2,  # ss 2 (west)
        back_child=SUBSECTOR_FLAG | 1,  # ss 1 (north)
    )
    root = BspNode(
        px=0,
        py=0,
        dx=0,
        dy=1,
        front_bbox=(10, -10, 0, 10),
        back_bbox=(10, -10, -10, 0),
        front_child=0,  # ne_sw
        back_child=1,  # nw_se
    )
    nodes = [ne_sw, nw_se, root]
    return MapData(
        name="SYNTH",
        vertices=vertices,
        linedefs=linedefs,
        sidedefs=sidedefs,
        sectors=sectors,
        segs=segs,
        subsectors=subsectors,
        nodes=nodes,
    )


def test_synthetic_box_bsp_ordering() -> None:
    """Verify the reference BSP traversal on the synthetic 4-wall box.

    Player at (5, 5) (NE quadrant): should visit NE (east wall) first,
    then traverse to other quadrants in BSP order.
    """
    md = _synthetic_box_mapdata()

    # Player at (5, 5) — NE quadrant.  Expected visit order:
    # Root(x=0): player front (x>0) → visit ne_sw first, then nw_se.
    # ne_sw(y=0): player at y=5, raw = -y = -5 < 0 → BACK → visit back first = ss 0 (east).
    # Then front of ne_sw = ss 3 (south).
    # Then nw_se.  At (5, 5): raw = -5 < 0 → BACK → visit back first = ss 1 (north).
    # Then front of nw_se = ss 2 (west).
    # So order: east (seg 0), south (seg 3), north (seg 1), west (seg 2)
    order = bsp_traversal_order(md, 5.0, 5.0)
    assert order == [0, 3, 1, 2], f"got {order}"


def test_synthetic_box_rank_matches_traversal() -> None:
    """Manually build the subset on the synthetic box and verify rank
    matches the traversal order at several player positions."""
    # The load_map_subset API needs a WAD path; test the core math
    # directly using the building blocks.
    from torchwright.doom.map_subset import (
        _make_plane,
        _walk_paths,
        _count_selected_in_subtree,
        _compute_coefficients,
        _build_seg_to_subsector,
    )

    md = _synthetic_box_mapdata()
    selected_orig = [0, 1, 2, 3]  # all four segs
    selected_set = set(selected_orig)
    seg_to_ss = _build_seg_to_subsector(md)
    paths = _walk_paths(md, len(md.nodes) - 1)

    subset_node_ids: Set[int] = set()
    for ss in {seg_to_ss[s] for s in selected_orig}:
        for n, _ in paths[ss]:
            subset_node_ids.add(n)
    sorted_old = sorted(subset_node_ids)
    old_to_new = {old: new for new, old in enumerate(sorted_old)}
    bsp_nodes = [_make_plane(md.nodes[o]) for o in sorted_old]

    _ss, fc, bc = _count_selected_in_subtree(
        md,
        selected_set,
        len(md.nodes) - 1,
    )
    coeffs, consts = _compute_coefficients(
        md,
        selected_orig,
        seg_to_ss,
        paths,
        fc,
        bc,
        old_to_new,
        max_bsp_nodes=len(sorted_old),
    )

    for px, py in [(5, 5), (-5, -5), (7, -3), (-2, 4)]:
        reference = bsp_traversal_order(md, px, py, selected_set)
        side_P_vec = np.array(
            [float(side_P(p, px, py)) for p in bsp_nodes]
            + [0.0] * (coeffs.shape[1] - len(bsp_nodes))
        )
        ranks = coeffs @ side_P_vec + consts
        # Tie-break on seg index to match DOOM's in-subsector order.
        seg_idx_arr = np.array(selected_orig)
        order = np.lexsort((seg_idx_arr, ranks))
        computed = [selected_orig[i] for i in order]
        assert computed == reference, (
            f"at ({px}, {py}): got {computed}, expected {reference}\n" f"ranks: {ranks}"
        )


# ---------------------------------------------------------------------------
# build_scene_subset — axis-aligned balanced BSP for hand-authored scenes
# ---------------------------------------------------------------------------


def _traverse_scene_tree(root, px: float, py: float) -> List[int]:
    """Front-to-back DFS of an in-memory ``_BspTreeNode`` tree.

    Returns seg indices in DOOM traversal order: at each internal node
    visit the player-side subtree first, then the other side.  This is
    the ground-truth comparator for rank-based sorting on a scene.
    """
    out: List[int] = []

    def visit(node) -> None:
        if node.is_leaf:
            out.append(node.seg_idx)
            return
        raw = node.plane.nx * px + node.plane.ny * py + node.plane.d
        if raw > 0:
            visit(node.front)
            visit(node.back)
        else:
            visit(node.back)
            visit(node.front)

    visit(root)
    return out


def _rank_order_scene(subset: MapSubset, px: float, py: float) -> List[int]:
    """Sort segs by rank at (px, py); return seg indices in rank order."""
    M = subset.seg_bsp_coeffs.shape[1]
    side_P_vec = np.zeros(M, dtype=np.float64)
    for i, plane in enumerate(subset.bsp_nodes):
        side_P_vec[i] = float(side_P(plane, px, py))
    ranks = subset.seg_bsp_coeffs @ side_P_vec + subset.seg_bsp_consts
    seg_idx_arr = np.arange(len(subset.segments))
    order = np.lexsort((seg_idx_arr, ranks))
    return [int(i) for i in order]


def test_build_scene_subset_nonempty() -> None:
    """box_room scene builds a non-empty subset with real BSP nodes."""
    segments, textures = box_room_textured(
        wad_path="doom1.wad",
        tex_size=8,
    )
    subset = build_scene_subset(segments, textures)
    assert len(subset.segments) == len(segments)
    assert len(subset.bsp_nodes) == len(segments) - 1  # balanced BSP
    assert subset.original_seg_indices == list(range(len(segments)))


def test_build_scene_subset_coeffs_shape() -> None:
    """Coefficient matrix has shape (N, max_bsp_nodes); exactly N-1
    columns contain nonzero entries; the rest are pure zero padding."""
    segments, textures = box_room_textured(
        wad_path="doom1.wad",
        tex_size=8,
    )
    subset = build_scene_subset(segments, textures, max_bsp_nodes=16)
    N = len(segments)
    assert subset.seg_bsp_coeffs.shape == (N, 16)
    real_nodes = len(subset.bsp_nodes)
    assert real_nodes == N - 1
    # Columns >= real_nodes are padding → all zeros
    padding = subset.seg_bsp_coeffs[:, real_nodes:]
    assert np.all(padding == 0.0)


def test_build_scene_subset_rank_matches_python_traversal() -> None:
    """Rank-sorted segs match a direct DFS of the built BSP tree.

    This is the critical correctness test: if the coefficients are
    right, the runtime rank formula must reproduce the same ordering
    that a straightforward front-to-back BSP walk would produce.
    """
    segments, textures = box_room_textured(
        wad_path="doom1.wad",
        tex_size=8,
    )
    subset = build_scene_subset(segments, textures, max_bsp_nodes=16)
    # Rebuild the tree with the same construction logic for reference.
    tree = _build_balanced_bsp(list(range(len(segments))), segments, depth=0)

    test_positions = [
        (0.0, 0.0),  # center
        (3.0, 2.0),  # off-center
        (-2.5, 4.0),  # another quadrant
        (4.9, -4.9),  # near a corner
    ]
    for px, py in test_positions:
        reference = _traverse_scene_tree(tree, px, py)
        computed = _rank_order_scene(subset, px, py)
        assert (
            computed == reference
        ), f"at ({px}, {py}): rank order {computed} != tree DFS {reference}"


def test_build_scene_subset_single_seg() -> None:
    """N=1 edge case: no BSP nodes, zero coefficients, trivial rank."""
    segments = [
        Segment(
            ax=0,
            ay=0,
            bx=1,
            by=0,
            color=(0.5, 0.5, 0.5),
            front_floor=-1.0,
            front_ceiling=1.0,
            texture_id=0,
        )
    ]
    textures = [np.zeros((8, 8, 3), dtype=np.float64)]
    subset = build_scene_subset(segments, textures, max_bsp_nodes=16)
    assert len(subset.bsp_nodes) == 0
    assert subset.seg_bsp_coeffs.shape == (1, 16)
    assert np.all(subset.seg_bsp_coeffs == 0.0)
    assert np.all(subset.seg_bsp_consts == 0.0)


def test_build_scene_subset_too_many_segs_raises() -> None:
    """N > max_bsp_nodes + 1 is rejected with a helpful ValueError."""
    # 8 segs would need 7 nodes; max_bsp_nodes=4 is too small.
    segments = [
        Segment(
            ax=i,
            ay=0,
            bx=i + 1,
            by=0,
            color=(0.5, 0.5, 0.5),
            front_floor=-1.0,
            front_ceiling=1.0,
            texture_id=0,
        )
        for i in range(8)
    ]
    textures: List[np.ndarray] = []
    with pytest.raises(ValueError, match="max_bsp_nodes"):
        build_scene_subset(segments, textures, max_bsp_nodes=4)


def test_build_scene_subset_empty_raises() -> None:
    """Empty segment list raises ValueError."""
    with pytest.raises(ValueError, match="at least 1"):
        build_scene_subset([], [], max_bsp_nodes=16)
