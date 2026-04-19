"""Tests for WAD map geometry + BSP parsing in ``torchwright.doom.wad``.

Ground truth values for E1M1 (first map of ``doom1.wad``):

- 467 VERTEXES
- 475 LINEDEFS
- 648 SIDEDEFS
- 85  SECTORS
- 732 SEGS
- 237 SSECTORS
- 236 NODES

Tests use the shipped ``doom1.wad`` in the repo root.
"""

from dataclasses import replace
from typing import Set

import numpy as np
import pytest

from torchwright.doom.wad import (
    SUBSECTOR_FLAG,
    BspNode,
    Linedef,
    MapData,
    Seg,
    Sector,
    Sidedef,
    Subsector,
    Vertex,
    WADReader,
    _assign_tex_id,
    _pick_seg_texture,
    sector_color,
    seg_list_to_segments,
)
from torchwright.reference_renderer.render import render_frame
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wad() -> WADReader:
    return WADReader("doom1.wad")


@pytest.fixture(scope="module")
def e1m1(wad: WADReader) -> MapData:
    return wad.get_map("E1M1")


# ---------------------------------------------------------------------------
# 1. Lump discovery
# ---------------------------------------------------------------------------


def test_find_map_lumps_e1m1(wad: WADReader) -> None:
    """All seven geometry lumps are found for E1M1."""
    lumps = wad._find_map_lumps("E1M1")
    expected = {
        "VERTEXES",
        "LINEDEFS",
        "SIDEDEFS",
        "SECTORS",
        "SEGS",
        "SSECTORS",
        "NODES",
    }
    assert expected.issubset(lumps.keys()), f"Missing lumps: {expected - lumps.keys()}"
    # All lump sizes are positive
    for name in expected:
        _, size = lumps[name]
        assert size > 0, f"{name} has zero size"


def test_find_map_missing_raises(wad: WADReader) -> None:
    """Looking up a non-existent map raises ``KeyError``."""
    with pytest.raises(KeyError):
        wad.get_map("NOPE")


# ---------------------------------------------------------------------------
# 2. Per-lump counts (ground truth for E1M1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lump,expected",
    [
        ("vertices", 467),
        ("linedefs", 475),
        ("sidedefs", 648),
        ("sectors", 85),
        ("segs", 732),
        ("subsectors", 237),
        ("nodes", 236),
    ],
)
def test_e1m1_lump_counts(e1m1: MapData, lump: str, expected: int) -> None:
    assert len(getattr(e1m1, lump)) == expected


# ---------------------------------------------------------------------------
# 3. Per-lump data integrity
# ---------------------------------------------------------------------------


def test_vertex_bounds_sane(e1m1: MapData) -> None:
    """Vertex coordinates fit in int16 and span a reasonable map extent."""
    xs = [v.x for v in e1m1.vertices]
    ys = [v.y for v in e1m1.vertices]
    assert all(-32768 <= x <= 32767 for x in xs)
    assert all(-32768 <= y <= 32767 for y in ys)
    # E1M1 is at least 100 units across in each axis
    assert max(xs) - min(xs) > 100
    assert max(ys) - min(ys) > 100


def test_linedef_vertex_refs_valid(e1m1: MapData) -> None:
    """Every linedef's v1/v2 indices are valid."""
    n = len(e1m1.vertices)
    for i, ld in enumerate(e1m1.linedefs):
        assert 0 <= ld.v1 < n, f"linedef {i} has invalid v1 {ld.v1}"
        assert 0 <= ld.v2 < n, f"linedef {i} has invalid v2 {ld.v2}"


def test_linedef_sidedef_refs(e1m1: MapData) -> None:
    """Front sidedefs are always valid; back sidedefs are -1 or valid.

    At least one linedef must be two-sided (has a valid back sidedef)
    — E1M1 has many interior walls between sectors.
    """
    n_sd = len(e1m1.sidedefs)
    two_sided_count = 0
    for i, ld in enumerate(e1m1.linedefs):
        assert (
            0 <= ld.front_sidedef < n_sd
        ), f"linedef {i} has bad front_sidedef {ld.front_sidedef}"
        assert (
            ld.back_sidedef == -1 or 0 <= ld.back_sidedef < n_sd
        ), f"linedef {i} has bad back_sidedef {ld.back_sidedef}"
        if ld.back_sidedef != -1:
            two_sided_count += 1
    assert two_sided_count > 0, "expected at least one two-sided linedef"


def test_sidedef_sector_refs_valid(e1m1: MapData) -> None:
    """Every sidedef points to a valid sector."""
    n = len(e1m1.sectors)
    for i, sd in enumerate(e1m1.sidedefs):
        assert 0 <= sd.sector < n, f"sidedef {i} has bad sector {sd.sector}"


def test_sidedef_texture_dash_preserved(e1m1: MapData) -> None:
    """At least one sidedef has ``"-"`` for a texture slot.

    This confirms that the parser preserves the ``"-"`` sentinel
    literally rather than stripping it to the empty string.
    """
    has_dash = any(
        sd.upper == "-" or sd.middle == "-" or sd.lower == "-" for sd in e1m1.sidedefs
    )
    assert has_dash, "expected at least one sidedef with '-' texture"


def test_sector_floor_ceiling_order(e1m1: MapData) -> None:
    """Every sector's ceiling is at or above its floor."""
    for i, s in enumerate(e1m1.sectors):
        assert (
            s.ceiling_h >= s.floor_h
        ), f"sector {i}: ceiling {s.ceiling_h} < floor {s.floor_h}"


def test_seg_linedef_refs_valid(e1m1: MapData) -> None:
    """Every seg references a valid linedef."""
    n = len(e1m1.linedefs)
    for i, seg in enumerate(e1m1.segs):
        assert 0 <= seg.linedef < n, f"seg {i} has invalid linedef {seg.linedef}"


def test_seg_vertex_refs_valid(e1m1: MapData) -> None:
    """Every seg's v1/v2 indices are valid vertices."""
    n = len(e1m1.vertices)
    for i, seg in enumerate(e1m1.segs):
        assert 0 <= seg.v1 < n, f"seg {i} has invalid v1 {seg.v1}"
        assert 0 <= seg.v2 < n, f"seg {i} has invalid v2 {seg.v2}"


def test_subsector_seg_range_valid(e1m1: MapData) -> None:
    """Each subsector's seg range is a valid slice of the SEGS list."""
    n = len(e1m1.segs)
    for i, ss in enumerate(e1m1.subsectors):
        assert ss.first_seg >= 0
        assert ss.seg_count > 0, f"subsector {i} has zero segs"
        assert ss.first_seg + ss.seg_count <= n, (
            f"subsector {i} seg range [{ss.first_seg}, "
            f"{ss.first_seg + ss.seg_count}) overflows SEGS (n={n})"
        )


# ---------------------------------------------------------------------------
# 4. BSP tree integrity
# ---------------------------------------------------------------------------


def test_bsp_node_children_valid(e1m1: MapData) -> None:
    """Each BSP node child is either a valid subsector or a valid node."""
    n_nodes = len(e1m1.nodes)
    n_ss = len(e1m1.subsectors)
    for i, node in enumerate(e1m1.nodes):
        for side, child in [("front", node.front_child), ("back", node.back_child)]:
            if child & SUBSECTOR_FLAG:
                idx = child & ~SUBSECTOR_FLAG
                assert 0 <= idx < n_ss, f"node {i} {side} subsector {idx} out of range"
            else:
                assert (
                    0 <= child < n_nodes
                ), f"node {i} {side} node {child} out of range"


def test_bsp_tree_reaches_all_subsectors(e1m1: MapData) -> None:
    """Traversing from the root visits every subsector exactly once.

    DOOM convention: the root is the last BSP node in the NODES lump.
    """
    visited_ss: Set[int] = set()
    visited_nodes: Set[int] = set()

    def walk(child_ref: int) -> None:
        if child_ref & SUBSECTOR_FLAG:
            idx = child_ref & ~SUBSECTOR_FLAG
            assert idx not in visited_ss, f"subsector {idx} reached twice"
            visited_ss.add(idx)
            return
        assert child_ref not in visited_nodes, f"node {child_ref} reached twice"
        visited_nodes.add(child_ref)
        node = e1m1.nodes[child_ref]
        walk(node.front_child)
        walk(node.back_child)

    root = len(e1m1.nodes) - 1
    walk(root)
    assert visited_ss == set(range(len(e1m1.subsectors))), (
        f"unvisited subsectors: " f"{set(range(len(e1m1.subsectors))) - visited_ss}"
    )
    assert visited_nodes == set(range(len(e1m1.nodes)))


def test_bsp_splitting_line_nonzero(e1m1: MapData) -> None:
    """Every BSP node has a nonzero splitting direction vector."""
    for i, node in enumerate(e1m1.nodes):
        assert node.dx != 0 or node.dy != 0, f"node {i} has zero splitting direction"


# ---------------------------------------------------------------------------
# 5. Segment conversion
# ---------------------------------------------------------------------------


def test_get_map_segments_nonempty(wad: WADReader) -> None:
    """E1M1 produces a non-trivial number of renderable segments."""
    segments, _, _ = wad.get_map_segments("E1M1", tex_size=8)
    assert len(segments) > 500


def test_get_map_segments_count_close_to_segs(wad: WADReader, e1m1: MapData) -> None:
    """Nearly all SEGS convert to Segments (>= 95%).

    Some may be skipped due to invalid sidedef refs, but in a
    well-formed WAD this is rare.
    """
    segments, _, _ = wad.get_map_segments("E1M1", tex_size=8)
    ratio = len(segments) / len(e1m1.segs)
    assert ratio >= 0.95, (
        f"only {len(segments)}/{len(e1m1.segs)} segs converted " f"({ratio:.1%})"
    )


def test_segment_coords_match_vertices(wad: WADReader, e1m1: MapData) -> None:
    """A known seg's output coordinates match its vertices exactly."""
    segments, _, _ = wad.get_map_segments("E1M1", tex_size=8)
    # Pick the first seg with a valid front sidedef
    for seg in e1m1.segs:
        if seg.side != 0:
            continue
        ld = e1m1.linedefs[seg.linedef]
        if ld.front_sidedef < 0:
            continue
        v1 = e1m1.vertices[seg.v1]
        v2 = e1m1.vertices[seg.v2]
        break
    else:
        pytest.skip("no seg with valid front sidedef")

    # Find the corresponding Segment by matching endpoints
    match = [
        s
        for s in segments
        if s.ax == float(v1.x)
        and s.ay == float(v1.y)
        and s.bx == float(v2.x)
        and s.by == float(v2.y)
    ]
    assert len(match) >= 1, "converted segment has different coordinates"


def test_segment_coords_are_doom_native(wad: WADReader, e1m1: MapData) -> None:
    """Returned coordinates are in DOOM's native int16 range.

    Confirms we return raw world units, not a rescaled version — callers
    can apply their own scaling if they want smaller numeric ranges.
    """
    segments, _, _ = wad.get_map_segments("E1M1", tex_size=8)
    # E1M1 spans roughly ±4000 in each axis, not ±1 or ±100.
    max_abs = max(max(abs(s.ax), abs(s.ay), abs(s.bx), abs(s.by)) for s in segments)
    assert max_abs > 100, f"max |coord| is {max_abs} — expected DOOM-native (>100)"


def test_segment_back_side_reversed_vs_front(e1m1: MapData) -> None:
    """For a two-sided linedef, front and back SEGS have swapped vertex order.

    DOOM emits one seg per linedef side; the back side's v1/v2 are
    reversed relative to the front side.
    """
    # Find a two-sided linedef that has both front and back segs.
    seg_pairs: dict[int, dict[int, Seg]] = {}  # linedef -> {side: seg}
    for seg in e1m1.segs:
        seg_pairs.setdefault(seg.linedef, {})[seg.side] = seg
    for ld_idx, sides in seg_pairs.items():
        if 0 in sides and 1 in sides:
            front, back = sides[0], sides[1]
            # The endpoints should be swapped (v1/v2 reversed)
            v1f = e1m1.vertices[front.v1]
            v2f = e1m1.vertices[front.v2]
            v1b = e1m1.vertices[back.v1]
            v2b = e1m1.vertices[back.v2]
            assert v1f == v2b and v2f == v1b, (
                f"linedef {ld_idx} front/back endpoints not reversed: "
                f"front=({v1f}, {v2f}), back=({v1b}, {v2b})"
            )
            return
    pytest.fail("no two-sided linedef with both front and back segs found")


def test_texture_id_mapping_unique(wad: WADReader) -> None:
    """``name_to_id`` assigns distinct ids to distinct names, never ``"-"``."""
    _, _, name_to_id = wad.get_map_segments("E1M1", tex_size=8)
    assert "-" not in name_to_id
    ids = list(name_to_id.values())
    assert len(ids) == len(set(ids)), "duplicate texture ids"
    # Ids are 0..len-1
    assert set(ids) == set(range(len(ids)))


def test_texture_atlas_shape(wad: WADReader) -> None:
    """Every atlas entry has ``(tex_size, tex_size, 3)`` and values in [0,1]."""
    tex_size = 8
    _, textures, _ = wad.get_map_segments("E1M1", tex_size=tex_size)
    assert len(textures) > 0
    for i, t in enumerate(textures):
        assert t.shape == (tex_size, tex_size, 3), f"texture {i} has shape {t.shape}"
        assert (
            t.min() >= 0.0 and t.max() <= 1.0
        ), f"texture {i} has out-of-range values ({t.min()}, {t.max()})"


def test_texture_atlas_count_matches_ids(wad: WADReader) -> None:
    """``len(textures) == len(name_to_id)``."""
    _, textures, name_to_id = wad.get_map_segments("E1M1", tex_size=8)
    assert len(textures) == len(name_to_id)


def test_segment_texture_ids_in_range(wad: WADReader) -> None:
    """Every segment has ``-1 <= texture_id < len(textures)``."""
    segments, textures, _ = wad.get_map_segments("E1M1", tex_size=8)
    for i, seg in enumerate(segments):
        assert -1 <= seg.texture_id < len(textures), (
            f"segment {i} has out-of-range texture_id {seg.texture_id} "
            f"(atlas size {len(textures)})"
        )


# ---------------------------------------------------------------------------
# 6. Helper unit tests (synthetic MapData)
# ---------------------------------------------------------------------------


def test_assign_tex_id_dash_returns_minus_one() -> None:
    name_to_id: dict = {}
    assert _assign_tex_id("-", name_to_id) == -1
    assert _assign_tex_id("", name_to_id) == -1
    assert name_to_id == {}


def test_assign_tex_id_assigns_incrementally() -> None:
    name_to_id: dict = {}
    assert _assign_tex_id("BRICK", name_to_id) == 0
    assert _assign_tex_id("STONE", name_to_id) == 1
    # Re-use returns the same id
    assert _assign_tex_id("BRICK", name_to_id) == 0
    assert name_to_id == {"BRICK": 0, "STONE": 1}


@pytest.mark.parametrize(
    "upper,lower,middle,expected",
    [
        ("UP", "LO", "MID", "MID"),  # prefers middle
        ("UP", "LO", "-", "LO"),  # falls back to lower
        ("UP", "-", "-", "UP"),  # falls back to upper
        ("-", "-", "-", "-"),  # none → dash
    ],
    ids=[
        "prefers_middle",
        "falls_back_to_lower",
        "falls_back_to_upper",
        "none_returns_dash",
    ],
)
def test_pick_seg_texture(upper: str, lower: str, middle: str, expected: str) -> None:
    sd = Sidedef(
        x_offset=0,
        y_offset=0,
        upper=upper,
        lower=lower,
        middle=middle,
        sector=0,
    )
    assert _pick_seg_texture(sd) == expected


def test_sector_color_deterministic() -> None:
    """Same sector index yields same color; different indices yield
    different colors (with high probability)."""
    assert sector_color(0) == sector_color(0)
    assert sector_color(1) == sector_color(1)
    assert sector_color(5) != sector_color(17)
    # All channels in [0, 1]
    r, g, b = sector_color(42)
    assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1


def _synthetic_map_data() -> MapData:
    """Build a minimal well-formed MapData with 4 walls in a square."""
    return MapData(
        name="TEST",
        vertices=[Vertex(0, 0), Vertex(10, 0), Vertex(10, 10), Vertex(0, 10)],
        linedefs=[
            Linedef(
                v1=0, v2=1, flags=0, special=0, tag=0, front_sidedef=0, back_sidedef=-1
            ),
            Linedef(
                v1=1, v2=2, flags=0, special=0, tag=0, front_sidedef=1, back_sidedef=-1
            ),
            Linedef(
                v1=2, v2=3, flags=0, special=0, tag=0, front_sidedef=2, back_sidedef=-1
            ),
            Linedef(
                v1=3, v2=0, flags=0, special=0, tag=0, front_sidedef=3, back_sidedef=-1
            ),
        ],
        sidedefs=[
            Sidedef(
                x_offset=0, y_offset=0, upper="-", lower="-", middle="BRICK", sector=0
            ),
            Sidedef(
                x_offset=0, y_offset=0, upper="-", lower="-", middle="STONE", sector=0
            ),
            Sidedef(
                x_offset=0, y_offset=0, upper="-", lower="-", middle="BRICK", sector=0
            ),
            Sidedef(x_offset=0, y_offset=0, upper="-", lower="-", middle="-", sector=0),
        ],
        sectors=[
            Sector(
                floor_h=0,
                ceiling_h=128,
                floor_tex="FLOOR",
                ceiling_tex="CEIL",
                light=255,
                special=0,
                tag=0,
            )
        ],
        segs=[
            Seg(v1=0, v2=1, angle=0, linedef=0, side=0, offset=0),
            Seg(v1=1, v2=2, angle=0, linedef=1, side=0, offset=0),
            Seg(v1=2, v2=3, angle=0, linedef=2, side=0, offset=0),
            Seg(v1=3, v2=0, angle=0, linedef=3, side=0, offset=0),
        ],
        subsectors=[Subsector(seg_count=4, first_seg=0)],
        nodes=[],
    )


def test_seg_list_to_segments_basic() -> None:
    """Basic synthetic map produces expected segments."""
    md = _synthetic_map_data()
    segments, name_to_id = seg_list_to_segments(md)
    assert len(segments) == 4
    # BRICK and STONE get ids (middle texture); the "-" wall has id -1.
    assert name_to_id == {"BRICK": 0, "STONE": 1}
    assert segments[0].texture_id == 0  # BRICK
    assert segments[1].texture_id == 1  # STONE
    assert segments[2].texture_id == 0  # BRICK (reused)
    assert segments[3].texture_id == -1  # no texture
    # All in sector 0 → same color
    assert segments[0].color == segments[1].color == sector_color(0)


def test_seg_list_to_segments_skips_invalid_sidedef() -> None:
    """A seg referencing an out-of-range sidedef is silently skipped."""
    md = _synthetic_map_data()
    # Break linedef 0's front sidedef
    broken_ld = replace(md.linedefs[0], front_sidedef=9999)
    md = replace(md, linedefs=[broken_ld] + md.linedefs[1:])
    segments, _ = seg_list_to_segments(md)
    assert len(segments) == 3  # seg 0 skipped


def test_seg_list_to_segments_skips_invalid_vertex() -> None:
    """A seg referencing an out-of-range vertex is silently skipped."""
    md = _synthetic_map_data()
    broken_seg = Seg(v1=9999, v2=1, angle=0, linedef=0, side=0, offset=0)
    md = replace(md, segs=[broken_seg] + md.segs[1:])
    segments, _ = seg_list_to_segments(md)
    assert len(segments) == 3


def test_seg_list_to_segments_skips_invalid_linedef() -> None:
    """A seg referencing an out-of-range linedef is silently skipped."""
    md = _synthetic_map_data()
    broken_seg = Seg(v1=0, v2=1, angle=0, linedef=9999, side=0, offset=0)
    md = replace(md, segs=[broken_seg] + md.segs[1:])
    segments, _ = seg_list_to_segments(md)
    assert len(segments) == 3


# ---------------------------------------------------------------------------
# 7. Render smoke tests
# ---------------------------------------------------------------------------


def _player_midpoint(md: MapData):
    xs = [v.x for v in md.vertices]
    ys = [v.y for v in md.vertices]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _rescale_segments(segments, factor: float):
    """Return a copy of ``segments`` with all coordinates multiplied.

    DOOM's native coords span ±4000 — the reference renderer's
    small-screen defaults (24-row) need smaller coords to avoid
    subpixel wall heights.  Tests that render through the renderer
    apply the scaling themselves.
    """
    from torchwright.reference_renderer.types import Segment

    return [
        Segment(
            ax=s.ax * factor,
            ay=s.ay * factor,
            bx=s.bx * factor,
            by=s.by * factor,
            color=s.color,
            texture_id=s.texture_id,
        )
        for s in segments
    ]


def test_render_smoke_e1m1(wad: WADReader, e1m1: MapData) -> None:
    """Render a frame from several candidate positions in E1M1.

    The loader returns DOOM-native coords (±4000).  For this smoke
    test we rescale locally to fit the reference renderer's small-
    screen defaults.  Asserts the frame shape/range is correct and
    at least one candidate produces visible wall pixels.
    """
    factor = 1.0 / 32.0
    raw_segments, textures, _ = wad.get_map_segments("E1M1", tex_size=8)
    segments = _rescale_segments(raw_segments, factor)
    config = RenderConfig(
        screen_width=32,
        screen_height=24,
        fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )
    # Sample player positions from actual vertex neighborhoods so we
    # are near walls.  Offset slightly to avoid landing exactly on a
    # vertex (which the renderer may treat as inside a wall).
    offsets = [(0.3, 0.3), (-0.3, 0.3), (0.3, -0.3), (-0.3, -0.3)]
    # Take every Nth vertex for broad coverage.
    sample = e1m1.vertices[:: max(1, len(e1m1.vertices) // 16)]

    saw_wall = False
    for v in sample:
        for dx, dy in offsets:
            px = v.x * factor + dx
            py = v.y * factor + dy
            for angle in (0, 64, 128, 192):
                frame = render_frame(
                    float(px),
                    float(py),
                    int(angle),
                    segments,
                    config,
                    textures=textures,
                )
                assert frame.shape == (24, 32, 3)
                assert frame.min() >= 0.0 - 1e-6
                assert frame.max() <= 1.0 + 1e-6
                ceil = np.array(config.ceiling_color)
                floor = np.array(config.floor_color)
                diff_ceil = np.abs(frame - ceil).sum(axis=-1)
                diff_floor = np.abs(frame - floor).sum(axis=-1)
                wall_mask = (diff_ceil > 1e-3) & (diff_floor > 1e-3)
                if wall_mask.any():
                    saw_wall = True
                    break
            if saw_wall:
                break
        if saw_wall:
            break
    assert saw_wall, (
        "rendering E1M1 from any candidate position/angle failed to " "show walls"
    )


def test_render_deterministic(wad: WADReader, e1m1: MapData) -> None:
    """Rendering the same inputs twice yields bit-identical output."""
    factor = 1.0 / 32.0
    raw_segments, textures, _ = wad.get_map_segments("E1M1", tex_size=8)
    segments = _rescale_segments(raw_segments, factor)
    config = RenderConfig(
        screen_width=16,
        screen_height=16,
        fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )
    # Pick the first vertex + a small offset as player start.
    px = e1m1.vertices[0].x * factor + 1.0
    py = e1m1.vertices[0].y * factor + 1.0

    frame_a = render_frame(px, py, 0, segments, config, textures=textures)
    frame_b = render_frame(px, py, 0, segments, config, textures=textures)
    assert np.array_equal(frame_a, frame_b)
