"""Integration tests for BSP-driven wall ordering in the compiled graph.

Step 1 (WAD parsing) and Step 2 (``load_map_subset``) are tested in
isolation.  This module exercises Step 3 — the BSP_NODE tokens in
``game_graph.py`` plus the prefill changes in ``compile.py`` — by:

1. Compiling the game graph with ``max_bsp_nodes=16``.
2. Feeding a :class:`MapSubset` through ``step_frame``.
3. Checking that the rendered frame is non-trivial (walls appear) and,
   where possible, matches the reference renderer.

A synthetic 4-wall box room with a hand-authored 3-node BSP is used as
the shared fixture, because it compiles quickly and the expected sort
order is easy to reason about.
"""

from typing import List

import numpy as np
import pytest
import torch

from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import (
    MapSubset,
    _compute_coefficients,
    _count_selected_in_subtree,
    _make_plane,
    _walk_paths,
    bsp_traversal_order,
    side_P,
)
from torchwright.doom.wad import (
    BspNode,
    Linedef,
    MapData,
    SUBSECTOR_FLAG,
    Seg,
    Sector,
    Sidedef,
    Subsector,
    Vertex,
)
from torchwright.reference_renderer.render import render_frame
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


# ---------------------------------------------------------------------------
# Shared synthetic map: 4 walls forming an axis-aligned 10x10 box
# with a tight 3-node BSP (root splits on x=0, each half splits on y=0).
# ---------------------------------------------------------------------------


def _synthetic_box_md() -> MapData:
    """A 4-wall box + 3-node BSP (identical to the one in test_map_subset)."""
    vertices = [
        Vertex(5, -5), Vertex(5, 5),    # east endpoints
        Vertex(-5, 5), Vertex(-5, -5),  # west endpoints
    ]
    sectors = [Sector(
        floor_h=0, ceiling_h=128,
        floor_tex="F", ceiling_tex="C",
        light=255, special=0, tag=0,
    )]
    sidedefs = [
        Sidedef(x_offset=0, y_offset=0, upper="-", lower="-",
                middle=name, sector=0)
        for name in ("BRICK", "STONE", "STRIPE", "CHECKER")
    ]
    linedefs = [
        Linedef(v1=0, v2=1, flags=0, special=0, tag=0,
                front_sidedef=0, back_sidedef=-1),  # east
        Linedef(v1=1, v2=2, flags=0, special=0, tag=0,
                front_sidedef=1, back_sidedef=-1),  # north
        Linedef(v1=2, v2=3, flags=0, special=0, tag=0,
                front_sidedef=2, back_sidedef=-1),  # west
        Linedef(v1=3, v2=0, flags=0, special=0, tag=0,
                front_sidedef=3, back_sidedef=-1),  # south
    ]
    segs = [
        Seg(v1=0, v2=1, angle=0, linedef=0, side=0, offset=0),  # east
        Seg(v1=1, v2=2, angle=0, linedef=1, side=0, offset=0),  # north
        Seg(v1=2, v2=3, angle=0, linedef=2, side=0, offset=0),  # west
        Seg(v1=3, v2=0, angle=0, linedef=3, side=0, offset=0),  # south
    ]
    subsectors = [
        Subsector(seg_count=1, first_seg=0),  # east (NE)
        Subsector(seg_count=1, first_seg=1),  # north (NW)
        Subsector(seg_count=1, first_seg=2),  # west (SW)
        Subsector(seg_count=1, first_seg=3),  # south (SE)
    ]
    # 3 BSP nodes:
    #   root (idx 2): x=0 split, front = NE/SE subtree, back = NW/SW
    #   node ne_sw (idx 0): y=0 split in x>0 half
    #   node nw_se (idx 1): y=0 split in x<0 half
    ne_sw = BspNode(
        px=0, py=0, dx=1, dy=0,
        front_bbox=(0, -5, 0, 5), back_bbox=(5, 0, 0, 5),
        front_child=SUBSECTOR_FLAG | 3,   # south (y<0, x>0)
        back_child=SUBSECTOR_FLAG | 0,    # east (y>0, x>0)
    )
    nw_se = BspNode(
        px=0, py=0, dx=1, dy=0,
        front_bbox=(0, -5, -5, 0), back_bbox=(5, 0, -5, 0),
        front_child=SUBSECTOR_FLAG | 2,   # west (y<0, x<0)
        back_child=SUBSECTOR_FLAG | 1,    # north (y>0, x<0)
    )
    root = BspNode(
        px=0, py=0, dx=0, dy=1,
        front_bbox=(5, -5, 0, 5), back_bbox=(5, -5, -5, 0),
        front_child=0, back_child=1,
    )
    return MapData(
        name="TEST",
        vertices=vertices, linedefs=linedefs, sidedefs=sidedefs,
        sectors=sectors, segs=segs, subsectors=subsectors,
        nodes=[ne_sw, nw_se, root],
    )


def _build_synthetic_subset(max_bsp_nodes: int = 16) -> MapSubset:
    """Hand-assemble a MapSubset from the synthetic box + BSP."""
    md = _synthetic_box_md()
    selected_orig = [0, 1, 2, 3]
    seg_to_ss = {s: s for s in range(4)}  # 1-seg-per-ss
    root_idx = len(md.nodes) - 1
    paths = _walk_paths(md, root_idx)
    subset_node_ids = set()
    for ss in {seg_to_ss[s] for s in selected_orig}:
        for (n, _) in paths[ss]:
            subset_node_ids.add(n)
    sorted_old = sorted(subset_node_ids)
    old_to_new = {old: new for new, old in enumerate(sorted_old)}
    bsp_nodes = [_make_plane(md.nodes[o]) for o in sorted_old]
    _ss, fc, bc = _count_selected_in_subtree(
        md, set(selected_orig), root_idx,
    )
    coeffs, consts = _compute_coefficients(
        md, selected_orig, seg_to_ss, paths, fc, bc,
        old_to_new, max_bsp_nodes=max_bsp_nodes,
    )

    # 4 simple solid-color textures (seg-0..3 → textures 0..3)
    textures = default_texture_atlas()
    segments = []
    for W_idx in selected_orig:
        seg = md.segs[W_idx]
        v1 = md.vertices[seg.v1]
        v2 = md.vertices[seg.v2]
        segments.append(Segment(
            ax=float(v1.x), ay=float(v1.y),
            bx=float(v2.x), by=float(v2.y),
            color=(0.5, 0.5, 0.5),
            texture_id=W_idx,
        ))

    return MapSubset(
        segments=segments,
        textures=textures,
        tex_name_to_id={f"TEX{i}": i for i in range(4)},
        bsp_nodes=bsp_nodes,
        seg_bsp_coeffs=coeffs,
        seg_bsp_consts=consts,
        original_seg_indices=list(selected_orig),
    )


def _small_config() -> RenderConfig:
    return RenderConfig(
        screen_width=16, screen_height=20, fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBspIntegration:
    """Tests sharing a single compiled graph to amortize compile cost."""

    @pytest.fixture(scope="class")
    def subset(self) -> MapSubset:
        return _build_synthetic_subset(max_bsp_nodes=16)

    @pytest.fixture(scope="class")
    def module(self, subset):
        config = _small_config()
        return compile_game(
            config, subset.textures,
            max_walls=8, max_bsp_nodes=16,
            d=2048, d_head=32, verbose=False,
        )

    def test_module_metadata_includes_max_bsp_nodes(self, module) -> None:
        assert int(module.metadata.get("max_bsp_nodes", 0)) == 16

    def test_module_has_bsp_inputs(self, module) -> None:
        input_names = {name for name, _, _ in module._input_specs}
        for required in (
            "bsp_plane_nx", "bsp_plane_ny", "bsp_plane_d",
            "bsp_node_id_onehot",
            "wall_bsp_coeffs", "wall_bsp_const",
        ):
            assert required in input_names, f"missing input: {required}"

    def test_renders_box_room_via_subset(self, module, subset) -> None:
        """Using a real BSP-aware MapSubset, the compiled graph renders
        a plausible frame of the 4-wall box room."""
        config = _small_config()
        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, PlayerInput(), subset, config)

        assert frame.shape == (20, 16, 3)
        # Has some wall pixels (not entirely ceiling/floor)
        ceil = np.array(config.ceiling_color)
        floor = np.array(config.floor_color)
        diff_ceil = np.abs(frame - ceil).sum(axis=-1)
        diff_floor = np.abs(frame - floor).sum(axis=-1)
        wall_mask = (diff_ceil > 1e-3) & (diff_floor > 1e-3)
        assert wall_mask.any(), "no wall pixels rendered"

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_renders_in_all_four_directions(self, module, subset, angle) -> None:
        """Looking in any cardinal direction inside the box should produce
        a visible wall."""
        config = _small_config()
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, PlayerInput(), subset, config)

        ceil = np.array(config.ceiling_color)
        floor = np.array(config.floor_color)
        diff_ceil = np.abs(frame - ceil).sum(axis=-1)
        diff_floor = np.abs(frame - floor).sum(axis=-1)
        wall_mask = (diff_ceil > 1e-3) & (diff_floor > 1e-3)
        assert wall_mask.any(), f"angle={angle}: no wall pixels rendered"

    def test_second_frame_renders(self, module, subset) -> None:
        """Calling step_frame twice in a row must produce a valid second
        frame.  Regression test for a bug where subsequent frames
        early-terminate at 4 render steps (one per wall, no chunks),
        producing a blank ceiling/floor-only image.
        """
        config = _small_config()
        state1 = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        frame1, state2 = step_frame(module, state1, PlayerInput(forward=True),
                                    subset, config)
        frame2, _ = step_frame(module, state2, PlayerInput(forward=True),
                               subset, config)

        ceil = np.array(config.ceiling_color)
        floor = np.array(config.floor_color)

        def wall_pixel_count(frame):
            diff_ceil = np.abs(frame - ceil).sum(axis=-1)
            diff_floor = np.abs(frame - floor).sum(axis=-1)
            return int(((diff_ceil > 1e-3) & (diff_floor > 1e-3)).sum())

        n1 = wall_pixel_count(frame1)
        n2 = wall_pixel_count(frame2)
        assert n1 > 0, "frame 1 should have wall pixels"
        # Allow some variation (different player position), but frame 2
        # should not collapse to ~zero wall pixels.
        assert n2 > n1 // 4, (
            f"frame 2 has {n2} wall pixels vs frame 1's {n1} — "
            "second frame appears to have early-terminated"
        )

    def test_matches_reference_render(self, module, subset) -> None:
        """Compiled output is close to the reference renderer's frame.

        The BSP rank sort should produce the same visual output as the
        reference renderer (which uses per-column closest-hit).  Some
        numerical error is expected (fp32 accumulation, piecewise-
        linear approximations).
        """
        config = _small_config()
        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, PlayerInput(), subset, config)

        ref = np.zeros_like(frame)
        for col in range(config.screen_width):
            from torchwright.reference_renderer.render import render_column
            ref[:, col, :] = render_column(
                col, 0.0, 0.0, 0, subset.segments, config,
                textures=subset.textures,
            )
        max_err = np.abs(frame - ref).max()
        assert max_err < 0.65, (
            f"compiled render diverges from reference: max_err={max_err:.3f}"
        )


# ---------------------------------------------------------------------------
# Rank math sanity (no compile needed)
# ---------------------------------------------------------------------------


def test_python_side_P_matches_graph_encoding() -> None:
    """Verify the 0/1 side_P encoding is consistent with the graph math.

    The graph computes:
        raw = nx*px + ny*py + d
        side_P = +1 (graph bool) if raw > 0 else -1; bool_to_01 → {0, 1}
    Python's :func:`side_P` must agree: 1 ↔ front (raw>0), 0 ↔ back.
    """
    md = _synthetic_box_md()
    root = md.nodes[-1]  # root BSP node
    plane = _make_plane(root)
    # Player at (5, 0): x>0, raw=x=5>0 → FRONT → side_P=1
    assert side_P(plane, 5, 0) == 1
    # Player at (-5, 0): x<0, raw=-5<0 → BACK → side_P=0
    assert side_P(plane, -5, 0) == 0


def test_bsp_traversal_order_consistent() -> None:
    """Reference BSP traversal produces stable order in the synthetic box."""
    md = _synthetic_box_md()
    order = bsp_traversal_order(md, 5.0, 5.0)
    # At (5, 5): front of root (x>0), back of ne_sw (y>0) → east first,
    # then front of ne_sw (south), then front of root's back (nw_se):
    # back of nw_se (y>0) = north, then front = west.
    # Expected: [east=seg0, south=seg3, north=seg1, west=seg2]
    assert order == [0, 3, 1, 2]
