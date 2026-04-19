"""Affine bound soundness and tightness on the DOOM game graph.

Builds the full game graph with a box-room scene, runs reference_eval
with realistic prefill inputs, and verifies:

1. **Soundness**: every node's observed values fall within its affine
   bound intervals.  Nodes inside the token-type detection chain and
   gate-MLP internals that take detection flags as conditions are
   excluded — the E8 spherical codes have components up to +/-30 but
   the token_type InputNode declares value_range=(-1, 1).  This is a
   known pre-existing mismatch.  Semantic overrides at cond_gate/select
   boundaries firewall the downstream game logic, so nodes after those
   boundaries ARE checked.
2. **Finiteness**: no non-input node has infinite affine bounds.
3. **Tightness**: bound widths and cond_gate M values stay within
   ratcheted thresholds.  As affine bounds tighten, lower them.
"""

import math
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pytest
import torch

from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import reference_eval
from torchwright.doom.game_graph import (
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
    build_game_graph,
)
from torchwright.doom.map_subset import build_scene_subset
from torchwright.graph.misc import Assert, Concatenate, InputNode, Placeholder
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.session import fresh_graph_session
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

MAX_WALLS = 4
MAX_BSP_NODES = 8
MAX_COORD = 20.0


def _config() -> RenderConfig:
    return RenderConfig(
        screen_width=8,
        screen_height=10,
        fov_columns=8,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _segments(half=5.0) -> List[Segment]:
    return [
        Segment(
            ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0
        ),
        Segment(
            ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1
        ),
        Segment(
            ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2
        ),
        Segment(
            ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3
        ),
    ]


# ---------------------------------------------------------------------------
# Input builder for reference_eval
# ---------------------------------------------------------------------------


def _build_input_values(
    input_nodes: List[InputNode],
    subset,
    textures: List[np.ndarray],
    *,
    px: float,
    py: float,
    angle: float,
) -> Dict[str, torch.Tensor]:
    """Build a realistic prefill input dict for reference_eval.

    Token sequence: TEX_COL x (num_tex x tex_w) + INPUT + BSP_NODE x N_bsp
    + WALL x N_wall + EOS.
    """
    num_tex = len(textures)
    tex_w = textures[0].shape[0]
    n_bsp = min(len(subset.bsp_nodes), MAX_BSP_NODES)
    n_walls = len(subset.segments)

    n_tex_col = num_tex * tex_w
    n_pos = n_tex_col + 1 + MAX_BSP_NODES + n_walls + 1

    vals: Dict[str, torch.Tensor] = {}
    for n in input_nodes:
        vals[n.name] = torch.zeros(n_pos, n.d_output)

    def _set(name: str, pos: int, value):
        if isinstance(value, (int, float)):
            value = torch.tensor([value])
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value).float()
        if value.dim() == 0:
            value = value.unsqueeze(0)
        vals[name][pos, : len(value)] = value

    def _set_common(pos: int):
        _set("player_x", pos, px)
        _set("player_y", pos, py)
        _set("player_angle", pos, angle)

    p = 0

    for tex_idx in range(num_tex):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            _set("token_type", p, tex_e8)
            _set("texture_id_e8", p, tex_e8)
            _set("tex_col_input", p, float(col))
            pixel_data = textures[tex_idx][col].flatten()
            _set("tex_pixels", p, pixel_data)
            _set_common(p)
            p += 1

    _set("token_type", p, E8_INPUT)
    _set("input_forward", p, 1.0)
    _set_common(p)
    p += 1

    for i in range(MAX_BSP_NODES):
        _set("token_type", p, E8_BSP_NODE)
        onehot = torch.zeros(MAX_BSP_NODES)
        onehot[i] = 1.0
        _set("bsp_node_id_onehot", p, onehot)
        if i < n_bsp:
            plane = subset.bsp_nodes[i]
            _set("bsp_plane_nx", p, plane.nx)
            _set("bsp_plane_ny", p, plane.ny)
            _set("bsp_plane_d", p, plane.d)
        _set_common(p)
        p += 1

    for i, seg in enumerate(subset.segments):
        _set("token_type", p, E8_WALL)
        _set("wall_ax", p, seg.ax)
        _set("wall_ay", p, seg.ay)
        _set("wall_bx", p, seg.bx)
        _set("wall_by", p, seg.by)
        _set("wall_tex_id", p, float(seg.texture_id))
        _set("wall_index", p, float(i))
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :MAX_BSP_NODES], dtype=torch.float32
        )
        _set("wall_bsp_coeffs", p, coeffs)
        _set("wall_bsp_const", p, float(subset.seg_bsp_consts[i]))
        _set_common(p)
        p += 1

    _set("token_type", p, E8_EOS)
    _set_common(p)
    p += 1

    assert p == n_pos
    return vals


# ---------------------------------------------------------------------------
# Token-type taint tracking
# ---------------------------------------------------------------------------

_GATE_MLP_INTERNAL_NAMES = frozenset(
    [
        "select_linear1",
        "select_relu",
        "cond_gate_linear1",
        "cond_gate_relu",
        "cond_gate_c_off_linear1",
        "cond_gate_c_off_relu",
        "cond_gate_c_off_linear2",
    ]
)


def _find_tainted_nodes(
    all_nodes,
    start_node,
    *,
    stop_at_overrides: bool = True,
) -> Set[int]:
    """Find nodes whose affine bounds are tainted by a wrong input range.

    Walks forward from *start_node* through the graph.

    When *stop_at_overrides* is True, stops propagation at
    semantic-override boundaries (select_linear2, cond_gate_linear2)
    — their affine bounds depend only on the gated data, not the
    condition.  Use this for E8-coded inputs where the condition
    channel is wrong but the data channel is correct.

    When *stop_at_overrides* is False, propagates through everything.
    Use this for feedback inputs whose declared range is wrong — the
    data itself (not just the condition) has an incorrect range, so
    semantic overrides don't firewall the taint.
    """
    consumers: Dict[int, Set] = defaultdict(set)
    for n in all_nodes:
        for inp in n.inputs:
            consumers[inp.node_id].add(n)

    tainted = {start_node.node_id}
    queue = [start_node]
    while queue:
        n = queue.pop(0)
        for c in consumers.get(n.node_id, set()):
            if c.node_id in tainted:
                continue
            tainted.add(c.node_id)
            if stop_at_overrides:
                name = getattr(c, "name", "") or ""
                if "select_linear2" in name or "cond_gate_linear2" in name:
                    continue
            queue.append(c)

    return tainted


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def doom_graph_eval():
    """Build doom graph + run reference_eval; shared across all tests."""
    config = _config()
    textures = default_texture_atlas()
    segs = _segments()
    subset = build_scene_subset(segs, textures, max_bsp_nodes=MAX_BSP_NODES)

    with fresh_graph_session():
        gio, pos_enc = build_game_graph(
            config,
            textures,
            max_walls=MAX_WALLS,
            max_bsp_nodes=MAX_BSP_NODES,
            max_coord=MAX_COORD,
            chunk_size=8,
        )
        output = gio.concat_output()
        all_nodes = get_ancestor_nodes({output, pos_enc})
        input_nodes = [n for n in all_nodes if isinstance(n, InputNode)]

        # Known input-range mismatches:
        # - token_type, texture_id_e8: E8 spherical codes have components
        #   up to +/-30, but these InputNodes declare value_range=(-1, 1).
        # - sort_feedback, render_feedback: composite vectors whose
        #   sub-fields have heterogeneous ranges (E8 codes, column
        #   indices, coordinates) covered by a single (-max_coord,
        #   max_coord) declaration that doesn't fit all sub-fields.
        # Taint all nodes whose affine bounds are derived from these.
        # E8 inputs: condition channel is wrong, data channel is
        # correct after semantic overrides.
        e8_names = {"token_type", "texture_id_e8"}
        # Feedback inputs: the data itself has wrong declared range,
        # so taint propagates through semantic overrides.
        feedback_names = {"sort_feedback", "render_feedback"}

        tainted: Set[int] = set()
        for n in input_nodes:
            if n.name in e8_names:
                tainted |= _find_tainted_nodes(all_nodes, n, stop_at_overrides=True)
            elif n.name in feedback_names:
                tainted |= _find_tainted_nodes(all_nodes, n, stop_at_overrides=False)

        input_values = _build_input_values(
            input_nodes,
            subset,
            textures,
            px=0.0,
            py=0.0,
            angle=45.0,
        )

        n_pos = next(iter(input_values.values())).shape[0]

        orig_check = Assert._check
        Assert._check = lambda self, x: None
        try:
            cache = reference_eval(output, input_values, n_pos)
        finally:
            Assert._check = orig_check

    return {
        "all_nodes": all_nodes,
        "cache": cache,
        "tainted": tainted,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_SKIP_TYPES = (InputNode, Concatenate, Placeholder, PosEncoding)


def _node_label(node) -> str:
    ann = node.annotation or ""
    return f"{node.node_type()}(id={node.node_id}, name='{node.name}', ann='{ann}')"


class TestAffineBoundSoundness:
    """Observed values from reference_eval must be within affine bounds.

    Nodes tainted by the token_type value_range mismatch are excluded:
    the E8 spherical codes have components up to +/-30 but the InputNode
    declares value_range=(-1, 1).  The taint propagates through the
    detection chain and gate-MLP internals, but semantic overrides at
    cond_gate/select boundaries create firewalls — everything after a
    semantic override IS checked.
    """

    def test_all_nodes_within_bounds(self, doom_graph_eval):
        all_nodes = doom_graph_eval["all_nodes"]
        cache = doom_graph_eval["cache"]
        tainted = doom_graph_eval["tainted"]

        checked = 0
        violations = []
        for node in all_nodes:
            if isinstance(node, _SKIP_TYPES):
                continue
            if node.node_id in tainted:
                continue
            tensor = cache.get(node)
            if tensor is None:
                continue
            checked += 1
            intervals = node.affine_bound.to_interval()
            for j in range(node.d_output):
                observed_lo = tensor[:, j].min().item()
                observed_hi = tensor[:, j].max().item()
                bound_lo = intervals[j].lo
                bound_hi = intervals[j].hi
                tol = 0.1
                if observed_lo < bound_lo - tol:
                    violations.append(
                        f"{_node_label(node)} component {j}: "
                        f"observed_lo={observed_lo:.6f} < bound_lo={bound_lo:.6f}"
                    )
                if observed_hi > bound_hi + tol:
                    violations.append(
                        f"{_node_label(node)} component {j}: "
                        f"observed_hi={observed_hi:.6f} > bound_hi={bound_hi:.6f}"
                    )

        assert checked > 0, "No nodes were checked — filter too aggressive"
        assert not violations, (
            f"{len(violations)} affine bound violations "
            f"({checked} nodes checked):\n" + "\n".join(violations[:30])
        )


class TestAffineBoundFiniteness:
    """Every non-input node should have finite affine bounds."""

    def test_no_infinite_bounds(self, doom_graph_eval):
        all_nodes = doom_graph_eval["all_nodes"]

        infinite_nodes = []
        for node in all_nodes:
            if isinstance(node, _SKIP_TYPES):
                continue
            r = node.value_type.value_range
            if not r.is_finite():
                infinite_nodes.append(f"{_node_label(node)}: range=[{r.lo}, {r.hi}]")

        assert (
            not infinite_nodes
        ), f"{len(infinite_nodes)} nodes with infinite bounds:\n" + "\n".join(
            infinite_nodes[:30]
        )


class TestAffineBoundTightness:
    """Bound widths should stay reasonable across the graph.

    Thresholds are ratchets: as affine bounds tighten, lower them.
    """

    def test_max_width(self, doom_graph_eval):
        """Track the worst-case bound width across the graph.

        Current worst case: ~42 million in wall/visibility
        (squared-distance chain through multiply_const).  Threshold
        is a ratchet — lower it as bounds tighten.
        """
        all_nodes = doom_graph_eval["all_nodes"]

        MAX_WIDTH = 50_000_000
        wide_nodes = []
        for node in all_nodes:
            if isinstance(node, _SKIP_TYPES):
                continue
            r = node.value_type.value_range
            if not r.is_finite():
                continue
            width = r.hi - r.lo
            if width > MAX_WIDTH:
                wide_nodes.append(
                    f"{_node_label(node)}: width={width:.0f} "
                    f"range=[{r.lo:.0f}, {r.hi:.0f}]"
                )

        assert (
            not wide_nodes
        ), f"{len(wide_nodes)} nodes with width > {MAX_WIDTH}:\n" + "\n".join(
            wide_nodes[:30]
        )

    def test_bound_coverage(self, doom_graph_eval):
        """Affine bounds should not be wildly wider than observed values.

        For each non-tainted node with a non-trivial observed range,
        computes the overestimation ratio:

            overestimation = bound_width / observed_width

        A ratio of 1 is perfect tightness; 1000 means the bound is
        1000x wider than what the game actually produces at this
        scene/camera.

        The test checks the 90th-percentile overestimation across all
        such nodes.  With a single prefill scene the observed ranges
        are narrow (many code paths inactive), so some overestimation
        is expected.  Threshold is a ratchet.
        """
        all_nodes = doom_graph_eval["all_nodes"]
        cache = doom_graph_eval["cache"]
        tainted = doom_graph_eval["tainted"]

        MIN_OBSERVED_WIDTH = 0.01
        ratios = []
        for node in all_nodes:
            if isinstance(node, _SKIP_TYPES):
                continue
            if node.node_id in tainted:
                continue
            tensor = cache.get(node)
            if tensor is None:
                continue
            r = node.value_type.value_range
            if not r.is_finite():
                continue
            bound_width = r.hi - r.lo
            if bound_width < MIN_OBSERVED_WIDTH:
                continue
            observed_lo = tensor.min().item()
            observed_hi = tensor.max().item()
            observed_width = observed_hi - observed_lo
            if observed_width < MIN_OBSERVED_WIDTH:
                continue
            ratios.append(bound_width / observed_width)

        assert len(ratios) > 0, "No nodes with non-trivial observed range"

        ratios.sort()
        p50 = ratios[len(ratios) // 2]
        p90 = ratios[int(len(ratios) * 0.9)]
        worst = ratios[-1]

        # Current state: p50=20x, p90=1199x, max=21064419x.
        # The long tail is wall/visibility squared-distance chains.
        MAX_P90 = 5_000
        assert p90 <= MAX_P90, (
            f"90th-percentile overestimation {p90:.0f}x exceeds "
            f"{MAX_P90}x (median={p50:.0f}x, worst={worst:.0f}x, "
            f"n={len(ratios)} nodes)"
        )

    def test_gate_coverage(self, doom_graph_eval):
        """Gate M values should not be wildly larger than needed.

        For each cond_gate/select output with non-trivial observed
        range, computes:

            M_overestimation = M_from_bound / M_from_observed

        A ratio of 1 means M is exactly as large as needed; 100
        means M (and the associated precision loss) is 100x larger
        than the game actually requires.

        Threshold is a ratchet — lower it as bounds tighten.
        """
        from torchwright.graph.linear import Linear
        from torchwright.ops.logic_ops import _GATE_OFFSET_SAFETY_FACTOR

        all_nodes = doom_graph_eval["all_nodes"]
        cache = doom_graph_eval["cache"]

        MIN_OBSERVED_ABS = 0.01
        ratios = []
        for node in all_nodes:
            if not isinstance(node, Linear):
                continue
            name = node.name or ""
            if "cond_gate_linear2" not in name and "select_linear2" not in name:
                continue
            r = node.value_type.value_range
            if not r.is_finite():
                continue
            tensor = cache.get(node)
            if tensor is None:
                continue
            observed_abs = max(abs(tensor.min().item()), abs(tensor.max().item()))
            if observed_abs < MIN_OBSERVED_ABS:
                continue
            bound_abs = max(abs(r.lo), abs(r.hi))
            ratios.append(bound_abs / observed_abs)

        assert len(ratios) > 0, "No gate sites with non-trivial observed range"

        ratios.sort()
        p50 = ratios[len(ratios) // 2]
        p90 = ratios[int(len(ratios) * 0.9)]
        worst = ratios[-1]

        # Current state: p50=5x, p90=1231x, max=4990x.
        MAX_P90 = 5_000
        assert p90 <= MAX_P90, (
            f"90th-percentile gate M overestimation {p90:.0f}x exceeds "
            f"{MAX_P90}x (median={p50:.0f}x, worst={worst:.0f}x, "
            f"n={len(ratios)} gates)"
        )

    def test_gate_M_bounded(self, doom_graph_eval):
        """cond_gate/select offset M should not exceed 50000.

        M = 2 * max(|lo|, |hi|) on the gated input.  Current worst
        case is ~42742 in wall/collision (squared-distance chain).
        Threshold is a ratchet.
        """
        from torchwright.graph.linear import Linear
        from torchwright.ops.logic_ops import _GATE_OFFSET_SAFETY_FACTOR

        all_nodes = doom_graph_eval["all_nodes"]

        MAX_M = 50_000
        bad_gates = []
        for node in all_nodes:
            if not isinstance(node, Linear):
                continue
            name = node.name or ""
            if "cond_gate_linear2" not in name and "select_linear2" not in name:
                continue
            r = node.value_type.value_range
            if not r.is_finite():
                bad_gates.append(f"{_node_label(node)}: infinite range")
                continue
            M = _GATE_OFFSET_SAFETY_FACTOR * max(abs(r.lo), abs(r.hi))
            if M > MAX_M:
                bad_gates.append(
                    f"{_node_label(node)}: M={M:.1f} " f"range=[{r.lo:.4f}, {r.hi:.4f}]"
                )

        assert (
            not bad_gates
        ), f"{len(bad_gates)} gate sites with M > {MAX_M}:\n" + "\n".join(
            bad_gates[:30]
        )
