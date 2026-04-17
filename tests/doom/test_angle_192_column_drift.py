"""Angle-192 diagnostic regressions: residual-column drift + SORTED attention.

These tests pin down behaviour that the full-game compile exhibits at
``angle=192``: wall-input InputNodes must keep their host-fed values
through every post-MLP snapshot where they're alive, the collected
``score_gap_at_least`` assertion must hold on the compiled module, and
SORTED's above-threshold argmin must concentrate ≈100 % of its softmax
mass on the single renderable wall (south) at ``sort[0]``.

Each test drives the same synthetic subset + graph as the legacy
``compile_game`` pipeline and delegates the heavy lifting to the
diagnostic harness in :mod:`torchwright.debug.probe`.  The harness is
the reusable surface; this file is a concrete scene consumer.
"""

from typing import List, Tuple

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import (
    check_asserts_on_compiled,
    probe_attention,
    probe_layer_diff,
)
from torchwright.doom.compile import _build_row
from torchwright.doom.game_graph import (
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
    build_game_graph,
)
from torchwright.graph.attn import Attn
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.attention_ops import _ABOVE_BONUS

from tests.doom.test_bsp_rank_integration import (
    _build_synthetic_subset,
    _small_config,
)

_MAX_WALLS = 8
_MAX_BSP_NODES = 16
_D = 2048
_D_HEAD = 32


# ---------------------------------------------------------------------------
# Fixture: build graph + compile it with direct node references retained.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_and_module():
    """Build the full game graph with the same knobs as ``compile_game``
    and keep ``graph_io`` around so tests have direct references to
    wall_ax / wall_by / pos_encoding / outputs.
    """
    config = _small_config()
    subset = _build_synthetic_subset(max_bsp_nodes=_MAX_BSP_NODES)

    graph_io, pos_encoding = build_game_graph(
        config,
        subset.textures,
        max_walls=_MAX_WALLS,
        max_coord=20.0,
        move_speed=0.3,
        turn_speed=4,
        chunk_size=20,
        max_bsp_nodes=_MAX_BSP_NODES,
    )

    io: dict = {}
    for name, node in graph_io.inputs.items():
        io[name] = (node, graph_io.overlaid_outputs.get(name))
    for name, node in graph_io.overflow_outputs.items():
        assert name not in io
        io[name] = (None, node)

    from torchwright.graph.optimize import fuse_consecutive_linears

    output_nodes = set(graph_io.overlaid_outputs.values())
    output_nodes.update(graph_io.overflow_outputs.values())
    output_nodes.add(pos_encoding)
    while True:
        if fuse_consecutive_linears(output_nodes, verbose=False) == 0:
            break

    from torchwright.graph.asserts import collect_asserts

    collected_asserts: list = []
    _seen_ids: set = set()
    for out_node in list(graph_io.overlaid_outputs.values()) + list(
        graph_io.overflow_outputs.values()
    ):
        for a in collect_asserts(out_node):
            if a.node_id not in _seen_ids:
                _seen_ids.add(a.node_id)
                collected_asserts.append(a)

    module = compile_headless(
        pos_encoding,
        io=io,
        d=_D,
        d_head=_D_HEAD,
        max_layers=400,
        verbose=False,
        extra_metadata={
            "chunk_size": 20,
            "max_walls": _MAX_WALLS,
            "max_bsp_nodes": _MAX_BSP_NODES,
            "tex_h": subset.textures[0].shape[1],
            "asserts": collected_asserts,
        },
    )
    module.eval()
    return graph_io, module, subset, config


# ---------------------------------------------------------------------------
# Prefill builder — mirrors step_frame's construction for player at origin.
# ---------------------------------------------------------------------------


def _build_prefill(
    module,
    subset,
    config,
    *,
    angle: float,
    px: float = 0.0,
    py: float = 0.0,
) -> Tuple[torch.Tensor, int, int]:
    """Build the ``(n_prefill, d_input)`` flat input for the given pose.

    Layout: TEX_COL × (num_tex × tex_w), INPUT, BSP_NODE × max_bsp_nodes,
    WALL × n_walls, EOS.  Returns ``(prefill, wall_base_idx, eos_idx)``.
    Wall order matches ``subset.segments`` — east, north, west, south.
    """
    max_walls = int(module.metadata["max_walls"])
    max_bsp_nodes = int(module.metadata["max_bsp_nodes"])
    num_tex = len(subset.textures)
    tex_w = subset.textures[0].shape[0]

    common = dict(
        player_x=torch.tensor([px]),
        player_y=torch.tensor([py]),
        player_angle=torch.tensor([float(angle)]),
    )

    rows: List[torch.Tensor] = []

    for tex_idx in range(num_tex):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = subset.textures[tex_idx][col].flatten()
            rows.append(
                _build_row(
                    module,
                    max_walls,
                    token_type=E8_TEX_COL,
                    texture_id_e8=tex_e8,
                    tex_col_input=torch.tensor([float(col)]),
                    tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                    **common,
                )
            )

    rows.append(_build_row(module, max_walls, token_type=E8_INPUT, **common))

    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            plane = subset.bsp_nodes[i]
            nx, ny, d = plane.nx, plane.ny, plane.d
        else:
            nx, ny, d = 0.0, 0.0, 0.0
        rows.append(
            _build_row(
                module,
                max_walls,
                token_type=E8_BSP_NODE,
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
                **common,
            )
        )

    wall_base_idx = len(rows)
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes],
            dtype=torch.float32,
        )
        const = torch.tensor(
            [float(subset.seg_bsp_consts[i])],
            dtype=torch.float32,
        )
        rows.append(
            _build_row(
                module,
                max_walls,
                token_type=E8_WALL,
                wall_ax=torch.tensor([float(seg.ax)]),
                wall_ay=torch.tensor([float(seg.ay)]),
                wall_bx=torch.tensor([float(seg.bx)]),
                wall_by=torch.tensor([float(seg.by)]),
                wall_tex_id=torch.tensor([float(seg.texture_id)]),
                wall_index=torch.tensor([float(i)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
                **common,
            )
        )

    eos_idx = len(rows)
    rows.append(_build_row(module, max_walls, token_type=E8_EOS, **common))
    return torch.cat(rows, dim=0), wall_base_idx, eos_idx


def _sort_argmin_attn(graph_io) -> Attn:
    """Return the SORTED argmin Attn node — the one under
    ``sort/attention`` whose V-input width matches ``13 + max_walls``.
    """
    roots = set(graph_io.overlaid_outputs.values()) | set(
        graph_io.overflow_outputs.values()
    )
    candidates = [
        n
        for n in get_ancestor_nodes(roots)
        if isinstance(n, Attn) and (n.annotation or "").startswith("sort/attention")
    ]
    expected_v_width = 13 + _MAX_WALLS
    for attn in candidates:
        v_input = attn.inputs[2]
        if v_input is not None and len(v_input) == expected_v_width:
            return attn
    raise AssertionError(
        f"no sort/attention Attn with V-width={expected_v_width}; "
        f"update the filter if the SORTED payload layout changed"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wall_inputs_stable_through_layers_at_angle_192(graph_and_module):
    """wall_ax/ay/bx/by InputNodes must hold their host-fed value at
    every post-MLP snapshot that materialises them.  Any drift indicates
    residual-column aliasing — a compiler-side bug, not a numerical
    approximation issue.
    """
    graph_io, module, subset, config = graph_and_module
    prefill, wall_base_idx, _ = _build_prefill(
        module,
        subset,
        config,
        angle=192.0,
    )
    wall_positions = list(range(wall_base_idx, wall_base_idx + len(subset.segments)))
    in_specs = {n: (s, w) for n, s, w in module._input_specs}

    for field in ("wall_ax", "wall_ay", "wall_bx", "wall_by"):
        node = graph_io.inputs[field]
        s, w = in_specs[field]
        reference = prefill[wall_positions, s : s + w]
        report = probe_layer_diff(
            module,
            prefill,
            node,
            reference=reference,
            positions=wall_positions,
            drift_threshold=1e-3,
        )
        assert report.records, f"{field}: no layers materialised the node"
        assert report.first_drift_layer is None, (
            f"{field} drifted at layer {report.first_drift_layer} "
            f"(max_abs_delta={report.records[0].max_abs_delta:.4g}): "
            f"a compiled InputNode value must match its host-fed input."
        )


def test_score_gap_assert_at_angle_192_compiled(graph_and_module):
    """The compiled ``score_gap_at_least`` predicate must hold.

    At angle-192 only south is renderable, so the valid set at sort[0]
    is a singleton and the margin-1.0 check is vacuously satisfied.
    The test locks that invariant: if renderability routing regresses
    and two walls become simultaneously valid, the clean-integer rank
    spacing (1.0) would be eaten by any compile-side precision loss
    and this assertion would fire.
    """
    _graph_io, module, subset, config = graph_and_module
    asserts = module.metadata["asserts"]
    score_gap_asserts = [
        a for a in asserts if "score_gap_at_least" in (a.message or "")
    ]
    assert score_gap_asserts, "expected at least one score_gap_at_least assert"

    prefill, _wbi, _eos = _build_prefill(
        module,
        subset,
        config,
        angle=192.0,
    )
    n_pos = prefill.shape[0]
    input_values = {
        name: prefill[:, start : start + width]
        for name, start, width in module._input_specs
    }
    check_asserts_on_compiled(module, score_gap_asserts, input_values, n_pos)


def test_sorted_argmin_attention_concentrates_on_south_at_sort0_angle_192(
    graph_and_module,
):
    """SORTED argmin-above-integer must put ≈100 % of its softmax mass
    on the single renderable wall (south) at ``sort[0]``.

    Non-renderable walls (east, west — parallel to view; north — behind
    player) have all-zero ``indicators_above``, so the ``_ABOVE_BONUS``
    additive bias in the query matrix cannot fire for them.  The logit
    gap between south and any invalid wall is the full
    ``_ABOVE_BONUS`` at reference eval; the compiled path must retain
    enough of that margin to keep south's softmax weight above 0.999.
    """
    graph_io, module, subset, config = graph_and_module
    net = module._net
    argmin_attn = _sort_argmin_attn(graph_io)

    prefill, wall_base_idx, _ = _build_prefill(
        module,
        subset,
        config,
        angle=192.0,
    )
    wall_positions = list(range(wall_base_idx, wall_base_idx + len(subset.segments)))

    # Drive the prefill to establish the KV cache, then build the
    # sort[0] input from the EOS row's overlaid outputs and probe the
    # attention weights on that single decode step.
    past = module.empty_past()
    with torch.no_grad():
        pre_out, past = module.step(prefill, past, past_len=0)
    step = prefill.shape[0]

    out_specs = {n: (s, w) for n, s, w in module._output_specs}
    in_specs = {n: (s, w) for n, s, w in module._input_specs}
    overlaid_names = [n for n in in_specs if n in out_specs]
    d_input = max(s + w for _, s, w in module._input_specs)
    device = module._net.device

    sort0_in = torch.zeros(1, d_input, device=device)
    for name in overlaid_names:
        in_s, w = in_specs[name]
        out_s, _ = out_specs[name]
        sort0_in[0, in_s : in_s + w] = pre_out[-1, out_s : out_s + w]

    past_K, past_V = past
    past_kvs = [(past_K[i], past_V[i]) for i in range(len(past_K))]
    report = probe_attention(
        module,
        sort0_in,
        argmin_attn,
        query_pos=0,
        past_len=step,
        past_kvs=past_kvs,
    )

    # Find the head that concentrates on a WALL position at sort[0].
    wall_pos_set = set(wall_positions)
    head_idx, best_wall_mass = -1, -1.0
    for h in range(report.weights.shape[0]):
        wall_mass = sum(float(report.weights[h, p]) for p in wall_pos_set)
        if wall_mass > best_wall_mass:
            head_idx, best_wall_mass = h, wall_mass
    assert head_idx >= 0
    south_pos = wall_positions[3]
    w_south = float(report.weights[head_idx, south_pos])
    l_south = float(report.logits[head_idx, south_pos])
    invalid_logits = [
        float(report.logits[head_idx, wall_positions[i]]) for i in (0, 1, 2)
    ]
    min_gap = min(l_south - li for li in invalid_logits)

    assert w_south > 0.999, (
        f"south should dominate at sort[0]: weight={w_south:.6f}, "
        f"head={head_idx}, wall-mass={best_wall_mass:.6f}, "
        f"logit gap to worst invalid = {min_gap:.2f}"
    )
    assert min_gap > 0.5 * _ABOVE_BONUS, (
        f"above/below logit gap {min_gap:.0f} is less than half of the "
        f"design separation {_ABOVE_BONUS:.0f} — compile-side precision "
        f"loss has eroded the above-threshold signal."
    )
