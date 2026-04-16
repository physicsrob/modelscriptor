"""Phase E demo trace — exercise the harness against the formerly-xfailed
``(px=3, py=2, angle=20)`` scene in ``box_room``.

This test demonstrates :mod:`torchwright.debug.probe` against the
Phase E regression that motivated the de-risking initiative.  The
original symptom was a deterministic ``-_ABOVE_BONUS`` (== -1000)
landing in the ``sel_bsp_rank`` residual columns at ``sort[0]``;
commit ``a979f69`` (the sum_nodes-thermometer refactor, 100→70 layers)
collaterally fixed the underlying precision loss so the sentinel no
longer surfaces on main.  That leaves the xfail at
``test_renders_off_center_oblique[3.0-2.0-20]`` in
``XPASS(strict)=fail`` state — a follow-up cleanup that belongs to
plan 6, out of scope here.

What this test still buys:

1. It demonstrates the harness end-to-end on the same compile config
   (``compile_game``-equivalent, ``d=2048``, ``d_head=32``) and scene
   that triggered the original regression.
2. It prints a per-layer trace of ``sel_bsp_rank`` at ``sort[0]`` so
   humans (and future regression triage) can see the actual value
   trajectory — the same kind of output plan 6 will need.
3. It sentinel-probes for ``-_ABOVE_BONUS`` and records whether it
   surfaces; today it must not, which lock-steps the fix in place.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import probe_layer_diff
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
from torchwright.doom.map_subset import build_scene_subset
from torchwright.graph.misc import Concatenate, LiteralValue
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.attention_ops import _ABOVE_BONUS
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


# Exactly the knobs that ``tests/doom/test_game_graph.py::TestGameGraph``
# uses — the xfail's compile config.  Reproducing the bug requires the
# same compile path, same texture atlas, and same segments.
_MAX_WALLS = 8
_MAX_BSP_NODES = 48
_D = 2048
_D_HEAD = 32


def _box_room_config() -> RenderConfig:
    return RenderConfig(
        screen_width=16, screen_height=20, fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _box_room_segments(half: float = 5.0):
    return [
        Segment(ax=half, ay=-half, bx=half, by=half,
                color=(0.8, 0.2, 0.1), texture_id=0),
        Segment(ax=-half, ay=-half, bx=-half, by=half,
                color=(0.8, 0.2, 0.1), texture_id=1),
        Segment(ax=-half, ay=half, bx=half, by=half,
                color=(0.8, 0.2, 0.1), texture_id=2),
        Segment(ax=-half, ay=-half, bx=half, by=-half,
                color=(0.8, 0.2, 0.1), texture_id=3),
    ]


@pytest.fixture(scope="module")
def compiled_box_room():
    """Rebuild + compile the box_room game graph, keeping ``graph_io``.

    Mirrors ``compile_game``'s internals so we can resolve graph-IR
    nodes (specifically ``sel_bsp_rank``) against the compiled module's
    ``residual_assignment``.  Cannot use ``compile_game`` directly
    because it discards ``graph_io`` after compile returns.
    """
    config = _box_room_config()
    textures = default_texture_atlas()
    segs = _box_room_segments()
    subset = build_scene_subset(segs, textures)

    graph_io, pos_encoding = build_game_graph(
        config, textures,
        max_walls=_MAX_WALLS, max_coord=20.0,
        move_speed=0.3, turn_speed=4,
        chunk_size=20, max_bsp_nodes=_MAX_BSP_NODES,
    )

    from torchwright.graph.optimize import fuse_consecutive_linears
    output_nodes = set(graph_io.overlaid_outputs.values())
    output_nodes.update(graph_io.overflow_outputs.values())
    output_nodes.add(pos_encoding)
    while True:
        if fuse_consecutive_linears(output_nodes, verbose=False) == 0:
            break

    io: dict = {}
    for name, node in graph_io.inputs.items():
        io[name] = (node, graph_io.overlaid_outputs.get(name))
    for name, node in graph_io.overflow_outputs.items():
        io[name] = (None, node)

    module = compile_headless(
        pos_encoding, io=io,
        d=_D, d_head=_D_HEAD, max_layers=400,
        verbose=False,
        extra_metadata={
            "chunk_size": 20, "max_walls": _MAX_WALLS,
            "max_bsp_nodes": _MAX_BSP_NODES,
            "tex_h": textures[0].shape[1],
        },
    )
    module.eval()
    return graph_io, module, subset, textures


def _find_sel_bsp_rank(graph_io):
    """Walk ``graph_io.overlaid_outputs["sort_feedback"]`` and return
    the ``sel_bsp_rank`` Node — child index 2 of the SORTED
    feedback-packing Concatenate identified by its LiteralValue
    first-child named ``"sort_type"``.
    """
    roots = set(graph_io.overlaid_outputs.values())
    for node in get_ancestor_nodes(roots):
        if not isinstance(node, Concatenate) or len(node.inputs) != 6:
            continue
        first = node.inputs[0]
        if isinstance(first, LiteralValue) and first.name == "sort_type":
            return node.inputs[2]
    raise AssertionError(
        "could not locate the sort_feedback-packing Concatenate — did the "
        "SORTED payload layout change?  Expected Concatenate([sort_type, "
        "sel_wall_data, sel_bsp_rank, vis_lo, vis_hi, sel_onehot])."
    )


def _build_prefill(module, subset, *, px: float, py: float, angle: float):
    """Mirror ``step_frame``'s prefill construction for a single scene."""
    max_walls = int(module.metadata["max_walls"])
    max_bsp_nodes = int(module.metadata["max_bsp_nodes"])
    common = dict(
        player_x=torch.tensor([px]),
        player_y=torch.tensor([py]),
        player_angle=torch.tensor([float(angle)]),
    )
    tex_w = subset.textures[0].shape[0]

    rows = []
    for tex_idx in range(len(subset.textures)):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = subset.textures[tex_idx][col].flatten()
            rows.append(_build_row(
                module, max_walls,
                token_type=E8_TEX_COL,
                texture_id_e8=tex_e8,
                tex_col_input=torch.tensor([float(col)]),
                tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                **common,
            ))
    rows.append(_build_row(module, max_walls, token_type=E8_INPUT, **common))
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            plane = subset.bsp_nodes[i]
            nx, ny, d = plane.nx, plane.ny, plane.d
        else:
            nx, ny, d = 0.0, 0.0, 0.0
        rows.append(_build_row(
            module, max_walls,
            token_type=E8_BSP_NODE,
            bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
            bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
            bsp_plane_d=torch.tensor([d], dtype=torch.float32),
            bsp_node_id_onehot=onehot,
            **common,
        ))
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes], dtype=torch.float32,
        )
        const = torch.tensor(
            [float(subset.seg_bsp_consts[i])], dtype=torch.float32,
        )
        rows.append(_build_row(
            module, max_walls,
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
        ))
    rows.append(_build_row(module, max_walls, token_type=E8_EOS, **common))
    return torch.cat(rows, dim=0)


def test_sel_bsp_rank_trace_at_sort0_formerly_phase_e_scene(compiled_box_room):
    """Trace ``sel_bsp_rank`` across layers at ``sort[0]`` for scene
    (px=3, py=2, angle=20) in ``box_room`` using the harness's
    :func:`probe_layer_diff`.

    Demonstrates the full cached-decode probe path (past_len + past_kvs)
    on the exact compile config of the Phase E xfail.  The sentinel
    probe for ``-_ABOVE_BONUS`` must NOT fire today — commit a979f69
    collaterally fixed the underlying precision loss — and the assertion
    here locks that in: any future regression that re-surfaces -1000 in
    ``sel_bsp_rank`` breaks this test immediately, with the layer index
    printed in the trace above.
    """
    graph_io, module, subset, _textures = compiled_box_room
    sel_bsp_rank = _find_sel_bsp_rank(graph_io)
    assert len(sel_bsp_rank) == 1, (
        f"sel_bsp_rank width {len(sel_bsp_rank)} != 1 — update the walk"
    )

    prefill = _build_prefill(module, subset, px=3.0, py=2.0, angle=20.0)

    # Drive prefill → EOS to establish the KV cache, then build the
    # sort[0] input row from the EOS row's overlaid outputs (matches
    # ``step_frame``'s host bookkeeping).
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
        sort0_in[0, in_s:in_s + w] = pre_out[-1, out_s:out_s + w]

    past_K, past_V = past
    past_kvs = [(past_K[i], past_V[i]) for i in range(len(past_K))]

    report = probe_layer_diff(
        module, sort0_in, sel_bsp_rank,
        # Host-truth reference is irrelevant for sentinel-only tracing —
        # zeros here; callers who want drift flagging would populate
        # this from ``reference_eval(...)[sel_bsp_rank][[0]]``.
        reference=torch.zeros(1, 1),
        positions=[0],
        sentinel=-_ABOVE_BONUS,
        sentinel_tol=0.5,
        past_len=step,
        past_kvs=past_kvs,
    )
    assert report.records, (
        "sel_bsp_rank was not materialised at any post-MLP snapshot — "
        "probe found no layers"
    )

    print(f"\nsel_bsp_rank at sort[0] (scene px=3, py=2, angle=20):")
    for rec in report.records:
        print(
            f"  L{rec.layer_index:3d}  {rec.state_name:22s}  "
            f"val={rec.value[0, 0].item():+10.4f}"
        )

    assert report.first_sentinel_layer is None, (
        f"regression: -_ABOVE_BONUS (= {-_ABOVE_BONUS}) resurfaced in "
        f"sel_bsp_rank at layer {report.first_sentinel_layer} — Phase E "
        f"is back.  Inspect the printed trace above for the first layer "
        f"where the value crossed the sentinel tolerance; that's the "
        f"locus of the precision regression."
    )
