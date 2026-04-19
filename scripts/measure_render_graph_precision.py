"""Measure per-node fp32/fp64 divergence across the compiled DOOM graph.

Purpose: answer the question "what precision envelope do we have today?"
for the compiled DOOM renderer, across a handful of canonical scenes.
The output informs whether a regression-test version of this walk would
assert single aggregate thresholds, per-node-class envelopes, or a
committed baseline JSON.

Walks every materialised node in the compiled graph, compares to the
``node.compute()`` oracle, and reports per-node ``max|compiled - oracle|``.

- fp32 GPU pass at 4 scenes (the production regime).
- fp64 CPU replay at 1 scene (separates construction bugs from fp32
  drift — if fp64 errors don't collapse to <1e-6, the compiler is
  wiring something wrong).

Run via ``make modal-run MODULE=scripts.measure_render_graph_precision``.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled, reference_eval
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
from torchwright.graph import Concatenate, Linear
from torchwright.graph.attn import Attn
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

_MAX_WALLS = 8
_MAX_BSP_NODES = 48
_D = 2048
_D_HEAD = 32

_SCENES = [
    (0.0, 0.0, 0.0),  # axis-aligned, centered
    (0.0, 0.0, 45.0),  # oblique, centered
    (3.0, 2.0, 20.0),  # the Phase E scene
    (-2.0, 3.0, 240.0),  # off-center + oblique (existing test scene)
]


def _config():
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _segments(half=5.0):
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


def _build_prefill(module, subset, *, px, py, angle):
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
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes],
            dtype=torch.float32,
        )
        const = torch.tensor([float(subset.seg_bsp_consts[i])], dtype=torch.float32)
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
    rows.append(_build_row(module, max_walls, token_type=E8_EOS, **common))
    return torch.cat(rows, dim=0)


def _input_values_from_prefill(module, prefill):
    in_by_name = {name: (s, w) for name, s, w in module._input_specs}
    return {name: prefill[:, s : s + w].clone() for name, (s, w) in in_by_name.items()}


def _summarize_errors(label, errors):
    """Print distribution summary + top-20 table for a list of (node, max_err)."""
    vals = sorted((e for _, e in errors), reverse=True)
    n = len(vals)
    if n == 0:
        print(f"  {label}: no data")
        return
    p = lambda q: vals[min(n - 1, int(q * n))]  # noqa: E731
    print(
        f"  {label}:  n={n}  max={vals[0]:.4g}  "
        f"p99={p(0.01):.4g}  p95={p(0.05):.4g}  p90={p(0.10):.4g}  "
        f"p50={p(0.50):.4g}  min={vals[-1]:.4g}"
    )
    buckets = [
        (0, 1e-6),
        (1e-6, 1e-4),
        (1e-4, 1e-3),
        (1e-3, 1e-2),
        (1e-2, 1e-1),
        (1e-1, 1.0),
        (1.0, 10.0),
        (10.0, 100.0),
        (100.0, 1e6),
    ]
    for lo, hi in buckets:
        c = sum(1 for v in vals if lo <= v < hi)
        if c:
            print(f"    |Δ| ∈ [{lo:>9.2g}, {hi:<9.2g}):  {c:>5}")


def _top_nodes(errors, k=20):
    srt = sorted(errors, key=lambda x: -x[1])
    for node, err in srt[:k]:
        nm = (node.name or "").strip() or f"<{type(node).__name__}>"
        print(
            f"    id={node.node_id:>6}  |Δ|={err:>10.4g}  "
            f"{type(node).__name__:<15}  name={nm!r}"
        )


def _top_nodes_with_rel(records, k=20):
    """Print top-k nodes with absolute error, oracle scale, and relative error."""
    srt = sorted(records, key=lambda r: -r.max_abs_error)
    for rec in srt[:k]:
        n = rec.node
        nm = (n.name or "").strip() or f"<{type(n).__name__}>"
        oracle_scale = max(abs(rec.oracle_min), abs(rec.oracle_max))
        rel = rec.max_abs_error / oracle_scale if oracle_scale > 0 else float("inf")
        # Declared-range scale: max|value_range|
        vr = n.value_type.value_range
        if vr.is_finite():
            declared_scale = max(abs(vr.lo), abs(vr.hi))
            rel_decl = (
                rec.max_abs_error / declared_scale
                if declared_scale > 0
                else float("inf")
            )
            decl_str = f"rel/decl={rel_decl:>8.3%}"
        else:
            decl_str = "rel/decl=  (inf)"
        print(
            f"    id={n.node_id:>6}  |Δ|={rec.max_abs_error:>8.3g}  "
            f"oracle∈[{rec.oracle_min:+.2f},{rec.oracle_max:+.2f}]  "
            f"rel/obs={rel:>8.3%}  {decl_str}  "
            f"{type(n).__name__:<15} name={nm!r}"
        )


def main():
    print("=" * 72)
    print("DOOM render graph: per-node precision measurement")
    print("=" * 72)

    config = _config()
    textures = default_texture_atlas()
    segs = _segments()
    subset = build_scene_subset(segs, textures)

    print("\nBuilding game graph...")
    graph_io, pos_encoding = build_game_graph(
        config,
        textures,
        max_walls=_MAX_WALLS,
        max_coord=20.0,
        move_speed=0.3,
        turn_speed=4,
        chunk_size=20,
        max_bsp_nodes=_MAX_BSP_NODES,
    )

    from torchwright.graph.optimize import fuse_consecutive_linears

    output_nodes = set(graph_io.overlaid_outputs.values())
    output_nodes.update(graph_io.overflow_outputs.values())
    output_nodes.add(pos_encoding)
    while True:
        if fuse_consecutive_linears(output_nodes, verbose=False) == 0:
            break

    io: Dict[str, tuple] = {}
    for name, node in graph_io.inputs.items():
        io[name] = (node, graph_io.overlaid_outputs.get(name))
    for name, node in graph_io.overflow_outputs.items():
        io[name] = (None, node)

    print("Compiling (d=2048)...")
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
            "tex_h": textures[0].shape[1],
        },
    )
    module.eval()
    # Probe root: concat of all graph outputs, so the probe walks the
    # full forward graph (pos_encoding alone has no consumers under the
    # probe — its children feed the real outputs).
    final_outputs = list(graph_io.overlaid_outputs.values()) + list(
        graph_io.overflow_outputs.values()
    )
    probe_root = Concatenate(final_outputs)
    print(f"  compiled: {len(module._net.layers)} layers")

    # Per-scene fp32 GPU probes.
    all_errors_by_scene: Dict[tuple, List] = {}
    combined_errors: Dict[int, tuple] = {}  # node_id -> (node, max_err_across_scenes)
    # Collect per-scene ProbeReport records for relative-error analysis.
    combined_records: Dict[int, object] = {}  # node_id -> NodeDivergence w/ worst rec

    for px, py, angle in _SCENES:
        scene = (px, py, angle)
        print(f"\n--- Scene (px={px}, py={py}, angle={angle}) fp32 GPU ---")
        prefill = _build_prefill(module, subset, px=px, py=py, angle=angle)
        input_values = _input_values_from_prefill(module, prefill)
        n_pos = prefill.shape[0]
        # atol=1e9 so first_divergent classification is moot; we want the
        # full per-node distribution, not just divergent ones.
        report = probe_compiled(module, probe_root, input_values, n_pos, atol=1e9)
        errors = [(rec.node, rec.max_abs_error) for rec in report.per_node.values()]
        all_errors_by_scene[scene] = errors
        print(
            f"  checked {len(report.nodes_checked)} nodes, skipped {len(report.skipped)}"
        )
        _summarize_errors("distribution", errors)
        print("  top-10 noisiest:")
        _top_nodes(errors, k=10)
        # Accumulate into combined envelope: max_err over scenes.
        for rec in report.per_node.values():
            node = rec.node
            err = rec.max_abs_error
            prev = combined_errors.get(node.node_id)
            if prev is None or err > prev[1]:
                combined_errors[node.node_id] = (node, err)
                combined_records[node.node_id] = rec

    # Cross-scene envelope.
    print("\n=== Combined fp32 envelope (max|Δ| across all scenes per node) ===")
    combined_list = list(combined_errors.values())
    _summarize_errors("combined", combined_list)

    print("\n  top-30 nodes by worst-case fp32 error (with rel error):")
    _top_nodes_with_rel(list(combined_records.values()), k=30)

    # Per-op-class aggregation — useful for envelope strategy.
    print("\n  Per-op-class worst-case (max over nodes of that class):")
    by_class: Dict[str, tuple] = {}
    for node, err in combined_list:
        cls = type(node).__name__
        name_hint = (node.name or "").strip()
        # Separate linear1 vs linear2 vs other Linears.
        if cls == "Linear":
            if "linear1" in name_hint:
                cls = "Linear(linear1)"
            elif "linear2" in name_hint:
                cls = "Linear(linear2)"
            elif name_hint.startswith("bsp_"):
                cls = "Linear(bsp_*)"
        prev = by_class.get(cls)
        if prev is None or err > prev[1]:
            by_class[cls] = (node, err)
    for cls, (node, err) in sorted(by_class.items(), key=lambda kv: -kv[1][1]):
        nm = (node.name or "").strip()
        print(f"    {cls:<25}  worst |Δ|={err:>10.4g}  (worst node={nm!r})")

    # fp64 CPU replay on the Phase E scene only.
    print("\n=== fp64 CPU replay (scene = (3, 2, 20)) ===")
    px, py, angle = 3.0, 2.0, 20.0
    prefill_fp32 = _build_prefill(module, subset, px=px, py=py, angle=angle)
    n_pos = prefill_fp32.shape[0]

    print("  deepcopy + convert to fp64 on CPU...")
    _prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        module_cpu = copy.deepcopy(module)
        module_cpu._net.to("cpu")
        for _layer in module_cpu._net.layers:
            for _attr in (
                "query_matrix",
                "key_matrix",
                "value_matrix",
                "output_matrix",
            ):
                setattr(
                    _layer.attn.attn,
                    _attr,
                    getattr(_layer.attn.attn, _attr).to(torch.float64),
                )
            for _comp in (_layer.mlp.linear1, _layer.mlp.linear2):
                _comp.output_matrix = _comp.output_matrix.to(torch.float64)
                _comp.output_bias = _comp.output_bias.to(torch.float64)

        from torchwright.compiler.utils import get_ancestor_nodes

        all_graph_nodes = get_ancestor_nodes({probe_root})
        for _n in all_graph_nodes:
            if isinstance(_n, Linear):
                _n.output_matrix = _n.output_matrix.to(torch.float64)
                _n.output_bias = _n.output_bias.to(torch.float64)
            if isinstance(_n, Attn):
                _n.query_matrix = _n.query_matrix.to(torch.float64)
                _n.key_matrix = _n.key_matrix.to(torch.float64)
                _n.value_matrix = _n.value_matrix.to(torch.float64)
                _n.output_matrix = _n.output_matrix.to(torch.float64)
            if hasattr(_n, "value") and isinstance(_n.value, torch.Tensor):
                _n.value = _n.value.to(torch.float64)

        prefill_cpu_f64 = prefill_fp32.to(device="cpu", dtype=torch.float64)
        in_by_name = {name: (s, w) for name, s, w in module_cpu._input_specs}
        input_values_f64 = {
            name: prefill_cpu_f64[:, s : s + w].to(torch.float64)
            for name, (s, w) in in_by_name.items()
        }
        report_f64 = probe_compiled(
            module_cpu,
            probe_root,
            input_values_f64,
            n_pos=n_pos,
            atol=1e9,
        )
    finally:
        torch.set_default_dtype(_prev)

    errors_f64 = [(rec.node, rec.max_abs_error) for rec in report_f64.per_node.values()]
    _summarize_errors("fp64 distribution", errors_f64)
    print("  top-10 fp64 divergences:")
    _top_nodes(errors_f64, k=10)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
