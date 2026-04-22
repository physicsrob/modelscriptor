"""Oracle-vs-compiled divergence probe on angle=210 prefill.

Step 3 of the session-4 plan.  Runs ``probe_compiled(..., atol=500)``
on the angle=210 box-room prefill (all 86 positions — TEX_COL +
INPUT + BSP_NODE + WALL + EOS, with ``player_angle=210``).  The
plan's atol=500 matches the empirical floor for the DOOM renderer
documented in ``tests/debug/test_probe.py::test_probe_clean_on_v2_box_room``
and in CLAUDE.md's *Debugging compiled graphs* section.

Expected outcome: ``report.first_divergent is None`` on the fix-applied
compile.  If non-None, the offending node is the lead for investigating
the 3 pre-existing full-suite ``[210]`` regressions (Problem 2).

Usage:
    make modal-run MODULE=scripts.probe_oracle_prefill_210
"""

from __future__ import annotations

import torch

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
from torchwright.graph import Concatenate
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

TRIG = generate_trig_table()

_ATOL = 500.0
_ANGLE = 210
_PX = 0.0
_PY = 0.0


def _config():
    return RenderConfig(
        screen_width=64,
        screen_height=80,
        fov_columns=32,
        trig_table=TRIG,
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
            subset.seg_bsp_coeffs[i, :max_bsp_nodes], dtype=torch.float32
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


def _input_values(module, prefill):
    in_by_name = {name: (s, w) for name, s, w in module._input_specs}
    return {name: prefill[:, s : s + w].clone() for name, (s, w) in in_by_name.items()}


def main():
    config = _config()
    textures = default_texture_atlas()
    subset = build_scene_subset(_segments(), textures)
    print(f"Compiling game graph for angle={_ANGLE}, px={_PX}, py={_PY}...")

    graph_io, pos_encoding = build_game_graph(
        config,
        textures,
        max_walls=8,
        max_coord=20.0,
        move_speed=0.3,
        turn_speed=4,
        chunk_size=20,
        max_bsp_nodes=48,
    )
    output_nodes = set(graph_io.overlaid_outputs.values())
    output_nodes.update(graph_io.overflow_outputs.values())
    output_nodes.add(pos_encoding)
    while fuse_consecutive_linears(output_nodes, verbose=False) > 0:
        pass

    io = {}
    for name, node in graph_io.inputs.items():
        io[name] = (node, graph_io.overlaid_outputs.get(name))
    for name, node in graph_io.overflow_outputs.items():
        io[name] = (None, node)

    module = compile_headless(
        pos_encoding,
        io=io,
        d=2048,
        d_head=32,
        max_layers=400,
        verbose=False,
        extra_metadata={
            "chunk_size": 20,
            "max_walls": 8,
            "max_bsp_nodes": 48,
            "tex_h": textures[0].shape[1],
        },
    )
    module.eval()
    print(f"Compiled: {len(module._net.layers)} layers")

    probe_root = Concatenate(
        list(graph_io.overlaid_outputs.values())
        + list(graph_io.overflow_outputs.values())
    )

    prefill = _build_prefill(module, subset, px=_PX, py=_PY, angle=_ANGLE)
    inputs = _input_values(module, prefill)
    n_pos = prefill.shape[0]
    print(f"Prefill rows: {n_pos}; running probe_compiled at atol={_ATOL}...")

    report = probe_compiled(module, probe_root, inputs, n_pos=n_pos, atol=_ATOL)

    print("\n=== Report ===")
    print(report.format_short(show_top_k=10))

    if report.first_divergent is None:
        print(f"\nCLEAN: no node exceeds atol={_ATOL} on prefill")
    else:
        fd = report.first_divergent
        print(
            f"\nFIRST DIVERGENT: node_id={fd.node.node_id} "
            f"type={type(fd.node).__name__} name={fd.node.name!r} "
            f"ann={fd.node.annotation!r}"
        )
        print(f"  max_abs_error={fd.max_abs_error:.4f}")

    # Top 5 per_node errors regardless of divergence flag.
    ranked = sorted(
        report.per_node.values(), key=lambda r: r.max_abs_error, reverse=True
    )[:5]
    print("\nTop 5 per-node max_abs_error:")
    for rec in ranked:
        n = rec.node
        print(
            f"  {rec.max_abs_error:>12.4f}  id={n.node_id} "
            f"{type(n).__name__} name={n.name!r} ann={n.annotation!r}"
        )


if __name__ == "__main__":
    main()
