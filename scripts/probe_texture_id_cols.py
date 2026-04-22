"""Verify whether `texture_id_e8` InputNode's residual cols are aliased.

The texture_id_e8 InputNode's compiled value (via ``compiled.debug_value``)
shows min=-30, max=10 across (86, 8) at the first captured state — but its
declared value_range is [-1, 1] and the input tensor only contains E8
spherical codes (bounded [-1, 1]) at TEX_COL rows and 0 elsewhere.

This probe:
1. Finds the InputNode's assigned residual cols.
2. Reads the actual prefill tensor values at those cols.
3. Reads the residual-stream values at those cols at EVERY captured state,
   so we can see whether they change layer-to-layer.
4. Reports which (state, col) pairs show values outside [-1.1, 1.1].

Usage:
    make modal-run MODULE=scripts.probe_texture_id_cols
"""

import torch

from torchwright.doom.compile import (
    compile_game,
    _build_row,
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_SORTED_WALL,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
)
from torchwright.doom.map_subset import build_scene_subset
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.reference_renderer.render import Segment
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig

TRIG = generate_trig_table()


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


def main():
    config = _config()
    textures = default_texture_atlas()
    segs = _segments()
    subset = build_scene_subset(segs, textures)
    print("Compiling...")
    module = compile_game(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False
    )
    print(f"Compiled: {len(module._net.layers)} layers")

    max_walls = 8
    tex_w = textures[0].shape[0]

    def row(**kwargs):
        return _build_row(module, max_walls, **kwargs)

    rows = []
    for tex_idx in range(len(textures)):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for c in range(tex_w):
            pixel_data = textures[tex_idx][c].flatten()
            rows.append(
                row(
                    token_type=E8_TEX_COL,
                    texture_id_e8=tex_e8,
                    tex_col_input=torch.tensor([float(c)]),
                    tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                )
            )
    rows.append(row(token_type=E8_INPUT))
    max_bsp_nodes = int(module.metadata["max_bsp_nodes"])
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            p = subset.bsp_nodes[i]
            nx, ny, d_ = p.nx, p.ny, p.d
        else:
            nx, ny, d_ = 0.0, 0.0, 0.0
        rows.append(
            row(
                token_type=E8_BSP_NODE,
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d_], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
            )
        )
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes], dtype=torch.float32
        )
        const = torch.tensor([float(subset.seg_bsp_consts[i])], dtype=torch.float32)
        rows.append(
            row(
                token_type=E8_WALL,
                wall_ax=torch.tensor([float(seg.ax)]),
                wall_ay=torch.tensor([float(seg.ay)]),
                wall_bx=torch.tensor([float(seg.bx)]),
                wall_by=torch.tensor([float(seg.by)]),
                wall_tex_id=torch.tensor([float(seg.texture_id)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
            )
        )
    rows.append(row(token_type=E8_EOS))
    prefill = torch.cat(rows, dim=0)
    print(f"Prefill rows: {prefill.shape[0]}")

    # Run debug=True (ignore the assert)
    past = module.empty_past()
    try:
        module.step(prefill, past, past_len=0, debug=True)
    except AssertionError:
        pass

    ds = module._debug_state
    ra = ds.ra

    # Find the InputNode by annotation/name.
    from torchwright.graph.node import Node

    tex_id_node = None
    for a in module._asserts:
        # Walk upstream from each tex_attention assert to find texture_id_e8 InputNode
        stack = [a]
        seen = set()
        while stack:
            n = stack.pop()
            if n.node_id in seen:
                continue
            seen.add(n.node_id)
            if (n.name or "") == "texture_id_e8":
                tex_id_node = n
                break
            for inp in n.inputs:
                stack.append(inp)
        if tex_id_node:
            break

    if tex_id_node is None:
        # Fallback: scan all nodes
        print("Couldn't find via asserts; scanning full graph.")
        stack = [module._asserts[0]]
        seen = set()
        while stack:
            n = stack.pop()
            if n.node_id in seen:
                continue
            seen.add(n.node_id)
            if (n.name or "") == "texture_id_e8":
                tex_id_node = n
                break
            for inp in n.inputs:
                stack.append(inp)

    if tex_id_node is None:
        print("texture_id_e8 node not found; bailing")
        return

    print(
        f"\ntexture_id_e8 node: id={tex_id_node.node_id} type={type(tex_id_node).__name__}"
    )
    print(f"  declared value_type: {tex_id_node.value_type}")
    print(f"  width: {tex_id_node.d_output}")

    # Find input spec offset for texture_id_e8 so we can read actual input vals.
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    tex_in_s, tex_in_w = in_by_name.get("texture_id_e8", (None, None))
    tex_out = out_by_name.get("texture_id_e8", None)
    print(f"  input spec: offset={tex_in_s} width={tex_in_w}")
    print(f"  output spec: {tex_out}")

    if tex_in_s is not None:
        actual = prefill[:, tex_in_s : tex_in_s + tex_in_w]
        print(
            f"  actual prefill values: shape={tuple(actual.shape)} min={float(actual.min()):.4f} max={float(actual.max()):.4f}"
        )
        for pi in (0, 16, 32, 48, 63, 64, 80):
            if pi < actual.shape[0]:
                row_vals = actual[pi].tolist()
                print(f"    prefill[{pi}]={row_vals}")

    # Now walk every captured state and read residual at the InputNode's cols.
    print("\nResidual values at texture_id_e8 InputNode cols per captured state:")
    any_col_found = False
    for state in ds.ordered_states:
        if not ra.has_node(state, tex_id_node):
            continue
        any_col_found = True
        label = ds.state_tensor[state][1]
        tensor = ds.state_tensor[state][0]
        cols = list(ra.get_node_indices(state, tex_id_node))
        val = tensor[:, cols]
        vmin = float(val.min())
        vmax = float(val.max())
        # Compare with actual.
        if tex_in_s is not None:
            diff = (val - actual).abs().max().item()
        else:
            diff = None
        print(
            f"  {label}: cols={cols}  min={vmin:.4f} max={vmax:.4f}"
            + (f"  diff_vs_actual={diff:.4f}" if diff is not None else "")
        )
    if not any_col_found:
        print("  (no captured state has this node allocated)")

    # Also enumerate all nodes whose cols overlap texture_id_e8's cols (over all states).
    if ra.has_node(ds.ordered_states[0], tex_id_node):
        cols0 = set(ra.get_node_indices(ds.ordered_states[0], tex_id_node))
    else:
        cols0 = set()
    print(
        f"\nSearching for other nodes whose cols overlap texture_id_e8's cols ({sorted(cols0)[:10]}...):"
    )
    overlaps = {}
    for state in ds.ordered_states:
        for n in ra.get_nodes(state):
            if n.node_id == tex_id_node.node_id:
                continue
            try:
                ncols = set(ra.get_node_indices(state, n))
            except KeyError:
                continue
            inter = cols0 & ncols
            if inter:
                overlaps.setdefault(n.node_id, (n, set()))[1].update(inter)
    print(f"  Nodes with overlap: {len(overlaps)}")
    for nid, (n, cols_shared) in overlaps.items():
        nm = n.annotation or n.name or f"node_{nid}"
        print(
            f"    id={nid} type={type(n).__name__} name='{n.name}' ann='{n.annotation}' cols={sorted(cols_shared)}"
        )


if __name__ == "__main__":
    main()
