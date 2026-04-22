"""Trace the per-layer trajectory of Linear 2870's residual columns.

Runs the DOOM prefill through ``module.step(debug=True)`` with a very
loose ``debug_atol`` (so the self-consistency check doesn't stop
after the first divergence), then walks every captured per-layer
snapshot and prints Linear 2870's value at every layer boundary.

Pinpoints the exact layer where the 0 → -1 transition happens, so we
can match that layer against the scheduler's op log.

Usage:
    make modal-run MODULE=scripts.probe_per_layer_trajectory
"""

import torch

from torchwright.compiler.export import CompiledHeadless
from torchwright.doom.compile import (
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
    _build_row,
    compile_game,
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
    module = compile_game(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False
    )

    max_walls = 8
    tex_w = textures[0].shape[0]

    def row(**kwargs):
        return _build_row(module, max_walls, **kwargs)

    # Build prefill (same as probe_debug_true).
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

    # Run with a huge atol so the check doesn't abort — we just want the
    # state capture.  Asserts will still fire; _debug_state is populated
    # before Assert checks run, so we can inspect state regardless.
    past = module.empty_past()
    try:
        out, past = module.step(prefill, past, past_len=0, debug=True, debug_atol=1e6)
        print(f"Prefill done ({prefill.shape[0]} positions), states captured.")
    except (AssertionError, RuntimeError) as e:
        print(f"(step raised, ok — state was captured before the raise)")
        print(f"  {type(e).__name__}: {e!s:.120s}...")

    # Now walk the captured state and for each (layer, sublayer) extract
    # residual[:, cols_of_Linear_2870].
    ds = module._debug_state
    assert ds is not None

    # Linear 2870's residual cols (from the earlier audit / from the
    # self-consistency failure message).
    TARGET_COLS = [
        41,
        66,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        130,
        133,
        135,
        140,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        174,
        175,
        194,
        198,
        202,
        206,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
    ]

    # Build an ordered list of (label, tensor) pairs, sorted by layer
    # index then attn/mlp sublayer order.
    def sort_key(item):
        label = item[0]
        # Labels look like "layer_N_attn_skip_out_state" or "layer_N_mlp_out_state".
        parts = label.split("_")
        try:
            layer_idx = int(parts[1])
        except (IndexError, ValueError):
            return (10**9, 9, label)
        sublayer = 0 if "attn" in label else 1
        return (layer_idx, sublayer, label)

    rows = []
    for state, (tensor, label) in ds.state_tensor.items():
        rows.append((label, tensor))
    rows.sort(key=sort_key)

    print("\n=== Trajectory of Linear 2870's residual columns across layers ===")
    print(f"TARGET_COLS: {TARGET_COLS[:6]}... ({len(TARGET_COLS)} cols total)")
    print(
        f"{'label':40s}  {'col[25]=160':>11s}  {'col[0]=41':>11s}  "
        f"{'col[32]=167':>11s}  {'summary':>40s}"
    )
    prev_sig = None
    for label, tensor in rows:
        # residual shape: (n_pos, d).  Gather at TARGET_COLS -> (n_pos, 60).
        gathered = tensor[:, TARGET_COLS]
        # Summary: unique values, approximate to 4 decimals
        rounded = torch.round(gathered * 10000) / 10000
        flat = rounded.flatten().tolist()
        unique_vals = sorted(set(flat))
        summary = (
            f"uniq={len(unique_vals)}" f" min={min(flat):+.4f}" f" max={max(flat):+.4f}"
        )
        v_160 = float(gathered[0, 25].item())  # col index 25 = residual[160]
        v_41 = float(gathered[0, 0].item())  # col index 0 = residual[41]
        v_167 = float(gathered[0, 32].item())  # col index 32 = residual[167]
        sig = f"{v_160:.4f}/{v_41:.4f}/{v_167:.4f}"
        marker = ""
        if prev_sig is not None and sig != prev_sig:
            marker = "  <-- CHANGED"
        prev_sig = sig
        print(
            f"{label:40s}  {v_160:+11.4f}  {v_41:+11.4f}  {v_167:+11.4f}  {summary:>40s}{marker}"
        )


if __name__ == "__main__":
    main()
