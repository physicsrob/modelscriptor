"""Run angle=210 compile with ``debug=True`` at every position to surface
the first failure (Assert fire or self-consistency violation).

Compiles DOOM with the overlay-reserve fix reverted, then re-runs the
prefill+player+SORT sequence using ``module.step(..., debug=True)`` so
the per-layer self-consistency check and Assert nodes fire as soon as
the compile diverges from the graph's stated invariants.

Usage:
    make modal-run MODULE=scripts.probe_debug_true
"""

import torch

from torchwright.doom.compile import (
    compile_game,
    _build_row,
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_PLAYER_ANGLE,
    E8_PLAYER_X,
    E8_PLAYER_Y,
    E8_RENDER,
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


def _out_to_input(raw_out, module):
    """Map overlaid output fields back into a d_input row for the next step."""
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    d_input = sum(w for _, _, w in module._input_specs)
    device = raw_out.device
    r = torch.zeros(1, d_input, device=device)
    for name, (in_s, in_w) in in_by_name.items():
        if name in out_by_name:
            os_, ow = out_by_name[name]
            r[0, in_s : in_s + in_w] = raw_out[0, os_ : os_ + ow]
    return r


def main():
    config = _config()
    textures = default_texture_atlas()
    segs = _segments()
    subset = build_scene_subset(segs, textures)
    print("Compiling...")
    module = compile_game(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False
    )
    print(
        f"Compiled: {len(module._net.layers)} layers, "
        f"{len(module._asserts)} asserts, {len(module._watches)} watches"
    )

    max_walls = 8
    tex_w = textures[0].shape[0]

    def row(**kwargs):
        return _build_row(module, max_walls, **kwargs)

    # Prefill
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

    # Run the whole prefill with debug=True in one shot.
    print("\n=== Prefill with debug=True ===")
    try:
        past = module.empty_past()
        out, past = module.step(prefill, past, past_len=0, debug=True)
        step = prefill.shape[0]
        print(f"Prefill debug=True passed ({step} positions)")
    except AssertionError as e:
        print(f"!!! Prefill debug=True raised: {e}")
        return

    # EOS-resolved state.
    eos = out[-1:]
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    px = float(eos[0, out_by_name["eos_resolved_x"][0]])
    py = float(eos[0, out_by_name["eos_resolved_y"][0]])
    angle = float(eos[0, out_by_name["eos_new_angle"][0]])
    print(f"EOS: px={px:.3f} py={py:.3f} angle={angle:.3f}")

    # Player rows.
    for ttype, field, val in [
        (E8_PLAYER_X, "player_x", px),
        (E8_PLAYER_Y, "player_y", py),
        (E8_PLAYER_ANGLE, "player_angle", angle),
    ]:
        try:
            prow = row(token_type=ttype, **{field: torch.tensor([val])})
            out, past = module.step(prow, past, past_len=step, debug=True)
            step += 1
            print(f"  Player step {step} debug=True passed")
        except AssertionError as e:
            print(f"!!! Player step {step+1} debug=True raised: {e}")
            return

    # SORT + RENDER — one step at a time with debug=True.
    prev = row(token_type=E8_SORTED_WALL, wall_counter=torch.tensor([0.0]))
    for k in range(5):
        try:
            out, past = module.step(prev, past, past_len=step, debug=True)
            step += 1
            raw = out[0].detach().cpu().numpy()
            wc = raw[out_by_name["wall_counter"][0]]
            wi = raw[out_by_name["render_wall_index"][0]]
            tt_s = out_by_name["token_type"][0]
            tt = raw[tt_s : tt_s + 8]
            print(
                f"  step {step} debug=True PASSED  wc={wc:.4f} wi={wi:.4f} "
                f"tt_sum={float(tt.sum()):.3f} tt_argmax={int(tt.argmax())}"
            )
        except AssertionError as e:
            print(f"\n!!! step {step+1} debug=True RAISED:")
            print(f"    {e}")
            import traceback

            traceback.print_exc()
            return
        prev = _out_to_input(out, module)


if __name__ == "__main__":
    main()
