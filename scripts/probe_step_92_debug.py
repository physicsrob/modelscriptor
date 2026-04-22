"""Check what debug=True fires at step 92 (the observed drift point).

Runs the full angle=210 sequence up through the second RENDER step
(step 92), with debug=True on each step, so we can see which
self-consistency or Assert failure actually fires at a render
position (vs. the prefill-only checks I ran earlier).
"""

import torch

from torchwright.doom.compile import (
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_PLAYER_ANGLE,
    E8_PLAYER_X,
    E8_PLAYER_Y,
    E8_SORTED_WALL,
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


def _out_to_input(raw_out, module):
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    d_input = sum(w for _, _, w in module._input_specs)
    r = torch.zeros(1, d_input, device=raw_out.device)
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
    module = compile_game(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False
    )

    max_walls = 8
    tex_w = textures[0].shape[0]

    def row(**kwargs):
        return _build_row(module, max_walls, **kwargs)

    # Build prefill (same as before).
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

    # Prefill with HUGE atol so we don't abort, but print any check result.
    print("=== Prefill ===")
    past = module.empty_past()
    try:
        out, past = module.step(prefill, past, past_len=0, debug=True, debug_atol=1e6)
        print(f"  ok ({prefill.shape[0]} positions)")
    except (RuntimeError, AssertionError) as e:
        print(f"  raised: {type(e).__name__}: {str(e)[:200]}")
        return

    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    eos = out[-1:]
    px = float(eos[0, out_by_name["eos_resolved_x"][0]])
    py = float(eos[0, out_by_name["eos_resolved_y"][0]])
    angle = float(eos[0, out_by_name["eos_new_angle"][0]])
    step = prefill.shape[0]

    print("=== Player steps ===")
    for ttype, field, val in [
        (E8_PLAYER_X, "player_x", px),
        (E8_PLAYER_Y, "player_y", py),
        (E8_PLAYER_ANGLE, "player_angle", angle),
    ]:
        prow = row(token_type=ttype, **{field: torch.tensor([val])})
        try:
            out, past = module.step(
                prow, past, past_len=step, debug=True, debug_atol=1e6
            )
            step += 1
            print(f"  player step {step} ok")
        except (RuntimeError, AssertionError) as e:
            print(f"  player step {step+1} raised: {type(e).__name__}: {str(e)[:300]}")
            return

    prev = row(token_type=E8_SORTED_WALL, wall_counter=torch.tensor([0.0]))
    for k in range(4):  # SORT 1, RENDER 1, RENDER 2, RENDER 3
        try:
            out, past = module.step(
                prev, past, past_len=step, debug=True, debug_atol=1e6
            )
            step += 1
            raw = out[0].detach().cpu().numpy()
            wc = raw[out_by_name["wall_counter"][0]]
            print(f"  step {step} ok  wc={wc:.6f}")
        except (RuntimeError, AssertionError) as e:
            print(f"  step {step+1} raised: {type(e).__name__}: {str(e)[:400]}")
            return
        prev = _out_to_input(out, module)


if __name__ == "__main__":
    main()
