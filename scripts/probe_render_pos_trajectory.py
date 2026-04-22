"""Per-layer trajectory of residual[160] during STEP 91's forward pass.

Hypothesis #2 from docs/postmortems/angle_210_overlay_reserve.md:
an op scheduled in layers 40-53 writes to col 160 at RENDER
positions (gated on ``is_render=+1``) but not at prefill.  If true,
col 160 at a RENDER position would transition to a non-zero value
*before* layer 54 — different from the prefill trajectory where col
160 stays at 0 until the delta layer.

Strategy: prefill without debug=True (avoids the unrelated hardness
Assert firing during prefill), then run the first SORT and first
RENDER one step at a time with ``debug=True, debug_atol=1e6``.  The
Assert on ``render/wall_vis_attention`` will still fire during the
debug run — but ``_debug_state`` is populated before the Assert
phase, so we can catch the exception and read out the per-layer
state ourselves.

Usage:
    make modal-run MODULE=scripts.probe_render_pos_trajectory
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


def _dump_col_trajectory(module, label_prefix, target_col):
    """After a step(..., debug=True), walk every captured state and
    print residual[target_col] at every layer boundary."""
    ds = module._debug_state
    assert ds is not None

    def sort_key(item):
        lbl = item[0]
        parts = lbl.split("_")
        try:
            layer_idx = int(parts[1])
        except (IndexError, ValueError):
            return (10**9, 9, lbl)
        sublayer = 0 if "attn" in lbl else 1
        return (layer_idx, sublayer, lbl)

    rows = [(label, tensor) for state, (tensor, label) in ds.state_tensor.items()]
    rows.sort(key=sort_key)

    print(f"\n--- {label_prefix}: residual[{target_col}] per layer ---")
    prev = None
    for lbl, tensor in rows:
        v = float(tensor[0, target_col].item())
        marker = ""
        if prev is not None and abs(v - prev) > 1e-6:
            marker = f"  <-- CHANGED ({v - prev:+.4f})"
        prev = v
        print(f"  {lbl:40s}  residual[{target_col}]={v:+.6f}{marker}")


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

    # Prefill WITHOUT debug=True so the hardness Assert doesn't stop us.
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

    past = module.empty_past()
    out, past = module.step(prefill, past, past_len=0)  # no debug
    step = prefill.shape[0]

    eos = out[-1:]
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    px = float(eos[0, out_by_name["eos_resolved_x"][0]])
    py = float(eos[0, out_by_name["eos_resolved_y"][0]])
    angle = float(eos[0, out_by_name["eos_new_angle"][0]])

    for ttype, field, val in [
        (E8_PLAYER_X, "player_x", px),
        (E8_PLAYER_Y, "player_y", py),
        (E8_PLAYER_ANGLE, "player_angle", angle),
    ]:
        prow = row(token_type=ttype, **{field: torch.tensor([val])})
        out, past = module.step(prow, past, past_len=step)
        step += 1

    # First SORT.  No debug (output drives next input).
    prev = row(token_type=E8_SORTED_WALL, wall_counter=torch.tensor([0.0]))
    out, past = module.step(prev, past, past_len=step)
    step += 1
    prev = _out_to_input(out, module)

    # First RENDER with debug=True + huge atol (so self-consistency doesn't
    # abort).  Assert will still fire; catch it.
    print(f"\n=== step {step+1} (first RENDER) with debug=True ===")
    try:
        out2, past2 = module.step(prev, past, past_len=step, debug=True, debug_atol=1e6)
        print("  step completed without raising")
    except (AssertionError, RuntimeError) as e:
        print(f"  step raised ({type(e).__name__}): ok, state captured beforehand")
        print(f"  first 180 chars: {str(e)[:180]}")

    # Dump col 160's trajectory at the render-position forward.
    _dump_col_trajectory(module, f"RENDER step {step+1}", target_col=160)

    # Also dump a non-target col for comparison — col 53 is Linear 2900's
    # residual home (source for the advance_wall delta).
    _dump_col_trajectory(module, f"RENDER step {step+1}", target_col=53)


if __name__ == "__main__":
    main()
