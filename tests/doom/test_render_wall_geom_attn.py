"""Regression test: wall geometry attention in the compiled RENDER stage.

The wall geometry attend_argmax_dot reads (ax, ay, bx, by) from WALL
positions in the KV cache.  At tex_size=16 (larger graph) the compiled
Attn output is all zeros despite correct V cache and perfect softmax
concentration.
"""

import numpy as np
import pytest
import torch

from torchwright.doom.compile import compile_game, _build_row
from torchwright.doom.map_subset import build_scene_subset
from torchwright.doom.game_graph import (
    E8_SORTED_WALL,
    E8_INPUT,
    E8_EOS,
    E8_WALL,
    E8_BSP_NODE,
    E8_PLAYER_X,
    E8_PLAYER_Y,
    E8_PLAYER_ANGLE,
    E8_RENDER,
)
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.doom.graph_constants import TEX_E8_OFFSET


def _solid_textures(tex_size, n=4):
    colors = [(0.8, 0.2, 0.1), (0.1, 0.8, 0.2), (0.2, 0.1, 0.8), (0.5, 0.5, 0.1)]
    textures = []
    for i in range(n):
        r, g, b = colors[i % len(colors)]
        t = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
        t[:, :, 0] = r
        t[:, :, 1] = g
        t[:, :, 2] = b
        textures.append(t)
    return textures


def _compile_and_check_wall_geom(tex_size):
    """Compile the game graph, run prefill+sort+one render step, return
    the wall geometry attention's compiled output via DebugWatch."""
    config = RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )
    textures = _solid_textures(tex_size)
    segments = [
        Segment(ax=5, ay=-5, bx=5, by=5, color=(0.8, 0.2, 0.1), texture_id=0),
        Segment(ax=-5, ay=-5, bx=-5, by=5, color=(0.1, 0.8, 0.2), texture_id=1),
        Segment(ax=-5, ay=5, bx=5, by=5, color=(0.2, 0.1, 0.8), texture_id=2),
        Segment(ax=-5, ay=-5, bx=5, by=-5, color=(0.5, 0.5, 0.1), texture_id=3),
    ]
    subset = build_scene_subset(segments, textures)
    N = len(segments)
    max_walls = 8
    num_tex = len(textures)
    tex_w = textures[0].shape[0]

    module = compile_game(
        config,
        textures,
        max_walls=max_walls,
        max_coord=10.0,
        d=2048,
        chunk_size=4,
        verbose=False,
        optimize=True,
    )
    max_bsp_nodes = int(module.metadata.get("max_bsp_nodes", 48))

    past = module.empty_past()
    step = 0
    px, py, angle = 0.0, 0.0, 0.0
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}

    def _common(**extra):
        return _build_row(
            module,
            max_walls,
            player_x=torch.tensor([px]),
            player_y=torch.tensor([py]),
            player_angle=torch.tensor([angle]),
            **extra,
        )

    # Prefill: TEX_COL + INPUT + BSP + WALL + EOS
    rows = []
    for tex_idx in range(num_tex):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = textures[tex_idx][col].flatten()
            rows.append(
                _common(
                    token_type=tex_e8,
                    texture_id_e8=tex_e8,
                    tex_col_input=torch.tensor([float(col)]),
                    tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                )
            )
    rows.append(_common(token_type=E8_INPUT))
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        kw = dict(token_type=E8_BSP_NODE, bsp_node_id_onehot=onehot)
        if i < len(subset.bsp_nodes):
            p = subset.bsp_nodes[i]
            kw.update(
                bsp_plane_nx=torch.tensor([p.nx]),
                bsp_plane_ny=torch.tensor([p.ny]),
                bsp_plane_d=torch.tensor([p.d]),
            )
        rows.append(_common(**kw))
    for i, s in enumerate(segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes], dtype=torch.float32
        )
        const = torch.tensor([float(subset.seg_bsp_consts[i])], dtype=torch.float32)
        rows.append(
            _common(
                token_type=E8_WALL,
                wall_ax=torch.tensor([s.ax]),
                wall_ay=torch.tensor([s.ay]),
                wall_bx=torch.tensor([s.bx]),
                wall_by=torch.tensor([s.by]),
                wall_tex_id=torch.tensor([float(s.texture_id)]),
                wall_index=torch.tensor([float(i)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
            )
        )
    rows.append(_common(token_type=E8_EOS))
    prefill = torch.cat(rows, dim=0)
    with torch.no_grad():
        out, past = module.step(prefill, past, past_len=0)
    step = prefill.shape[0]

    # Player tokens
    for tt, kw in [
        (E8_PLAYER_X, {"player_x": torch.tensor([px])}),
        (E8_PLAYER_Y, {"player_y": torch.tensor([py])}),
        (E8_PLAYER_ANGLE, {"player_angle": torch.tensor([angle])}),
    ]:
        row = _build_row(module, max_walls, token_type=tt, **kw)
        with torch.no_grad():
            out, past = module.step(row, past, past_len=step)
        step += 1

    # Sort loop
    wj_s, wj_w = out_by_name["render_wall_j_onehot"]
    vlo_s, _ = out_by_name["render_vis_lo"]
    vhi_s, _ = out_by_name["render_vis_hi"]
    tid_s, _ = out_by_name["render_tex_id"]
    sd_s, _ = out_by_name["sort_done"]

    for k in range(N):
        sort_row = _build_row(
            module,
            max_walls,
            token_type=E8_SORTED_WALL,
            sort_position_index=torch.tensor([float(k)]),
        )
        with torch.no_grad():
            out, past = module.step(sort_row, past, past_len=step)
        step += 1

    # Use the LAST sorted wall's outputs (matches the reproducer)
    wall_oh = out[0, wj_s : wj_s + wj_w].clone()
    vlo = out[0, vlo_s].item()
    vhi = out[0, vhi_s].item()
    tid = out[0, tid_s].item()

    render_row = _build_row(
        module,
        max_walls,
        token_type=E8_RENDER,
        render_wall_j_onehot=wall_oh,
        render_vis_lo=torch.tensor([vlo]),
        render_vis_hi=torch.tensor([vhi]),
        render_tex_id=torch.tensor([tid]),
        render_col=torch.tensor([vlo]),
        render_chunk_k=torch.tensor([0.0]),
    )
    with torch.no_grad():
        rout, past = module.step(render_row, past, past_len=step)

    # Read wall height from outputs
    length_s, _ = out_by_name["length"]
    start_s, _ = out_by_name["start"]
    return rout[0, length_s].item(), rout[0, start_s].item()


@pytest.mark.parametrize("tex_size", [8, 16])
def test_wall_geom_attention_nonzero_height(tex_size):
    length, start = _compile_and_check_wall_geom(tex_size)
    assert length > 1.0, (
        f"tex_size={tex_size}: wall height is ~zero (length={length:.6f}, "
        f"start={start:.4f}). Wall geometry attention likely produced zeros."
    )
