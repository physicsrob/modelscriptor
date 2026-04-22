"""Inspect layer 54's attention matrices for any head that would write
a non-zero constant into residual[160].

Approach: for each attention head in layer 54, examine the output
projection matrix O.  Residual[160] receives
    sum_h sum_d O[h, d, 160] * V_head[h, d]
per position.  For a head to write a position-independent -1 into
col 160, it needs:
  - a nonzero O[h, d, 160] at some d,
  - such that the corresponding V_head[h, d] is a position-independent
    constant (likely produced by attending to a literal value, or by
    the head's V-matrix reading from a "constant" column like
    pos_encoding bias).

We just print any head whose O column 160 has a nonzero entry and
the V/Q/K pattern that backs it.

Usage:
    make modal-run MODULE=scripts.probe_layer_54
"""

import torch

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
    net = module._net

    print(
        f"Net: {len(net.layers)} layers, d={net.layers[0].attn.attn.d}, "
        f"d_head={net.layers[0].attn.attn.d_head}"
    )

    # Inspect layer 54's attention.
    TARGET_LAYER = 54
    TARGET_COL = 160
    layer = net.layers[TARGET_LAYER]
    attn = layer.attn.attn
    print(f"\nLayer {TARGET_LAYER}: n_heads={attn.n_heads}, d_head={attn.d_head}")
    print(
        f"  Q shape {attn.query_matrix.shape}, "
        f"K shape {attn.key_matrix.shape}, "
        f"V shape {attn.value_matrix.shape}, "
        f"O shape {attn.output_matrix.shape}"
    )

    # Matrices are per-head: Q/K/V shape (n_heads, d, d_head), O shape
    # (n_heads, d_head, d).  Residual col c receives:
    #   residual[c] += sum_h sum_{d_h} O[h, d_h, c] * head_out[h, d_h]
    # So O[:, :, TARGET_COL] tells us which head slots contribute to c=160.
    o_col = attn.output_matrix[:, :, TARGET_COL]  # (n_heads, d_head)
    nonzero = (o_col != 0).nonzero(as_tuple=False)  # list of (head, slot)
    print(
        f"\n  O[..., :, col={TARGET_COL}] nonzero entries: "
        f"{nonzero.shape[0]} (out of {o_col.numel()})"
    )
    if nonzero.numel() == 0:
        print(
            "    (col 160 receives NO attention output at layer 54 — bug is elsewhere)"
        )
        return

    for row in nonzero.tolist():
        h, d_h = row
        o_coef = o_col[h, d_h].item()
        # Per-head Q/K/V slices: shape (d, d_head).  We want the column
        # d_head_idx=d_h (what this head produced at head dim d_h).
        q_col = attn.query_matrix[h, :, d_h]
        k_col = attn.key_matrix[h, :, d_h]
        v_col = attn.value_matrix[h, :, d_h]
        q_nz = (q_col != 0).nonzero(as_tuple=True)[0].tolist()
        k_nz = (k_col != 0).nonzero(as_tuple=True)[0].tolist()
        v_nz = (v_col != 0).nonzero(as_tuple=True)[0].tolist()
        print(f"\n    head {h}, slot {d_h}: O coef = {o_coef:+.6f}")
        print(f"      Q nonzero input cols: {q_nz}")
        print(f"      Q values: " f"{[q_col[c].item() for c in q_nz]}")
        print(f"      K nonzero input cols: {k_nz}")
        print(f"      K values: " f"{[k_col[c].item() for c in k_nz]}")
        print(f"      V nonzero input cols: {v_nz}")
        print(f"      V values: " f"{[v_col[c].item() for c in v_nz]}")


if __name__ == "__main__":
    main()
