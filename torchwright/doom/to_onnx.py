"""Compile the DOOM v2 walls-as-tokens game graph and export to ONNX.

The v2 graph receives wall geometry at runtime via wall tokens, so the
ONNX model is scene-independent.  Collision detection (optional) is
still baked from segments when ``--collision`` is passed.

Input schema (per position):
    token_type (8)   — E8 spherical code identifying the token role
    player_{x,y,angle}, input_* — player state and controls
    wall_{ax,ay,bx,by,tex_id}, wall_index — wall geometry (WALL tokens)
    wall_counter (1) — sort position index for SORTED_WALL / RENDER tokens
    render_col, render_chunk_k, render_wall_index — RENDER state

Output schema: token-type-dependent (see game_graph module docstring).

Usage:
    python -m torchwright.doom.to_onnx [--scene box|multi]
"""

import argparse

from torchwright.compiler.export import compile_headless_to_onnx
from torchwright.doom.game_graph import build_game_graph
from torchwright.reference_renderer.scenes import (
    box_room_textured,
    multi_room_textured,
)
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


def main():
    parser = argparse.ArgumentParser(
        description="Compile the DOOM v2 game graph and save it to ONNX",
    )
    parser.add_argument("--scene", choices=["box", "multi"], default="box")
    parser.add_argument(
        "--wad",
        type=str,
        default="doom1.wad",
        help="Path to doom1.wad for DOOM textures",
    )
    parser.add_argument(
        "--tex-size",
        type=int,
        default=64,
        help="Texture resolution (downscaled from WAD)",
    )
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--fov", type=int, default=32)
    parser.add_argument(
        "--max-walls",
        type=int,
        default=8,
        help="Maximum number of wall tokens per frame",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Render chunk height (pixels per render token).",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=2048,
        help="Residual stream width.",
    )
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output .onnx path. Default: doom_game_v2_<scene>.onnx",
    )
    args = parser.parse_args()

    trig_table = generate_trig_table()
    config = RenderConfig(
        screen_width=args.width,
        screen_height=args.height,
        fov_columns=args.fov,
        trig_table=trig_table,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )

    output_path = args.output or f"doom_game_{args.scene}.onnx"

    if args.scene == "box":
        segments, textures = box_room_textured(
            wad_path=args.wad,
            tex_size=args.tex_size,
        )
        max_coord = 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path=args.wad,
            tex_size=args.tex_size,
        )
        max_coord = 15.0

    print(f"Building game graph (d={args.d}, max_walls={args.max_walls})...")
    graph_io, pos_encoding = build_game_graph(
        config,
        textures,
        max_walls=args.max_walls,
        max_coord=max_coord,
        move_speed=0.3,
        turn_speed=4,
        chunk_size=args.chunk_size,
    )
    output_node = graph_io.concat_output()

    # Sequence length: TEX_COL + INPUT + WALL*N + EOS + SORTED_WALL*N + RENDER (dynamic)
    cs = args.chunk_size
    n_walls = args.max_walls
    num_tex = len(textures)
    tex_w = textures[0].shape[0]
    n_tex_col = num_tex * tex_w
    # Upper bound on render tokens: each wall covers W columns, each column ceil(H/cs) chunks
    max_render = n_walls * config.screen_width * ((config.screen_height + cs - 1) // cs)
    max_seq_len = n_tex_col + 1 + n_walls + 1 + n_walls + max_render

    compile_headless_to_onnx(
        output_node=output_node,
        pos_encoding=pos_encoding,
        output_path=output_path,
        d=args.d,
        d_head=args.d_head,
        max_seq_len=max_seq_len,
        max_layers=400,
        verbose=True,
        extra_metadata={
            "chunk_size": cs,
            "max_walls": n_walls,
        },
    )


if __name__ == "__main__":
    main()
