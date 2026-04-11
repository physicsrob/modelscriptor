"""Compile the DOOM renderer or game graph and export it to ONNX.

Usage:
    python -m torchwright.doom.to_onnx [--mode renderer|game] [--scene box|multi]
"""

import argparse

from torchwright.compiler.export import compile_headless_to_onnx
from torchwright.doom.renderer import build_renderer_graph
from torchwright.reference_renderer.scenes import (
    box_room,
    box_room_textured,
    multi_room,
    multi_room_textured,
)
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


def main():
    parser = argparse.ArgumentParser(
        description="Compile the DOOM renderer/game and save it to ONNX",
    )
    parser.add_argument(
        "--mode", choices=["renderer", "game"], default="renderer",
        help="renderer: flat-shaded wall renderer only. "
             "game: full game logic + textured rendering.",
    )
    parser.add_argument("--scene", choices=["box", "multi"], default="box")
    parser.add_argument("--wad", type=str, default="doom1.wad",
                        help="Path to doom1.wad for DOOM textures")
    parser.add_argument("--tex-size", type=int, default=8,
                        help="Texture resolution (downscaled from WAD)")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--fov", type=int, default=32)
    parser.add_argument(
        "--rows-per-patch", type=int, default=8,
        help="Vertical patch height (game mode only). Must divide --height. "
             "Default: 8 (the analysis-picked shipping target for full DOOM). "
             "Pass --rows-per-patch <height> for the unsharded phase-α path.",
    )
    parser.add_argument(
        "--d", type=int, default=None,
        help="Residual stream width. Defaults: 512 (renderer), 2048 (game). "
             "Game-mode default is the measured dev sweet spot at rp=8.",
    )
    parser.add_argument(
        "--d-hidden", type=int, default=None,
        help="Per-layer MLP hidden width. Defaults: d (renderer), 4096 (game). "
             "Decoupling d_hidden from d_model cuts layer count and KV "
             "traffic at rp=8; 4096 is the measured optimum at d=2048.",
    )
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output .onnx path. Default: doom_<mode>_<scene>.onnx",
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

    output_path = args.output or f"doom_{args.mode}_{args.scene}.onnx"

    if args.mode == "renderer":
        if args.scene == "box":
            segments = box_room()
            max_coord = 10.0
        else:
            segments = multi_room()
            max_coord = 15.0
        d = args.d if args.d is not None else 512
        d_hidden = args.d_hidden  # None → falls back to d inside forward_compile
        print(f"Building renderer graph (d={d})...")
        output_node, pos_encoding = build_renderer_graph(
            segments, config, max_coord,
        )
        max_layers = 100
    else:
        from torchwright.doom.game_graph import build_game_graph

        if args.scene == "box":
            segments, textures = box_room_textured(
                wad_path=args.wad, tex_size=args.tex_size,
            )
            max_coord = 10.0
        else:
            segments, textures = multi_room_textured(
                wad_path=args.wad, tex_size=args.tex_size,
            )
            max_coord = 15.0
        d = args.d if args.d is not None else 2048
        d_hidden = args.d_hidden if args.d_hidden is not None else 4096
        print(f"Building game graph (d={d}, d_hidden={d_hidden})...")
        output_node, pos_encoding = build_game_graph(
            segments, config, max_coord,
            move_speed=0.3, turn_speed=4,
            textures=textures,
            rows_per_patch=args.rows_per_patch,
        )
        max_layers = 200

    rp = args.rows_per_patch if args.rows_per_patch is not None else config.screen_height
    if args.mode == "game":
        max_seq_len = config.screen_width * (config.screen_height // rp)
        extra_metadata = {"rows_per_patch": rp}
    else:
        max_seq_len = config.screen_width
        extra_metadata = None

    compile_headless_to_onnx(
        output_node=output_node,
        pos_encoding=pos_encoding,
        output_path=output_path,
        d=d,
        d_head=args.d_head,
        d_hidden=d_hidden,
        max_seq_len=max_seq_len,
        max_layers=max_layers,
        verbose=True,
        extra_metadata=extra_metadata,
    )


if __name__ == "__main__":
    main()
