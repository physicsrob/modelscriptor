"""Compile the DOOM game graph and export it to ONNX.

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
        description="Compile the DOOM game graph and save it to ONNX",
    )
    parser.add_argument("--scene", choices=["box", "multi"], default="box")
    parser.add_argument("--wad", type=str, default="doom1.wad",
                        help="Path to doom1.wad for DOOM textures")
    parser.add_argument("--tex-size", type=int, default=64,
                        help="Texture resolution (downscaled from WAD)")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--fov", type=int, default=32)
    parser.add_argument(
        "--rows-per-patch", type=int, default=10,
        help="Vertical patch height. Must divide --height. "
             "Pass --rows-per-patch <height> for the unsharded phase-α path.",
    )
    parser.add_argument(
        "--d", type=int, default=1024,
        help="Residual stream width. Use a larger value (e.g. 4096) for "
             "higher quality.",
    )
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output .onnx path. Default: doom_game_<scene>.onnx",
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
            wad_path=args.wad, tex_size=args.tex_size,
        )
        max_coord = 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path=args.wad, tex_size=args.tex_size,
        )
        max_coord = 15.0

    print(f"Building game graph (d={args.d})...")
    output_node, pos_encoding = build_game_graph(
        segments, config, max_coord,
        move_speed=0.3, turn_speed=4,
        textures=textures,
        rows_per_patch=args.rows_per_patch,
    )

    max_seq_len = config.screen_width * (config.screen_height // args.rows_per_patch)

    compile_headless_to_onnx(
        output_node=output_node,
        pos_encoding=pos_encoding,
        output_path=output_path,
        d=args.d,
        d_head=args.d_head,
        max_seq_len=max_seq_len,
        max_layers=200,
        verbose=True,
        extra_metadata={"rows_per_patch": args.rows_per_patch},
    )


if __name__ == "__main__":
    main()
