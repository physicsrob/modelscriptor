"""Compile the DOOM renderer or game graph and export it to ONNX.

Usage:
    python -m torchwright.doom.to_onnx [--mode renderer|game] [--scene box|multi]
"""

import argparse

from torchwright.compiler.export import export_headless_to_onnx
from torchwright.doom.compile import compile_game, compile_renderer
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
        "--d", type=int, default=None,
        help="Residual stream width. Defaults: 512 (renderer), 1024 (game). "
             "Use a larger value (e.g. 4096 for game) for higher quality.",
    )
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
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
        print(f"Compiling renderer graph (d={d})...")
        module = compile_renderer(
            segments, config, max_coord=max_coord,
            d=d, d_head=args.d_head, device=args.device,
        )
    else:
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
        d = args.d if args.d is not None else 1024
        print(f"Compiling game graph (d={d})...")
        module = compile_game(
            segments, config, max_coord=max_coord,
            textures=textures,
            d=d, d_head=args.d_head, device=args.device,
        )

    export_headless_to_onnx(module, output_path)


if __name__ == "__main__":
    main()
