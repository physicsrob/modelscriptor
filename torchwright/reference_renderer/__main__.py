"""CLI entry point: python -m torchwright.reference_renderer [output.png]"""

import argparse

from torchwright.reference_renderer import (
    RenderConfig,
    generate_trig_table,
    render_frame,
)
from torchwright.reference_renderer.render import save_png
from torchwright.reference_renderer.scenes import box_room


def main():
    parser = argparse.ArgumentParser(description="Render the box room to a PNG file.")
    parser.add_argument(
        "output", nargs="?", default="box_room.png", help="Output PNG path"
    )
    parser.add_argument("--width", type=int, default=320, help="Screen width in pixels")
    parser.add_argument(
        "--height", type=int, default=200, help="Screen height in pixels"
    )
    parser.add_argument(
        "--fov", type=int, default=64, help="FOV in angle indices (64 ≈ 90°)"
    )
    parser.add_argument("--angle", type=int, default=0, help="Player angle (0-255)")
    args = parser.parse_args()

    config = RenderConfig(
        screen_width=args.width,
        screen_height=args.height,
        fov_columns=args.fov,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )

    segments = box_room()
    frame = render_frame(0.0, 0.0, args.angle, segments, config)
    save_png(frame, args.output)
    print(f"Saved {args.output} ({args.width}x{args.height}, angle={args.angle})")


if __name__ == "__main__":
    main()
