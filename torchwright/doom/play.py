"""Interactive DOOM game loop with pygame display.

Usage:
    python -m torchwright.doom.play [--scene box|multi] [--mode transformer|reference]
"""

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer.render import render_frame
from torchwright.reference_renderer.scenes import (
    box_room_textured, multi_room_textured,
)
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


def _make_alt_walls(half=5.0):
    """Build an alternative L-shaped room layout for level-swap demo."""
    return [
        {"ax": half, "ay": -half, "bx": half, "by": 0.0, "tex_id": 0.0},
        {"ax": half, "ay": 0.0, "bx": 0.0, "by": 0.0, "tex_id": 1.0},
        {"ax": 0.0, "ay": 0.0, "bx": 0.0, "by": half, "tex_id": 2.0},
        {"ax": 0.0, "ay": half, "bx": -half, "by": half, "tex_id": 3.0},
        {"ax": -half, "ay": half, "bx": -half, "by": -half, "tex_id": 0.0},
        {"ax": -half, "ay": -half, "bx": half, "by": -half, "tex_id": 1.0},
    ]


def play(
    segments: List[Segment],
    config: RenderConfig,
    start_x: float,
    start_y: float,
    start_angle: int,
    max_coord: float = 20.0,
    scale: int = 8,
    mode: str = "transformer",
    textures=None,
    rows_per_patch: int = 10,
) -> None:
    """Run an interactive game loop with pygame display.

    Args:
        segments: Wall segments defining the map.
        config: Render configuration.
        start_x, start_y: Starting position.
        start_angle: Starting facing direction (0-255).
        max_coord: Upper bound on coordinate magnitudes.
        scale: Pixel scaling factor for display window.
        mode: "transformer" compiles the v2 walls-as-tokens graph and
            runs inference.  "reference" uses the Python implementation
            as a ground-truth baseline.
        textures: Optional texture atlas.

    Press L during play to swap between two level layouts (same
    compiled weights, different wall tokens) — demonstrates level
    independence.
    """
    try:
        import pygame
    except ImportError:
        print("pygame is required for interactive play: pip install pygame")
        sys.exit(1)

    if mode == "transformer":
        from torchwright.doom.compile import (
            compile_game, step_frame, segments_to_walls,
        )

        walls_a = segments_to_walls(segments)
        walls_b = _make_alt_walls()
        max_walls = max(8, len(walls_a), len(walls_b))

        print(f"Compiling game graph ({len(walls_a)} walls, max_walls={max_walls})...")
        module = compile_game(
            config, textures,
            max_walls=max_walls,
            max_coord=max_coord,
            d=2048, d_head=32,
            rows_per_patch=rows_per_patch,
        )
        current_walls = [walls_a]  # mutable container for level-swap

        def frame_fn(state, inputs):
            return step_frame(module, state, inputs, current_walls[0], config)
    else:
        current_walls = [None]
        trig_table = config.trig_table

        def frame_fn(state, inputs):
            new_state = update_state(state, inputs, segments, trig_table)
            frame = render_frame(
                new_state.x, new_state.y, new_state.angle, segments, config,
                textures=textures,
            )
            return frame, new_state

    state = GameState(x=start_x, y=start_y, angle=start_angle)

    pygame.init()
    W, H = config.screen_width, config.screen_height
    screen = pygame.display.set_mode((W * scale, H * scale))
    label = "DOOM Transformer" if mode == "transformer" else "DOOM Reference"
    pygame.display.set_caption(label)

    running = True
    frame_count = 0
    last_frame_ms = 0.0
    level_name = "A"
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_l and mode == "transformer":
                    if current_walls[0] is walls_a:
                        current_walls[0] = walls_b
                        level_name = "B"
                    else:
                        current_walls[0] = walls_a
                        level_name = "A"
                    print(f"Level swap → {level_name} (no recompile)")

        keys = pygame.key.get_pressed()
        inputs = PlayerInput(
            forward=keys[pygame.K_w] or keys[pygame.K_UP],
            backward=keys[pygame.K_s] or keys[pygame.K_DOWN],
            strafe_left=keys[pygame.K_a],
            strafe_right=keys[pygame.K_d],
            turn_left=keys[pygame.K_LEFT],
            turn_right=keys[pygame.K_RIGHT],
        )

        t0 = time.perf_counter()
        frame, state = frame_fn(state, inputs)
        last_frame_ms = (time.perf_counter() - t0) * 1000.0

        pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surface, (W * scale, H * scale))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        frame_count += 1
        if frame_count % 10 == 0:
            fps = 1000.0 / last_frame_ms if last_frame_ms > 0 else 0
            lev = f" level={level_name}" if mode == "transformer" else ""
            pygame.display.set_caption(
                f"{label} | pos=({state.x:.1f}, {state.y:.1f}) "
                f"angle={state.angle}{lev} | {last_frame_ms:.0f}ms ({fps:.1f} fps)"
            )

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Play DOOM in a transformer")
    parser.add_argument("--scene", choices=["box", "multi"], default="box")
    parser.add_argument(
        "--mode", choices=["transformer", "reference"], default="transformer",
        help="transformer: game logic + rendering in compiled transformer (default). "
             "reference: pure Python implementation.",
    )
    parser.add_argument("--wad", type=str, default="doom1.wad",
                        help="Path to doom1.wad for DOOM textures")
    parser.add_argument("--tex-size", type=int, default=8,
                        help="Texture resolution (downscaled from WAD)")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--fov", type=int, default=32)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument(
        "--rows-per-patch", type=int, default=10,
        help="Vertical patch height. Must divide --height.",
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

    if args.scene == "box":
        segments, textures = box_room_textured(
            wad_path=args.wad, tex_size=args.tex_size,
        )
        start_x, start_y, start_angle = 0.0, 0.0, 0
        max_coord = 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path=args.wad, tex_size=args.tex_size,
        )
        start_x, start_y, start_angle = -8.0, 0.0, 0
        max_coord = 15.0

    play(segments, config, start_x, start_y, start_angle,
         max_coord=max_coord, scale=args.scale, mode=args.mode,
         textures=textures, rows_per_patch=args.rows_per_patch)


if __name__ == "__main__":
    main()
