"""Interactive DOOM game loop with pygame display.

Usage:
    python -m torchwright.doom.play [--scene box|multi] [--mode transformer|reference]
"""

import argparse
import sys
from typing import List

import numpy as np

from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer.render import render_frame
from torchwright.reference_renderer.scenes import box_room, box_room_textured, multi_room
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


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
) -> None:
    """Run an interactive game loop with pygame display.

    Args:
        segments: Wall segments defining the map.
        config: Render configuration.
        start_x, start_y: Starting position.
        start_angle: Starting facing direction (0-255).
        max_coord: Upper bound on coordinate magnitudes.
        scale: Pixel scaling factor for display window.
        mode: "transformer" compiles game logic + rendering into a
            transformer. "reference" uses the Python implementation.
        textures: Optional texture atlas for wall textures.
    """
    try:
        import pygame
    except ImportError:
        print("pygame is required for interactive play: pip install pygame")
        sys.exit(1)

    if mode == "transformer":
        from torchwright.doom.compile import compile_game, step_frame_compiled

        print("Compiling game graph...")
        module = compile_game(
            segments, config, max_coord,
            textures=textures,
            d=2048 if textures else 1024, d_head=16,
        )

        def frame_fn(state, inputs):
            frame, new_state = step_frame_compiled(module, state, inputs, config)
            return frame, new_state
    else:
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
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        inputs = PlayerInput(
            forward=keys[pygame.K_w] or keys[pygame.K_UP],
            backward=keys[pygame.K_s] or keys[pygame.K_DOWN],
            strafe_left=keys[pygame.K_a],
            strafe_right=keys[pygame.K_d],
            turn_left=keys[pygame.K_LEFT],
            turn_right=keys[pygame.K_RIGHT],
        )

        frame, state = frame_fn(state, inputs)

        pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surface, (W * scale, H * scale))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        frame_count += 1
        if frame_count % 10 == 0:
            pygame.display.set_caption(
                f"{label} | pos=({state.x:.1f}, {state.y:.1f}) "
                f"angle={state.angle}"
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
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--fov", type=int, default=32)
    parser.add_argument("--scale", type=int, default=8)
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

    textures = None
    if args.scene == "box":
        segments, textures = box_room_textured()
        start_x, start_y, start_angle = 0.0, 0.0, 0
        max_coord = 10.0
    else:
        segments = multi_room()
        start_x, start_y, start_angle = -8.0, 0.0, 0
        max_coord = 15.0

    play(segments, config, start_x, start_y, start_angle,
         max_coord=max_coord, scale=args.scale, mode=args.mode,
         textures=textures)


if __name__ == "__main__":
    main()
