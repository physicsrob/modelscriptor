"""Interactive DOOM game loop with pygame display.

Usage:
    python -m torchwright.doom.play [--scene box|multi] [--width 32] [--height 40]
"""

import argparse
import sys
from typing import List

import numpy as np

from torchwright.doom.compile import compile_renderer, render_frame_compiled
from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer.scenes import box_room, multi_room
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
) -> None:
    """Run an interactive game loop with pygame display.

    Args:
        segments: Wall segments defining the map.
        config: Render configuration.
        start_x, start_y: Starting position.
        start_angle: Starting facing direction (0-255).
        max_coord: Upper bound on coordinate magnitudes.
        scale: Pixel scaling factor for display window.
    """
    try:
        import pygame
    except ImportError:
        print("pygame is required for interactive play: pip install pygame")
        sys.exit(1)

    print("Compiling renderer...")
    module = compile_renderer(segments, config, max_coord, d=1024, d_head=16)

    trig_table = config.trig_table
    state = GameState(x=start_x, y=start_y, angle=start_angle)

    pygame.init()
    W, H = config.screen_width, config.screen_height
    screen = pygame.display.set_mode((W * scale, H * scale))
    pygame.display.set_caption("DOOM Transformer")
    clock = pygame.time.Clock()

    running = True
    frame_count = 0
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Read keyboard state
        keys = pygame.key.get_pressed()
        inputs = PlayerInput(
            forward=keys[pygame.K_w] or keys[pygame.K_UP],
            backward=keys[pygame.K_s] or keys[pygame.K_DOWN],
            strafe_left=keys[pygame.K_a],
            strafe_right=keys[pygame.K_d],
            turn_left=keys[pygame.K_LEFT],
            turn_right=keys[pygame.K_RIGHT],
        )

        # Update game state
        state = update_state(state, inputs, segments, trig_table)

        # Render frame
        frame = render_frame_compiled(module, state.x, state.y, state.angle, config)

        # Display
        pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surface, (W * scale, H * scale))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        frame_count += 1
        if frame_count % 10 == 0:
            pygame.display.set_caption(
                f"DOOM Transformer | pos=({state.x:.1f}, {state.y:.1f}) "
                f"angle={state.angle}"
            )

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Play DOOM in a transformer")
    parser.add_argument("--scene", choices=["box", "multi"], default="box")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--fov", type=int, default=16)
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

    if args.scene == "box":
        segments = box_room()
        start_x, start_y, start_angle = 0.0, 0.0, 0
        max_coord = 10.0
    else:
        segments = multi_room()
        start_x, start_y, start_angle = -8.0, 0.0, 0
        max_coord = 15.0

    play(segments, config, start_x, start_y, start_angle,
         max_coord=max_coord, scale=args.scale)


if __name__ == "__main__":
    main()
