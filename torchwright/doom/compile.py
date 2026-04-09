"""Compile renderer and game graphs to HeadlessTransformerModule and run them."""

from typing import List, Tuple

import numpy as np
import torch

from torchwright.compiler.export import compile_headless
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.renderer import build_renderer_graph
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


def compile_renderer(
    segments: List[Segment],
    config: RenderConfig,
    max_coord: float = 20.0,
    d: int = 512,
    d_head: int = 16,
    device: str = "cpu",
    verbose: bool = True,
):
    """Compile a flat-shaded renderer to a HeadlessTransformerModule.

    Args:
        segments: Wall segments (geometry baked into weights).
        config: Render configuration.
        max_coord: Upper bound on coordinate magnitudes.
        d: Residual stream width (d_model).
        d_head: Attention head dimension.
        device: Target device.
        verbose: Print compilation stats.

    Returns:
        HeadlessTransformerModule ready for inference.
    """
    output_node, pos_encoding = build_renderer_graph(segments, config, max_coord)
    module = compile_headless(
        output_node, pos_encoding,
        d=d, d_head=d_head,
        device=device, verbose=verbose,
    )
    module.eval()
    return module


def render_frame_compiled(
    module,
    player_x: float,
    player_y: float,
    player_angle: int,
    config: RenderConfig,
) -> np.ndarray:
    """Render a full frame using a compiled HeadlessTransformerModule.

    Pre-computes ray_angle and perp_cos per column, runs the module,
    and reshapes the output into an (H, W, 3) image array.

    Args:
        module: Compiled HeadlessTransformerModule.
        player_x, player_y: Player world coordinates.
        player_angle: Player facing direction (0-255).
        config: Render configuration.

    Returns:
        np.ndarray of shape (H, W, 3) with float RGB values.
    """
    W = config.screen_width
    H = config.screen_height
    trig = config.trig_table

    # Build input tensor: (W, 4) with columns [perp_cos, player_x, player_y, ray_angle]
    # (alphabetical order matching input_names)
    inputs = torch.zeros(W, 4)
    for col in range(W):
        col_offset = col - W // 2
        ray_angle = (player_angle + col_offset * config.fov_columns // W) % 256
        angle_diff = (ray_angle - player_angle) % 256
        perp_cos = trig[angle_diff, 0]

        inputs[col, 0] = perp_cos     # perp_cos
        inputs[col, 1] = player_x     # player_x
        inputs[col, 2] = player_y     # player_y
        inputs[col, 3] = ray_angle    # ray_angle

    with torch.no_grad():
        output = module(inputs)  # (W, H*3)

    # Reshape to (H, W, 3)
    frame = output.cpu().numpy().reshape(W, H, 3).transpose(1, 0, 2)
    return frame


# ---------------------------------------------------------------------------
# Phase 3: Game graph (game logic + rendering in one forward pass)
# ---------------------------------------------------------------------------


def compile_game(
    segments: List[Segment],
    config: RenderConfig,
    max_coord: float = 20.0,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    d: int = 1024,
    d_head: int = 16,
    device: str = "cpu",
    verbose: bool = True,
):
    """Compile the game + rendering graph to a HeadlessTransformerModule.

    Args:
        segments: Wall segments (geometry baked into weights).
        config: Render configuration.
        max_coord: Upper bound on coordinate magnitudes.
        move_speed: Player movement speed per frame.
        turn_speed: Angle units per turn input per frame.
        d: Residual stream width (d_model).
        d_head: Attention head dimension.
        device: Target device.
        verbose: Print compilation stats.

    Returns:
        HeadlessTransformerModule ready for inference.
    """
    from torchwright.doom.game_graph import build_game_graph

    output_node, pos_encoding = build_game_graph(
        segments, config, max_coord, move_speed, turn_speed,
    )
    module = compile_headless(
        output_node, pos_encoding,
        d=d, d_head=d_head,
        max_layers=200,
        device=device, verbose=verbose,
    )
    module.eval()
    return module


def step_frame_compiled(
    module,
    state: GameState,
    inputs: PlayerInput,
    config: RenderConfig,
) -> Tuple[np.ndarray, GameState]:
    """Run one game frame: update state + render.

    Builds the input tensor (11 inputs per column), runs the compiled
    game graph, and extracts the rendered frame and updated state.

    Args:
        module: Compiled game HeadlessTransformerModule.
        state: Current game state.
        inputs: Player input flags for this frame.
        config: Render configuration.

    Returns:
        (frame, new_state) where frame is (H, W, 3) float array
        and new_state is the updated GameState.
    """
    W = config.screen_width
    H = config.screen_height
    trig = config.trig_table

    # Build input tensor: (W, 11) — alphabetical input order
    inp = torch.zeros(W, 11)
    for col in range(W):
        col_offset = col - W // 2
        angle_offset = col_offset * config.fov_columns // W
        perp_cos_val = float(trig[angle_offset % 256, 0])

        inp[col, 0] = float(angle_offset)
        inp[col, 1] = float(inputs.backward)
        inp[col, 2] = float(inputs.forward)
        inp[col, 3] = float(inputs.strafe_left)
        inp[col, 4] = float(inputs.strafe_right)
        inp[col, 5] = float(inputs.turn_left)
        inp[col, 6] = float(inputs.turn_right)
        inp[col, 7] = float(state.angle)
        inp[col, 8] = float(state.x)
        inp[col, 9] = float(state.y)
        inp[col, 10] = perp_cos_val

    with torch.no_grad():
        output = module(inp)  # (W, H*3 + 3)

    # Extract pixels: first H*3 columns
    pixels = output[:, :H * 3].cpu().numpy().reshape(W, H, 3).transpose(1, 0, 2)

    # Extract state from position 0 (all positions compute the same state)
    new_x = output[0, H * 3].item()
    new_y = output[0, H * 3 + 1].item()
    new_angle = round(output[0, H * 3 + 2].item()) % 256

    new_state = GameState(
        x=new_x, y=new_y, angle=new_angle,
        move_speed=state.move_speed, turn_speed=state.turn_speed,
    )
    return pixels, new_state
