"""Compile a renderer graph to a HeadlessTransformerModule and run it."""

from typing import List

import numpy as np
import torch

from torchwright.compiler.export import compile_headless
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
