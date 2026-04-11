"""Compile DOOM graphs and run them via the cached-protocol KV-cache loop.

A compiled game graph is executed as a vanilla-transformer autoregressive
rollout: one ``module.step()`` call per screen column, threading the KV
cache across calls.  Step 0 receives the seed state and real player
inputs; steps 1..W-1 pass a zero row and rely on phase α's
``get_prev_value`` attention to read the seed from the cache.  The
graph is unchanged between prefill and cached execution — causal
attention makes the two modes mathematically identical.
"""

from typing import Iterator, List, Optional, Tuple

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
    textures=None,
    d: int = 1024,
    d_head: int = 16,
    device: str = "cpu",
    verbose: bool = True,
    rows_per_patch: Optional[int] = None,
):
    """Compile the game + rendering graph to a HeadlessTransformerModule.

    Args:
        segments: Wall segments (geometry baked into weights).
        config: Render configuration.
        max_coord: Upper bound on coordinate magnitudes.
        move_speed: Player movement speed per frame.
        turn_speed: Angle units per turn input per frame.
        textures: Optional list of (W, H, 3) texture arrays for wall
            textures.  When provided, segments must have ``texture_id``
            set to index into this list.
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
        textures=textures,
        rows_per_patch=rows_per_patch,
    )
    rp = rows_per_patch if rows_per_patch is not None else config.screen_height
    module = compile_headless(
        output_node, pos_encoding,
        d=d, d_head=d_head,
        max_layers=200,
        device=device, verbose=verbose,
        extra_metadata={"rows_per_patch": rp},
    )
    module.eval()
    return module


EXPECTED_INPUT_NAMES = (
    "input_backward",
    "input_forward",
    "input_strafe_left",
    "input_strafe_right",
    "input_turn_left",
    "input_turn_right",
    "seed_angle",
    "seed_x",
    "seed_y",
)


def _seed_row(state: GameState, inputs: PlayerInput) -> torch.Tensor:
    """Build step-0's (1, 9) input row from the seed state + real inputs."""
    row = torch.zeros(1, 9)
    row[0, 0] = float(inputs.backward)
    row[0, 1] = float(inputs.forward)
    row[0, 2] = float(inputs.strafe_left)
    row[0, 3] = float(inputs.strafe_right)
    row[0, 4] = float(inputs.turn_left)
    row[0, 5] = float(inputs.turn_right)
    row[0, 6] = float(state.angle)
    row[0, 7] = float(state.x)
    row[0, 8] = float(state.y)
    return row


def step_frame_iter(
    module,
    state: GameState,
    inputs: PlayerInput,
    config: RenderConfig,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Run one game frame as a cached autoregressive rollout.

    Calls ``module.step()`` once per (col, patch) position, threading
    the KV cache across calls.  Step 0 receives the seed state + real
    player inputs; subsequent steps pass a zero row and rely on phase
    α's ``get_prev_value`` attention to retrieve the seed from the
    cache.  The graph emits ``col_idx`` and ``patch_row_start`` as
    trailing scalars in each step's output, so the host is a dumb
    stitcher: it just reads those indices and pastes the patch.

    Yields:
        ``(col_idx, patch_row_start, patch_pixels)`` per position, where
        ``patch_pixels`` is a ``(rows_per_patch, 3)`` ``float32`` RGB
        array slicing a single screen column.

    Returns (via ``StopIteration.value``):
        The updated ``GameState`` for the next frame.
    """
    assert tuple(module.input_names) == EXPECTED_INPUT_NAMES, (
        f"Compiled module has stale input names {module.input_names}; "
        f"expected {EXPECTED_INPUT_NAMES}."
    )

    W = config.screen_width
    H = config.screen_height
    rp = int(module.metadata.get("rows_per_patch", H))
    assert H % rp == 0, (
        f"screen_height {H} must be divisible by rows_per_patch {rp}"
    )
    shards_per_col = H // rp
    total_steps = W * shards_per_col
    pixel_width = rp * 3

    past = module.empty_past()

    # Step 0: real inputs + seed state. Populates the KV cache so
    # subsequent zero-input decodes can attend back to the seed.
    with torch.no_grad():
        out, past = module.step(_seed_row(state, inputs), past)

    col_idx = int(round(out[0, pixel_width].item()))
    patch_row_start = int(round(out[0, pixel_width + 1].item()))
    new_x = out[0, pixel_width + 2].item()
    new_y = out[0, pixel_width + 3].item()
    new_angle_raw = out[0, pixel_width + 4].item()
    patch = out[0, :pixel_width].cpu().numpy().reshape(rp, 3)
    yield col_idx, patch_row_start, patch

    zero_row = torch.zeros(1, 9)
    for _ in range(1, total_steps):
        with torch.no_grad():
            out, past = module.step(zero_row, past)
        col_idx = int(round(out[0, pixel_width].item()))
        patch_row_start = int(round(out[0, pixel_width + 1].item()))
        patch = out[0, :pixel_width].cpu().numpy().reshape(rp, 3)
        yield col_idx, patch_row_start, patch

    return GameState(
        x=new_x,
        y=new_y,
        angle=round(new_angle_raw) % 256,
        move_speed=state.move_speed,
        turn_speed=state.turn_speed,
    )


def step_frame_compiled(
    module,
    state: GameState,
    inputs: PlayerInput,
    config: RenderConfig,
) -> Tuple[np.ndarray, GameState]:
    """Run one game frame via the cached-protocol generate loop.

    Consumes :func:`step_frame_iter` to the end, assembling the
    self-identifying patch yields into an ``(H, W, 3)`` frame and
    returning the new game state. The host is a dumb stitcher: it
    doesn't know anything about the iteration order or patch layout —
    the graph tells it where each patch belongs.
    """
    W = config.screen_width
    H = config.screen_height

    frame = np.zeros((H, W, 3), dtype=np.float32)
    it = step_frame_iter(module, state, inputs, config)
    try:
        while True:
            col_idx, patch_row_start, patch_pixels = next(it)
            rp = patch_pixels.shape[0]
            frame[patch_row_start : patch_row_start + rp, col_idx, :] = patch_pixels
    except StopIteration as stop:
        new_state: GameState = stop.value

    return frame, new_state
