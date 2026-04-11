"""Compile DOOM graphs and run them via the cached-protocol KV-cache loop.

A compiled game graph is executed as a vanilla-transformer autoregressive
rollout: one ``module.step()`` call per screen column, threading the KV
cache across calls.  Step 0 receives the seed state and real player
inputs; steps 1..W-1 pass a zero row and rely on phase α's
``get_prev_value`` attention to read the seed from the cache.  The
graph is unchanged between prefill and cached execution — causal
attention makes the two modes mathematically identical.
"""

from typing import Iterator, List, Tuple

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
    )
    module = compile_headless(
        output_node, pos_encoding,
        d=d, d_head=d_head,
        max_layers=200,
        device=device, verbose=verbose,
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
) -> Iterator[Tuple[int, np.ndarray]]:
    """Run one game frame as a cached autoregressive rollout.

    Calls ``module.step()`` once per screen column, threading the KV
    cache across calls.  Step 0 receives the seed state + real player
    inputs; steps 1..W-1 pass a zero row and rely on phase α's
    ``get_prev_value`` attention to retrieve the seed from the cache.

    Yields:
        ``(col_idx, col_pixels)`` per column, where ``col_pixels`` is an
        ``(H, 3)`` ``float32`` RGB array.  Consumers can render columns
        progressively as they arrive.

    Returns (via ``StopIteration.value``):
        The updated ``GameState`` for the next frame.  All positions in
        the rollout compute the same post-game-logic state because
        steps 1..W-1 receive zero player inputs, so any column's state
        slice would suffice — we read it from step 0.
    """
    assert tuple(module.input_names) == EXPECTED_INPUT_NAMES, (
        f"Compiled module has stale input names {module.input_names}; "
        f"expected {EXPECTED_INPUT_NAMES}. Recompile after the phase-α refactor."
    )

    W = config.screen_width
    H = config.screen_height
    past = module.empty_past()

    # Step 0: real inputs + seed state.  Produces column 0 and populates
    # the KV cache for subsequent steps to attend back into.
    with torch.no_grad():
        out0, past = module.step(_seed_row(state, inputs), past)

    col0 = out0[0, : H * 3].cpu().numpy().reshape(H, 3)
    yield 0, col0

    new_x = out0[0, H * 3].item()
    new_y = out0[0, H * 3 + 1].item()
    new_angle_raw = out0[0, H * 3 + 2].item()

    # Steps 1..W-1: zero-row decodes.  The graph's get_prev_value
    # attention reads step 0's seed from the cache at every step, so
    # game logic is effectively the identity on the cached state.
    zero_row = torch.zeros(1, 9)
    for col in range(1, W):
        with torch.no_grad():
            out, past = module.step(zero_row, past)
        yield col, out[0, : H * 3].cpu().numpy().reshape(H, 3)

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

    Consumes :func:`step_frame_iter` to the end, assembling the per-column
    yields into an ``(H, W, 3)`` frame and returning the new game state.

    Args:
        module: Compiled DOOM game module — any :class:`HeadlessRuntime`
            (``CompiledHeadless`` or ``OnnxHeadlessModule``) with
            ``.step()`` + ``.empty_past()``.
        state: Current game state (seeded at step 0).
        inputs: Player input flags for this frame.
        config: Render configuration.

    Returns:
        ``(frame, new_state)`` where frame is ``(H, W, 3)`` float RGB
        and ``new_state`` is the post-game-logic state.
    """
    W = config.screen_width
    H = config.screen_height

    frame = np.zeros((H, W, 3), dtype=np.float32)
    it = step_frame_iter(module, state, inputs, config)
    try:
        while True:
            col_idx, col_pixels = next(it)
            frame[:, col_idx, :] = col_pixels
    except StopIteration as stop:
        new_state: GameState = stop.value

    return frame, new_state
