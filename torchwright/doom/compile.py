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
from torchwright.reference_renderer.types import RenderConfig, Segment


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
    d_hidden: Optional[int] = None,
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
        d_hidden: Per-layer MLP hidden width; defaults to ``d``.

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
        d_hidden=d_hidden,
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
    "prev_col_idx",
    "prev_patch_idx_in_col",
    "seed_angle",
    "seed_x",
    "seed_y",
)

N_INPUTS = len(EXPECTED_INPUT_NAMES)
_IDX_PREV_COL = EXPECTED_INPUT_NAMES.index("prev_col_idx")
_IDX_PREV_PATCH = EXPECTED_INPUT_NAMES.index("prev_patch_idx_in_col")


def _seed_row(state: GameState, inputs: PlayerInput) -> torch.Tensor:
    """Build step-0's (1, N_INPUTS) input row from the seed state + real inputs.

    The prev_col_idx / prev_patch_idx_in_col slots are left at 0; at
    position 0 the graph's is_pos_0 select overrides the delta output
    to (0, 0) regardless of what's passed.
    """
    row = torch.zeros(1, N_INPUTS)
    row[0, 0] = float(inputs.backward)
    row[0, 1] = float(inputs.forward)
    row[0, 2] = float(inputs.strafe_left)
    row[0, 3] = float(inputs.strafe_right)
    row[0, 4] = float(inputs.turn_left)
    row[0, 5] = float(inputs.turn_right)
    row[0, EXPECTED_INPUT_NAMES.index("seed_angle")] = float(state.angle)
    row[0, EXPECTED_INPUT_NAMES.index("seed_x")] = float(state.x)
    row[0, EXPECTED_INPUT_NAMES.index("seed_y")] = float(state.y)
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
    player inputs; subsequent steps pass a zero row for the player
    inputs (the graph broadcasts the seed from step 0 via
    ``get_prev_value``) plus the previous step's emitted
    ``(col_idx, patch_idx_in_col)`` so the graph can compute the new
    values via a local delta.  The graph emits ``col_idx`` and
    ``patch_row_start`` as trailing scalars in each step's output, so
    the host is a dumb stitcher: it just reads those indices and pastes
    the patch without knowing the sharding layout.

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
    # subsequent zero-input decodes can attend back to the seed.  The
    # prev_col / prev_patch slots are ignored at pos 0 (is_pos_0
    # override forces the outputs to (0, 0)).
    with torch.no_grad():
        out, past = module.step(_seed_row(state, inputs), past)

    col_idx = int(round(out[0, pixel_width].item()))
    patch_row_start = int(round(out[0, pixel_width + 1].item()))
    patch_idx_in_col = patch_row_start // rp
    new_x = out[0, pixel_width + 2].item()
    new_y = out[0, pixel_width + 3].item()
    new_angle_raw = out[0, pixel_width + 4].item()
    patch = out[0, :pixel_width].cpu().numpy().reshape(rp, 3)
    yield col_idx, patch_row_start, patch

    for _ in range(1, total_steps):
        row = torch.zeros(1, N_INPUTS)
        row[0, _IDX_PREV_COL] = float(col_idx)
        row[0, _IDX_PREV_PATCH] = float(patch_idx_in_col)
        with torch.no_grad():
            out, past = module.step(row, past)
        col_idx = int(round(out[0, pixel_width].item()))
        patch_row_start = int(round(out[0, pixel_width + 1].item()))
        patch_idx_in_col = patch_row_start // rp
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
