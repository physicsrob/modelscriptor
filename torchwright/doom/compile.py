"""Compile DOOM graphs and run them via the cached-protocol KV-cache loop.

A compiled game graph is executed as a vanilla-transformer autoregressive
rollout: one ``module.step()`` call per screen column, threading the KV
cache across calls.  Step 0 receives the seed state and real player
inputs; steps 1..W-1 pass a zero row and rely on phase α's
``get_prev_value`` attention to read the seed from the cache.  The
graph is unchanged between prefill and cached execution — causal
attention makes the two modes mathematically identical.
"""

import time
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
    "cur_col_idx",
    "cur_patch_idx_in_col",
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

N_INPUTS = len(EXPECTED_INPUT_NAMES)
_IDX_CUR_COL = EXPECTED_INPUT_NAMES.index("cur_col_idx")
_IDX_CUR_PATCH = EXPECTED_INPUT_NAMES.index("cur_patch_idx_in_col")
_IDX_INPUT_BACKWARD = EXPECTED_INPUT_NAMES.index("input_backward")
_IDX_INPUT_FORWARD = EXPECTED_INPUT_NAMES.index("input_forward")
_IDX_INPUT_STRAFE_LEFT = EXPECTED_INPUT_NAMES.index("input_strafe_left")
_IDX_INPUT_STRAFE_RIGHT = EXPECTED_INPUT_NAMES.index("input_strafe_right")
_IDX_INPUT_TURN_LEFT = EXPECTED_INPUT_NAMES.index("input_turn_left")
_IDX_INPUT_TURN_RIGHT = EXPECTED_INPUT_NAMES.index("input_turn_right")
_IDX_SEED_ANGLE = EXPECTED_INPUT_NAMES.index("seed_angle")
_IDX_SEED_X = EXPECTED_INPUT_NAMES.index("seed_x")
_IDX_SEED_Y = EXPECTED_INPUT_NAMES.index("seed_y")


def _seed_row(state: GameState, inputs: PlayerInput) -> torch.Tensor:
    """Build step-0's (1, N_INPUTS) input row.

    cur_col_idx / cur_patch_idx_in_col are 0 (top-left of the frame).
    seed_{x,y,angle} and input_* are the real values the game logic
    consumes to produce the post-update state.
    """
    row = torch.zeros(1, N_INPUTS)
    row[0, _IDX_CUR_COL] = 0.0
    row[0, _IDX_CUR_PATCH] = 0.0
    row[0, _IDX_INPUT_BACKWARD] = float(inputs.backward)
    row[0, _IDX_INPUT_FORWARD] = float(inputs.forward)
    row[0, _IDX_INPUT_STRAFE_LEFT] = float(inputs.strafe_left)
    row[0, _IDX_INPUT_STRAFE_RIGHT] = float(inputs.strafe_right)
    row[0, _IDX_INPUT_TURN_LEFT] = float(inputs.turn_left)
    row[0, _IDX_INPUT_TURN_RIGHT] = float(inputs.turn_right)
    row[0, _IDX_SEED_ANGLE] = float(state.angle)
    row[0, _IDX_SEED_X] = float(state.x)
    row[0, _IDX_SEED_Y] = float(state.y)
    return row


def _trim_past(past, keep: int):
    """Slice a ``(past_K_tuple, past_V_tuple)`` along dim=1 to the last
    ``keep`` entries.  No-op when the cache already has ``<= keep``
    entries.  Used by :func:`step_frame_iter` when ``cache_lookback`` is
    set to bound the KV cache it hands back to ``module.step`` without
    shrinking the graph's notion of ``past_len``.
    """
    past_K, past_V = past
    if past_K and past_K[0].shape[1] <= keep:
        return past
    trimmed_K = tuple(k[:, -keep:, :] for k in past_K)
    trimmed_V = tuple(v[:, -keep:, :] for v in past_V)
    return (trimmed_K, trimmed_V)


def step_frame_iter(
    module,
    state: GameState,
    inputs: PlayerInput,
    config: RenderConfig,
    cache_lookback: Optional[int] = 10,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Run one game frame as an autoregressive rollout with full feedback.

    The graph is fully position-independent: every step's input row
    carries ``(cur_col_idx, cur_patch_idx_in_col, seed_{x,y,angle},
    input_*)``, and every step's output row emits the patch's rendered
    pixels plus ``(next_col_idx, next_patch_idx_in_col, new_x, new_y,
    new_angle)`` — everything the next step needs to reconstruct its own
    input row.  Step 0 carries real player inputs; the game-logic update
    fires there for real, and its post-update state becomes the
    ``seed_*`` values the host threads forward to all subsequent steps.
    Steps ≥ 1 pass zero player inputs, so the graph's game-logic chain
    degenerates to a passthrough and simply re-emits the carried state.

    Because there is no cross-position attention in the graph, the host
    pastes each patch at the *input-side* ``(cur_col, cur_patch * rp)``
    — which it already knows, since it just wrote those values into the
    input row.

    Args:
        cache_lookback: Optional bound on how many past KV entries to
            thread back into ``module.step``.  ``None`` keeps the full
            cache (stock decode protocol); an integer ``k`` trims the
            cache to the last ``k`` entries before each call.  Default
            is ``10``.  The graph's ``past_len`` input is always set to
            the true global step number, so positional encodings for
            the new row are correct regardless of how much of the cache
            was trimmed (see ``_emit_cached_preamble``).  Trimming is
            only *correct* for graphs whose compiled attention does not
            actually read cross-position K/V to carry semantic signal —
            the post-Option-A DOOM graph qualifies.

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

    # Output layout: [pixels..., next_col, next_patch, new_x, new_y, new_angle]
    _OUT_NEXT_COL = pixel_width
    _OUT_NEXT_PATCH = pixel_width + 1
    _OUT_NEW_X = pixel_width + 2
    _OUT_NEW_Y = pixel_width + 3
    _OUT_NEW_ANGLE = pixel_width + 4

    past = module.empty_past()

    # Step 0: pass cur=(0, 0), real seed state, real player inputs.
    _t0 = time.perf_counter()
    with torch.no_grad():
        # past_len=0 both by shape and by convention.
        out, past = module.step(_seed_row(state, inputs), past, past_len=0)
    _dt_ms = (time.perf_counter() - _t0) * 1000
    print(f"  module.step  step=0/{total_steps}  past_len=0  {_dt_ms:.1f}ms")

    # Paste at the step-0 input position (0, 0).
    cur_col = 0
    cur_patch = 0
    patch = out[0, :pixel_width].cpu().numpy().reshape(rp, 3)
    yield cur_col, cur_patch * rp, patch

    # Read step-0's feedback scalars: they become the next step's input.
    next_col = int(round(out[0, _OUT_NEXT_COL].item()))
    next_patch = int(round(out[0, _OUT_NEXT_PATCH].item()))
    carried_x = out[0, _OUT_NEW_X].item()
    carried_y = out[0, _OUT_NEW_Y].item()
    carried_angle_raw = out[0, _OUT_NEW_ANGLE].item()

    for _step in range(1, total_steps):
        cur_col = next_col
        cur_patch = next_patch
        row = torch.zeros(1, N_INPUTS)
        row[0, _IDX_CUR_COL] = float(cur_col)
        row[0, _IDX_CUR_PATCH] = float(cur_patch)
        row[0, _IDX_SEED_X] = float(carried_x)
        row[0, _IDX_SEED_Y] = float(carried_y)
        row[0, _IDX_SEED_ANGLE] = float(carried_angle_raw)
        # Player-input slots stay zero — the game-logic chain degenerates
        # to a passthrough and re-emits the carried state.

        # Trim the cache handed to the graph while keeping past_len at
        # the true global step number so positional encodings stay
        # correct.
        if cache_lookback is not None:
            past_for_step = _trim_past(past, cache_lookback)
        else:
            past_for_step = past

        _t0 = time.perf_counter()
        with torch.no_grad():
            out, past = module.step(row, past_for_step, past_len=_step)
        _dt_ms = (time.perf_counter() - _t0) * 1000
        cached_len = past[0][0].shape[1] if past[0] else 0
        print(
            f"  module.step  step={_step}/{total_steps}  "
            f"past_len={_step}  cache_len={cached_len}  {_dt_ms:.1f}ms"
        )
        patch = out[0, :pixel_width].cpu().numpy().reshape(rp, 3)
        yield cur_col, cur_patch * rp, patch

        next_col = int(round(out[0, _OUT_NEXT_COL].item()))
        next_patch = int(round(out[0, _OUT_NEXT_PATCH].item()))
        carried_x = out[0, _OUT_NEW_X].item()
        carried_y = out[0, _OUT_NEW_Y].item()
        carried_angle_raw = out[0, _OUT_NEW_ANGLE].item()

    # The post-update game state is whatever we carried forward from
    # step 0 — it's been passing through the zero-input game-logic chain
    # at every subsequent step unchanged.
    return GameState(
        x=carried_x,
        y=carried_y,
        angle=round(carried_angle_raw) % 256,
        move_speed=state.move_speed,
        turn_speed=state.turn_speed,
    )


def step_frame_compiled(
    module,
    state: GameState,
    inputs: PlayerInput,
    config: RenderConfig,
    cache_lookback: Optional[int] = 10,
) -> Tuple[np.ndarray, GameState]:
    """Run one game frame via the cached-protocol generate loop.

    Consumes :func:`step_frame_iter` to the end, assembling the
    self-identifying patch yields into an ``(H, W, 3)`` frame and
    returning the new game state. The host is a dumb stitcher: it
    doesn't know anything about the iteration order or patch layout —
    the graph tells it where each patch belongs.

    ``cache_lookback`` is forwarded to :func:`step_frame_iter` — see
    its docstring for the semantics and safety guarantees.
    """
    W = config.screen_width
    H = config.screen_height

    frame = np.zeros((H, W, 3), dtype=np.float32)
    it = step_frame_iter(module, state, inputs, config, cache_lookback=cache_lookback)
    try:
        while True:
            col_idx, patch_row_start, patch_pixels = next(it)
            rp = patch_pixels.shape[0]
            frame[patch_row_start : patch_row_start + rp, col_idx, :] = patch_pixels
    except StopIteration as stop:
        new_state: GameState = stop.value

    return frame, new_state
