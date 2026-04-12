"""Compile and run the walls-as-tokens game graph (v2).

Replaces ``compile.py``'s baked-geometry protocol with a multi-phase
autoregressive rollout:

    Phase 0 — Prefill:  START + WALL×N + EOS  (host-driven)
    Phase 1 — Sort:     SORTED_WALL×N         (autoregressive, mask feedback)
    Phase 2 — Render:   RENDER×(W × H/rp)     (autoregressive, pixels out)

The host is a dumb token feeder and pixel stitcher.  It feeds player
state at every position, wall geometry at WALL positions, the
accumulated mask at SORTED_WALL positions, and (col, patch) indices
at RENDER positions.  The only "intelligence" is updating the sort
mask from the returned one-hot — a bitwise OR.
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from torchwright.compiler.export import compile_headless
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.game_graph_v2 import (
    E8_EOS,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_START,
    E8_WALL,
    build_game_graph_v2,
)
from torchwright.reference_renderer.types import RenderConfig, Segment


def compile_game_v2(
    config: RenderConfig,
    textures: List[np.ndarray],
    max_walls: int = 8,
    max_coord: float = 20.0,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    d: int = 2048,
    d_head: int = 32,
    device: str = "cpu",
    verbose: bool = True,
    rows_per_patch: Optional[int] = None,
    d_hidden: Optional[int] = None,
):
    """Compile the v2 game graph to a HeadlessTransformerModule."""
    output_node, pos_encoding = build_game_graph_v2(
        config, textures, max_walls, max_coord,
        move_speed, turn_speed,
        rows_per_patch=rows_per_patch,
    )
    rp = rows_per_patch if rows_per_patch is not None else config.screen_height
    module = compile_headless(
        output_node, pos_encoding,
        d=d, d_head=d_head, max_layers=400,
        device=device, verbose=verbose,
        extra_metadata={"rows_per_patch": rp, "max_walls": max_walls},
        d_hidden=d_hidden,
    )
    module.eval()
    return module


def _build_row(compiled, max_walls, **kwargs):
    """Build a (1, d_input) row for module.step()."""
    defaults = {
        "col_idx": torch.zeros(1),
        "input_backward": torch.zeros(1),
        "input_forward": torch.zeros(1),
        "input_strafe_left": torch.zeros(1),
        "input_strafe_right": torch.zeros(1),
        "input_turn_left": torch.zeros(1),
        "input_turn_right": torch.zeros(1),
        "patch_idx": torch.zeros(1),
        "player_angle": torch.zeros(1),
        "player_x": torch.zeros(1),
        "player_y": torch.zeros(1),
        "sort_mask": torch.zeros(max_walls),
        "token_type": torch.zeros(8),
        "wall_ax": torch.zeros(1),
        "wall_ay": torch.zeros(1),
        "wall_bx": torch.zeros(1),
        "wall_by": torch.zeros(1),
        "wall_index": torch.zeros(1),
        "wall_tex_id": torch.zeros(1),
    }
    defaults.update(kwargs)
    d_input = max(s + w for _, s, w in compiled._input_specs)
    row = torch.zeros(1, d_input)
    for name, start, width in compiled._input_specs:
        v = defaults[name]
        if isinstance(v, (int, float)):
            v = torch.tensor([v])
        if v.dim() == 1:
            v = v.unsqueeze(0)
        row[:, start:start + width] = v
    return row


def step_frame_v2(
    module,
    state: GameState,
    inputs: PlayerInput,
    walls: List[dict],
    config: RenderConfig,
) -> Tuple[np.ndarray, GameState]:
    """Run one frame via the v2 multi-phase rollout.

    Args:
        module: Compiled v2 module from :func:`compile_game_v2`.
        state: Current game state (x, y, angle).
        inputs: Player inputs for this frame.
        walls: List of wall dicts with keys ax, ay, bx, by, tex_id.
        config: Render configuration.

    Returns:
        ``(frame, new_state)`` where frame is ``(H, W, 3)`` float32.
    """
    N = len(walls)
    max_walls = int(module.metadata.get("max_walls", 8))
    rp = int(module.metadata.get("rows_per_patch", config.screen_height))
    H = config.screen_height
    W = config.screen_width
    shards_per_col = H // rp

    assert N <= max_walls, f"Too many walls ({N}) for max_walls={max_walls}"
    assert H % rp == 0

    past = module.empty_past()
    step = 0
    px, py, angle = float(state.x), float(state.y), float(state.angle)

    # Output layout indices
    pixel_sl = slice(8, 8 + rp * 3)
    onehot_sl = slice(8 + 5, 8 + 5 + max_walls)

    def _common(**extra):
        return _build_row(
            module, max_walls,
            player_x=torch.tensor([px]),
            player_y=torch.tensor([py]),
            player_angle=torch.tensor([angle]),
            **extra,
        )

    # --- Phase 0: Prefill ---

    # START (with real player inputs)
    row = _common(
        token_type=E8_START,
        input_forward=torch.tensor([float(inputs.forward)]),
        input_backward=torch.tensor([float(inputs.backward)]),
        input_turn_left=torch.tensor([float(inputs.turn_left)]),
        input_turn_right=torch.tensor([float(inputs.turn_right)]),
        input_strafe_left=torch.tensor([float(inputs.strafe_left)]),
        input_strafe_right=torch.tensor([float(inputs.strafe_right)]),
    )
    with torch.no_grad():
        out, past = module.step(row, past, past_len=step)
    step += 1

    # Read updated state from START output
    new_x = out[0, 8].item()
    new_y = out[0, 9].item()
    new_angle_raw = out[0, 10].item()
    # Use updated state for all subsequent positions
    px, py, angle = new_x, new_y, new_angle_raw

    # WALL × N
    for i, w in enumerate(walls):
        row = _common(
            token_type=E8_WALL,
            wall_ax=torch.tensor([w["ax"]]),
            wall_ay=torch.tensor([w["ay"]]),
            wall_bx=torch.tensor([w["bx"]]),
            wall_by=torch.tensor([w["by"]]),
            wall_tex_id=torch.tensor([w["tex_id"]]),
            wall_index=torch.tensor([float(i)]),
        )
        with torch.no_grad():
            out, past = module.step(row, past, past_len=step)
        step += 1

    # EOS
    row = _common(token_type=E8_EOS)
    with torch.no_grad():
        out, past = module.step(row, past, past_len=step)
    step += 1

    # --- Phase 1: Sort ---
    mask = np.zeros(max_walls)
    for k in range(N):
        row = _common(
            token_type=E8_SORTED_WALL,
            sort_mask=torch.tensor(mask, dtype=torch.float32),
        )
        with torch.no_grad():
            out, past = module.step(row, past, past_len=step)
        step += 1
        onehot = out[0, onehot_sl].detach().cpu().numpy()
        mask = np.maximum(mask, np.round(onehot))

    # --- Phase 2: Render ---
    frame = np.zeros((H, W, 3), dtype=np.float32)
    for col in range(W):
        for shard in range(shards_per_col):
            row = _common(
                token_type=E8_RENDER,
                col_idx=torch.tensor([float(col)]),
                patch_idx=torch.tensor([float(shard)]),
            )
            with torch.no_grad():
                out, past = module.step(row, past, past_len=step)
            step += 1
            pixels = out[0, pixel_sl].detach().cpu().numpy().reshape(rp, 3)
            row_start = shard * rp
            frame[row_start:row_start + rp, col, :] = pixels

    new_state = GameState(
        x=px, y=py,
        angle=round(new_angle_raw) % 256,
        move_speed=state.move_speed,
        turn_speed=state.turn_speed,
    )
    return frame, new_state
