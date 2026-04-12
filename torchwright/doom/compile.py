"""Compile and run the walls-as-tokens game graph.

Multi-phase autoregressive rollout:

    Phase 0 — Prefill:  START + WALL×N + EOS  (host-driven)
    Phase 1 — Sort:     SORTED_WALL×N         (autoregressive, mask feedback)
    Phase 2 — Render:   RENDER×(W × H/rp)     (autoregressive, pixels out)

The host is a dumb token feeder and pixel stitcher.  It feeds player
state at every position, wall geometry at WALL positions, the
accumulated mask at SORTED_WALL positions, and (col, patch) indices
at RENDER positions.  The only "intelligence" is updating the sort
mask from the returned one-hot — a bitwise OR.
"""

import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from torchwright.compiler.export import compile_headless
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.game_graph import (
    E8_EOS,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_START,
    E8_WALL,
    build_game_graph,
)
from torchwright.reference_renderer.types import RenderConfig, Segment


def segments_to_walls(segments: List[Segment]) -> List[dict]:
    """Convert Segment objects to the wall dict format expected by step_frame."""
    return [
        {
            "ax": seg.ax, "ay": seg.ay,
            "bx": seg.bx, "by": seg.by,
            "tex_id": float(seg.texture_id if seg.texture_id is not None else 0),
        }
        for seg in segments
    ]


def compile_game(
    config: RenderConfig,
    textures: List[np.ndarray],
    max_walls: int = 8,
    max_coord: float = 20.0,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    d: int = 2048,
    d_head: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
    rows_per_patch: Optional[int] = None,
    d_hidden: Optional[int] = None,
):
    """Compile the game graph to a HeadlessTransformerModule."""
    output_node, pos_encoding = build_game_graph(
        config, textures, max_walls, max_coord,
        move_speed, turn_speed,
        rows_per_patch=rows_per_patch,
    )
    # The render attention d_head = W + 6 (visibility mask + bias + value
    # passthrough).  Round up to a power of 2 that divides d.
    if d_head is None:
        W = config.screen_width
        min_d_head = W + 6
        d_head = 1
        while d_head < min_d_head:
            d_head *= 2
        assert d % d_head == 0, f"d={d} not divisible by d_head={d_head}"
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
    device = compiled._net.device
    defaults = {
        "col_idx": torch.zeros(1, device=device),
        "input_backward": torch.zeros(1, device=device),
        "input_forward": torch.zeros(1, device=device),
        "input_strafe_left": torch.zeros(1, device=device),
        "input_strafe_right": torch.zeros(1, device=device),
        "input_turn_left": torch.zeros(1, device=device),
        "input_turn_right": torch.zeros(1, device=device),
        "patch_idx": torch.zeros(1, device=device),
        "player_angle": torch.zeros(1, device=device),
        "player_x": torch.zeros(1, device=device),
        "player_y": torch.zeros(1, device=device),
        "sort_mask": torch.zeros(max_walls, device=device),
        "token_type": torch.zeros(8, device=device),
        "wall_ax": torch.zeros(1, device=device),
        "wall_ay": torch.zeros(1, device=device),
        "wall_bx": torch.zeros(1, device=device),
        "wall_by": torch.zeros(1, device=device),
        "wall_index": torch.zeros(1, device=device),
        "wall_tex_id": torch.zeros(1, device=device),
    }
    defaults.update(kwargs)
    d_input = max(s + w for _, s, w in compiled._input_specs)
    row = torch.zeros(1, d_input, device=device)
    for name, start, width in compiled._input_specs:
        v = defaults[name]
        if isinstance(v, (int, float)):
            v = torch.tensor([v], device=device)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        row[:, start:start + width] = v.to(device)
    return row


def step_frame(
    module,
    state: GameState,
    inputs: PlayerInput,
    walls: List[dict],
    config: RenderConfig,
) -> Tuple[np.ndarray, GameState]:
    """Run one frame via the multi-phase rollout.

    Args:
        module: Compiled module from :func:`compile_game`.
        state: Current game state (x, y, angle).
        inputs: Player inputs for this frame.
        walls: List of wall dicts with keys ax, ay, bx, by, tex_id.
        config: Render configuration.

    Returns:
        ``(frame, new_state)`` where frame is ``(H, W, 3)`` float32.

    Host protocol — the host is a dumb token feeder:
        1. START + WALL×N — feed real inputs + wall geometry
        2. EOS — feed real inputs → read (resolved_x, resolved_y, new_angle)
        3. SORTED_WALL×N — feed resolved state + sort mask feedback
        4. RENDER×(W×H/rp) — feed resolved state + (col, patch)

    Collision detection and wall sliding are handled entirely inside
    the graph (WALL tokens compute per-wall hit flags, EOS aggregates
    them via attention and resolves the position).
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
    total_steps = 1 + N + 1 + N + W * shards_per_col
    px, py, angle = float(state.x), float(state.y), float(state.angle)

    # Output layout indices
    pixel_sl = slice(8, 8 + rp * 3)
    onehot_sl = slice(8 + 5, 8 + 5 + max_walls)

    # Player input tensors (reused at START, WALL, and EOS positions
    # so the graph computes velocity consistently at all three)
    input_kw = dict(
        input_forward=torch.tensor([float(inputs.forward)]),
        input_backward=torch.tensor([float(inputs.backward)]),
        input_turn_left=torch.tensor([float(inputs.turn_left)]),
        input_turn_right=torch.tensor([float(inputs.turn_right)]),
        input_strafe_left=torch.tensor([float(inputs.strafe_left)]),
        input_strafe_right=torch.tensor([float(inputs.strafe_right)]),
    )

    def _common(**extra):
        return _build_row(
            module, max_walls,
            player_x=torch.tensor([px]),
            player_y=torch.tensor([py]),
            player_angle=torch.tensor([angle]),
            **extra,
        )

    def _kv_len(past):
        return past[0][0].shape[1] if past[0][0].numel() > 0 else 0

    def _step(row, past, step):
        with torch.no_grad():
            return module.step(row, past, past_len=step)

    t_frame = time.perf_counter()

    # --- Phase 0: Prefill (START + WALL×N + EOS) ---
    t0 = time.perf_counter()

    # START
    row = _common(token_type=E8_START, **input_kw)
    out, past = _step(row, past, step)
    step += 1

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
            **input_kw,
        )
        out, past = _step(row, past, step)
        step += 1

    # EOS — graph resolves collision via attention to WALL positions
    row = _common(token_type=E8_EOS, **input_kw)
    out, past = _step(row, past, step)
    step += 1

    t_prefill = time.perf_counter() - t0
    print(f"  prefill  {step} steps  kv={_kv_len(past)}  {t_prefill*1000:.0f}ms")

    # Read collision-resolved state from EOS output
    px = out[0, 8].item()
    py = out[0, 9].item()
    new_angle_raw = out[0, 10].item()
    angle = new_angle_raw

    # --- Phase 1: Sort ---
    t0 = time.perf_counter()
    mask = np.zeros(max_walls)
    for k in range(N):
        row = _common(
            token_type=E8_SORTED_WALL,
            sort_mask=torch.tensor(mask, dtype=torch.float32),
        )
        out, past = _step(row, past, step)
        step += 1
        raw_sort = out[0].detach().cpu().numpy()
        # Sort output: [type(8), wall_data(5), onehot(max_walls)]
        wall_data = raw_sort[8:13]
        onehot = raw_sort[onehot_sl]
        # Also read the sel_dist from the sort value (position 7 = 8+5+2 = offset 15 in raw output)
        # Actually, onehot_sl starts at 8+5 = 13, so dist would be at position... let me print more context
        print(f"  sort[{k}]: wall=[{wall_data[0]:.2f},{wall_data[1]:.2f},{wall_data[2]:.2f},{wall_data[3]:.2f}] "
              f"tex={wall_data[4]:.1f} oh_max={np.argmax(onehot)} raw[13:16]={raw_sort[13:16].tolist()}")
        mask = np.maximum(mask, np.round(onehot))

    t_sort = time.perf_counter() - t0
    print(f"  sort     {N} steps  kv={_kv_len(past)}  {t_sort*1000:.0f}ms")

    # Print sort results for debugging
    print(f"  sort results: mask={mask.tolist()}")
    print(f"  resolved state: px={px:.3f} py={py:.3f} angle={angle:.3f}")
    # Re-read the last sort output to see what wall was selected
    print(f"  last sort out[:20]: {out[0, :20].detach().cpu().numpy().tolist()}")

    # --- Phase 2: Render ---
    t0 = time.perf_counter()
    frame = np.zeros((H, W, 3), dtype=np.float32)
    _diag_printed = False
    for col in range(W):
        for shard in range(shards_per_col):
            row = _common(
                token_type=E8_RENDER,
                col_idx=torch.tensor([float(col)]),
                patch_idx=torch.tensor([float(shard)]),
            )
            out, past = _step(row, past, step)
            step += 1
            raw_out = out[0].detach().cpu().numpy()
            if not _diag_printed and col == W // 2 and shard == 0:
                # Dump first 20 output values at center column
                print(f"  render diag col={col} shard={shard}: "
                      f"out[:20]={raw_out[:20].tolist()}")
                _diag_printed = True
            pixels = raw_out[pixel_sl].reshape(rp, 3)
            row_start = shard * rp
            frame[row_start:row_start + rp, col, :] = pixels

    t_render = time.perf_counter() - t0
    render_steps = W * shards_per_col
    t_total = time.perf_counter() - t_frame
    print(
        f"  render   {render_steps} steps  kv={_kv_len(past)}  "
        f"{t_render*1000:.0f}ms ({t_render/render_steps*1000:.1f}ms/step)  "
        f"total={t_total*1000:.0f}ms"
    )

    new_state = GameState(
        x=px, y=py,
        angle=round(new_angle_raw) % 256,
        move_speed=state.move_speed,
        turn_speed=state.turn_speed,
    )
    return frame, new_state
