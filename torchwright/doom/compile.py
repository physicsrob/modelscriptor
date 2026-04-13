"""Compile and run the walls-as-tokens game graph.

Multi-phase autoregressive rollout:

    Phase -1 — Tex:     TEX_COL×(num_tex × tex_w)  (texture column prefill)
    Phase 0  — Prefill: INPUT + WALL×N + EOS        (host-driven)
    Phase 1  — Sort:    SORTED_WALL×N               (pure autoregressive)
    Phase 2  — Render:  RENDER×(W × H/rp)           (autoregressive, pixels out)

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
    E8_INPUT,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
    build_game_graph,
)
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.reference_renderer.types import RenderConfig, Segment


def print_graph_stats(output_node, pos_encoding=None):
    """Print a breakdown of graph nodes and params by annotation."""
    from collections import defaultdict
    from torchwright.compiler.utils import get_ancestor_nodes

    start = {output_node}
    if pos_encoding is not None:
        start.add(pos_encoding)
    all_nodes = get_ancestor_nodes(start)

    stats = defaultdict(lambda: {"nodes": 0, "params": 0})
    for node in all_nodes:
        key = node.annotation or "(none)"
        stats[key]["nodes"] += 1
        stats[key]["params"] += node.num_params()

    total_nodes = sum(s["nodes"] for s in stats.values())
    total_params = sum(s["params"] for s in stats.values())

    # Sort by top-level group, then sub-path
    rows = sorted(stats.items())

    print(f"\nGraph stats: {total_nodes:,} nodes, {total_params:,} params\n")
    print(f"  {'Annotation':<35s} {'Nodes':>7s} {'Params':>12s} {'% params':>9s}")
    print(f"  {'─' * 35} {'─' * 7} {'─' * 12} {'─' * 9}")

    # Group by top-level for subtotals
    from itertools import groupby
    def top_level(item):
        return item[0].split("/")[0]

    for group_key, group_items in groupby(rows, key=top_level):
        group_list = list(group_items)
        for key, s in group_list:
            pct = 100.0 * s["params"] / total_params if total_params else 0
            print(f"  {key:<35s} {s['nodes']:>7,} {s['params']:>12,} {pct:>8.1f}%")
        if len(group_list) > 1:
            gn = sum(s["nodes"] for _, s in group_list)
            gp = sum(s["params"] for _, s in group_list)
            gpct = 100.0 * gp / total_params if total_params else 0
            print(f"  {'  ' + group_key + ' (total)':<35s} {gn:>7,} {gp:>12,} {gpct:>8.1f}%")
        print()

    print(f"  {'TOTAL':<35s} {total_nodes:>7,} {total_params:>12,} {'100.0%':>9s}")
    print()


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
    device: str = "auto",
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
    # d_head must be >= max d_qk across all Attn nodes.
    # Render attention: d_qk = W + 1.  TEX_COL attention: d_qk = 8 + tex_w + 1.
    if d_head is None:
        W = config.screen_width
        tex_w = textures[0].shape[0]
        tex_d_qk = 8 + tex_w + 1
        render_d_qk = W + 2  # col_onehot(W) + bias(1) + tiebreak(1)
        min_d_head = max(render_d_qk, tex_d_qk)
        d_head = 1
        while d_head < min_d_head:
            d_head *= 2
        assert d % d_head == 0, f"d={d} not divisible by d_head={d_head}"
    rp = rows_per_patch if rows_per_patch is not None else config.screen_height
    tex_h = textures[0].shape[1]
    module = compile_headless(
        output_node, pos_encoding,
        d=d, d_head=d_head, max_layers=400,
        device=device, verbose=verbose,
        extra_metadata={
            "rows_per_patch": rp,
            "max_walls": max_walls,
            "tex_h": tex_h,
        },
        d_hidden=d_hidden,
    )
    module.eval()
    return module


def _build_row(compiled, max_walls, **kwargs):
    """Build a (1, d_input) row for module.step()."""
    device = compiled._net.device
    tex_h = int(compiled.metadata.get("tex_h", 8))
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
        "sort_feedback": torch.zeros(8 + 5 + 2 * max_walls, device=device),
        "tex_col_input": torch.zeros(1, device=device),
        "tex_pixels": torch.zeros(tex_h * 3, device=device),
        "texture_id_e8": torch.zeros(8, device=device),
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
    textures: List[np.ndarray] = None,
) -> Tuple[np.ndarray, GameState]:
    """Run one frame via the multi-phase rollout.

    Args:
        module: Compiled module from :func:`compile_game`.
        state: Current game state (x, y, angle).
        inputs: Player inputs for this frame.
        walls: List of wall dicts with keys ax, ay, bx, by, tex_id.
        config: Render configuration.
        textures: List of texture arrays, each (tex_w, tex_h, 3).

    Returns:
        ``(frame, new_state)`` where frame is ``(H, W, 3)`` float32.

    Host protocol — the host is a dumb token feeder:
        0. TEX_COL×(num_tex × tex_w) — feed texture column pixel data
        1. INPUT — feed player state + controls
        2. WALL×N — feed wall geometry + player position
        3. EOS — feed player position → read resolved state from output
        4. SORTED_WALL×N — pure autoregressive: feed prev output back
        5. RENDER×(W×H/rp) — feed (col, patch) → read pixels

    Every cross-position dependency flows through attention.  The host
    never computes on intermediate results — it just forwards outputs
    back as inputs for the sort phase.
    """
    assert textures is not None, "textures is required"
    N = len(walls)
    max_walls = int(module.metadata.get("max_walls", 8))
    rp = int(module.metadata.get("rows_per_patch", config.screen_height))
    H = config.screen_height
    W = config.screen_width
    shards_per_col = H // rp
    num_tex = len(textures)
    tex_w = textures[0].shape[0]

    assert N <= max_walls, f"Too many walls ({N}) for max_walls={max_walls}"
    assert H % rp == 0

    past = module.empty_past()
    step = 0
    total_steps = num_tex * tex_w + 1 + N + 1 + N + W * shards_per_col
    px, py, angle = float(state.x), float(state.y), float(state.angle)

    # Output layout indices
    pixel_sl = slice(8, 8 + rp * 3)

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

    # --- Batched prefill: TEX_COL + INPUT + WALL + EOS in one forward ---
    # All prefill tokens are causally ordered and don't depend on each
    # other's outputs, so we process them in a single batched call.
    t0 = time.perf_counter()
    rows = []

    # TEX_COL × (num_tex × tex_w)
    for tex_idx in range(num_tex):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = textures[tex_idx][col].flatten()
            rows.append(_common(
                token_type=E8_TEX_COL,
                texture_id_e8=tex_e8,
                tex_col_input=torch.tensor([float(col)]),
                tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
            ))

    # INPUT (controls only here)
    rows.append(_common(token_type=E8_INPUT, **input_kw))

    # WALL × N
    for i, w in enumerate(walls):
        rows.append(_common(
            token_type=E8_WALL,
            wall_ax=torch.tensor([w["ax"]]),
            wall_ay=torch.tensor([w["ay"]]),
            wall_bx=torch.tensor([w["bx"]]),
            wall_by=torch.tensor([w["by"]]),
            wall_tex_id=torch.tensor([w["tex_id"]]),
            wall_index=torch.tensor([float(i)]),
        ))

    # EOS
    rows.append(_common(token_type=E8_EOS))

    prefill = torch.cat(rows, dim=0)  # (n_prefill, d_input)
    with torch.no_grad():
        out, past = module.step(prefill, past, past_len=0)
    step = prefill.shape[0]

    t_prefill = time.perf_counter() - t0
    print(f"  prefill  {step} steps  kv={_kv_len(past)}  {t_prefill*1000:.0f}ms")

    # Read collision-resolved state from EOS output (last row)
    eos_out = out[-1:]  # (1, d_output)
    px = eos_out[0, 8].item()
    py = eos_out[0, 9].item()
    new_angle_raw = eos_out[0, 10].item()
    angle = new_angle_raw

    # --- Phase 1: Sort ---
    # Pure autoregressive sort: EOS output seeds the loop, each step's
    # output feeds back as the next input.  The transformer owns the mask
    # lifecycle — the host does zero computation on intermediate results.
    d_sf = 8 + 5 + 2 * max_walls
    prev = eos_out[0, :d_sf].detach().clone()  # EOS output seeds sort
    t0 = time.perf_counter()
    for k in range(N):
        row = _build_row(
            module, max_walls,
            token_type=prev[:8],
            sort_feedback=prev,
        )
        out, past = _step(row, past, step)
        step += 1
        prev = out[0, :d_sf].detach().clone()
        raw_sort = prev.cpu().numpy()
        wall_data = raw_sort[8:13]
        onehot = raw_sort[13:13 + max_walls]
        print(f"  sort[{k}]: wall=[{wall_data[0]:.2f},{wall_data[1]:.2f},{wall_data[2]:.2f},{wall_data[3]:.2f}] "
              f"tex={wall_data[4]:.1f} oh_max={np.argmax(onehot)}")

    t_sort = time.perf_counter() - t0
    print(f"  sort     {N} steps  kv={_kv_len(past)}  {t_sort*1000:.0f}ms")
    print(f"  resolved state: px={px:.3f} py={py:.3f} angle={angle:.3f}")

    # --- Phase 2: Render ---
    # RENDER positions read wall data from SORTED KV and texture data
    # from TEX_COL KV.  No player state needed — just col/patch indices.
    t0 = time.perf_counter()
    frame = np.zeros((H, W, 3), dtype=np.float32)
    _diag_printed = False
    for col in range(W):
        for shard in range(shards_per_col):
            row = _build_row(
                module, max_walls,
                token_type=E8_RENDER,
                col_idx=torch.tensor([float(col)]),
                patch_idx=torch.tensor([float(shard)]),
            )
            out, past = _step(row, past, step)
            step += 1
            raw_out = out[0].detach().cpu().numpy()
            if not _diag_printed and col == W // 2 and shard == 0:
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
