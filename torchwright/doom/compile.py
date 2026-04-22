"""Compile and run the walls-as-tokens game graph.

Multi-phase autoregressive rollout:

    Phase -1 — Tex:     TEX_COL×(num_tex × tex_w)  (texture column prefill)
    Phase 0  — Prefill: INPUT + BSP_NODE×M + WALL×N + EOS
    Phase 0b — Player:  PLAYER_X + PLAYER_Y + PLAYER_ANGLE
    Phase 1+2 — Sort+Render (interleaved):
        SORTED_WALL → RENDER×k → SORTED_WALL → RENDER×k → ...

The host is a dumb token feeder and pixel bitblitter.  It seeds the
first SORTED_WALL token, then loops: step the transformer, blit any
pixels from overflow output, copy overlaid output to next input.  The
transformer controls the token sequence via the ``token_type`` overlaid
output — SORTED emits E8_RENDER, RENDER emits E8_SORTED_WALL on wall
transitions.  The host never inspects token type, never caches wall
data, never patches inputs.
"""

import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph.node import Node
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.game_graph import (
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_PLAYER_ANGLE,
    E8_PLAYER_X,
    E8_PLAYER_Y,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
    build_game_graph,
)
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.doom.trace import FrameTrace, RenderStepTrace, SortStepTrace
from torchwright.reference_renderer.types import RenderConfig


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
            print(
                f"  {'  ' + group_key + ' (total)':<35s} {gn:>7,} {gp:>12,} {gpct:>8.1f}%"
            )
        print()

    print(f"  {'TOTAL':<35s} {total_nodes:>7,} {total_params:>12,} {'100.0%':>9s}")
    print()


def compute_min_d_head(max_walls: int, tex_w: int) -> int:
    """Minimum d_head required by the game graph's attention heads.

    Three attention patterns drive the requirement:
    - Sort (attend_argmin_above_integer):
          1 + n_thresholds + d_value = 1 + max_walls + (8 + max_walls)
    - Render wall geometry (attend_argmax_dot):
          max_walls + 4  (query/key = one-hot, value = geometry)
    - TEX_COL (attend_argmax_dot): 8 + tex_w
    """
    d_sort_val = 8 + max_walls
    sort_d = 1 + max_walls + d_sort_val
    render_geom_d = max_walls + 4
    tex_d = 8 + tex_w
    return max(sort_d, render_geom_d, tex_d)


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
    chunk_size: int = 20,
    d_hidden: Optional[int] = None,
    max_bsp_nodes: int = 48,
    optimize: bool = True,
):
    """Compile the game graph to a HeadlessTransformerModule.

    Args:
        optimize: Run graph optimization passes (Linear fusion) before
            compilation. Fuses consecutive Linear nodes to reduce layers.
    """
    graph_io, pos_encoding = build_game_graph(
        config,
        textures,
        max_walls,
        max_coord,
        move_speed,
        turn_speed,
        chunk_size=chunk_size,
        max_bsp_nodes=max_bsp_nodes,
    )

    # Run graph optimizations (Linear fusion)
    if optimize:
        from torchwright.graph.optimize import fuse_consecutive_linears

        # Collect all output nodes from both overlaid and overflow outputs
        output_nodes = set(graph_io.overlaid_outputs.values())
        output_nodes.update(graph_io.overflow_outputs.values())
        output_nodes.add(pos_encoding)
        total_fused = 0
        while True:
            fused = fuse_consecutive_linears(output_nodes, verbose=False)
            if fused == 0:
                break
            total_fused += fused
        if verbose and total_fused > 0:
            print(f"Optimized: fused {total_fused} Linear pairs")

    if d_head is None:
        tex_w = textures[0].shape[0]
        min_d_head = compute_min_d_head(max_walls, tex_w)
        d_head = 1
        while d_head < min_d_head:
            d_head *= 2
        assert d % d_head == 0, f"d={d} not divisible by d_head={d_head}"
    tex_h = textures[0].shape[1]

    # Build io dict: overlaid entries share a name between input and
    # output (compiler places the output at the input's columns via
    # delta transfer).  Input-only entries have no output node.  Overflow
    # entries have no input and live in overflow columns after d_input.
    io: dict[str, tuple[Node | None, Node | None]] = {}
    for name, node in graph_io.inputs.items():
        overlaid_out: Node | None = graph_io.overlaid_outputs.get(name)
        io[name] = (node, overlaid_out)
    for name, node in graph_io.overflow_outputs.items():
        assert name not in io, f"overflow name collides with input: {name}"
        io[name] = (None, node)

    module = compile_headless(
        pos_encoding,
        io=io,
        d=d,
        d_head=d_head,
        max_layers=400,
        device=device,
        verbose=verbose,
        extra_metadata={
            "chunk_size": chunk_size,
            "max_walls": max_walls,
            "max_bsp_nodes": max_bsp_nodes,
            "tex_h": tex_h,
            "overflow_names": list(graph_io.overflow_outputs),
        },
        d_hidden=d_hidden,
    )
    module.eval()
    return module


def _build_row(compiled, max_walls, **kwargs):
    """Build a (1, d_input) row for module.step()."""
    device = compiled._net.device
    tex_h = int(compiled.metadata.get("tex_h", 8))
    max_walls_meta = int(compiled.metadata.get("max_walls", 8))
    max_bsp_nodes = int(compiled.metadata.get("max_bsp_nodes", 48))
    defaults = {
        "input_backward": torch.zeros(1, device=device),
        "input_forward": torch.zeros(1, device=device),
        "input_strafe_left": torch.zeros(1, device=device),
        "input_strafe_right": torch.zeros(1, device=device),
        "input_turn_left": torch.zeros(1, device=device),
        "input_turn_right": torch.zeros(1, device=device),
        "player_angle": torch.zeros(1, device=device),
        "player_x": torch.zeros(1, device=device),
        "player_y": torch.zeros(1, device=device),
        "render_col": torch.zeros(1, device=device),
        "render_chunk_k": torch.zeros(1, device=device),
        "wall_counter": torch.zeros(1, device=device),
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
        # BSP-related inputs (zero at positions that don't use them).
        "bsp_plane_nx": torch.zeros(1, device=device),
        "bsp_plane_ny": torch.zeros(1, device=device),
        "bsp_plane_d": torch.zeros(1, device=device),
        "bsp_node_id_onehot": torch.zeros(max_bsp_nodes, device=device),
        "wall_bsp_coeffs": torch.zeros(max_bsp_nodes, device=device),
        "wall_bsp_const": torch.zeros(1, device=device),
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
        row[:, start : start + width] = v.to(device)
    return row


def step_frame(
    module,
    state: GameState,
    inputs: PlayerInput,
    subset,
    config: RenderConfig,
    textures: Optional[List[np.ndarray]] = None,
    trace: Optional[FrameTrace] = None,
) -> Tuple[np.ndarray, GameState]:
    """Run one frame via the multi-phase rollout.

    Args:
        module: Compiled module from :func:`compile_game`.
        state: Current game state (x, y, angle).
        inputs: Player inputs for this frame.
        subset: :class:`~torchwright.doom.map_subset.MapSubset` carrying
            segments, BSP planes, and precomputed rank coefficients.
            Build one with :func:`build_scene_subset` (hand-authored
            scenes) or :func:`load_map_subset` (WAD maps).
        config: Render configuration.
        textures: List of texture arrays, each (tex_w, tex_h, 3).
            Defaults to the subset's textures.

    Returns:
        ``(frame, new_state)`` where frame is ``(H, W, 3)`` float32.

    Host protocol — the host is a dumb token feeder + bitblitter:
        0. TEX_COL×(num_tex × tex_w) — feed texture column pixel data
        1. INPUT — feed player state + controls
        2. BSP_NODE×max_bsp_nodes — feed splitting plane coefficients
        3. WALL×N — feed wall geometry + BSP rank coefficients
        4. EOS — feed player position → read resolved state from overflow
        5. PLAYER_X/Y/ANGLE — feed resolved state
        6. SORTED_WALL×N — host feeds position_index, caches sorted wall data
        7. RENDER×(dynamic) — host copies overlaid fields, detects wall
           transitions, feeds next wall identity.  Reads pixels from
           overflow and bitblits to framebuffer.
    """
    max_walls = int(module.metadata.get("max_walls", 8))
    cs = int(module.metadata.get("chunk_size", 20))
    max_bsp_nodes = int(module.metadata.get("max_bsp_nodes", 48))

    if textures is None:
        textures = subset.textures

    # The prefill loop below iterates walls as dicts; convert once.
    walls = [
        {"ax": s.ax, "ay": s.ay, "bx": s.bx, "by": s.by, "tex_id": float(s.texture_id)}
        for s in subset.segments
    ]

    N = len(walls)
    H = config.screen_height
    W = config.screen_width
    num_tex = len(textures)
    tex_w = textures[0].shape[0]

    assert N <= max_walls, f"Too many walls ({N}) for max_walls={max_walls}"

    past = module.empty_past()
    step = 0
    px, py, angle = float(state.x), float(state.y), float(state.angle)

    # Compute layout from compiled module
    d_input = max(s + w for _, s, w in module._input_specs)
    device = module._net.device

    # Field offsets resolved by name — host stays layout-agnostic.
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}

    # Output-side offsets in the gathered output tensor
    pix_out_s, _ = out_by_name["pixels"]
    col_out_s, _ = out_by_name["col"]
    start_out_s, _ = out_by_name["start"]
    length_out_s, _ = out_by_name["length"]
    done_out_s, _ = out_by_name["done"]
    sort_done_out_s, _ = out_by_name["sort_done"]
    eos_rx_out_s, _ = out_by_name["eos_resolved_x"]
    eos_ry_out_s, _ = out_by_name["eos_resolved_y"]
    eos_angle_out_s, _ = out_by_name["eos_new_angle"]

    # Overlaid fields: any name appearing in both input and output specs.
    overlaid_names = [n for n, _s, _w in module._input_specs if n in out_by_name]

    # Player input tensors
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
            module,
            max_walls,
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

    # --- Batched prefill: TEX_COL + INPUT + BSP_NODE + WALL + EOS ---
    t0 = time.perf_counter()
    rows = []

    # TEX_COL × (num_tex × tex_w)
    for tex_idx in range(num_tex):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = textures[tex_idx][col].flatten()
            rows.append(
                _common(
                    token_type=E8_TEX_COL,
                    texture_id_e8=tex_e8,
                    tex_col_input=torch.tensor([float(col)]),
                    tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                )
            )

    # INPUT (controls only here)
    rows.append(_common(token_type=E8_INPUT, **input_kw))

    # BSP_NODE × max_bsp_nodes
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            plane = subset.bsp_nodes[i]
            nx, ny, d = plane.nx, plane.ny, plane.d
        else:
            nx, ny, d = 0.0, 0.0, 0.0
        rows.append(
            _common(
                token_type=E8_BSP_NODE,
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
            )
        )

    # WALL × N
    for i, w in enumerate(walls):
        if i < subset.seg_bsp_coeffs.shape[0]:
            coeffs = torch.tensor(
                subset.seg_bsp_coeffs[i, :max_bsp_nodes],
                dtype=torch.float32,
            )
            const = torch.tensor(
                [float(subset.seg_bsp_consts[i])],
                dtype=torch.float32,
            )
        else:
            coeffs = torch.zeros(max_bsp_nodes, dtype=torch.float32)
            const = torch.zeros(1, dtype=torch.float32)
        rows.append(
            _common(
                token_type=E8_WALL,
                wall_ax=torch.tensor([w["ax"]]),
                wall_ay=torch.tensor([w["ay"]]),
                wall_bx=torch.tensor([w["bx"]]),
                wall_by=torch.tensor([w["by"]]),
                wall_tex_id=torch.tensor([w["tex_id"]]),
                wall_index=torch.tensor([float(i)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
            )
        )

    # EOS
    rows.append(_common(token_type=E8_EOS))

    prefill = torch.cat(rows, dim=0)  # (n_prefill, d_input)
    with torch.no_grad():
        out, past = module.step(prefill, past, past_len=0)
    step = prefill.shape[0]

    t_prefill = time.perf_counter() - t0
    print(f"  prefill  {step} steps  kv={_kv_len(past)}  {t_prefill*1000:.0f}ms")

    # Read collision-resolved state from EOS overflow outputs.
    eos_out = out[-1:]  # (1, d_output)
    px = eos_out[0, eos_rx_out_s].item()
    py = eos_out[0, eos_ry_out_s].item()
    new_angle_raw = eos_out[0, eos_angle_out_s].item()
    angle = new_angle_raw

    if trace is not None:
        trace.eos_resolved_x = px
        trace.eos_resolved_y = py
        trace.eos_new_angle = new_angle_raw

    def _out_to_input(raw_out):
        """Map overlaid output fields to a flat input row."""
        row = torch.zeros(1, d_input, device=device)
        for name in overlaid_names:
            in_s, w = in_by_name[name]
            out_s, _ = out_by_name[name]
            row[0, in_s : in_s + w] = raw_out[0, out_s : out_s + w]
        return row

    # --- Phase 0b: Player state tokens ---
    t0 = time.perf_counter()
    player_row_x = _build_row(
        module,
        max_walls,
        token_type=E8_PLAYER_X,
        player_x=torch.tensor([px]),
    )
    out, past = _step(player_row_x, past, step)
    step += 1

    player_row_y = _build_row(
        module,
        max_walls,
        token_type=E8_PLAYER_Y,
        player_y=torch.tensor([py]),
    )
    out, past = _step(player_row_y, past, step)
    step += 1

    player_row_angle = _build_row(
        module,
        max_walls,
        token_type=E8_PLAYER_ANGLE,
        player_angle=torch.tensor([angle]),
    )
    out, past = _step(player_row_angle, past, step)
    step += 1

    t_player = time.perf_counter() - t0
    print(f"  player   3 steps  kv={_kv_len(past)}  {t_player*1000:.0f}ms")

    # --- Sort + Render (interleaved, pure bitblit) ---
    #
    # The transformer controls the token sequence: SORTED_WALL picks a
    # wall and emits token_type=E8_RENDER; RENDER tokens paint pixels
    # and emit token_type=E8_SORTED_WALL on wall transitions.  The host
    # just copies overlaid outputs to the next input (_out_to_input) and
    # blits pixels to the framebuffer.  No sorting cache, no advance_wall
    # detection, no wall-identity patching.

    # Output field offsets for trace recording.
    # Phase A M3: wall identity is no longer an overlaid input; RENDER
    # reads it from the KV cache via attention.  The SORTED stage still
    # emits its picked wall index as an overflow field ("sort_wall_index")
    # so the trace harness can record which wall the transformer chose
    # at each sort step — this field is host-visible but not fed back.
    wi_out_s, _ = out_by_name["sort_wall_index"]
    wc_out_s, _ = out_by_name["wall_counter"]
    col_overlay_s, _ = out_by_name["render_col"]
    svhi_out_s, _ = out_by_name["sort_vis_hi"]
    t0 = time.perf_counter()
    frame = np.full((H, W, 3), -1.0, dtype=np.float32)
    filled = np.zeros((H, W), dtype=bool)

    prev = _build_row(
        module,
        max_walls,
        token_type=E8_SORTED_WALL,
        wall_counter=torch.tensor([0.0]),
    )

    max_steps = N * (W * (H // cs + 1) + 1) + 10
    total_steps = 0
    prev_wc = 0.0

    for k in range(max_steps):
        out, past = _step(prev, past, step)
        step += 1
        total_steps += 1
        raw = out[0].detach().cpu().numpy()

        # Read overflow outputs (zero/neg at non-RENDER positions).
        col = int(round(raw[col_out_s]))
        start_y = int(round(raw[start_out_s]))
        length = int(round(raw[length_out_s]))
        done = raw[done_out_s]
        sort_done = raw[sort_done_out_s]
        pix = raw[pix_out_s : pix_out_s + cs * 3].reshape(cs, 3)

        # Trace: detect token type from wall_counter changes.
        # SORTED tokens increment wall_counter; RENDER tokens forward it.
        if trace is not None:
            cur_wc = raw[wc_out_s]
            if cur_wc > prev_wc + 0.5:
                wall_idx = int(round(raw[wi_out_s]))
                trace.sort_steps.append(
                    SortStepTrace(
                        position_index=len(trace.sort_steps),
                        wall_j_onehot=np.eye(max_walls)[wall_idx],
                        selected_wall_index=wall_idx,
                        vis_lo=raw[col_overlay_s],
                        vis_hi=raw[svhi_out_s],
                        tex_id=0.0,
                        sort_done=sort_done > 0.0,
                    )
                )
                if sort_done <= 0.0:
                    trace.n_renderable = len(trace.sort_steps)
            if length > 0:
                sort_idx = max(0, int(round(cur_wc)) - 1)
                trace.render_steps.append(
                    RenderStepTrace(
                        col=col,
                        start=start_y,
                        length=length,
                        pixels=pix.copy(),
                        done=done > 0.0,
                        wall_index=sort_idx,
                    )
                )
            prev_wc = cur_wc

        # Bitblit pixels (length=0 at SORTED positions → no-op).
        for row_idx in range(length):
            y = start_y + row_idx
            if 0 <= y < H and 0 <= col < W and not filled[y, col]:
                frame[y, col] = pix[row_idx]
                filled[y, col] = True

        if done > 0.0 or sort_done > 0.0:
            break

        prev = _out_to_input(out)

    print(f"  resolved state: px={px:.3f} py={py:.3f} angle={angle:.3f}")

    # Fill unfilled pixels with ceiling/floor
    # (Known violation of dumb-host principle — rendering logic that
    # belongs in the transformer.  Tracked separately.)
    ceil = np.array(config.ceiling_color, dtype=np.float32)
    floor_c = np.array(config.floor_color, dtype=np.float32)
    center_y = H // 2
    for c in range(W):
        for y in range(H):
            if not filled[y, c]:
                frame[y, c] = ceil if y < center_y else floor_c

    t_render = time.perf_counter() - t0
    t_total = time.perf_counter() - t_frame
    if total_steps > 0:
        print(
            f"  sort+render {total_steps} steps  kv={_kv_len(past)}  "
            f"{t_render*1000:.0f}ms ({t_render/total_steps*1000:.1f}ms/step)  "
            f"total={t_total*1000:.0f}ms"
        )
    else:
        print(f"  sort+render 0 steps  total={t_total*1000:.0f}ms")

    new_state = GameState(
        x=px,
        y=py,
        angle=round(new_angle_raw) % 256,
        move_speed=state.move_speed,
        turn_speed=state.turn_speed,
    )
    return frame, new_state
