"""Compile and run the walls-as-tokens game graph.

Multi-phase autoregressive rollout:

    Phase -1 — Tex:     TEX_COL×(num_tex × tex_w)  (texture column prefill)
    Phase 0  — Prefill: INPUT + WALL×N + EOS        (host-driven)
    Phase 1  — Sort:    SORTED_WALL×N               (pure autoregressive)
    Phase 2  — Render:  RENDER×(dynamic)             (autoregressive, pixels out)

The host is a dumb token feeder and pixel bitblitter.  It feeds player
state at every position, wall geometry at WALL positions, the
accumulated mask at SORTED_WALL positions, and render feedback
at RENDER positions.  The only host-side logic is copying feedback
from each render output to the next input and bitblitting pixels
to the framebuffer with skip-filled compositing.
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
    E8_RENDER,
    E8_SORTED_WALL,
    E8_TEX_COL,
    E8_THINKING,
    E8_WALL,
    TEX_E8_OFFSET,
    build_game_graph,
)
from torchwright.graph.spherical_codes import index_to_vector
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
          1 + n_thresholds + d_value = 1 + max_walls + (13 + max_walls)
    - Render (attend_argmin_unmasked):
          1 + max_walls + (8 + max_walls)
    - TEX_COL (attend_argmax_dot): 8 + tex_w + 1
    """
    d_sort_val = 13 + max_walls
    sort_d = 1 + max_walls + d_sort_val
    render_d = 1 + max_walls + 8 + max_walls
    tex_d = 8 + tex_w + 1
    return max(sort_d, render_d, tex_d)


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

    # Collect Assert nodes *before* compile strips them so callers can
    # later run their predicates against the compiled residual stream
    # via ``check_asserts_on_compiled`` without rebuilding the graph.
    from torchwright.graph.asserts import collect_asserts

    collected_asserts: list = []
    _seen_ids: set = set()
    for out_node in list(graph_io.overlaid_outputs.values()) + list(
        graph_io.overflow_outputs.values()
    ):
        for a in collect_asserts(out_node):
            if a.node_id not in _seen_ids:
                _seen_ids.add(a.node_id)
                collected_asserts.append(a)

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
            # List[Assert] captured pre-strip so callers can run
            # compiled-side predicate checks via
            # ``torchwright.debug.probe.check_asserts_on_compiled``.
            "asserts": collected_asserts,
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
    d_render_fb = 2 * max_walls_meta + 11
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
        "render_feedback": torch.zeros(d_render_fb, device=device),
        "sort_feedback": torch.zeros(8 + 5 + 3 + max_walls_meta, device=device),
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
        4. EOS — feed player position → read resolved state from output
        5. SORTED_WALL×N — pure autoregressive: output IS next input
        6. THINKING/RENDER×(dynamic) — autoregressive: output IS next
           input.  THINKING selects wall, RENDER renders pixels.  Host
           reads pixels from overflow region, bitblits to framebuffer.

    For autoregressive phases (sort + render), the output tensor's first
    d_input values are laid out at input field offsets.  The host feeds
    ``output[:d_input]`` directly as the next input — no remapping.
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
    d_sort_out = 8 + 5 + 3 + max_walls
    d_render_fb = 2 * max_walls + 11
    device = module._net.device

    # Field offsets resolved by name — host stays layout-agnostic.
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}

    # Input-side offsets (overlaid fields' write targets on the next input)
    tt_off, _ = in_by_name["token_type"]
    sf_off, _ = in_by_name["sort_feedback"]
    rf_off, _ = in_by_name["render_feedback"]

    # Output-side offsets in the gathered output tensor
    sf_out_s, _ = out_by_name["sort_feedback"]
    rf_out_s, _ = out_by_name["render_feedback"]
    pix_out_s, _ = out_by_name["pixels"]
    col_out_s, _ = out_by_name["col"]
    start_out_s, _ = out_by_name["start"]
    length_out_s, _ = out_by_name["length"]
    done_out_s, _ = out_by_name["done"]

    # Overlaid fields: any name appearing in both input and output specs.
    # The host copies these from output to input each step (delta transfer
    # places them at the same columns in the residual stream, but the
    # gathered output tensor has its own running layout, so we copy
    # explicitly).
    overlaid_names = [n for n, _s, _w in module._input_specs if n in out_by_name]

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
    # All prefill tokens are causally ordered and don't depend on each
    # other's outputs, so we process them in a single batched call.
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

    # BSP_NODE × max_bsp_nodes — real planes first, pad with null planes.
    # The null plane (nx=0, ny=0, d=0) produces raw=0 at any player
    # position, so side_P = compare(0,0) is BACK (cond_gate emits zero),
    # and the padding contributes nothing to any wall's rank.
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

    # WALL × N — now each wall carries BSP rank coefficients precomputed
    # by the host.  Rank = dot(coeffs, side_P_vec) + const is evaluated
    # in the graph.
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

    # Read collision-resolved state from EOS output (last row).
    # sort_feedback begins with E8_SORTED_WALL(8), then resolved x, y, angle.
    eos_out = out[-1:]  # (1, d_output)
    px = eos_out[0, sf_out_s + 8].item()
    py = eos_out[0, sf_out_s + 9].item()
    new_angle_raw = eos_out[0, sf_out_s + 10].item()
    angle = new_angle_raw

    def _out_to_input(raw_out):
        """Map overlaid output fields to a flat input row.

        Every name present in both input and output specs is overlaid —
        iterate the intersection rather than naming fields explicitly so
        the host stays layout-agnostic.
        """
        row = torch.zeros(1, d_input, device=device)
        for name in overlaid_names:
            in_s, w = in_by_name[name]
            out_s, _ = out_by_name[name]
            row[0, in_s : in_s + w] = raw_out[0, out_s : out_s + w]
        return row

    # --- Phase 1: Sort ---
    prev = _out_to_input(eos_out)
    t0 = time.perf_counter()
    for k in range(N):
        out, past = _step(prev, past, step)
        step += 1
        prev = _out_to_input(out)
        # Diagnostics: read sort_feedback from compact output
        raw_sf = out[0, sf_out_s : sf_out_s + d_sort_out].cpu().numpy()
        wall_data = raw_sf[8:13]
        s_rank = raw_sf[13]
        s_col_lo = raw_sf[14]
        s_col_hi = raw_sf[15]
        onehot = raw_sf[16 : 16 + max_walls]
        print(
            f"  sort[{k}]: wall=[{wall_data[0]:.2f},{wall_data[1]:.2f},{wall_data[2]:.2f},{wall_data[3]:.2f}] "
            f"tex={wall_data[4]:.1f} rank={s_rank:.0f} cols=[{s_col_lo:.1f},{s_col_hi:.1f}) oh_max={np.argmax(onehot)}"
        )

    t_sort = time.perf_counter() - t0
    print(f"  sort     {N} steps  kv={_kv_len(past)}  {t_sort*1000:.0f}ms")
    print(f"  resolved state: px={px:.3f} py={py:.3f} angle={angle:.3f}")

    # --- Phase 2: Render (THINKING + RENDER interleaved) ---
    # THINKING tokens select a wall.  RENDER tokens render pixels.
    # The graph decides the next token type in the output.
    t0 = time.perf_counter()
    frame = np.full((H, W, 3), -1.0, dtype=np.float32)
    filled = np.zeros((H, W), dtype=bool)

    # Seed: E8_THINKING + render_feedback with is_new_wall=+1, chunk_start=-1
    prev = torch.zeros(1, d_input, device=device)
    prev[0, tt_off : tt_off + 8] = E8_THINKING.to(device)
    seed_rf = torch.zeros(d_render_fb, device=device)
    seed_rf[max_walls + 1] = 1.0  # is_new_wall = +1
    seed_rf[max_walls + 2] = -1.0  # chunk_start = sentinel
    prev[0, rf_off : rf_off + d_render_fb] = seed_rf

    max_render_steps = N * W * (H // cs + 1) + N + 10
    render_steps = 0

    for k in range(max_render_steps):
        out, past = _step(prev, past, step)
        step += 1
        render_steps += 1
        raw = out[0].detach().cpu().numpy()

        # Pixels and metadata from compact output
        col = int(round(raw[col_out_s]))
        start_y = int(round(raw[start_out_s]))
        length = int(round(raw[length_out_s]))
        done = raw[done_out_s]
        pix = raw[pix_out_s : pix_out_s + cs * 3].reshape(cs, 3)

        # Bitblit with skip-filled (length=0 for THINKING tokens → no-op)
        for row_idx in range(length):
            y = start_y + row_idx
            if 0 <= y < H and 0 <= col < W and not filled[y, col]:
                frame[y, col] = pix[row_idx]
                filled[y, col] = True

        if done > 0.0:
            break

        # Map compact output to input layout
        prev = _out_to_input(out)

        # Host-side termination: check render_mask from render_feedback
        mask_vals = out[0, rf_out_s : rf_out_s + max_walls].cpu().numpy()
        n_masked = int(np.sum(np.round(mask_vals).clip(0, 1)))
        if n_masked >= N:
            break

    # Fill unfilled pixels with ceiling/floor
    ceil = np.array(config.ceiling_color, dtype=np.float32)
    floor_c = np.array(config.floor_color, dtype=np.float32)
    center_y = H // 2
    for c in range(W):
        for y in range(H):
            if not filled[y, c]:
                frame[y, c] = ceil if y < center_y else floor_c

    t_render = time.perf_counter() - t0
    t_total = time.perf_counter() - t_frame
    print(
        f"  render   {render_steps} steps  kv={_kv_len(past)}  "
        f"{t_render*1000:.0f}ms ({t_render/render_steps*1000:.1f}ms/step)  "
        f"total={t_total*1000:.0f}ms"
    )

    new_state = GameState(
        x=px,
        y=py,
        angle=round(new_angle_raw) % 256,
        move_speed=state.move_speed,
        turn_speed=state.turn_speed,
    )
    return frame, new_state
