"""Compile and run the walls-as-tokens game graph.

Multi-phase autoregressive rollout:

    Phase -1 — Tex:     TEX_COL×(num_tex × tex_w)  (texture column prefill)
    Phase 0  — Prefill: INPUT + BSP_NODE×M + WALL×N + EOS
    Phase 0b — Player:  PLAYER_X + PLAYER_Y + PLAYER_ANGLE
    Phase 1+2 — Sort+Render (interleaved):
        SORTED_WALL → RENDER×k → SORTED_WALL → RENDER×k → ...

The host is a dumb token feeder and pixel bitblitter.  Every
autoregressive step produces exactly one ``next_token_id`` (argmaxed
against ``W_EMBED.T`` inside :class:`CompiledToken`), which becomes
the next step's ``token_ids`` input.  Bypass fields (player state,
wall geometry, overlaid autoregressive state) flow alongside as a
separate dict.  The host never inspects token identity, never caches
wall data, never patches inputs.
"""

import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from torchwright.compiler.token_model import CompiledToken, compile_token
from torchwright.graph.node import Node
from torchwright.doom.embedding import (
    N_VALUES,
    V,
    VALUE_RANGE_BY_NAME,
    vocab_id,
)
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.game_graph import (
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
    render_pixels: bool = True,
):
    """Compile the game graph to a HeadlessTransformerModule.

    Args:
        optimize: Run graph optimization passes (Linear fusion) before
            compilation. Fuses consecutive Linear nodes to reduce layers.
        render_pixels: When False, skip the texture-attention and
            texture-coordinate blocks in RENDER (and the TEX_COL prefill
            in step_frame).  The state machine — chunked column iteration,
            wall heights, sort/render handoff — is unchanged, so the
            token stream still exercises the full autoregressive loop.
            Used by rollout tests that assert on tokens, not pixels.
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
        render_pixels=render_pixels,
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

    module = compile_token(
        pos_encoding,
        graph_io.embedding,
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
            "render_pixels": render_pixels,
        },
        d_hidden=d_hidden,
        token_id_input_name="token_ids",
        logit_output_name="next_token_embedding",
    )
    module.eval()
    return module


def _build_inputs(compiled, *, token_id: int, **kwargs) -> dict:
    """Build a single-position inputs dict for :meth:`CompiledToken.step`.

    ``token_id`` is the integer vocab ID for this position.  Every
    other bypass field defaults to zero at the module's declared
    width; pass ``**kwargs`` to override by name.

    ``compiled`` may be a :class:`CompiledToken` or a
    :class:`CompiledHeadless` — this helper only needs the shared
    ``device`` / ``metadata`` / ``input_names`` surface, so the
    annotation stays duck-typed for test / script reuse.
    """
    device = compiled.device
    tex_h = int(compiled.metadata.get("tex_h", 8))
    max_bsp_nodes = int(compiled.metadata.get("max_bsp_nodes", 48))
    defaults = {
        "token_ids": torch.tensor([float(token_id)], device=device),
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
        "wall_ax": torch.zeros(1, device=device),
        "wall_ay": torch.zeros(1, device=device),
        "wall_bx": torch.zeros(1, device=device),
        "wall_by": torch.zeros(1, device=device),
        "wall_index": torch.zeros(1, device=device),
        "wall_tex_id": torch.zeros(1, device=device),
        "bsp_plane_nx": torch.zeros(1, device=device),
        "bsp_plane_ny": torch.zeros(1, device=device),
        "bsp_plane_d": torch.zeros(1, device=device),
        "bsp_node_id_onehot": torch.zeros(max_bsp_nodes, device=device),
        "wall_bsp_coeffs": torch.zeros(max_bsp_nodes, device=device),
        "wall_bsp_const": torch.zeros(1, device=device),
    }
    defaults.update(kwargs)
    # Normalise every field to (1, width) float32 on the target device.
    out: dict = {}
    for name in compiled.input_names:
        v = defaults[name]
        if isinstance(v, (int, float)):
            v = torch.tensor([v], device=device)
        v = v.to(device=device, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        out[name] = v
    return out


def _stack_inputs(rows: List[dict]) -> dict:
    """Concatenate a list of single-position inputs dicts along dim 0."""
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {k: torch.cat([r[k] for r in rows], dim=0) for k in keys}


def step_frame(
    module,
    state: GameState,
    inputs: PlayerInput,
    subset,
    config: RenderConfig,
    textures: Optional[List[np.ndarray]] = None,
    trace: Optional[FrameTrace] = None,
    stop_after_thinking: bool = False,
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
        stop_after_thinking: If ``True``, break the autoregressive
            loop as soon as the transformer emits the SORTED_WALL
            handoff token — i.e. skip the entire SORTED + RENDER
            phase.  The returned frame stays unfilled (ceiling /
            floor only); the trace's ``token_id_log`` still covers
            the full thinking phase + the SORTED_WALL token.  Use
            for thinking-token tests that only need the token-stream
            trace and not pixel output.

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
    render_pixels = bool(module.metadata.get("render_pixels", True))

    if textures is None:
        textures = subset.textures

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
    # Phase A Part 4: the host feeds pre-collision (frame-start)
    # player_x/y to PLAYER_X / PLAYER_Y; collision resolution now lives
    # in the thinking phase and the post-collision state is read from
    # the RESOLVED_X / RESOLVED_Y thinking tokens.  The pre-collision
    # bypass slot at thinking-token positions is gone — thinking_wall
    # reads player_x/y from the PLAYER broadcast (which is
    # pre-collision under the Part 4 semantics).

    # Host-visible overlaid bypass names (fields shared between input
    # and output specs).  The host copies each one's value from the
    # previous step's overflow back into the next step's bypass dict.
    input_names = set(module.input_names)
    output_names = set(module.output_names)
    overlaid_names = [n for n in module.input_names if n in output_names]

    # Player input tensors
    input_kw = dict(
        input_forward=torch.tensor([float(inputs.forward)]),
        input_backward=torch.tensor([float(inputs.backward)]),
        input_turn_left=torch.tensor([float(inputs.turn_left)]),
        input_turn_right=torch.tensor([float(inputs.turn_right)]),
        input_strafe_left=torch.tensor([float(inputs.strafe_left)]),
        input_strafe_right=torch.tensor([float(inputs.strafe_right)]),
    )

    def _common(token_id: int, **extra):
        return _build_inputs(
            module,
            token_id=token_id,
            player_x=torch.tensor([px]),
            player_y=torch.tensor([py]),
            player_angle=torch.tensor([angle]),
            **extra,
        )

    def _kv_len(past):
        return past[0][0].shape[1] if past[0][0].numel() > 0 else 0

    def _step(inputs_dict, past, step):
        with torch.no_grad():
            return module.step(inputs_dict, past, past_len=step)

    t_frame = time.perf_counter()

    # --- Batched prefill: TEX_COL + INPUT + BSP_NODE + WALL + EOS ---
    t0 = time.perf_counter()
    rows: List[dict] = []

    # TEX_COL × (num_tex × tex_w) — skipped when render_pixels=False
    # (the headless graph wires zero literals through the texture path
    # and TEX_COL prefill positions are unused).
    if render_pixels:
        for tex_idx in range(num_tex):
            tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
            for col in range(tex_w):
                pixel_data = textures[tex_idx][col].flatten()
                rows.append(
                    _common(
                        token_id=vocab_id("TEX_COL"),
                        texture_id_e8=tex_e8,
                        tex_col_input=torch.tensor([float(col)]),
                        tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                    )
                )

    # INPUT (controls only here)
    rows.append(_common(token_id=vocab_id("INPUT"), **input_kw))

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
                token_id=vocab_id("BSP_NODE"),
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
            )
        )

    # WALL × max_walls — pad beyond N real walls with zero geometry so
    # every thinking-wall position has a matching K at some WALL prefill
    # slot.  Without this, padded thinking positions' quad-equality
    # wall_geom_attention concentrates on the highest real WALL (nearest
    # by score), causing them to inherit that wall's is_renderable /
    # bsp_rank — which pollutes SORTED's softmax whenever the donor wall
    # is renderable.  With padding, each padded wall attends to its own
    # zero-geometry WALL, which fails the renderability gate
    # (``|sort_den| > 0.05`` on rotated (0,0)→(0,0)) and is sentinel-gated
    # out of SORTED.
    for i in range(max_walls):
        if i < len(walls):
            w = walls[i]
            ax, ay, bx, by, tex_id = w["ax"], w["ay"], w["bx"], w["by"], w["tex_id"]
        else:
            ax = ay = bx = by = tex_id = 0.0
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
                token_id=vocab_id("WALL"),
                wall_ax=torch.tensor([ax]),
                wall_ay=torch.tensor([ay]),
                wall_bx=torch.tensor([bx]),
                wall_by=torch.tensor([by]),
                wall_tex_id=torch.tensor([tex_id]),
                wall_index=torch.tensor([float(i)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
            )
        )

    # EOS
    rows.append(_common(token_id=vocab_id("EOS")))

    prefill = _stack_inputs(rows)
    _, _, past = _step(prefill, past, 0)
    step = prefill["token_ids"].shape[0]

    t_prefill = time.perf_counter() - t0
    print(f"  prefill  {step} steps  kv={_kv_len(past)}  {t_prefill*1000:.0f}ms")

    def _next_inputs_from_overflow(next_id_scalar: int, overflow_row: dict) -> dict:
        """Build a single-position inputs dict for the NEXT step.

        Copies overlaid bypass fields (render_col, render_chunk_k,
        wall_counter) from the previous step's overflow.  Post-Part-4
        the thinking-position player_x/y bypass is gone — the graph
        reads pre-collision player state from the PLAYER broadcast
        rather than a raw input slot, so default-zero is fine.
        """
        kwargs: dict = {}
        for name in overlaid_names:
            kwargs[name] = overflow_row[name][0]  # (width,) 1-D tensor
        return _build_inputs(module, token_id=int(next_id_scalar), **kwargs)

    # --- Phase 0b: Player state tokens ---
    # Host feeds pre-collision (frame-start) px, py to PLAYER_X /
    # PLAYER_Y.  PLAYER_ANGLE's ``player_angle`` input is ignored by
    # the player stage under Part 4 (trig is sourced from INPUT's
    # post-turn ``new_angle`` broadcast instead) — but we still feed
    # the pre-turn angle here so the declared input range isn't
    # violated.  PLAYER_ANGLE's next-token argmax yields
    # ``THINKING_WALL_0`` (the graph emits that at PLAYER_ANGLE
    # positions), which becomes the first token of the autoregressive
    # thinking loop — the host no longer synthesizes it.
    t0 = time.perf_counter()
    player_row_x = _build_inputs(
        module,
        token_id=vocab_id("PLAYER_X"),
        player_x=torch.tensor([px]),
    )
    _, _, past = _step(player_row_x, past, step)
    step += 1

    player_row_y = _build_inputs(
        module,
        token_id=vocab_id("PLAYER_Y"),
        player_y=torch.tensor([py]),
    )
    _, _, past = _step(player_row_y, past, step)
    step += 1

    player_row_angle = _build_inputs(
        module,
        token_id=vocab_id("PLAYER_ANGLE"),
        player_angle=torch.tensor([angle]),
    )
    next_ids_after_pa, overflow_after_pa, past = _step(player_row_angle, past, step)
    step += 1
    first_thinking_id = int(next_ids_after_pa[0].item())

    t_player = time.perf_counter() - t0
    print(f"  player   3 steps  kv={_kv_len(past)}  {t_player*1000:.0f}ms")

    # --- Thinking + Sort + Render (single autoregressive loop) ---
    #
    # Phase A Part 4: the PLAYER_ANGLE step emits ``THINKING_WALL_0``
    # as its next-token prediction (via the graph's next-token-embedding
    # output at PLAYER_ANGLE positions).  The host no longer
    # synthesizes the first thinking token — it just feeds the argmaxed
    # ID back in.  From there the transformer drives the full sequence:
    # marker → identifier → value → … → RESOLVED_X/Y/ANGLE →
    # SORTED_WALL → RENDER×k → …, until done.  The host loop is
    # identical across all phases — copy overlaid bypass overflow back
    # into the next bypass dict, blit pixels for any RENDER positions,
    # terminate on done/sort_done.

    t0 = time.perf_counter()
    frame = np.full((H, W, 3), -1.0, dtype=np.float32)
    filled = np.zeros((H, W), dtype=bool)

    prev = _next_inputs_from_overflow(first_thinking_id, overflow_after_pa)

    # Thinking phase budget (Phase B Part 1): per wall, 1 marker + 17 identifiers
    # + 17 values = 35 steps.  After the last wall, 3 RESOLVED identifiers
    # + 3 RESOLVED values = 6 steps.  Total = max_walls * 35 + 6, plus a
    # small safety margin.
    n_thinking = max_walls * 35 + 6 + 4
    # Per wall in the SORT/RENDER loop (Phase B Part 2): SORTED_WALL
    # marker + SORT_RESULT id + SORT_RESULT VALUE + up to RENDER tokens.
    max_steps = n_thinking + N * (W * (H // cs + 1) + 3) + 10
    total_steps = 0
    prev_wc = 0.0
    # Capture the token-stream trace as the test sees it: one entry per
    # autoregressive position T, holding the token at that position.
    # Position 0 is ``THINKING_WALL_0`` (the PLAYER_ANGLE step's
    # argmax); every subsequent position's ID is the previous step's
    # ``next_ids``.
    token_id_log: List[int] = [first_thinking_id]

    for k in range(max_steps):
        next_ids, overflow, past = _step(prev, past, step)
        step += 1
        total_steps += 1

        next_id_scalar = int(next_ids[0].item())
        token_id_log.append(next_id_scalar)

        col = int(round(overflow["col"][0, 0].item()))
        start_y = int(round(overflow["start"][0, 0].item()))
        length = int(round(overflow["length"][0, 0].item()))
        done = overflow["done"][0, 0].item()
        sort_done = overflow["sort_done"][0, 0].item()
        pix = overflow["pixels"][0].detach().cpu().numpy().reshape(cs, 3)

        # Trace: detect token type from wall_counter changes.
        # Phase B Part 2: wall_counter increments at the SORT_RESULT id
        # position (not at the SORTED_WALL marker anymore), so the
        # increment fires one token later than before.  ``render_col``
        # still carries ``vis_lo`` but it's seeded at the SORT_RESULT
        # VALUE position (one step AFTER the id).  The trace below is
        # only used by the walkthrough harness for sanity diagnostics;
        # the reference comparison itself goes through pixel output, so
        # we fill ``vis_lo`` / ``vis_hi`` / ``tex_id`` with 0.0 here
        # and let the pixel compare catch any divergence.
        if trace is not None:
            cur_wc = overflow["wall_counter"][0, 0].item()
            if cur_wc > prev_wc + 0.5:
                # Garbage picks (after sort exhaustion) can produce
                # nonsensical wall_idx values from soft-averaged
                # attention; clamp to keep the trace harness from
                # crashing on np.eye() out-of-bounds.  The next
                # iteration's done/sort_done check will break the loop.
                raw_wi = overflow["sort_wall_index"][0, 0].item()
                if 0.0 <= raw_wi < max_walls:
                    wall_idx = int(round(raw_wi))
                else:
                    wall_idx = max(0, min(max_walls - 1, 0))
                trace.sort_steps.append(
                    SortStepTrace(
                        position_index=len(trace.sort_steps),
                        wall_j_onehot=np.eye(max_walls)[wall_idx],
                        selected_wall_index=wall_idx,
                        vis_lo=0.0,
                        vis_hi=0.0,
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

        # Bitblit pixels (length=0 at non-RENDER positions → no-op).
        for row_idx in range(length):
            y = start_y + row_idx
            if 0 <= y < H and 0 <= col < W and not filled[y, col]:
                frame[y, col] = pix[row_idx]
                filled[y, col] = True

        if done > 0.0 or sort_done > 0.0:
            break

        # Thinking-only fast path: the SORTED_WALL token is the
        # handoff out of the thinking phase.  Tests that only need
        # the token-stream trace can stop here and skip the SORTED +
        # RENDER work that follows.
        if stop_after_thinking and next_id_scalar == vocab_id("SORTED_WALL"):
            break

        prev = _next_inputs_from_overflow(next_id_scalar, overflow)

    if trace is not None:
        trace.token_id_log = token_id_log

    # Read post-collision (x, y) and post-turn angle from the RESOLVED
    # thinking tokens.  Phase B Part 1: per wall, 1 marker + 17 id-value
    # pairs = 35 steps; after all walls, offsets 0..5 of the RESOLVED
    # tail = [X_ID, X_VAL, Y_ID, Y_VAL, ANGLE_ID, ANGLE_VAL].  So the
    # VALUE positions land at ``max_walls * 35 + {1, 3, 5}``.
    def _dequant(token_id: int, name: str) -> float:
        lo, hi = VALUE_RANGE_BY_NAME[name]
        q = max(0, min(N_VALUES - 1, int(token_id)))
        return lo + q * (hi - lo) / float(N_VALUES - 1)

    resolved_x_offset = max_walls * 35 + 1
    resolved_y_offset = max_walls * 35 + 3
    resolved_angle_offset = max_walls * 35 + 5

    if len(token_id_log) > resolved_angle_offset:
        resolved_x = _dequant(token_id_log[resolved_x_offset], "RESOLVED_X")
        resolved_y = _dequant(token_id_log[resolved_y_offset], "RESOLVED_Y")
        resolved_angle = _dequant(token_id_log[resolved_angle_offset], "RESOLVED_ANGLE")
        px = resolved_x
        py = resolved_y
        new_angle_raw = resolved_angle
    else:
        # The loop exited before reaching the RESOLVED tokens (e.g.
        # ``stop_after_thinking`` tripped early or a degenerate scene
        # short-circuited).  Fall back to pre-collision state so
        # downstream trace/state construction doesn't NaN.
        new_angle_raw = angle

    if trace is not None:
        trace.resolved_x = px
        trace.resolved_y = py
        trace.new_angle = new_angle_raw

    print(f"  resolved state: px={px:.3f} py={py:.3f} angle={new_angle_raw:.3f}")

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
