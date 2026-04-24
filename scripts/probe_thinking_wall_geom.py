"""Probe thinking_wall/wall_geom_attention at angle 192.

Phase C Part 3 (commit 68f9c9a) migrated this attention to 2-wide
quadratic-equality and introduced 3 pipeline regressions at angles
160/192/210: SORTED misses wall 3 in the box_room scene.  Part 3's
commit message states the regression is "somewhere in the thinking_wall
quad attention's behavior at specific angles" but match_gain 20→1000
didn't help.  This probe captures the softmax weights and logits for
the attention at every thinking-identifier position so we can see
exactly what's happening.

Runs locally on GPU (the local box has an L4):

    uv run python -m scripts.probe_thinking_wall_geom
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.compiler.token_model import compile_token
from torchwright.debug.probe import probe_attention
from torchwright.doom.compile import _build_inputs, _stack_inputs


def _pack_flat(module, fields: dict) -> torch.Tensor:
    """Pack a dict of (n, w) per-field tensors into a flat (n, d_input) tensor."""
    n = fields["token_ids"].shape[0]
    d_input = max(s + w for _, s, w in module._input_specs)
    out = torch.zeros((n, d_input), dtype=torch.float32, device=module.device)
    for name, start, width in module._input_specs:
        if name in fields:
            t = fields[name].to(module.device)
            if t.shape[-1] != width:
                # auto-broadcast 1-wide to match
                assert t.shape[-1] == width, f"{name}: got width {t.shape[-1]} expected {width}"
            out[:, start : start + width] = t
    return out
from torchwright.doom.embedding import IDENTIFIER_NAMES, vocab_id
from torchwright.doom.game_graph import build_game_graph
from torchwright.graph import Attn
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment
from torchwright.doom.map_subset import build_scene_subset

_TRIG = generate_trig_table()


def _config() -> RenderConfig:
    return RenderConfig(
        screen_width=64,
        screen_height=80,
        fov_columns=32,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _box_room(half: float = 5.0) -> List[Segment]:
    return [
        Segment(ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0),
        Segment(ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1),
        Segment(ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2),
        Segment(ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3),
    ]


def find_attn_by_annotation(roots: set, annotation_prefix: str) -> list[Attn]:
    """Walk backwards from a set of output nodes and find all Attn
    nodes whose annotation starts with the given prefix."""
    ancestors = get_ancestor_nodes(roots)
    hits = []
    for node in ancestors:
        if isinstance(node, Attn):
            ann = node.annotation or ""
            if ann.startswith(annotation_prefix):
                hits.append(node)
    return hits


def main():
    config = _config()
    textures = default_texture_atlas()
    segs = _box_room()
    subset = build_scene_subset(segs, textures)
    max_walls = 8
    max_bsp_nodes = 48
    chunk_size = 20

    print("=" * 70)
    print("Building game graph ...")
    graph_io, pos_encoding = build_game_graph(
        config,
        textures,
        max_walls,
        max_coord=20.0,
        move_speed=0.3,
        turn_speed=4,
        chunk_size=chunk_size,
        max_bsp_nodes=max_bsp_nodes,
    )

    # Find the Attn node for thinking_wall/wall_geom_attention
    output_nodes = set(graph_io.overlaid_outputs.values())
    output_nodes.update(graph_io.overflow_outputs.values())
    output_nodes.add(pos_encoding)
    candidates = find_attn_by_annotation(output_nodes, "thinking_wall/wall_geom_attention")
    assert len(candidates) == 1, f"found {len(candidates)} candidates for thinking_wall/wall_geom_attention"
    wall_geom_attn = candidates[0]
    print(f"  found Attn: annotation={wall_geom_attn.annotation!r}")
    print(f"  d_query_in={wall_geom_attn.d_query_in}")
    print(f"  d_key_in={wall_geom_attn.d_key_in}")
    print(f"  d_value_in={wall_geom_attn.d_value_in}")
    print(f"  d_qk={wall_geom_attn.d_qk}")
    print(f"  d_v={wall_geom_attn.d_v}")
    print(f"  d_output={wall_geom_attn.d_output}")

    # Also find the find_current_wall attention (the upstream hop).
    current_wall_candidates = find_attn_by_annotation(
        output_nodes, "thinking_wall/find_current_wall"
    )
    assert len(current_wall_candidates) == 1
    current_wall_attn = current_wall_candidates[0]

    print(f"\n{'=' * 70}")
    print("Compiling ...")
    tex_w = textures[0].shape[0]
    tex_h = textures[0].shape[1]
    from torchwright.doom.compile import compute_min_d_head

    min_d_head = compute_min_d_head(max_walls, tex_w)
    d_head = 1
    while d_head < min_d_head:
        d_head *= 2
    d = 2048

    io: dict = {}
    for name, node in graph_io.inputs.items():
        io[name] = (node, graph_io.overlaid_outputs.get(name))
    for name, node in graph_io.overflow_outputs.items():
        assert name not in io
        io[name] = (None, node)

    module = compile_token(
        pos_encoding,
        graph_io.embedding,
        io=io,
        d=d,
        d_head=d_head,
        max_layers=400,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
        extra_metadata={
            "chunk_size": chunk_size,
            "max_walls": max_walls,
            "max_bsp_nodes": max_bsp_nodes,
            "tex_h": tex_h,
            "overflow_names": list(graph_io.overflow_outputs),
        },
        token_id_input_name="token_ids",
        logit_output_name="next_token_embedding",
    )
    module.eval()
    headless = module._headless  # type: ignore[attr-defined]
    print(f"  device={module.device}  total_layers={len(headless._net.layers)}")

    # Build the prefill for angle=192
    import sys
    angle = float(sys.argv[1]) if len(sys.argv) > 1 else 192.0
    print(f"\n{'=' * 70}")
    print(f"Building prefill for scene (px=0, py=0, angle={angle}) ...")
    px, py = 0.0, 0.0
    walls = [
        {"ax": s.ax, "ay": s.ay, "bx": s.bx, "by": s.by, "tex_id": float(s.texture_id)}
        for s in subset.segments
    ]
    num_tex = len(textures)
    rows: list[dict] = []
    input_kw = dict(
        input_forward=torch.tensor([0.0]),
        input_backward=torch.tensor([0.0]),
        input_turn_left=torch.tensor([0.0]),
        input_turn_right=torch.tensor([0.0]),
        input_strafe_left=torch.tensor([0.0]),
        input_strafe_right=torch.tensor([0.0]),
    )

    def _common(token_id, **extra):
        return _build_inputs(
            module,
            token_id=token_id,
            player_x=torch.tensor([px]),
            player_y=torch.tensor([py]),
            player_angle=torch.tensor([angle]),
            **extra,
        )

    # TEX_COL
    from torchwright.doom.embedding import index_to_vector
    from torchwright.doom.graph_constants import TEX_E8_OFFSET

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
    # INPUT
    rows.append(_common(token_id=vocab_id("INPUT"), **input_kw))
    # BSP_NODE × max_bsp_nodes
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            plane = subset.bsp_nodes[i]
            nx, ny, d_ = plane.nx, plane.ny, plane.d
        else:
            nx, ny, d_ = 0.0, 0.0, 0.0
        rows.append(
            _common(
                token_id=vocab_id("BSP_NODE"),
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d_], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
            )
        )
    wall_positions_start = len(rows)
    # WALL × N
    for i, w in enumerate(walls):
        if i < subset.seg_bsp_coeffs.shape[0]:
            coeffs = torch.tensor(subset.seg_bsp_coeffs[i, :max_bsp_nodes], dtype=torch.float32)
            const = torch.tensor([float(subset.seg_bsp_consts[i])], dtype=torch.float32)
        else:
            coeffs = torch.zeros(max_bsp_nodes, dtype=torch.float32)
            const = torch.zeros(1, dtype=torch.float32)
        rows.append(
            _common(
                token_id=vocab_id("WALL"),
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
    eos_pos = len(rows)
    rows.append(_common(token_id=vocab_id("EOS")))
    prefill_dict = _stack_inputs(rows)
    prefill = _pack_flat(module, prefill_dict)

    print(f"  prefill shape: {prefill.shape}")
    print(f"  wall positions: {wall_positions_start}..{wall_positions_start + len(walls) - 1}")
    print(f"  EOS position:   {eos_pos}")

    # Run prefill via CompiledHeadless then autoregressive steps for
    # player_x/y/angle + thinking to observe the thinking phase.
    #
    # For the probe we need to build a longer prefill that includes the
    # PLAYER_X/Y/ANGLE + thinking tokens so we can probe the attention
    # at a thinking-identifier position.  Simplest: run step_frame up
    # through the thinking phase and capture the full past_kvs, then
    # feed one more prefill row and probe.
    #
    # Instead of reimplementing that, use step_frame.  We need a way
    # to probe after the thinking tokens exist.  But probe_attention
    # wants a prefill tensor and past_kvs for the cached path.

    # Easier approach: probe the attention on a full-prefill tensor that
    # includes the entire thinking phase.  We build this by running
    # step_frame first and concatenating all inputs.
    #
    # But the thinking tokens are autoregressive — their token_ids
    # depend on the previous model output.  So we have to actually run
    # the model to generate them.

    print(f"\n{'=' * 70}")
    print("Running step_frame to generate the full prefill+thinking trace ...")
    from torchwright.doom.compile import step_frame
    from torchwright.doom.game import GameState
    from torchwright.doom.input import PlayerInput
    from torchwright.doom.trace import FrameTrace

    state = GameState(x=px, y=py, angle=angle, move_speed=0.3, turn_speed=4)
    inp = PlayerInput(forward=False)
    trace = FrameTrace()
    frame, new_state = step_frame(
        module, state, inp, subset, config, textures=textures, trace=trace,
        stop_after_thinking=True,
    )
    print(f"  token_id_log length: {len(trace.token_id_log)}")
    print(f"  last 50 token IDs: {trace.token_id_log[-50:]}")

    # Now find the BSP_RANK identifier positions.  Each wall has its own
    # sequence.  The identifier token_ids come from IDENTIFIER_NAMES.
    bsp_rank_id = vocab_id("BSP_RANK")
    print(f"\n  BSP_RANK vocab_id = {bsp_rank_id}")

    # The thinking phase starts after PLAYER_X/Y/ANGLE.  Prefill positions
    # = TEX_COL (num_tex*tex_w) + INPUT (1) + BSP_NODE (48) + WALL (N) +
    # EOS (1).  Then 3 PLAYER tokens fed autoregressively, then thinking.
    # The BSP_RANK id positions fire within each wall's thinking block.

    # Find BSP_RANK positions in token_id_log (after the first THINKING_WALL marker)
    # prefill_len = initial rows + PLAYER_X/Y/ANGLE.  token_id_log[0] is
    # first_thinking_id (THINKING_WALL_0), fed at position prefill_len.
    wall_markers = {vocab_id(f"THINKING_WALL_{i}"): i for i in range(max_walls)}
    bsp_rank_positions_per_wall: dict[int, int] = {}
    current_wall = -1
    prefill_len = eos_pos + 1 + 3  # +3 for PLAYER_X/Y/ANGLE
    for step_idx, tid in enumerate(trace.token_id_log):
        if tid in wall_markers:
            current_wall = wall_markers[tid]
        elif tid == bsp_rank_id and current_wall >= 0 and current_wall not in bsp_rank_positions_per_wall:
            # Position in the full token stream = prefill_len + step_idx
            bsp_rank_positions_per_wall[current_wall] = prefill_len + step_idx
    print(f"  BSP_RANK positions per wall:")
    for w_i in sorted(bsp_rank_positions_per_wall):
        print(f"    wall {w_i}: position {bsp_rank_positions_per_wall[w_i]}")

    # Probe at EACH wall's BSP_RANK position.  For the quad-eq attention:
    #   Q at BSP_RANK(wall=k) = [2·k, 1]
    #   K at WALL_k = [k, -k²]
    #   score(WALL_k) = -(wall_i - k)² + k²
    # Concentrates on WALL_k at BSP_RANK(wall=k).
    # Plan: use forward_cached with the full prefill+autoregressive sequence.

    # Build the full sequence tensor: prefill + autoregressive steps.
    # The autoregressive inputs for each step are produced internally by
    # step_frame via the host overlay.  To re-run, I can re-feed the same
    # token_ids.  But non-token-id fields (player_x, etc.) are consistent.

    # Simplest: use the past_kvs from step_frame.  But step_frame doesn't
    # return past_kvs directly.  Let me reimplement the minimum needed.

    # Alternative: use probe_attention with a fresh full-sequence prefill
    # by replaying the token_id_log.  Fresh prefill means past_kvs=None.
    # This builds a full res_stream including the thinking phase, and
    # probe_attention patches the attention layer to capture softmax.

    # Build the full prefill rows = initial prefill (TEX_COL..EOS) +
    # PLAYER_X/Y/ANGLE rows + thinking rows.  For each thinking row,
    # set token_id from trace.token_id_log and carry forward the state.
    # Non-token-id fields are mostly zero at thinking positions (host
    # didn't feed wall geometry etc., those come from attention).

    # Build PLAYER tokens.  Host feeds resolved_x/y/angle for these.
    player_x_id = vocab_id("PLAYER_X")
    player_y_id = vocab_id("PLAYER_Y")
    player_angle_id = vocab_id("PLAYER_ANGLE")
    player_rows = [
        _common(token_id=player_x_id),
        _common(token_id=player_y_id),
        _common(token_id=player_angle_id),
    ]
    # Thinking rows: just feed token_ids; other fields default.
    thinking_rows = []
    for tid in trace.token_id_log:
        thinking_rows.append(_common(token_id=tid))
    full_prefill_dict = _stack_inputs(rows + player_rows + thinking_rows)
    full_prefill = _pack_flat(module, full_prefill_dict)
    print(f"\n  full prefill shape: {full_prefill.shape}")

    # Position labels for readability
    position_labels = []
    for i in range(full_prefill.shape[0]):
        tid = int(full_prefill[i, 0].item())
        if wall_positions_start <= i < wall_positions_start + len(walls):
            wi = i - wall_positions_start
            position_labels.append(f"WALL_{wi}")
        else:
            position_labels.append(f"pos{i}")

    # Find which head in the captured weights corresponds to our logical Attn.
    # attention_capture returns (n_heads, n_new, n_keys) where n_heads is the
    # hardware head count at that layer.  The logical Attn is placed on some
    # subset of those heads.  We identify it by checking which head shows the
    # expected quadratic logit at wall 3 BSP_RANK.

    # Probe each wall's BSP_RANK position (including padded walls 4-7).
    for target_wall in sorted(bsp_rank_positions_per_wall):
        pad_marker = " (PADDED)" if target_wall >= len(walls) else ""
        bsp_pos = bsp_rank_positions_per_wall[target_wall]
        tid_at_pos = int(full_prefill[bsp_pos, 0].item())
        print(f"\n--- wall {target_wall}{pad_marker} BSP_RANK position {bsp_pos} (token_id={tid_at_pos}, expected {bsp_rank_id}) ---")
        expected_peak = min(target_wall, len(walls) - 1)
        if target_wall >= len(walls):
            print(f"  (padded wall — no matching WALL_{target_wall} exists; attention will pick nearest real wall)")
        ap = probe_attention(
            headless,
            prefill=full_prefill.to(module.device),
            attn_node=wall_geom_attn,
            query_pos=bsp_pos,
            past_len=0,
            past_kvs=None,
            position_labels=position_labels,
        )

        # Identify the "correct" head by finding the one whose logit at
        # WALL_k matches the expected quad score.
        # expected_score(target_wall, wall_i) = 2*target_wall*wall_i - wall_i²
        # logit = match_gain * score = 20 * score
        # (For padded walls with target_wall >= len(walls), the argmax of
        # score over wall_i ∈ [0, len(walls)-1] is len(walls)-1 — the
        # attention picks the highest real wall.)
        expected_scores = {
            wi: (2 * target_wall * wi - wi * wi) for wi in range(len(walls))
        }
        n_heads = ap.logits.shape[0]
        best_head = 0
        best_err = float("inf")
        for h in range(n_heads):
            err = 0.0
            for wi, exp_s in expected_scores.items():
                pos = wall_positions_start + wi
                actual_logit = ap.logits[h, pos].item()
                expected_logit = 20.0 * exp_s
                err += abs(actual_logit - expected_logit)
            if err < best_err:
                best_err = err
                best_head = h
        print(f"  matching head: {best_head} (logit_err={best_err:.2f} vs expected 20*quad)")

        w_row = ap.weights[best_head].cpu().numpy()
        l_row = ap.logits[best_head].cpu().numpy()
        top = np.argsort(w_row)[::-1][:6]
        print(f"  top-6 keys by weight at head {best_head}:")
        for k_i in top:
            lbl = position_labels[k_i] if k_i < len(position_labels) else f"pos{k_i}"
            print(f"    pos {k_i:4d} ({lbl:16s})  logit={l_row[k_i]:+9.3f}  weight={w_row[k_i]:.6f}")

        # Weight at each WALL position
        print(f"  WALL weights at head {best_head}:")
        for w_i in range(len(walls)):
            pos = wall_positions_start + w_i
            exp_s = expected_scores[w_i]
            print(
                f"    WALL_{w_i} (pos {pos}): weight={w_row[pos]:.6f} "
                f"logit={l_row[pos]:+.3f} expected_logit={20.0*exp_s:+.3f}"
            )


if __name__ == "__main__":
    main()
