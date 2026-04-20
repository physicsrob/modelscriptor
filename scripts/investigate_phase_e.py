"""Phase E root-cause investigation.

Traces the RAW sel_bsp_rank (pre-sentinel), sort_done, and threshold_onehot
at sort[0] for scene (px=3, py=2, angle=20) in box_room — the conditions
that trigger the xfail at ``test_renders_off_center_oblique[3.0-2.0-20]``.

The existing ``test_probe_phase_e_trace.py`` only traces the
post-sentinel sel_bsp_rank.  Since our SORTED stage replaces the raw
value with 99 when sort_done fires, that trace can only tell us
whether sort_done fired — not why.  This script drills into the
pre-sentinel values.
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import (
    attention_capture,
    probe_attention,
    probe_layer_diff,
    probe_residual,
)
from torchwright.doom.compile import _build_row
from torchwright.doom.game_graph import (
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
    build_game_graph,
)
from torchwright.doom.map_subset import build_scene_subset
from torchwright.graph import Concatenate, Linear
from torchwright.graph.attn import Attn
from torchwright.graph.misc import LiteralValue
from torchwright.graph.relu import ReLU
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

_MAX_WALLS = 8
_MAX_BSP_NODES = 48
_D = 2048
_D_HEAD = 32


def _config() -> RenderConfig:
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _segments(half=5.0):
    return [
        Segment(
            ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0
        ),
        Segment(
            ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1
        ),
        Segment(
            ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2
        ),
        Segment(
            ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3
        ),
    ]


def _find_key_in_concat(attn_node):
    """After Linear fusion, attn_node.key_in may be a Linear wrapping a
    Concatenate. BFS inward for a Concatenate with 3 inputs
    ([pos_enc, score, indicators_above])."""
    seen = set()
    stack = [attn_node.inputs[1]]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Concatenate) and len(node.inputs) == 3:
            # Heuristic: pos_encoding is wide (d_pos), score is 1-wide,
            # indicators_above is max_walls-wide.
            widths = [len(inp) for inp in node.inputs]
            if widths[1] == 1 and widths[2] == _MAX_WALLS:
                return node
        for inp in getattr(node, "inputs", []):
            stack.append(inp)
    raise AssertionError("could not find key_in Concatenate with 3 inputs")


def _find_sort_score_node(attn_node):
    """Walk Attn.key_in → Concatenate → inputs[1] = score."""
    from torchwright.graph.misc import Assert

    key_concat = _find_key_in_concat(attn_node)
    # Concatenate inputs: [pos_encoding, score, indicators_above]
    # But pos_encoding might be expressed as multiple inputs — depends
    # on PosEncoding layout.  Assert-unwrap the score slot.
    # Actually the Concatenate has exactly 3 inputs in the new primitive:
    # pos_encoding (a single Node), score, indicators_above.
    if len(key_concat.inputs) == 3:
        score = key_concat.inputs[1]
    else:
        # Fall back: find a 1-wide node that's not pos_encoding or indicators.
        score = None
        for inp in key_concat.inputs:
            if len(inp) == 1:
                score = inp
                break
    assert score is not None
    while isinstance(score, Assert):
        score = score.inputs[0]
    return score


def _find_indicators_above_node(attn_node):
    """Last Concatenate slot is indicators_above (max_walls-wide)."""
    from torchwright.graph.misc import Assert

    key_concat = _find_key_in_concat(attn_node)
    indicators = key_concat.inputs[-1]
    while isinstance(indicators, Assert):
        indicators = indicators.inputs[0]
    return indicators


def _find_sel_bsp_rank_effective(graph_io):
    """Find the post-sentinel sel_bsp_rank node in the graph.

    Walks all output nodes looking for the sentinel literal (99.0)
    used in the sort_done select. The sel_bsp_rank_effective is the
    select node whose true-branch is the 99.0 sentinel.
    """
    roots = set(graph_io.overlaid_outputs.values())
    roots.update(graph_io.overflow_outputs.values())
    for node in get_ancestor_nodes(roots):
        if isinstance(node, Linear) and len(node.inputs) == 1:
            inner = node.inputs[0]
            if isinstance(inner, ReLU) and len(inner.inputs) == 1:
                first_lin = inner.inputs[0]
                if isinstance(first_lin, Linear) and len(first_lin.inputs) == 1:
                    cat = first_lin.inputs[0]
                    if isinstance(cat, Concatenate) and len(cat.inputs) == 3:
                        true_node = cat.inputs[1]
                        if (
                            isinstance(true_node, LiteralValue)
                            and true_node.name == "sort_done_sentinel"
                        ):
                            return node
    raise AssertionError("sel_bsp_rank_effective (sort_done sentinel select) not found")


def _unwrap_asserts(node):
    """Peel off Assert wrappers to get the inner node."""
    from torchwright.graph.misc import Assert

    while isinstance(node, Assert):
        node = node.inputs[0]
    return node


def _find_raw_sel_bsp_rank(effective_node):
    """Walk backward through select() to find the raw (false-branch) input.

    select(cond, true, false) is built as:
        linear_relu_linear(Concatenate([cond, true, false]))
    which is Linear -> ReLU -> Linear.  Possibly wrapped in an Assert.
    """
    node = _unwrap_asserts(effective_node)
    assert isinstance(
        node, Linear
    ), f"expected select's output Linear, got {type(node).__name__}"
    # Second Linear -> its input is ReLU
    relu = node.inputs[0]
    assert isinstance(relu, ReLU), f"expected ReLU, got {type(relu).__name__}"
    # ReLU -> its input is first Linear
    first_linear = relu.inputs[0]
    assert isinstance(
        first_linear, Linear
    ), f"expected Linear, got {type(first_linear).__name__}"
    # First Linear -> its input is Concatenate([cond, true, false])
    concat = first_linear.inputs[0]
    assert isinstance(
        concat, Concatenate
    ), f"expected Concatenate, got {type(concat).__name__}"
    assert (
        len(concat.inputs) == 3
    ), f"expected 3 inputs to select's Concatenate, got {len(concat.inputs)}"
    cond, true_node, false_node = concat.inputs
    return {"cond": cond, "true": true_node, "false": false_node}


def _build_prefill(module, subset, *, px, py, angle):
    max_walls = int(module.metadata["max_walls"])
    max_bsp_nodes = int(module.metadata["max_bsp_nodes"])
    common = dict(
        player_x=torch.tensor([px]),
        player_y=torch.tensor([py]),
        player_angle=torch.tensor([float(angle)]),
    )
    tex_w = subset.textures[0].shape[0]
    rows = []
    for tex_idx in range(len(subset.textures)):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = subset.textures[tex_idx][col].flatten()
            rows.append(
                _build_row(
                    module,
                    max_walls,
                    token_type=E8_TEX_COL,
                    texture_id_e8=tex_e8,
                    tex_col_input=torch.tensor([float(col)]),
                    tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                    **common,
                )
            )
    rows.append(_build_row(module, max_walls, token_type=E8_INPUT, **common))
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            plane = subset.bsp_nodes[i]
            nx, ny, d = plane.nx, plane.ny, plane.d
        else:
            nx, ny, d = 0.0, 0.0, 0.0
        rows.append(
            _build_row(
                module,
                max_walls,
                token_type=E8_BSP_NODE,
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
                **common,
            )
        )
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes],
            dtype=torch.float32,
        )
        const = torch.tensor(
            [float(subset.seg_bsp_consts[i])],
            dtype=torch.float32,
        )
        rows.append(
            _build_row(
                module,
                max_walls,
                token_type=E8_WALL,
                wall_ax=torch.tensor([float(seg.ax)]),
                wall_ay=torch.tensor([float(seg.ay)]),
                wall_bx=torch.tensor([float(seg.bx)]),
                wall_by=torch.tensor([float(seg.by)]),
                wall_tex_id=torch.tensor([float(seg.texture_id)]),
                wall_index=torch.tensor([float(i)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
                **common,
            )
        )
    rows.append(_build_row(module, max_walls, token_type=E8_EOS, **common))
    return torch.cat(rows, dim=0)


def main():
    config = _config()
    textures = default_texture_atlas()
    segs = _segments()
    subset = build_scene_subset(segs, textures)

    print("Building game graph...")
    graph_io, pos_encoding = build_game_graph(
        config,
        textures,
        max_walls=_MAX_WALLS,
        max_coord=20.0,
        move_speed=0.3,
        turn_speed=4,
        chunk_size=20,
        max_bsp_nodes=_MAX_BSP_NODES,
    )

    from torchwright.graph.optimize import fuse_consecutive_linears

    output_nodes = set(graph_io.overlaid_outputs.values())
    output_nodes.update(graph_io.overflow_outputs.values())
    output_nodes.add(pos_encoding)
    while True:
        if fuse_consecutive_linears(output_nodes, verbose=False) == 0:
            break

    io = {}
    for name, node in graph_io.inputs.items():
        io[name] = (node, graph_io.overlaid_outputs.get(name))
    for name, node in graph_io.overflow_outputs.items():
        io[name] = (None, node)

    print("Walking graph for key nodes...")
    sel_effective = _find_sel_bsp_rank_effective(graph_io)
    select_inputs = _find_raw_sel_bsp_rank(sel_effective)
    sort_done = select_inputs["cond"]
    raw_sel = select_inputs["false"]
    true_lit = select_inputs["true"]

    # Asserts get stripped at compile; the underlying node is what ends up
    # in residual_assignment.  Unwrap so probe_* can find them.
    sel_effective_inner = _unwrap_asserts(sel_effective)
    sort_done_inner = _unwrap_asserts(sort_done)
    raw_sel_inner = _unwrap_asserts(raw_sel)
    print(
        f"  sel_eff:   {type(sel_effective).__name__} -> {type(sel_effective_inner).__name__}"
    )
    print(
        f"  sort_done: {type(sort_done).__name__} -> {type(sort_done_inner).__name__}"
    )
    print(f"  raw_sel:   {type(raw_sel).__name__} -> {type(raw_sel_inner).__name__}")

    print("Compiling module...")
    module = compile_headless(
        pos_encoding,
        io=io,
        d=_D,
        d_head=_D_HEAD,
        max_layers=400,
        verbose=False,
        extra_metadata={
            "chunk_size": 20,
            "max_walls": _MAX_WALLS,
            "max_bsp_nodes": _MAX_BSP_NODES,
            "tex_h": textures[0].shape[1],
        },
    )
    module.eval()

    print("Building prefill for (px=3, py=2, angle=20)...")
    prefill = _build_prefill(module, subset, px=3.0, py=2.0, angle=20.0)

    # Drive prefill -> EOS, then build sort[0] input from scratch.
    print("Running prefill...")
    past = module.empty_past()
    with torch.no_grad():
        pre_out, past = module.step(prefill, past, past_len=0)
    step = prefill.shape[0]

    d_input = max(s + w for _, s, w in module._input_specs)
    device = module._net.device

    from torchwright.doom.game_graph import E8_SORTED_WALL

    sort0_in = _build_row(
        module,
        _MAX_WALLS,
        token_type=E8_SORTED_WALL,
        sort_position_index=torch.tensor([0.0]),
        player_x=torch.tensor([3.0]),
        player_y=torch.tensor([2.0]),
        player_angle=torch.tensor([20.0]),
    )

    past_K, past_V = past
    past_kvs = [(past_K[i], past_V[i]) for i in range(len(past_K))]

    print("\n=== probe_layer_diff: raw_sel_bsp_rank (pre-sentinel) at sort[0] ===")
    report_raw = probe_layer_diff(
        module,
        sort0_in,
        raw_sel_inner,
        reference=torch.zeros(1, 1),
        positions=[0],
        past_len=step,
        past_kvs=past_kvs,
    )
    for rec in report_raw.records:
        print(
            f"  L{rec.layer_index:3d}  {rec.state_name:22s}  "
            f"val={rec.value[0, 0].item():+12.4f}"
        )

    print("\n=== probe_layer_diff: sort_done at sort[0] ===")
    report_done = probe_layer_diff(
        module,
        sort0_in,
        sort_done_inner,
        reference=torch.zeros(1, 1),
        positions=[0],
        past_len=step,
        past_kvs=past_kvs,
    )
    for rec in report_done.records:
        print(
            f"  L{rec.layer_index:3d}  {rec.state_name:22s}  "
            f"val={rec.value[0, 0].item():+12.4f}"
        )

    print(
        "\n=== probe_layer_diff: sel_bsp_rank_effective (post-sentinel) at sort[0] ==="
    )
    report_eff = probe_layer_diff(
        module,
        sort0_in,
        sel_effective_inner,
        reference=torch.zeros(1, 1),
        positions=[0],
        past_len=step,
        past_kvs=past_kvs,
    )
    for rec in report_eff.records:
        print(
            f"  L{rec.layer_index:3d}  {rec.state_name:22s}  "
            f"val={rec.value[0, 0].item():+12.4f}"
        )

    # --- Locate SORTED's attend_argmin_above_integer node & probe it ---
    print("\n=== SORTED attention softmax at sort[0] ===")

    # Walk: sel_effective's select reads raw_sel from unpack(attn_output).
    # The attn_output is the Attn node feeding unpack_wall_payload.  We
    # already have raw_sel_inner (a Linear from extract_from).  Its input
    # chain eventually reaches the Attn.
    def _find_attn_ancestor(node, depth=0):
        """BFS for the closest Attn ancestor."""
        seen = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            if isinstance(n, Attn):
                return n
            for inp in getattr(n, "inputs", []):
                stack.append(inp)
        return None

    attn_node = _find_attn_ancestor(raw_sel_inner)
    assert attn_node is not None, "could not find Attn ancestor of raw_sel"
    print(
        f"  found Attn: value_width={len(attn_node.inputs[2]) if attn_node.inputs[2] else 'None'}"
    )

    # Label the positions in the prefill to make the probe output readable.
    tex_w = subset.textures[0].shape[0]
    labels = []
    for tex_idx in range(len(subset.textures)):
        for c in range(tex_w):
            labels.append(f"TEX_COL[{tex_idx}][{c}]")
    labels.append("INPUT")
    for i in range(_MAX_BSP_NODES):
        labels.append(f"BSP_NODE[{i}]")
    for i, seg in enumerate(subset.segments):
        name = {0: "east", 1: "west", 2: "north", 3: "south"}.get(i, f"wall{i}")
        labels.append(f"WALL[{i}] ({name})")
    labels.append("EOS")
    labels.append("SORTED[0]")
    n_total_positions = len(labels)
    print(f"  total positions in full seq: {n_total_positions}")

    attn_probe = probe_attention(
        module,
        sort0_in,
        attn_node,
        query_pos=0,
        past_len=step,
        past_kvs=past_kvs,
        position_labels=labels,
    )
    print(
        f"  attention at layer {attn_probe.layer_index}, n_heads={attn_probe.weights.shape[0]}"
    )
    # Find the head that does the argmin work — probably head 0.
    for head_i in range(attn_probe.weights.shape[0]):
        w = attn_probe.weights[head_i]
        top = attn_probe.top(k=6, head=head_i)
        wall_mass = 0.0
        for idx, weight, label in [
            (i, w[i].item(), labels[i] if i < len(labels) else f"pos{i}")
            for i in range(w.shape[0])
        ]:
            if label.startswith("WALL["):
                wall_mass += weight
        sorted_mass = sum(
            w[i].item()
            for i in range(w.shape[0])
            if i < len(labels) and labels[i].startswith("SORTED")
        )
        print(
            f"\n  Head {head_i}: WALL-mass={wall_mass:.6f}, SORTED-mass={sorted_mass:.6f}"
        )
        print(f"    top-6:")
        for pos_k, weight_k, label_k in top:
            logit_k = attn_probe.logits[head_i, pos_k].item()
            print(
                f"      pos={pos_k:3d}  logit={logit_k:+12.4f}  weight={weight_k:.6f}  {label_k}"
            )

    # Also look at the raw logits at the WALL positions specifically.
    wall_start = next(i for i, l in enumerate(labels) if l.startswith("WALL["))
    n_walls = sum(1 for l in labels if l.startswith("WALL["))
    print(f"\n  Logits at all WALL positions (head 0):")
    for w_i in range(n_walls):
        pos = wall_start + w_i
        logit = attn_probe.logits[0, pos].item()
        weight = attn_probe.weights[0, pos].item()
        print(
            f"    pos={pos:3d}  logit={logit:+12.4f}  weight={weight:.6f}  {labels[pos]}"
        )

    # --- Probe sort_score (bsp_rank) and indicators_above values ---
    # at WALL positions using reference_eval on the prefill (not the
    # decode step).  This tells us the ORACLE (graph-level) values.
    from torchwright.debug.probe import reference_eval

    sort_score_node = _find_sort_score_node(attn_node)
    indicators_node = _find_indicators_above_node(attn_node)
    print(
        f"\n  sort_score node: {type(sort_score_node).__name__}  width={len(sort_score_node)}"
    )
    print(
        f"  indicators_above node: {type(indicators_node).__name__}  width={len(indicators_node)}"
    )

    # Build input_values dict from the prefill for reference_eval.
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    input_values = {}
    for name, (s, w) in in_by_name.items():
        input_values[name] = prefill[:, s : s + w]

    n_prefill = prefill.shape[0]
    print(f"\n  Running reference_eval over {n_prefill} prefill positions...")
    try:
        ref_vals = reference_eval(
            Concatenate([sort_score_node, indicators_node]),
            input_values,
            n_pos=n_prefill,
        )
    except Exception as e:
        print(f"  reference_eval failed: {e}")
        ref_vals = {}

    if ref_vals:
        score_ref = ref_vals[sort_score_node]
        ind_ref = ref_vals[indicators_node]
        print(f"\n  ORACLE sort_score + indicators_above at relevant positions:")
        for pos in [wall_start + i for i in range(n_walls)] + [
            n_prefill - 1
        ]:  # walls + EOS
            label = labels[pos] if pos < len(labels) else f"pos{pos}"
            s = score_ref[pos, 0].item()
            ind = ind_ref[pos].tolist()
            print(
                f"    pos={pos:3d}  {label:25s}  score={s:+8.2f}  ind_above={[f'{v:.2f}' for v in ind]}"
            )

    # Compare against what the COMPILED module produces at the same
    # positions.  Use probe_residual on the prefill run.
    print(f"\n  COMPILED sort_score + indicators_above at relevant positions:")
    from torchwright.debug.probe import probe_residual

    # Re-compute from prefill (not decode step) so positions line up.
    score_probe = probe_residual(module, prefill, sort_score_node)
    ind_probe = probe_residual(module, prefill, indicators_node)
    if score_probe.value is not None and ind_probe.value is not None:
        for pos in [wall_start + i for i in range(n_walls)] + [n_prefill - 1]:
            label = labels[pos] if pos < len(labels) else f"pos{pos}"
            s = score_probe.value[pos, 0].item()
            ind = ind_probe.value[pos].tolist()
            print(
                f"    pos={pos:3d}  {label:25s}  score={s:+8.2f}  ind_above={[f'{v:+5.2f}' for v in ind]}"
            )

    # Summary
    raw_final = (
        report_raw.records[-1].value[0, 0].item() if report_raw.records else None
    )
    done_final = (
        report_done.records[-1].value[0, 0].item() if report_done.records else None
    )
    eff_final = (
        report_eff.records[-1].value[0, 0].item() if report_eff.records else None
    )
    print("\n=== Summary at sort[0] for scene (3, 2, 20) ===")
    print(f"  raw sel_bsp_rank = {raw_final}")
    print(f"  sort_done        = {done_final}")
    print(f"  sel_eff          = {eff_final}")
    pos_idx = 0.0  # sort[0]
    if raw_final is not None:
        expected_sort_done = +1.0 if (pos_idx - raw_final) > 0.5 else -1.0
        print(f"  position_index = {pos_idx}")
        print(
            f"  pos_idx - raw = {pos_idx - raw_final:+.4f} → sort_done expected = {expected_sort_done:+.0f}"
        )


if __name__ == "__main__":
    main()
