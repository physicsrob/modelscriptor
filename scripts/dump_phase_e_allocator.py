"""Phase E allocator-state dump.

Companion to ``scripts/investigate_phase_e.py``.  That script probes
residual *values* during a decode step.  This one adds the missing
dimension — **who owns each column** in the residual stream at the
SORTED attention's read layer — so we can directly confirm or refute
the "column aliasing among simultaneously-live nodes" hypothesis from
``docs/postmortems/phase_e_xfail.md``.

Outputs three blocks at the layer-state feeding the SORTED attention:

1. I1 sanity check — verifies every column is owned by at most one node.
   If I1 ever fires on a real compile, it surfaces here.

2. Ownership of ``key_in``'s leaves — for each column in the sort
   attention's ``(pos_encoding, score, indicators_above)`` Concatenate,
   prints the owning node.  If any column is owned by a *different*
   node, that's aliasing.  If every column is owned by its expected
   leaf, aliasing is ruled out and the bug is elsewhere (numerical
   drift through the compute chain, schedule ordering, etc.).

3. Compiled-vs-oracle value comparison — for the same columns over a
   full prefill, shows the compiled residual value alongside
   ``reference_eval``'s oracle.  If compiled == oracle, the residual
   stream at the attention's read layer is exactly what the graph
   says — any downstream corruption lives inside the attention head
   itself (K/V matrices, softmax precision) rather than in its input.
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import (
    _extract_compiled_value,
    _run_with_states,
    probe_compiled,
    probe_residual,
    reference_eval,
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
from torchwright.graph.misc import Assert, LiteralValue
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


def _unwrap_asserts(node):
    while isinstance(node, Assert):
        node = node.inputs[0]
    return node


def _find_key_in_concat(attn_node):
    seen = set()
    stack = [attn_node.inputs[1]]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Concatenate) and len(node.inputs) == 3:
            widths = [len(inp) for inp in node.inputs]
            if widths[1] == 1 and widths[2] == _MAX_WALLS:
                return node
        for inp in getattr(node, "inputs", []):
            stack.append(inp)
    raise AssertionError("could not find key_in Concatenate with 3 inputs")


def _find_sort_attn(graph_io):
    """Walk forward from graph outputs; return the Attn node whose
    key_in is a 3-slot Concatenate [pos_enc, score, indicators]."""
    roots = set(graph_io.overlaid_outputs.values())
    roots.update(graph_io.overflow_outputs.values())
    for node in get_ancestor_nodes(roots):
        if not isinstance(node, Attn):
            continue
        try:
            _find_key_in_concat(node)
        except AssertionError:
            continue
        return node
    raise AssertionError("could not locate SORTED attention node")


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


def _build_labels(subset):
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
    return labels


def _dump_i1_sanity(state_to_cols):
    """Verify pairwise-disjointness at this state.  If it ever fires,
    I1 was violated on the real compile — previously thought impossible
    per residual_map._check_invariants."""
    seen = {}
    for node, cols in state_to_cols.items():
        for c in cols:
            if c in seen:
                other = seen[c]
                print(
                    f"  !!! I1 VIOLATION at this state: column {c} "
                    f"owned by BOTH {node!r} AND {other!r}"
                )
                return False
            seen[c] = node
    print(
        f"  I1 holds at this state: {len(seen)} live columns across "
        f"{len(state_to_cols)} nodes, all disjoint."
    )
    return True


def _dump_ownership(label, node, state_to_cols, extra_lookup=None):
    """Print the column set owned by ``node`` at this state, plus any
    unexpected owners for columns that should belong to ``node``."""
    leaves = flatten_concat_nodes([node]) if isinstance(node, Concatenate) else [node]
    total_cols = 0
    for leaf in leaves:
        cols = state_to_cols.get(leaf)
        if cols is None:
            # Maybe the leaf is not directly in the snapshot — try the
            # caller-provided lookup (e.g., for nodes wrapped by Assert).
            if extra_lookup is not None:
                cols = extra_lookup(leaf)
        if cols is None:
            print(f"  {label:20s}  {leaf!r}  NOT IN STATE")
            continue
        total_cols += len(cols)
        cols_preview = cols if len(cols) <= 10 else cols[:5] + ["..."] + cols[-3:]
        print(f"  {label:20s}  {leaf!r}  width={len(cols)}  cols={cols_preview}")
    print(f"  {label:20s}  TOTAL columns: {total_cols}")


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

    print("Locating SORTED attention node...")
    attn_node = _find_sort_attn(graph_io)
    key_concat = _find_key_in_concat(attn_node)
    # key_concat.inputs: [pos_encoding-like, score, indicators_above]
    pos_enc_node = _unwrap_asserts(key_concat.inputs[0])
    score_node = _unwrap_asserts(key_concat.inputs[1])
    indicators_node = _unwrap_asserts(key_concat.inputs[2])
    print(f"  Attn={attn_node!r}  d_qk={attn_node.d_qk}  d_v={attn_node.d_v}")
    print(f"  key_in = Concat[{pos_enc_node!r}, {score_node!r}, {indicators_node!r}]")

    print("\nCompiling module...")
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

    net = module._net
    ra = net.residual_assignment
    assert ra is not None

    # Find the layer that emits attn_node (its value appears at
    # layer_i.mlp.out_state for the layer whose attention sublayer
    # scheduled this Attn).  The input to this attention is layer_{i-1}'s
    # mlp.out_state.
    attn_layer_index = None
    for i, layer in enumerate(net.layers):
        if ra.has_node(layer.mlp.out_state, attn_node):
            attn_layer_index = i
            break
    assert (
        attn_layer_index is not None
    ), "Attn node not materialised in any layer's state"
    print(f"\nSORTED attention emits at layer {attn_layer_index}.")

    if attn_layer_index == 0:
        read_state = net.layers[0].attn.in_state
        read_label = f"L0.attn_in (pre-layer-0)"
    else:
        read_state = net.layers[attn_layer_index - 1].mlp.out_state
        read_label = f"L{attn_layer_index - 1}.mlp_out (feeds L{attn_layer_index} attn)"
    print(f"Reading state:  {read_label}")

    state_to_cols = ra.mapping.get(read_state, {})
    print(f"Nodes live at this state: {len(state_to_cols)}")

    print("\n=== Block 1: I1 sanity check at read state ===")
    _dump_i1_sanity(state_to_cols)

    print("\n=== Block 2: ownership of key_in's leaves ===")
    _dump_ownership("pos_encoding", pos_enc_node, state_to_cols)
    _dump_ownership("score", score_node, state_to_cols)
    _dump_ownership("indicators_above", indicators_node, state_to_cols)

    # Invert the map: which node owns each column read by K?
    def _cols_for(node):
        if isinstance(node, Concatenate):
            out = []
            for leaf in flatten_concat_nodes([node]):
                if leaf in state_to_cols:
                    out += state_to_cols[leaf]
            return out
        return state_to_cols.get(node, [])

    k_cols = (
        _cols_for(pos_enc_node) + _cols_for(score_node) + _cols_for(indicators_node)
    )
    print(
        f"\n  key_in total resolved columns: {len(k_cols)}  "
        f"(expected d_qk head rows x d_head, see attn_node.d_qk)"
    )

    # Cross-check: are any of these columns ALSO owned by a non-key_in
    # node at this state?  This is the direct "aliasing" test.
    # ``expected_owners`` must contain the LEAVES of any Concatenate
    # inputs, because state_to_cols holds leaves, not Concatenate
    # wrappers.
    expected_owners = set()
    for wrapper in (pos_enc_node, score_node, indicators_node):
        if isinstance(wrapper, Concatenate):
            expected_owners.update(flatten_concat_nodes([wrapper]))
        else:
            expected_owners.add(wrapper)
    k_col_set = set(k_cols)
    aliasing_pairs = []
    for node, cols in state_to_cols.items():
        if node in expected_owners:
            continue
        overlap = k_col_set & set(cols)
        if overlap:
            aliasing_pairs.append((node, sorted(overlap)[:8]))
    if aliasing_pairs:
        print("\n  !!! ALIASING FOUND — these nodes share key_in columns:")
        for node, cols_preview in aliasing_pairs:
            print(f"    {node!r}  overlap={cols_preview}")
    else:
        print(
            "\n  No aliasing: key_in's columns are owned exclusively by "
            "pos_encoding / score / indicators_above at this state.  "
            "The 'column aliasing among live nodes' hypothesis from the "
            "post-mortem is REFUTED at this layer."
        )

    print("\n=== Block 3: compiled vs oracle values on prefill ===")
    prefill = _build_prefill(module, subset, px=3.0, py=2.0, angle=20.0)
    labels = _build_labels(subset)
    wall_start = next(i for i, l in enumerate(labels) if l.startswith("WALL["))
    n_walls = sum(1 for l in labels if l.startswith("WALL["))
    # Include a few TEX_COL and BSP_NODE positions for baseline, plus all
    # WALLs and EOS — so we can see whether the compiled-vs-oracle gap is
    # WALL-specific or universal.
    positions_to_show = [
        0,
        4,
        16,  # TEX_COL samples
        32,  # INPUT
        33,
        40,
        50,
        60,
        70,  # BSP_NODE samples
    ]
    positions_to_show += [wall_start + i for i in range(n_walls)]
    positions_to_show.append(len(labels) - 1)  # EOS

    # Compiled values at the read state.
    _net, _ra, state_tensor = _run_with_states(module, prefill, past_len=0)
    res_tensor, _ = state_tensor[read_state]

    def _extract(node):
        val = _extract_compiled_value(node, ra, read_state, res_tensor)
        return val

    compiled_score = _extract(score_node)
    compiled_indicators = _extract(indicators_node)

    # Oracle via reference_eval.
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    input_values = {}
    for name, (s, w) in in_by_name.items():
        input_values[name] = prefill[:, s : s + w]
    try:
        ref = reference_eval(
            Concatenate([score_node, indicators_node]),
            input_values,
            n_pos=prefill.shape[0],
        )
        oracle_score = ref[score_node]
        oracle_indicators = ref[indicators_node]
    except Exception as exc:
        print(f"  reference_eval failed: {exc}")
        oracle_score = None
        oracle_indicators = None

    print(
        f"\n  {'pos':>4}  {'label':<25}  {'score':>10}  "
        f"{'oracle':>10}  {'|Δ|':>8}  {'ind_above (compiled // oracle)':<60}"
    )
    for pos in positions_to_show:
        label = labels[pos]
        c_s = (
            compiled_score[pos, 0].item()
            if compiled_score is not None
            else float("nan")
        )
        o_s = oracle_score[pos, 0].item() if oracle_score is not None else float("nan")
        d_s = abs(c_s - o_s)
        c_ind = (
            compiled_indicators[pos].tolist() if compiled_indicators is not None else []
        )
        o_ind = oracle_indicators[pos].tolist() if oracle_indicators is not None else []
        c_ind_s = "[" + ",".join(f"{v:+.2f}" for v in c_ind) + "]"
        o_ind_s = "[" + ",".join(f"{v:+.2f}" for v in o_ind) + "]"
        print(
            f"  {pos:>4}  {label:<25}  {c_s:>+10.4f}  {o_s:>+10.4f}  "
            f"{d_s:>8.4f}  {c_ind_s} // {o_ind_s}"
        )

    print(
        "\nInterpretation guide:\n"
        "  * If compiled == oracle (small |Δ|) at all positions above, the\n"
        "    residual stream feeding the SORTED attention is exactly what\n"
        "    the graph specifies.  Any downstream wrongness must live\n"
        "    inside the attention head (K/V weight corruption or softmax\n"
        "    precision), not in its input.\n"
        "  * If compiled ≠ oracle, the divergence is numerical drift\n"
        "    through the compute chain feeding this state — NOT column\n"
        "    aliasing (Block 2 would have surfaced that directly).\n"
    )

    print("\n=== Block 4: first divergent node via probe_compiled ===")
    # Run the probe against the full output so we can find the earliest
    # node whose compiled value diverges from oracle by > atol.  This
    # localizes the bug to a specific graph op rather than "somewhere in
    # the score/indicators chain".
    from torchwright.graph.attn import Attn as _Attn  # local import

    final_outputs = list(graph_io.overlaid_outputs.values()) + list(
        graph_io.overflow_outputs.values()
    )
    probe_root = Concatenate(final_outputs)
    report = probe_compiled(
        module,
        probe_root,
        input_values,
        n_pos=prefill.shape[0],
        atol=1e-4,
    )
    print(f"  nodes checked: {len(report.nodes_checked)}")
    print(f"  nodes skipped: {len(report.skipped)}")
    if report.first_divergent is None:
        print("  no divergence above atol=1e-2 — compiled == oracle everywhere!")
    else:
        fd = report.first_divergent
        print(f"  FIRST divergent node: {fd.node!r}")
        print(f"    max_abs_error = {fd.max_abs_error:.4f}")
        print(
            f"    compiled range [{fd.compiled_min:+.4f}, {fd.compiled_max:+.4f}]  "
            f"mean {fd.compiled_mean:+.4f}"
        )
        print(
            f"    oracle   range [{fd.oracle_min:+.4f}, {fd.oracle_max:+.4f}]  "
            f"mean {fd.oracle_mean:+.4f}"
        )
        print(f"    state = {fd.state.name or fd.state.state_id}")

    # Print the top 10 most-divergent nodes regardless of position in
    # topological order.  Useful for spotting whether divergence is
    # concentrated in one op family (e.g., all indicators, all
    # bsp_rank-related) or scattered.
    sorted_records = sorted(
        report.per_node.values(),
        key=lambda r: -r.max_abs_error,
    )
    print(f"\n  top-10 divergent nodes (by max_abs_error):")
    for r in sorted_records[:10]:
        print(
            f"    |Δ|={r.max_abs_error:>8.4f}  {r.node!r:80s}  "
            f"compiled~{r.compiled_mean:+.3f}  oracle~{r.oracle_mean:+.3f}"
        )

    # Show the top-30 EARLIEST-divergent nodes in topological order,
    # capped at atol=1e-4.  This traces where drift first surfaces and
    # how it grows op-by-op.
    print("\n  earliest-20 divergent nodes (topological order, atol=1e-4):")
    count = 0
    first_divergent_mlp_node = None
    for rec in report.per_node.values():
        if rec.max_abs_error <= 1e-4:
            continue
        print(
            f"    |Δ|={rec.max_abs_error:>10.6f}  {rec.node!r:80s}  "
            f"compiled~{rec.compiled_mean:+.4f}  oracle~{rec.oracle_mean:+.4f}"
        )
        count += 1
        # Record the first divergent Linear that came from an MLP chain
        # (typical name pattern: *_linear2) for Block 5 weight inspection.
        if (
            first_divergent_mlp_node is None
            and isinstance(rec.node, Linear)
            and "linear2" in (rec.node.name or "")
        ):
            first_divergent_mlp_node = rec.node
        if count >= 20:
            break

    print("\n=== Block 5: CPU fp64 replay of the compiled module ===")
    # Move the whole compiled module to CPU + float64 and re-run
    # probe_compiled.  If the same per-node errors persist, the
    # compiler is wiring something wrong (a construction bug).  If
    # errors drop to ~1e-10, the discrepancy we saw on GPU was
    # fp32/TF32 accumulation-order noise that compounds through the
    # chain.  This is the cleanest way to distinguish precision from
    # construction without slicing individual weight submatrices.
    print("  Moving compiled module to CPU + float64...")
    # Set default dtype to float64 so any transient tensor creation inside
    # compute() (e.g., PosEncoding.get_pos_encoding allocating zeros) lands
    # in fp64.  Restored in a try/finally below.
    _prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    import copy

    module_cpu = copy.deepcopy(module)
    module_cpu._net.to("cpu")
    # HeadlessTransformer.to() only takes device — manually convert each
    # weight tensor to float64 for the fp64 replay.
    for _layer in module_cpu._net.layers:
        for _attr in (
            "query_matrix",
            "key_matrix",
            "value_matrix",
            "output_matrix",
        ):
            setattr(
                _layer.attn.attn,
                _attr,
                getattr(_layer.attn.attn, _attr).to(torch.float64),
            )
        for _comp in (_layer.mlp.linear1, _layer.mlp.linear2):
            _comp.output_matrix = _comp.output_matrix.to(torch.float64)
            _comp.output_bias = _comp.output_bias.to(torch.float64)

    # reference_eval walks via node.compute() and reads the graph's own
    # weight tensors on each Linear node.  Convert those to fp64 too so
    # the oracle pass matches the compiled pass precision.  This mutates
    # the graph in place — fine because we're at the end of the script.
    all_graph_nodes = get_ancestor_nodes({probe_root})
    for _n in all_graph_nodes:
        if isinstance(_n, Linear):
            _n.output_matrix = _n.output_matrix.to(torch.float64)
            _n.output_bias = _n.output_bias.to(torch.float64)
        if isinstance(_n, Attn):
            _n.query_matrix = _n.query_matrix.to(torch.float64)
            _n.key_matrix = _n.key_matrix.to(torch.float64)
            _n.value_matrix = _n.value_matrix.to(torch.float64)
            _n.output_matrix = _n.output_matrix.to(torch.float64)
        if hasattr(_n, "value") and isinstance(_n.value, torch.Tensor):
            _n.value = _n.value.to(torch.float64)
    # The residual_assignment links are pickled into the deepcopy so we
    # can just rerun probe_compiled.  Use a fresh prefill in float64 so
    # the input matches.
    prefill_cpu_f64 = prefill.to(device="cpu", dtype=torch.float64)
    # Monkey-patch module_cpu to expose the new float64 prefill.  We
    # pass it through input_values as float64 so reference_eval also
    # runs in float64 (for bit-identical comparison).
    input_values_f64 = {
        name: prefill_cpu_f64[:, s : s + w].to(torch.float64)
        for name, (s, w) in in_by_name.items()
    }
    try:
        report_f64 = probe_compiled(
            module_cpu,
            probe_root,
            input_values_f64,
            n_pos=prefill.shape[0],
            atol=1e-8,
        )
    finally:
        torch.set_default_dtype(_prev_default_dtype)

    print("\n=== Block 6: per-bit noise on side_P_vec (48-wide) ===")
    # side_P_vec is the shared ``is player on front side of BSP node i?``
    # vector that every wall's bsp_rank dot product reads from.  Locate
    # it by finding any ``bsp_s_*`` extract_from Linear in the graph:
    # its inputs[0] is side_P_vec.
    side_P_vec_node = None
    for _n in all_graph_nodes:
        if isinstance(_n, Linear) and (_n.name or "").startswith("bsp_s_"):
            side_P_vec_node = _n.inputs[0]
            break
    if side_P_vec_node is None:
        print("  (could not locate side_P_vec — skipping)")
    else:
        print(
            f"  Found side_P_vec: {type(side_P_vec_node).__name__} "
            f"width={len(side_P_vec_node)}  name={side_P_vec_node.name!r}"
        )
        # Oracle + compiled at full fp32 on GPU (the production regime).
        # Re-run a fresh fp32 probe since we just nuked the graph's weights
        # to fp64 for Block 5; restore via reference_eval on fp32 inputs.
        for _n in all_graph_nodes:
            if isinstance(_n, Linear):
                _n.output_matrix = _n.output_matrix.to(torch.float32)
                _n.output_bias = _n.output_bias.to(torch.float32)
            if isinstance(_n, Attn):
                _n.query_matrix = _n.query_matrix.to(torch.float32)
                _n.key_matrix = _n.key_matrix.to(torch.float32)
                _n.value_matrix = _n.value_matrix.to(torch.float32)
                _n.output_matrix = _n.output_matrix.to(torch.float32)
            if hasattr(_n, "value") and isinstance(_n.value, torch.Tensor):
                _n.value = _n.value.to(torch.float32)
        oracle_fp32 = reference_eval(
            side_P_vec_node,
            input_values,
            n_pos=prefill.shape[0],
        ).get(side_P_vec_node)
        compiled_fp32 = probe_residual(module, prefill, side_P_vec_node)
        # Pick any layer where side_P_vec is materialised.
        if not compiled_fp32.per_layer:
            print("  (side_P_vec not materialised in any captured state)")
        else:
            # Report noise at every materialised layer — side_P_vec is
            # broadcast via attend_mean_where, so it should be position-
            # invariant after the broadcast; if earliest layer is pre-
            # broadcast that explains systematic offsets.
            print(f"  side_P_vec materialised at layers: {compiled_fp32.layers}")
            oracle_sp_cpu = oracle_sp_ref = oracle_fp32.detach().cpu()
            for _lyr in compiled_fp32.layers:
                _val = compiled_fp32.at(_lyr).detach().cpu()
                _diff = (_val - oracle_sp_cpu).abs()
                print(
                    f"    layer {_lyr:3d}  "
                    f"overall_max|Δ|={_diff.max().item():.4f}  "
                    f"wall_max|Δ|={_diff[81:85].max().item():.4f}  "
                    f"bsp_node_max|Δ|={_diff[33:81].max().item():.4f}"
                )
            # Use LAST materialised layer for detailed breakdown — that's
            # where bsp_rank reads from it.
            layer_i = compiled_fp32.layers[-1]
            compiled_sp = compiled_fp32.at(layer_i).detach().cpu()
            oracle_sp = oracle_fp32.detach().cpu()
            diff = (compiled_sp - oracle_sp).abs()
            print(
                f"\n  Using layer {layer_i} (last materialised) for detailed "
                f"breakdown.  Shape={tuple(compiled_sp.shape)}"
            )

            wall_positions = list(
                range(
                    next(i for i, l in enumerate(labels) if l.startswith("WALL[")),
                    next(i for i, l in enumerate(labels) if l.startswith("WALL["))
                    + sum(1 for l in labels if l.startswith("WALL[")),
                )
            )
            bsp_node_positions = [
                i for i, l in enumerate(labels) if l.startswith("BSP_NODE[")
            ]

            def _summarize(label, rows):
                pos_diff = diff[rows]
                print(
                    f"\n  {label} (n_pos={len(rows)}):\n"
                    f"    overall   max|Δ|={pos_diff.max().item():.4f}  "
                    f"mean|Δ|={pos_diff.mean().item():.6f}\n"
                    f"    per-bit  max|Δ| across bits: "
                    f"{pos_diff.amax(dim=0).tolist()[:16]}... (first 16 of "
                    f"{pos_diff.shape[1]})"
                )

            _summarize(
                "At BSP_NODE positions (where side_P is computed)",
                bsp_node_positions[:20],
            )
            _summarize("At WALL positions (where bsp_rank reads it)", wall_positions)

            # Show one WALL position in detail.
            if wall_positions:
                w0 = wall_positions[0]
                print(f"\n  Detailed dump at pos {w0} (WALL[0] east):")
                print(
                    f"    oracle:   {[f'{v:+.3f}' for v in oracle_sp[w0].tolist()[:16]]}... (first 16)"
                )
                print(
                    f"    compiled: {[f'{v:+.3f}' for v in compiled_sp[w0].tolist()[:16]]}... (first 16)"
                )
                print(
                    f"    |Δ|:      {[f'{v:.4f}' for v in diff[w0].tolist()[:16]]}... (first 16)"
                )

                # Count how many bits are "cleanly 0/1" (|Δ| < 0.05) vs noisy.
                clean = (diff[w0] < 0.05).sum().item()
                medium = ((diff[w0] >= 0.05) & (diff[w0] < 0.2)).sum().item()
                noisy = (diff[w0] >= 0.2).sum().item()
                print(
                    f"    bit classification: "
                    f"clean(<0.05)={clean} / "
                    f"medium(0.05-0.2)={medium} / "
                    f"noisy(>=0.2)={noisy}   (of {diff.shape[1]})"
                )
                # Show the worst bits specifically.
                topk = torch.topk(diff[w0], k=min(8, diff.shape[1]))
                print(f"    worst 8 bits (index, |Δ|, compiled, oracle):")
                for idx_t, v in zip(topk.indices.tolist(), topk.values.tolist()):
                    print(
                        f"      bit[{idx_t:2d}]  |Δ|={v:.4f}  "
                        f"compiled={compiled_sp[w0, idx_t].item():+.4f}  "
                        f"oracle={oracle_sp[w0, idx_t].item():+.4f}"
                    )
    print(f"  nodes checked: {len(report_f64.nodes_checked)}")
    if report_f64.first_divergent is None:
        print(
            "  CLEAN: no divergence > 1e-8 anywhere.  GPU fp32/TF32 "
            "precision was the sole source of drift — the compiler's "
            "weight wiring is faithful to the graph."
        )
    else:
        fd = report_f64.first_divergent
        print(
            f"  STILL DIVERGES in fp64: first node "
            f"{fd.node!r}  max|Δ|={fd.max_abs_error:.3e}"
        )
        print("  top-5 divergent in fp64:")
        sorted_f64 = sorted(
            report_f64.per_node.values(),
            key=lambda r: -r.max_abs_error,
        )
        for r in sorted_f64[:5]:
            print(
                f"    |Δ|={r.max_abs_error:>10.3e}  {r.node!r:70s}  "
                f"compiled~{r.compiled_mean:+.4f}  oracle~{r.oracle_mean:+.4f}"
            )

    print("\n=== Block 7: approximate-gate call sites by M ===")
    # After the big_offset refactor, each cond_gate / select derives its
    # M from its input's value_type (scaled by GATE_OFFSET_SAFETY_FACTOR).
    # Large M → large M·ε_cond error on the approximate path.  Enumerate
    # every approximate gate's output Linear, compute the M that was
    # baked in, and show the fp32 error next to it so we can see which
    # call sites are the actual offenders.
    try:
        from torchwright.ops.logic_ops import (
            _GATE_OFFSET_SAFETY_FACTOR as _SAFETY,
        )
    except ImportError:
        _SAFETY = 2.0
    try:
        from torchwright.ops.const import step_sharpness as _STEP_SHARPNESS
    except ImportError:
        _STEP_SHARPNESS = 10.0

    def _gate_M(node):
        """Estimate the M baked into a *_linear2 gate-output node.
        For cond_gate: M = safety * max|inp.value_type|.  The output's
        value_range is [min(0, inp.lo), max(0, inp.hi)], so recovering
        max|inp| from the output range works for cond_gate (≈ max(|range|)).
        For select: output range covers both branches; max|range| is a
        good proxy for M since the gate routes up to max of either branch.
        Multiplied by GATE_OFFSET_SAFETY_FACTOR (= 2 by default).
        """
        r = node.value_type.value_range
        import math as _m

        if not _m.isfinite(r.lo) or not _m.isfinite(r.hi):
            return float("inf")
        return _SAFETY * max(abs(r.lo), abs(r.hi))

    gate_records = []
    for n in all_graph_nodes:
        if not isinstance(n, Linear):
            continue
        nm = n.name or ""
        if ("select_linear2" not in nm) and ("cond_gate_linear2" not in nm):
            continue
        rec = report.per_node.get(n)  # fp32 report from Block 4
        rec_f64 = report_f64.per_node.get(n) if "report_f64" in dir() else None
        M_est = _gate_M(n)
        predicted_err = M_est / _STEP_SHARPNESS
        gate_records.append(
            {
                "node": n,
                "M": M_est,
                "predicted_err": predicted_err,
                "fp32_err": rec.max_abs_error if rec else None,
                "fp64_err": rec_f64.max_abs_error if rec_f64 else None,
            }
        )
    print(f"  Found {len(gate_records)} approximate *_linear2 gate outputs.")

    # Sort by M descending (big-M sites are the actual offenders; fp32
    # errors are scene-dependent but M is static).
    gate_records.sort(key=lambda r: -(r["M"] if r["M"] != float("inf") else 1e20))
    print(
        f"\n  Top 15 by M  "
        f"(predicted_err = M / step_sharpness; step_sharpness={_STEP_SHARPNESS}):"
    )
    print(
        f"  {'id':>6}  {'M':>12}  {'pred|Δ|':>10}  "
        f"{'fp32|Δ|':>10}  {'fp64|Δ|':>10}   name"
    )
    for r in gate_records[:15]:
        n = r["node"]
        nm = (n.name or "")[:30]
        M = r["M"]
        pred = r["predicted_err"]
        fp32 = r["fp32_err"]
        fp64 = r["fp64_err"]
        pred_s = f"{pred:>10.2f}" if pred < float("inf") else "       inf"
        fp32_s = f"{fp32:>10.2f}" if fp32 is not None else "         -"
        fp64_s = f"{fp64:>10.2e}" if fp64 is not None else "         -"
        print(f"  {n.node_id:>6}  {M:>12.2f}  {pred_s}  {fp32_s}  {fp64_s}   {nm}")

    # For the top-3 biggest-M sites, walk back through the select's
    # Linear→ReLU→Linear chain to find the true/false branch nodes
    # and their declared value_types.
    print(
        "\n  Walking back from the top-3 biggest-M sites to find the "
        "true/false branches whose loose value_type is setting M:"
    )

    def _find_branches(select_linear2_node):
        """Walk: select_linear2 -> its Linear's input is a ReLU whose input
        is a Linear whose input is a Concatenate([cond, true, false])."""
        l2 = select_linear2_node
        if not l2.inputs:
            return None
        relu = l2.inputs[0]
        if not isinstance(relu, ReLU) or not relu.inputs:
            return None
        l1 = relu.inputs[0]
        if not isinstance(l1, Linear) or not l1.inputs:
            return None
        concat = l1.inputs[0]
        if not isinstance(concat, Concatenate):
            return None
        branches = concat.inputs
        return branches

    for r in gate_records[:3]:
        n = r["node"]
        print(f"\n  --- id={n.node_id} (M={r['M']:.2f}):")
        branches = _find_branches(n)
        if branches is None:
            print("    (could not extract branches — non-standard structure)")
            continue
        for i, b in enumerate(branches):
            vr = b.value_type.value_range
            name_str = (b.name or "").strip() or f"<anonymous id={b.node_id}>"
            if isinstance(b, Concatenate):
                # Walk leaves for detail.
                leaves = flatten_concat_nodes([b])
                print(
                    f"    branch[{i}] ({type(b).__name__}): "
                    f"width={len(b)} with {len(leaves)} leaves"
                )
                # Show leaves with widest ranges.
                leaf_ranges = []
                for lf in leaves:
                    lr = lf.value_type.value_range
                    m = max(abs(lr.lo), abs(lr.hi))
                    leaf_ranges.append((m, lf, lr))
                leaf_ranges.sort(key=lambda t: -t[0])
                for m, lf, lr in leaf_ranges[:3]:
                    print(
                        f"      leaf max|range|={m:>12.2f}  "
                        f"[{lr.lo:+.2f}, {lr.hi:+.2f}]  "
                        f"{type(lf).__name__}  name={lf.name!r}  "
                        f"id={lf.node_id}"
                    )
            else:
                m = max(abs(vr.lo), abs(vr.hi))
                print(
                    f"    branch[{i}] max|range|={m:>12.2f}  "
                    f"[{vr.lo:+.2f}, {vr.hi:+.2f}]  "
                    f"{type(b).__name__}  name={name_str}  id={b.node_id}"
                )

    # Scan ALL graph nodes for the widest declared ranges — useful for
    # seeing whether a handful of parent nodes are polluting many
    # downstream consumers.
    print("\n  Top 15 widest declared value_ranges in the graph:")
    import math as _math

    range_recs = []
    for _n in all_graph_nodes:
        vr = _n.value_type.value_range
        if not (_math.isfinite(vr.lo) and _math.isfinite(vr.hi)):
            continue
        m = max(abs(vr.lo), abs(vr.hi))
        range_recs.append((m, _n, vr))
    range_recs.sort(key=lambda t: -t[0])
    seen_names = set()
    shown = 0
    for m, _n, vr in range_recs:
        # Dedupe by (type, name) so we don't see 1000 copies of the same op.
        key = (type(_n).__name__, _n.name or "")
        if key in seen_names:
            continue
        seen_names.add(key)
        print(
            f"    max|range|={m:>12.2f}  [{vr.lo:+.2f}, {vr.hi:+.2f}]  "
            f"{type(_n).__name__}  name={_n.name!r}  id={_n.node_id}"
        )
        shown += 1
        if shown >= 15:
            break

    # For the TOP 5 nodes with non-zero observed fp32 error, walk FORWARD
    # from each to see what render-visible outputs they feed into.  If a
    # noisy node only affects dead-end paths, it's not a priority.  If it
    # flows into wall_data, render_data, pixel outputs, or overlaid graph
    # outputs, it's the next target for tightening.
    print("\n  Forward-trace from top-5 observed-error sites:")
    err_sorted = [
        r for r in gate_records if r["fp32_err"] is not None and r["fp32_err"] > 1.0
    ]
    err_sorted.sort(key=lambda r: -r["fp32_err"])

    def _walk_forward(start_node, max_depth=5):
        reachable_roots = list(graph_io.overlaid_outputs.values()) + list(
            graph_io.overflow_outputs.values()
        )
        all_nodes_local = get_ancestor_nodes(set(reachable_roots))
        rev = {n: [] for n in all_nodes_local}
        for n in all_nodes_local:
            for inp in getattr(n, "inputs", []):
                if inp in rev:
                    rev[inp].append(n)
        frontier = [(start_node, 0)]
        seen = {start_node}
        visited = []
        while frontier:
            n, d = frontier.pop(0)
            if d > max_depth:
                continue
            visited.append((n, d))
            for c in rev.get(n, []):
                if c not in seen:
                    seen.add(c)
                    frontier.append((c, d + 1))
        return visited

    for r in err_sorted[:5]:
        n = r["node"]
        fp32 = r["fp32_err"]
        rec = report.per_node[n]
        print(
            f"\n    id={n.node_id} name={n.name!r}  fp32|Δ|={fp32:.2f}  "
            f"compiled~{rec.compiled_mean:+.3f} "
            f"oracle~{rec.oracle_mean:+.3f}"
        )
        visited = _walk_forward(n, max_depth=4)
        shown = 0
        for d_n, depth in visited[1:]:
            nm = d_n.name or ""
            if nm and shown < 10:
                print(
                    f"      depth={depth} {type(d_n).__name__:<15} "
                    f"name={nm!r}  id={d_n.node_id}"
                )
                shown += 1

    # Also: histogram of M across all approximate gates.
    import statistics

    Ms = [r["M"] for r in gate_records if r["M"] < float("inf")]
    if Ms:
        print(
            f"\n  Distribution of M across {len(Ms)} approximate gates: "
            f"min={min(Ms):.2f}  median={statistics.median(Ms):.2f}  "
            f"max={max(Ms):.2f}  mean={statistics.mean(Ms):.2f}"
        )
        buckets = [(0, 2), (2, 10), (10, 100), (100, 1000), (1000, 1e6)]
        print("  Bucketed count:")
        for lo, hi in buckets:
            count = sum(1 for m in Ms if lo <= m < hi)
            print(f"    M ∈ [{lo:>6g}, {hi:<7g}):  {count}")


if __name__ == "__main__":
    main()
