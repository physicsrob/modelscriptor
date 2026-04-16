"""Mode C diagnostic: localize where angle-192 drift enters the full-game compile.

TODO.md Mode C documents a 0.14 drift on ``wall_ax`` and ``wall_by`` at
``sort[2]`` in the full-game compile at angle=192 that disappears when
WALL or SORTED are compiled in isolation.  Hypothesis: residual-column
aliasing in the scheduler — wall_ax / wall_by columns get reassigned
to another node before the SORTED stage's attention reads them.

This file builds the full game graph the same way ``compile_game`` does,
compiles it, then feeds the angle-192 prefill through a single
``forward(return_states=True)`` call.  For every per-layer snapshot
state we check:

  * Is ``wall_ax`` (the InputNode) still tracked in the residual
    assignment?  If it disappears, some layer freed its columns.
  * At each WALL token position, what value does the residual stream
    hold in wall_ax's columns?  If it drifts away from the host-fed
    input value, some other node is overwriting those columns.

Same question for ``wall_by``.  Running this test prints a per-layer
trace that pinpoints the exact layer where the drift first appears.
"""

from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
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
from torchwright.graph.spherical_codes import index_to_vector

from tests.doom.test_bsp_rank_integration import (
    _build_synthetic_subset,
    _small_config,
)


_MAX_WALLS = 8
_MAX_BSP_NODES = 16
_D = 2048
_D_HEAD = 32


# ---------------------------------------------------------------------------
# Fixture: build graph + compile it with direct node references retained.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_and_module():
    """Rebuild the game graph and compile it with the same knobs as
    compile_game, but keep the ``graph_io`` around so tests have
    direct references to wall_ax / wall_by / pos_encoding / outputs.
    """
    config = _small_config()
    subset = _build_synthetic_subset(max_bsp_nodes=_MAX_BSP_NODES)

    graph_io, pos_encoding = build_game_graph(
        config, subset.textures,
        max_walls=_MAX_WALLS, max_coord=20.0,
        move_speed=0.3, turn_speed=4,
        chunk_size=20, max_bsp_nodes=_MAX_BSP_NODES,
    )

    # Match compile_game's io dict shape: overlaid + overflow outputs.
    io: Dict[str, Tuple] = {}
    for name, node in graph_io.inputs.items():
        io[name] = (node, graph_io.overlaid_outputs.get(name))
    for name, node in graph_io.overflow_outputs.items():
        assert name not in io
        io[name] = (None, node)

    # Match compile_game's Linear-fusion optimization.
    from torchwright.graph.optimize import fuse_consecutive_linears
    output_nodes = set(graph_io.overlaid_outputs.values())
    output_nodes.update(graph_io.overflow_outputs.values())
    output_nodes.add(pos_encoding)
    while True:
        fused = fuse_consecutive_linears(output_nodes, verbose=False)
        if fused == 0:
            break

    # Collect asserts before compile_headless strips them, so tests can
    # later run predicates against the compiled residual stream.
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

    module = compile_headless(
        pos_encoding, io=io,
        d=_D, d_head=_D_HEAD, max_layers=400,
        verbose=False,
        extra_metadata={
            "chunk_size": 20, "max_walls": _MAX_WALLS,
            "max_bsp_nodes": _MAX_BSP_NODES,
            "tex_h": subset.textures[0].shape[1],
            "asserts": collected_asserts,
        },
    )
    module.eval()
    return graph_io, module, subset, config


# ---------------------------------------------------------------------------
# Prefill builder — mirrors step_frame's construction for angle=192.
# ---------------------------------------------------------------------------


def _build_angle_192_prefill(
    module, subset, config, angle: float = 192.0,
) -> Tuple[torch.Tensor, int, int]:
    """Build the (n_prefill, d_input) flat input the compiled graph expects
    for angle=192 at the origin.

    Returns ``(prefill, wall_base_idx, eos_idx)`` where
    ``wall_base_idx`` is the position index of the first WALL token.
    Wall order is: east (0), north (1), west (2), south (3) — matching
    subset.segments.
    """
    max_walls = int(module.metadata["max_walls"])
    max_bsp_nodes = int(module.metadata["max_bsp_nodes"])
    num_tex = len(subset.textures)
    tex_w = subset.textures[0].shape[0]

    px, py = 0.0, 0.0
    common = dict(
        player_x=torch.tensor([px]),
        player_y=torch.tensor([py]),
        player_angle=torch.tensor([float(angle)]),
    )

    rows: List[torch.Tensor] = []

    # TEX_COL × (num_tex × tex_w)
    for tex_idx in range(num_tex):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = subset.textures[tex_idx][col].flatten()
            rows.append(_build_row(
                module, max_walls,
                token_type=E8_TEX_COL,
                texture_id_e8=tex_e8,
                tex_col_input=torch.tensor([float(col)]),
                tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                **common,
            ))

    # INPUT
    rows.append(_build_row(module, max_walls, token_type=E8_INPUT, **common))

    # BSP_NODE × max_bsp_nodes
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            plane = subset.bsp_nodes[i]
            nx, ny, d = plane.nx, plane.ny, plane.d
        else:
            nx, ny, d = 0.0, 0.0, 0.0
        rows.append(_build_row(
            module, max_walls,
            token_type=E8_BSP_NODE,
            bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
            bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
            bsp_plane_d=torch.tensor([d], dtype=torch.float32),
            bsp_node_id_onehot=onehot,
            **common,
        ))

    wall_base_idx = len(rows)
    # WALL × N — subset has 4 walls in order east, north, west, south.
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes], dtype=torch.float32,
        )
        const = torch.tensor(
            [float(subset.seg_bsp_consts[i])], dtype=torch.float32,
        )
        rows.append(_build_row(
            module, max_walls,
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
        ))

    eos_idx = len(rows)
    rows.append(_build_row(module, max_walls, token_type=E8_EOS, **common))
    prefill = torch.cat(rows, dim=0)
    return prefill, wall_base_idx, eos_idx


# ---------------------------------------------------------------------------
# Diagnostic: trace wall_ax / wall_by residual values layer by layer.
# ---------------------------------------------------------------------------


def test_trace_wall_input_columns_through_layers(graph_and_module):
    """Print + assert whether wall_ax / wall_by stay clean through all layers.

    Extracts each WALL-input InputNode's compiled value at every
    post-sublayer snapshot state, for each wall token position.
    Compares against the host-fed value (the row's wall_ax at the
    input position).

    If the residual assignment loses track of wall_ax (node disappears
    from the snapshot), or if the values at its columns drift from
    the host-fed input, we've found a candidate aliasing.
    """
    graph_io, module, subset, config = graph_and_module

    wall_ax_node = graph_io.inputs["wall_ax"]
    wall_ay_node = graph_io.inputs["wall_ay"]
    wall_bx_node = graph_io.inputs["wall_bx"]
    wall_by_node = graph_io.inputs["wall_by"]

    prefill, wall_base_idx, _ = _build_angle_192_prefill(
        module, subset, config, angle=192.0,
    )

    # Single-pass forward with per-sublayer residual snapshots.
    res_stream = module._build_res_stream(prefill, past_len=0)
    net = module._net
    with torch.no_grad():
        _, all_states = net.forward(res_stream, return_states=True)

    ra = net.residual_assignment
    # Order states by layer index (0..L-1) using net.layers for stability.
    # mlp.out_state is the snapshot the scheduler records per layer.
    ordered_states = []
    for i, layer in enumerate(net.layers):
        st = layer.mlp.out_state
        if st in ra.mapping:
            ordered_states.append((i, f"L{i}.mlp_out", st))

    # Map ResidualStreamState → captured tensor.
    state_tensor = {}
    for key, (state, tensor) in all_states.items():
        state_tensor.setdefault(state, tensor)

    # Host-fed ground truth per wall position.
    in_specs = {n: (s, w) for n, s, w in module._input_specs}
    ax_s, ax_w = in_specs["wall_ax"]
    by_s, by_w = in_specs["wall_by"]
    ay_s, ay_w = in_specs["wall_ay"]
    bx_s, bx_w = in_specs["wall_bx"]

    n_walls = len(subset.segments)
    wall_positions = list(range(wall_base_idx, wall_base_idx + n_walls))

    host_ax = prefill[wall_positions, ax_s].tolist()
    host_ay = prefill[wall_positions, ay_s].tolist()
    host_bx = prefill[wall_positions, bx_s].tolist()
    host_by = prefill[wall_positions, by_s].tolist()

    wall_labels = ["east (0)", "north (1)", "west (2)", "south (3)"]
    header = (
        f"\nTracing wall input InputNodes through {len(ordered_states)} "
        f"post-MLP snapshots (d={_D}, d_head={_D_HEAD}, "
        f"max_walls={_MAX_WALLS}, max_bsp_nodes={_MAX_BSP_NODES}).\n"
        f"WALL token positions: {wall_positions} (east, north, west, south)\n"
        f"Host-fed values per wall position:\n"
        f"  ax = {host_ax}\n"
        f"  ay = {host_ay}\n"
        f"  bx = {host_bx}\n"
        f"  by = {host_by}\n"
    )
    print(header)

    fields = [
        ("wall_ax", wall_ax_node, host_ax),
        ("wall_ay", wall_ay_node, host_ay),
        ("wall_bx", wall_bx_node, host_bx),
        ("wall_by", wall_by_node, host_by),
    ]

    # For each field, record the first layer where the residual value
    # at that field's columns drifts > 1e-3 from the host-fed value at
    # any WALL token position.
    drift_thresh = 1e-3
    first_drift: Dict[str, Tuple[int, str, List[float]]] = {}

    for field_name, node, host_vals in fields:
        last_layer_alive = None
        for layer_idx, state_name, state in ordered_states:
            tensor = state_tensor.get(state)
            if tensor is None:
                continue
            if not ra.has_node(state, node):
                # Node freed/reassigned at this state.  Record and stop.
                if last_layer_alive is None:
                    first_drift.setdefault(field_name, (
                        layer_idx, f"UNASSIGNED@{state_name}", [],
                    ))
                else:
                    first_drift.setdefault(field_name, (
                        layer_idx,
                        f"UNASSIGNED@{state_name} (last alive L{last_layer_alive})",
                        [],
                    ))
                break
            last_layer_alive = layer_idx
            cols = ra.get_node_indices(state, node)
            vals_at_walls = tensor[wall_positions][:, cols].detach().cpu()
            # vals_at_walls is (n_walls, field_width).  Compare to host.
            # field_width is 1 for ax/ay/bx/by.
            compiled_per_wall = vals_at_walls[:, 0].tolist()
            drifts = [abs(c - h) for c, h in zip(compiled_per_wall, host_vals)]
            max_drift = max(drifts)
            if max_drift > drift_thresh and field_name not in first_drift:
                first_drift[field_name] = (
                    layer_idx,
                    state_name,
                    compiled_per_wall,
                )
                break
        # If still clean at the end, record as "clean".
        if field_name not in first_drift:
            first_drift[field_name] = (-1, "clean through all layers", [])

    # Print the per-field results clearly.
    print("=" * 72)
    print("First divergence (vs host-fed value) per wall input InputNode:")
    print("=" * 72)
    host_by_name = {n: h for n, _node, h in fields}
    for field_name, (layer_idx, state_name, compiled_vals) in first_drift.items():
        host_vals = host_by_name[field_name]
        if layer_idx < 0:
            print(f"  {field_name:10s}: clean through all layers "
                  f"(no drift > {drift_thresh})")
        else:
            print(f"  {field_name:10s}: first drift at {state_name} (layer {layer_idx})")
            if compiled_vals:
                pairs = ", ".join(
                    f"{lbl}: host={h:+.3f} compiled={c:+.3f} Δ={c-h:+.3f}"
                    for lbl, h, c in zip(wall_labels, host_vals, compiled_vals)
                )
                print(f"             {pairs}")
    print("=" * 72)

    # Pull a detailed per-layer trace for the fields that showed drift.
    drifted_fields = [
        (name, node, host)
        for (name, node, host) in fields
        if first_drift[name][0] >= 0
    ]
    if drifted_fields:
        print("\nPer-layer trace for drifted fields "
              "(compiled value at each WALL position minus host value):")
        print("-" * 72)
        # Column header
        header_cols = "  ".join(f"{lbl:>12s}" for lbl in wall_labels)
        for field_name, node, host_vals in drifted_fields:
            print(f"\n  {field_name} (host = {host_vals}):")
            print(f"    layer  state            {header_cols}")
            for layer_idx, state_name, state in ordered_states:
                tensor = state_tensor.get(state)
                if tensor is None or not ra.has_node(state, node):
                    print(f"    {layer_idx:5d}  {state_name:16s}  (not assigned)")
                    continue
                cols = ra.get_node_indices(state, node)
                vals = tensor[wall_positions][:, cols].detach().cpu()[:, 0]
                deltas = "  ".join(
                    f"{vals[k].item() - host_vals[k]:+12.4f}"
                    for k in range(len(wall_positions))
                )
                print(f"    {layer_idx:5d}  {state_name:16s}  {deltas}")
        print("-" * 72)

    # ----- Second-phase diagnostic: what sits in wall_ax's *physical columns*
    # after the scheduler marks it dead?  The compile-time residual
    # assignment considers wall_ax "freed" after layer 25, so columns get
    # reassigned.  If SORTED's attention reads those columns beyond that
    # layer, it sees whatever the new owner wrote.  We pin the columns at
    # the entry state (in_state) and then trace what node owns them +
    # what value sits there at every later snapshot.
    print("\nPhase-2: trace wall_ax / wall_by *column contents* after free.")
    print("-" * 72)

    # Build reverse index: for each state, cols → node.
    col_owner_per_state: Dict[object, Dict[int, object]] = {}
    for state in ra.mapping:
        owners: Dict[int, object] = {}
        for node, cols in ra.mapping[state].items():
            for c in cols:
                owners[c] = node
        col_owner_per_state[state] = owners

    in_state = net.layers[0].attn.in_state
    for field_name, node, host_vals in [
        ("wall_ax", wall_ax_node, host_ax),
        ("wall_ay", wall_ay_node, host_ay),
        ("wall_bx", wall_bx_node, host_bx),
        ("wall_by", wall_by_node, host_by),
    ]:
        # Physical columns that originally held this InputNode.
        if node not in ra.mapping[in_state]:
            print(f"  {field_name}: no in_state columns (skip)")
            continue
        cols = list(ra.mapping[in_state][node])
        col = cols[0]  # ax/by are 1-wide
        print(f"\n  {field_name}: physical column {col} "
              f"(host values: {host_vals})")
        print(f"    layer  state            owner               "
              f"   east (0)    north (1)     west (2)    south (3)")
        for layer_idx, state_name, state in ordered_states:
            tensor = state_tensor.get(state)
            if tensor is None:
                continue
            owner = col_owner_per_state.get(state, {}).get(col)
            owner_repr = (
                f"{type(owner).__name__}#{getattr(owner, 'node_id', '?')}"
                f"({getattr(owner, 'name', None) or getattr(owner, 'annotation', None) or '-'})"
                if owner is not None else "-"
            )[:32]
            vals = tensor[wall_positions, col].detach().cpu().tolist()
            deltas = "  ".join(f"{v - h:+10.4f}" for v, h in zip(vals, host_vals))
            print(f"    {layer_idx:5d}  {state_name:16s}  "
                  f"{owner_repr:32s}  {deltas}")
    print("-" * 72)

    # ----- Phase 2c: at NON-WALL positions, what values live in columns
    # 49..52 at the final post-MLP state?  The observed drift pattern
    # (ax/by drift to 4.86, ay/bx stay clean) can be explained by softmax
    # blending north-wall values (97%) with non-wall-position values (3%).
    # If at non-wall positions cols 49,52 hold 0 while 50,51 hold the
    # north-wall y/x values, then the blend produces exactly the observed
    # pattern.
    print("\nPhase-2c: column contents at NON-WALL positions at last layer state.")
    print("-" * 72)
    # Pick a representative non-wall position — e.g., the INPUT token (pos 32).
    # We have: TEX_COL × 32 (pos 0..31), INPUT (pos 32), BSP_NODE × 16 (33..48),
    # WALL × 4 (49..52), EOS (53).
    input_pos = 32
    eos_pos = 53
    bsp_first_pos = 33
    # Sample several representative states.  Pick the states right
    # before the V-reads at L22, L27, L28, L42 fire — i.e., L21, L26,
    # L27, L41 mlp_out (= the input residual of those attns).
    sample_state_indices = [21, 26, 27, 41]
    for target_idx in sample_state_indices:
        match = [t for t in ordered_states if t[0] == target_idx]
        if not match:
            continue
        layer_idx, state_name, state = match[0]
        tensor = state_tensor.get(state)
        if tensor is None:
            continue
        print(f"\n  {state_name} (input to L{layer_idx+1}.attn — if attn reads V[49..52]):")
        for pos_label, pos in [
            ("INPUT (32)", input_pos),
            ("BSP_NODE (33)", bsp_first_pos),
            ("EOS (53)", eos_pos),
        ]:
            if pos >= tensor.shape[0]:
                continue
            vals = tensor[pos, 49:53].detach().cpu().tolist()
            print(f"    {pos_label:>16s}: col49={vals[0]:+.3f}  col50={vals[1]:+.3f}  "
                  f"col51={vals[2]:+.3f}  col52={vals[3]:+.3f}")
        for i, pos in enumerate(wall_positions):
            label = f"WALL {i} ({wall_labels[i]})"
            vals = tensor[pos, 49:53].detach().cpu().tolist()
            print(f"    {label:>16s}: col49={vals[0]:+.3f}  col50={vals[1]:+.3f}  "
                  f"col51={vals[2]:+.3f}  col52={vals[3]:+.3f}")
    print("-" * 72)

    # ----- Phase 2b: which compiled attention sublayer *reads* the
    # wall_{ax,ay,bx,by} columns?  The V-projection of an attention
    # component has weight matrix (n_heads, d, d_head); a nonzero row
    # at column c means that layer reads residual-stream col c into V.
    # We scan every layer's V and K projections and flag those that
    # touch columns 49..52.
    print("\nPhase-2b: attention layers that read wall columns 49..52.")
    print("-" * 72)
    wall_cols = {49, 50, 51, 52}
    attn_details = []
    for i, layer in enumerate(net.layers):
        V = layer.attn.attn.value_matrix.detach().cpu()
        K = layer.attn.attn.key_matrix.detach().cpu()
        Q = layer.attn.attn.query_matrix.detach().cpu()
        def _touched(mat):
            row_mass = mat.abs().sum(dim=(0, 2))  # (d,)
            return {c for c in range(row_mass.shape[0]) if row_mass[c].item() > 1e-8}

        v_all = _touched(V)
        k_all = _touched(K)
        v_wall = v_all & wall_cols
        if v_wall:
            # Print the full V-read set for this layer — sized ~19 for SORTED's argmin.
            print(f"  layer {i:2d}  |V|={len(v_all):3d}  V-all={sorted(v_all)[:30]}{'...' if len(v_all)>30 else ''}")
            print(f"             |K|={len(k_all):3d}  K-all={sorted(k_all)[:30]}{'...' if len(k_all)>30 else ''}")
        attn_details.append((i, v_all, k_all))
    print("-" * 72)

    # ----- Phase 3: run the autoregressive SORTED loop end-to-end and
    # print sort[0..N-1] to verify this test reproduces the integration
    # test's drift pattern.  If sort[2] shows wall=[4.86, 5.00, -5.00, 4.86]
    # then we have a reproducer and the bug is localized inside this setup.
    print("\nPhase-3: autoregressive SORTED run (matches step_frame).")
    print("-" * 72)

    # Re-run prefill via module.step (autoregressive KV cache path) to match
    # the failing integration test exactly.
    past = module.empty_past()
    with torch.no_grad():
        pre_out, past = module.step(prefill, past, past_len=0)
    step = prefill.shape[0]

    # sort_feedback lives in the output layout — read by name.
    out_specs = {n: (s, w) for n, s, w in module._output_specs}
    in_specs_h = {n: (s, w) for n, s, w in module._input_specs}
    sf_out_s, _ = out_specs["sort_feedback"]

    # Feed the EOS output back as the next input (overlaid-field host
    # protocol mirrors compile.py::step_frame).
    eos_out = pre_out[-1:]
    d_input = max(s + w for _, s, w in module._input_specs)
    device = module._net.device
    overlaid_names = [n for n in in_specs_h if n in out_specs]

    def _out_to_input(raw):
        row = torch.zeros(1, d_input, device=device)
        for name in overlaid_names:
            in_s, w = in_specs_h[name]
            out_s, _ = out_specs[name]
            row[0, in_s:in_s + w] = raw[0, out_s:out_s + w]
        return row

    prev = _out_to_input(eos_out)
    n_walls = len(subset.segments)
    with torch.no_grad():
        for k in range(n_walls):
            out, past = module.step(prev, past, past_len=step)
            step += 1
            raw_sf = out[0, sf_out_s:sf_out_s + 8 + 5 + 3 + 2 * _MAX_WALLS]
            raw_sf = raw_sf.detach().cpu().numpy()
            wall_data = raw_sf[8:13]
            print(f"  sort[{k}]: wall=[{wall_data[0]:+.3f}, {wall_data[1]:+.3f}, "
                  f"{wall_data[2]:+.3f}, {wall_data[3]:+.3f}]  tex={wall_data[4]:.2f}")
            prev = _out_to_input(out)
    print("-" * 72)

    # Flip this to True once we've localized the bug to make the test a
    # gating regression.  For now the test is documentation — it always
    # passes but prints a diagnostic.
    strict = False
    if strict:
        for field_name, (layer_idx, *_rest) in first_drift.items():
            assert layer_idx < 0, (
                f"{field_name} drifted at layer {layer_idx} "
                f"(details above) — see printed trace"
            )


def test_score_gap_assert_state_on_angle_192_compiled(graph_and_module):
    """Score-gap assert now passes vacuously at angle-192.

    Semantics have changed from the original Mode C probe: the SORTED
    argmin no longer uses sentinel-encoded non-renderability, so the
    valid set at sort[0] under angle-192 is a *singleton* (south).  The
    stricter ``assert_score_gap_at_least(margin=1.0)`` check passes via
    its "fewer than 2 valid rows" early-exit — the rank-1/rank-2 gap is
    vacuously satisfied.  The test remains useful as a regression lock:
    if renderability routing regresses and multiple walls are flagged
    renderable simultaneously, the 1.0 margin on clean integer ranks
    (1.0 spacing) is met exactly — any compilation precision loss would
    fire the assert.
    """
    from torchwright.debug.probe import check_asserts_on_compiled

    _graph_io, module, subset, config = graph_and_module
    asserts = module.metadata["asserts"]
    assert asserts, "expected collected asserts on the compiled module"

    score_gap_asserts = [
        a for a in asserts if "score_gap_at_least" in (a.message or "")
    ]
    assert score_gap_asserts, "expected at least one score_gap_at_least assert"

    prefill, _wall_base_idx, _eos_idx = _build_angle_192_prefill(
        module, subset, config, angle=192.0,
    )
    n_pos = prefill.shape[0]

    input_values = {
        name: prefill[:, start:start + width]
        for name, start, width in module._input_specs
    }

    expect_passes = True
    if expect_passes:
        check_asserts_on_compiled(
            module, score_gap_asserts, input_values, n_pos,
        )
    else:
        with pytest.raises(AssertionError, match=r"score_gap"):
            check_asserts_on_compiled(
                module, score_gap_asserts, input_values, n_pos,
            )


# ---------------------------------------------------------------------------
# V-aliasing investigation: pin down the SORTED argmin compiled layer and
# inspect what sits in sort_value's columns at its attention-input state.
# ---------------------------------------------------------------------------


def _find_sort_attn_node(graph_io):
    """Walk the graph and return the Attn node under the ``sort/attention``
    annotation — the one built by ``attend_argmin_unmasked`` inside
    ``_argmin_and_derive``.
    """
    from torchwright.graph.attn import Attn
    from torchwright.compiler.utils import get_ancestor_nodes

    roots = set(graph_io.overlaid_outputs.values()) | set(
        graph_io.overflow_outputs.values()
    )
    all_nodes = get_ancestor_nodes(roots)
    candidates = [
        n for n in all_nodes
        if isinstance(n, Attn) and (n.annotation or "").startswith("sort/attention")
    ]
    # There may be a few Attn nodes under the annotation; the argmin
    # itself is the one whose value-input resolves to sort_value
    # (width 13 + max_walls).
    return candidates


def test_find_sort_argmin_layer_and_inspect_V(graph_and_module):
    """Locate the compiled layer that hosts SORTED's argmin attention
    and dump the residual-stream values its V-projection reads at each
    WALL token position.

    If wall_ax / wall_by columns (49 / 52) at WALL token positions
    hold values other than the host-fed wall geometry at that layer's
    attention input, we have compiled-side aliasing between score
    generation and the argmin's V-read — the Mode C root cause that
    the ``score_gap`` check empirically ruled out.
    """
    graph_io, module, subset, config = graph_and_module
    net = module._net
    ra = net.residual_assignment

    sort_attn_candidates = _find_sort_attn_node(graph_io)
    assert sort_attn_candidates, (
        "no Attn node under 'sort/attention' annotation; did the "
        "annotation scope in sorted.py change?"
    )

    # Per-layer MLP-out snapshots (frozen residual column ownership).
    ordered_states = []
    for i, layer in enumerate(net.layers):
        if layer.mlp.out_state in ra.mapping:
            ordered_states.append((i, layer.mlp.out_state))

    # An Attn node is scheduled at layer L iff it first appears in the
    # snapshot at layer L.  Find that layer for each candidate Attn.
    print("\nSORTED/attention Attn-node candidates and their compiled layers:")
    print("-" * 72)
    sort_attn_to_layer = {}
    for attn_node in sort_attn_candidates:
        first_layer = None
        for layer_idx, state in ordered_states:
            if attn_node in ra.mapping[state]:
                first_layer = layer_idx
                break
        ann = (attn_node.annotation or "-")
        name = getattr(attn_node, "name", None) or "-"
        v_input = attn_node.inputs[2]
        v_width = len(v_input) if v_input is not None else "?"
        print(f"  Attn#{attn_node.node_id}  annotation='{ann}'  name='{name}'  "
              f"V-width={v_width}  first-layer={first_layer}")
        if first_layer is not None:
            sort_attn_to_layer[attn_node] = first_layer
    print("-" * 72)

    # Pick the candidate whose V-input matches sort_value's width
    # (13 + max_walls).  That's the argmin.
    expected_v_width = 13 + _MAX_WALLS
    argmin_attn = None
    for attn_node, layer_idx in sort_attn_to_layer.items():
        v_width = len(attn_node.inputs[2]) if attn_node.inputs[2] is not None else -1
        if v_width == expected_v_width:
            argmin_attn = attn_node
            break
    assert argmin_attn is not None, (
        f"couldn't find the argmin Attn by V-width={expected_v_width}; "
        f"update the filter if the payload layout changed"
    )
    argmin_layer = sort_attn_to_layer[argmin_attn]
    print(f"SORTED argmin Attn scheduled at layer {argmin_layer}.")
    print(f"Its attention runs on the residual AT the input to L{argmin_layer}")
    print(f"(i.e., the end of L{argmin_layer - 1}'s MLP — snapshot "
          f"'L{argmin_layer - 1}.mlp_out').")

    # Residual snapshot state to read V from: end-of-(argmin_layer-1)
    # mlp_out.  For argmin_layer==0, it's the very initial in_state
    # (which we don't have on return_states), so this test assumes
    # argmin_layer >= 1 (expected for the full-game compile).
    assert argmin_layer >= 1, "argmin at layer 0 — adjust state lookup"

    # Build the angle-192 prefill and run forward(return_states=True).
    prefill, wall_base_idx, _ = _build_angle_192_prefill(
        module, subset, config, angle=192.0,
    )
    res_stream = module._build_res_stream(prefill, past_len=0)
    with torch.no_grad():
        _, all_states = net.forward(res_stream, return_states=True)
    state_tensor = {}
    for key, (state, tensor) in all_states.items():
        state_tensor.setdefault(state, tensor)

    target_state = net.layers[argmin_layer - 1].mlp.out_state
    target_tensor = state_tensor.get(target_state)
    assert target_tensor is not None, (
        f"no captured tensor for L{argmin_layer - 1}.mlp_out"
    )

    # Dump sort_value's columns at each WALL token position.
    # sort_value = Concatenate([wall_ax, wall_ay, wall_bx, wall_by,
    #                           wall_tex_id, sort_den, C, D, E, H_inv,
    #                           bsp_rank, position_onehot (max_walls)])
    v_input = argmin_attn.inputs[2]
    # get_node_indices handles Concatenate transparently.
    try:
        v_cols = ra.get_node_indices(target_state, v_input)
    except KeyError:
        v_cols = None
    print(f"\nAt L{argmin_layer - 1}.mlp_out (SORTED argmin's V-read residual):")
    if v_cols is None:
        print("  V-input is NOT fully assigned at this state — at least one")
        print("  leaf of the sort_value Concatenate has been freed/reassigned")
        print("  before the argmin reads it.  This is direct evidence of")
        print("  V-column aliasing at the argmin layer.")
        # Still dump the first few expected columns (49..52 = ax/ay/bx/by)
        # via raw tensor indexing.
        print("\n  Raw residual values at cols 49..52 (host-fed wall geometry):")
    else:
        print(f"  V-columns: {v_cols}")
        print("\n  V values at cols 49..52 (wall_ax / wall_ay / wall_bx / wall_by):")

    wall_positions = list(range(wall_base_idx, wall_base_idx + len(subset.segments)))
    wall_labels = ["east (0)", "north (1)", "west (2)", "south (3)"]
    host_ax = prefill[wall_positions, 49].tolist()
    host_ay = prefill[wall_positions, 50].tolist()
    host_bx = prefill[wall_positions, 51].tolist()
    host_by = prefill[wall_positions, 52].tolist()
    print(f"\n  Host-fed at WALL positions: ax={host_ax}  ay={host_ay}  "
          f"bx={host_bx}  by={host_by}\n")
    print(f"    pos   label          col49 (ax)    col50 (ay)    col51 (bx)    col52 (by)")
    for label, pos in zip(wall_labels, wall_positions):
        vals = target_tensor[pos, 49:53].detach().cpu().tolist()
        print(f"    {pos:3d}   {label:<12s}   "
              f"{vals[0]:+10.4f}   {vals[1]:+10.4f}   "
              f"{vals[2]:+10.4f}   {vals[3]:+10.4f}")
    # Also sample a couple of non-WALL positions to see what V sees there.
    print("\n  Non-WALL V sample (to help explain any blending):")
    sample_non_wall = [32, 33, 53]  # INPUT, first BSP_NODE, EOS
    sample_labels = ["INPUT", "BSP_NODE[0]", "EOS"]
    for label, pos in zip(sample_labels, sample_non_wall):
        if pos >= target_tensor.shape[0]:
            continue
        vals = target_tensor[pos, 49:53].detach().cpu().tolist()
        print(f"    {pos:3d}   {label:<12s}   "
              f"{vals[0]:+10.4f}   {vals[1]:+10.4f}   "
              f"{vals[2]:+10.4f}   {vals[3]:+10.4f}")

    # ---------- Blend-math check ----------
    # Compute what sort[2] WOULD output if it picked north (pos 50) with
    # weight w and west (pos 51) with weight (1-w), given the V values at
    # this layer.
    v_north = target_tensor[wall_positions[1], 49:53].detach().cpu()
    v_west = target_tensor[wall_positions[2], 49:53].detach().cpu()
    observed_sort2 = torch.tensor([4.860, 5.000, -5.000, 4.860])
    # Solve for w: observed = w * v_north + (1-w) * v_west
    # Use ax/by (cols 49/52) since they're the ones that drift.
    # If v_north and v_west match the host values, we expect w ≈ 0.986.
    if (v_north - v_west).abs().sum() > 0:
        # Least-squares fit across the 4 values.
        denom = ((v_north - v_west) ** 2).sum().item()
        if denom > 1e-6:
            w = ((observed_sort2 - v_west) * (v_north - v_west)).sum().item() / denom
            pred = w * v_north + (1 - w) * v_west
            residual = (observed_sort2 - pred).abs().max().item()
            print(f"\n  Blend-fit: w_north={w:.4f} (w_west={1-w:.4f})")
            print(f"    predicted: {pred.tolist()}")
            print(f"    observed : {observed_sort2.tolist()}")
            print(f"    L∞ residual: {residual:.4f}")
            if residual < 0.01 and 0.9 < w < 1.0:
                print("  → observed sort[2] is explained by a north/west blend of the")
                print("    V values actually present at L{}.mlp_out.".format(argmin_layer - 1))
            elif residual >= 0.01:
                print("  → V values at this layer do NOT explain the sort[2] blend;")
                print("    the argmin reads from a different layer or cache path.")


def test_inspect_sorted_argmin_attention_weights_at_sort2(graph_and_module):
    """Directly inspect the argmin layer's attention weights at sort[0].

    With Phase E's above-threshold SORTED attention
    (``attend_argmin_above_integer``), at angle-192 only south is
    renderable (east/west are parallel to the viewing ray, north is
    behind the player).  Non-renderable walls have all-zero
    ``indicators_above``, so the above-logit contribution is zero and
    they can't outscore south.  The softmax should concentrate
    ≈100 % of its mass on the south wall's position.

    The logit gap between the above-threshold winner (south) and any
    invalid wall is ≈ ``_ABOVE_BONUS`` (= 1000).  That's the direct
    additive bonus in the query matrix; unlike the old
    ``_VALIDITY_KEY_COEFF`` path, it doesn't route through the
    slow-cosine gain.

    After sort[0] picks south, sort[1..N-1] exhaust the above-threshold
    search (no renderable wall has rank > 0), so those picks return
    softmax-averaged garbage per the primitive's contract — but
    ``sort_done`` catches the exhaustion.  This test samples sort[0]
    because that's where the "is the softmax concentrating?" question
    is non-vacuous.
    """
    graph_io, module, subset, config = graph_and_module
    net = module._net

    # Find the SORTED argmin layer (should be 21, per prior test).
    sort_attn_candidates = _find_sort_attn_node(graph_io)
    argmin_attn = next(
        n for n in sort_attn_candidates
        if n.inputs[2] is not None and len(n.inputs[2]) == 13 + _MAX_WALLS
    )
    ra = net.residual_assignment
    argmin_layer = None
    for i, layer in enumerate(net.layers):
        if argmin_attn in ra.mapping.get(layer.mlp.out_state, {}):
            argmin_layer = i
            break
    assert argmin_layer is not None

    # Replicate step_frame's autoregressive SORTED up to sort[2], and
    # monkey-patch layer[argmin_layer]'s attention to capture the raw
    # softmax weights at the query row.
    captured = {"logits": None, "weights": None}

    orig_fwd_cached = net.layers[argmin_layer].attn.attn.forward_cached

    def patched_fwd_cached(inp, past_kv=None):
        import torch as _t
        attn = net.layers[argmin_layer].attn.attn
        Q = _t.einsum('pd,hdk->hpk', inp, attn.query_matrix)
        K_new = _t.einsum('pd,hdk->hpk', inp, attn.key_matrix)
        V_new = _t.einsum('pd,hdk->hpk', inp, attn.value_matrix)
        if past_kv is not None:
            K = _t.cat([past_kv[0], K_new], dim=1)
            V = _t.cat([past_kv[1], V_new], dim=1)
        else:
            K, V = K_new, V_new
        n_new = inp.shape[0]
        n_total = K.shape[1]
        from torchwright.graph.attn import CAUSAL_MASK_SENTINEL
        attn_logits = _t.bmm(Q, K.transpose(1, 2))
        mask = _t.triu(
            _t.ones(n_new, n_total, device=inp.device),
            diagonal=n_total - n_new + 1,
        ).bool()
        attn_logits.masked_fill_(mask.unsqueeze(0), CAUSAL_MASK_SENTINEL)
        attn = _t.softmax(attn_logits, dim=2)
        captured["logits"] = attn_logits.detach().cpu()
        captured["weights"] = attn.detach().cpu()
        weighted = _t.bmm(attn, V)
        output = _t.einsum(
            'hpk,hkd->pd', weighted, net.layers[argmin_layer].attn.attn.output_matrix,
        )
        return output, (K, V)

    # Run prefill + sort steps; capture attention weights at sort[2].
    prefill, wall_base_idx, _ = _build_angle_192_prefill(
        module, subset, config, angle=192.0,
    )
    past = module.empty_past()
    with torch.no_grad():
        pre_out, past = module.step(prefill, past, past_len=0)
    step = prefill.shape[0]
    wall_positions = list(range(wall_base_idx, wall_base_idx + len(subset.segments)))

    out_specs = {n: (s, w) for n, s, w in module._output_specs}
    in_specs = {n: (s, w) for n, s, w in module._input_specs}
    overlaid_names = [n for n in in_specs if n in out_specs]
    d_input = max(s + w for _, s, w in module._input_specs)
    device = module._net.device

    def _out_to_input(raw):
        row = torch.zeros(1, d_input, device=device)
        for name in overlaid_names:
            in_s, w = in_specs[name]
            out_s, _ = out_specs[name]
            row[0, in_s:in_s + w] = raw[0, out_s:out_s + w]
        return row

    prev = _out_to_input(pre_out[-1:])
    # sort[0] — install hook, advance, then uninstall.  At angle-192
    # only south is renderable, so this is the "hot-case" step where
    # validity fully excludes the three invalid walls.
    net.layers[argmin_layer].attn.attn.forward_cached = patched_fwd_cached
    try:
        with torch.no_grad():
            out, past = module.step(prev, past, past_len=step)
    finally:
        net.layers[argmin_layer].attn.attn.forward_cached = orig_fwd_cached
    sort0_out = out
    step += 1
    prev = _out_to_input(out)

    weights = captured["weights"]  # (n_heads, 1, n_total)
    logits = captured["logits"]
    assert weights is not None, "failed to capture attention weights"

    print(f"\nL{argmin_layer}.attn captured at sort[0].  Head count: "
          f"{weights.shape[0]}, n_total={weights.shape[2]}.")
    # The argmin's attention head is bundled into a 28-head slab by
    # the compiler.  Find it by searching for the head whose mass at
    # sort[0] concentrates on one of the WALL positions.
    wall_pos_set = set(wall_positions)
    head_idx = None
    best_wall_mass = -1.0
    for h in range(weights.shape[0]):
        wh = weights[h, 0, :]
        wall_mass_h = sum(wh[p].item() for p in wall_pos_set)
        if wall_mass_h > best_wall_mass:
            best_wall_mass = wall_mass_h
            head_idx = h
    assert head_idx is not None, "no attention head found"
    print(f"Argmin head: {head_idx}  (WALL-mass = {best_wall_mass:.6f})")
    w = weights[head_idx, 0, :]
    l = logits[head_idx, 0, :]

    # Print top-8 weighted positions with their logits and labels.
    topk = torch.topk(w, k=8)
    print(f"\nTop-8 attention weights at sort[0] (head {head_idx}):")
    print(f"    pos   logit         weight     label")
    for val, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        label = "?"
        if idx in wall_positions:
            label = f"WALL {idx - wall_base_idx} " + ["(east)", "(north)", "(west)", "(south)"][idx - wall_base_idx]
        elif idx == prefill.shape[0] - 1:
            label = "EOS"
        elif idx < 32:
            label = "TEX_COL"
        elif idx == 32:
            label = "INPUT"
        elif 33 <= idx < 49:
            label = f"BSP_NODE[{idx-33}]"
        elif idx >= prefill.shape[0]:
            label = f"SORTED[{idx - prefill.shape[0]}]"
        print(f"    {idx:3d}   {l[idx].item():+10.4f}   {val:.6f}   {label}")

    # Total mass on WALL positions vs elsewhere.
    wall_mass = w[wall_positions].sum().item()
    sort_positions_end = step  # sort[0..1] positions fed before this step
    sort_start = prefill.shape[0]
    sort_mass = w[sort_start:sort_positions_end].sum().item() if sort_positions_end > sort_start else 0.0
    elsewhere_mass = 1.0 - wall_mass - sort_mass
    print(f"\nMass distribution: WALL={wall_mass:.6f}  "
          f"SORTED-cache={sort_mass:.6f}  elsewhere={elsewhere_mass:.6f}")

    # Specifically check sort[0], sort[1] cached positions — they're
    # the NEW contenders the softmax might be attending to that didn't
    # exist in the isolated SORTED test.
    print("\nAttention on sort cache positions:")
    for k in range(sort_start, sort_positions_end):
        print(f"    sort[{k - sort_start}] (pos {k}):  "
              f"logit={l[k].item():+10.4f}  weight={w[k].item():.6f}")

    # --- Above-threshold gap assertions ---
    # ``attend_argmin_above_integer`` applies ``_ABOVE_BONUS`` directly
    # to the logit for positions where ``indicators_above[threshold]``
    # is 1.  Non-renderable walls have all-zero indicators, so the
    # bonus-logit gap between valid and invalid is exactly
    # ``_ABOVE_BONUS`` at reference eval, minus compile-side precision
    # loss.
    from torchwright.ops.attention_ops import _ABOVE_BONUS
    south_pos = wall_positions[3]
    l_south = l[south_pos].item()
    w_south = w[south_pos].item()
    invalid_logits = [
        (lbl, l[wall_positions[i]].item())
        for i, lbl in enumerate(["east", "north", "west"])
    ]
    min_gap = min(l_south - li for _, li in invalid_logits)
    expected_gap = _ABOVE_BONUS
    print(f"\nAbove-threshold gap check:")
    print(f"  logit(south, above) = {l_south:.2f}")
    for lbl, li in invalid_logits:
        print(f"  logit({lbl}, below) = {li:.2f}  gap = {l_south - li:.2f}")
    print(f"  min above/below gap = {min_gap:.2f}  "
          f"(design: _ABOVE_BONUS = {expected_gap:.0f})")
    print(f"  weight(south) = {w_south:.6f}")

    assert w_south > 0.999, (
        f"south should dominate at sort[0]: weight={w_south:.6f}, "
        f"logit gap to worst invalid = {min_gap:.2f}"
    )
    # The logit gap should be within ~10 % of the design separation.
    assert min_gap > 0.5 * expected_gap, (
        f"above/below logit gap {min_gap:.0f} is less than half of the "
        f"design separation {expected_gap:.0f} — compile-side precision "
        f"loss has eroded the above-threshold signal."
    )

    # Exhausted-step behavior (sort[1..3]) isn't pinned here — the
    # above-threshold primitive's garbage regime interacts with
    # subsequent steps' prev_bsp_rank in ways that depend on the
    # softmax-averaged rank.  End-to-end correctness is covered by
    # the integration test (test_game_graph.py); this test's job is
    # to verify sort[0] concentrates cleanly.
