"""Trace the `render/tex_attention` cond_gate assert upstream.

Runs angle=210 compile (fix applied — current state), runs prefill with
``module.step(..., debug=True)`` inside try/except. When the
``assert_matches_value_type`` on the cond_gate output fires, locates the
failing Assert node in ``module._asserts``, walks backward through its
inputs, and calls ``compiled.debug_value(node)`` to inspect each
upstream node's compiled value.

Usage:
    make modal-run MODULE=scripts.probe_cond_gate_tex
"""

import torch

from torchwright.doom.compile import (
    compile_game,
    _build_row,
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
)
from torchwright.doom.map_subset import build_scene_subset
from torchwright.graph.misc import Assert, DebugWatch
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.reference_renderer.render import Segment
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig

TRIG = generate_trig_table()


def _config():
    return RenderConfig(
        screen_width=64,
        screen_height=80,
        fov_columns=32,
        trig_table=TRIG,
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


def _describe(node, label, compiled):
    """Print a one-line summary of a node's compiled value."""
    val = compiled.debug_value(node)
    ann = node.annotation or "<no ann>"
    tname = type(node).__name__
    nid = node.node_id
    name = node.name or ""
    vt = node.value_type
    r = vt.value_range
    if val is None:
        print(
            f"  {label}: id={nid} {tname}('{name}') ann={ann!r} vt={r} → <no residual>"
        )
        return None
    # Summary across all positions, per column.
    n_pos, d = val.shape
    vmin = float(val.min())
    vmax = float(val.max())
    print(
        f"  {label}: id={nid} {tname}('{name}') ann={ann!r} vt={r}  shape=({n_pos},{d})"
        f"  min={vmin:.4f} max={vmax:.4f}"
    )
    # Per-position max-abs — highlight the rows that blow the range.
    abs_max_per_row = val.abs().max(dim=-1).values
    worst_rows = torch.argsort(abs_max_per_row, descending=True)[:5].tolist()
    for pi in worst_rows:
        row = val[pi]
        print(
            f"      pos {pi}: row min={float(row.min()):.4f} max={float(row.max()):.4f} "
            f"| [0]={float(row[0]):.4f}"
            + (f" [1]={float(row[1]):.4f}" if d > 1 else "")
            + (f" [last]={float(row[-1]):.4f}" if d > 2 else "")
        )
    return val


def _walk(root, compiled, max_depth=8):
    """Walk upstream from root, print each unique node's compiled value.

    BFS by graph order (so close-to-root comes first).
    """
    seen = set()
    stack = [(root, 0, "root")]
    order = []
    while stack:
        node, depth, tag = stack.pop(0)
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        order.append((node, depth, tag))
        if depth >= max_depth:
            continue
        for i, inp in enumerate(node.inputs):
            stack.append((inp, depth + 1, f"{tag}.inputs[{i}]"))

    print(
        f"\nWalking {len(order)} unique nodes upstream of root (max_depth={max_depth}):"
    )
    for node, depth, tag in order:
        label = f"  [{depth:>2}] {tag}"
        _describe(node, label, compiled)


def main():
    config = _config()
    textures = default_texture_atlas()
    segs = _segments()
    subset = build_scene_subset(segs, textures)
    print("Compiling...")
    module = compile_game(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False
    )
    print(
        f"Compiled: {len(module._net.layers)} layers, "
        f"{len(module._asserts)} asserts, {len(module._watches)} watches"
    )

    max_walls = 8
    tex_w = textures[0].shape[0]

    def row(**kwargs):
        return _build_row(module, max_walls, **kwargs)

    rows = []
    for tex_idx in range(len(textures)):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for c in range(tex_w):
            pixel_data = textures[tex_idx][c].flatten()
            rows.append(
                row(
                    token_type=E8_TEX_COL,
                    texture_id_e8=tex_e8,
                    tex_col_input=torch.tensor([float(c)]),
                    tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                )
            )
    rows.append(row(token_type=E8_INPUT))
    max_bsp_nodes = int(module.metadata["max_bsp_nodes"])
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            p = subset.bsp_nodes[i]
            nx, ny, d_ = p.nx, p.ny, p.d
        else:
            nx, ny, d_ = 0.0, 0.0, 0.0
        rows.append(
            row(
                token_type=E8_BSP_NODE,
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d_], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
            )
        )
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes], dtype=torch.float32
        )
        const = torch.tensor([float(subset.seg_bsp_consts[i])], dtype=torch.float32)
        rows.append(
            row(
                token_type=E8_WALL,
                wall_ax=torch.tensor([float(seg.ax)]),
                wall_ay=torch.tensor([float(seg.ay)]),
                wall_bx=torch.tensor([float(seg.bx)]),
                wall_by=torch.tensor([float(seg.by)]),
                wall_tex_id=torch.tensor([float(seg.texture_id)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
            )
        )
    rows.append(row(token_type=E8_EOS))
    prefill = torch.cat(rows, dim=0)
    print(f"Prefill rows: {prefill.shape[0]}")

    # Run prefill with debug=True so the assert fires and _debug_state is populated.
    past = module.empty_past()
    try:
        out, past = module.step(prefill, past, past_len=0, debug=True)
        print("Prefill debug=True passed (unexpected)")
        return
    except AssertionError as e:
        err_msg = str(e)
        print(f"Caught assert (expected): {err_msg[:200]}")

    # Locate the failing assert node.
    # The cond_gate value-type assert has message "matches NodeValueType(...)" and
    # annotation under "render/tex_attention".
    candidates = []
    for a in module._asserts:
        ann = a.annotation or ""
        msg = a.message or ""
        if "render/tex_attention" in ann and "matches NodeValueType" in msg:
            candidates.append(a)
    print(
        f"\nFound {len(candidates)} asserts matching 'render/tex_attention' + 'matches NodeValueType'"
    )

    # Pick the one that actually fires.
    target_assert = None
    for a in candidates:
        val = module.debug_value(a.inputs[0])
        if val is None:
            continue
        ok, detail = a.predicate(val)
        print(f"  assert id={a.node_id} ann={a.annotation!r} msg={a.message!r}")
        print(
            f"    vt.value_range={a.claimed_type.value_range if a.claimed_type else None}"
        )
        print(f"    predicate ok={ok} detail={detail[:200]}")
        if not ok:
            target_assert = a

    if target_assert is None:
        print("No firing candidate assert identified; bailing")
        return

    print("\n====== Walking upstream from failing assert ======")
    print(
        f"Failing Assert: id={target_assert.node_id} ann={target_assert.annotation!r} msg={target_assert.message!r}"
    )
    # The assert wraps the cond_gate's output; walk from there.
    _walk(target_assert.inputs[0], module, max_depth=6)

    # Additional per-position analysis of the cond_gate output.
    cond_gate_out = target_assert.inputs[0]
    while isinstance(cond_gate_out, (Assert, DebugWatch)):
        cond_gate_out = cond_gate_out.inputs[0]
    val = module.debug_value(cond_gate_out)
    if val is not None:
        # atol from the assert predicate is 0.1; replicate the bad-mask test.
        lo = -1.0 - 0.1
        hi = 10.0 + 0.1
        bad = (val < lo) | (val > hi)
        bad_rows = bad.any(dim=-1).nonzero(as_tuple=False).flatten().tolist()
        print(f"\nPositions with ANY out-of-range slot (n={len(bad_rows)}):")
        print(f"  {bad_rows}")
        # For the first few bad rows, show the pattern.
        for pi in bad_rows[:5]:
            row = val[pi]
            neg_20 = ((row + 20).abs() < 0.1).sum().item()
            ok_slots = ((row >= lo) & (row <= hi)).sum().item()
            print(
                f"  pos {pi}: n_slots_at_-20={neg_20}  n_slots_in_range={ok_slots}  min={float(row.min()):.4f} max={float(row.max()):.4f}"
            )


if __name__ == "__main__":
    main()
