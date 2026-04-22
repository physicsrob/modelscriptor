"""Trace the ``wall/visibility`` cond-near-±1 assert at decode step 89.

After the fix to ``assert_picked_from`` (skip inactive queries) landed,
the first remaining assert on angle=210 debug=True walkthrough sometimes
fires at player step 89 (PLAYER_ANGLE) inside ``wall/visibility``:

    Assert failed at wall/visibility: cond near ±1 (c_tol=0.005)
      (expected ||cond| - 1| <= 0.005; bad at [0]=-0.9949, [0]=-0.9949)

**Finding (2026-04-22):** the fire is INTERMITTENT across runs of the
same code on the same inputs.  Comparing multiple runs of
``scripts.probe_debug_true``:

* Some runs show prefill sort/attention ``selected_positions`` ending
  ``[...53,53,...,81,81,81,81,81]`` and step 89 fires with cond=-0.9949.
* Other runs show ``[...73,73,...,81,81,81,81,81]`` and step 89 passes
  cleanly.

The violating cond deviates from ±1 by 0.0051 vs a tolerance of 0.005
— 0.0001 (2%) over budget.  The two prefill-sort modes show
GPU FP nondeterminism (TF32, cuBLAS algorithm selection, or atomics
order) is producing enough variation in downstream cascaded ops to
shift this cond across the tolerance boundary.

Also verified that running prefill with ``debug=False`` but player
step 89 with ``debug=True`` passes reliably — suggesting the manual
per-layer walk inside ``_run_debug_checks`` for decode has a different
fp-ordering path from ``net.forward_cached``, which can also swing
a borderline cond value.

This probe was the reproducer that established the intermittency.  It
is kept under ``scripts/`` so the analysis survives the session.

Usage:
    make modal-run MODULE=scripts.probe_wall_visibility_cond
"""

import torch
import traceback

from torchwright.doom.compile import (
    compile_game,
    _build_row,
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


def _out_to_input(raw_out, module):
    in_by_name = {n: (s, w) for n, s, w in module._input_specs}
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    d_input = sum(w for _, _, w in module._input_specs)
    device = raw_out.device
    r = torch.zeros(1, d_input, device=device)
    for name, (in_s, in_w) in in_by_name.items():
        if name in out_by_name:
            os_, ow = out_by_name[name]
            r[0, in_s : in_s + in_w] = raw_out[0, os_ : os_ + ow]
    return r


def _describe(node, label, compiled):
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
    n_pos, d = val.shape
    vmin = float(val.min())
    vmax = float(val.max())
    abs_dev_from_pm1 = (val.abs() - 1.0).abs()
    max_dev = float(abs_dev_from_pm1.max()) if val.numel() > 0 else 0.0
    print(
        f"  {label}: id={nid} {tname}('{name}') ann={ann!r} vt={r}  "
        f"shape=({n_pos},{d}) min={vmin:.6f} max={vmax:.6f} "
        f"max||v|-1|={max_dev:.6f}"
    )
    for pi in range(min(n_pos, 3)):
        row = val[pi]
        print(
            f"      pos {pi}: "
            + " ".join(f"[{j}]={float(row[j]):.6f}" for j in range(min(d, 6)))
        )
    return val


def _walk(root, compiled, max_depth=10):
    """BFS upstream from root."""
    seen = set()
    queue = [(root, 0, "root")]
    order = []
    while queue:
        node, depth, tag = queue.pop(0)
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        order.append((node, depth, tag))
        if depth >= max_depth:
            continue
        for i, inp in enumerate(node.inputs):
            queue.append((inp, depth + 1, f"{tag}.inputs[{i}]"))

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

    # Run prefill WITH debug=True.  Empirically, prefill with debug=True
    # is what produces the KV cache that makes step 89 fire — running
    # prefill without debug but then player-steps with debug gets to
    # step 89 without any assert.  That itself is a finding, noted below.
    past = module.empty_past()
    out, past = module.step(prefill, past, past_len=0, debug=True)
    step = prefill.shape[0]
    out_by_name = {n: (s, w) for n, s, w in module._output_specs}
    px = float(out[-1, out_by_name["eos_resolved_x"][0]])
    py = float(out[-1, out_by_name["eos_resolved_y"][0]])
    angle = float(out[-1, out_by_name["eos_new_angle"][0]])
    print(f"EOS: px={px:.3f} py={py:.3f} angle={angle:.3f}")

    # Player steps 87, 88 (no debug), 89 with debug=True.
    for idx, (ttype, field, val) in enumerate(
        [
            (E8_PLAYER_X, "player_x", px),
            (E8_PLAYER_Y, "player_y", py),
            (E8_PLAYER_ANGLE, "player_angle", angle),
        ]
    ):
        prow = row(token_type=ttype, **{field: torch.tensor([val])})
        debug = True  # debug all 3 player steps (matches probe_debug_true)
        try:
            out, past = module.step(prow, past, past_len=step, debug=debug)
            step += 1
            print(f"  Player step {step} passed (debug={debug})")
        except AssertionError as e:
            step += 1
            print(f"  !!! Player step {step} raised: {str(e)[:200]}")
            break
    else:
        print("No assert fired — unexpected; abort probe")
        return

    # Now identify the firing wall/visibility cond assert.
    candidates = []
    for a in module._asserts:
        ann = a.annotation or ""
        msg = a.message or ""
        if "wall/visibility" in ann and "cond near" in msg:
            candidates.append(a)
    print(f"\nFound {len(candidates)} asserts matching 'wall/visibility' + 'cond near'")

    target_assert = None
    for a in candidates:
        val = module.debug_value(a.inputs[0])
        if val is None:
            continue
        ok, detail = a.predicate(val)
        if not ok:
            print(
                f"  FIRING assert id={a.node_id} ann={a.annotation!r} "
                f"msg={a.message!r} detail={detail[:200]}"
            )
            target_assert = a
            break
        else:
            # suppress noise: only print first few that pass
            pass

    if target_assert is None:
        print("No firing candidate assert found; bailing")
        return

    print(
        f"\n====== Walking upstream from failing assert "
        f"id={target_assert.node_id} ======"
    )
    # The assert wraps the cond input to select().
    cond_node = target_assert.inputs[0]
    _walk(cond_node, module, max_depth=8)

    # The direct input is the cond (1-wide boolean). Print exact value.
    val = module.debug_value(cond_node)
    if val is not None:
        print(f"\nFiring cond value: shape={tuple(val.shape)}  value={val.tolist()}")
        print(f"  |cond|-1 = {(val.abs()-1.0).tolist()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
