"""Sweep chunk_size values and report compile-time stats.

Builds + compiles the DOOM game graph at several patch heights and
prints a table showing:

    rp            rows per patch
    shards        H // rp
    positions     W * shards (max_seq_len the runtime would need)
    layers        len(compiled.layers)
    peak_d        max stream in/out over layers (minimum viable d_model)
    layer_params  sum over layers
    time          sum of per-layer compile times

Compile-only: does not run inference, does not export ONNX. The load
bearing question at full DOOM dims is "at what rp does peak_d drop
below 8192" — that's the shipping configuration.

Usage:
    # Fixture dims (fast, no WAD needed)
    python -m torchwright.doom.profile_patches

    # Specific patch sizes
    python -m torchwright.doom.profile_patches --rows-per-patch 1 4 6

    # Full DOOM dims
    python -m torchwright.doom.profile_patches \\
        --width 320 --height 200 --fov 64 --tex-size 64
"""

import argparse
import io
import re
import sys
import time
from typing import List, Optional, Tuple

from torchwright.compiler.forward.compile import forward_compile
from torchwright.doom.game_graph import build_game_graph
from torchwright.reference_renderer.scenes import (
    box_room_textured,
    multi_room_textured,
)
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig

# Matches the per-layer verbose line from forward_compile.
#
# Example with dense layer (density >10%):
#   "  0         30 ops     18,677,788/2,516,... (74.2%)  656/20480 ( 3%)  674/20480 ( 3%)  2270.8ms ..."
# Example with sparse layer (density <10%):
#   " 46          7 ops  1,835,520/33,560,576 ( 5.5%)    267/2048 (13%) ..."
#
# The density is printed as {:>4.1f}%, so densities <10.0% are prefixed
# with a space — `( 5.5%)`, not `(5.5%)`. Easy to miss; cost us a round
# of analysis before we caught it.
_LAYER_LINE_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+ops\s+"
    r"([\d,]+)/[\d,]+\s*\(\s*[\d.]+%\)\s+"
    r"(\d+)/\d+\s*\(\s*\d+%\)\s+"
    r"(\d+)/\d+\s*\(\s*\d+%\)\s+"
    r"([\d.]+)ms"
)


def _divisors(n: int) -> List[int]:
    return sorted({d for d in range(1, n + 1) if n % d == 0}, reverse=True)


# ---------------------------------------------------------------------------
# Cost model
#
# The profile measures peak_d (the minimum viable d_model) at a fixed
# provisioned d. A real deployment would ship at d_ship = peak_d rounded
# up to a d_head-aligned width, so the cost model uses d_ship everywhere.
#
# Two bounds on per-frame wall time on an H100-class GPU:
#
#   compute_s  = total FLOPs / fp16 tensor-core throughput
#   mem_s      = total HBM bytes / HBM bandwidth
#
# Both account for the autoregressive-decode-with-KV-cache workload:
# batch=1, one new position per step, attending to all past positions
# via a cached K/V. That matters a lot — arithmetic intensity is ~1
# FLOP/byte for both matmuls and attention at batch=1, so the
# memory-bound regime rules at every rp on an H100 (~295 FLOPs/byte
# machine ratio).
#
# Per step, per layer:
#
#   matmul reads    = 12·d² bytes            (QKVO + W1/W2 weights, fp16)
#   matmul FLOPs    = 12·d²                  (6 matrices × 2·d²)
#   KV-past reads   = 4·past_len·d bytes     (past K + past V, fp16)
#   KV-new writes   = 4·d bytes              (append new K, V)
#   attention FLOPs = 4·past_len·d           (Q·K^T + attn·V)
#
# Summed over past_len = 0..P-1 (per layer):
#
#   matmul bytes    = 12·d²·P
#   matmul FLOPs    = 12·d²·P
#   KV  bytes       = 2·d·P·(P+1)            (past reads + new writes)
#   attention FLOPs = 2·d·P·(P-1)
#
# Multiplied by n_layers and divided by throughput to get the two bounds.
# For P ≫ d, KV bytes swamp matmul bytes — this is why rp=1 is NOT
# the minimum-wall-time config on an H100, even though it has the
# smallest model.
# ---------------------------------------------------------------------------

# H100 SXM5 nominal peak, fp16 tensor cores without sparsity.
GPU_FP16_TFLOPS = 989.0
# H100 SXM5 HBM3 bandwidth.
GPU_HBM_GBPS = 3350.0


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _cost_model(
    peak_d: int,
    n_layers: int,
    total_positions: int,
    d_head: int,
) -> dict:
    """Physics-of-a-GPU estimate of per-frame inference time.

    Uses the torchwright per-layer weight shape `6*d*d + 2*d`
    (QKVO attention + W1/W2 MLP, no 4x MLP expansion ratio).
    Accounts for KV cache reads: at batch=1 the cached K/V traffic
    is the dominant cost for small d + large P.
    """
    d_ship = max(_round_up(peak_d, d_head), d_head)
    d = d_ship
    L = n_layers
    P = total_positions

    # --- Dense fp16 weight footprint (informational, separate from traffic) ---
    params_per_layer = 6 * d * d + 2 * d
    dense_params_total = L * params_per_layer
    dense_bytes_fp16 = dense_params_total * 2
    dense_gb = dense_bytes_fp16 / (1024**3)

    # --- Compute: matmul + attention ---
    matmul_flops = L * 12 * d * d * P
    attn_flops = L * 2 * d * P * (P - 1)  # 4·d·past_len summed over 0..P-1
    total_flops = matmul_flops + attn_flops

    # --- Memory traffic: weight reads + KV reads + KV writes ---
    # Per-step weight reads (fp16). The +2·d·2 accounts for bias vectors
    # alongside the 6·d² matmul weights.
    weight_bytes_per_step = params_per_layer * 2
    weight_bytes = L * weight_bytes_per_step * P

    # KV traffic: at step t, read 4·t·d bytes of past K+V and write 4·d
    # bytes of new K+V. Summed over t=0..P-1 that's 4·d·P·(P+1)/2·(read+write).
    # The closed form: 2·d·P·(P+1) bytes per layer.
    kv_bytes = L * 2 * d * P * (P + 1)

    total_bytes = weight_bytes + kv_bytes

    compute_s = total_flops / (GPU_FP16_TFLOPS * 1e12)
    mem_s = total_bytes / (GPU_HBM_GBPS * 1e9)

    # Also report the split so the dominant term is visible in the table.
    weight_s = weight_bytes / (GPU_HBM_GBPS * 1e9)
    kv_s = kv_bytes / (GPU_HBM_GBPS * 1e9)

    return {
        "d_ship": d_ship,
        "dense_gb": dense_gb,
        "compute_s": compute_s,
        "mem_s": mem_s,
        "weight_s": weight_s,
        "kv_s": kv_s,
    }


def _parse_layer_stats(stdout_text: str) -> List[Tuple[int, int, int, int, int, float]]:
    """Return (layer_idx, n_ops, layer_params, stream_in, stream_out, time_ms)."""
    rows = []
    for line in stdout_text.splitlines():
        m = _LAYER_LINE_RE.match(line)
        if m:
            rows.append(
                (
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3).replace(",", "")),
                    int(m.group(4)),
                    int(m.group(5)),
                    float(m.group(6)),
                )
            )
    return rows


def _free_layer_weights(_i: int, layer) -> None:
    """Null out a freshly-compiled layer's weight tensors.

    Called by forward_compile right after each layer's stats are
    printed, so the parser in _profile_one still sees them. Keeps peak
    memory to ~1 layer's weights regardless of depth — critical at
    larger d, where 6·d²·4 bytes per layer adds up fast.
    """
    attn = layer.attn.attn
    mlp = layer.mlp
    attn.query_matrix = None
    attn.key_matrix = None
    attn.value_matrix = None
    attn.output_matrix = None
    mlp.linear1.output_matrix = None
    mlp.linear1.output_bias = None
    mlp.linear2.output_matrix = None
    mlp.linear2.output_bias = None


def _profile_one(
    segments,
    textures,
    config: RenderConfig,
    max_coord: float,
    chunk_size: int,
    d: int,
    d_head: int,
) -> dict:
    graph_io, pos_encoding = build_game_graph(
        config,
        textures,
        max_walls=max(8, len(segments)),
        max_coord=max_coord,
        move_speed=0.3,
        turn_speed=4,
        chunk_size=chunk_size,
    )
    output_node = graph_io.concat_output()

    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = buf
    t0 = time.perf_counter()
    try:
        net = forward_compile(
            d=d,
            d_head=d_head,
            output_node=output_node,
            pos_encoding=pos_encoding,
            verbose=True,
            max_layers=400,
            device=None,
            on_layer_compiled=_free_layer_weights,
        )
    finally:
        sys.stdout = real_stdout
    wall_time = time.perf_counter() - t0

    layer_rows = _parse_layer_stats(buf.getvalue())
    n_layers = len(net.layers)
    if not layer_rows:
        raise RuntimeError(
            "Could not parse layer stats from forward_compile verbose output"
        )
    # Sanity: the parser must catch every layer line. A silent undercount
    # here once cost us an entire round of wrong recommendations — density
    # <10% prints as "( 5.5%)" (leading space), and the regex needs to
    # allow that. If this assertion fires, fix the regex, don't just
    # accept the shorter list.
    if len(layer_rows) != n_layers:
        sample = "\n".join(buf.getvalue().splitlines()[-5:])
        raise RuntimeError(
            f"Parser mismatch: forward_compile reported {n_layers} layers "
            f"but regex matched only {len(layer_rows)}. Last 5 verbose "
            f"lines:\n{sample}"
        )
    layer_params = sum(r[2] for r in layer_rows)
    peak_d = max(max(r[3], r[4]) for r in layer_rows)
    compile_time = sum(r[5] for r in layer_rows) / 1000.0

    # Render is autoregressive — step count is dynamic at runtime.
    # For cost modeling, estimate max render steps (N_walls * W * ceil(H/cs)).
    W = config.screen_width
    H = config.screen_height
    max_walls_est = max(8, len(segments))
    max_render_steps = max_walls_est * W * ((H + chunk_size - 1) // chunk_size)
    total_positions = max_render_steps
    cost = _cost_model(
        peak_d=peak_d,
        n_layers=n_layers,
        total_positions=total_positions,
        d_head=16,
    )

    return {
        "cs": chunk_size,
        "chunks_per_col": (H + chunk_size - 1) // chunk_size,
        "positions": total_positions,
        "layers": n_layers,
        "peak_d": peak_d,
        "d_ship": cost["d_ship"],
        "dense_gb": cost["dense_gb"],
        "compute_s": cost["compute_s"],
        "mem_s": cost["mem_s"],
        "weight_s": cost["weight_s"],
        "kv_s": cost["kv_s"],
        "layer_params": layer_params,
        "time": compile_time,
        "wall_time": wall_time,
    }


def _print_table(rows: List[dict]) -> None:
    print(
        f"Cost model: H100 SXM5 ({GPU_FP16_TFLOPS:.0f} fp16 TFLOPS, "
        f"{GPU_HBM_GBPS:.0f} GB/s HBM), batch=1 KV-cached decode.\n"
        f"  d_ship     = round-up(peak_d, 16)\n"
        f"  dense_GB   = n_layers · (6·d² + 2·d) · 2 bytes (fp16)\n"
        f"  weight_s   = weight streaming bound (linear in P)\n"
        f"  kv_s       = past-K/V read + new-K/V write bound (~ P² at small d)\n"
        f"  mem_s      = weight_s + kv_s\n"
        f"  compute_s  = (matmul + attention) FLOPs / fp16 tensor throughput\n"
        f"  frame_s    = max(mem_s, compute_s)  — always mem_s on H100 at batch=1\n"
    )
    header = (
        f"{'cs':>4}  {'ch/col':>6}  {'positions':>9}  "
        f"{'peak_d':>7}  {'d_ship':>7}  {'dense_GB':>9}  "
        f"{'weight_s':>9}  {'kv_s':>8}  "
        f"{'compute_s':>10}  {'frame_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("error"):
            print(
                f"{r['cs']:>4}  {r['chunks_per_col']:>6}  {r['positions']:>9}  "
                f"{'—':>7}  {'—':>7}  {'—':>9}  "
                f"{'—':>9}  {'—':>8}  {'—':>10}  {'—':>8}   ({r['error']})"
            )
            continue
        frame_s = max(r["compute_s"], r["mem_s"])
        print(
            f"{r['cs']:>4}  {r['chunks_per_col']:>6}  {r['positions']:>9}  "
            f"{r['peak_d']:>7}  {r['d_ship']:>7}  "
            f"{r['dense_gb']:>8.2f}G  "
            f"{r['weight_s']:>8.2f}s  {r['kv_s']:>7.2f}s  "
            f"{r['compute_s']:>9.2f}s  {frame_s:>7.2f}s"
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Profile compile stats across chunk_size values",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--fov", type=int, default=64)
    parser.add_argument("--tex-size", type=int, default=64)
    parser.add_argument("--scene", choices=["box", "multi"], default="box")
    parser.add_argument("--wad", type=str, default="doom1.wad")
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="+",
        default=None,
        help="Chunk sizes to profile. "
        "Default: all divisors of --height (descending).",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=1024,
        help="Provisioned d_model for the compile call. Must be large "
        "enough that no config runs out of stream width; raise if "
        "you see a compile error at a tight rp. The reported "
        "peak_d is the real minimum.",
    )
    parser.add_argument("--d-head", type=int, default=16)
    args = parser.parse_args(argv)

    H = args.height
    rps = args.chunk_size or _divisors(H)

    trig_table = generate_trig_table()
    config = RenderConfig(
        screen_width=args.width,
        screen_height=H,
        fov_columns=args.fov,
        trig_table=trig_table,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )

    # Scene selection: at fixture dims we use the built-in procedural
    # textures (no WAD needed). At larger tex-size the user will
    # usually pass a real WAD.
    wad_path = args.wad if args.tex_size > 8 else None
    if args.scene == "box":
        segments, textures = box_room_textured(
            wad_path=wad_path,
            tex_size=args.tex_size,
        )
        max_coord = 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path=wad_path,
            tex_size=args.tex_size,
        )
        max_coord = 15.0

    print(
        f"Profiling {len(rps)} patch configurations at "
        f"W={args.width}, H={H}, fov={args.fov}, tex={args.tex_size}"
    )
    print()

    results: List[dict] = []
    for rp in rps:
        print(f"  compiling rp={rp} ...", flush=True)
        try:
            row = _profile_one(
                segments,
                textures,
                config,
                max_coord,
                chunk_size=rp,
                d=args.d,
                d_head=args.d_head,
            )
        except Exception as exc:
            # Capture failures per-config: we want to see how far down
            # the sweep succeeds, which is the whole point of finding
            # the minimum-viable rp.
            results.append(
                {
                    "cs": rp,
                    "chunks_per_col": (H + rp - 1) // rp,
                    "positions": max(8, len(segments))
                    * args.width
                    * ((H + rp - 1) // rp),
                    "layers": None,
                    "peak_d": None,
                    "layer_params": None,
                    "time": None,
                    "wall_time": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append(row)

    print()
    _print_table(results)


if __name__ == "__main__":
    main()
