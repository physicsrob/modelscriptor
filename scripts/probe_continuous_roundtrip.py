"""Diagnose test_emit_continuous_roundtrip[CROSS_A-test_values0] failure.

Test reports: CROSS_A at -12.1 → argmaxed VALUE_22599, expected VALUE_22855.
They differ by 256 = 2^8, i.e. exactly bit 8 of the gray code.

This script:
1. Encodes -12.1 via emit_continuous_value_embedding
2. Prints every column of the emit row
3. Compares to the expected W_EMBED[VALUE_22855] row
4. Computes dot products against VALUE_22855 and VALUE_22599 to see
   which one wins and why.
"""

from __future__ import annotations

import math
import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.doom.embedding import (
    D_EMBED,
    E8_VALUE,
    VALUE_RANGE_BY_NAME,
    W_EMBED,
    build_doom_embedding,
    embed_lookup,
    gray_code_16,
)
from torchwright.doom.thinking_readback import (
    emit_continuous_value_embedding,
    encode_value_binary,
)
from torchwright.graph import Linear
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.quantization import quantize_to_range


def main():
    name = "CROSS_A"
    test_val = -12.1
    lo, hi = VALUE_RANGE_BY_NAME[name]
    lsb = (hi - lo) / 65535.0

    # Expected k and alternates
    k_expected = int(round((test_val - lo) * 65535.0 / (hi - lo)))
    k_alternate = k_expected - 256  # differs in bit 8

    print(f"name={name}, test_val={test_val}, lo={lo}, hi={hi}, LSB={lsb:.6f}")
    print(f"  expected k = round((test_val - lo) * 65535 / (hi - lo)) = {k_expected}")
    print(f"  alternate k (bit-8 flip) = {k_alternate}")

    # Step 1: verify quantize_to_range in isolation
    print(f"\n--- Isolated quantize_to_range ---")
    pos_encoding_1 = create_pos_encoding()
    value_in_1 = create_input("value", 1)
    q_node = quantize_to_range(value_in_1, lo, hi)
    q_out = Linear(q_node, torch.eye(1), name="q_out")
    net_q = forward_compile(
        d=512, d_head=32, output_node=q_out, pos_encoding=pos_encoding_1,
        verbose=False,
    )
    vals = {"value": torch.tensor([[test_val]])}
    q_val = net_q.compute(1, vals)[q_out].squeeze().item()
    print(f"  quantize_to_range({test_val}, {lo}, {hi}) = {q_val} (expected ~{(test_val-lo)*65535/(hi-lo)})")

    # Step 1c: multiply_const with direct q input (bypass quantize_to_range)
    for d_test in (512,):
        print(f"\n--- Direct multiply_const(q, 2/131072) test at d={d_test} (value_range=0..65535) ---")
        pos_encoding_3 = create_pos_encoding()
        q_direct = create_input("q_direct", 1, value_range=(0.0, 65535.0))
        from torchwright.ops.arithmetic_ops import multiply_const as mc_op
        out_mc = mc_op(q_direct, 2.0 / 131072.0)
        out_mc_wrap = Linear(out_mc, torch.eye(1), name="out_mc_wrap")
        net_mc = forward_compile(
            d=d_test, d_head=32, output_node=out_mc_wrap,
            pos_encoding=pos_encoding_3, verbose=False,
        )
        vals = {"q_direct": torch.tensor([[22855.33]])}
        mc_result = net_mc.compute(1, vals)[out_mc_wrap].squeeze().item()
        expected_mc = 22855.33 * 2.0 / 131072.0
        print(f"  multiply_const(22855.33, 2/131072) @ d={d_test} = {mc_result}  (expected {expected_mc})")
        print(f"  diff = {mc_result - expected_mc}")

        # Also try smaller q
        vals = {"q_direct": torch.tensor([[100.0]])}
        mc_result = net_mc.compute(1, vals)[out_mc_wrap].squeeze().item()
        expected_mc = 100.0 * 2.0 / 131072.0
        print(f"  multiply_const(100.0, 2/131072) @ d={d_test} = {mc_result}  (expected {expected_mc})")
        print(f"  diff = {mc_result - expected_mc}")

        # Larger q
        vals = {"q_direct": torch.tensor([[65535.0]])}
        mc_result = net_mc.compute(1, vals)[out_mc_wrap].squeeze().item()
        expected_mc = 65535.0 * 2.0 / 131072.0
        print(f"  multiply_const(65535, 2/131072) @ d={d_test} = {mc_result}  (expected {expected_mc})")
        print(f"  diff = {mc_result - expected_mc}")

        # Fractional q (matches the failing case)
        for q_test in [22855.0, 22855.1, 22855.2, 22855.25, 22855.3, 22855.33, 22855.5, 22856.0, 22855.33000001, 22855.33000002]:
            vals = {"q_direct": torch.tensor([[q_test]])}
            mc_result = net_mc.compute(1, vals)[out_mc_wrap].squeeze().item()
            expected_mc = q_test * 2.0 / 131072.0
            print(f"  multiply_const({q_test}, 2/131072) @ d={d_test} = {mc_result:.6f}  expected={expected_mc:.6f}  diff={mc_result - expected_mc:+.6f}")

        # Use full-precision torch value (avoid string→float loss)
        for q_test in [torch.tensor([[22855.0]]), torch.tensor([[22855.001]]),
                       torch.tensor([[22855.5]]), torch.tensor([[22855.9999]])]:
            vals = {"q_direct": q_test}
            mc_result = net_mc.compute(1, vals)[out_mc_wrap].squeeze().item()
            expected_mc = q_test.item() * 2.0 / 131072.0
            print(f"  multiply_const({q_test.item()}, 2/131072) @ d={d_test} = {mc_result:.6f}  expected={expected_mc:.6f}  diff={mc_result - expected_mc:+.6f}")

    # Step 1b: quantize + affine (matches encoder input handling)
    print(f"\n--- Quantize + encoder affine to x (raw slot input) ---")
    from torchwright.ops.arithmetic_ops import add_const, clamp, multiply_const
    from torchwright.graph import Concatenate
    pos_encoding_2 = create_pos_encoding()
    value_in_2 = create_input("value", 1)
    q_node_2 = quantize_to_range(value_in_2, lo, hi)
    q_clamped = clamp(q_node_2, 0.0, 65535.0)
    x_raw = add_const(multiply_const(q_clamped, 2.0 / 131072.0), 1.0 / 131072.0)
    x = clamp(x_raw, 0.0, 1.0)
    combined = Concatenate([q_clamped, x_raw, x])
    combined_out = Linear(combined, torch.eye(3), name="combined_out")
    net_x = forward_compile(
        d=512, d_head=32, output_node=combined_out,
        pos_encoding=pos_encoding_2, verbose=False,
    )
    vals = {"value": torch.tensor([[test_val]])}
    out = net_x.compute(1, vals)[combined_out].squeeze()
    q_val_2 = out[0].item()
    x_raw_val = out[1].item()
    x_val = out[2].item()
    print(f"  q_clamped = {q_val_2}  (expected ~22855.33)")
    print(f"  x_raw = (2q+1)/131072 = {x_raw_val}  (expected ~0.34878)")
    print(f"  x = clamp(x_raw, 0, 1) = {x_val}")

    # Step 2: combined pipeline
    print(f"\n--- Full emit pipeline ---")
    # Compile minimal encode graph
    pos_encoding = create_pos_encoding()
    value_in = create_input("value", 1)
    emitted = emit_continuous_value_embedding(value_in, name)
    emit_node = Linear(emitted, torch.eye(D_EMBED), name="emit_passthrough")
    net = forward_compile(
        d=512, d_head=32, output_node=emit_node, pos_encoding=pos_encoding,
        verbose=False,
    )
    print(f"  D_EMBED = {D_EMBED}")
    print(f"  W_EMBED shape: {W_EMBED.shape}")

    vals = {"value": torch.tensor([[test_val]])}
    emit_row = net.compute(1, vals)[emit_node].squeeze(0)
    print(f"\nemit_row shape: {emit_row.shape}")
    print(f"emit_row: {emit_row.cpu().numpy().round(4).tolist()}")

    # Compare columns to expected and alternate
    w_expected = W_EMBED[k_expected].cpu()
    w_alternate = W_EMBED[k_alternate].cpu()

    emit_row_cpu = emit_row.cpu()

    print(f"\nColumn-by-column comparison:")
    print(f"  col  emit        W[VAL_{k_expected}]  W[VAL_{k_alternate}]  diff*W_exp  diff*W_alt")
    diff_exp = 0.0
    diff_alt = 0.0
    for c in range(D_EMBED):
        e = emit_row_cpu[c].item()
        we = w_expected[c].item()
        wa = w_alternate[c].item()
        de = e * we
        da = e * wa
        diff_exp += de
        diff_alt += da
        label = ""
        if c < 8:
            label = "E8"
        elif c == 8:
            label = "raw"
        elif 9 <= c < 25:
            label = f"gray_bit_{c-9}"
        elif c == 25:
            label = "K"
        elif c == 26:
            label = "K_NS"
        print(f"  {c:3d}  {e:+10.5f}  {we:+10.5f}  {wa:+10.5f}  {de:+10.5f}  {da:+10.5f}  [{label}]")

    print(f"\nTotal dot products:")
    print(f"  emit . W[VALUE_{k_expected}] = {diff_exp:.5f}")
    print(f"  emit . W[VALUE_{k_alternate}] = {diff_alt:.5f}")
    print(f"  winner: VALUE_{k_expected if diff_exp > diff_alt else k_alternate}")

    # Full argmax
    logits = emit_row_cpu @ W_EMBED.T.cpu()
    k_argmax = int(logits.argmax().item())
    print(f"\nFull argmax: VALUE_{k_argmax} (logit {logits[k_argmax].item():.5f})")
    # Top 5
    top5 = torch.topk(logits, 5)
    print(f"Top-5 argmax candidates:")
    for rank, (v, i) in enumerate(zip(top5.values.tolist(), top5.indices.tolist())):
        print(f"  rank {rank}: VALUE_{i}  logit={v:.5f}")

    # Check bit 8 specifically
    print(f"\nBit 8 of gray code (differs between VALUE_{k_expected} and VALUE_{k_alternate}):")
    col_bit8 = 9 + 8  # gray_start + 8
    print(f"  emit.col_{col_bit8} = {emit_row_cpu[col_bit8].item():.6f}")
    print(f"  W[VALUE_{k_expected}].col_{col_bit8} = {w_expected[col_bit8].item():.6f}")
    print(f"  W[VALUE_{k_alternate}].col_{col_bit8} = {w_alternate[col_bit8].item():.6f}")

    # Check what gray_code_16 returns for the expected k
    gc_expected = gray_code_16(k_expected)
    gc_alternate = gray_code_16(k_alternate)
    print(f"  gray_code_16({k_expected})[8] = {gc_expected[8].item():.6f}")
    print(f"  gray_code_16({k_alternate})[8] = {gc_alternate[8].item():.6f}")


if __name__ == "__main__":
    main()
