"""TDD tests for the forward compiler's WeightWriter.

Each test builds a small graph, sets up a residual stream with known values,
writes weights into one TransformerLayer, runs the forward pass, and verifies
the output matches node.compute().

Conventions:
- n_pos > 1 for all attention tests to exercise causal mask behavior.
- For add_into: the Add node must be constructed as Add(dead_addend, live_addend).
  inputs[0] is the dead addend whose columns (target_cols) are being reused.
  inputs[1] is the live addend whose values are copied via attention.
  The scheduler (Phase 3) enforces this ordering.
"""

import torch
import torch.nn.functional as F

from modelscriptor.compiler.forward.residual_map import ResidualStreamMap
from modelscriptor.compiler.forward.weight_writer import (
    AttnHeadOp,
    FFNOp,
    write_attn_sublayer,
    write_ffn_sublayer,
)
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.graph import Linear, ReLU, Attn, Add, Concatenate
from modelscriptor.graph.misc import InputNode, Constant
from modelscriptor.graph.pos_encoding import PosEncoding, attention_hardness

D = 64
D_HEAD = 16
N_POS = 4


def _make_pos_encoding():
    return PosEncoding(d_pos=D_HEAD)


def _build_residual_stream(
    residual_map: ResidualStreamMap, node_values: dict
) -> torch.Tensor:
    """Build a residual stream tensor with known values at each node's columns."""
    res = torch.zeros(N_POS, D)
    for node, values in node_values.items():
        indices = residual_map.get_indices(node)
        for i, idx in enumerate(indices):
            res[:, idx] = values[:, i]
    return res


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def test_identity_layer():
    """A layer with no ops written is pure identity via skip connections."""
    layer = TransformerLayer(D, D_HEAD)
    inp = torch.randn(N_POS, D)
    out = layer.attn.forward(inp)
    out = layer.ffn.forward(out)
    assert torch.allclose(inp, out, atol=1e-6)


# ---------------------------------------------------------------------------
# Attention — compute_attn
# ---------------------------------------------------------------------------


def test_attn_compute():
    """Compile a basic Attn node into one attention head."""
    pos = _make_pos_encoding()
    value_in = InputNode("v", 4)
    # Build an Attn node that does current-position attention and passes through value
    d_head = D_HEAD
    attn_node = Attn(
        query_in=pos,
        key_in=pos,
        value_in=value_in,
        query_matrix=attention_hardness * torch.eye(len(pos), d_head),
        key_matrix=torch.eye(len(pos), d_head),
        value_matrix=torch.eye(len(value_in), d_head),
        output_matrix=torch.eye(d_head, len(value_in)),
    )

    rmap = ResidualStreamMap(D)
    pos_cols = rmap.allocate(pos)
    v_cols = rmap.allocate(value_in)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_attn", node=attn_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    # Build input residual stream
    v_values = torch.randn(N_POS, len(value_in))
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, value_in: v_values})

    # Run attention sublayer only (skip adds input, so output cols get 0 + attn_output)
    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values})
    assert torch.allclose(result, expected, atol=1e-4)


def test_attn_compute_small_d_head():
    """Attn node with d_head smaller than layer d_head — needs padding."""
    pos = _make_pos_encoding()
    value_in = InputNode("v", 4)
    small_d_head = 8  # smaller than D_HEAD=16

    attn_node = Attn(
        query_in=pos,
        key_in=pos,
        value_in=value_in,
        query_matrix=attention_hardness * torch.eye(len(pos), small_d_head),
        key_matrix=torch.eye(len(pos), small_d_head),
        value_matrix=torch.eye(len(value_in), small_d_head),
        output_matrix=torch.eye(small_d_head, len(value_in)),
    )

    rmap = ResidualStreamMap(D)
    pos_cols = rmap.allocate(pos)
    v_cols = rmap.allocate(value_in)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_attn", node=attn_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    v_values = torch.randn(N_POS, len(value_in))
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, value_in: v_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values})
    assert torch.allclose(result, expected, atol=1e-4)


def test_attn_compute_shared_inputs():
    """Attn node where query_in == key_in (like get_last_value)."""
    pos = _make_pos_encoding()
    value_in = InputNode("v", 4)

    # get_last_value pattern: query and key both use pos_encoding
    attn_node = pos.get_last_value(value_in, delta_pos=-1)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(value_in)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_attn", node=attn_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    v_values = torch.randn(N_POS, len(value_in))
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, value_in: v_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values})
    assert torch.allclose(result, expected, atol=1e-4)


def test_attn_compute_multiposition():
    """Attn node with cross-position attention (get_prev_value pattern)."""
    pos = _make_pos_encoding()
    value_in = InputNode("v", 4)
    cond_in = InputNode("c", 1)

    attn_node = pos.get_prev_value(value_in, cond_in)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(value_in)
    rmap.allocate(cond_in)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_attn", node=attn_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    v_values = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]
    )
    c_values = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    pe_values = pos.compute(N_POS, {})

    # get_prev_value needs Concatenate([pos, cond]) as key_in
    # We need to also place the concatenated node's children
    concat_node = attn_node.inputs[1]  # key_in is Concatenate([pos, cond])
    res = _build_residual_stream(
        rmap,
        {
            pos: pe_values,
            value_in: v_values,
            cond_in: c_values,
        },
    )

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values, "c": c_values})
    assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Attention — compute_linear (zero-bias)
# ---------------------------------------------------------------------------


def test_linear_zero_bias():
    """Zero-bias Linear compiled via current-position attention."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4)
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_linear", node=linear_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    x_values = torch.randn(N_POS, 4)
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result, expected, atol=1e-4)


def test_linear_large_input():
    """Zero-bias Linear with input dim > d_head — requires multiple attention heads.

    This is the sum_nodes pattern from the adder: Concatenate(4 × 8-dim) → Linear.
    With d_input=32 and d_head=16, needs ceil(32/16) = 2 heads.
    """
    pos = _make_pos_encoding()
    # 4 inputs of 8 dims each, concatenated → 32-dim input
    inputs = [InputNode(f"x{i}", 8) for i in range(4)]
    cat = Concatenate(inputs)
    # Summing matrix: each output dim accumulates from all 4 inputs
    d_out = 8
    W = torch.zeros(32, d_out)
    for i in range(32):
        W[i, i % d_out] = 1.0
    linear_node = Linear(cat, W, torch.zeros(d_out), name="sum")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    for inp in inputs:
        rmap.allocate(inp)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_linear", node=linear_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    input_values = {f"x{i}": torch.randn(N_POS, 8) for i in range(4)}
    pe_values = pos.compute(N_POS, {})
    node_values = {pos: pe_values}
    for inp in inputs:
        node_values[inp] = input_values[inp.name]
    res = _build_residual_stream(rmap, node_values)

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, input_values)
    assert torch.allclose(result, expected, atol=1e-4)


def test_linear_different_dims():
    """Zero-bias Linear where input dim != output dim."""
    pos = _make_pos_encoding()
    x = InputNode("x", 8)
    W = torch.randn(8, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_linear", node=linear_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    x_values = torch.randn(N_POS, 8)
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Attention — cancel
# ---------------------------------------------------------------------------


def test_cancel():
    """Cancel a node: columns should become zero after attn sublayer."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    x_cols = rmap.allocate(x)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="cancel", node=x, target_cols=x_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    x_values = torch.randn(N_POS, 4)
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    # After attn sublayer (includes skip): x + (-x) = 0
    out = layer.attn.forward(res)
    result = out[:, x_cols]

    assert torch.allclose(result, torch.zeros_like(result), atol=1e-4)


def test_cancel_multiple():
    """Cancel two nodes in the same layer using different heads."""
    pos = _make_pos_encoding()
    a = InputNode("a", 3)
    b = InputNode("b", 5)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)
    b_cols = rmap.allocate(b)

    layer = TransformerLayer(D, D_HEAD, pos)
    ops = [
        AttnHeadOp(op_type="cancel", node=a, target_cols=a_cols),
        AttnHeadOp(op_type="cancel", node=b, target_cols=b_cols),
    ]
    write_attn_sublayer(layer, ops, rmap, pos)

    a_values = torch.randn(N_POS, 3)
    b_values = torch.randn(N_POS, 5)
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, a: a_values, b: b_values})

    out = layer.attn.forward(res)
    assert torch.allclose(out[:, a_cols], torch.zeros(N_POS, 3), atol=1e-4)
    assert torch.allclose(out[:, b_cols], torch.zeros(N_POS, 5), atol=1e-4)


# ---------------------------------------------------------------------------
# Attention — add_into
# ---------------------------------------------------------------------------


def test_add_into():
    """Add(A, B) where A is dead — write B into A's columns via skip."""
    pos = _make_pos_encoding()
    a = InputNode("a", 4)
    b = InputNode("b", 4)
    # inputs[0]=a is dead (at target_cols), inputs[1]=b is live (copied via attention)
    add_node = Add(a, b)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)  # A occupies these columns (will become Add result)
    rmap.allocate(b)

    # Simulate scheduler: reassign dead addend's columns to the Add node
    rmap.reassign(a, add_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="add_into", node=add_node, target_cols=a_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    a_values = torch.randn(N_POS, 4)
    b_values = torch.randn(N_POS, 4)
    pe_values = pos.compute(N_POS, {})
    # a's columns now belong to add_node, but still hold a's values
    res = _build_residual_stream(
        rmap, {pos: pe_values, add_node: a_values, b: b_values}
    )

    # After attn sublayer: A's columns get A + B (skip adds A, attn writes B)
    out = layer.attn.forward(res)
    result = out[:, a_cols]

    expected = add_node.compute(N_POS, {"a": a_values, "b": b_values})
    assert torch.allclose(result, expected, atol=1e-4)


def test_add_into_dead_at_inputs1():
    """Add(live, dead) where dead is inputs[1] — still works correctly.

    This matches the adder's cond_add_vector pattern: Add(inp, chain_output)
    where chain_output is dead but lives at inputs[1].
    """
    pos = _make_pos_encoding()
    live = InputNode("live", 4)
    dead = InputNode("dead", 4)
    # Dead addend is inputs[1], not inputs[0]
    add_node = Add(live, dead)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(live)
    dead_cols = rmap.allocate(dead)

    # Simulate what the scheduler does: reassign dead's columns to the Add node
    rmap.reassign(dead, add_node)

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="add_into", node=add_node, target_cols=dead_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    live_values = torch.randn(N_POS, 4)
    dead_values = torch.randn(N_POS, 4)
    pe_values = pos.compute(N_POS, {})
    # dead's columns now belong to add_node, but still hold dead's values
    res = _build_residual_stream(
        rmap, {pos: pe_values, live: live_values, add_node: dead_values}
    )

    out = layer.attn.forward(res)
    result = out[:, dead_cols]

    expected = add_node.compute(N_POS, {"live": live_values, "dead": dead_values})
    assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# FFN — compute_relu
# ---------------------------------------------------------------------------


def test_ffn_relu_chain():
    """Linear -> ReLU -> Linear chain compiled via FFN."""
    x = InputNode("x", 4)
    W1 = torch.randn(4, 8)
    b1 = torch.randn(8)
    W2 = torch.randn(8, 3)
    b2 = torch.randn(3)
    l1 = Linear(x, W1, b1, name="l1")
    r = ReLU(l1, name="r")
    l2 = Linear(r, W2, b2, name="l2")

    rmap = ResidualStreamMap(D)
    x_cols = rmap.allocate(x)
    out_cols = rmap.allocate(l2)

    ffn_slots = list(range(0, 8))  # 8 internal FFN slots for the 8-dim intermediate

    layer = TransformerLayer(D, D_HEAD)
    op = FFNOp(
        op_type="compute_relu", node=l2, target_cols=out_cols, ffn_slots=ffn_slots
    )
    write_ffn_sublayer(layer, [op], rmap)

    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    # Run FFN sublayer only
    out = layer.ffn.forward(res)
    result = out[:, out_cols]

    expected = l2.compute(N_POS, {"x": x_values})
    assert torch.allclose(result, expected, atol=1e-4)


def test_ffn_relu_chain_multiple():
    """Two L->R->L chains in same FFN, using different slot ranges."""
    x = InputNode("x", 4)
    y = InputNode("y", 3)

    # Chain 1: x -> l1a -> relu -> l1b
    l1a = Linear(x, torch.randn(4, 6), torch.randn(6), name="l1a")
    r1 = ReLU(l1a)
    l1b = Linear(r1, torch.randn(6, 2), torch.randn(2), name="l1b")

    # Chain 2: y -> l2a -> relu -> l2b
    l2a = Linear(y, torch.randn(3, 5), torch.randn(5), name="l2a")
    r2 = ReLU(l2a)
    l2b = Linear(r2, torch.randn(5, 2), torch.randn(2), name="l2b")

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    rmap.allocate(y)
    out1_cols = rmap.allocate(l1b)
    out2_cols = rmap.allocate(l2b)

    layer = TransformerLayer(D, D_HEAD)
    ops = [
        FFNOp(
            op_type="compute_relu",
            node=l1b,
            target_cols=out1_cols,
            ffn_slots=list(range(0, 6)),
        ),
        FFNOp(
            op_type="compute_relu",
            node=l2b,
            target_cols=out2_cols,
            ffn_slots=list(range(6, 11)),
        ),
    ]
    write_ffn_sublayer(layer, ops, rmap)

    x_values = torch.randn(N_POS, 4)
    y_values = torch.randn(N_POS, 3)
    res = _build_residual_stream(rmap, {x: x_values, y: y_values})

    out = layer.ffn.forward(res)

    expected1 = l1b.compute(N_POS, {"x": x_values})
    expected2 = l2b.compute(N_POS, {"y": y_values})
    assert torch.allclose(out[:, out1_cols], expected1, atol=1e-4)
    assert torch.allclose(out[:, out2_cols], expected2, atol=1e-4)


# ---------------------------------------------------------------------------
# FFN — compute_standalone_relu
# ---------------------------------------------------------------------------


def test_ffn_standalone_relu():
    """Standalone ReLU compiled via FFN with identity linear1/linear2."""
    x = InputNode("x", 4)
    relu_node = ReLU(x, name="standalone_relu")

    rmap = ResidualStreamMap(D)
    x_cols = rmap.allocate(x)
    out_cols = rmap.allocate(relu_node)

    ffn_slots = list(range(0, 4))  # 4 slots for 4-dim ReLU

    layer = TransformerLayer(D, D_HEAD)
    op = FFNOp(
        op_type="compute_standalone_relu",
        node=relu_node,
        target_cols=out_cols,
        ffn_slots=ffn_slots,
    )
    write_ffn_sublayer(layer, [op], rmap)

    # Input with mix of positive and negative values to exercise ReLU
    x_values = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0],
            [-1.0, 2.0, -3.0, 4.0],
            [0.5, -0.5, 0.0, 1.0],
            [-0.1, 0.1, -0.2, 0.2],
        ]
    )
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.ffn.forward(res)
    result = out[:, out_cols]

    expected = relu_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result, expected, atol=1e-4)


def test_ffn_standalone_relu_preserves_input():
    """Standalone ReLU doesn't corrupt the input node's columns."""
    x = InputNode("x", 4)
    relu_node = ReLU(x, name="standalone_relu")

    rmap = ResidualStreamMap(D)
    x_cols = rmap.allocate(x)
    out_cols = rmap.allocate(relu_node)

    ffn_slots = list(range(0, 4))

    layer = TransformerLayer(D, D_HEAD)
    op = FFNOp(
        op_type="compute_standalone_relu",
        node=relu_node,
        target_cols=out_cols,
        ffn_slots=ffn_slots,
    )
    write_ffn_sublayer(layer, [op], rmap)

    x_values = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0],
            [-1.0, 2.0, -3.0, 4.0],
            [0.5, -0.5, 0.0, 1.0],
            [-0.1, 0.1, -0.2, 0.2],
        ]
    )
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.ffn.forward(res)

    # Input columns should be preserved by the skip connection
    assert torch.allclose(out[:, x_cols], x_values, atol=1e-4)


# ---------------------------------------------------------------------------
# FFN — compute_constant
# ---------------------------------------------------------------------------


def test_ffn_constant():
    """Constant value written via FFN output bias."""
    const_value = torch.tensor([1.0, -2.0, 3.5])
    const = Constant(const_value)

    rmap = ResidualStreamMap(D)
    out_cols = rmap.allocate(const)

    layer = TransformerLayer(D, D_HEAD)
    op = FFNOp(
        op_type="compute_constant", node=const, target_cols=out_cols, ffn_slots=[]
    )
    write_ffn_sublayer(layer, [op], rmap)

    res = torch.zeros(N_POS, D)

    out = layer.ffn.forward(res)
    result = out[:, out_cols]

    expected = const.compute(N_POS, {})
    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Biased Linear split (attention Wx + FFN b)
# ---------------------------------------------------------------------------


def test_biased_linear_split():
    """Linear with non-zero bias: attention computes Wx, FFN adds b."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4)
    W = torch.randn(4, 3)
    b = torch.randn(3)
    linear_node = Linear(x, W, b, name="biased")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD, pos)

    # Attention writes Wx (zero-bias part)
    attn_op = AttnHeadOp(
        op_type="compute_linear", node=linear_node, target_cols=out_cols
    )
    write_attn_sublayer(layer, [attn_op], rmap, pos)

    # FFN adds bias
    ffn_op = FFNOp(
        op_type="compute_bias", node=linear_node, target_cols=out_cols, ffn_slots=[]
    )
    write_ffn_sublayer(layer, [ffn_op], rmap)

    x_values = torch.randn(N_POS, 4)
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    # Run full layer: attn sublayer then ffn sublayer
    out = layer.attn.forward(res)
    out = layer.ffn.forward(out)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Non-contiguous columns
# ---------------------------------------------------------------------------


def test_non_contiguous_columns():
    """Operations work with scattered (non-contiguous) column indices."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4)
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    # Manually create a residual map and force non-contiguous allocation
    # by allocating and freeing intermediate nodes
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)  # takes first 16 cols
    dummy1 = InputNode("d1", 2)
    rmap.allocate(x)  # takes next 4
    d1_cols = rmap.allocate(dummy1)  # takes next 2
    out_cols = rmap.allocate(linear_node)  # takes next 3
    rmap.free(dummy1)  # frees 2 cols in the middle

    # Verify output cols are non-contiguous with input cols
    # (they're in different regions of the stream)
    assert set(rmap.get_indices(x)) & set(out_cols) == set()

    layer = TransformerLayer(D, D_HEAD, pos)
    op = AttnHeadOp(op_type="compute_linear", node=linear_node, target_cols=out_cols)
    write_attn_sublayer(layer, [op], rmap, pos)

    x_values = torch.randn(N_POS, 4)
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Mixed layer (attn + FFN together)
# ---------------------------------------------------------------------------


def test_mixed_layer():
    """One layer with both attention ops and FFN ops, verifying composition."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4)

    # Attention: zero-bias linear
    W_attn = torch.randn(4, 3)
    lin_attn = Linear(x, W_attn, torch.zeros(3), name="lin_attn")

    # FFN: constant
    const_value = torch.tensor([7.0, -3.0])
    const = Constant(const_value)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    attn_out_cols = rmap.allocate(lin_attn)
    const_cols = rmap.allocate(const)

    layer = TransformerLayer(D, D_HEAD, pos)

    # Write attention ops
    attn_op = AttnHeadOp(
        op_type="compute_linear", node=lin_attn, target_cols=attn_out_cols
    )
    write_attn_sublayer(layer, [attn_op], rmap, pos)

    # Write FFN ops
    ffn_op = FFNOp(
        op_type="compute_constant", node=const, target_cols=const_cols, ffn_slots=[]
    )
    write_ffn_sublayer(layer, [ffn_op], rmap)

    x_values = torch.randn(N_POS, 4)
    pe_values = pos.compute(N_POS, {})
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    # Run full layer
    out = layer.attn.forward(res)
    out = layer.ffn.forward(out)

    # Check attention result
    expected_attn = lin_attn.compute(N_POS, {"x": x_values})
    assert torch.allclose(out[:, attn_out_cols], expected_attn, atol=1e-4)

    # Check FFN result
    expected_const = const.compute(N_POS, {})
    assert torch.allclose(out[:, const_cols], expected_const, atol=1e-4)
