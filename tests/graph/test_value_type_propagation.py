"""Propagation-rule tests for per-op ``compute_value_type()`` implementations.

These exercise representative subgraphs (not the full test suite). The
runtime verifier (``TW_VERIFY_VALUE_TYPES=1``) on the full suite is the
primary guard against soundness bugs.
"""

import math

import torch

from torchwright.graph import (
    Add,
    Attn,
    Concatenate,
    Embedding,
    InputNode,
    Linear,
    LiteralValue,
    NodeValueType,
    PosEncoding,
    Range,
    ReLU,
    ValueLogger,
)
from torchwright.graph.misc import Placeholder

# --- Leaf nodes -----------------------------------------------------


def test_input_node_is_unknown():
    n = InputNode("x", 4)
    assert n.value_type == NodeValueType.unknown()


def test_placeholder_is_unknown():
    n = Placeholder(d=3)
    assert n.value_type == NodeValueType.unknown()


def test_pos_encoding_bounded_to_pm1():
    n = PosEncoding(d_pos=8)
    vt = n.value_type
    assert vt.value_range == Range(-1.0, 1.0)
    assert not vt.is_integer


def test_literal_integer_inference():
    lit = LiteralValue(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    vt = lit.value_type
    assert vt.is_integer
    assert vt.value_range == Range(0.0, 3.0)
    assert not vt.is_binary


def test_literal_binary_inference():
    lit = LiteralValue(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    vt = lit.value_type
    assert vt.is_binary and vt.is_integer
    assert not vt.is_one_hot  # sum > 1


def test_literal_one_hot_inference():
    lit = LiteralValue(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    vt = lit.value_type
    assert vt.is_one_hot and vt.is_binary


def test_literal_sign_inference():
    lit = LiteralValue(torch.tensor([-1.0, 1.0, -1.0]))
    vt = lit.value_type
    assert vt.is_sign and vt.is_integer
    assert vt.value_range == Range(-1.0, 1.0)


def test_literal_non_integer():
    lit = LiteralValue(torch.tensor([0.3, 0.7]))
    vt = lit.value_type
    assert not vt.is_integer


def test_embedding_has_finite_range():
    emb = Embedding(vocab=["1", "2", "3"])
    vt = emb.value_type
    assert math.isfinite(vt.value_range.lo)
    assert math.isfinite(vt.value_range.hi)
    assert vt.value_range.lo <= vt.value_range.hi


# --- Add / Concatenate ---------------------------------------------


def test_add_sums_ranges_and_keeps_integer():
    a = LiteralValue(torch.tensor([1.0, 2.0]))
    b = LiteralValue(torch.tensor([3.0, 4.0]))
    s = Add(a, b)
    vt = s.value_type
    assert vt.is_integer
    assert vt.value_range == Range(1.0 + 3.0, 2.0 + 4.0)
    assert not vt.is_binary


def test_add_drops_integer_when_one_side_float():
    a = LiteralValue(torch.tensor([1.0, 2.0]))
    pe = PosEncoding(d_pos=8)
    # Widths differ, but Add doesn't enforce width symmetry on value_type
    # rules — only on compute. We'll use matched-width literals instead.
    b = LiteralValue(torch.tensor([0.5] * len(pe)))
    # fallback: use a float literal matching a's width.
    b = LiteralValue(torch.tensor([0.5, 0.5]))
    s = Add(a, b)
    assert not s.value_type.is_integer


def test_concatenate_intersects_element_props():
    a = LiteralValue(torch.tensor([0.0, 1.0]))  # binary
    b = LiteralValue(torch.tensor([2.0, 3.0]))  # integer, not binary
    c = Concatenate([a, b])
    vt = c.value_type
    assert vt.is_integer
    assert not vt.is_binary
    assert vt.value_range == Range(0.0, 3.0)


# --- Linear ---------------------------------------------------------


def test_linear_with_integer_weights_and_bounded_input():
    inp = LiteralValue(torch.tensor([1.0, 2.0, 3.0]))  # integer, range [1, 3]
    W = torch.tensor([[1.0], [0.0], [-1.0]])
    b = torch.tensor([0.0])
    lin = Linear(inp, W, b)
    vt = lin.value_type
    assert vt.is_integer
    # Range over-approximation per column: [1*1 + 0 + 3*-1, 3*1 + 0 + 1*-1] = [-2, 2]
    assert vt.value_range == Range(-2.0, 2.0)


def test_linear_non_integer_matrix_drops_integer():
    inp = LiteralValue(torch.tensor([1.0, 2.0]))
    W = torch.tensor([[0.5], [0.5]])
    lin = Linear(inp, W)
    assert not lin.value_type.is_integer


def test_linear_unbounded_input_gives_unbounded_output():
    inp = InputNode("x", 2)
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    lin = Linear(inp, W)
    vt = lin.value_type
    assert vt.value_range == Range.unbounded()


# --- ReLU -----------------------------------------------------------


def test_relu_clamps_range_and_keeps_integer():
    inp = LiteralValue(torch.tensor([-2.0, 1.0, 3.0]))
    r = ReLU(inp)
    vt = r.value_type
    assert vt.value_range == Range(0.0, 3.0)
    assert vt.is_integer


def test_relu_of_sign_becomes_binary():
    inp = LiteralValue(torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    r = ReLU(inp)
    vt = r.value_type
    assert vt.is_binary
    assert not vt.is_sign


def test_relu_preserves_one_hot():
    inp = LiteralValue(torch.tensor([0.0, 0.0, 1.0, 0.0]))
    r = ReLU(inp)
    assert r.value_type.is_one_hot


# --- ValueLogger (pass-through) ------------------------------------


def test_value_logger_passes_through():
    inp = LiteralValue(torch.tensor([0.0, 1.0]))
    vl = ValueLogger(inp, name="debug")
    assert vl.value_type == inp.value_type


# --- Attn default is unknown (claims come via Assert wrappers) -----


def test_attn_propagates_value_range_from_value_input():
    # Softmax produces non-negative weights summing to 1, so each output
    # position is a convex combination of rows of (V @ output_matrix).
    # Convex combinations don't expand per-element range, so Attn's output
    # range is bounded by the input range propagated through
    # ``value_matrix @ output_matrix``.  Structural claims (is_integer,
    # is_binary, is_one_hot) are NOT preserved — those callers should
    # wrap with assert_matches_value_type.
    from torchwright.graph.value_type import Range

    pe = PosEncoding(d_pos=8)
    value = LiteralValue(torch.tensor([2.0, 3.0]))
    attn = Attn(
        query_in=pe,
        key_in=pe,
        value_in=value,
        query_matrix=torch.eye(8, 2),
        key_matrix=torch.eye(8, 2),
        value_matrix=torch.eye(2),
        output_matrix=torch.eye(2),
    )
    assert attn.value_type.value_range == Range(2.0, 3.0)
    # Structural claims not carried from value_in even though the literal
    # is integer-valued — softmax mixing breaks integer invariance.
    assert not attn.value_type.is_integer


# --- Op propagation rules ----------------------------------------


def test_compare_default_is_sign():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0)
    assert out.value_type.is_sign


def test_compare_01_levels_is_binary():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0, true_level=1.0, false_level=0.0)
    assert out.value_type.is_binary


def test_compare_arbitrary_integer_levels():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0, true_level=5.0, false_level=-3.0)
    vt = out.value_type
    assert vt.is_integer
    assert vt.value_range == Range(-3.0, 5.0)


def test_compare_float_levels_no_propagation():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0, true_level=0.5, false_level=-0.5)
    assert not out.value_type.is_integer


def test_equals_vector_is_sign():
    from torchwright.ops.logic_ops import equals_vector

    inp = LiteralValue(torch.tensor([1.0, 0.0, 0.0]))
    out = equals_vector(inp, torch.tensor([1.0, 0.0, 0.0]))
    assert out.value_type.is_sign


def test_select_sign_cond_integer_branches():
    from torchwright.ops.map_select import select
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    a = LiteralValue(torch.tensor([2.0, 3.0]))
    b = LiteralValue(torch.tensor([5.0, 7.0]))
    out = select(cond, a, b)
    vt = out.value_type
    assert vt.is_integer
    assert vt.value_range.lo == 2.0
    assert vt.value_range.hi == 7.0


def test_select_sign_cond_binary_branches():
    from torchwright.ops.map_select import select
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    a = LiteralValue(torch.tensor([0.0, 1.0]))
    b = LiteralValue(torch.tensor([1.0, 0.0]))
    out = select(cond, a, b)
    assert out.value_type.is_binary


def test_select_sign_cond_onehot_branches():
    from torchwright.ops.map_select import select
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    a = LiteralValue(torch.tensor([1.0, 0.0, 0.0]))
    b = LiteralValue(torch.tensor([0.0, 1.0, 0.0]))
    out = select(cond, a, b)
    assert out.value_type.is_one_hot


def test_select_unknown_cond_no_propagation():
    from torchwright.ops.map_select import select

    cond = InputNode("cond", 1)
    a = LiteralValue(torch.tensor([2.0, 3.0]))
    b = LiteralValue(torch.tensor([5.0, 7.0]))
    out = select(cond, a, b)
    assert not out.value_type.is_integer


def test_cond_gate_sign_cond_integer_inp():
    from torchwright.ops.logic_ops import cond_gate
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    inp = LiteralValue(torch.tensor([3.0, 5.0]))
    out = cond_gate(cond, inp)
    vt = out.value_type
    assert vt.is_integer
    assert vt.value_range.lo == 0.0
    assert vt.value_range.hi == 5.0


def test_cond_gate_sign_cond_binary_inp():
    from torchwright.ops.logic_ops import cond_gate
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    inp = LiteralValue(torch.tensor([0.0, 1.0]))
    out = cond_gate(cond, inp)
    assert out.value_type.is_binary


def test_cond_gate_unknown_cond_no_propagation():
    from torchwright.ops.logic_ops import cond_gate

    cond = InputNode("cond", 1)
    inp = LiteralValue(torch.tensor([3.0, 5.0]))
    out = cond_gate(cond, inp)
    assert not out.value_type.is_integer


# --- Phase 3: in_range, map_to_table, floor/ceil ----------------


def test_in_range_is_sign():
    from torchwright.ops.map_select import in_range

    lower = LiteralValue(torch.tensor([1.0]))
    upper = LiteralValue(torch.tensor([3.0]))
    out = in_range(lower, upper, 5)
    assert out.value_type.is_sign


def test_floor_int_is_integer():
    from torchwright.ops.arithmetic_ops import floor_int
    from torchwright.graph.asserts import assert_in_range

    inp = assert_in_range(LiteralValue(torch.tensor([2.5])), 0.0, 10.0)
    out = floor_int(inp, 0, 10)
    vt = out.value_type
    assert vt.is_integer
    assert vt.value_range.lo == 0.0
    assert vt.value_range.hi == 10.0


def test_ceil_int_is_integer():
    from torchwright.ops.arithmetic_ops import ceil_int
    from torchwright.graph.asserts import assert_in_range

    inp = assert_in_range(LiteralValue(torch.tensor([2.5])), 0.0, 10.0)
    out = ceil_int(inp, 0, 10)
    vt = out.value_type
    assert vt.is_integer


def test_thermometer_floor_div_is_integer():
    from torchwright.ops.arithmetic_ops import thermometer_floor_div

    inp = LiteralValue(torch.tensor([35.0]))
    out = thermometer_floor_div(inp, 10, 100)
    vt = out.value_type
    assert vt.is_integer
    assert vt.value_range.lo == 0.0
    assert vt.value_range.hi == 10.0


# --- Guarantee level propagation ----------------------------------------


def test_compare_output_is_approximate():
    from torchwright.graph.value_type import Guarantee
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([5.0]))
    out = compare(inp, 3.0)
    assert out.value_type.is_sign is Guarantee.APPROXIMATE


def test_floor_int_is_approximate():
    from torchwright.graph.value_type import Guarantee
    from torchwright.ops.arithmetic_ops import floor_int
    from torchwright.graph.asserts import assert_in_range

    inp = assert_in_range(LiteralValue(torch.tensor([2.5])), 0.0, 10.0)
    out = floor_int(inp, 0, 10)
    assert out.value_type.is_integer is Guarantee.APPROXIMATE


def test_select_approximate_cond_demotes_output():
    from torchwright.graph.value_type import Guarantee
    from torchwright.ops.arithmetic_ops import compare
    from torchwright.ops.map_select import select

    cond = compare(LiteralValue(torch.tensor([5.0])), 3.0)
    a = LiteralValue(torch.tensor([1.0]))
    b = LiteralValue(torch.tensor([2.0]))
    out = select(cond, a, b)
    # cond is APPROXIMATE sign, both branches are ALWAYS integer
    # → output is APPROXIMATE integer
    assert out.value_type.is_integer is Guarantee.APPROXIMATE


def test_linear_preserves_approximate():
    from torchwright.graph.value_type import Guarantee
    from torchwright.ops.arithmetic_ops import floor_int
    from torchwright.graph.asserts import assert_in_range

    inp = assert_in_range(LiteralValue(torch.tensor([2.5])), 0.0, 10.0)
    idx = floor_int(inp, 0, 10)
    # Linear with integer weights preserves the input's guarantee level
    out = Linear(idx, torch.tensor([[1.0]]), torch.tensor([0.0]))
    assert out.value_type.is_integer is Guarantee.APPROXIMATE


def test_add_always_plus_approximate_gives_approximate():
    from torchwright.graph.value_type import Guarantee
    from torchwright.ops.arithmetic_ops import floor_int
    from torchwright.graph.asserts import assert_in_range

    a = LiteralValue(torch.tensor([3.0]))  # ALWAYS integer
    inp = assert_in_range(LiteralValue(torch.tensor([2.5])), 0.0, 10.0)
    b = floor_int(inp, 0, 10)  # APPROXIMATE integer
    out = Add(a, b)
    assert out.value_type.is_integer is Guarantee.APPROXIMATE
