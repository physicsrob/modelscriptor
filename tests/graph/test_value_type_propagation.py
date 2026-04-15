"""Propagation-rule tests for per-op ``compute_value_type()`` implementations.

These exercise representative subgraphs (not the full test suite). The
runtime verifier (``TW_VERIFY_VALUE_TYPES=1``) on the full suite is the
primary guard against soundness bugs.
"""

import math
import os

import pytest
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
    n = InputNode("x", d_output=4)
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
    inp = InputNode("x", d_output=2)
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


# --- Attn default vs declared --------------------------------------


def test_attn_default_uses_value_range_through_v_o():
    pe = PosEncoding(d_pos=8)
    value = LiteralValue(torch.tensor([2.0, 3.0]))  # range [2, 3]
    query_matrix = torch.zeros(8, 4)
    key_matrix = torch.zeros(8, 4)
    value_matrix = torch.eye(2, 4)
    output_matrix = torch.eye(4, 2)
    attn = Attn(
        query_in=pe,
        key_in=pe,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=value_matrix,
        output_matrix=output_matrix,
    )
    vt = attn.value_type
    # Over-approximated via V then O matrix interval arithmetic: zero-
    # padding columns in V pull the low bound down to 0, and the O
    # projection preserves that.  Still a sound superset of [2, 3].
    assert vt.value_range.contains(Range(2.0, 3.0))
    # No other properties declared without an explicit declaration.
    assert not vt.is_integer


@pytest.fixture
def verify_value_types_env():
    prior = os.environ.get("TW_VERIFY_VALUE_TYPES")
    os.environ["TW_VERIFY_VALUE_TYPES"] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("TW_VERIFY_VALUE_TYPES", None)
        else:
            os.environ["TW_VERIFY_VALUE_TYPES"] = prior


def test_runtime_verifier_passes_on_consistent_graph(verify_value_types_env):
    inp = LiteralValue(torch.tensor([1.0, 2.0, 3.0]))
    r = ReLU(inp)
    out = r.compute(n_pos=2, input_values={})
    assert out.shape == (2, 3)


def test_runtime_verifier_rejects_bad_declaration(verify_value_types_env):
    pe = PosEncoding(d_pos=8)
    value = LiteralValue(torch.tensor([0.5, 0.7]))  # non-integer
    liar = NodeValueType.integer(0, 1)  # falsely claims integer
    attn = Attn(
        query_in=pe,
        key_in=pe,
        value_in=value,
        query_matrix=torch.zeros(8, 4),
        key_matrix=torch.zeros(8, 4),
        value_matrix=torch.eye(2, 4),
        output_matrix=torch.eye(4, 2),
        declared_output_type=liar,
    )
    with pytest.raises(AssertionError):
        attn.compute(n_pos=3, input_values={})


def test_attn_declared_output_type_wins():
    pe = PosEncoding(d_pos=8)
    value = LiteralValue(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    declared = NodeValueType.integer(0, 3)
    attn = Attn(
        query_in=pe,
        key_in=pe,
        value_in=value,
        query_matrix=torch.zeros(8, 4),
        key_matrix=torch.zeros(8, 4),
        value_matrix=torch.eye(4, 4),
        output_matrix=torch.eye(4, 4),
        declared_output_type=declared,
    )
    assert attn.value_type == declared
