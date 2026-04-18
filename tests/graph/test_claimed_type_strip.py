"""Strip-pass transfer: an Assert's ``claimed_type`` must survive the
``GraphAnalyzer`` scheduling-strip and appear on the wrapped node's
``value_type``.
"""

import torch

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.graph import InputNode, LiteralValue, NodeValueType, Range
from torchwright.graph.asserts import assert_integer, assert_01, assert_onehot
from torchwright.graph.value_type import tightened_with


def test_tightened_with_ors_claims_and_intersects_range():
    a = NodeValueType.bounded(-5.0, 5.0)
    b = NodeValueType.integer(0, 3)
    m = tightened_with(a, b)
    assert m.is_integer
    assert m.value_range == Range(0.0, 3.0)


def test_tightened_with_forces_range_under_binary():
    a = NodeValueType.integer(0, 10)
    b = NodeValueType.binary()
    m = tightened_with(a, b)
    assert m.is_binary and m.is_integer
    assert m.value_range == Range(0.0, 1.0)


def test_strip_transfers_integer_claim_to_wrapped_node():
    inp = InputNode("x", 3, value_range=(-100.0, 100.0))  # unknown value_type
    wrapped = assert_integer(inp)
    analyzer = GraphAnalyzer(wrapped)
    out = analyzer.get_output_node()
    assert out is inp
    assert inp.value_type.is_integer


def test_strip_transfers_binary_claim():
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    wrapped = assert_01(inp)
    GraphAnalyzer(wrapped)
    assert inp.value_type.is_binary
    assert inp.value_type.value_range == Range(0.0, 1.0)


def test_strip_transfers_one_hot_claim():
    inp = InputNode("x", 5, value_range=(-100.0, 100.0))
    wrapped = assert_onehot(inp)
    GraphAnalyzer(wrapped)
    assert inp.value_type.is_one_hot


def test_strip_chained_asserts_compose_claims():
    # Two Asserts stacked — integer range [0, 9] plus binary (0/1).
    # After stripping both should have applied to the innermost node.
    inp = InputNode("x", 2, value_range=(-100.0, 100.0))
    inner = assert_integer(inp)  # integer
    outer = assert_01(inner)  # binary (which implies integer)
    GraphAnalyzer(outer)
    vt = inp.value_type
    assert vt.is_binary and vt.is_integer
    assert vt.value_range == Range(0.0, 1.0)


def test_strip_does_not_regress_existing_inferred_type():
    # LiteralValue infers integer range [1, 3].  A redundant
    # assert_integer shouldn't weaken the range.
    lit = LiteralValue(torch.tensor([1.0, 2.0, 3.0]))
    before = lit.value_type
    wrapped = assert_integer(lit)
    GraphAnalyzer(wrapped)
    assert lit.value_type.is_integer
    # Range is intersection of [1, 3] and the Assert's [1, 3] claim
    # (assert_integer copies the input's bounds), so unchanged.
    assert lit.value_type.value_range == before.value_range
