"""Strip-pass transfer: an Assert's ``claimed_type`` must survive the
``GraphAnalyzer`` scheduling-strip and appear on the wrapped node's
``value_type``.
"""

import torch

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.graph import InputNode, LiteralValue, NodeValueType, Range
from torchwright.graph.asserts import assert_integer, assert_01, assert_onehot
from torchwright.graph.value_type import tightened_with


def test_tightened_with_intersects_range():
    a = NodeValueType.bounded(-5.0, 5.0)
    b = NodeValueType.bounded(0.0, 3.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 3.0)


def test_tightened_with_bounded_and_01():
    a = NodeValueType.bounded(0.0, 10.0)
    b = NodeValueType.bounded(0.0, 1.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 1.0)


def test_strip_transfers_integer_range_to_wrapped_node():
    inp = InputNode("x", 3, value_range=(-100.0, 100.0))
    wrapped = assert_integer(inp)
    analyzer = GraphAnalyzer(wrapped)
    out = analyzer.get_output_node()
    assert out is inp
    assert inp.value_type.value_range == Range(-100.0, 100.0)


def test_strip_transfers_binary_range():
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    wrapped = assert_01(inp)
    GraphAnalyzer(wrapped)
    assert inp.value_type.value_range == Range(0.0, 1.0)


def test_strip_transfers_onehot_range():
    inp = InputNode("x", 5, value_range=(-100.0, 100.0))
    wrapped = assert_onehot(inp)
    GraphAnalyzer(wrapped)
    assert inp.value_type.value_range == Range(0.0, 1.0)


def test_strip_chained_asserts_compose_ranges():
    inp = InputNode("x", 2, value_range=(-100.0, 100.0))
    inner = assert_integer(inp)
    outer = assert_01(inner)
    GraphAnalyzer(outer)
    vt = inp.value_type
    assert vt.value_range == Range(0.0, 1.0)


def test_strip_does_not_regress_existing_inferred_type():
    lit = LiteralValue(torch.tensor([1.0, 2.0, 3.0]))
    before = lit.value_type
    wrapped = assert_integer(lit)
    GraphAnalyzer(wrapped)
    assert lit.value_type.value_range == before.value_range
