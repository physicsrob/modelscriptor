import pytest
import torch

from modelscriptor.compiler.feature_assignment import ResidualStreamState
from modelscriptor.compiler.forward.residual_map import ResidualStreamMap
from modelscriptor.graph import Node
from modelscriptor.graph.misc import InputNode, Constant


def test_allocate_and_free():
    """Allocate a node, verify indices, free it, verify columns recovered."""
    rmap = ResidualStreamMap(64)
    node = InputNode("x", 8)

    indices = rmap.allocate(node)
    assert len(indices) == 8
    assert len(set(indices)) == 8  # all unique
    assert all(0 <= i < 64 for i in indices)
    assert rmap.get_free_count() == 56
    assert rmap.is_allocated(node)
    assert rmap.get_indices(node) == indices

    rmap.free(node)
    assert rmap.get_free_count() == 64
    assert not rmap.is_allocated(node)


def test_multiple_allocations():
    """Three nodes get non-overlapping index sets."""
    rmap = ResidualStreamMap(64)
    a = InputNode("a", 8)
    b = InputNode("b", 16)
    c = InputNode("c", 4)

    idx_a = rmap.allocate(a)
    idx_b = rmap.allocate(b)
    idx_c = rmap.allocate(c)

    # No overlap
    all_indices = set(idx_a) | set(idx_b) | set(idx_c)
    assert len(all_indices) == 8 + 16 + 4
    assert rmap.get_free_count() == 64 - 28
    assert rmap.get_allocated_nodes() == {a, b, c}


def test_full_stream():
    """Fill the stream exactly, then verify next allocation raises."""
    rmap = ResidualStreamMap(16)
    a = InputNode("a", 8)
    b = InputNode("b", 8)
    c = InputNode("c", 1)

    rmap.allocate(a)
    rmap.allocate(b)
    assert rmap.get_free_count() == 0

    with pytest.raises(ValueError):
        rmap.allocate(c)


def test_reassign():
    """Reassign columns from one node to another."""
    rmap = ResidualStreamMap(64)
    old = InputNode("old", 8)
    new = InputNode("new", 8)

    indices = rmap.allocate(old)
    rmap.reassign(old, new)

    assert not rmap.is_allocated(old)
    assert rmap.is_allocated(new)
    assert rmap.get_indices(new) == indices
    assert rmap.get_free_count() == 56  # unchanged


def test_build_feature_assignment():
    """Build a FeatureAssignment and verify get_node_indices works."""
    rmap = ResidualStreamMap(64)
    inp = InputNode("x", 8)
    const = Constant(torch.ones(4))
    out = InputNode("out", 3)

    idx_inp = rmap.allocate(inp)
    idx_const = rmap.allocate(const)
    idx_out = rmap.allocate(out)

    in_state = ResidualStreamState(name="in")
    out_state = ResidualStreamState(name="out")

    fa = rmap.build_feature_assignment(
        in_state=in_state,
        out_state=out_state,
        input_nodes=[inp, const],
        output_node=out,
    )

    assert fa.get_node_indices(in_state, inp) == idx_inp
    assert fa.get_node_indices(in_state, const) == idx_const
    assert fa.get_node_indices(out_state, out) == idx_out


def test_no_fragmentation():
    """Non-contiguous free space is usable because contiguity is not required."""
    rmap = ResidualStreamMap(32)
    a = InputNode("a", 8)
    b = InputNode("b", 8)
    c = InputNode("c", 8)

    rmap.allocate(a)
    rmap.allocate(b)
    rmap.allocate(c)
    assert rmap.get_free_count() == 8

    # Free the middle node — creates a gap
    rmap.free(b)
    assert rmap.get_free_count() == 16

    # Allocate a node larger than the gap — succeeds because columns
    # don't need to be contiguous
    d = InputNode("d", 16)
    indices = rmap.allocate(d)
    assert len(indices) == 16
    assert len(set(indices)) == 16

    # No overlap with a or c
    assert not (set(indices) & set(rmap.get_indices(a)))
    assert not (set(indices) & set(rmap.get_indices(c)))
