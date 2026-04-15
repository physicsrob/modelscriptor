"""Tests for SiblingClusterAnalyzer.

Verify the static pattern detection for admission control: what gets
recognized as a cluster, what doesn't, and what node-to-chain mapping
results.
"""

import torch

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.sibling_clusters import (
    SiblingClusterAnalyzer,
)
from torchwright.graph import Concatenate, Linear
from torchwright.graph.misc import InputNode


def _linear(inp, d_out, name=""):
    return Linear(
        inp, torch.zeros(len(inp), d_out), torch.zeros(d_out), name=name,
    )


def test_detects_wide_sibling_cluster():
    """6 parallel width-64 branches feeding one Concatenate → 1 cluster."""
    shared_in = InputNode(1, name="shared")
    terminals = []
    for i in range(6):
        # Each branch: shared_in -> Linear(→64) -> Linear(→3)
        wide = _linear(shared_in, 64, name=f"wide_{i}")
        narrow = _linear(wide, 3, name=f"narrow_{i}")
        terminals.append(narrow)
    join = Concatenate(terminals)

    graph = GraphAnalyzer(join)
    clusters = SiblingClusterAnalyzer(graph).analyze()

    assert len(clusters.clusters) == 1
    (cluster,) = clusters.clusters.values()
    assert len(cluster.chains) == 6
    assert cluster.peak_chain_width == 64


def test_no_cluster_when_branches_narrow():
    """6 branches but each max width=1 → rejected by min_peak_width."""
    shared_in = InputNode(1, name="shared")
    terminals = []
    for i in range(6):
        n = _linear(shared_in, 1, name=f"scalar_{i}")
        terminals.append(n)
    join = Concatenate(terminals)

    graph = GraphAnalyzer(join)
    clusters = SiblingClusterAnalyzer(graph, min_peak_width=32).analyze()
    assert clusters.is_empty()


def test_no_cluster_when_few_branches():
    """3 wide branches — below default min_chains=4, rejected."""
    shared_in = InputNode(1, name="shared")
    terminals = []
    for i in range(3):
        wide = _linear(shared_in, 64, name=f"wide_{i}")
        terminals.append(wide)
    join = Concatenate(terminals)

    graph = GraphAnalyzer(join)
    clusters = SiblingClusterAnalyzer(graph).analyze()
    assert clusters.is_empty()


def test_custom_thresholds():
    """With min_chains=3, the 3-branch case should be detected."""
    shared_in = InputNode(1, name="shared")
    terminals = []
    for i in range(3):
        wide = _linear(shared_in, 64, name=f"wide_{i}")
        terminals.append(wide)
    join = Concatenate(terminals)

    graph = GraphAnalyzer(join)
    clusters = SiblingClusterAnalyzer(graph, min_chains=3).analyze()
    assert len(clusters.clusters) == 1


def test_node_to_chain_mapping():
    """Each branch-exclusive node should map back to its (cluster, chain)."""
    shared_in = InputNode(1, name="shared")
    branches = []
    for i in range(4):
        wide = _linear(shared_in, 64, name=f"wide_{i}")
        narrow = _linear(wide, 3, name=f"narrow_{i}")
        branches.append((wide, narrow))
    join = Concatenate([b[1] for b in branches])

    graph = GraphAnalyzer(join)
    clusters = SiblingClusterAnalyzer(graph).analyze()

    assert len(clusters.clusters) == 1
    (cluster_id,) = clusters.clusters.keys()

    # Each wide and narrow node should be mapped; they belong to their
    # own chain, and the chain ids should be distinct.
    seen_chain_ids = set()
    for wide, narrow in branches:
        assert wide in clusters.node_to_chain
        assert narrow in clusters.node_to_chain
        c1, ch1 = clusters.node_to_chain[wide]
        c2, ch2 = clusters.node_to_chain[narrow]
        assert c1 == c2 == cluster_id
        assert ch1 == ch2
        seen_chain_ids.add(ch1)
    assert len(seen_chain_ids) == 4


def test_terminal_mapping():
    """The branch terminal (direct input to join) is registered separately."""
    shared_in = InputNode(1, name="shared")
    terminals = []
    for i in range(4):
        wide = _linear(shared_in, 64, name=f"wide_{i}")
        narrow = _linear(wide, 3, name=f"narrow_{i}")
        terminals.append(narrow)
    join = Concatenate(terminals)

    graph = GraphAnalyzer(join)
    clusters = SiblingClusterAnalyzer(graph).analyze()

    # All four narrows should be registered as terminals.
    for narrow in terminals:
        assert narrow in clusters.terminal_to_chain


def test_diamond_dependency_excluded():
    """A node shared across two branches is excluded from both."""
    shared_in = InputNode(1, name="shared")
    # Create a shared intermediate used by two different branches.
    shared_mid = _linear(shared_in, 64, name="shared_mid")
    # Four branches, two of which use shared_mid (diamond), two independent.
    branches = []
    for i in range(2):
        narrow = _linear(shared_mid, 3, name=f"diamond_{i}")
        branches.append(narrow)
    for i in range(2):
        wide = _linear(shared_in, 64, name=f"wide_{i}")
        narrow = _linear(wide, 3, name=f"indep_{i}")
        branches.append(narrow)
    join = Concatenate(branches)

    graph = GraphAnalyzer(join)
    clusters = SiblingClusterAnalyzer(graph).analyze()

    # shared_mid should NOT be mapped to any chain (it's in two branches).
    assert shared_mid not in clusters.node_to_chain
    # If clusters were created: the two diamond branches have no
    # branch-exclusive wide nodes and may be rejected entirely.  The
    # key invariant is that shared_mid isn't misclassified.


def test_concatenate_feeding_linear_still_detected():
    """A Concatenate feeding a Linear is still a join — peak width matters."""
    shared_in = InputNode(1, name="shared")
    branches = [_linear(shared_in, 64, name=f"br_{i}") for i in range(4)]
    concat = Concatenate(branches)
    final = _linear(concat, 4, name="final")

    graph = GraphAnalyzer(final)
    clusters = SiblingClusterAnalyzer(graph).analyze()

    assert len(clusters.clusters) == 1
    (cluster,) = clusters.clusters.values()
    assert cluster.join is concat
    assert cluster.peak_chain_width == 64
