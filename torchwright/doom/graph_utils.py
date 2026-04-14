"""Small graph-construction utilities shared across the DOOM stage files."""

import torch

from torchwright.graph import Linear, Node


def extract_from(node: Node, d_total: int, start: int, width: int, name: str) -> Node:
    """Extract ``width`` columns starting at ``start`` from a ``d_total``-wide node.

    Implemented as a single Linear with a one-hot selection matrix, so the
    compiler can fuse it with upstream/downstream linear ops.
    """
    m = torch.zeros(d_total, width)
    for i in range(width):
        m[start + i, i] = 1.0
    return Linear(node, m, name=name)
