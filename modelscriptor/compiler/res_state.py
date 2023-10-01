from typing import Set, Dict, List, NamedTuple
from abc import ABC, abstractmethod

import torch

from modelscriptor.graph import Node, Concatenate, Linear


class AllocationError(Exception):
    ...


class ResState(ABC):
    # Represents the state of the residual stream at one point in time.
    d: int
    nodes: Set[Node]
    node_to_indices: Dict[Node, List[int]]

    def __init__(self, d):
        self.d = d
        self.nodes = set()
        self.node_to_indices = {}

    def print(self, prefix: str = ""):
        if len(prefix) and not prefix.endswith(" "):
            prefix = prefix + " "
        print(f"{prefix}Residual State", end="")
        sorted_nodes = sorted(self.nodes, key=lambda n: min(self.node_to_indices[n]))
        for node in sorted_nodes:
            print(
                f" {node} [{' '.join(str(idx) for idx in sorted(self.node_to_indices[node]))}]",
                end="",
            )
        print()

    def allocate_node(self, node: Node):
        # If the node is already allocated, this is a no-op.
        # This can happen due to skip connections
        if node in self.nodes:
            return

        used_indices = set()
        for indices in self.node_to_indices.values():
            used_indices |= set(indices)
        available_indices = {idx for idx in range(self.d) if idx not in used_indices}
        if len(node) > len(available_indices):
            raise AllocationError(
                "Insufficient space in residual stream for allocation"
            )
        indices = sorted(available_indices)[0 : len(node)]
        self.node_to_indices[node] = indices
        self.nodes.add(node)

    def connect_allocation(
        self, other_state: "ResState", other_node: Node, this_node: Node
    ):
        # other_node in other_state should have the same allocation as this_node in this state.
        existing_in_node_indices = {
            idx for indices in self.node_to_indices.values() for idx in indices
        }

        node_indices = set(other_state.node_to_indices[other_node])
        assert not bool(node_indices & existing_in_node_indices)

        self.nodes.add(this_node)
        self.node_to_indices[this_node] = other_state.node_to_indices[other_node]

    def update_from(self, other: "ResState"):
        # other represents the same state as this residual state, and it has already been
        # allocated.
        self.nodes.update(other.nodes)
        self.node_to_indices.update(other.node_to_indices)
