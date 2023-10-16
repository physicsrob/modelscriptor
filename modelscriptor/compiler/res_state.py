from collections import defaultdict
from typing import Set, Dict, List, Iterable, Optional

from modelscriptor.graph import Node, Concatenate
from modelscriptor.graph.misc import Placeholder


class AllocationError(Exception):
    ...


class ResState:
    # Represents the state of the residual stream at one point in time.
    d: int
    _node_to_indices: Dict[Node, List[int]]
    _linked_from: Optional["ResState"]

    def __init__(self, d):
        self.d = d
        self._node_to_indices = {}
        self._linked_from = None

    def _consistency_check(self):
        # This function checks that the state is consistent.
        idx_to_nodes = defaultdict(list)
        for node, indices in self._node_to_indices.items():
            for idx in indices:
                idx_to_nodes[idx].append(node)
        assert all(len(nodes) <= 1 for nodes in idx_to_nodes.values())

    def has_node_indices(self, node: Node):
        self._consistency_check()
        if node in self._node_to_indices:
            return True
        if isinstance(node, Placeholder):
            return True
        if isinstance(node, Concatenate):
            inputs = node.simplify_inputs()
            # If node is a concatenation, we have the node if we have all the inputs
            return all(self.has_node_indices(inp) for inp in inputs)
        return False

    def get_node_indices(self, node: Node) -> List[int]:
        self._consistency_check()
        if node in self._node_to_indices:
            return self._node_to_indices[node]
        elif isinstance(node, Concatenate):
            inputs = node.simplify_inputs()
            assert all(inp in self._node_to_indices for inp in inputs)
            indices = []
            for inp in inputs:
                indices += self.get_node_indices(inp)
            return indices
        elif isinstance(node, Placeholder):
            return []
        else:
            assert False

    def _get_free_indices(self) -> Set[int]:
        """
        Get all available indices that have not been allocated.
        """
        self._consistency_check()
        used_indices = set()
        for indices in self._node_to_indices.values():
            used_indices |= set(indices)
        return {idx for idx in range(self.d) if idx not in used_indices}

    def get_nodes_with_indices(self) -> Set[Node]:
        self._consistency_check()
        return set(self._node_to_indices.keys())

    def print(self, prefix: str = ""):
        self._consistency_check()
        if len(prefix) and not prefix.endswith(" "):
            prefix = prefix + " "
        print(f"{prefix}Residual Stream {id(self)=}:", end="")
        if self._linked_from:
            print(f" linked from {id(self._linked_from)}")
            return

        sorted_nodes = sorted(
            self._node_to_indices.keys(),
            key=lambda n: min(self._node_to_indices[n]),
        )
        for node in sorted_nodes:
            print(
                f" {node} [{' '.join(str(idx) for idx in sorted(self._node_to_indices[node]))}]",
                end="",
            )
        print()

    def allocate_node(self, node: Node):
        self._consistency_check()
        # If the node is already allocated, this is a no-op.
        # This can happen due to skip connections
        if self.has_node_indices(node):
            return

        if isinstance(node, Concatenate):
            inputs = node.simplify_inputs()
            for inp in inputs:
                self.allocate_node(inp)
            return

        if isinstance(node, Placeholder):
            return

        available_indices = self._get_free_indices()
        if len(node) > len(available_indices):
            raise AllocationError(
                "Insufficient space in residual stream for allocation"
            )
        indices = sorted(available_indices)[0 : len(node)]
        self._node_to_indices[node] = indices

    # def pre_allocate_node(self, node: Node):
    #     self._pre_allocated_nodes.add(node)
    #
    # def get_pre_allocated_nodes(self) -> Set[Node]:
    #     return self._pre_allocated_nodes

    def _connect_allocations(
        self, other_state: "ResState", other_nodes: List[Node], this_nodes: List[Node]
    ):
        self._consistency_check()
        # This forces the allocation for this_nodes to be the same as the allocation for other_nodes
        # in other_state.

        # Simplify all concatenations
        simplified_other_nodes = []
        for n in other_nodes:
            if isinstance(n, Concatenate):
                simplified_other_nodes += n.simplify_inputs()
            else:
                simplified_other_nodes.append(n)

        other_indices = []
        for node in simplified_other_nodes:
            other_indices += other_state.get_node_indices(node)

        simplified_this_nodes = []
        for n in this_nodes:
            if isinstance(n, Concatenate):
                simplified_this_nodes += n.simplify_inputs()
            else:
                simplified_this_nodes.append(n)

        # Assert that the allocation is the same length
        assert (
            sum(len(n) for n in simplified_this_nodes)
            == sum(len(n) for n in simplified_other_nodes)
            == len(other_indices)
        )

        offset = 0
        for node in simplified_this_nodes:
            indices = other_indices[offset : offset + len(node)]
            offset += len(node)
            self._node_to_indices[node] = indices

    def connect_allocation(
        self, other_state: "ResState", other_node: Node, this_node: Node
    ):
        self._consistency_check()
        """
        This method copies an allocation from other_state to this state, forcing this_node
        to have the same allocation as other_node on other_state.

        Since this is a bit hard to follow, let's illustrate by example:
        Let's say we have a node "Input", and a node "Output".  Let's further say that
        "Output" is ReLU("Input").

        This can be implemented in a relu layer, but doing so requires the representation of
        "Input" in the incoming residual stream to be the same as the representation of "Output"
        in the outgoing residual stream.
        """

        self._connect_allocations(other_state, [other_node], [this_node])

    def link(self, other: "ResState"):
        if id(other._node_to_indices) == id(self._node_to_indices):
            return  # Already linked
        elif self._linked_from is not None:
            self._linked_from.link(other)
        elif other._linked_from is not None:
            other._linked_from.link(self)
        else:
            other._node_to_indices = self._node_to_indices
            other._linked_from = self

    def get_min_width(self):
        if not len(self._node_to_indices):
            return 0

        return (
            max(idx for indices in self._node_to_indices.values() for idx in indices)
            + 1
        )

    def resize(self, new_d):
        self.d = new_d
