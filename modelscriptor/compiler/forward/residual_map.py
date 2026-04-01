from typing import Dict, List, Set

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignment,
    ResidualStreamState,
    simplify_nodes,
)
from modelscriptor.graph import Node, Concatenate


class ResidualStreamMap:
    """Set-based column allocator for the residual stream.

    Assigns arbitrary free columns to nodes — no contiguity required.
    Weight matrices scatter/gather via index lists, so physical adjacency
    is never needed.
    """

    def __init__(self, d: int):
        self.d = d
        self._free: Set[int] = set(range(d))
        self._node_to_indices: Dict[Node, List[int]] = {}

    def allocate(self, node: Node) -> List[int]:
        n = len(node)
        if n > len(self._free):
            raise ValueError(
                f"Cannot allocate {n} columns for {node}: "
                f"only {len(self._free)} free of {self.d}"
            )
        indices = sorted(list(self._free)[:n])
        self._free -= set(indices)
        self._node_to_indices[node] = indices
        return indices

    def free(self, node: Node):
        if node not in self._node_to_indices:
            raise KeyError(f"Node {node} is not allocated")
        self._free |= set(self._node_to_indices[node])
        del self._node_to_indices[node]

    def reassign(self, old_node: Node, new_node: Node):
        if old_node not in self._node_to_indices:
            raise KeyError(f"Node {old_node} is not allocated")
        indices = self._node_to_indices.pop(old_node)
        self._node_to_indices[new_node] = indices

    def get_indices(self, node: Node) -> List[int]:
        return self._node_to_indices[node]

    def get_node_indices(self, node: Node) -> List[int]:
        """Get column indices for a node, resolving through Concatenate.

        Unlike get_indices(), this handles Concatenate nodes by gathering
        the indices of their children in order. Needed by the weight writer
        when an Attn node's input is a Concatenate (e.g. get_prev_value).
        """
        if isinstance(node, Concatenate):
            indices = []
            for child in simplify_nodes([node]):
                indices += self.get_indices(child)
            return indices
        return self.get_indices(node)

    def is_allocated(self, node: Node) -> bool:
        return node in self._node_to_indices

    def get_free_count(self) -> int:
        return len(self._free)

    def get_allocated_nodes(self) -> Set[Node]:
        return set(self._node_to_indices.keys())

    def build_feature_assignment(
        self,
        in_state: ResidualStreamState,
        out_state: ResidualStreamState,
        input_nodes: List[Node],
        output_node: Node,
    ) -> FeatureAssignment:
        """Build a FeatureAssignment bridge for HeadlessTransformer.compute().

        Populates exactly two states:
        - in_state with all input_nodes at their allocated columns
          (read by get_input_res_stream)
        - out_state with output_node at its allocated columns
          (read by compute)
        """
        fa = FeatureAssignment({in_state, out_state})
        for node in input_nodes:
            fa.assign(in_state, node, self.get_indices(node))
        fa.assign(out_state, output_node, self.get_indices(output_node))
        return fa
