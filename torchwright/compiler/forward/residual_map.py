from typing import Dict, List, Set

from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
    flatten_concat_nodes,
)
from torchwright.graph import Node, Concatenate


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
        # Cols whose current value is unknown (may contain garbage from a
        # caller-provided residual stream).  A write op whose target lands on
        # a dirty col must first cancel the prior value, or the additive
        # sublayer write produces `garbage + value` instead of `value`.
        # Cols become clean when they are explicitly written by
        # get_input_res_stream (pos encoding + input nodes) or cancelled
        # inline in the forward pass.
        self._dirty: Set[int] = set(range(d))

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

    def resolve_indices(self, node: Node) -> List[int]:
        """Get column indices for a node, resolving through Concatenate.

        Unlike get_indices(), this handles Concatenate nodes by gathering
        the indices of their children in order. Needed by the weight writer
        when an Attn node's input is a Concatenate (e.g. get_prev_value).
        """
        if isinstance(node, Concatenate):
            indices = []
            for child in flatten_concat_nodes([node]):
                indices += self.get_indices(child)
            return indices
        return self.get_indices(node)

    def is_allocated(self, node: Node) -> bool:
        return node in self._node_to_indices

    def get_free_count(self) -> int:
        return len(self._free)

    def mark_clean(self, cols) -> None:
        """Record that ``cols`` now hold a known value and no longer need
        cancellation before the next additive write."""
        self._dirty.difference_update(cols)

    def dirty_subset(self, cols: List[int]) -> List[int]:
        """Return the subset of ``cols`` still marked dirty, preserving order."""
        return [c for c in cols if c in self._dirty]

    def get_allocated_nodes(self) -> Set[Node]:
        return set(self._node_to_indices.keys())

    def build_residual_assignment(
        self,
        in_state: ResidualStreamState,
        out_state: ResidualStreamState,
        input_nodes: List[Node],
        output_node: Node,
    ) -> ResidualAssignment:
        """Build a ResidualAssignment bridge for HeadlessTransformer.compute().

        Populates exactly two states:
        - in_state with all input_nodes at their allocated columns
          (read by get_input_res_stream)
        - out_state with output_node at its allocated columns
          (read by compute)
        """
        ra = ResidualAssignment({in_state, out_state})
        for node in input_nodes:
            ra.assign(in_state, node, self.get_indices(node))
        ra.assign(out_state, output_node, self.get_indices(output_node))
        return ra
