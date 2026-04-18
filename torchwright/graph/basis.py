"""Shared basis for affine bound propagation.

A ``Basis`` maps every ``InputNode`` component (element) to a column
index in the affine coefficient matrices ``A_lo`` and ``A_hi``.  All
``AffineBound`` objects in the same graph share the same ``Basis`` so
their coefficient vectors are directly addable / composable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from torchwright.graph.misc import InputNode


@dataclass(frozen=True)
class Basis:
    """Column layout for affine coefficient matrices.

    Attributes:
        n: Total number of basis columns (sum of all InputNode widths).
        slices: Mapping from InputNode → (start_col, width).
    """

    n: int
    slices: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    @staticmethod
    def from_input_nodes(input_nodes: "List[InputNode]") -> "Basis":
        slices: Dict[int, Tuple[int, int]] = {}
        offset = 0
        for node in input_nodes:
            slices[node.node_id] = (offset, node.d_output)
            offset += node.d_output
        return Basis(n=offset, slices=slices)

    def index_of(self, node: "InputNode") -> Tuple[int, int]:
        """Return ``(start_col, width)`` for *node* in this basis."""
        key = node.node_id
        if key not in self.slices:
            raise KeyError(
                f"InputNode(id={node.node_id}, name={node.name!r}) "
                f"is not in this Basis"
            )
        return self.slices[key]
