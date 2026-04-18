"""Placeholder nodes for composite ops that need finalized bounds.

During graph construction, bound-consumer ops (``cond_gate``,
``select``, ``floor_int``, ``attend_mean_where``) return a
``ConsumerPlaceholder`` instead of building their real subgraph.
``finalize()`` materializes these using the tightened affine bounds,
then rebinds downstream consumers to the real output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from torchwright.graph.node import Node
from torchwright.graph.value_type import NodeValueType

if TYPE_CHECKING:
    pass


class ConsumerPlaceholder(Node):
    """Placeholder for a composite op that needs finalized bounds."""

    def __init__(self, d_output: int, inputs: list, name: str = ""):
        self._materialized_output: Optional[Node] = None
        super().__init__(d_output, inputs, name=name)

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType.unknown()

    def materialize(self) -> Node:
        raise NotImplementedError

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        if self._materialized_output is not None:
            return self._materialized_output.compute(n_pos, input_values)
        raise RuntimeError("ConsumerPlaceholder not materialized")


class CondGatePlaceholder(ConsumerPlaceholder):
    """Placeholder for ``cond_gate``."""

    def __init__(self, cond: Node, inp: Node, *, approximate: bool = True):
        self._approximate = approximate
        super().__init__(len(inp), [cond, inp], name="cond_gate_placeholder")

    def compute_value_type(self) -> NodeValueType:
        from torchwright.ops.logic_ops import _cond_gate_output_type

        return _cond_gate_output_type(self.inputs[0], self.inputs[1])

    def materialize(self) -> Node:
        from torchwright.ops.logic_ops import _build_cond_gate

        cond, inp = self.inputs
        self._materialized_output = _build_cond_gate(
            cond, inp, approximate=self._approximate
        )
        return self._materialized_output


class SelectPlaceholder(ConsumerPlaceholder):
    """Placeholder for ``select``."""

    def __init__(
        self, cond: Node, true_node: Node, false_node: Node, *, approximate: bool = True
    ):
        self._approximate = approximate
        super().__init__(
            len(true_node), [cond, true_node, false_node], name="select_placeholder"
        )

    def compute_value_type(self) -> NodeValueType:
        from torchwright.ops.map_select import _select_output_type

        return _select_output_type(self.inputs[0], self.inputs[1], self.inputs[2])

    def materialize(self) -> Node:
        from torchwright.ops.map_select import _build_select

        cond, true_node, false_node = self.inputs
        self._materialized_output = _build_select(
            cond, true_node, false_node, approximate=self._approximate
        )
        return self._materialized_output
