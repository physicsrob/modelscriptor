from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.graph import Node, Concatenate, Linear, ReLU
from modelscriptor.compiler.components.component import NodeComponentStrategy, Component


class ReLUNodeComponentStrategy(NodeComponentStrategy):
    def __init__(
        self,
        in_node: Node,
        out_node: Node,
        output_matrix: torch.Tensor,
        output_bias: torch.Tensor,
    ):
        super().__init__(in_nodes=[in_node], out_node=out_node)
        self.output_matrix = output_matrix
        self.output_bias = output_bias

    def __repr__(self):
        return (
            f"ReLUComponentStrategy(in_nodes={self.in_nodes}, out_node={self.out_node})"
        )


class ReLULayerComponent(Component):
    def __init__(self, d: int):
        super().__init__(d)

    def __repr__(self):
        return f"ReLULayerComponent()"

    def get_strategies(self, node: Node) -> List[NodeComponentStrategy]:
        if isinstance(node, ReLU):
            # Only one strategy for relus.
            return [NodeComponentStrategy(in_nodes=node.inputs, out_node=node)]
        else:
            # No strategy for non-relu
            return []

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert self.out_state.has_node_indices(
            strategy.out_node
        ), "Strategy applied before output allocated"
        in_node = strategy.in_nodes[0]
        self.in_state.connect_allocation(self.out_state, strategy.out_node, in_node)

    def forward(self, inp: torch.Tensor):
        return torch.clamp(inp, min=0)

    def num_params(self) -> int:
        return 0
