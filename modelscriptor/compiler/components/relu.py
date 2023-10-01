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
        points: int,
    ):
        self.in_nodes = {in_node}
        self.out_node = out_node
        self.output_matrix = output_matrix
        self.output_bias = output_bias
        self.points = points


class ReLULayerComponent(Component):
    def __init__(self, d: int):
        super().__init__(d)

    def get_strategies(self, node: Node) -> List[NodeComponentStrategy]:
        if isinstance(node, ReLU):
            # Only one strategy for relus.
            return [
                NodeComponentStrategy(
                    in_nodes={node.inputs[0]}, out_node=node, points=1
                )
            ]
        else:
            # No strategy for non-relu
            return []

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert (
            strategy.out_node in self.out_state.nodes
        ), "Strategy applied before output allocated"
        in_node = next(iter(strategy.in_nodes))
        self.in_state.connect_allocation(self.out_state, strategy.out_node, in_node)
