from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
)
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
    def __init__(self, d: int, name: str = ""):
        super().__init__(d, name)

    def __repr__(self):
        return f"ReLULayerComponent(name='{self.name}')"

    def get_strategies(self, node: Node) -> List[NodeComponentStrategy]:
        if isinstance(node, ReLU):
            # Only one strategy for relus.
            return [NodeComponentStrategy(in_nodes=node.inputs, out_node=node)]
        else:
            # No strategy for non-relu
            return []

    def get_constraints_for_strategy(self, strategy: NodeComponentStrategy):
        in_node = strategy.in_nodes[0]
        out_node = strategy.out_node

        constraints = FeatureAssignmentConstraints()
        constraints.add_node_to_state(out_node, self.out_state)
        constraints.add_node_to_state(in_node, self.in_state)

        constraints.add_shared_features_constraint(
            self.in_state, in_node, self.out_state, strategy.out_node
        )
        return constraints

    def apply_strategy(
        self, feature_assignment: FeatureAssignment, strategy: NodeComponentStrategy
    ):
        pass

    def forward(self, inp: torch.Tensor):
        return torch.clamp(inp, min=0)

    def num_params(self) -> int:
        return 0
