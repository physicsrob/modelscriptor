from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
    ResidualStreamState,
)
from modelscriptor.graph import Node, Concatenate, Linear, Add
from modelscriptor.modelscript.inout_nodes import create_constant


class SkipNodeComponentStrategy(NodeComponentStrategy):
    skip_node: Node
    in_node: Node  # The non-skip node

    def __init__(self, skip_node: Node, in_node: Node, out_node: Node):
        super().__init__([in_node], out_node)
        self.skip_node = skip_node
        self.in_node = in_node

    def __repr__(self):
        return f"SkipNodeComponentStrategy(in_node={self.in_node}, skip_node={self.skip_node}, out_node={self.out_node})"


class SkipLayerComponent(Component):
    d: int
    skip_state: ResidualStreamState

    def __init__(self, d, name: str = ""):
        super().__init__(d, name)
        self.skip_state = ResidualStreamState()

    def __repr__(self):
        return f"SkipLayerComponent(name='{self.name}')"

    def get_strategies(self, node: Node) -> List[SkipNodeComponentStrategy]:
        strategies = []
        zero = create_constant(torch.zeros(len(node)))

        # We always have two skip strategies where we don't compile anything.
        strategies.append(
            SkipNodeComponentStrategy(skip_node=zero, in_node=node, out_node=node)
        )
        strategies.append(
            SkipNodeComponentStrategy(skip_node=node, in_node=zero, out_node=node)
        )

        if isinstance(node, Add):
            # If node is Add, we have two additional strategies:
            # - add(input0, input1)
            # - add(input1, input0)
            addend0 = node.inputs[0]
            addend1 = node.inputs[1]
            strategies.append(
                SkipNodeComponentStrategy(
                    skip_node=addend0, in_node=addend1, out_node=node
                )
            )
            strategies.append(
                SkipNodeComponentStrategy(
                    skip_node=addend1, in_node=addend0, out_node=node
                )
            )
        return strategies

    def get_constraints_for_strategy(self, strategy: NodeComponentStrategy):
        assert isinstance(strategy, SkipNodeComponentStrategy)
        in_node = strategy.in_node
        skip_node = strategy.skip_node
        out_node = strategy.out_node

        constraints = FeatureAssignmentConstraints()
        constraints.add_node_to_state(out_node, self.out_state)
        constraints.add_node_to_state(in_node, self.in_state)
        constraints.add_node_to_state(skip_node, self.skip_state)

        constraints.add_shared_features_constraint(
            self.in_state, in_node, self.out_state, strategy.out_node
        )
        constraints.add_shared_features_constraint(
            self.skip_state, skip_node, self.out_state, strategy.out_node
        )
        return constraints

    def apply_strategy(
        self, feature_assignment: FeatureAssignment, strategy: NodeComponentStrategy
    ):
        pass

    def num_params(self) -> int:
        return 0

    def resize(self, new_d):
        self.d = new_d
