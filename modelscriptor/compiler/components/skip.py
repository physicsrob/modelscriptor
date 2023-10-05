from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.compiler.res_state import ResState
from modelscriptor.graph import Node, Concatenate, Linear, Add
from modelscriptor.modelscript.inout_nodes import create_constant


class SkipNodeComponentStrategy(NodeComponentStrategy):
    skip_node: Node
    in_node: Node  # The non-skip node

    def __init__(self, skip_node: Node, in_node: Node, out_node: Node, points: int):
        super().__init__({in_node}, out_node, points)
        self.skip_node = skip_node
        self.in_node = in_node

    def __repr__(self):
        return f"SkipNodeComponentStrategy(in_node={self.in_node}, skip_node={self.skip_node}, out_node={self.out_node})"


class SkipLayerComponent(Component):
    d: int
    skip_state: ResState
    in_state: ResState
    out_state: ResState

    def __init__(self, d):
        super().__init__(d)
        self.skip_state = ResState(d)
        self.in_state = ResState(d)

    def __repr__(self):
        return f"SkipLayerComponent()"

    def get_strategies(self, node: Node) -> List[SkipNodeComponentStrategy]:
        strategies = []

        # If node is add, we have two strategies
        if isinstance(node, Add):
            addend0 = node.inputs[0]
            addend1 = node.inputs[1]
            strategies.append(
                SkipNodeComponentStrategy(
                    skip_node=addend0, in_node=addend1, out_node=node, points=1
                )
            )
            strategies.append(
                SkipNodeComponentStrategy(
                    skip_node=addend1, in_node=addend0, out_node=node, points=1
                )
            )
        else:
            # If node is not add
            zero = create_constant(torch.zeros(len(node)))
            strategies.append(
                SkipNodeComponentStrategy(
                    skip_node=zero, in_node=node, out_node=node, points=0
                )
            )
            strategies.append(
                SkipNodeComponentStrategy(
                    skip_node=node, in_node=zero, out_node=node, points=0
                )
            )
        return strategies

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert isinstance(strategy, SkipNodeComponentStrategy)
        assert self.out_state.has_node(
            strategy.out_node
        ), "Strategy applied before output allocated"

        self.skip_state.connect_allocation(
            self.out_state, strategy.out_node, strategy.skip_node
        )
        self.in_state.connect_allocation(
            self.out_state, strategy.out_node, strategy.in_node
        )
        print("After applying skip strategy indices:")
        print(f"{self.in_state.get_node_indices(strategy.in_node)=}")
        print(f"{self.skip_state.get_node_indices(strategy.skip_node)=}")
        print(f"{self.skip_state._node_to_indices.get(strategy.skip_node)=}")
        print(f"{self.out_state.get_node_indices(strategy.out_node)=}")
        print()
        assert self.in_state.get_node_indices(
            strategy.in_node
        ) == self.skip_state.get_node_indices(strategy.skip_node)
        assert self.in_state.get_node_indices(
            strategy.in_node
        ) == self.out_state.get_node_indices(strategy.out_node)
