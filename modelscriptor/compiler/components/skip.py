from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.compiler.res_state import ResState
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
        assert self.in_state.get_node_indices(
            strategy.in_node
        ) == self.skip_state.get_node_indices(strategy.skip_node)
        assert self.in_state.get_node_indices(
            strategy.in_node
        ) == self.out_state.get_node_indices(strategy.out_node)

    def num_params(self) -> int:
        return 0

    def resize(self, new_d):
        self.d = new_d
        self.in_state.resize(new_d)
        self.out_state.resize(new_d)
        self.skip_state.resize(new_d)

    def get_min_width(self):
        return max(
            self.in_state.get_min_width(),
            self.out_state.get_min_width() + self.skip_state.get_min_width(),
        )
