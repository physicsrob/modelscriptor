from typing import List

import torch

from modelscriptor.compiler.components.linear import LinearLayerComponent
from modelscriptor.compiler.components.relu import ReLULayerComponent
from modelscriptor.compiler.components.skip import SkipLayerComponent
from modelscriptor.compiler.groups.group import Group, GroupStrategy
from modelscriptor.compiler.groups.strategy import get_sequential_placement_strategies
from modelscriptor.compiler.res_state import ResState
from modelscriptor.graph import Node, Add


class FFNSubLayer(Group):
    linear1: LinearLayerComponent
    relu: ReLULayerComponent
    linear2: LinearLayerComponent
    skip: SkipLayerComponent
    in_state: ResState
    out_state: ResState

    def __init__(self, d: int):
        super().__init__(d)
        self.linear1 = LinearLayerComponent(d)
        self.relu = ReLULayerComponent(d)
        self.linear2 = LinearLayerComponent(d)
        self.skip = SkipLayerComponent(d)
        self.out_state = ResState(d)
        self.in_state = ResState(d)

    def print(self):
        print("FFNSubLayer")
        self.linear2.out_state.print("linear2 out")
        self.linear2.in_state.print("linear2 in")
        self.relu.out_state.print("relu out")
        self.relu.in_state.print("relu in")
        self.linear1.out_state.print("linear1 out")
        self.linear1.in_state.print("linear1 in")
        self.skip.in_state.print("skip in")
        self.skip.skip_state.print("skip skip_in")
        self.skip.out_state.print("skip out")

    def get_strategies(self, output_node: Node) -> List[GroupStrategy]:
        return get_sequential_placement_strategies(
            {output_node}, [self.linear1, self.relu, self.linear2, self.skip]
        )

    def apply_strategy(self, strategy: GroupStrategy):
        # Connect skip out to group output
        self.skip.out_state.update_from(self.out_state)

        # Apply all skip strategies
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(s)

        # Connect linear2 output to group output
        self.linear2.out_state.update_from(self.skip.in_state)

        # Apply all linear2 strategies
        for s in strategy.get_component_strategies(self.linear2):
            self.linear2.apply_strategy(s)

        # Connect relu out to linear2 in
        self.relu.out_state.update_from(self.linear2.in_state)

        # Apply relu strategies
        for s in strategy.get_component_strategies(self.relu):
            self.relu.apply_strategy(s)

        # Connect linear1 out to relu in
        self.linear1.out_state.update_from(self.relu.in_state)

        # Copy skip connection to linear1 in state
        self.linear1.in_state.update_from(self.skip.skip_state)

        # Apply linear1 strategies
        for s in strategy.get_component_strategies(self.linear1):
            self.linear1.apply_strategy(s)

        # Connect group input to linear1 input
        self.in_state.update_from(self.linear1.in_state)

    def forward(self, inp: torch.Tensor):
        x = self.linear1.forward(inp)
        x = self.relu.forward(x)
        x = self.linear2.forward(x)
        return x + inp
