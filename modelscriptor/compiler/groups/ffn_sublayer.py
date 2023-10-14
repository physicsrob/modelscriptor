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

    def print_strategy(self, strategy: GroupStrategy):
        strategy.print(
            layer_components=[self.skip, self.linear2, self.relu, self.linear1],
            layer_names=["skip", "linear2", "relu", "linear1"],
        )

    def get_strategies(self, output_node: Node) -> List[GroupStrategy]:
        strategies = get_sequential_placement_strategies(
            {output_node}, [self.linear1, self.relu, self.linear2, self.skip]
        )
        return strategies

    def apply_skip_allocation(self, strategy: GroupStrategy):
        self.skip.out_state.update_from(self.out_state)
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(s)
        self.linear1.in_state.update_from(self.skip.skip_state)

    def apply_strategy(self, strategy: GroupStrategy):
        # Connect skip out to group output
        self.skip.out_state.update_from(self.out_state)
        # self.skip.skip_state.update_from(self.in_state)

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

    def forward(self, inp: torch.Tensor, return_states=False):
        """
        Forward pass through the FFNSublayer.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor for the forward pass.

        return_states : bool, optional (default=False)
            If True, return the internal states of each layer alongside the output.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor and dict
            The output tensor if `return_states=False`.
            A tuple of the output tensor and a dictionary containing intermediate states
            if `return_states=True`. The dictionary keys are the names of the layers or
            operations, and the values are tuples containing the state object and the
            output tensor of that layer.

        """
        states = {}

        x = self.linear1.forward(inp)
        states["linear1"] = (self.linear1.out_state, x)
        x = self.relu.forward(x)
        states["relu"] = (self.relu.out_state, x)
        x = self.linear2.forward(x)
        states["linear2"] = (self.linear2.out_state, x)
        x = x + inp
        states["skip"] = (self.skip.out_state, x)
        if return_states:
            return x, states
        else:
            return x

    def num_params(self):
        return self.linear1.num_params() + self.linear2.num_params()

    def resize(self, new_d):
        self.d = new_d
        self.linear1.resize(new_d)
        self.linear2.resize(new_d)
        self.relu.resize(new_d)
        self.skip.resize(new_d)
        self.in_state.resize(new_d)
        self.out_state.resize(new_d)

    def get_min_width(self):
        return max(
            self.linear1.get_min_width(),
            self.linear2.get_min_width(),
            self.relu.get_min_width(),
            self.skip.get_min_width(),
            self.in_state.get_min_width(),
            self.out_state.get_min_width(),
        )
