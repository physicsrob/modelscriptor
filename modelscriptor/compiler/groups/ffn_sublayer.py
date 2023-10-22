from typing import List

import torch

from modelscriptor.compiler.components.linear import LinearLayerComponent
from modelscriptor.compiler.components.relu import ReLULayerComponent
from modelscriptor.compiler.components.skip import SkipLayerComponent
from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
)
from modelscriptor.compiler.groups.group import Group, GroupStrategy
from modelscriptor.compiler.groups.strategy import get_sequential_placement_strategies
from modelscriptor.graph import Node, Add


class FFNSubLayer(Group):
    linear1: LinearLayerComponent
    relu: ReLULayerComponent
    linear2: LinearLayerComponent
    skip: SkipLayerComponent

    def __init__(
        self,
        d: int,
    ):
        super().__init__(d, name="FFNSubLayer")
        self.linear1 = LinearLayerComponent(d, name="linear1")
        self.relu = ReLULayerComponent(d, name="relu")
        self.linear2 = LinearLayerComponent(d, name="linear2")
        self.skip = SkipLayerComponent(d, name="linear_skip")

    def get_strategies_for_node(self, output_node: Node) -> List[GroupStrategy]:
        strategies = get_sequential_placement_strategies(
            {output_node},
            [self.linear1, self.relu, self.linear2, self.skip],
            component_names=["linear1", "relu", "linear2", "skip"],
        )
        return strategies

    def get_constraints(self, strategy: GroupStrategy) -> FeatureAssignmentConstraints:
        constraints = strategy.get_constraints_for_strategy()
        constraints.add_equivalency(self.in_state, self.linear1.in_state)
        constraints.add_equivalency(self.linear1.out_state, self.relu.in_state)
        constraints.add_equivalency(self.relu.out_state, self.linear2.in_state)
        constraints.add_equivalency(self.linear2.out_state, self.skip.in_state)
        constraints.add_equivalency(self.skip.skip_state, self.in_state)
        constraints.add_equivalency(self.skip.out_state, self.out_state)
        return constraints

    def apply_strategy(
        self, feature_assignment: FeatureAssignment, strategy: GroupStrategy
    ):
        # Apply all skip strategies
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(feature_assignment, s)

        # Apply all linear2 strategies
        for s in strategy.get_component_strategies(self.linear2):
            self.linear2.apply_strategy(feature_assignment, s)

        # Apply relu strategies
        for s in strategy.get_component_strategies(self.relu):
            self.relu.apply_strategy(feature_assignment, s)

        # Apply linear1 strategies
        for s in strategy.get_component_strategies(self.linear1):
            self.linear1.apply_strategy(feature_assignment, s)

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
        states["linear1_out_state"] = (self.linear1.out_state, x)
        x = self.relu.forward(x)
        states["relu_out_state"] = (self.relu.out_state, x)
        x = self.linear2.forward(x)
        states["linear2_out_state"] = (self.linear2.out_state, x)
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
