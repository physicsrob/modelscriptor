from typing import List, Optional

import torch

from modelscriptor.compiler.components.attn import AttnLayerComponent
from modelscriptor.compiler.components.skip import SkipLayerComponent
from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
)
from modelscriptor.compiler.groups.group import Group
from modelscriptor.compiler.groups.strategy import (
    GroupStrategy,
    get_sequential_placement_strategies,
)
from modelscriptor.graph import Node, PosEncoding


class AttnSubLayer(Group):
    attn: AttnLayerComponent
    skip: SkipLayerComponent

    def __init__(
        self,
        d: int,
        d_head: int,
        pos_encoding: Optional[PosEncoding] = None,
    ):
        super().__init__(d, name="AttnSubLayer")
        self.attn = AttnLayerComponent(d, d_head, pos_encoding, name="attn")
        self.skip = SkipLayerComponent(d, name="attn_skip")

    def get_strategies_for_node(self, output_node: Node) -> List[GroupStrategy]:
        strategies = get_sequential_placement_strategies(
            {output_node}, [self.attn, self.skip], ["attn", "skip"]
        )
        return strategies

    def get_constraints(self, strategy: GroupStrategy) -> FeatureAssignmentConstraints:
        constraints = strategy.get_constraints_for_strategy()
        constraints.add_equivalency(self.in_state, self.attn.in_state)
        constraints.add_equivalency(self.in_state, self.skip.skip_state)
        constraints.add_equivalency(self.out_state, self.skip.out_state)
        constraints.add_equivalency(self.attn.out_state, self.skip.in_state)
        return constraints

    def apply_strategy(
        self, feature_assignment: FeatureAssignment, strategy: GroupStrategy
    ):
        # Apply all skip strategies
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(feature_assignment, s)

        # Apply all attention strategies
        for s in strategy.get_component_strategies(self.attn):
            self.attn.apply_strategy(feature_assignment, s)

    def forward(self, inp: torch.Tensor, return_states=False):
        """
        Forward pass through the AttnSublayer.

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

        x = self.attn.forward(inp)
        states["attn_out_state"] = (self.attn.out_state, x)
        x = x + inp
        states["skip_out_state"] = (self.skip.out_state, x)
        if return_states:
            return x, states
        else:
            return x

    def num_params(self):
        return self.attn.num_params()

    def resize(self, new_d):
        self.d = new_d
        self.attn.resize(new_d)
        self.in_state.resize(new_d)
        self.out_state.resize(new_d)
