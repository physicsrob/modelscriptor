from typing import List

import torch

from modelscriptor.compiler.components.attn import AttnLayerComponent
from modelscriptor.compiler.components.skip import SkipLayerComponent
from modelscriptor.compiler.groups.group import Group
from modelscriptor.compiler.groups.strategy import (
    GroupStrategy,
    get_sequential_placement_strategies,
)
from modelscriptor.compiler.res_state import ResState
from modelscriptor.graph import Node


class AttnSubLayer(Group):
    attn: AttnLayerComponent
    skip: SkipLayerComponent
    in_state: ResState
    out_state: ResState

    def __init__(self, d: int, d_head: int = 64):
        super().__init__(d)
        self.attn = AttnLayerComponent(d, d_head)
        self.skip = SkipLayerComponent(d)
        self.out_state = ResState(d)
        self.in_state = ResState(d)

    def print_strategy(self, strategy: GroupStrategy):
        strategy.print(
            layer_components=[self.skip, self.attn],
            layer_names=["skip", "attn"],
        )

    def get_strategies(self, output_node: Node) -> List[GroupStrategy]:
        strategies = get_sequential_placement_strategies(
            {output_node}, [self.attn, self.skip]
        )
        return strategies

    def apply_skip_allocation(self, strategy: GroupStrategy):
        self.skip.out_state.update_from(self.out_state)
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(s)
        self.attn.in_state.update_from(self.skip.skip_state)

    def apply_strategy(self, strategy: GroupStrategy):
        # Connect skip out to group output
        self.skip.out_state.update_from(self.out_state)

        # Apply all skip strategies
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(s)

        self.attn.out_state.update_from(self.skip.in_state)
        self.attn.in_state.update_from(self.skip.skip_state)

        # Apply all attention strategies
        for s in strategy.get_component_strategies(self.attn):
            self.attn.apply_strategy(s)

        # Connect group input to attention input
        self.in_state.update_from(self.attn.in_state)

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
        states["attn"] = (self.attn.out_state, x)
        x = x + inp
        states["skip"] = (self.skip.out_state, x)
        if return_states:
            return x, states
        else:
            return x
