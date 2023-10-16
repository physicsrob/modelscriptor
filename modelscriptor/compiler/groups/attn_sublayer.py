from typing import List, Optional

import torch

from modelscriptor.compiler.components.attn import AttnLayerComponent
from modelscriptor.compiler.components.skip import SkipLayerComponent
from modelscriptor.compiler.groups.group import Group
from modelscriptor.compiler.groups.strategy import (
    GroupStrategy,
    get_sequential_placement_strategies,
)
from modelscriptor.compiler.res_state import ResState
from modelscriptor.graph import Node, PosEncoding


class AttnSubLayer(Group):
    attn: AttnLayerComponent
    skip: SkipLayerComponent
    in_state: ResState
    out_state: ResState

    def __init__(
        self, d: int, d_head: int = 64, pos_encoding: Optional[PosEncoding] = None
    ):
        super().__init__(d)
        self.attn = AttnLayerComponent(d, d_head, pos_encoding)
        self.skip = SkipLayerComponent(d)
        self.out_state = self.skip.out_state
        self.in_state = self.skip.skip_state
        self.attn.in_state.link(self.in_state)
        self.attn.out_state.link(self.skip.in_state)

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

    def apply_pre_allocation(self, strategy: GroupStrategy):
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(s)

    def apply_strategy(self, strategy: GroupStrategy):
        # Apply all skip strategies
        for s in strategy.get_component_strategies(self.skip):
            self.skip.apply_strategy(s)

        # Apply all attention strategies
        for s in strategy.get_component_strategies(self.attn):
            self.attn.apply_strategy(s)

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

    def num_params(self):
        return self.attn.num_params()

    def resize(self, new_d):
        self.d = new_d
        self.attn.resize(new_d)
        self.in_state.resize(new_d)
        self.out_state.resize(new_d)

    def get_min_width(self):
        return max(
            self.attn.get_min_width(),
            self.skip.get_min_width(),
            self.in_state.get_min_width(),
            self.out_state.get_min_width(),
        )
