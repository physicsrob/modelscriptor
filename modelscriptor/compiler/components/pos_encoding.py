from typing import List, Optional

import torch

from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.graph import Node, PosEncoding


class PosEncodingNodeComponentStrategy(NodeComponentStrategy):
    out_node: PosEncoding

    def __init__(
        self,
        out_node: PosEncoding,
    ):
        super().__init__(in_nodes=[], out_node=out_node)


class PosEncodingLayerComponent(Component):
    pos_encoding: Optional[PosEncoding]
    out_indices: List[int]

    def __init__(self, d: int):
        super().__init__(d)
        self.out_indices = []
        self.pos_encoding = None

    def __repr__(self):
        return f"PosEncodingLayerComponent()"

    def get_strategies(self, node: Node) -> List[NodeComponentStrategy]:
        strategies: List[NodeComponentStrategy] = []

        # If the node is pos encoding, we can compile it!
        if isinstance(node, PosEncoding):
            strategies.append(
                PosEncodingNodeComponentStrategy(
                    out_node=node,
                )
            )

        return strategies

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert self.out_state.has_node_indices(
            strategy.out_node
        ), "Strategy applied before output allocated"
        out_indices = self.out_state.get_node_indices(strategy.out_node)

        if isinstance(strategy, PosEncodingNodeComponentStrategy):
            self.pos_encoding = strategy.out_node
            self.out_indices = out_indices
        else:
            assert False

    def forward(self, inp: torch.Tensor):
        # inp shape (n_pos)
        assert len(inp.shape) == 1
        assert self.pos_encoding
        n_pos = inp.shape[0]
        x = torch.zeros((n_pos, self.d))
        for i, idx in enumerate(self.out_indices):
            x[:, idx] = self.pos_encoding.get_pos_encoding(n_pos)[:, i]
        return x

    def num_params(self) -> int:
        return 0

    def resize(self, new_d):
        self.d = new_d

    def get_min_width(self):
        return max(self.out_indices) + 1
