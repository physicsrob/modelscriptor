from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.graph import Node, Concatenate, Linear, Constant
from modelscriptor.graph.misc import Placeholder


class LinearNodeComponentStrategy(NodeComponentStrategy):
    output_matrix: torch.Tensor  # d_input x d_output (both len(out_node))
    output_bias: torch.Tensor  # d_output

    def __init__(
        self,
        in_node: Node,
        out_node: Node,
        output_matrix: torch.Tensor,
        output_bias: torch.Tensor,
    ):
        super().__init__(in_nodes=[in_node], out_node=out_node)
        self.output_matrix = output_matrix
        self.output_bias = output_bias

    def __repr__(self):
        if self.in_nodes == [self.out_node]:
            return f"LinearNodeComponentStrategy(passthrough=True, node={self.in_nodes[0]})"
        else:
            return f"LinearNodeComponentStrategy(in_node={self.in_nodes[0]}, out_node={self.out_node})"


class LinearLayerComponent(Component):
    output_matrix: torch.Tensor  # d_input x d_output
    output_bias: torch.Tensor  # d_output

    def __init__(self, d: int):
        super().__init__(d)
        self.output_matrix = torch.zeros(d, d)
        self.output_bias = torch.zeros(d)

    def __repr__(self):
        return f"LinearLayerComponent()"

    def get_strategies(self, node: Node) -> List[LinearNodeComponentStrategy]:
        strategies = []

        # Always have the pass-through option.
        strategies.append(
            LinearNodeComponentStrategy(
                in_node=node,
                out_node=node,
                output_matrix=torch.eye(len(node)),
                output_bias=torch.zeros(len(node)),
            )
        )

        # If the node is linear, we can compile it!
        if isinstance(node, Linear):
            strategies.append(
                LinearNodeComponentStrategy(
                    in_node=node.inputs[0],
                    out_node=node,
                    output_matrix=node.output_matrix,
                    output_bias=node.output_bias,
                )
            )

        # If the node is constant, we can compile it
        if isinstance(node, Constant):
            strategies.append(
                LinearNodeComponentStrategy(
                    in_node=Placeholder(),
                    out_node=node,
                    output_matrix=torch.zeros(0, 0),
                    output_bias=node.value,
                )
            )

        return strategies

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert isinstance(strategy, LinearNodeComponentStrategy)
        assert self.out_state.has_node_indices(
            strategy.out_node
        ), "Strategy applied before output allocated"
        in_node = strategy.in_nodes[0]
        self.in_state.allocate_node(in_node)

        # Copy the matrix
        in_indices = self.in_state.get_node_indices(in_node)
        out_indices = self.out_state.get_node_indices(strategy.out_node)

        for i, in_idx in enumerate(in_indices):
            for j, out_idx in enumerate(out_indices):
                self.output_matrix[in_idx, out_idx] = strategy.output_matrix[i, j]

        for j, out_idx in enumerate(out_indices):
            self.output_bias[out_idx] = strategy.output_bias[j]

    def forward(self, inp: torch.Tensor):
        # inp shape (n_pos, d)
        x = inp @ self.output_matrix
        return x + self.output_bias

    def num_params(self) -> int:
        return self.output_matrix.numel() + self.output_bias.numel()

    def resize(self, new_d):
        super().resize(new_d)
        self.output_matrix = self.output_matrix[:new_d, :new_d]
        self.output_bias = self.output_bias[:new_d]
