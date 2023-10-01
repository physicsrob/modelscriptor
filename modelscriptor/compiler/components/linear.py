from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.graph import Node, Concatenate, Linear, Constant


class LinearNodeComponentStrategy(NodeComponentStrategy):
    output_matrix: torch.Tensor  # d_input x d_output (both len(out_node))
    output_bias: torch.Tensor  # d_output

    def __init__(
        self,
        in_node: Node,
        out_node: Node,
        output_matrix: torch.Tensor,
        output_bias: torch.Tensor,
        points: int,
    ):
        self.in_nodes = {in_node}
        self.out_node = out_node
        self.output_matrix = output_matrix
        self.output_bias = output_bias
        self.points = points


class LinearLayerComponent(Component):
    output_matrix: torch.Tensor  # d_input x d_output
    output_bias: torch.Tensor  # d_output

    def __init__(self, d: int):
        super().__init__(d)
        self.output_matrix = torch.zeros(d, d)
        self.output_bias = torch.zeros(d)

    def get_strategies(self, node: Node) -> List[LinearNodeComponentStrategy]:
        strategies = []

        # Always have the pass-through option.
        strategies.append(
            LinearNodeComponentStrategy(
                in_node=node,
                out_node=node,
                output_matrix=torch.eye(len(node)),
                output_bias=torch.zeros(len(node)),
                points=0,
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
                    points=1,
                )
            )
        return strategies

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert isinstance(strategy, LinearNodeComponentStrategy)
        self.out_state.print("pre alloc")
        print("strategy out", strategy.out_node)
        assert (
            strategy.out_node in self.out_state.nodes
        ), "Strategy applied before output allocated"
        in_node = next(iter(strategy.in_nodes))
        self.in_state.allocate_node(in_node)

        # Copy the matrix
        in_indices = self.in_state.node_to_indices[in_node]
        out_indices = self.out_state.node_to_indices[strategy.out_node]

        for i, in_idx in enumerate(in_indices):
            for j, out_idx in enumerate(out_indices):
                self.output_matrix[in_idx, out_idx] = strategy.output_matrix[i, j]

        for j, out_idx in enumerate(out_indices):
            self.output_bias[out_idx] = strategy.output_bias[j]
