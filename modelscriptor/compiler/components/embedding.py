from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
)
from modelscriptor.graph import Node, Concatenate, Linear, Constant, Embedding
from modelscriptor.graph.misc import Placeholder


class EmbeddingNodeComponentStrategy(NodeComponentStrategy):
    output_matrix: torch.Tensor  # vocab x d_output

    def __init__(
        self,
        out_node: Embedding,
    ):
        super().__init__(in_nodes=[], out_node=out_node)
        self.output_matrix = out_node.table


class EmbeddingConstantNodeComponentStrategy(NodeComponentStrategy):
    value: torch.Tensor

    def __init__(
        self,
        out_node: Constant,
    ):
        super().__init__(in_nodes=[], out_node=out_node)
        self.value = out_node.value


class EmbeddingLayerComponent(Component):
    output_matrix: torch.Tensor  # max_vocab x d

    def __init__(self, d: int, max_vocab: int):
        super().__init__(d)
        self.output_matrix = torch.zeros(max_vocab, d)

    def __repr__(self):
        return f"EmbeddingLayerComponent()"

    def get_strategies(self, node: Node) -> List[NodeComponentStrategy]:
        # If the node is embedding, we can compile it!
        if isinstance(node, Embedding):
            return [
                EmbeddingNodeComponentStrategy(
                    out_node=node,
                )
            ]
        elif isinstance(node, Constant):
            return [
                EmbeddingConstantNodeComponentStrategy(
                    out_node=node,
                )
            ]
        else:
            return []

    def apply_strategy(
        self, feature_assignment: FeatureAssignment, strategy: NodeComponentStrategy
    ):
        out_indices = feature_assignment.get_node_indices(
            self.out_state, strategy.out_node
        )

        if isinstance(strategy, EmbeddingNodeComponentStrategy):
            vocab_len = self.output_matrix.shape[0]
            for j, out_idx in enumerate(out_indices):
                self.output_matrix[:, out_idx] = strategy.output_matrix[:vocab_len, j]
        elif isinstance(strategy, EmbeddingConstantNodeComponentStrategy):
            for j, out_idx in enumerate(out_indices):
                self.output_matrix[:, out_idx] = strategy.value[j]
        else:
            assert False

    def forward(self, inp: torch.Tensor):
        # inp shape (n_pos)
        assert len(inp.shape) == 1
        assert inp.max() < self.output_matrix.shape[0]
        return self.output_matrix[inp]

    def deembed_forward(self, inp: torch.Tensor):
        # inp shape (n_pos, d)
        n_pos = inp.shape[0]
        result = torch.zeros(n_pos, dtype=torch.long)
        for pos in range(n_pos):
            probs = self.output_matrix @ inp[pos]
            token_id = probs.argmax().item()
            result[pos] = token_id
        return result

    def num_params(self) -> int:
        return self.output_matrix.numel()

    def resize(self, new_d):
        self.d = new_d
        self.output_matrix = self.output_matrix[:, :new_d]
