from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.graph import Node, Concatenate, Linear, Constant, Attn
from modelscriptor.compiler.components.component import NodeComponentStrategy, Component
from modelscriptor.graph.misc import Placeholder


class AttnNodeComponentStrategy(NodeComponentStrategy):
    # We apply one attention head per strategy.

    # query_matrix shape (d_query_in, d_head)
    query_matrix: torch.Tensor

    # key_matrix shape (d_key_in, d_head)
    key_matrix: torch.Tensor

    # value_matrix shape (d_value_in, d_head)
    value_matrix: torch.Tensor

    # output_matrix shape (d_head, d_output)
    output_matrix: torch.Tensor

    d_head: int  # d_head of the node, could be less than the layer's d_head

    def __init__(
        self,
        query_in: Node,
        key_in: Node,
        value_in: Node,
        out_node: Node,
        query_matrix: torch.Tensor,
        key_matrix: torch.Tensor,
        value_matrix: torch.Tensor,
        output_matrix: torch.Tensor,
        points: int,
        d_head: int,
    ):
        super().__init__(
            in_nodes={query_in, key_in, value_in}, out_node=out_node, points=points
        )
        self.query_in = query_in
        self.key_in = key_in
        self.value_in = value_in

        self.query_matrix = query_matrix
        self.key_matrix = key_matrix
        self.value_matrix = value_matrix
        self.output_matrix = output_matrix
        self.d_head = d_head
        assert self.query_matrix.shape == (len(query_in), d_head)
        assert self.key_matrix.shape == (len(key_in), d_head)
        assert self.value_matrix.shape == (len(value_in), d_head)
        assert self.output_matrix.shape == (d_head, len(out_node))


class AttnLayerComponent(Component):
    # query_matrix shape (n_head, d, d_head)
    query_matrix: torch.Tensor

    # key_matrix shape (n_head, d, d_head)
    key_matrix: torch.Tensor

    # value_matrix shape (n_head, d, d_head)
    value_matrix: torch.Tensor

    # output_matrix shape (n_head, d_head, d_output)
    output_matrix: torch.Tensor
    d_head: int
    n_heads: int
    used_heads: int

    def __init__(self, d: int, d_head: int):
        super().__init__(d)
        assert (d % d_head) == 0, "Invalid combination of d and d_head"
        self.d_head = d_head
        self.n_heads = d // d_head
        self.used_heads = 0

        self.query_matrix = torch.zeros(self.n_heads, d, d_head)
        self.key_matrix = torch.zeros(self.n_heads, d, d_head)
        self.value_matrix = torch.zeros(self.n_heads, d, d_head)
        self.output_matrix = torch.zeros(self.n_heads, d_head, d)

    def __repr__(self):
        return f"AttnLayerComponent()"

    def get_strategies(self, node: Node) -> List[AttnNodeComponentStrategy]:
        strategies = []

        # We can compile attention components.
        if isinstance(node, Attn):
            node_d_head = node.query_matrix.shape[1]
            assert node_d_head <= self.d_head

            query_in, key_in, value_in = node.inputs
            strategies.append(
                AttnNodeComponentStrategy(
                    query_in=query_in,
                    key_in=key_in,
                    value_in=value_in,
                    out_node=node,
                    query_matrix=node.query_matrix,
                    key_matrix=node.key_matrix,
                    value_matrix=node.value_matrix,
                    output_matrix=node.output_matrix,
                    points=1,
                    d_head=node_d_head,
                )
            )

        # We can compile a zero constant.
        if isinstance(node, Constant) and node.is_zero() and len(node) < self.d_head:
            strategies.append(
                AttnNodeComponentStrategy(
                    query_in=Placeholder(),
                    key_in=Placeholder(),
                    value_in=Placeholder(),
                    out_node=node,
                    query_matrix=torch.zeros(0, self.d_head),
                    key_matrix=torch.zeros(0, self.d_head),
                    value_matrix=torch.zeros(0, self.d_head),
                    output_matrix=torch.zeros(self.d_head, len(node)),
                    points=1,
                    d_head=self.d_head,
                )
            )

        return strategies

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert isinstance(strategy, AttnNodeComponentStrategy)
        assert self.out_state.has_node(
            strategy.out_node
        ), "Strategy applied before output allocated"
        if self.used_heads >= self.n_heads:
            assert False, "Ran out of heads to apply strategy"
        n_head = self.used_heads
        self.used_heads += 1

        # Allocate the input nodes
        for node in strategy.in_nodes:
            self.in_state.allocate_node(node)

        # Copy the matrices
        query_in_indices = self.in_state.get_node_indices(strategy.query_in)
        key_in_indices = self.in_state.get_node_indices(strategy.key_in)
        value_in_indices = self.in_state.get_node_indices(strategy.value_in)
        out_indices = self.out_state.get_node_indices(strategy.out_node)

        # Copy the query matrix from the strategy (d_query_in, d_head) to self.query_matrix (n_head, d_query_in, d_head)
        for i, in_idx in enumerate(query_in_indices):
            for j in range(strategy.d_head):
                self.query_matrix[n_head, in_idx, j] = strategy.query_matrix[i, j]

        # Copy the key matrix from the strategy (d_key_in, d_head) to self.key_matrix (n_head, d_key_in, d_head)
        for i, in_idx in enumerate(key_in_indices):
            for j in range(strategy.d_head):
                self.key_matrix[n_head, in_idx, j] = strategy.key_matrix[i, j]

        # Copy the value matrix from the strategy (d_value_in, d_head) to self.value_matrix (n_head, d_value_in, d_head)
        for i, in_idx in enumerate(value_in_indices):
            for j in range(strategy.d_head):
                self.value_matrix[n_head, in_idx, j] = strategy.value_matrix[i, j]

        # Copy the output matrix from the strategy (d_head, d_output) to self.output_matrix (n_head, d_head, d_output)
        for i in range(strategy.d_head):
            for j, out_idx in enumerate(out_indices):
                self.output_matrix[n_head, i, out_idx] = strategy.output_matrix[i, j]

    def forward(self, inp: torch.Tensor):
        # inp shape (n_pos, d)
        assert inp.shape[1] == self.d
        n_pos = inp.shape[0]

        output = torch.zeros(n_pos, self.d)

        # Apply the attention heads
        for n_head in range(self.n_heads):
            query_values = (
                inp @ self.query_matrix[n_head]
            )  # query_values shape is (n_pos, d_head)

            key_values = (
                inp @ self.key_matrix[n_head]
            )  # key_values shape is (n_pos, d_head)

            attn_logits = (
                query_values @ key_values.t()
            )  # attn_logits shape is (n_pos, n_pos)

            # Apply attention mask
            mask = torch.triu(torch.ones_like(attn_logits), diagonal=1)
            attn_logits = torch.where(
                mask == 1, -1000 * torch.ones_like(attn_logits), attn_logits
            )
            attn = torch.softmax(attn_logits, dim=1)  # attn shape is (n_pos, n_pos)
            value_values = (
                inp @ self.value_matrix[n_head]
            )  # value_values shape is (n_pos, d_head)
            values = attn @ value_values  # values shape is now (n_pos, d_head)
            output += values @ self.output_matrix[n_head]

        return output
