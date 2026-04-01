from typing import Optional

import torch

from modelscriptor.compiler.components.component import Component
from modelscriptor.graph import PosEncoding


class AttnLayerComponent(Component):
    """Multi-head attention component.

    Weight matrices:
        query_matrix:  (n_heads, d, d_head)
        key_matrix:    (n_heads, d, d_head)
        value_matrix:  (n_heads, d, d_head)
        output_matrix: (n_heads, d_head, d)
    """

    def __init__(
        self, d: int, d_head: int, pos_encoding: Optional[PosEncoding], name: str = ""
    ):
        super().__init__(d, name)
        assert (d % d_head) == 0, "Invalid combination of d and d_head"
        self.d_head = d_head
        self.n_heads = d // d_head
        self.used_heads = 0
        self.pos_encoding = pos_encoding

        self.query_matrix = torch.zeros(self.n_heads, d, d_head)
        self.key_matrix = torch.zeros(self.n_heads, d, d_head)
        self.value_matrix = torch.zeros(self.n_heads, d, d_head)
        self.output_matrix = torch.zeros(self.n_heads, d_head, d)

    def __repr__(self):
        return f"AttnLayerComponent(name='{self.name}')"

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

    def num_params(self) -> int:
        return (
            self.query_matrix.numel()
            + self.key_matrix.numel()
            + self.value_matrix.numel()
            + self.output_matrix.numel()
        )
