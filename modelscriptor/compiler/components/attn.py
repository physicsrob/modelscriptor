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

        # All heads in parallel via batched ops
        Q = torch.einsum('pd,hdk->hpk', inp, self.query_matrix)
        K = torch.einsum('pd,hdk->hpk', inp, self.key_matrix)
        V = torch.einsum('pd,hdk->hpk', inp, self.value_matrix)

        attn_logits = torch.bmm(Q, K.transpose(1, 2))  # (n_heads, n_pos, n_pos)
        mask = torch.triu(torch.ones(n_pos, n_pos, device=inp.device), diagonal=1).bool()
        attn_logits.masked_fill_(mask.unsqueeze(0), -1000.0)
        attn = torch.softmax(attn_logits, dim=2)

        weighted = torch.bmm(attn, V)  # (n_heads, n_pos, d_head)
        output = torch.einsum('hpk,hkd->pd', weighted, self.output_matrix)

        return output

    def num_params(self) -> int:
        return (
            self.query_matrix.numel()
            + self.key_matrix.numel()
            + self.value_matrix.numel()
            + self.output_matrix.numel()
        )
