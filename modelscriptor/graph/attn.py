import torch
from modelscriptor.graph import Node


class Attn(Node):
    def __init__(
        self,
        query_in: Node,
        key_in: Node,
        value_in: Node,
        query_matrix: torch.Tensor,
        key_matrix: torch.Tensor,
        value_matrix: torch.Tensor,
        output_matrix: torch.Tensor,
    ):
        self.query_in = query_in
        self.key_in = key_in
        self.value_in = value_in

        # query_matrix shape (d_query_in, d_head)
        # key_matrix shape (d_key_in, d_head)
        # value_matrix shape (d_value_in, d_head)
        # output_matrix shape (d_head, d_output)
        self.d_head = query_matrix.shape[1]
        self.d_query_in = query_matrix.shape[0]
        self.d_key_in = key_matrix.shape[0]
        self.d_value_in = value_matrix.shape[0]
        super().__init__(output_matrix.shape[1])

        assert query_matrix.shape[1] == self.d_head
        assert key_matrix.shape[1] == self.d_head
        assert value_matrix.shape[1] == self.d_head

        self.query_matrix = query_matrix
        self.key_matrix = key_matrix
        self.value_matrix = value_matrix
        self.output_matrix = output_matrix

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        query_in = self.query_in.compute(n_pos, input_values)
        key_in = self.key_in.compute(n_pos, input_values)
        value_in = self.value_in.compute(n_pos, input_values)

        assert query_in.shape == (n_pos, self.d_query_in)
        assert key_in.shape == (n_pos, self.d_key_in)
        assert value_in.shape == (n_pos, self.d_value_in)

        key_values = torch.matmul(key_in, self.key_matrix)
        # key_values shape is (pos, d_head)
        query_values = torch.matmul(query_in, self.query_matrix)
        # query_values shape is (pos, d_head)
        attn_logits = query_values.matmul(key_values.t())
        # attn_logits shape is (query pos, key pos)

        # Apply attention mask
        mask = torch.triu(torch.ones_like(attn_logits), diagonal=1)
        attn_logits = torch.where(
            mask == 1, -1000 * torch.ones_like(attn_logits), attn_logits
        )

        attn = torch.softmax(attn_logits, dim=1)
        value_values = torch.matmul(value_in, self.value_matrix)
        # value_values shape is (pos, d_head)
        values = attn.matmul(value_values)
        # values shape is now (query pos, d_head)

        values_output = values.matmul(self.output_matrix)
        # values shape is now (query pos, d_output)
        return values_output
