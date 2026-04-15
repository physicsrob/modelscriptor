from typing import Optional

import torch
from torchwright.graph import Node
from torchwright.graph.value_type import NodeValueType, linear_output_range

# Causal mask sentinel: future positions are filled with this value before
# softmax.  Must be large enough that no valid logit ever falls below it,
# otherwise the softmax will prefer "hidden" future positions over the
# real current position.  With _QUERY_GAIN = 80 and |score| up to 120,
# the worst valid logit is 80 × (−120) = −9600, still far above −1e6.
CAUSAL_MASK_SENTINEL = -1e6


class Attn(Node):
    """Single causal attention head with explicit Q/K/V/O weight matrices.

    Computes causal (lower-triangular masked) attention:
    ``softmax(Q @ K^T, masked) @ V @ O``

    Inputs are three nodes: ``query_in``, ``key_in``, ``value_in``.
    """

    # query_matrix shape (d_query_in, d_qk)
    query_matrix: torch.Tensor

    # key_matrix shape (d_key_in, d_qk)
    key_matrix: torch.Tensor

    # value_matrix shape (d_value_in, d_v)
    value_matrix: torch.Tensor

    # output_matrix shape (d_v, d_output)
    output_matrix: torch.Tensor

    def __init__(
        self,
        query_in: Node,
        key_in: Node,
        value_in: Node,
        query_matrix: torch.Tensor,
        key_matrix: torch.Tensor,
        value_matrix: torch.Tensor,
        output_matrix: torch.Tensor,
        declared_output_type: Optional[NodeValueType] = None,
    ):
        self.d_qk = query_matrix.shape[1]
        self.d_v = value_matrix.shape[1]
        self.d_query_in = query_matrix.shape[0]
        self.d_key_in = key_matrix.shape[0]
        self.d_value_in = value_matrix.shape[0]

        assert key_matrix.shape[1] == self.d_qk
        assert output_matrix.shape[0] == self.d_v

        self.query_matrix = query_matrix
        self.key_matrix = key_matrix
        self.value_matrix = value_matrix
        self.output_matrix = output_matrix
        # Stashed for compute_value_type (runs inside super().__init__).
        self._declared_output_type = declared_output_type
        super().__init__(output_matrix.shape[1], inputs=[query_in, key_in, value_in])

    def compute_value_type(self) -> NodeValueType:
        if self._declared_output_type is not None:
            return self._declared_output_type
        # Weak default: attention is a convex combination of
        # ``value_in @ V`` vectors, so the post-attention range is a
        # subset of that. Then multiply by O for the final output range.
        value_in = self.inputs[2]
        v_range = value_in.value_type.value_range
        v_after_vm = linear_output_range(v_range, self.value_matrix)
        out_range = linear_output_range(v_after_vm, self.output_matrix)
        return NodeValueType(value_range=out_range)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        query_in_node, key_in_node, value_in_node = self.inputs
        query_in = query_in_node.compute(n_pos, input_values)
        key_in = key_in_node.compute(n_pos, input_values)
        value_in = value_in_node.compute(n_pos, input_values)

        assert query_in.shape == (n_pos, self.d_query_in)
        assert key_in.shape == (n_pos, self.d_key_in)
        assert value_in.shape == (n_pos, self.d_value_in)

        key_values = torch.matmul(key_in, self.key_matrix)
        # key_values shape is (pos, d_qk)
        query_values = torch.matmul(query_in, self.query_matrix)
        # query_values shape is (pos, d_qk)
        attn_logits = query_values.matmul(key_values.t())
        # attn_logits shape is (query pos, key pos)

        # Apply attention mask
        mask = torch.triu(torch.ones_like(attn_logits), diagonal=1)
        attn_logits = torch.where(
            mask == 1,
            torch.full_like(attn_logits, CAUSAL_MASK_SENTINEL),
            attn_logits,
        )

        attn = torch.softmax(attn_logits, dim=1)
        value_values = torch.matmul(value_in, self.value_matrix)
        # value_values shape is (pos, d_v)
        values = attn.matmul(value_values)
        # values shape is now (query pos, d_v)

        values_output = values.matmul(self.output_matrix)
        # values shape is now (query pos, d_output)
        return values_output

    def num_params(self):
        return (
            self.query_matrix.numel()
            + self.key_matrix.numel()
            + self.value_matrix.numel()
            + self.output_matrix.numel()
        )
