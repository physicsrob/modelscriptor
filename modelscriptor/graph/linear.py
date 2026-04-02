from typing import List, Dict, Optional
from modelscriptor.graph import Node

import torch


class Linear(Node):
    # This is a base type which can be implemented as either a FFN or an Attention Head (attending
    # to the current token), or optimized to within one of the above.
    # TODO: Consider updating FFN and AttnHead.
    # e.g. add ReLU layer
    # FFN = LinearBLock(ReLU(LinearBlock(input_node, ...), ...)
    # AttnHead = Attn(LinearBlock(query_in, ...), LinearBlock(key_in, ...), LinearBlock(value_in, ...)
    output_matrix: torch.Tensor  # d_input x d_output
    output_bias: torch.Tensor  # d_output

    def __init__(
        self,
        input_node: Node,
        output_matrix: torch.Tensor,
        output_bias: Optional[torch.Tensor] = None,
        name: str = "",
    ):
        # output_matrix shape (d_input, d_output)
        self.d_input = output_matrix.shape[0]
        self.d_output = output_matrix.shape[1]
        assert len(input_node) == self.d_input
        self.output_matrix = output_matrix

        if output_bias is None:
            self.output_bias = torch.zeros(self.d_output)
        else:
            assert len(output_bias) == self.d_output
            self.output_bias = output_bias

        super().__init__(self.d_output, [input_node], name=name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        value_in = self.inputs[0].compute(n_pos, input_values)

        assert value_in.shape == (n_pos, self.d_input)
        return torch.matmul(value_in, self.output_matrix) + self.output_bias

    def num_params(self):
        return self.d_input * self.d_output + self.d_output
