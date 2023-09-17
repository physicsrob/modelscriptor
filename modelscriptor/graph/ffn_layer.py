import torch
from modelscriptor.graph import Node


class FFNLayer(Node):
    def __init__(
        self,
        input_node: Node,
        input_proj: torch.Tensor,
        input_bias: torch.Tensor,
        output_proj: torch.Tensor,
        output_bias: torch.Tensor,
    ):
        self.input_node = input_node

        if len(input_proj.shape) == 1:
            input_proj = input_proj.unsqueeze(0)
        if len(input_bias.shape) == 0:
            input_bias = input_bias.unsqueeze(0)
        if len(output_proj.shape) == 1:
            output_proj = output_proj.unsqueeze(0)

        self.d_int = input_proj.shape[0]
        self.d_input = input_proj.shape[1]
        super().__init__(output_proj.shape[1])

        assert input_proj.shape == (self.d_int, self.d_input)
        assert input_bias.shape == (self.d_int,)
        assert output_proj.shape == (self.d_int, self.d_output)
        assert output_bias.shape == (self.d_output,)

        self.input_proj = input_proj
        self.input_bias = input_bias
        self.output_proj = output_proj
        self.output_bias = output_bias

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.input_node.compute(n_pos, input_values)
        assert x.shape == (n_pos, self.d_input)
        # Compute the dot product for each n_pos entry.
        # Shape: (n_pos, d_input) * (d_input, d_int) = (n_pos, d_int)
        batch_dot_product = torch.matmul(x, self.input_proj.t())

        # Add the biases.
        biased_output = batch_dot_product + self.input_bias

        # Apply ReLU operation
        relu_output = torch.clamp(biased_output, min=0.0)
        # print(f"{relu_output=}")
        assert relu_output.shape == (n_pos, self.d_int)

        # Multiply by self.output_vec. Resultant shape: (batch, N, D)
        # Shape: (n_pos, d_int) * (d_int, d_output) = (n_pos, d_output)
        y = torch.matmul(relu_output, self.output_proj)
        # print(f"{self.output_proj=}")
        # print(f"{y=}")

        return y + self.output_bias.unsqueeze(0)
