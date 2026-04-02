import torch

from modelscriptor.compiler.components.component import Component


class LinearLayerComponent(Component):
    """Linear projection component.

    Weight matrices:
        output_matrix: (d, d)
        output_bias:   (d,)

    Forward: out = inp @ output_matrix + output_bias
    """

    def __init__(self, d: int, name: str = ""):
        super().__init__(d, name)
        self.output_matrix = torch.zeros(d, d)
        self.output_bias = torch.zeros(d)

    def __repr__(self):
        return f"LinearLayerComponent(name='{self.name}')"

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
