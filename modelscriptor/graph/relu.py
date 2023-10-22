import torch
from modelscriptor.graph import Node


class ReLU(Node):
    def __init__(self, input_node: Node, name: str = ""):
        super().__init__(len(input_node), [input_node], name=name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.inputs[0].compute(n_pos, input_values)
        assert x.shape == (n_pos, len(self.inputs[0]))

        # Apply ReLU operation
        return torch.clamp(x, min=0.0)
