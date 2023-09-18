import torch
from modelscriptor.graph import Node


class ReLU(Node):
    def __init__(
        self,
        input_node: Node,
    ):
        self.input_node = input_node
        super().__init__(len(input_node), [input_node])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.input_node.compute(n_pos, input_values)
        assert x.shape == (n_pos, len(self.input_node))

        # Apply ReLU operation
        return torch.clamp(x, min=0.0)
