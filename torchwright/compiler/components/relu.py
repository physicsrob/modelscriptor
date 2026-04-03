import torch

from torchwright.compiler.components.component import Component


class ReLULayerComponent(Component):
    """ReLU activation component. No parameters."""

    def __init__(self, d: int, name: str = ""):
        super().__init__(d, name)

    def __repr__(self):
        return f"ReLULayerComponent(name='{self.name}')"

    def forward(self, inp: torch.Tensor):
        return torch.clamp(inp, min=0)

    def num_params(self) -> int:
        return 0
