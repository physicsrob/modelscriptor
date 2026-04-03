import torch

from torchwright.compiler.components.linear import LinearLayerComponent
from torchwright.compiler.components.relu import ReLULayerComponent
from torchwright.compiler.feature_assignment import ResidualStreamState


class FFNSubLayer:
    """Feed-forward sublayer: linear1 -> relu -> linear2 + residual skip.

    Forward: out = linear2(relu(linear1(inp))) + inp
    """

    def __init__(self, d: int):
        self.d = d
        self.in_state = ResidualStreamState(name="FFNSubLayer In State")
        self.out_state = ResidualStreamState(name="FFNSubLayer Out State")
        self.linear1 = LinearLayerComponent(d, name="linear1")
        self.relu = ReLULayerComponent(d, name="relu")
        self.linear2 = LinearLayerComponent(d, name="linear2")

    def forward(self, inp: torch.Tensor, return_states=False):
        states = {}

        x = self.linear1.forward(inp)
        states["linear1_out_state"] = (self.linear1.out_state, x)
        x = self.relu.forward(x)
        states["relu_out_state"] = (self.relu.out_state, x)
        x = self.linear2.forward(x)
        states["linear2_out_state"] = (self.linear2.out_state, x)
        x = x + inp
        states["skip"] = (self.out_state, x)
        if return_states:
            return x, states
        else:
            return x

    def num_params(self):
        return self.linear1.num_params() + self.linear2.num_params()

    def to(self, device):
        self.linear1.to(device)
        self.relu.to(device)
        self.linear2.to(device)
        return self

    def resize(self, new_d):
        self.d = new_d
        self.linear1.resize(new_d)
        self.linear2.resize(new_d)
        self.relu.resize(new_d)
