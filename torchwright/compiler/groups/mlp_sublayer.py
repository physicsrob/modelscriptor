from typing import Optional

import torch

from torchwright.compiler.components.linear import LinearLayerComponent
from torchwright.compiler.components.relu import ReLULayerComponent
from torchwright.compiler.residual_assignment import ResidualStreamState


class MLPSubLayer:
    """MLP sublayer: linear1 -> relu -> linear2 + residual skip.

    Forward: out = linear2(relu(linear1(inp))) + inp

    ``d`` is the residual stream width; ``d_hidden`` is the MLP hidden
    width (the number of neurons / packed slots per layer).  They are
    independent — passing ``d_hidden=None`` defaults it to ``d``.
    """

    def __init__(self, d: int, d_hidden: Optional[int] = None):
        if d_hidden is None:
            d_hidden = d
        self.d = d
        self.d_hidden = d_hidden
        self.in_state = ResidualStreamState(name="MLPSubLayer In State")
        self.out_state = ResidualStreamState(name="MLPSubLayer Out State")
        self.linear1 = LinearLayerComponent(d, d_hidden, name="linear1")
        self.relu = ReLULayerComponent(d_hidden, name="relu")
        self.linear2 = LinearLayerComponent(d_hidden, d, name="linear2")

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

    def trim_unused_slots(self):
        """Remove trailing unused (all-zero) hidden slots after compilation.

        Slots are allocated contiguously from index 0, so slicing
        [:used] is safe.  Keeps at least 1 slot to avoid degenerate
        empty-tensor shapes.
        """
        # linear1 output columns and linear2 input rows correspond to slots.
        # A slot is unused if its linear1 column and linear2 row are both zero.
        l1_used = (self.linear1.output_matrix != 0).any(dim=0)  # (d_hidden,)
        l2_used = (self.linear2.output_matrix != 0).any(dim=1)  # (d_hidden,)
        used = l1_used | l2_used
        if used.any():
            last_used = used.nonzero()[-1].item() + 1
        else:
            last_used = 1
        n = max(last_used, 1)
        if n < self.d_hidden:
            self.linear1.output_matrix = self.linear1.output_matrix[:, :n].contiguous()
            self.linear1.output_bias = self.linear1.output_bias[:n].contiguous()
            self.linear2.output_matrix = self.linear2.output_matrix[:n, :].contiguous()
            self.d_hidden = n
            self.relu.d = n

    def num_params(self):
        return self.linear1.num_params() + self.linear2.num_params()

    def to(self, device):
        self.linear1.to(device)
        self.relu.to(device)
        self.linear2.to(device)
        return self
