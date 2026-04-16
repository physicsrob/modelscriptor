import torch
from torchwright.graph import Node
from torchwright.graph.value_type import NodeValueType


class ReLU(Node):
    def __init__(self, input_node: Node, name: str = ""):
        super().__init__(len(input_node), [input_node], name=name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.inputs[0].compute(n_pos, input_values)
        assert x.shape == (n_pos, len(self.inputs[0]))

        # Apply ReLU operation
        return torch.clamp(x, min=0.0)

    def compute_value_type(self) -> NodeValueType:
        from torchwright.graph.value_type import _max_guarantee, _min_guarantee

        t = self.inputs[0].value_type
        new_range = t.value_range.relu()
        # sign-valued input ({-1, +1}) maps to binary {0, 1}; binary stays
        # binary; one-hot-ness is preserved only for already-binary inputs.
        new_binary = _max_guarantee(t.is_binary, _min_guarantee(t.is_sign, t.is_integer))
        new_one_hot = _min_guarantee(t.is_one_hot, t.is_binary)
        return NodeValueType(
            value_range=new_range,
            is_integer=t.is_integer,
            is_binary=new_binary,
            is_sign=False,
            is_one_hot=new_one_hot,
        )
