from typing import List, Dict, Optional
from torchwright.graph import Node

import torch


class InputNode(Node):
    def __init__(self, d_output_or_name, d_output_or_nothing=None, name: str = ""):
        # Support both old and new constructor patterns:
        # - InputNode(d_output) - new anonymous pattern
        # - InputNode(d_output, name=name) - new named pattern
        # - InputNode(name, d_output) - legacy pattern
        if isinstance(d_output_or_name, str):
            # Legacy pattern: InputNode(name, d_output)
            if d_output_or_nothing is None:
                raise ValueError("d_output is required when name is the first argument")
            d_output = d_output_or_nothing
            name = d_output_or_name
        else:
            # New pattern: InputNode(d_output) or InputNode(d_output, name=name)
            d_output = d_output_or_name
            if d_output_or_nothing is not None and isinstance(d_output_or_nothing, str):
                name = d_output_or_nothing
        super().__init__(d_output, [], name=name)

    def compute(self, n_pos: int, input_values: dict, name: str = None) -> torch.Tensor:
        # Use provided name or fall back to stored name
        lookup_name = name if name is not None else self.name
        if not lookup_name:
            raise ValueError(
                "InputNode has no name set and none was provided to compute()"
            )
        if lookup_name not in input_values:
            raise ValueError(f"Did not specify value for input variable {lookup_name}")

        val = input_values[lookup_name]
        assert isinstance(val, torch.Tensor)
        assert val.shape == (
            n_pos,
            self.d_output,
        ), "Input must be of shape (n_pos, d_output)"
        return val

    def __len__(self):
        return self.d_output


class Concatenate(Node):
    def __init__(self, inputs: List[Node]):
        super().__init__(sum(len(x) for x in inputs), inputs)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return torch.cat([x.compute(n_pos, input_values) for x in self.inputs], dim=-1)

    def flatten_inputs(self: Node) -> List[Node]:
        # Flatten concatenation and return the list of nodes
        inputs = []
        for n in self.inputs:
            if isinstance(n, Concatenate):
                inputs += n.flatten_inputs()
            else:
                inputs.append(n)
        return inputs


class Add(Node):
    def __init__(self, input1: Node, input2: Node, name: str = ""):
        super().__init__(len(input1), [input1, input2], name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        input1, input2 = self.inputs
        return input1.compute(n_pos, input_values) + input2.compute(n_pos, input_values)

    def other_input(self, node: Node):
        if self.inputs[0] == node:
            return self.inputs[1]
        elif self.inputs[1] == node:
            return self.inputs[0]
        else:
            raise ValueError("Invalid node choice")


class LiteralValue(Node):
    def __init__(self, value: torch.Tensor, name: str = ""):
        assert len(value.shape) == 1
        self.value = value
        super().__init__(len(value), [], name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.value.unsqueeze(0).expand(n_pos, -1)
        assert x.shape == (n_pos, len(self))
        return x

    def is_zero(self):
        return self.value.eq(0).all()

    def node_type(self):
        if self.is_zero():
            return "Zero"
        else:
            return "LiteralValue"


class Placeholder(Node):
    """Zero-width sentinel used as a stand-in when a real node is not yet available."""

    def __init__(self, d: int = 0):
        super().__init__(d, [])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return torch.zeros(self.d_output)


class ValueLogger(Node):
    def __init__(self, inp: Node, name: str):
        self.name = name
        super().__init__(len(inp), [inp])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        inp = self.inputs[0]
        x = inp.compute(n_pos, input_values)
        print(f"ValueLogger({self.name}): shape={x.shape} value={x}")
        return x
