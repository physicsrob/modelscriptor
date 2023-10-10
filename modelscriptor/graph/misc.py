from typing import List, Dict, Optional
from modelscriptor.graph import Node

import torch


class InputNode(Node):
    def __init__(self, name: str, d_output: int):
        self.name = name
        super().__init__(d_output, [])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        if self.name not in input_values:
            raise ValueError(f"Did not specify value for input variable {self.name}")

        val = input_values[self.name]
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

    def simplify_inputs(self: Node) -> List[Node]:
        # Flatten concatenation and return the list of nodes
        inputs = []
        for n in self.inputs:
            if isinstance(n, Concatenate):
                inputs += n.simplify_inputs()
            else:
                inputs.append(n)
        return inputs


class Add(Node):
    def __init__(self, input1: Node, input2: Node):
        super().__init__(len(input1), [input1, input2])

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


class Constant(Node):
    def __init__(self, value: torch.Tensor):
        assert len(value.shape) == 1
        self.value = value
        super().__init__(len(value), [])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.value.unsqueeze(0).expand(n_pos, -1)
        assert x.shape == (n_pos, len(self))
        return x

    def is_zero(self):
        return self.value.eq(0).all()

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False
        return torch.equal(self.value, other.value)

    def __hash__(self):
        native_floats = map(float, self.value.view(-1).tolist())
        return hash(tuple(native_floats))

    def node_type(self):
        if self.is_zero():
            return "Zero"
        else:
            return "Constant"


class Placeholder(Node):
    # This node-type is length 0 placeholder
    def __init__(self):
        super().__init__(0, [])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return torch.zeros(0)


class ValueLogger(Node):
    def __init__(self, inp: Node, name: str):
        self.name = name
        super().__init__(len(inp), [inp])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        inp = self.inputs[0]
        x = inp.compute(n_pos, input_values)
        print(f"ValueLogger({self.name}): shape={x.shape} value={x}")
        return x
