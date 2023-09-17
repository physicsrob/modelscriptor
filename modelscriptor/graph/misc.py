from typing import List, Dict, Optional
from modelscriptor.graph import Node

import torch


class InputNode(Node):
    def __init__(self, name: str, d_output: int):
        self.name = name
        super().__init__(d_output)

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
        super().__init__(sum(len(x) for x in inputs))
        self.inputs = inputs

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return torch.cat([x.compute(n_pos, input_values) for x in self.inputs], dim=-1)


class Add(Node):
    def __init__(self, input1: Node, input2: Node):
        super().__init__(len(input1))
        self.input1 = input1
        self.input2 = input2

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return self.input1.compute(n_pos, input_values) + self.input2.compute(
            n_pos, input_values
        )


class Constant(Node):
    def __init__(self, value: torch.Tensor):
        assert len(value.shape) == 1
        self.value = value
        super().__init__(len(value))

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.value.unsqueeze(0).expand(n_pos, -1)
        assert x.shape == (n_pos, len(self))
        return x


class ValueLogger(Node):
    def __init__(self, inp: Node, name: str):
        self.name = name
        self.inp = inp
        super().__init__(len(inp))

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.inp.compute(n_pos, input_values)
        print(f"ValueLogger({self.name}): shape={x.shape} value={x}")
        return x
