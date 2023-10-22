from typing import List, Dict, Optional

import torch


global_node_id = 0


class Node:
    inputs: List["Node"]
    d_output: int
    node_id: int
    name: str

    def __init__(self, d_output: int, inputs: List["Node"], name: str = ""):
        global global_node_id
        self.d_output = d_output
        self.inputs = inputs
        self.node_id = global_node_id
        self.name = name
        global_node_id += 1

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self):
        return self.d_output

    def replace_input(self, old_input: "Node", new_input: "Node"):
        for i, _ in enumerate(self.inputs):
            if self.inputs[i] == old_input:
                self.inputs[i] = new_input

    def node_type(self):
        return type(self).__name__

    def __repr__(self):
        type_name = self.node_type()
        if len(self.inputs) == 0:
            return f"{type_name}(id={self.node_id}, name='{self.name}', d={len(self)})"
        elif len(self.inputs) == 1:
            inp = self.inputs[0]
            inp_type_name = inp.node_type()
            return f"{type_name}(id={self.node_id}, name='{self.name}', inp={inp_type_name}(id={inp.node_id}, name='{inp.name}', d={len(inp)}), d={len(self)})"
        else:
            inp_strings = []
            for i, inp in enumerate(self.inputs):
                inp_type_name = inp.node_type()
                inp_strings.append(
                    f"inp{i}={inp_type_name}(id={inp.node_id}, name='{inp.name}', d={len(inp)})"
                )
            inp_str = ", ".join(inp_strings)
            return f"{type_name}(id={self.node_id}, name='{self.name}', {inp_str}, d={len(self)})"

    def num_params(self):
        return 0

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __hash__(self):
        return self.node_id
