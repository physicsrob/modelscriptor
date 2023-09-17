from typing import List, Dict, Optional

import torch


class Node:
    def __init__(self, d_output: int):
        self.d_output = d_output

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self):
        return self.d_output
