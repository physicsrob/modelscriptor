from typing import Set, Dict, List, NamedTuple, Tuple
from abc import ABC, abstractmethod

import torch

from modelscriptor.compiler.components.component import Component, NodeComponentStrategy
from modelscriptor.compiler.groups.strategy import GroupStrategy
from modelscriptor.compiler.res_state import ResState
from modelscriptor.graph import Node, Concatenate, Linear


class Group:
    d: int
    in_state: ResState
    out_state: ResState

    def __init__(self, d):
        self.d = d
        self.in_state = ResState(d)
        self.out_state = ResState(d)

    @abstractmethod
    def forward(self, inp: torch.Tensor, return_states=False):
        ...

    @abstractmethod
    def print_strategy(self, strategy: GroupStrategy):
        ...

    @abstractmethod
    def get_strategies(self, node: Node) -> List[GroupStrategy]:
        ...

    @abstractmethod
    def apply_skip_allocation(self, strategy: GroupStrategy):
        ...

    @abstractmethod
    def apply_strategy(self, strategy: GroupStrategy):
        ...
