from typing import Set, Dict, List, NamedTuple, Tuple
from abc import ABC, abstractmethod

import torch

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
    ResidualStreamState,
)
from modelscriptor.compiler.groups.strategy import GroupStrategy
from modelscriptor.graph import Node, Concatenate, Linear


class Group:
    d: int
    in_state: ResidualStreamState
    out_state: ResidualStreamState

    def __init__(self, d):
        self.d = d
        self.in_state = ResidualStreamState()
        self.out_state = ResidualStreamState()

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
    def get_constraints(self, strategy: GroupStrategy) -> FeatureAssignmentConstraints:
        ...

    @abstractmethod
    def apply_strategy(
        self, feature_assignment: FeatureAssignment, strategy: GroupStrategy
    ):
        ...

    @abstractmethod
    def resize(self, new_d):
        ...
