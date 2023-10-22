from typing import Set, Dict, List, NamedTuple, Tuple
from abc import ABC, abstractmethod

import torch

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
    ResidualStreamState,
)
from modelscriptor.compiler.groups.strategy import (
    GroupStrategy,
    get_combined_strategies,
)
from modelscriptor.graph import Node, Concatenate, Linear


class Group:
    d: int
    in_state: ResidualStreamState
    out_state: ResidualStreamState

    def __init__(self, d, name: str = ""):
        self.d = d
        self.in_state = ResidualStreamState(name=f"{name} Group In State")
        self.out_state = ResidualStreamState(name=f"{name} Group Out State")

    @abstractmethod
    def forward(self, inp: torch.Tensor, return_states=False):
        ...

    @abstractmethod
    def get_strategies_for_node(self, node: Node) -> List[GroupStrategy]:
        ...

    def get_strategies(self, nodes: Set[Node]) -> List[GroupStrategy]:
        node_to_strategies = {
            node: self.get_strategies_for_node(node) for node in nodes
        }
        for node, strategies in node_to_strategies.items():
            if len(strategies) > 1 and strategies[0].get_score() == 0:
                node_to_strategies[node] = [strategies[0]]

        return get_combined_strategies(node_to_strategies)

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
