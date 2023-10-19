from typing import Set, Dict, List, NamedTuple, TypeVar, Generic
from abc import ABC, abstractmethod

import torch

from modelscriptor.compiler.feature_assignment import (
    ResidualStreamState,
    FeatureAssignmentConstraints,
    FeatureAssignment,
)
from modelscriptor.graph import Node, Concatenate, Linear


class NodeComponentStrategy:
    # Represents the way a node passes through and is transformed by one component
    in_nodes: List[Node]  # Input nodes used for this computation
    out_node: Node  # Output node fhor this computation

    def __init__(self, in_nodes: List[Node], out_node: Node):
        self.in_nodes = in_nodes
        self.out_node = out_node

    def __repr__(self):
        return (
            f"NodeComponentStrategy(in_nodes={self.in_nodes}, out_node={self.out_node})"
        )


T = TypeVar("T", bound=NodeComponentStrategy)


class Component(Generic[T], ABC):
    d: int
    in_state: ResidualStreamState
    out_state: ResidualStreamState

    def __init__(self, d: int):
        self.d = d
        self.in_state = ResidualStreamState()
        self.out_state = ResidualStreamState()

    @abstractmethod
    def get_strategies(self, node: Node) -> List[T]:
        ...

    def get_constraints_for_strategy(
        self, strategy: NodeComponentStrategy
    ) -> FeatureAssignmentConstraints:
        constraints = FeatureAssignmentConstraints()
        constraints.add_node_to_state(strategy.out_node, self.out_state)
        for node in strategy.in_nodes:
            constraints.add_node_to_state(node, self.in_state)
        return constraints

    @abstractmethod
    def apply_strategy(self, feature_assignment: FeatureAssignment, strategy: T):
        ...

    @abstractmethod
    def num_params(self) -> int:
        ...

    def resize(self, new_d):
        self.d = new_d
