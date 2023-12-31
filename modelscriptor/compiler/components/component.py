from typing import Set, Dict, List, NamedTuple, TypeVar, Generic
from abc import ABC, abstractmethod

import torch

from modelscriptor.compiler.res_state import ResState
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
    in_state: ResState
    out_state: ResState

    def __init__(self, d):
        self.d = d
        self.in_state = ResState(d)
        self.out_state = ResState(d)

    def print(self):
        print(f"{repr(self)}")
        self.in_state.print("in ")
        self.out_state.print("in ")

    @abstractmethod
    def get_strategies(self, node: Node) -> List[T]:
        ...

    @abstractmethod
    def apply_strategy(self, strategy: T):
        ...

    @abstractmethod
    def num_params(self) -> int:
        ...

    def resize(self, new_d):
        self.d = new_d
        self.in_state.resize(new_d)
        self.out_state.resize(new_d)

    def get_min_width(self):
        return max(self.in_state.get_min_width(), self.out_state.get_min_width())
