from abc import ABC, abstractmethod
from typing import Set, List

from modelscriptor.compiler.plan.placement import (
    NodePlacementType,
    CompileStrategy,
)
from modelscriptor.graph import Linear, Concatenate, Add, ReLU, Node, Attn


class LayerComponent(ABC):
    in_nodes: Set[Node]
    out_nodes: Set[Node]

    def __init__(self):
        self.in_nodes = set()
        self.out_nodes = set()

    def print(self, prefix=""):
        for node in self.in_nodes:
            print(f"{prefix} {repr(self)}   in:   {repr(node)}")
        for node in self.out_nodes:
            print(f"{prefix} {repr(self)}  out:   {repr(node)}")

    @abstractmethod
    def can_compile_node(self, node):
        ...

    @abstractmethod
    def can_pass_node(self, node):
        ...

    def get_strategies(self, output_node: Node) -> List[CompileStrategy]:
        # For a single layer plan there are only two strategies: pass or compile.
        result = []

        if self.can_pass_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self, output_node, NodePlacementType.pass_through)
            result.append(strategy)

        if self.can_compile_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self, output_node, NodePlacementType.compile)
            result.append(strategy)

        return result

    def apply_strategy(self, output_node: Node, strategy: CompileStrategy):
        self.in_nodes.update(strategy.get_layer_inputs(self))
        self.out_nodes.update(strategy.get_layer_outputs(self))


class LinearLayerComponent(LayerComponent):
    def __repr__(self):
        return "LinearLayerComponent()"

    def can_compile_node(self, node):
        if isinstance(node, Linear):
            return {
                "in": node.inputs[0],
                "out": node,
                "output_matrix": node.output_matrix,
                "output_bias": node.output_bias,
            }
            return True
        if isinstance(node, Concatenate):
            return True
        if isinstance(node, Add):
            return True
        return False

    def can_pass_node(self, node):
        return True


class ReLULayerComponent(LayerComponent):
    def __repr__(self):
        return "ReLULayerComponent()"

    def can_compile_node(self, node):
        return isinstance(node, ReLU)

    def can_pass_node(self, node):
        return False


class AttnLayerComponent(LayerComponent):
    def __repr__(self):
        return "AttnLayerComponent()"

    def can_compile_node(self, node):
        return isinstance(node, Attn)

    def can_pass_node(self, node):
        # We could do this later, which would allow Add() to be implemented in attention layers
        return False


class SkipLayerComponent(LayerComponent):
    def __repr__(self):
        return "SkipLayerComponent()"

    def can_compile_node(self, node):
        return isinstance(node, Add)

    def can_pass_node(self, node):
        return True
