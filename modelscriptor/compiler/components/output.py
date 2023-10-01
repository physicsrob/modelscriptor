from typing import List

from modelscriptor.compiler.components.component import Component, NodeComponentStrategy
from modelscriptor.graph import Node


class OutputLayer(Component):
    def __init__(self, d: int, output_node: Node):
        super().__init__(d)
        self.out_state.allocate_node(output_node)
        self.in_state = self.out_state

    def apply_strategy(self, strategy: NodeComponentStrategy):
        raise NotImplemented

    def get_strategies(self, node: Node) -> List[NodeComponentStrategy]:
        return []
