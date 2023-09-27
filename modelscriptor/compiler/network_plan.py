from typing import List, Set

from modelscriptor.compiler.plan.layer_plan import AttnSkipFFNSkipPlan
from modelscriptor.graph import Node


class NetworkPlan:
    layers: List[AttnSkipFFNSkipPlan]

    def __init__(self, output_node: Node):
        self.layers = []

        current_input_nodes: Set[Node] = {output_node}

        while True:
            next_layer = AttnSkipFFNSkipPlan()
            for node in current_input_nodes:
                next_layer.add_output(node)

            if next_layer.is_plan_empty():
                break

            # Plan was not empty, let's keep going
            current_input_nodes = next_layer.get_input_nodes()
            self.layers.append(next_layer)
