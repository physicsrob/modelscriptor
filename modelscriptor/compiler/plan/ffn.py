from typing import List

from modelscriptor.compiler.plan.layer_component import (
    LinearLayerComponent,
    ReLULayerComponent,
)
from modelscriptor.compiler.plan.placement import CompileStrategy
from modelscriptor.compiler.plan.sequential_layer_components import (
    get_sequential_placement_strategies,
)
from modelscriptor.graph import Node


class FFN:
    linear1: LinearLayerComponent
    relu: ReLULayerComponent
    linear2: LinearLayerComponent

    def __init__(self):
        self.linear1 = LinearLayerComponent()
        self.relu = ReLULayerComponent()
        self.linear2 = LinearLayerComponent()

    def get_placement_strategies(self, output_node: Node) -> List[CompileStrategy]:
        strategies = get_sequential_placement_strategies(
            {output_node}, [self.linear1, self.relu, self.linear2]
        )


class FFNSkip:
    linear1: LinearLayerComponent
    relu: ReLULayerComponent
    linear2: LinearLayerComponent
    res: ResidualLayerComponent

    def __init__(self):
        self.linear1 = LinearLayerComponent()
        self.relu = ReLULayerComponent()
        self.linear2 = LinearLayerComponent()

    def get_placement_strategies(self, output_node: Node) -> List[CompileStrategy]:
        strategies = get_sequential_placement_strategies(
            {output_node}, [self.linear1, self.relu, self.linear2]
        )
