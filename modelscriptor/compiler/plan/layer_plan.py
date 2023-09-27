from abc import abstractmethod, ABC
from typing import Set, List

from modelscriptor.compiler.plan.layer_component import LayerComponent
from modelscriptor.compiler.plan.placement import (
    NodePlacementType,
    CompileStrategy,
)
from modelscriptor.graph import Node, Linear, Add, Attn
from modelscriptor.graph.relu import ReLU


class LayerPlan(ABC):
    @abstractmethod
    def get_strategies(self, output_node: Node) -> List[CompileStrategy]:
        ...

    @abstractmethod
    def add_output(self, output_node: Node, strategy: CompileStrategy):
        ...


class SingleLayerPlan(LayerPlan):
    layer: LayerComponent

    def __init__(self, layer: LayerComponent):
        super().__init__()
        self.layer = layer

    def get_strategies(self, output_node: Node) -> List[CompileStrategy]:
        # For a single layer plan there are only two strategies: pass or compile.
        result = []

        if self.layer.can_pass_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self.layer, output_node, NodePlacementType.pass_through)
            result.append(strategy)

        if self.layer.can_compile_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self.layer, output_node, NodePlacementType.compile)
            result.append(strategy)

        return result

    def add_output(self, output_node: Node, strategy: CompileStrategy):
        self.out_nodes.add(output_node)

        if (
            strategy.get_layer_node_strategy(self.layer, output_node)
            == NodePlacementType.pass_through
        ):
            self.in_nodes.add(output_node)
        elif (
            strategy.get_layer_node_strategy(self.layer, output_node)
            == NodePlacementType.compile
        ):
            self.in_nodes.update(output_node.inputs)
        else:
            assert False


class TwoLayerPlan(LayerPlan):
    plan1: SingleLayerPlan
    plan2: SingleLayerPlan

    def __init__(self, layer1: LayerComponent, layer2: LayerComponent):
        super().__init__()
        self.plan1 = SingleLayerPlan(layer1)
        self.plan2 = SingleLayerPlan(layer2)

    def print(self, prefix=""):
        self.plan1.print(prefix)
        self.plan2.print(prefix)

    def get_strategies(self, output_node: Node) -> List[CompileStrategy]:
        # For a single layer plan there are only two strategies: pass or compile.
        result = []

        if self.layer.can_pass_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self.layer, output_node, NodePlacementType.pass_through)
            result.append(strategy)

        if self.layer.can_compile_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self.layer, output_node, NodePlacementType.compile)
            result.append(strategy)

        return result

    def get_strategies(self, output_node: Node):
        # For a single layer plan there are only two strategies: pass or compile.
        result = []
        if self.layer.can_pass_node(output_node):
            result.add("pass")

        if self.layer.can_compile_node(output_node):
            result.add("compile")
        return result

    def can_pass_output(self, output_node: Node) -> bool:
        return self.node_type == Linear

    def add_output(self, output_node: Node) -> Set[Node]:
        if self.node_type == Linear and not isinstance(output_node, Linear):
            # We can pass the output node through.
            self.in_nodes.add(output_node)
            self.out_nodes.add(output_node)
            return {output_node}
        else:
            assert isinstance(output_node, self.node_type)
            self.out_nodes.add(output_node)
            self.in_nodes.update(output_node.inputs)
            return set(output_node.inputs)


class MultiLayerPlan(LayerPlan):
    layers: List[LayerComponent]

    def __init__(self, layers: List[LayerComponent]):
        super().__init__()
        self.layers = layers

    def print(self, prefix=""):
        for node in self.in_nodes:
            print(f"{prefix}  in:   {repr(node)}")
        for node in self.out_nodes:
            print(f"{prefix} out:   {repr(node)}")

    def get_strategies(self, output_node: Node) -> List[CompileStrategy]:
        # For a single layer plan there are only two strategies: pass or compile.
        result = []

        if self.layer.can_pass_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self.layer, output_node, NodePlacementType.pass_through)
            result.append(strategy)

        if self.layer.can_compile_node(output_node):
            strategy = CompileStrategy()
            strategy.place_node(self.layer, output_node, NodePlacementType.compile)
            result.append(strategy)

        return result


class FFNPlan(LayerPlan):
    # Plan for FFN, not counting skip.
    linear1_plan: SingleLayerPlan
    relu_plan: SingleLayerPlan
    linear2_plan: SingleLayerPlan

    def __init__(self):
        super().__init__()
        self.linear1_plan = SingleLayerPlan(Linear)
        self.relu_plan = SingleLayerPlan(ReLU)
        self.linear2_plan = SingleLayerPlan(Linear)

    def print(self):
        print("FFNPlan:")
        self.linear1_plan.print("linear1")
        self.relu_plan.print("relu   ")
        self.linear2_plan.print("linear2")

    def add_output(self, output_node: Node):
        relu_nodes = self.linear2_plan.add_output(output_node)
        assert all(self.relu_plan.can_add_output(relu_node) for relu_node in relu_nodes)

        linear1_out_nodes = set()
        for node in relu_nodes:
            linear1_out_nodes.update(self.relu_plan.add_output(node))

        assert all(
            self.linear1_plan.can_add_output(n) or self.linear1_plan.can_pass_output(n)
            for n in linear1_out_nodes
        )

        for node in linear1_out_nodes:
            self.linear1_plan.add_output(node)


class FFNSkipPlan(LayerPlan):
    ffn_plan: FFNPlan
    res_input_nodes: Set[Node]
    res_output_nodes: Set[Node]

    def __init__(self):
        super().__init__()
        self.ffn_plan = FFNPlan()
        self.out_nodes = set()
        self.res_input_nodes = set()

    def add_output(self, output_node: Node):
        if isinstance(output_node, Add):
            if self.ffn_plan.can_add_output(output_node.inputs[0]):
                self.ffn_plan.add_output(output_node.inputs[0])
                self.res_input_nodes.add(output_node.inputs[1])
                self.res_output_nodes.add(output_node)
            elif self.ffn_plan.can_add_output(output_node.inputs[1]):
                self.ffn_plan.add_output(output_node.inputs[1])
                self.res_input_nodes.add(output_node.inputs[0])
                self.res_output_nodes.add(output_node)
            else:
                # Neither plan worked, which means we need to skip both branches of the add.
                self.res_input_nodes.add(output_node)
                self.res_output_nodes.add(output_node)
        else:
            # output_node is not an addition.  Try to compile it to the FCN
            if self.ffn_plan.can_add_output(output_node):
                self.ffn_plan.add_output(output_node)
                self.res_output_nodes.add(
                    output_node
                )  # We'll add an Add(0) at compile time.
            else:
                # We can't plan anything!
                self.res_input_nodes.add(output_node)
                self.res_output_nodes.add(output_node)


class AttnSkipPlan(LayerPlan):
    attn_nodes: Set[Attn]
    res_input_nodes: Set[Node]
    res_output_nodes: Set[Node]

    def __init__(self):
        super().__init__()
        self.attn_nodes = set()
        self.res_input_nodes = set()
        self.res_output_nodes = set()

    def add_output(self, output_node: Node):
        if isinstance(output_node, Add):
            if isinstance(output_node.inputs[0], Attn):
                self.attn_nodes.add(output_node.inputs[0])
                self.res_input_nodes.add(output_node.inputs[1])
                self.res_output_nodes.add(output_node)
            elif isinstance(output_node.inputs[1], Attn):
                self.attn_nodes.add(output_node.inputs[1])
                self.res_input_nodes.add(output_node.inputs[0])
                self.res_output_nodes.add(output_node)
            else:
                self.res_input_nodes.add(output_node)
                self.res_output_nodes.add(output_node)
        elif isinstance(output_node, Attn):
            self.attn_nodes.add(output_node)
            self.res_output_nodes.add(
                output_node
            )  # We'll add an Add(0) at compile time.
        else:
            self.res_input_nodes.add(output_node)
            self.res_output_nodes.add(output_node)


class AttnSkipFFNSkipPlan(LayerPlan):
    attn_plan: AttnSkipPlan
    ffn_plan: FFNSkipPlan

    def __init__(self):
        super().__init__()
        self.attn_plan = AttnSkipPlan()
        self.ffn_plan = FFNSkipPlan()

    def add_output(self, output_node: Node):
        self.ffn_plan.add_output(output_node)

        for ffn_input_node in self.ffn_plan.get_input_nodes():
            self.attn_plan.add_output(ffn_input_node)
