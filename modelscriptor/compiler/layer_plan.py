from abc import abstractmethod, ABC
from typing import Set, Optional, Type, List

from modelscriptor.graph import Node, Linear, Add, Attn
from modelscriptor.graph.relu import ReLU


class LayerPlan(ABC):
    def __init__(self):
        ...

    def is_plan_empty(self) -> bool:
        # A plan is empty iff all input nodes are output nodes
        return len(self.get_input_nodes().union(self.get_output_nodes())) == len(
            self.get_input_nodes()
        )

    @abstractmethod
    def get_input_nodes(self) -> Set[Node]:
        ...

    @abstractmethod
    def get_output_nodes(self) -> Set[Node]:
        ...

    @abstractmethod
    def can_add_output(self, output_node: Node) -> bool:
        ...

    @abstractmethod
    def can_pass_output(self, output_node: Node) -> bool:
        ...

    @abstractmethod
    def add_output(self, output_node: Node) -> Set[Node]:
        ...


class SingleLayerPlan(LayerPlan):
    in_nodes: Set[Node]
    out_nodes: Set[Node]
    node_type: Type[Node]

    def __init__(self, node_type: Type[Node]):
        super().__init__()
        self.node_type = node_type
        self.in_nodes = set()
        self.out_nodes = set()

    def print(self, prefix=""):
        for node in self.in_nodes:
            print(f"{prefix}  in:   {repr(node)}")
        for node in self.out_nodes:
            print(f"{prefix} out:   {repr(node)}")

    def get_input_nodes(self) -> Set[Node]:
        return self.in_nodes

    def get_output_nodes(self) -> Set[Node]:
        return self.out_nodes

    def can_add_output(self, output_node: Node) -> bool:
        return isinstance(output_node, self.node_type)

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


class TwoLayerPlan(LayerPlan):
    node_type1: Type[Node]
    node_type2: Type[Node]
    plan1: SingleLayerPlan
    plan2: SingleLayerPlan

    def __init__(self, node_type1: Type[Node], node_type2: Type[Node]):
        super().__init__()
        self.node_type1 = node_type1
        self.node_type2 = node_type2
        self.plan1 = SingleLayerPlan(node_type1)
        self.plan2 = SingleLayerPlan(node_type2)

    def print(self, prefix=""):
        self.plan1.print(f"{prefix} {self.node_type1}")
        self.plan2.print(f"{prefix} {self.node_type2}")

    def get_input_nodes(self) -> Set[Node]:
        return self.plan1.get_input_nodes()

    def get_output_nodes(self) -> Set[Node]:
        return self.plan2.get_output_nodes()

    def can_add_output(self, output_node: Node) -> bool:
        for strategy2 in ['pass', 'output']:
            if strategy2 == 'pass':
                if not self.plan2.can_pass_output(output_node):
                    continue
                plan1_outputs = [output_node]
            else:
                if not self.plan2.can_add_output(output_node):
                    continue
                plan2_outputs = output_node.inputs

            for strategy1 in ['pass', 'output']:
                if strategy1 == 'pass':
                    if not all(self.plan1.can_pass_output(inp)
        return (self.plan2.can_add_output(output_node) and all(self.plan1.can_add_output(inp) for inp in output_node.inputs)) or
        (self.plan2.can_pass_output(output_node) and self.plan1.can_add_output(output_node))
        return isinstance(output_node, self.node_type)

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
    plans: List[SingleLayerPlan]

    def __init__(self, node_types: List[Type[Node]]):
        super().__init__()
        self.node_types = node_types
        self.plans = [SingleLayerPlan(node_type) for node_type in node_types]

    def print(self, prefix=""):
        for node in self.in_nodes:
            print(f"{prefix}  in:   {repr(node)}")
        for node in self.out_nodes:
            print(f"{prefix} out:   {repr(node)}")

    def get_input_nodes(self) -> Set[Node]:
        return self.in_nodes

    def get_output_nodes(self) -> Set[Node]:
        return self.out_nodes

    def can_add_output(self, output_node: Node) -> bool:
        return isinstance(output_node, self.node_type)

    def can_pass_output(self, output_node: Node) -> bool:
        return all(plan.can_pass_output(output_node) for plan in self.plans)

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

    def get_input_nodes(self) -> Set[Node]:
        return self.linear1_plan.get_input_nodes()

    def get_output_nodes(self) -> Set[Node]:
        return self.linear2_plan.get_output_nodes()

    def can_add_output(self, output_node: Node) -> bool:
        for plan1, plan2, plan3 in [
            ("pass", "pass", "add"),
            ("pass", "pass", "pass"),
            ("pass", "add", "add"),
            ("pass", "add", "pass"),
            ("add", "pass", "add"),
            ("add", "pass", "pass"),
            ("add", "add", "add"),
            ("add", "add", "pass"),
        ]:
            check1 = self.linear1_plan.can_add_output()

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

    def get_input_nodes(self) -> Set[Node]:
        return self.ffn_plan.get_input_nodes().union(self.res_input_nodes)

    def get_output_nodes(self) -> Set[Node]:
        return self.res_output_nodes

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

    def get_input_nodes(self) -> Set[Node]:
        return self.res_input_nodes.union(
            {inp for attn in self.attn_nodes for inp in attn.inputs}
        )

    def get_output_nodes(self) -> Set[Node]:
        return self.res_output_nodes

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

    def get_input_nodes(self) -> Set[Node]:
        return self.attn_plan.get_input_nodes()

    def get_output_nodes(self) -> Set[Node]:
        return self.ffn_plan.get_output_nodes()

    def add_output(self, output_node: Node):
        self.ffn_plan.add_output(output_node)

        for ffn_input_node in self.ffn_plan.get_input_nodes():
            self.attn_plan.add_output(ffn_input_node)
