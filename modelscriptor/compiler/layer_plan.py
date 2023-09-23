from abc import abstractmethod, ABC
from typing import Set, Optional

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


class FFNPlan(LayerPlan):
    # Plan for FFN, not counting skip.
    linear1_output_nodes: Set[Node]
    relu_output_nodes: Set[Node]
    linear2_output_nodes: Set[Node]

    def __init__(self):
        super().__init__()
        self.linear1_output_nodes = set()
        self.relu_output_nodes = set()
        self.linear2_output_nodes = set()

    def print(self):
        print("FFNPlan:")
        for node in self.linear1_output_nodes:
            print(f"linear1:   {repr(node)}")
        for node in self.relu_output_nodes:
            print(f"relu   :   {repr(node)}")
        for node in self.linear2_output_nodes:
            print(f"linear2:   {repr(node)}")

    def get_input_nodes(self) -> Set[Node]:
        result = set()
        for relu_inp in self.linear1_output_nodes:
            if isinstance(relu_inp, Linear):
                result.update(relu_inp.inputs)
            else:
                result.add(relu_inp)
        return result

    def get_output_nodes(self) -> Set[Node]:
        return self.linear2_output_nodes

    def can_add_output(self, output_node: Node) -> bool:
        # A node can be added to a FFN if it's a linear layer with a relu parent,
        # or just a relu.
        if isinstance(output_node, Linear):
            parent = output_node.inputs[0]
            return isinstance(parent, ReLU)
        else:
            return isinstance(output_node, ReLU)

    def add_output(self, output_node: Node):
        current_node = output_node

        # Process linear2 layer
        self.linear2_output_nodes.add(current_node)
        if isinstance(current_node, Linear):
            current_node = current_node.inputs[0]
        # Otherwise identity matrix will be added, we don't go up the graph.

        # Process relu layer
        assert isinstance(current_node, ReLU)

        self.relu_output_nodes.add(current_node)
        current_node = current_node.inputs[0]

        # Process relu inputs
        self.linear1_output_nodes.add(current_node)


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
