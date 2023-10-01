from typing import Set

from modelscriptor.compiler.transformer import FFNNetwork
from modelscriptor.graph import Node, InputNode, Constant


def _get_input_nodes(output_node: Node) -> Set[Node]:
    # Find all ancestors to node.
    if not output_node.inputs:
        return {output_node}
    else:
        result = set()
        for n in output_node.inputs:
            result |= _get_input_nodes(n)
        return result


def compile_ffn_network(d: int, output_node: Node) -> FFNNetwork:
    # Start with the first layer and try to compile as much as possible.

    net = FFNNetwork(d)

    to_compile_nodes = {output_node}
    layer = net.add_layer()
    layer.out_state.allocate_node(output_node)

    while True:
        print("Compiling")
        applied_strategy_cnt = 0
        print(f"{to_compile_nodes=}")
        for node in to_compile_nodes:
            strategies = sorted(
                layer.get_strategies(output_node), key=lambda s: -s.get_score()
            )
            if len(strategies):
                layer.apply_strategy(strategies[0])
                applied_strategy_cnt += 1

        print(f"Applied {applied_strategy_cnt} strategies.")
        to_compile_nodes = layer.in_state.nodes
        if all(
            isinstance(node, InputNode) or isinstance(node, Constant)
            for node in to_compile_nodes
        ):
            break
        else:
            new_layer = net.add_layer()
            new_layer.out_state.update_from(layer.in_state)
            layer = new_layer

    return net
