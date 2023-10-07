from typing import Set

from modelscriptor.compiler.report import make_report
from modelscriptor.compiler.transformer import FFNNetwork
from modelscriptor.graph import Node, InputNode, Constant

MAX_LAYERS = 100


def _get_input_nodes(output_node: Node) -> Set[Node]:
    # Find all ancestors to node.
    if not output_node.inputs:
        return {output_node}
    else:
        result = set()
        for n in output_node.inputs:
            result |= _get_input_nodes(n)
        return result


class CompilationError(Exception):
    pass


def compile_ffn_network(d: int, output_node: Node, verbose: bool = False) -> FFNNetwork:
    # Start with the first layer and try to compile as much as possible.

    net = FFNNetwork(d)

    to_compile_nodes = {output_node}
    layer = net.add_layer()
    layer.out_state.allocate_node(output_node)

    for layer_cnt in range(MAX_LAYERS):
        if verbose:
            print(f"\n\nCompiling layer {layer_cnt}")
            print(f"Nodes to be compiled: {to_compile_nodes}")
            layer.out_state.print("Layer Output")
            print("\n\n")

        applied_strategy_cnt = 0
        chosen_strategies = []
        for node in to_compile_nodes:
            strategies = sorted(
                layer.get_strategies(node), key=lambda s: -s.get_score()
            )
            if len(strategies):
                chosen_strategies.append(strategies[0])
                if verbose:
                    print("\n\nBest strategy: ")
                    layer.print_strategy(strategies[0])
                    print("\n\n")
                layer.apply_skip_allocation(strategies[0])
            elif verbose:
                print("No strategy for: ", node)

        for chosen_strategy in chosen_strategies:
            layer.apply_strategy(chosen_strategy)
            applied_strategy_cnt += 1

        if verbose:
            print(f"Applied {applied_strategy_cnt} strategies.")
        to_compile_nodes = layer.in_state.get_compilable_nodes()
        if not len(to_compile_nodes) and not applied_strategy_cnt:
            raise CompilationError("Could not compile network.")

        if all(
            isinstance(node, InputNode) or isinstance(node, Constant)
            for node in to_compile_nodes
        ):
            if verbose:
                print("Compilation complete")
                layer.out_state.print("Final Layer Output")
                layer.in_state.print("Final Layer Input")
            break
        else:
            new_layer = net.add_layer()
            new_layer.out_state.update_from(layer.in_state)
            if verbose:
                print("\n\nCreating new layer:")
                layer.out_state.print("Old Layer Output")
                layer.in_state.print("Old Layer Input")
                new_layer.out_state.print("New Layer Output")
                new_layer.in_state.print("New Layer Input")
            layer = new_layer

    make_report(net)

    return net
