from typing import Set

from modelscriptor.compiler.groups.strategy import _get_ancestor_nodes
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.compiler.report import make_report
from modelscriptor.compiler.transformer import Transformer
from modelscriptor.graph import Node, InputNode, Constant, PosEncoding

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


def is_compilation_complete(layer: TransformerLayer) -> bool:
    return all(
        isinstance(node, InputNode)
        or isinstance(node, Constant)
        or isinstance(node, PosEncoding)
        for node in layer.attn.in_state.get_compilable_nodes()
    )


def compile_network(
    d: int, d_head: int, output_node: Node, verbose: bool = True
) -> Transformer:
    # Start with the first layer and try to compile as much as possible.

    net = Transformer(d, d_head)

    layer = net.add_layer()
    layer.ffn.out_state.allocate_node(output_node)
    needed_nodes = _get_ancestor_nodes({output_node})

    for layer_cnt in range(MAX_LAYERS):
        if verbose:
            print(f"\n\nCompiling layer {layer_cnt}")
            layer.ffn.out_state.print("Layer FFN Output")
            print("\n\n")

        for sublayer in [layer.ffn, layer.attn]:
            to_compile_nodes = sublayer.out_state.get_compilable_nodes()

            chosen_strategies = []
            for node in to_compile_nodes:
                strategies = sorted(
                    sublayer.get_strategies(node),
                    key=lambda s: -len(s.get_represented_nodes() & needed_nodes),
                )
                if len(strategies):
                    chosen_strategies.append(strategies[0])
                    if verbose:
                        print("\n\nStrategies considered: ")
                        for s in strategies:
                            sublayer.print_strategy(s)
                            print(f"Represented Nodes: {s.get_represented_nodes()}\n")
                            print(f"Score: {s.get_score()}\n")
                        print("\n\n")
                    sublayer.apply_skip_allocation(strategies[0])
                else:
                    raise CompilationError(f"No strategy for {node}")

            for chosen_strategy in chosen_strategies:
                sublayer.apply_strategy(chosen_strategy)

            if sublayer == layer.ffn:
                layer.attn.out_state.update_from(layer.ffn.in_state)

        if (
            layer.ffn.out_state.get_compilable_nodes()
            == layer.attn.in_state.get_compilable_nodes()
        ):
            import pdb

            pdb.set_trace()
            raise CompilationError("Could not compile network.")
        else:
            print(
                f"Nodes changed from {layer.ffn.out_state.get_compilable_nodes()} to {layer.attn.in_state.get_compilable_nodes()}"
            )

        if str(layer.ffn.out_state.get_compilable_nodes()) == str(
            layer.attn.in_state.get_compilable_nodes()
        ):
            import pdb

            pdb.set_trace()

        if is_compilation_complete(layer):
            if verbose:
                print("Compilation complete")
                layer.ffn.out_state.print("Final Layer Output")
                layer.attn.in_state.print("Final Layer Input")
            break
        else:
            new_layer = net.add_layer()
            new_layer.ffn.out_state.update_from(layer.attn.in_state)
            if verbose:
                print("\n\nCreating new layer:")
                layer.ffn.out_state.print("Old Layer Output")
                layer.attn.in_state.print("Old Layer Input")
                new_layer.ffn.out_state.print("New Layer Output")
                new_layer.attn.in_state.print("New Layer Input")
            layer = new_layer

    if not is_compilation_complete(layer):
        raise CompilationError(f"Exceeded maximum number of layers {MAX_LAYERS}.")

    make_report(net)

    return net
