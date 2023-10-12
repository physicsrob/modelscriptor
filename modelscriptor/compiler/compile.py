from typing import Set

from modelscriptor.compiler.groups.strategy import (
    _get_ancestor_nodes,
    get_strategies_for_component,
    get_combined_strategies,
)
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.compiler.report import make_report
from modelscriptor.compiler.transformer import HeadlessTransformer, Transformer
from modelscriptor.graph import Node, InputNode, Constant, PosEncoding, Embedding

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
        or isinstance(node, Embedding)
        for node in layer.attn.in_state.get_compilable_nodes()
    )


def compile_network(
    d: int,
    d_head: int,
    output_node: Node,
    report_name: str = "",
    verbose: bool = True,
    optimize: bool = True,
) -> HeadlessTransformer:
    # Start with the first layer and try to compile as much as possible.

    net = HeadlessTransformer(d, d_head)

    layer = net.add_layer()
    layer.ffn.out_state.allocate_node(output_node)
    needed_nodes = _get_ancestor_nodes({output_node})

    for layer_cnt in range(MAX_LAYERS):
        for sublayer_type in ["ffn", "attn"]:
            if sublayer_type == "ffn":
                sublayer = layer.ffn
            elif sublayer_type == "attn":
                sublayer = layer.attn
            else:
                assert False

            if verbose:
                print(f"\n\nCompiling layer {layer_cnt} {sublayer_type}")
                sublayer.out_state.print(f"Sublayer {sublayer_type} Output")
                sublayer.in_state.print(f"Sublayer {sublayer_type} Starting Input")
                print("\n\n")

            to_compile_nodes = sublayer.out_state.get_compilable_nodes()

            group_strategies = get_combined_strategies(
                {node: sublayer.get_strategies(node) for node in to_compile_nodes}
            )

            print("Combined Strategies considered:")
            for s in group_strategies:
                print(s.sub_strategies)
                sublayer.print_strategy(s)
                print("Score: ", s.get_score())
                # s.print([sublayer], [sublayer_type])

            strategy = group_strategies[0]

            sublayer.in_state._sanity_check()
            sublayer.out_state._sanity_check()
            sublayer.apply_skip_allocation(strategy)
            sublayer.in_state._sanity_check()
            sublayer.out_state._sanity_check()

            sublayer.apply_strategy(strategy)
            sublayer.in_state._sanity_check()
            sublayer.out_state._sanity_check()

            if sublayer == layer.ffn:
                print("Connecting attn.out_state from layer.ffn.in_state")
                layer.attn.out_state.update_from(layer.ffn.in_state)
            sublayer.in_state.print("Sublayer Input")
            sublayer.out_state.print("Sublayer Output")

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

    if optimize:
        # Find the minimum width (d) for the network
        min_d = 0
        for layer in net.layers:
            for sublayer in [layer.ffn, layer.attn]:
                min_d = max(sublayer.get_min_width(), min_d)

        if verbose:
            print("Optimizing network from a width of {d} to {min_d}.")

        for layer in net.layers:
            for sublayer in [layer.ffn, layer.attn]:
                sublayer.resize(min_d)

        net.d = min_d

    make_report(net, output_node, report_name)

    return net


def compile_transformer(
    d: int,
    d_head: int,
    output_node: Node,
    report_name: str = "",
    verbose: bool = True,
    optimize: bool = True,
):
    # Compile everything but the embedding layer
    headless_net = compile_network(
        d, d_head, output_node, report_name, verbose, optimize
    )
    net = Transformer(headless_net.d, d_head)
    net.layers = headless_net.layers

    in_nodes = net.layers[0].attn.in_state.get_distinct_nodes()
    strategies = []
    for node in in_nodes:
        node_strategies = net.embed.get_strategies(node)
        if len(node_strategies) != 1:
            raise CompilationError(f"Expected 1 strategy for {node}")
        strategies.append(node_strategies[0])

    for stratey in strategies:
        net.embed.apply_strategy(stratey)
