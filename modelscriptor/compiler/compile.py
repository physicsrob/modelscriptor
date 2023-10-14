from typing import Union, Optional

from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.strategy import (
    get_combined_strategies,
)
from modelscriptor.compiler.utils import get_ancestor_nodes
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.compiler.report import make_report
from modelscriptor.compiler.transformer import HeadlessTransformer, Transformer
from modelscriptor.graph import Node, InputNode, Constant, PosEncoding, Embedding

MAX_LAYERS = 100


class CompilationError(Exception):
    pass


def is_compilation_complete(layer: TransformerLayer) -> bool:
    return all(
        isinstance(node, InputNode)
        or isinstance(node, Constant)
        or isinstance(node, PosEncoding)
        or isinstance(node, Embedding)
        for node in layer.attn.in_state.get_nodes()
    )


def compile_network(
    d: int,
    d_head: int,
    output_node: Node,
    report_name: str = "",
    verbose: bool = True,
    optimize: bool = True,
    pos_encoding: Optional[PosEncoding] = None,
) -> HeadlessTransformer:
    # Start with the first layer and try to compile as much as possible.

    net = HeadlessTransformer(d, d_head, pos_encoding)

    layer = net.add_layer()
    layer.ffn.out_state.allocate_node(output_node)
    needed_nodes = get_ancestor_nodes({output_node})

    for layer_cnt in range(MAX_LAYERS):
        for sublayer_type in ["ffn", "attn"]:
            sublayer: Union[FFNSubLayer, AttnSubLayer]
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

            to_compile_nodes = sublayer.out_state.get_nodes()

            group_strategies = get_combined_strategies(
                {node: sublayer.get_strategies(node) for node in to_compile_nodes}
            )

            strategy = group_strategies[0]
            print("Combined Strategy:")
            print(strategy.sub_strategies)
            sublayer.print_strategy(strategy)
            print("Score: ", strategy.get_score())

            sublayer.in_state._consistency_check()
            sublayer.out_state._consistency_check()
            sublayer.apply_skip_allocation(strategy)
            sublayer.in_state._consistency_check()
            sublayer.out_state._consistency_check()

            sublayer.apply_strategy(strategy)
            sublayer.in_state._consistency_check()
            sublayer.out_state._consistency_check()

            if sublayer == layer.ffn:
                print("Connecting attn.out_state from layer.ffn.in_state")
                layer.attn.out_state.update_from(layer.ffn.in_state)
            sublayer.in_state.print("Sublayer Input")
            sublayer.out_state.print("Sublayer Output")

        if layer.ffn.out_state.get_nodes() == layer.attn.in_state.get_nodes():
            import pdb

            pdb.set_trace()
            raise CompilationError("Could not compile network.")
        else:
            print(
                f"Nodes changed from {layer.ffn.out_state.get_nodes()} to {layer.attn.in_state.get_nodes()}"
            )

        if str(layer.ffn.out_state.get_nodes()) == str(layer.attn.in_state.get_nodes()):
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
            min_d = max(layer.ffn.get_min_width(), layer.attn.get_min_width(), min_d)

        if verbose:
            print("Optimizing network from a width of {d} to {min_d}.")

        for layer in net.layers:
            layer.ffn.resize(min_d)
            layer.attn.resize(min_d)

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
    pos_encoding: Optional[PosEncoding] = None,
):
    headless_net = compile_network(
        d=d,
        d_head=d_head,
        output_node=output_node,
        report_name=report_name,
        verbose=verbose,
        optimize=optimize,
        pos_encoding=pos_encoding,
    )
    net = Transformer(headless_net)

    in_nodes = net.headless_net.layers[0].attn.in_state.get_nodes()
    strategies = []
    for node in in_nodes:
        node_strategies = net.embed.get_strategies(node)
        if len(node_strategies) != 1:
            raise CompilationError(f"Expected 1 strategy for {node}")
        strategies.append(node_strategies[0])

    for stratey in strategies:
        net.embed.apply_strategy(stratey)
