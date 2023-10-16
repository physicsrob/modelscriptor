from typing import Union, Optional, Dict

from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.strategy import (
    get_combined_strategies,
    GroupStrategy,
)
from modelscriptor.compiler.utils import get_ancestor_nodes
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.compiler.report import make_report
from modelscriptor.compiler.transformer import HeadlessTransformer, Transformer
from modelscriptor.graph import Node, InputNode, Constant, PosEncoding, Embedding
from modelscriptor.graph.embedding import Unembedding

MAX_LAYERS = 100


class CompilationError(Exception):
    pass


def is_compilation_complete(layer: TransformerLayer) -> bool:
    return all(
        isinstance(node, InputNode)
        or isinstance(node, Constant)
        or isinstance(node, PosEncoding)
        or isinstance(node, Embedding)
        for node in layer.attn.in_state.get_nodes_with_indices()
    )


class OverallStrategy:
    layer_to_strategy: Dict[TransformerLayer, GroupStrategy]


def compile_layer(layer: TransformerLayer, verbose: bool = False):
    for sublayer_type in ["ffn", "attn"]:
        sublayer: Union[FFNSubLayer, AttnSubLayer]
        if sublayer_type == "ffn":
            sublayer = layer.ffn
        elif sublayer_type == "attn":
            sublayer = layer.attn
        else:
            assert False

        if verbose:
            sublayer.out_state.print(f"Sublayer {sublayer_type} Output")
            sublayer.in_state.print(f"Sublayer {sublayer_type} Starting Input")

        to_compile_nodes = sublayer.out_state.get_nodes_with_indices()

        node_to_strategies = {
            node: sublayer.get_strategies(node) for node in to_compile_nodes
        }
        for node, strategies in node_to_strategies.items():
            if len(strategies) > 1 and strategies[0].get_score() == 0:
                node_to_strategies[node] = [strategies[0]]

        group_strategies = get_combined_strategies(node_to_strategies)

        strategy = group_strategies[0]
        if verbose:
            print("Combined Strategy:")
            print(strategy.sub_strategies)
            sublayer.print_strategy(strategy)
            print("Score: ", strategy.get_score())

        sublayer.in_state._consistency_check()
        sublayer.out_state._consistency_check()
        sublayer.apply_pre_allocation(strategy)
        sublayer.in_state._consistency_check()
        sublayer.out_state._consistency_check()

        sublayer.apply_strategy(strategy)
        sublayer.in_state._consistency_check()
        sublayer.out_state._consistency_check()


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

    for layer_cnt in range(MAX_LAYERS):
        compile_layer(layer, verbose)
        if (
            layer.ffn.out_state.get_nodes_with_indices()
            == layer.attn.in_state.get_nodes_with_indices()
        ):
            raise CompilationError("Could not compile network.")

        if is_compilation_complete(layer):
            if verbose:
                print("Compilation complete")
                layer.ffn.out_state.print("Final Layer Output")
                layer.attn.in_state.print("Final Layer Input")
            break
        else:
            new_layer = net.add_layer()
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
    unembedding: Unembedding,
    pos_encoding: PosEncoding,
    report_name: str = "",
    verbose: bool = True,
    optimize: bool = True,
) -> Transformer:
    headless_net = compile_network(
        d=d,
        d_head=d_head,
        output_node=unembedding.inp,
        report_name=report_name,
        verbose=verbose,
        optimize=optimize,
        pos_encoding=pos_encoding,
    )
    net = Transformer(headless_net, unembedding.embedding.tokenizer)

    # We need to update the input to have the same position as the output.

    # Get the nodes at the input to the transformer layer stack
    in_state = net.headless_net.layers[0].attn.in_state
    out_state = net.headless_net.layers[-1].ffn.out_state

    # Compile the position encoding nodes
    for node in in_state.get_nodes_with_indices():
        pos_encoding_strategies = net.pos_encoding.get_strategies(node)
        embed_node_strategies = net.embed.get_strategies(node)
        if len(pos_encoding_strategies):
            net.pos_encoding.apply_strategy(pos_encoding_strategies[0])
        elif len(embed_node_strategies):
            net.embed.apply_strategy(embed_node_strategies[0])
        else:
            raise CompilationError(
                "Unsupport node type at input to transformer layer stack."
            )

    # Check to see if input and output have the same allocation, as they must.
    embed_node = next(
        node
        for node in in_state.get_nodes_with_indices()
        if isinstance(node, Embedding)
    )
    out_node = unembedding.inp
    if in_state.get_node_indices(embed_node) != out_state.get_node_indices(out_node):
        raise CompilationError("Input and output node must have the same allocation.")
        # # Rewire
        # layer = net.headless_net.add_layer(end=True)
        # layer.attn.in_state.copy_from(out_state)
        # # FIXME FIXME FIXME -- implement the rewrite here?
        # from_indices = out_state.get_node_indices(out_node)
        # to_indices = in_state.get_node_indices(embed_node)
        # for from_idx, to_idx in zip(from_indices, to_indices):
        #
        #
        # layer.ffn.in_state.copy_from(layer.attn.out_state)
        # layer.ffn.out_state.copy_from(layer.attn.out_state)

    return net
