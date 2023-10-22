from typing import Union, Optional, Dict, Set, List

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    solve,
)
from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.group import Group
from modelscriptor.compiler.groups.strategy import (
    GroupStrategy,
)
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.compiler.report import make_report
from modelscriptor.compiler.transformer import HeadlessTransformer, Transformer
from modelscriptor.graph import Node, InputNode, Constant, PosEncoding, Embedding
from modelscriptor.graph.embedding import Unembedding

MAX_LAYERS = 100


class CompilationError(Exception):
    pass


def is_compilation_complete(nodes: Set[Node]) -> bool:
    return all(
        isinstance(node, InputNode)
        or isinstance(node, Constant)
        or isinstance(node, PosEncoding)
        or isinstance(node, Embedding)
        for node in nodes
    )


def compile_network(
    d: int,
    d_head: int,
    output_node: Node,
    report_name: str = "",
    verbose: bool = True,
    pos_encoding: Optional[PosEncoding] = None,
) -> HeadlessTransformer:
    net = HeadlessTransformer(d, d_head, pos_encoding)

    # Passes:
    # First Pass: Choose strategy for each sublayer, build constraints
    # Second pass: Assign feature, generate report.
    # Third pass: Apply strategies

    #
    # First Pass
    #
    sublayer_to_strategy: Dict[Group, GroupStrategy] = {}
    constraints = FeatureAssignmentConstraints()

    layer = net.add_layer()
    constraints.add_node_to_state(output_node, layer.ffn.out_state)
    to_compile_nodes: Set[Node] = {output_node}

    for layer_cnt in range(MAX_LAYERS):
        prev_to_compile_nodes = to_compile_nodes
        for sublayer in [layer.ffn, layer.attn]:
            strategies = sublayer.get_strategies(to_compile_nodes)
            if len(strategies):
                strategy = strategies[0]
                # print("Applying strategy:")
                # strategy.print()
                # if len(strategies) > 1:
                #     print("Other strategies considered:")
                #     for strategy in strategies[1:]:
                #         strategy.print()

                sublayer_to_strategy[sublayer] = strategy
                constraints.update(sublayer.get_constraints(strategy))

                to_compile_nodes = strategy.get_compilable_input_nodes(
                    include_skip=True
                )
            else:
                print("No strategy found")
                breakpoint()
            if not len(to_compile_nodes):
                print()
                breakpoint()
                print()

        constraints.add_equivalency(layer.ffn.in_state, layer.attn.out_state)

        if prev_to_compile_nodes == to_compile_nodes:
            breakpoint()
            raise CompilationError("Could not compile network.  Failed at stage 1.")

        if verbose:
            print(
                f"Layer -{layer_cnt} input nodes: "
                + ", ".join(repr(n) for n in to_compile_nodes)
            )

        if is_compilation_complete(to_compile_nodes):
            if verbose:
                print("Compilation complete")
            break
        else:
            new_layer = net.add_layer()
            if verbose:
                print("\n\nCreating new layer")
            constraints.add_equivalency(layer.attn.in_state, new_layer.ffn.out_state)
            layer = new_layer

    if not is_compilation_complete(to_compile_nodes):
        try:
            net.feature_assignment = solve(constraints)
            make_report(net, output_node, report_name + "_failed")
        except:
            pass
        raise CompilationError(f"Exceeded maximum number of layers {MAX_LAYERS}.")

    #
    # Second pass: Assign Features
    #
    net.feature_assignment = solve(constraints)
    net.constraints = constraints

    if not net.feature_assignment:
        raise CompilationError("Could not solve feature assignment problem.")

    if not constraints.check_solution(net.feature_assignment):
        raise CompilationError("Feature assignment solution did not pass validation.")

    print(net.feature_assignment)

    make_report(net, output_node, report_name)

    #
    # Third pass: Apply strategies
    #
    for layer in net.layers:
        for sublayer in [layer.ffn, layer.attn]:
            strategy = sublayer_to_strategy[sublayer]
            sublayer.apply_strategy(net.feature_assignment, strategy)

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
        pos_encoding=pos_encoding,
    )
    feature_assignment = headless_net.feature_assignment
    assert feature_assignment

    net = Transformer(headless_net, unembedding.embedding.tokenizer)

    # We need to update the input to have the same position as the output.

    # Get the nodes at the input to the transformer layer stack
    in_state = net.headless_net.layers[0].attn.in_state
    out_state = net.headless_net.layers[-1].ffn.out_state

    # Compile the input nodes (embedding, pos encoding, constants)
    for node in feature_assignment.get_nodes(in_state):
        pos_encoding_strategies = net.pos_encoding.get_strategies(node)
        embed_node_strategies = net.embed.get_strategies(node)
        if len(pos_encoding_strategies):
            net.pos_encoding.apply_strategy(
                feature_assignment, pos_encoding_strategies[0]
            )
        elif len(embed_node_strategies):
            net.embed.apply_strategy(feature_assignment, embed_node_strategies[0])
        else:
            raise CompilationError(
                "Unsupport node type at input to transformer layer stack."
            )

    # Check to see if input and output have the same allocation, as they must.
    embed_node = next(
        node
        for node in feature_assignment.get_nodes(in_state)
        if isinstance(node, Embedding)
    )
    out_node = unembedding.inp
    if feature_assignment.get_node_indices(
        in_state, embed_node
    ) != feature_assignment.get_node_indices(out_state, out_node):
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
