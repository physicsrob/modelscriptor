from typing import Union, Optional, Dict, Set, List, Tuple

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


def build_network(
    d: int,
    d_head: int,
    output_node: Node,
    verbose: bool = True,
    pos_encoding: Optional[PosEncoding] = None,
) -> Tuple[
    List[TransformerLayer], FeatureAssignmentConstraints, Dict[Group, GroupStrategy]
]:
    #
    # First Pass of compilation:  choose strategy for each sublayer, building constraints, and strategies.
    # Returns tuple of:
    # - List of transformer layers.
    # - Feature assignment constraints
    # - Dictionary mapping each sublayer to a strategy.
    constraints = FeatureAssignmentConstraints()

    # Define last layer.  We start from the end and work our way to the beginning.
    layer = TransformerLayer(d, d_head, pos_encoding)
    sublayer_to_strategy: Dict[Group, GroupStrategy] = {}
    constraints.add_node_to_state(output_node, layer.ffn.out_state)
    print(f"Added constraint {output_node} in {layer.ffn.out_state}")
    layers = [layer]
    to_compile_nodes: Set[Node] = {output_node}

    for layer_cnt in range(MAX_LAYERS):
        prev_to_compile_nodes = to_compile_nodes
        constraints.add_equivalency(layer.ffn.in_state, layer.attn.out_state)
        for sublayer in [layer.ffn, layer.attn]:
            if not solve(constraints):
                print("Could not solve constraints")
                breakpoint()
            strategies = sublayer.get_strategies(to_compile_nodes, constraints)
            if len(strategies):
                strategy = strategies[0]
                sublayer_to_strategy[sublayer] = strategy
                constraints.update(sublayer.get_constraints(strategy))

                to_compile_nodes = strategy.get_compilable_input_nodes(
                    include_skip=True
                )
                if not solve(constraints):
                    breakpoint()
                    print("Failed to solve constraints")
                    # raise CompilationError("Failed to solve constraints")
            else:
                raise CompilationError("No strategies found for nodes.")

        if prev_to_compile_nodes == to_compile_nodes:
            raise CompilationError("Could not compile network.  Failed at stage 1.")

        if verbose:
            print(
                f"Layer -{layer_cnt} input nodes: "
                + ", ".join(repr(n) for n in to_compile_nodes)
            )

        if is_compilation_complete(to_compile_nodes):
            if verbose:
                print("Stage 1 compilation complete")
            break
        else:
            if verbose:
                print("\n\nCreating new layer")
            layer = TransformerLayer(d, d_head, pos_encoding)
            layers = [layer] + layers
            constraints.add_equivalency(layers[1].attn.in_state, layer.ffn.out_state)

    if not is_compilation_complete(to_compile_nodes):
        raise CompilationError(f"Exceeded maximum number of layers {MAX_LAYERS}.")

    return layers, constraints, sublayer_to_strategy


def compile_network(
    d: int,
    d_head: int,
    output_node: Node,
    report_name: str = "",
    verbose: bool = True,
    pos_encoding: Optional[PosEncoding] = None,
) -> HeadlessTransformer:
    # Passes:
    # First Pass: Choose strategy for each sublayer, build constraints
    # Second pass: Assign feature, generate report.
    # Third pass: Apply strategies

    #
    # First Pass
    #
    layers, constraints, sublayer_to_strategy = build_network(
        d, d_head, output_node, verbose, pos_encoding
    )
    net = HeadlessTransformer(d, d_head, pos_encoding)
    net.layers = layers

    #
    # Second pass: Assign Features
    #
    net.feature_assignment = solve(constraints)

    if not net.feature_assignment:
        raise CompilationError("Could not solve feature assignment problem.")

    if not constraints.check_solution(net.feature_assignment):
        raise CompilationError("Feature assignment solution did not pass validation.")

    net.feature_assignment.print()

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
    # Passes:
    # First Pass: Choose strategy for each sublayer, build constraints
    #       Part 2: Choose input layer strategies, add constraints
    #       Part 3: Chose output layer strategies, add constraints
    # Second pass: Assign feature, generate report.
    # Third pass: Apply strategies

    #
    # First pass part 1: Choose strategies for primary layers
    #
    output_node = unembedding.inp
    layers, constraints, sublayer_to_strategy = build_network(
        d, d_head, output_node, verbose, pos_encoding
    )
    headless_net = HeadlessTransformer(d, d_head, pos_encoding)
    headless_net.layers = layers
    net = Transformer(headless_net, unembedding.embedding.tokenizer)

    prelim_feature_assignment = solve(constraints)

    if not prelim_feature_assignment:
        raise CompilationError(
            "Could not solve feature assignment problem for primary layers."
        )

    #
    # First pass part 2: Add embedding layer.
    #
    in_state = layers[0].attn.in_state
    out_state = layers[-1].ffn.out_state

    first_strategy = sublayer_to_strategy[layers[0].attn]
    to_compile_nodes = first_strategy.get_compilable_input_nodes(include_skip=True)
    pos_strategies = []  # Strategies that will be placed on pos encoding component
    embed_strategies = []  # Strategies that will be placed on the embedding component

    # Compile the input nodes (embedding, pos encoding, constants)
    for node in to_compile_nodes:
        pos_strategies_for_node = net.pos_encoding.get_strategies(node)
        embed_strategies_for_node = net.embed.get_strategies(node)
        if len(pos_strategies_for_node):
            pos_strategies.append(pos_strategies_for_node[0])
            # constraints.update(
            #     net.pos_encoding.get_constraints_for_strategy(
            #         pos_strategies_for_node[0]
            #     )
            # )
        elif len(embed_strategies_for_node):
            embed_strategies.append(embed_strategies_for_node[0])
            # constraints.update(
            #     net.embed.get_constraints_for_strategy(embed_strategies_for_node[0])
            # )
        else:
            raise CompilationError(
                "Unsupport node type at input to transformer layer stack."
            )

    # constraints.add_equivalency(in_state, net.pos_encoding.out_state)
    # constraints.add_equivalency(in_state, net.embed.out_state)

    #
    # First pass part 3: Add last layer (unembedding)
    #
    embed_node = next(node for node in to_compile_nodes if isinstance(node, Embedding))
    # constraints.add_shared_features_constraint(
    #     in_state, embed_node, out_state, output_node
    # )

    #
    # Second pass: Assign Features
    #
    net.feature_assignment = solve(constraints)

    if not net.feature_assignment:
        raise CompilationError("Could not solve feature assignment problem.")

    if not constraints.check_solution(net.feature_assignment):
        raise CompilationError("Feature assignment solution did not pass validation.")

    print(net.feature_assignment)

    make_report(net.headless_net, output_node, report_name)

    #
    # Third pass: Apply strategies
    #
    for layer in headless_net.layers:
        for sublayer in [layer.ffn, layer.attn]:
            strategy = sublayer_to_strategy[sublayer]
            sublayer.apply_strategy(net.feature_assignment, strategy)

    # Apply input strategies
    for s in embed_strategies:
        net.embed.apply_strategy(net.feature_assignment, s)

    for s in pos_strategies:
        net.pos_encoding.apply_strategy(net.feature_assignment, s)

    return net
