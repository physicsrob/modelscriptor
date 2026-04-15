"""Graph optimization passes.

These passes transform the computation graph before compilation to reduce
layer count and parameter overhead.
"""

from typing import Dict, List, Set, Tuple

import torch

from torchwright.graph import Node, Concatenate
from torchwright.graph.linear import Linear


def fuse_consecutive_linears(
    output_nodes: Set[Node],
    verbose: bool = False,
) -> int:
    """Fuse consecutive Linear nodes without intervening nonlinearities.

    Given Linear(L1) -> Linear(L2) where L1's only consumer is L2,
    mutates L2 in place to use the fused matrix (L1's input, M1@M2, b1@M2+b2).

    Fusion math:
        L1: y1 = x @ M1 + b1
        L2: y2 = y1 @ M2 + b2
        Fused: y2 = x @ (M1 @ M2) + (b1 @ M2 + b2)

    This saves one layer in the compiled transformer and often reduces
    parameters significantly (especially for bottleneck patterns like
    768 -> 192 -> 3 becoming 768 -> 3).

    The pass mutates nodes in-place so that output_nodes remain valid
    references to the optimized graph.

    Args:
        output_nodes: The graph's output nodes (used to find all ancestors).
        verbose: Print fusion details.

    Returns:
        Number of fusions performed.
    """
    from torchwright.compiler.utils import get_ancestor_nodes

    all_nodes = get_ancestor_nodes(output_nodes)

    # Build consumer map: node -> list of nodes that use it as input
    consumers: Dict[Node, List[Node]] = {n: [] for n in all_nodes}
    for node in all_nodes:
        for inp in node.inputs:
            if inp in consumers:
                consumers[inp].append(node)

    # Find fusion candidates: Linear -> Linear where L1 has single consumer L2
    fusions: List[Tuple[Linear, Linear]] = []
    for node in all_nodes:
        if not isinstance(node, Linear):
            continue
        l2 = node
        inp = l2.inputs[0]

        # Skip if input is a Concatenate (would need special handling)
        if isinstance(inp, Concatenate):
            continue

        if not isinstance(inp, Linear):
            continue
        l1 = inp

        # Only fuse if L1's sole consumer is L2
        # (otherwise L1's output is needed elsewhere)
        if len(consumers[l1]) != 1:
            continue

        # Skip if fusion would increase params (bottleneck patterns)
        # This can happen when d_mid is much smaller than d_in or d_out
        d_in = l1.output_matrix.shape[0]
        d_mid = l1.output_matrix.shape[1]
        d_out = l2.output_matrix.shape[1]
        old_params = d_in * d_mid + d_mid + d_mid * d_out + d_out
        new_params = d_in * d_out + d_out
        if new_params > old_params:
            continue

        fusions.append((l1, l2))

    # Sort upstream-first so L1→L2 is always processed before L2→L3.
    # Without this, if set iteration visits L3 before L2, we'd process
    # (L2,L3) first, leaving L3 depending on L1 — then a second pass
    # finds a new (L1,L3) pair and reports one extra fusion.
    fusions.sort(key=lambda pair: pair[0].node_id)

    if verbose and fusions:
        print(f"Fusing {len(fusions)} consecutive Linear pairs:")

    # Perform fusions by mutating L2 in place
    fused_count = 0
    for l1, l2 in fusions:
        # Compute fused matrix and bias
        # L1: (d_in, d_mid), L2: (d_mid, d_out)
        # Fused: (d_in, d_out)
        fused_matrix = l1.output_matrix @ l2.output_matrix
        fused_bias = l1.output_bias @ l2.output_matrix + l2.output_bias

        d_in = l1.output_matrix.shape[0]
        d_mid = l1.output_matrix.shape[1]
        d_out = l2.output_matrix.shape[1]

        # Param savings
        old_params = d_in * d_mid + d_mid + d_mid * d_out + d_out
        new_params = d_in * d_out + d_out

        if verbose:
            ann = l2.annotation or "(none)"
            print(f"  {ann}: {d_in}x{d_mid}x{d_out} -> {d_in}x{d_out} "
                  f"({old_params:,} -> {new_params:,} params)")

        # Mutate L2 in place to become the fused Linear
        # This preserves L2's identity so consumers and output_nodes stay valid
        l2.inputs = [l1.inputs[0]]
        l2.output_matrix = fused_matrix
        l2.output_bias = fused_bias
        l2.d_input = d_in
        # d_output stays the same
        if l1.name and l2.name:
            l2.name = f"fused_{l1.name}_{l2.name}"

        # L1 is now orphaned (no consumers) - it will be excluded from
        # future get_ancestor_nodes() calls since nothing references it

        fused_count += 1

    return fused_count


def optimize_graph(
    output_nodes: Set[Node],
    verbose: bool = False,
) -> None:
    """Apply all graph optimization passes.

    Modifies the graph in-place by redirecting node inputs to optimized
    versions.

    Args:
        output_nodes: The graph's output nodes.
        verbose: Print optimization details.
    """
    fused = fuse_consecutive_linears(output_nodes, verbose=verbose)
    if verbose:
        print(f"Graph optimization: fused {fused} Linear pairs")
