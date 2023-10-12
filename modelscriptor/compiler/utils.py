from typing import Set

from modelscriptor.graph import Node


def get_ancestor_nodes(start_nodes: Set[Node]) -> Set[Node]:
    # Find all ancestors
    result = set()

    for node in start_nodes:
        result.add(node)
        if node.inputs:
            result |= get_ancestor_nodes(set(node.inputs))
    return result


def get_input_nodes(output_node: Node) -> Set[Node]:
    # Find all ancestors to node.
    if not output_node.inputs:
        return {output_node}
    else:
        result = set()
        for n in output_node.inputs:
            result |= get_input_nodes(n)
        return result
