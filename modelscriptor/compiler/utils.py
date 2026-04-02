from typing import Set

from modelscriptor.graph import Node


def get_ancestor_nodes(start_nodes: Set[Node]) -> Set[Node]:
    # Find all ancestors via iterative BFS
    result = set(start_nodes)
    queue = list(start_nodes)
    while queue:
        node = queue.pop()
        for inp in node.inputs:
            if inp not in result:
                result.add(inp)
                queue.append(inp)
    return result
