from typing import NamedTuple, Union, Optional, Set, Callable

from modelscriptor.graph import (
    Node,
    PosEncoding,
    InputNode,
    Constant,
    Attn,
    Linear,
    Concatenate,
    Add,
)
from modelscriptor.graph.embedding import Unembedding, Embedding
from modelscriptor.graph.relu import ReLU


class CategorizedNodes(NamedTuple):
    embedding: Set[Embedding]
    pos_encoding: Set[PosEncoding]
    input: Set[InputNode]
    constant: Set[Constant]
    linear: Set[Linear]
    relu: Set[ReLU]
    attn: Set[Attn]
    concatenate: Set[Concatenate]
    add: Set[Add]


def categorize_all_nodes(all_nodes: Set[Node]):
    embedding: Set[Embedding] = set()
    pos_encoding: Set[PosEncoding] = set()
    input_nodes: Set[InputNode] = set()
    constant: Set[Constant] = set()
    linear: Set[Linear] = set()
    relu: Set[ReLU] = set()
    attn: Set[Attn] = set()
    concatenate: Set[Concatenate] = set()
    add: Set[Add] = set()

    for node in all_nodes:
        if isinstance(node, Embedding):
            embedding.add(node)
        elif isinstance(node, PosEncoding):
            pos_encoding.add(node)
        elif isinstance(node, InputNode):
            input_nodes.add(node)
        elif isinstance(node, Constant):
            constant.add(node)
        elif isinstance(node, Linear):
            linear.add(node)
        elif isinstance(node, ReLU):
            relu.add(node)
        elif isinstance(node, Attn):
            attn.add(node)
        elif isinstance(node, Concatenate):
            concatenate.add(node)
        elif isinstance(node, Add):
            add.add(node)
        else:
            raise Exception("Unsupported node type")

    return CategorizedNodes(
        embedding=embedding,
        pos_encoding=pos_encoding,
        input=input_nodes,
        constant=constant,
        linear=linear,
        relu=relu,
        attn=attn,
        concatenate=concatenate,
        add=add,
    )


def find_ancestor_nodes(output_node: Node, key: Callable[[Node], bool]) -> Set[Node]:
    # Traverse the graph finding nodes that match the filter.
    result = set()
    for node in output_node.inputs:
        if key(node):
            result.add(key)
        result.update(find_ancestor_nodes(node, key))
    return result


def get_children(output_node: Node, node: Node) -> Set[Node]:
    # Find all nodes that are ancestors to output_node, and direct children of node.
    result = set()
    for n in output_node.inputs:
        if n == node:
            result.add(n)
        result.update(get_children(n, node))
    return result


def get_ancestors(node: Node, stop_nodes: Optional[Set[Node]] = None) -> Set[Node]:
    # Find all ancestors to node.
    result = set()
    for n in node.inputs:
        if stop_nodes is None or n not in stop_nodes:
            result.add(n)
            result.update(get_ancestors(n, stop_nodes))
    return result


def insert_node(parent_node: Node, child_node: Node, new_node: Node):
    assert parent_node in child_node.inputs
    assert (
        parent_node in new_node.inputs
    ), "New node must already use parent_node as input"

    replaced = 0
    for i, n in enumerate(child_node.inputs):
        if n == parent_node:
            child_node.inputs[i] = new_node
            replaced += 1
    assert replaced == 1
