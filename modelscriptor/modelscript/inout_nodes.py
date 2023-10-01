from typing import List

from modelscriptor.graph import Node, InputNode, Constant, Embedding, PosEncoding
import torch

from modelscriptor.graph.embedding import Unembedding


def create_input(name: str, d: int) -> Node:
    """
    Create an input node with a specified name and dimension.

    Args:
    - name (str): Name of the input node.
    - d (int): Dimension of the input node.

    Returns:
    - Node: The created input node.
    """
    return InputNode(name, d)


def create_constant(vector: torch.Tensor) -> Node:
    """
    Create a node with a constant value.

    Args:
    - vector (torch.Tensor): Tensor representing the constant value.

    Returns:
    - Node: Node with the specified constant value.
    """
    return Constant(vector)


def create_embedding(vocab: List[str]) -> Embedding:
    """
    Create an embedding input.

    Args:
    - vocab (List[str]): List of vocab words.

    Returns:
    - Node: Embedding node.
    """
    return Embedding(vocab)


def create_unembedding(inp: Node, embedding: Embedding) -> Unembedding:
    """
    Create an unembedding output.

    Args:
    - inp (Node): Node with embedding vector to unembed
    - embedding (Embedding): Embedding instance to use for unembedding.

    Returns:
    - Unembedding
    """
    return Unembedding(inp, embedding)


def create_pos_encoding() -> PosEncoding:
    """
    Create a position encoding.

    Returns:
    - Node: PosEncoding node.
    """
    return PosEncoding(d_pos=16)
