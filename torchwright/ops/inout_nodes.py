from typing import List

from torchwright.graph import Node, InputNode, LiteralValue, Embedding, PosEncoding
import torch

from torchwright.graph.embedding import Unembedding


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


def create_literal_value(vector: torch.Tensor, name: str = "") -> Node:
    """
    Create a node with a literal value.

    Args:
    - vector (torch.Tensor): Tensor representing the literal value.

    Returns:
    - Node: Node with the specified literal value.
    """
    return LiteralValue(vector, name)


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
