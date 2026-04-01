from typing import Tuple

import torch

from modelscriptor.compiler.compile import compile_network, compile_transformer
from modelscriptor.compiler.transformer import HeadlessTransformer
from modelscriptor.graph import Node, Embedding
from modelscriptor.modelscript.arithmetic_ops import concat
from examples.adder import create_network_parts

from modelscriptor.modelscript.inout_nodes import (
    create_embedding,
    create_pos_encoding,
    create_constant,
    create_unembedding,
)
from modelscriptor.modelscript.logic_ops import compare_to_vector
from modelscriptor.modelscript.map_select import select, map_to_table


def test_compile_embedding():
    vocab = ["1", "2", "3", "was1", "was2", "was3", "default"]
    embedding = create_embedding(vocab=vocab)
    table_values = {"1": "was1", "2": "was2", "3": "was3"}
    lookup = map_to_table(
        inp=embedding,
        key_to_value={
            embedding.get_embedding(text): embedding.get_embedding(value)
            for text, value in table_values.items()
        },
        default=embedding.get_embedding("default"),
    )

    net = compile_network(128, 16, lookup, report_name="compile_embedding")


def check_is_num(embedding_value: Node, embedding: Embedding) -> Node:
    return map_to_table(
        inp=embedding_value,
        key_to_value={
            embedding.get_embedding(str(i)): torch.tensor([1.0]) for i in range(10)
        },
        default=torch.tensor([-1.0]),
    )


def sum_numbers(embedding: Embedding, num1: Node, num2: Node) -> Tuple[Node, Node]:
    """
    Adds num1 with num2.
    Assumes num1 and num2 are both embedding-valued nodes.
    return result as embedding-valued node and carry as boolean.
    """
    result_table = {}
    carry_table = {}
    for A in range(10):
        for B in range(10):
            numcat = torch.cat(
                [embedding.get_embedding(str(A)), embedding.get_embedding(str(B))]
            )
            result_table[numcat] = embedding.get_embedding(str((A + B) % 10))
            carry_table[numcat] = torch.tensor([1.0 if (A + B) >= 10 else -1.0])

    num1_num2 = concat([num1, num2])
    return (
        map_to_table(
            inp=num1_num2,
            key_to_value=result_table,
            default=embedding.get_embedding("0"),
        ),
        map_to_table(num1_num2, key_to_value=carry_table, default=torch.tensor([-1.0])),
    )


def test_adder_1digit():
    # Define our vocabulary -- these are the tokens that will be used for our netowrk.
    vocab = list(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    #
    # Make network that adds 1 digit numbers
    #

    # Define current number.
    zero_constant = create_constant(embedding.get_embedding("0"))
    is_num = check_is_num(embedding_value=embedding, embedding=embedding)

    # current_num is the embedding of the current character if it is a number,
    # otherwise it is the embedding of 0.
    current_num = select(cond=is_num, true_node=embedding, false_node=zero_constant)

    # Define a flag for the end of the first number (when we hit the + symbol).
    is_first_num = compare_to_vector(inp=embedding, vector=embedding.get_embedding("+"))

    # # Define a flag for the end of the second number (when we hit the = symbol).
    is_second_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("=")
    )
    #
    just_completed_num = pos_encoding.get_last_value(current_num, delta_pos=-1)
    first_num = pos_encoding.get_prev_value(just_completed_num, is_first_num)
    second_num = pos_encoding.get_prev_value(just_completed_num, is_second_num)
    #
    # # Figure out how to calculate output index.
    summed, carry = sum_numbers(embedding, first_num, second_num)
    out = create_unembedding(summed, embedding)
    net = compile_network(
        1024,
        64,
        summed,
        pos_encoding=pos_encoding,
        report_name="summed",
        verbose=True,
    )
    # Verify the compiled network produces correct sums
    test_cases = [
        (["1", "+", "1", "="], "2"),
        (["2", "+", "3", "="], "5"),
        (["0", "+", "0", "="], "0"),
        (["4", "+", "5", "="], "9"),
    ]
    for tokens, expected in test_cases:
        result = net.compute(n_pos=4, input_values={"embedding_input": tokens})
        summed_output = result[summed]
        # Find nearest embedding at the "=" position (position 3)
        dists = torch.cdist(summed_output[3:4], embedding.table)
        predicted = embedding.tokenizer.decode_id(dists.argmin().item())
        assert predicted == expected, (
            f"For {''.join(tokens)}: expected '{expected}' but got '{predicted}'"
        )


def decode_token(embedding: Embedding, vector: torch.Tensor) -> str:
    """Decode a single embedding vector to its nearest token."""
    dists = torch.cdist(vector.unsqueeze(0), embedding.table)
    return embedding.tokenizer.decode_id(dists.argmin().item())


def run_autoregressive(
    net: HeadlessTransformer,
    output_node: Node,
    embedding: Embedding,
    input_tokens: list,
    max_new_tokens: int = 10,
) -> str:
    """Run a compiled network autoregressively, appending output tokens until <eos>."""
    tokens = list(input_tokens)
    for _ in range(max_new_tokens):
        result = net.compute(
            n_pos=len(tokens), input_values={"embedding_input": tokens}
        )
        next_token = decode_token(embedding, result[output_node][-1])
        if next_token == "<eos>":
            break
        tokens.append(next_token)
    # Return only the generated portion (after the input)
    return "".join(tokens[len(input_tokens) :])


def test_adder_autoregressive():
    """Compile the examples/adder network (1-digit) and verify autoregressive output."""
    import examples.adder as adder_module

    original_max_digits = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        output_node, pos_encoding, embedding = create_network_parts()
        net = compile_network(
            256,
            16,
            output_node,
            pos_encoding=pos_encoding,
            report_name="adder_autoregressive",
            verbose=True,
        )

        test_cases = [
            ("1+1=", "2"),
            ("2+3=", "5"),
            ("0+0=", "0"),
            ("4+5=", "9"),
        ]
        for input_str, expected in test_cases:
            tokens = ["<bos>"] + list(input_str)
            result = run_autoregressive(net, output_node, embedding, tokens)
            assert result == expected, (
                f"For {input_str}: expected '{expected}' but got '{result}'"
            )
    finally:
        adder_module.max_digits = original_max_digits
