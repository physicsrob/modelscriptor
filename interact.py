from examples.adder import create_network
from modelscriptor.graph.embedding import Unembedding


def eval_network(output_node: Unembedding, input_text: str):
    in_tokens = list(input_text)
    n_pos = len(in_tokens)
    output = output_node.compute(
        n_pos=n_pos,
        input_values={"embedding_input": in_tokens},
    )
    return "".join(output)


output = create_network()

while True:
    text = input(": ")
    if text == "Q" or text == "q":
        print("Bye")
        exit(0)
    output_txt = eval_network(output, text)
    print(output_txt)
