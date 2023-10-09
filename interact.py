from examples.adder import create_network
from modelscriptor.graph.embedding import Unembedding

max_chars = 10


def eval_network(output_node: Unembedding, input_text: str):
    in_tokens = ["<bos>"] + list(input_text)
    for i in range(max_chars):
        n_pos = len(in_tokens)
        out_tokens = output_node.compute(
            n_pos=n_pos,
            input_values={"embedding_input": in_tokens},
        )
        if out_tokens[-1] == "<eos>":
            break
        else:
            in_tokens.append(out_tokens[-1])

    return "".join(t for t in out_tokens if t != "<bos>" and t != "<eos>")


output = create_network()

while True:
    text = input(": ")
    if text == "Q" or text == "q":
        print("Bye")
        exit(0)
    output_txt = eval_network(output, text)
    print(output_txt)
