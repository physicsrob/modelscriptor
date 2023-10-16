from typing import List

from modelscriptor.graph import Node
import torch

from modelscriptor.graph.spherical_codes import get_spherical_codes

unk_token = "<unk>"
special_tokens = [unk_token]


class Tokenizer:
    vocab: List[str]

    def __init__(self, vocab: List[str]):
        self.vocab = special_tokens + vocab

    def get_token_id(self, text: str) -> int:
        if text in self.vocab:
            return self.vocab.index(text)
        else:
            return self.vocab.index(unk_token)

    def decode_id(self, token_id: int) -> str:
        if token_id < len(self.vocab):
            return self.vocab[token_id]
        else:
            return unk_token

    def __len__(self):
        return len(self.vocab)


class Embedding(Node):
    tokenizer: Tokenizer
    table: torch.Tensor
    max_vocab: int
    d_embed: int

    def __init__(self, vocab: List[str], d_embed: int = 8):
        assert d_embed == 8, "Only support length-8 embeddings at the moment"
        self.d_embed = d_embed
        self.table = get_spherical_codes(d_embed)
        self.max_vocab = self.table.shape[0]
        self.tokenizer = Tokenizer(vocab)
        assert self.table.shape == (self.max_vocab, d_embed)
        super().__init__(d_embed, [])

    def get_embedding(self, text: str) -> torch.Tensor:
        return self.table[self.tokenizer.get_token_id(text)]

    def compute(self, n_pos: int, input_values: dict):
        assert "embedding_input" in input_values
        embedding_input = input_values["embedding_input"]
        assert len(embedding_input) == n_pos
        token_ids = torch.tensor(
            [self.tokenizer.get_token_id(x) for x in embedding_input], dtype=torch.long
        )
        result = self.table[token_ids]
        assert result.shape == (len(embedding_input), self.d_embed)
        return result

    def num_params(self):
        return self.d_embed * self.max_vocab


class Unembedding:
    def __init__(self, inp: Node, embedding: Embedding):
        self.embedding = embedding
        self.inp = inp
        assert len(self.inp) == self.embedding.d_embed

    def compute(self, n_pos: int, input_values: dict) -> List[str]:
        input_value = self.inp.compute(n_pos, input_values)
        result = []
        for pos in range(n_pos):
            probs = self.embedding.table @ input_value[pos]
            token_id = probs.argmax().item()
            assert isinstance(token_id, int)
            result.append(self.embedding.tokenizer.decode_id(token_id))
        return result
