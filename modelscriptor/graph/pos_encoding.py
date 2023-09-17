from modelscriptor.graph import Node, Concatenate, Attn
import torch
import math

attention_hardness = 100.0  # Scales attention to be 1.0 and 0.0 everywhere elee


class PosEncoding(Node):
    def __init__(self, d_pos: int):
        self.d_pos = d_pos
        super().__init__(d_pos)

    def compute(self, n_pos: int, input_values: dict):
        result = torch.zeros((n_pos, self.d_pos))
        for pos in range(n_pos):
            result[pos] = get_pos_encoding(pos, self.d_pos)
        return result

    def get_last_value(self, value: Node, delta_pos=-1) -> Node:
        # We're applying the query/key to the position
        d_head = 64
        assert self.d_pos < d_head
        assert len(value) < d_head

        # key_matrix shape (d_key_in, d_head)
        key_matrix = torch.zeros((len(self), d_head))
        key_matrix[:, : self.d_pos] = get_pos_delta_matrix(delta_pos, self.d_pos)

        # query_matrix shape (d_query_in, d_head)
        query_matrix = attention_hardness * torch.eye(len(self), d_head)

        # value_matrix shape (d_value_in, d_head)
        value_matrix = torch.eye(len(value), d_head)

        # output_matrix shape (d_head, d_output)
        output_matrix = torch.eye(d_head, len(value))

        return Attn(
            query_in=self,
            key_in=self,
            value_in=value,
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            value_matrix=value_matrix,
            output_matrix=output_matrix,
        )

    def get_prev_value(self, value: Node, cond: Node) -> Node:
        """
        Get the most recent previous value when cond is true.  cond must be 1.0 for true values,
        otherwise we won't guarantee that we return the maximum value.
        """
        # The penultimate component of the position encoding is approximately
        # sin(pos / 10000.0), which for our purposes is monotonically increasing.

        key_in = Concatenate([self, cond])
        d_head = 64

        assert self.d_pos < d_head
        assert len(value) < d_head
        assert len(cond) == 1

        # We'll setup the key matrix such that query * pos_encoding = [(cond + pos[-2]) 0 ... 0]
        # key_matrix shape (d_key_in, d_head)
        key_matrix = torch.zeros((len(key_in), d_head))
        key_matrix[-1, 0] = 1.0  # Cond
        key_matrix[-3, 0] = 100.0  # 100 * pos[-2]

        # query_matrix shape (d_query_in, d_head)
        # query_in is self, and the project doesn't really matter as long as it's non-zero.
        query_matrix = attention_hardness * torch.ones(len(self), d_head)

        # value_matrix shape (d_value_in, d_head)
        value_matrix = torch.eye(len(value), d_head)

        # output_matrix shape (d_head, d_output)
        output_matrix = torch.eye(d_head, len(value))

        return Attn(
            query_in=self,
            key_in=key_in,
            value_in=value,
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            value_matrix=value_matrix,
            output_matrix=output_matrix,
        )


def get_pos_encoding(pos: int, d: int):
    # Compute the positional encodings once in log space.
    pe = torch.zeros(d)
    div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
    pe[0::2] = torch.sin(pos * div_term)
    pe[1::2] = torch.cos(pos * div_term)
    return pe


def get_pos_delta_matrix(delta_pos: int, d: int):
    # Calculate a matrix M, such that M * pos_encoding(pos) = pos_encoding(pos + delta_pos)
    # We can do this using the trig identity sin(A+B) = sin(A)cos(B) + cos(A)sin(B)
    M = torch.zeros((d, d))

    # div_term must match from get_pos_encoding.  Note it's length d//2.
    div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))

    # It's helpful to imagine d==2, in which case:
    # pos_encoding(pos) = [[sin(pos * div_term[0])], [cos(pos * div_term[0])]]

    for i in range(d // 2):
        k = div_term[i]
        # Loop through one sin/cos pair at a time
        # We want:
        # M*pos_encoding(pos) = pos_encoding(pos + delta_pos)
        # In the odd row case, we're calculating the new sin values
        # sin(k*pos + k*delta_pos) = sin(k*pos)*cos(k*delta_pos) + cos(k*pos)*sin(k*delta_pos)
        # Which is a linear combination, so can be expressed using our matrix.
        M[2 * i + 0, 2 * i + 0] = torch.cos(k * delta_pos)
        M[2 * i + 0, 2 * i + 1] = torch.sin(k * delta_pos)

        # In the even row case, we're calculating the new cosine values
        # cos(A + B) = cos(A)cos(B) − sin(A)sin(B)
        # cos(k*pos + k*delta_pos) = cos(k*pos)*cos(k*delta_pos) - sin(k*pos)*sin(k*delta_pos)
        M[2 * i + 1, 2 * i + 1] = torch.cos(k * delta_pos)
        M[2 * i + 1, 2 * i + 0] = -torch.sin(k * delta_pos)

    return M
