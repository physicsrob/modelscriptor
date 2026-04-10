"""Convert a compiled HeadlessTransformer to a standard PyTorch nn.Module.

The resulting CompiledTransformerModule accepts token IDs and returns logits,
with all operations expressed as pure tensor ops (ONNX-compatible).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

from torchwright.compiler.device import get_device
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Node, LiteralValue, PosEncoding, Embedding, Concatenate
from torchwright.graph.embedding import Tokenizer
from torchwright.graph.misc import InputNode


class _AttentionLayer(nn.Module):
    """Multi-head causal attention with skip connection.

    Vectorized equivalent of AttnLayerComponent.forward() + skip.
    """

    def __init__(
        self,
        W_Q: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        W_O: torch.Tensor,
        n_heads: int,
        d_head: int,
        causal_mask: torch.Tensor,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.causal_mask: torch.Tensor
        # W_Q, W_K, W_V: (d, d) fused across heads
        self.W_Q = nn.Parameter(W_Q)
        self.W_K = nn.Parameter(W_K)
        self.W_V = nn.Parameter(W_V)
        # W_O: (n_heads, d_head, d)
        self.W_O = nn.Parameter(W_O)
        # causal_mask: (max_seq_len, max_seq_len) bool, True = masked
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        seq_len = inp.shape[0]

        # Project Q, K, V: (seq_len, d) @ (d, d) -> (seq_len, d)
        # Then reshape to (seq_len, n_heads, d_head) and permute to (n_heads, seq_len, d_head)
        Q = (inp @ self.W_Q).view(seq_len, self.n_heads, self.d_head).permute(1, 0, 2)
        K = (inp @ self.W_K).view(seq_len, self.n_heads, self.d_head).permute(1, 0, 2)
        V = (inp @ self.W_V).view(seq_len, self.n_heads, self.d_head).permute(1, 0, 2)

        # Attention logits: (n_heads, seq_len, seq_len)
        attn_logits = torch.bmm(Q, K.transpose(-2, -1))

        # Causal mask: -1000 (not -inf) to match existing behavior
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)
        attn_logits = attn_logits.masked_fill(mask, -1000.0)

        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Apply attention to values: (n_heads, seq_len, d_head)
        attn_out = torch.bmm(attn_weights, V)

        # Output projection: (n_heads, seq_len, d_head) @ (n_heads, d_head, d) -> (n_heads, seq_len, d)
        out = torch.bmm(attn_out, self.W_O)

        # Sum over heads + skip connection
        return out.sum(dim=0) + inp

    def forward_cached(
        self,
        inp: torch.Tensor,
        past_K: torch.Tensor,
        past_V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cached forward.

        Args:
            inp:    (n_new, d) — residual stream for new positions only.
            past_K: (n_heads, n_past, d_head)
            past_V: (n_heads, n_past, d_head)

        Returns:
            (out, new_K, new_V) where out is (n_new, d) with the skip
            connection applied, and new_K / new_V are
            (n_heads, n_past + n_new, d_head).
        """
        n_new = inp.shape[0]

        Q = (inp @ self.W_Q).view(n_new, self.n_heads, self.d_head).permute(1, 0, 2)
        K_new = (inp @ self.W_K).view(n_new, self.n_heads, self.d_head).permute(1, 0, 2)
        V_new = (inp @ self.W_V).view(n_new, self.n_heads, self.d_head).permute(1, 0, 2)

        K = torch.cat([past_K, K_new], dim=1)
        V = torch.cat([past_V, V_new], dim=1)
        n_total = K.shape[1]

        attn_logits = torch.bmm(Q, K.transpose(-2, -1))  # (n_heads, n_new, n_total)

        # Row r (absolute position n_past + r) may attend to columns 0..n_past+r.
        # Mask column c when c > n_past + r, i.e. triu diagonal = n_total - n_new + 1.
        mask = torch.triu(
            torch.ones(n_new, n_total, dtype=torch.bool, device=inp.device),
            diagonal=n_total - n_new + 1,
        ).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(mask, -1000.0)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        out = torch.bmm(attn_out, self.W_O)
        return out.sum(dim=0) + inp, K, V


class _FFNLayer(nn.Module):
    """Feed-forward sublayer: linear1 -> ReLU -> linear2 + skip connection."""

    def __init__(
        self, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor
    ):
        super().__init__()
        self.W1 = nn.Parameter(W1)
        self.b1 = nn.Parameter(b1)
        self.W2 = nn.Parameter(W2)
        self.b2 = nn.Parameter(b2)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp @ self.W1 + self.b1
        x = torch.clamp(x, min=0)
        x = x @ self.W2 + self.b2
        return x + inp


class CompiledTransformerModule(nn.Module):
    """A compiled torchwright transformer as a standard nn.Module.

    Input: token_ids (seq_len,) LongTensor
    Output: logits (seq_len, vocab_size) FloatTensor
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        token_embedding: nn.Embedding,
        embedding_proj: torch.Tensor,
        pos_proj: torch.Tensor,
        constant_values: torch.Tensor,
        pos_encoding: torch.Tensor,
        output_gather_indices: torch.Tensor,
        unembed_table: torch.Tensor,
        tokenizer: Tokenizer,
    ):
        super().__init__()
        self.layers = layers
        self.token_embedding = token_embedding
        self.embedding_proj: torch.Tensor
        self.register_buffer("embedding_proj", embedding_proj)
        self.pos_proj: torch.Tensor
        self.register_buffer("pos_proj", pos_proj)
        self.constant_values: torch.Tensor
        self.register_buffer("constant_values", constant_values)
        self.pos_encoding: torch.Tensor
        self.register_buffer("pos_encoding", pos_encoding)
        self.output_gather_indices: torch.Tensor
        self.register_buffer("output_gather_indices", output_gather_indices)
        self.unembed_table: torch.Tensor
        self.register_buffer("unembed_table", unembed_table)
        self.tokenizer = tokenizer

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        orig_device = token_ids.device
        token_ids = token_ids.to(self.embedding_proj.device)
        seq_len = token_ids.shape[0]

        # Build initial residual stream
        embedded = self.token_embedding(token_ids)  # (seq_len, d_embed)
        pos = self.pos_encoding[:seq_len]  # (seq_len, d_pos)

        res = (
            embedded @ self.embedding_proj + pos @ self.pos_proj + self.constant_values
        )

        # Run transformer layers
        for layer_pair in self.layers:
            assert isinstance(layer_pair, nn.ModuleList)
            res = layer_pair[0](res)
            res = layer_pair[1](res)

        # Extract output embedding and compute logits
        output_emb = res[:, self.output_gather_indices]  # (seq_len, d_embed)
        logits = output_emb @ self.unembed_table.T  # (seq_len, vocab_size)
        return logits.to(orig_device)


class _CachedTransformerWrapper(nn.Module):
    """Export-friendly cached forward for CompiledTransformerModule.

    Signature matches the ONNX graph:
        forward(token_ids, past_len, past_K_0, past_V_0, ..., past_K_{N-1}, past_V_{N-1})
          -> (logits, new_K_0, new_V_0, ..., new_K_{N-1}, new_V_{N-1})

    Prefill is expressed by passing empty past tensors
    (shape (n_heads, 0, d_head)) and past_len=0.
    """

    def __init__(self, base: CompiledTransformerModule):
        super().__init__()
        self.base = base
        self.n_layers = len(base.layers)

    def forward(
        self,
        token_ids: torch.LongTensor,
        past_len: torch.LongTensor,
        *past_kvs: torch.Tensor,
    ):
        b = self.base
        n_new = token_ids.shape[0]

        embedded = b.token_embedding(token_ids)  # (n_new, d_embed)
        pos = b.pos_encoding[past_len : past_len + n_new]  # (n_new, d_pos)
        res = embedded @ b.embedding_proj + pos @ b.pos_proj + b.constant_values

        new_kvs: List[torch.Tensor] = []
        for i, layer_pair in enumerate(b.layers):
            assert isinstance(layer_pair, nn.ModuleList)
            attn_mod = layer_pair[0]
            ffn_mod = layer_pair[1]
            past_K_i = past_kvs[2 * i]
            past_V_i = past_kvs[2 * i + 1]
            res, new_K_i, new_V_i = attn_mod.forward_cached(res, past_K_i, past_V_i)
            res = ffn_mod(res)
            new_kvs.append(new_K_i)
            new_kvs.append(new_V_i)

        output_emb = res[:, b.output_gather_indices]
        logits = output_emb @ b.unembed_table.T
        return (logits, *new_kvs)


class HeadlessTransformerModule(nn.Module):
    """A compiled torchwright transformer with float I/O (no embedding heads).

    Input: ``inputs`` — (seq_len, d_input) FloatTensor of raw scalar values.
    Output: (seq_len, d_output) FloatTensor of raw scalar values.

    The input columns correspond to the graph's ``InputNode`` nodes,
    ordered alphabetically by node name.  The ``input_names`` attribute
    records this ordering.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        input_proj: torch.Tensor,
        pos_proj: torch.Tensor,
        constant_values: torch.Tensor,
        pos_encoding: torch.Tensor,
        output_gather_indices: torch.Tensor,
        input_names: List[str],
    ):
        super().__init__()
        self.layers = layers
        self.input_names = input_names
        self.register_buffer("input_proj", input_proj)
        self.register_buffer("pos_proj", pos_proj)
        self.register_buffer("constant_values", constant_values)
        self.register_buffer("pos_encoding", pos_encoding)
        self.register_buffer("output_gather_indices", output_gather_indices)

    def forward(self, inputs: torch.FloatTensor) -> torch.Tensor:
        # Do not export this directly — the .to(device) dance leaks a Cast
        # node into the traced graph. Export via _CachedHeadlessWrapper.
        orig_device = inputs.device
        inputs = inputs.to(self.input_proj.device)
        seq_len = inputs.shape[0]

        pos = self.pos_encoding[:seq_len]

        res = (
            inputs @ self.input_proj + pos @ self.pos_proj + self.constant_values
        )

        for layer_pair in self.layers:
            assert isinstance(layer_pair, nn.ModuleList)
            res = layer_pair[0](res)
            res = layer_pair[1](res)

        return res[:, self.output_gather_indices].to(orig_device)


class _CachedHeadlessWrapper(nn.Module):
    """Export-friendly cached forward for HeadlessTransformerModule.

    Signature matches the ONNX graph:
        forward(inputs, past_len, past_K_0, past_V_0, ..., past_K_{N-1}, past_V_{N-1})
          -> (output, new_K_0, new_V_0, ..., new_K_{N-1}, new_V_{N-1})

    Prefill is expressed by passing empty past tensors
    (shape (n_heads, 0, d_head)) and past_len=0.
    """

    def __init__(self, base: HeadlessTransformerModule):
        super().__init__()
        self.base = base
        self.n_layers = len(base.layers)

    def forward(
        self,
        inputs: torch.FloatTensor,
        past_len: torch.LongTensor,
        *past_kvs: torch.Tensor,
    ):
        b = self.base
        n_new = inputs.shape[0]

        pos = b.pos_encoding[past_len : past_len + n_new]
        res = inputs @ b.input_proj + pos @ b.pos_proj + b.constant_values

        new_kvs: List[torch.Tensor] = []
        for i, layer_pair in enumerate(b.layers):
            assert isinstance(layer_pair, nn.ModuleList)
            attn_mod = layer_pair[0]
            ffn_mod = layer_pair[1]
            past_K_i = past_kvs[2 * i]
            past_V_i = past_kvs[2 * i + 1]
            res, new_K_i, new_V_i = attn_mod.forward_cached(res, past_K_i, past_V_i)
            res = ffn_mod(res)
            new_kvs.append(new_K_i)
            new_kvs.append(new_V_i)

        output = res[:, b.output_gather_indices]
        return (output, *new_kvs)


def _compute_pos_encoding(d_pos: int, max_seq_len: int) -> torch.Tensor:
    """Precompute sinusoidal positional encoding buffer."""
    pe = torch.zeros(max_seq_len, d_pos)
    div_term = torch.exp(torch.arange(0, d_pos, 2) * -(math.log(10000.0) / d_pos))
    for pos in range(max_seq_len):
        pe[pos, 0::2] = torch.sin(pos * div_term)
        pe[pos, 1::2] = torch.cos(pos * div_term)
    return pe


def _convert_layers(
    compiled: HeadlessTransformer,
    max_seq_len: int,
) -> nn.ModuleList:
    """Convert HeadlessTransformer layers to nn.ModuleList of (attn, ffn) pairs."""
    d = compiled.d
    d_head = compiled.d_head
    n_heads = d // d_head
    causal_mask = torch.triu(
        torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
    )

    layer_modules = nn.ModuleList()
    for layer in compiled.layers:
        attn_comp = layer.attn.attn

        W_Q = attn_comp.query_matrix.permute(1, 0, 2).reshape(d, d)
        W_K = attn_comp.key_matrix.permute(1, 0, 2).reshape(d, d)
        W_V = attn_comp.value_matrix.permute(1, 0, 2).reshape(d, d)
        W_O = attn_comp.output_matrix.clone()

        attn_mod = _AttentionLayer(W_Q, W_K, W_V, W_O, n_heads, d_head, causal_mask)

        ffn_comp = layer.ffn
        W1 = ffn_comp.linear1.output_matrix.clone()
        b1 = ffn_comp.linear1.output_bias.clone()
        W2 = ffn_comp.linear2.output_matrix.clone()
        b2 = ffn_comp.linear2.output_bias.clone()

        ffn_mod = _FFNLayer(W1, b1, W2, b2)
        layer_modules.append(nn.ModuleList([attn_mod, ffn_mod]))

    return layer_modules


def to_module(
    compiled: HeadlessTransformer,
    embedding: Embedding,
    output_node: Node,
    max_seq_len: int = 512,
    device: Optional[str] = "auto",
) -> CompiledTransformerModule:
    """Convert a compiled HeadlessTransformer to an nn.Module.

    Args:
        compiled: The compiled transformer from forward_compile().
        embedding: The Embedding node used for tokenization.
        output_node: The graph node whose value is the model output.
        max_seq_len: Maximum sequence length (for precomputed pos encoding and causal mask).
        device: Target device — "auto" (default) uses GPU if available,
                "cpu"/"cuda" to force, or None to skip moving.

    Returns:
        A CompiledTransformerModule that accepts token_ids and returns logits.
    """
    assert compiled.feature_assignment is not None
    d = compiled.d
    d_head = compiled.d_head
    n_heads = d // d_head

    in_state = compiled.layers[0].attn.in_state
    out_state = compiled.layers[-1].ffn.out_state

    # --- Extract input scatter indices ---
    embedding_indices = None
    pos_indices = None
    constant_values = torch.zeros(d)

    for node in compiled.feature_assignment.get_nodes(in_state):
        indices = compiled.feature_assignment.get_node_indices(in_state, node)
        if isinstance(node, Embedding):
            embedding_indices = indices
        elif isinstance(node, PosEncoding):
            pos_indices = indices
        elif isinstance(node, LiteralValue):
            for i, idx in enumerate(indices):
                constant_values[idx] = node.value[i]
        elif isinstance(node, Concatenate):
            pass  # children handled individually

    assert (
        embedding_indices is not None
    ), "No Embedding node found in feature assignment"
    assert pos_indices is not None, "No PosEncoding node found in feature assignment"

    d_embed = len(embedding_indices)
    d_pos = len(pos_indices)

    # Build projection matrices: (d_embed, d) and (d_pos, d)
    embedding_proj = torch.zeros(d_embed, d)
    for i, idx in enumerate(embedding_indices):
        embedding_proj[i, idx] = 1.0

    pos_proj = torch.zeros(d_pos, d)
    for i, idx in enumerate(pos_indices):
        pos_proj[i, idx] = 1.0

    # --- Extract output gather indices ---
    output_indices = compiled.feature_assignment.get_node_indices(
        out_state, output_node
    )
    output_gather_indices = torch.tensor(output_indices, dtype=torch.long)

    # --- Precompute buffers ---
    pos_encoding_buf = _compute_pos_encoding(d_pos, max_seq_len)

    # --- Convert layers ---
    layer_modules = _convert_layers(compiled, max_seq_len)

    # --- Embedding ---
    token_emb = nn.Embedding(
        num_embeddings=embedding.table.shape[0],
        embedding_dim=embedding.table.shape[1],
    )
    token_emb.weight = nn.Parameter(embedding.table.clone())

    module = CompiledTransformerModule(
        layers=layer_modules,
        token_embedding=token_emb,
        embedding_proj=embedding_proj,
        pos_proj=pos_proj,
        constant_values=constant_values,
        pos_encoding=pos_encoding_buf,
        output_gather_indices=output_gather_indices,
        unembed_table=embedding.table.clone(),
        tokenizer=embedding.tokenizer,
    )

    if device == "auto":
        module.to(get_device(verbose=False))
    elif device is not None:
        module.to(torch.device(device))

    return module


def to_headless_module(
    compiled: HeadlessTransformer,
    output_node: Node,
    max_seq_len: int = 512,
    device: Optional[str] = "auto",
) -> HeadlessTransformerModule:
    """Convert a compiled HeadlessTransformer to a headless nn.Module.

    The resulting module accepts raw float inputs and returns raw float
    outputs — no embedding or unembedding.

    Args:
        compiled: The compiled transformer from forward_compile().
        output_node: The graph node whose value is the model output.
        max_seq_len: Maximum sequence length (for precomputed pos encoding
            and causal mask).
        device: Target device — "auto" (default) uses GPU if available,
                "cpu"/"cuda" to force, or None to skip moving.

    Returns:
        A HeadlessTransformerModule.  Its ``input_names`` attribute lists
        the InputNode names in the order they appear in the input tensor.
    """
    assert compiled.feature_assignment is not None
    d = compiled.d

    in_state = compiled.layers[0].attn.in_state
    out_state = compiled.layers[-1].ffn.out_state

    # --- Collect InputNode and PosEncoding indices ---
    input_nodes: List[tuple] = []  # (name, indices)
    pos_indices = None
    constant_values = torch.zeros(d)

    for node in compiled.feature_assignment.get_nodes(in_state):
        indices = compiled.feature_assignment.get_node_indices(in_state, node)
        if isinstance(node, InputNode):
            input_nodes.append((node.name, indices))
        elif isinstance(node, PosEncoding):
            pos_indices = indices
        elif isinstance(node, LiteralValue):
            for i, idx in enumerate(indices):
                constant_values[idx] = node.value[i]
        elif isinstance(node, (Concatenate, Embedding)):
            pass

    assert len(input_nodes) > 0, "No InputNode found in feature assignment"
    assert pos_indices is not None, "No PosEncoding node found in feature assignment"

    # Sort by name for deterministic input ordering
    input_nodes.sort(key=lambda x: x[0])
    input_names = [name for name, _ in input_nodes]

    # Build input scatter matrix: (d_input, d)
    all_input_indices = []
    for _, indices in input_nodes:
        all_input_indices.extend(indices)
    d_input = len(all_input_indices)

    input_proj = torch.zeros(d_input, d)
    for i, idx in enumerate(all_input_indices):
        input_proj[i, idx] = 1.0

    d_pos = len(pos_indices)
    pos_proj = torch.zeros(d_pos, d)
    for i, idx in enumerate(pos_indices):
        pos_proj[i, idx] = 1.0

    # --- Extract output gather indices ---
    output_indices = compiled.feature_assignment.get_node_indices(
        out_state, output_node
    )
    output_gather_indices = torch.tensor(output_indices, dtype=torch.long)

    # --- Precompute buffers ---
    pos_encoding_buf = _compute_pos_encoding(d_pos, max_seq_len)

    # --- Convert layers ---
    layer_modules = _convert_layers(compiled, max_seq_len)

    module = HeadlessTransformerModule(
        layers=layer_modules,
        input_proj=input_proj,
        pos_proj=pos_proj,
        constant_values=constant_values,
        pos_encoding=pos_encoding_buf,
        output_gather_indices=output_gather_indices,
        input_names=input_names,
    )

    if device == "auto":
        module.to(get_device(verbose=False))
    elif device is not None:
        module.to(torch.device(device))

    return module
