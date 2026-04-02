"""Compile a graph to ONNX + vocab files."""

import json
import os

import torch

from modelscriptor.compiler.forward.compile import forward_compile
from modelscriptor.compiler.module import CompiledTransformerModule, to_module
from modelscriptor.graph import Node
from modelscriptor.graph.embedding import Embedding
from modelscriptor.graph.pos_encoding import PosEncoding


def _export_onnx(
    module: CompiledTransformerModule,
    output_path: str,
    opset_version: int = 14,
    example_seq_len: int = 10,
) -> None:
    module.eval()
    dummy_input = torch.zeros(example_seq_len, dtype=torch.long)
    torch.onnx.export(
        module,
        (dummy_input,),
        output_path,
        opset_version=opset_version,
        input_names=["token_ids"],
        output_names=["logits"],
        dynamic_axes={
            "token_ids": {0: "seq_len"},
            "logits": {0: "seq_len"},
        },
        dynamo=False,
    )


def _vocab_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".vocab.json"


def compile_to_onnx(
    output_node: Node,
    pos_encoding: PosEncoding,
    embedding: Embedding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    verbose: bool = True,
) -> None:
    """Compile a computation graph to ONNX + vocab files.

    Produces two files:
        <output_path>          -- the ONNX model
        <output_path>.vocab.json -- the vocabulary (needed by the REPL)

    Args:
        output_node: The graph node whose value is the model output.
        pos_encoding: Positional encoding node.
        embedding: Embedding node (provides tokenizer + embedding table).
        output_path: Path for the .onnx output file.
        d: Residual stream dimension.
        d_head: Attention head dimension.
        max_seq_len: Maximum sequence length.
        verbose: Print progress.
    """
    if verbose:
        print("Compiling graph...")
    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        device=None,
    )

    if verbose:
        print("Converting to nn.Module...")
    module = to_module(net, embedding, output_node, max_seq_len=max_seq_len, device=None)
    module.eval()

    n_params = sum(p.numel() for p in module.parameters())
    n_layers = len(module.layers)

    if verbose:
        print(f"Exporting ONNX: {n_layers} layers, {n_params:,} parameters")
    _export_onnx(module, output_path)

    vocab_path = _vocab_path_for(output_path)
    with open(vocab_path, "w") as f:
        json.dump({"vocab": embedding.tokenizer.vocab}, f)

    if verbose:
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {vocab_path}")
