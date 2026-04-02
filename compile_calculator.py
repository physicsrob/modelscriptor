"""Compile the calculator graph (supports +, -, *) to ONNX."""

from examples.calculator import create_network_parts
from modelscriptor.compiler.export import compile_to_onnx

output_node, pos_encoding, embedding = create_network_parts()
compile_to_onnx(output_node, pos_encoding, embedding, "calculator.onnx", d=1536)
