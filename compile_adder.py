"""Compile the 3-digit adder graph to ONNX."""

from examples.adder import create_network_parts
from torchwright.compiler.export import compile_to_onnx

output_node, pos_encoding, embedding = create_network_parts()
compile_to_onnx(output_node, pos_encoding, embedding, "adder.onnx")
