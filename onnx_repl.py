"""REPL for an ONNX-compiled modelscriptor model.

Usage:
    python compile_interact.py model.onnx
"""

import argparse
from modelscriptor.compiler.repl import run_repl


def main():
    parser = argparse.ArgumentParser(
        description="Interactive REPL for a compiled ONNX model"
    )
    parser.add_argument("onnx_path", help="Path to the .onnx model file")
    parser.add_argument(
        "--max-tokens", type=int, default=20, help="Max tokens to generate"
    )
    args = parser.parse_args()
    run_repl(args.onnx_path, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
