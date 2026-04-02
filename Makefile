.PHONY: compile
compile: adder.onnx

adder.onnx: compile_adder.py examples/adder.py modelscriptor/compiler/*.py modelscriptor/graph/*.py
	uv run python compile_adder.py

.PHONY: lint
lint:
	uv run black .
	uv run mypy .

.PHONY: test
test:
	uv run pytest tests
