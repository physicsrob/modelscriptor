.PHONY: compile
compile: adder.onnx calculator.onnx

adder.onnx: compile_adder.py examples/adder.py modelscriptor/compiler/*.py modelscriptor/graph/*.py
	uv run python compile_adder.py

calculator.onnx: compile_calculator.py examples/calculator.py examples/adder.py modelscriptor/compiler/*.py modelscriptor/graph/*.py
	uv run python compile_calculator.py

.PHONY: lint
lint:
	uv run black .
	uv run mypy .

.PHONY: test
test:
	uv run pytest tests
