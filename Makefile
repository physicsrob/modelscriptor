.PHONY: compile
compile: adder.onnx

adder.onnx: compile_adder.py examples/adder.py modelscriptor/compiler/*.py modelscriptor/graph/*.py
	python compile_adder.py

.PHONY: lint
lint:
	black .
	mypy .

.PHONY: test
test:
	pytest tests
