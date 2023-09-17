.PHONY: lint
lint:
	black .
	mypy .

.PHONY: test
test:
	pytest tests
