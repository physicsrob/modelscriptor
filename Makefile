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

LOCKFILE := /tmp/modelscriptor-test.lock
LOGFILE := /tmp/modelscriptor-test.log

.PHONY: test
test:
	@echo "=== Waiting for test lock ($(LOCKFILE)) ===" | tee $(LOGFILE)
	@echo "=== Monitor: make test-logs ===" | tee -a $(LOGFILE)
	@flock $(LOCKFILE) bash -c ' \
		echo "=== Lock acquired, running tests ===" | tee -a $(LOGFILE) && \
		start=$$(date +%s) && \
		uv run pytest $(if $(FILE),$(FILE),tests) \
			-v --tb=short --no-header \
			--durations=0 \
			$(ARGS) 2>&1 | tee -a $(LOGFILE) ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) && \
		echo "" | tee -a $(LOGFILE) && \
		echo "=== Tests finished in $$((end - start))s (exit $$rc) ===" | tee -a $(LOGFILE) ; \
		exit $$rc \
	'

.PHONY: test-logs
test-logs:
	@tail -f $(LOGFILE)
