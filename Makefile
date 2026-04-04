# Auto-discover compilable examples (those defining create_network_parts)
COMPILABLE := $(shell grep -rl 'def create_network_parts' examples/*.py \
    | sed 's|examples/||; s|\.py||')
ONNX_FILES := $(addsuffix .onnx, $(COMPILABLE))

.PHONY: compile
compile: $(ONNX_FILES)

%.onnx: examples/%.py examples/compile.py torchwright/compiler/*.py torchwright/graph/*.py
	uv run python -m examples.compile $*

.PHONY: lint
lint:
	uv run black .
	uv run mypy .

LOCKFILE := /tmp/torchwright-test.lock
LOGFILE := /tmp/torchwright-test.log

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
