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

LOGFILE := /tmp/torchwright-test.log

.PHONY: test
test:
	@echo "=== Running tests on Modal ===" | tee $(LOGFILE)
	@echo "=== Monitor: make test-logs ===" | tee -a $(LOGFILE)
	@bash -c ' \
		start=$$(date +%s) && \
		uv run modal run modal_test.py \
			--file $(if $(FILE),$(FILE),tests) \
			$(if $(ARGS),--args "$(ARGS)") \
			2>&1 | tee -a $(LOGFILE) ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) && \
		echo "" | tee -a $(LOGFILE) && \
		echo "=== Tests finished in $$((end - start))s (exit $$rc) ===" | tee -a $(LOGFILE) ; \
		exit $$rc \
	'

.PHONY: test-logs
test-logs:
	@tail -f $(LOGFILE)

.PHONY: graph-stats
graph-stats:
	uv run python graph_stats.py $(ARGS)

.PHONY: walkthrough
walkthrough:
	uv run modal run modal_walkthrough.py $(ARGS)
	xdg-open walkthrough.gif
	xdg-open reference.gif
