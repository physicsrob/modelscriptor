# Run sandbox tests (pure Python, no GPU, no Modal).
#
#   make test                          # full sandbox suite
#   make test FILE=api/test_std.py     # single file or directory
#   make test ARGS="-k foo -v"         # extra pytest args
#
# `FILE` is relative to this directory.

TARGET := $(if $(FILE),$(FILE),.)

.PHONY: test
test:
	uv run pytest $(TARGET) $(ARGS)
