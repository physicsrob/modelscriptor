"""doom_sandbox.api — the agent-facing API surface.

This is the only `doom_sandbox` module phases should import from
(alongside `doom_sandbox.types` and `doom_sandbox.fixtures`).
"""

from .debug import (
    assert_,
    assert_bool,
    assert_close,
    assert_in_range,
    assert_integer,
    debug_watch,
    print_vec,
)
from .forward import (
    DEFAULT_MAX_VOCAB_CARDINALITY,
    Config,
    ForwardOutput,
    Pixel,
    RunOutput,
    TokenVocab,
    run,
)
from .past import Past
from .pwl import PWLDef, PWLDef2D, pwl_def
from .std import (
    clamp,
    compare_const,
    linear,
    multiply,
    one_hot,
    piecewise_linear,
    piecewise_linear_2d,
    reduce_sum,
    relu,
    sum,
    type_switch,
)
from .tokens import (
    FloatSlot,
    IntSlot,
    Slot,
    Token,
    TokenType,
    extract_float_slot,
    extract_int_slot,
    extract_type_slot,
    is_type,
    make_token,
)
from .utils import concat, split
from .vec import Vec, constant

__all__ = [
    # Core
    "Vec",
    "PWLDef",
    "PWLDef2D",
    "pwl_def",
    "constant",
    "concat",
    "split",
    # Tokens
    "TokenType",
    "Token",
    "IntSlot",
    "FloatSlot",
    "Slot",
    "make_token",
    "extract_int_slot",
    "extract_float_slot",
    "extract_type_slot",
    "is_type",
    # Past
    "Past",
    # Forward / lifecycle
    "ForwardOutput",
    "Pixel",
    "TokenVocab",
    "DEFAULT_MAX_VOCAB_CARDINALITY",
    "Config",
    "RunOutput",
    "run",
    # Stdlib
    "type_switch",
    "relu",
    "clamp",
    "compare_const",
    "piecewise_linear",
    "multiply",
    "piecewise_linear_2d",
    "linear",
    "sum",
    "reduce_sum",
    "one_hot",
    # Debug
    "print_vec",
    "debug_watch",
    "assert_in_range",
    "assert_close",
    "assert_bool",
    "assert_integer",
    "assert_",
]
