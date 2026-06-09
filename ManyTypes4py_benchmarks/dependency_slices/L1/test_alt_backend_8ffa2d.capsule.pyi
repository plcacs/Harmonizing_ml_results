# === Internal dependency: hypothesis ===
from hypothesis._settings import settings
from hypothesis.core import given

# === Internal dependency: hypothesis.database ===
class InMemoryExampleDatabase(ExampleDatabase):
    def __init__(self): ...

# === Internal dependency: hypothesis.errors ===
class InvalidArgument(_Trimmable, TypeError): ...

# === Internal dependency: hypothesis.internal.compat ===
def int_to_bytes(i, size): ...

# === Internal dependency: hypothesis.internal.conjecture.data ===
class PrimitiveProvider(abc.ABC):
    def draw_boolean(self, p=..., *, forced=..., fake_forced=...): ...
    def draw_integer(self, min_value=..., max_value=..., *, weights=..., shrink_towards=..., forced=..., fake_forced=...): ...
    def draw_float(self, *, min_value=..., max_value=..., allow_nan=..., smallest_nonzero_magnitude, forced=..., fake_forced=...): ...
    def draw_string(self, intervals, *, min_size=..., max_size=..., forced=..., fake_forced=...): ...
    def draw_bytes(self, size, *, forced=..., fake_forced=...): ...
AVAILABLE_PROVIDERS = {'hypothesis': 'hypothesis.internal.conjecture.data.HypothesisProvider'}

# === Internal dependency: hypothesis.internal.conjecture.engine ===
class ConjectureRunner:
    def __init__(self, test_function, *, settings=..., random=..., database_key=..., ignore_limits=...): ...
    def cached_test_function_ir(self, nodes, *, error_on_discard=...): ...

# === Internal dependency: hypothesis.internal.floats ===
def int_to_float(value, width=...): ...
SIGNALING_NAN = int_to_float(...)

# === Internal dependency: hypothesis.strategies ===
from hypothesis.strategies._internal.core import binary
from hypothesis.strategies._internal.core import booleans
from hypothesis.strategies._internal.core import text
from hypothesis.strategies._internal.numbers import floats
from hypothesis.strategies._internal.numbers import integers

# === Unresolved dependency: pytest ===
# Used unresolved symbols: mark, raises

# === Unresolved dependency: tests.common.debug ===
# Used unresolved symbols: minimal

# === Unresolved dependency: tests.conjecture.common ===
# Used unresolved symbols: ir_nodes