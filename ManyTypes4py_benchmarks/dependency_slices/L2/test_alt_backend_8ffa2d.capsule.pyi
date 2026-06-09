from typing import Any

# === Internal dependency: hypothesis ===
# re-export: from hypothesis._settings import settings
# re-export: from hypothesis.core import given

# === Internal dependency: hypothesis.database ===
class InMemoryExampleDatabase(ExampleDatabase):
    def __init__(self) -> Any: ...

# === Internal dependency: hypothesis.errors ===
class InvalidArgument(_Trimmable, TypeError): ...

# === Internal dependency: hypothesis.internal.compat ===
def int_to_bytes(i: int, size: int) -> bytes: ...

# === Internal dependency: hypothesis.internal.conjecture.data ===
class PrimitiveProvider(abc.ABC):
    def draw_boolean(self, p: float = ..., *, forced: Optional[bool] = ..., fake_forced: bool = ...) -> bool: ...
    def draw_integer(self, min_value: Optional[int] = ..., max_value: Optional[int] = ..., *, weights: Optional[Sequence[float]] = ..., shrink_towards: int = ..., forced: Optional[int] = ..., fake_forced: bool = ...) -> int: ...
    def draw_float(self, *, min_value: float = ..., max_value: float = ..., allow_nan: bool = ..., smallest_nonzero_magnitude: float, forced: Optional[float] = ..., fake_forced: bool = ...) -> float: ...
    def draw_string(self, intervals: IntervalSet, *, min_size: int = ..., max_size: Optional[int] = ..., forced: Optional[str] = ..., fake_forced: bool = ...) -> str: ...
    def draw_bytes(self, size: int, *, forced: Optional[bytes] = ..., fake_forced: bool = ...) -> bytes: ...
AVAILABLE_PROVIDERS: Any

# === Internal dependency: hypothesis.internal.conjecture.engine ===
class ConjectureRunner:
    def __init__(self, test_function: Callable[[ConjectureData], None], *, settings: Optional[Settings] = ..., random: Optional[Random] = ..., database_key: Optional[bytes] = ..., ignore_limits: bool = ...) -> None: ...
    def cached_test_function_ir(self, nodes: List[IRNode], *, error_on_discard: bool = ...) -> Union[ConjectureResult, _Overrun]: ...

# === Internal dependency: hypothesis.internal.floats ===
SIGNALING_NAN: Any

# === Internal dependency: hypothesis.strategies ===
# re-export: from hypothesis.strategies._internal.core import binary
# re-export: from hypothesis.strategies._internal.core import booleans
# re-export: from hypothesis.strategies._internal.core import text
# re-export: from hypothesis.strategies._internal.numbers import floats
# re-export: from hypothesis.strategies._internal.numbers import integers

# === Unresolved dependency: pytest ===
# Used unresolved symbols: mark, raises

# === Unresolved dependency: tests.common.debug ===
# Used unresolved symbols: minimal

# === Unresolved dependency: tests.conjecture.common ===
# Used unresolved symbols: ir_nodes