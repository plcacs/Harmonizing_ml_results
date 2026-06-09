from typing import Any

# === Internal dependency: hypothesis ===
# re-export: from hypothesis._settings import HealthCheck
# re-export: from hypothesis._settings import settings
# re-export: from hypothesis.core import given

# === Internal dependency: hypothesis.errors ===
class Unsatisfiable(_Trimmable): ...
class HypothesisWarning(HypothesisException, Warning): ...

# === Internal dependency: hypothesis.internal.filtering ===
def max_len(size: int, element: Collection) -> bool: ...
def min_len(size: int, element: Collection) -> bool: ...

# === Internal dependency: hypothesis.internal.floats ===
def next_up(value, width = ...) -> Any: ...
def next_down(value, width = ...) -> Any: ...

# === Internal dependency: hypothesis.internal.reflection ===
def get_pretty_function_description(f) -> Any: ...

# === Internal dependency: hypothesis.strategies ===
# re-export: from hypothesis.strategies._internal.core import binary
# re-export: from hypothesis.strategies._internal.core import data
# re-export: from hypothesis.strategies._internal.core import dictionaries
# re-export: from hypothesis.strategies._internal.core import frozensets
# re-export: from hypothesis.strategies._internal.core import lists
# re-export: from hypothesis.strategies._internal.core import permutations
# re-export: from hypothesis.strategies._internal.core import sampled_from
# re-export: from hypothesis.strategies._internal.core import sets
# re-export: from hypothesis.strategies._internal.core import text
# re-export: from hypothesis.strategies._internal.misc import none
# re-export: from hypothesis.strategies._internal.numbers import floats
# re-export: from hypothesis.strategies._internal.numbers import integers

# === Internal dependency: hypothesis.strategies._internal.core ===
def data() -> SearchStrategy[DataObject]: ...

# === Internal dependency: hypothesis.strategies._internal.lazy ===
def unwrap_strategies(s) -> Any: ...
class LazyStrategy(SearchStrategy): ...

# === Internal dependency: hypothesis.strategies._internal.numbers ===
class IntegersStrategy(SearchStrategy): ...
class FloatStrategy(SearchStrategy): ...

# === Internal dependency: hypothesis.strategies._internal.strategies ===
class MappedStrategy(SearchStrategy[Ex]): ...
class FilteredStrategy(SearchStrategy[Ex]): ...

# === Internal dependency: hypothesis.strategies._internal.strings ===
class TextStrategy(ListStrategy): ...

# === Unresolved dependency: pytest ===
# Used unresolved symbols: mark, raises, warns

# === Unresolved dependency: tests.common.debug ===
# Used unresolved symbols: check_can_generate_examples

# === Unresolved dependency: tests.common.utils ===
# Used unresolved symbols: fails_with