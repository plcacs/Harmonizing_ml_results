# === Internal dependency: hypothesis ===
from hypothesis._settings import HealthCheck
from hypothesis._settings import settings
from hypothesis.core import given

# === Internal dependency: hypothesis.errors ===
class Unsatisfiable(_Trimmable): ...
class HypothesisWarning(HypothesisException, Warning): ...

# === Internal dependency: hypothesis.internal.filtering ===
def max_len(size, element): ...
def min_len(size, element): ...

# === Internal dependency: hypothesis.internal.floats ===
def next_up(value, width=...): ...
def next_down(value, width=...): ...

# === Internal dependency: hypothesis.internal.reflection ===
def get_pretty_function_description(f): ...

# === Internal dependency: hypothesis.strategies ===
from hypothesis.strategies._internal.core import binary
from hypothesis.strategies._internal.core import data
from hypothesis.strategies._internal.core import dictionaries
from hypothesis.strategies._internal.core import frozensets
from hypothesis.strategies._internal.core import lists
from hypothesis.strategies._internal.core import permutations
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.core import sets
from hypothesis.strategies._internal.core import text
from hypothesis.strategies._internal.misc import none
from hypothesis.strategies._internal.numbers import floats
from hypothesis.strategies._internal.numbers import integers

# === Internal dependency: hypothesis.strategies._internal.core ===
def data(): ...

# === Internal dependency: hypothesis.strategies._internal.lazy ===
def unwrap_strategies(s): ...
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