from typing import Any

# === Internal dependency: hypothesis ===
# re-export: from hypothesis._settings import settings
# re-export: from hypothesis.core import given

# === Internal dependency: hypothesis.errors ===
class InvalidArgument(_Trimmable, TypeError): ...

# === Internal dependency: hypothesis.extra.array_api ===
def make_strategies_namespace(xp: Any, *, api_version: Optional[NominalVersion] = ...) -> SimpleNamespace: ...
REAL_NAMES: Any
COMPLEX_NAMES: Any

# === Internal dependency: hypothesis.internal.floats ===
width_smallest_normals: Any

# === Internal dependency: hypothesis.strategies ===
# re-export: from hypothesis.strategies._internal.core import builds
# re-export: from hypothesis.strategies._internal.core import complex_numbers
# re-export: from hypothesis.strategies._internal.core import composite
# re-export: from hypothesis.strategies._internal.core import data
# re-export: from hypothesis.strategies._internal.core import lists
# re-export: from hypothesis.strategies._internal.core import sampled_from
# re-export: from hypothesis.strategies._internal.core import shared
# re-export: from hypothesis.strategies._internal.misc import just
# re-export: from hypothesis.strategies._internal.misc import nothing
# re-export: from hypothesis.strategies._internal.numbers import floats
# re-export: from hypothesis.strategies._internal.numbers import integers

# === Unresolved dependency: pytest ===
# Used unresolved symbols: mark, param, raises, skip, xfail

# === Unresolved dependency: tests.array_api.common ===
# Used unresolved symbols: MIN_VER_FOR_COMPLEX, dtype_name_params, flushes_to_zero

# === Unresolved dependency: tests.common.debug ===
# Used unresolved symbols: assert_all_examples, check_can_generate_examples, find_any, minimal

# === Unresolved dependency: tests.common.utils ===
# Used unresolved symbols: flaky