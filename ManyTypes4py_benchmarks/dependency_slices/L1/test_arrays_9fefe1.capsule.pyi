# === Internal dependency: hypothesis ===
from hypothesis._settings import settings
from hypothesis.core import given

# === Internal dependency: hypothesis.errors ===
class InvalidArgument(_Trimmable, TypeError): ...

# === Internal dependency: hypothesis.extra.array_api ===
def make_strategies_namespace(xp, *, api_version=...): ...
INT_NAMES = ('int8', 'int16', 'int32', 'int64')
UINT_NAMES = ('uint8', 'uint16', 'uint32', 'uint64')
ALL_INT_NAMES = INT_NAMES + UINT_NAMES
FLOAT_NAMES = ('float32', 'float64')
REAL_NAMES = ALL_INT_NAMES + FLOAT_NAMES
COMPLEX_NAMES = ('complex64', 'complex128')

# === Internal dependency: hypothesis.internal.floats ===
width_smallest_normals = {16: 2 ** (-(2 ** (5 - 1) - 2)), 32: 2 ** (-(2 ** (8 - 1) - 2)), 64: 2 ** (-(2 ** (11 - 1) - 2))}

# === Internal dependency: hypothesis.strategies ===
from hypothesis.strategies._internal.core import builds
from hypothesis.strategies._internal.core import complex_numbers
from hypothesis.strategies._internal.core import composite
from hypothesis.strategies._internal.core import data
from hypothesis.strategies._internal.core import lists
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.core import shared
from hypothesis.strategies._internal.misc import just
from hypothesis.strategies._internal.misc import nothing
from hypothesis.strategies._internal.numbers import floats
from hypothesis.strategies._internal.numbers import integers

# === Unresolved dependency: pytest ===
# Used unresolved symbols: mark, param, raises, skip, xfail

# === Unresolved dependency: tests.array_api.common ===
# Used unresolved symbols: MIN_VER_FOR_COMPLEX, dtype_name_params, flushes_to_zero

# === Unresolved dependency: tests.common.debug ===
# Used unresolved symbols: assert_all_examples, check_can_generate_examples, find_any, minimal

# === Unresolved dependency: tests.common.utils ===
# Used unresolved symbols: flaky