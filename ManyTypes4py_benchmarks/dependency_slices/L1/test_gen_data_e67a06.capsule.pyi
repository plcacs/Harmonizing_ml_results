from typing import Any

# === Internal dependency: hypothesis ===
from hypothesis._settings import HealthCheck
from hypothesis._settings import settings
from hypothesis.control import assume
from hypothesis.control import note
from hypothesis.control import target
from hypothesis.core import given

# === Internal dependency: hypothesis.errors ===
class UnsatisfiedAssumption(HypothesisException): ...
class InvalidArgument(_Trimmable, TypeError): ...

# === Internal dependency: hypothesis.extra.numpy ===
def from_dtype(dtype, *, alphabet=..., min_size=..., max_size=..., min_value=..., max_value=..., allow_nan=..., allow_infinity=..., allow_subnormal=..., exclude_min=..., exclude_max=..., min_magnitude=..., max_magnitude=...): ...
class ArrayStrategy(st.SearchStrategy): ...
def arrays(dtype, shape, *, elements=..., fill=..., unique=...): ...
def scalar_dtypes(): ...
def unsigned_integer_dtypes(*, endianness=..., sizes): ...
def unsigned_integer_dtypes(*, endianness=..., sizes=...): ...
def integer_dtypes(*, endianness=..., sizes): ...
def integer_dtypes(*, endianness=..., sizes=...): ...
def floating_dtypes(*, endianness=..., sizes): ...
def floating_dtypes(*, endianness=..., sizes=...): ...
def byte_string_dtypes(*, endianness=..., min_len=..., max_len=...): ...
def unicode_string_dtypes(*, endianness=..., min_len=..., max_len=...): ...
def array_dtypes(subtype_strategy=..., *, min_size=..., max_size=..., allow_subarrays=...): ...
def nested_dtypes(subtype_strategy=..., *, max_leaves=..., max_itemsize=...): ...
def valid_tuple_axes(*args, **kwargs): ...
def mutually_broadcastable_shapes(*args, **kwargs): ...
def basic_indices(shape, *, min_dims=..., max_dims=..., allow_newaxis=..., allow_ellipsis=...): ...
def integer_array_indices(shape, *, result_shape=...): ...
def integer_array_indices(shape, *, result_shape=..., dtype): ...
def integer_array_indices(shape, *, result_shape=..., dtype=...): ...
from hypothesis.extra._array_helpers import Shape
from hypothesis.extra._array_helpers import array_shapes
from hypothesis.extra._array_helpers import broadcastable_shapes

# === Internal dependency: hypothesis.strategies ===
from hypothesis.strategies._internal.collections import tuples
from hypothesis.strategies._internal.core import binary
from hypothesis.strategies._internal.core import booleans
from hypothesis.strategies._internal.core import builds
from hypothesis.strategies._internal.core import complex_numbers
from hypothesis.strategies._internal.core import data
from hypothesis.strategies._internal.core import lists
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.core import text
from hypothesis.strategies._internal.misc import just
from hypothesis.strategies._internal.misc import none
from hypothesis.strategies._internal.misc import nothing
from hypothesis.strategies._internal.numbers import floats
from hypothesis.strategies._internal.numbers import integers
from hypothesis.strategies._internal.strategies import one_of

# === Internal dependency: hypothesis.strategies._internal.lazy ===
def unwrap_strategies(s): ...

# === Internal dependency: numpy ===
__version__: Any
all: Any
any: Any
arange: Any
broadcast_arrays: Any
count_nonzero: Any
dtype: Any
errstate: Any
isinf: Any
isnan: Any
isscalar: Any
logical_or: Any
nan: Any
nansum: Any
ndarray: Any
newaxis: Any
prod: Any
shares_memory: Any
sum: Any
testing: Any
uint32: Any
uint8: Any
zeros: Any

# === Unresolved dependency: pytest ===
# Used unresolved symbols: mark, raises

# === Unresolved dependency: tests.common.debug ===
# Used unresolved symbols: check_can_generate_examples, find_any, minimal

# === Unresolved dependency: tests.common.utils ===
# Used unresolved symbols: fails_with, flaky