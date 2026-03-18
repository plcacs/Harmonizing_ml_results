```python
import sys
from typing import Any, Callable, Optional, Tuple, Union, overload
import numpy as np
import hypothesis.strategies as st
from hypothesis.extra import numpy as nps
from hypothesis.errors import InvalidArgument, UnsatisfiedAssumption
from tests.common.debug import check_can_generate_examples, find_any, minimal
from tests.common.utils import fails_with, flaky
import pytest

ANY_SHAPE: Any = ...
ANY_NONZERO_SHAPE: Any = ...

@given(nps.arrays(float, ()))
def test_empty_dimensions_are_arrays(x: np.ndarray) -> None: ...

@given(nps.arrays(float, (1, 0, 1)))
def test_can_handle_zero_dimensions(x: np.ndarray) -> None: ...

@given(nps.arrays('uint32', (5, 5)))
def test_generates_unsigned_ints(x: np.ndarray) -> None: ...

@given(nps.arrays(int, (1,)))
def test_assert_fits_in_machine_size(x: np.ndarray) -> None: ...

def test_generates_and_minimizes() -> None: ...

def test_can_minimize_large_arrays() -> None: ...

@flaky(max_runs=50, min_passes=1)
def test_can_minimize_float_arrays() -> None: ...

class Foo:
    pass

foos: Any = ...

def test_can_create_arrays_of_composite_types() -> None: ...

@given(st.lists(st.integers()), st.data())
def test_can_create_zero_dim_arrays_of_lists(x: list[int], data: st.DataObject) -> None: ...

def test_can_create_arrays_of_tuples() -> None: ...

@given(nps.arrays(object, (2, 2), elements=st.tuples(st.integers())))
def test_does_not_flatten_arrays_of_tuples(arr: np.ndarray) -> None: ...

@given(nps.arrays(object, (2, 2), elements=st.lists(st.integers(), min_size=1, max_size=1)))
def test_does_not_flatten_arrays_of_lists(arr: np.ndarray) -> None: ...

@given(nps.array_shapes())
def test_can_generate_array_shapes(shape: tuple[int, ...]) -> None: ...

@settings(deadline=None, max_examples=10)
@given(st.integers(0, 10), st.integers(0, 9), st.integers(0), st.integers(0))
def test_minimise_array_shapes(min_dims: int, dim_range: int, min_side: int, side_range: int) -> None: ...

@pytest.mark.parametrize('kwargs', [{'min_side': 100}, {'min_dims': 15}, {'min_dims': 32}])
def test_interesting_array_shapes_argument(kwargs: dict[str, Any]) -> None: ...

@given(nps.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype: np.dtype) -> None: ...

@settings(max_examples=100)
@given(nps.nested_dtypes(subtype_strategy=st.one_of(nps.scalar_dtypes(), nps.byte_string_dtypes(), nps.unicode_string_dtypes())))
def test_can_generate_compound_dtypes(dtype: np.dtype) -> None: ...

@settings(max_examples=100)
@given(nps.nested_dtypes(subtype_strategy=st.one_of(nps.scalar_dtypes(), nps.byte_string_dtypes(), nps.unicode_string_dtypes())).flatmap(lambda dt: nps.arrays(dtype=dt, shape=1)))
def test_can_generate_data_compound_dtypes(arr: np.ndarray) -> None: ...

@given(nps.nested_dtypes())
def test_np_dtype_is_idempotent(dtype: np.dtype) -> None: ...

def test_minimise_scalar_dtypes() -> None: ...

def test_minimise_nested_types() -> None: ...

def test_minimise_array_strategy() -> None: ...

@given(nps.array_dtypes(allow_subarrays=False))
def test_can_turn_off_subarrays(dt: np.dtype) -> None: ...

def test_array_dtypes_may_have_field_titles() -> None: ...

@pytest.mark.parametrize('byteorder', ['<', '>'])
@given(data=st.data())
def test_can_restrict_endianness(data: st.DataObject, byteorder: str) -> None: ...

@given(nps.integer_dtypes(sizes=8))
def test_can_specify_size_as_an_int(dt: np.dtype) -> None: ...

@given(st.data())
def test_can_draw_arrays_from_scalars(data: st.DataObject) -> None: ...

@given(st.data())
def test_can_cast_for_arrays(data: st.DataObject) -> None: ...

@given(nps.arrays(dtype='int8', shape=st.integers(0, 20), unique=True))
def test_array_values_are_unique(arr: np.ndarray) -> None: ...

def test_cannot_generate_unique_array_of_too_many_elements() -> None: ...

@given(nps.arrays(elements=st.just(0.0), dtype=float, fill=st.just(np.nan), shape=st.integers(0, 20), unique=True))
def test_array_values_are_unique_high_collision(arr: np.ndarray) -> None: ...

@given(nps.arrays(dtype='int8', shape=(4,), elements=st.integers(0, 3), unique=True))
def test_generates_all_values_for_unique_array(arr: np.ndarray) -> None: ...

@given(nps.arrays(dtype='int8', shape=255, unique=True))
def test_efficiently_generates_all_unique_array(arr: np.ndarray) -> None: ...

@given(st.data(), st.integers(-100, 100), st.integers(1, 100))
def test_array_element_rewriting(data: st.DataObject, start: int, size: int) -> None: ...

def test_may_fill_with_nan_when_unique_is_set() -> None: ...

@given(nps.arrays(dtype=float, elements=st.floats(allow_nan=False), shape=10, unique=True, fill=st.just(np.nan)))
def test_is_still_unique_with_nan_fill(xs: np.ndarray) -> None: ...

@fails_with(InvalidArgument)
@given(nps.arrays(dtype=float, elements=st.floats(allow_nan=False), shape=10, unique=True, fill=st.just(0.0)))
def test_may_not_fill_with_non_nan_when_unique_is_set(arr: np.ndarray) -> None: ...

@fails_with(InvalidArgument)
@given(nps.arrays(dtype='U', shape=10, unique=True, fill=st.just('')))
def test_may_not_fill_with_non_nan_when_unique_is_set_and_type_is_not_number(arr: np.ndarray) -> None: ...

np_version: tuple[int, ...] = ...

@pytest.mark.parametrize('fill', [False, True])
@fails_with(InvalidArgument if np_version < (1, 24) else DeprecationWarning if np_version < (2, 0) else OverflowError)
@given(st.data())
def test_overflowing_integers_are_deprecated(fill: bool, data: st.DataObject) -> None: ...

@pytest.mark.parametrize('fill', [False, True])
@pytest.mark.parametrize('dtype,strat', [('float16', st.floats(min_value=65520, allow_infinity=False)), ('float32', st.floats(min_value=10 ** 40, allow_infinity=False)), ('complex64', st.complex_numbers(min_magnitude=10 ** 300, allow_infinity=False)), ('U1', st.text(min_size=2, max_size=2)), ('S1', st.binary(min_size=2, max_size=2))])
@fails_with(InvalidArgument)
@given(data=st.data())
def test_unrepresentable_elements_are_deprecated(fill: bool, dtype: str, strat: st.SearchStrategy[Any], data: st.DataObject) -> None: ...

@given(nps.arrays(dtype='float16', shape=(1,)))
def test_inferred_floats_do_not_overflow(arr: np.ndarray) -> None: ...

@given(nps.arrays(dtype='float16', shape=10, elements={'min_value': 0, 'max_value': 1}))
def test_inferred_floats_can_be_constrained_at_low_width(arr: np.ndarray) -> None: ...

@given(nps.arrays(dtype='float16', shape=10, elements={'min_value': 0, 'max_value': 1, 'exclude_min': True, 'exclude_max': True}))
def test_inferred_floats_can_be_constrained_at_low_width_excluding_endpoints(arr: np.ndarray) -> None: ...

@given(nps.arrays(dtype='float16', shape=10, unique=True, elements=st.integers(1, 9), fill=st.just(np.nan)))
def test_unique_array_with_fill_can_use_all_elements(arr: np.ndarray) -> None: ...

@given(nps.arrays(dtype='uint8', shape=25, unique=True, fill=st.nothing()))
def test_unique_array_without_fill(arr: np.ndarray) -> None: ...

@given(ndim=st.integers(0, 5), data=st.data())
def test_mapped_positive_axes_are_unique(ndim: int, data: st.DataObject) -> None: ...

@given(ndim=st.integers(0, 5), data=st.data())
def test_length_bounds_are_satisfied(ndim: int, data: st.DataObject) -> None: ...

@given(shape=nps.array_shapes(), data=st.data())
def test_axes_are_valid_inputs_to_sum(shape: tuple[int, ...], data: st.DataObject) -> None: ...

@settings(deadline=None, max_examples=10)
@given(ndim=st.integers(0, 3), data=st.data())
def test_minimize_tuple_axes(ndim: int, data: st.DataObject) -> None: ...

@settings(deadline=None, max_examples=10)
@given(ndim=st.integers(0, 3), data=st.data())
def test_minimize_negative_tuple_axes(ndim: int, data: st.DataObject) -> None: ...

@given(nps.broadcastable_shapes((), min_side=0, max_side=0, min_dims=0, max_dims=0))
def test_broadcastable_empty_shape(shape: tuple[int, ...]) -> None: ...

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(shape=ANY_SHAPE, data=st.data())
def test_broadcastable_shape_bounds_are_satisfied(shape: tuple[int, ...], data: st.DataObject) -> None: ...

@settings(deadline=None)
@given(num_shapes=st.integers(1, 4), base_shape=ANY_SHAPE, data=st.data())
def test_mutually_broadcastable_shape_bounds_are_satisfied(num_shapes: int, base_shape: tuple[int, ...], data: st.DataObject) -> None: ...

def _draw_valid_bounds(data: st.DataObject, shape: tuple[int, ...], max_dims: int, *, permit_none: bool = True) -> tuple[int, Optional[int]]: ...

def _broadcast_two_shapes(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]: ...

def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]: ...

@settings(deadline=None, max_examples=500)
@given(shapes=st.lists(nps.array_shapes(min_dims=0, min_side=0, max_dims=4, max_side=4), min_size=1))
def test_broadcastable_shape_util(shapes: list[tuple[int, ...]]) -> None: ...

@settings(deadline=None, max_examples=200)
@given(shape=ANY_NONZERO_SHAPE, data=st.data())
def test_broadcastable_shape_has_good_default_values(shape: tuple[int, ...], data: st.DataObject) -> None: ...

@settings(deadline=None, max_examples=200)
@given(base_shape=ANY_SHAPE, num_shapes=st.integers(1, 10), data=st.data())
def test_mutually_broadcastableshapes_has_good_default_values(num_shapes: int, base_shape: tuple[int, ...], data: st.DataObject) -> None: ...

@settings(deadline=None)
@given(min_dims=st.integers(0, 32), shape=ANY_SHAPE, data=st.data())
def test_broadcastable_shape_can_broadcast(min_dims: int, shape: tuple[int, ...], data: st.DataObject) -> None: ...

@settings(deadline=None)
@given(num_shapes=st.integers(1, 10), min_dims=st.integers(0, 32), base_shape=ANY_SHAPE, data=st.data())
def test_mutually_broadcastable_shape_can_broadcast(num_shapes: int, min_dims: int, base_shape: tuple[int, ...], data: st.DataObject) -> None: ...

@settings(deadline=None, max_examples=50)
@given(num_shapes=st.integers(1, 3), min_dims=st.integers(0, 5), base_shape=nps.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5), data=st.data())
def test_minimize_mutually_broadcastable_shape(num_shapes: int, min_dims: int, base_shape: tuple[int, ...], data: st.DataObject) -> None: ...

@settings(deadline=None)
@given(max_dims=st.integers(4, 6), data=st.data())
def test_broadcastable_shape_adjusts_max_dim_with_explicit_bounds(max_dims: int, data: st.DataObject) -> None: ...

@settings(deadline=None)
@given(max_side=st.sampled_from([3, None]), min_dims=st.integers(0, 4), num_shapes=st.integers(1, 3), data=st.data())
def test_mutually_broadcastable_shape_adjusts_max_dim_with_default_bounds(max_side: Optional[int], min_dims: int, num_shapes: int, data: st.DataObject) -> None: ...

@settings(deadline=None, max_examples=10)
@given(min_dims=st.integers(0, 32), min_side=st.integers(2, 3), data=st.data())
def test_broadcastable_shape_shrinking_with_singleton_out_of_bounds(min_dims: int, min_side: int, data: st.DataObject) -> None: ...

@settings(deadline=None, max_examples=50)
@given(num_shapes=st.integers(1, 4), min_dims=st.integers(0, 4), min_side=st.integers(2, 3), data=st.data())
def test_mutually_broadcastable_shapes_shrinking_with_singleton_out_of_bounds(num_shapes: int, min_dims: int, min_side: int, data: st.DataObject) -> None: ...

@settings(suppress_health_check=[HealthCheck.too_slow])
@given(num_shapes=st.integers(1, 4), min_dims=st.integers(1, 32), max_side=st.integers(1, 6), data=st.data())
def test_mutually_broadcastable_shapes_only_singleton_is_valid(num_shapes: int, min_dims: int, max_side: int, data: st.DataObject) -> None: ...

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(shape=nps.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5), max_dims=st.integers(0, 6), data=st.data())
def test_broadcastable_shape_can_generate_arbitrary_ndims(shape: tuple[int, ...], max_dims: int, data: st.DataObject) -> None: ...

@settings(deadline=None)
@given(num_shapes=st.integers(1, 3), base_shape=nps.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5), max_dims=st.integers(0, 4), data=st.data())
def test_mutually_broadcastable_shapes_can_generate_arbitrary_ndims(num_shapes: int, base_shape: tuple[int, ...], max_dims: int, data: st.DataObject) -> None: ...

@settings(deadline=None, suppress_health_check=list(HealthCheck))
@given(base_shape=nps.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=2), max_dims=st.integers(1, 4))
def test_mutually_broadcastable_shapes_can_generate_interesting_singletons(base_shape: tuple[int, ...], max_dims: int) -> None: ...

@pytest.mark.parametrize('base_shape', [(), (0,), (1,), (2,), (1, 2), (2, 1), (2, 2)])
def test_mutually_broadcastable_shapes_can_generate_mirrored_singletons(base_shape: tuple[int, ...]) -> None: ...

@settings