import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.extra.array_api import COMPLEX_NAMES, REAL_NAMES
from hypothesis.internal.floats import width_smallest_normals
from tests.array_api.common import MIN_VER_FOR_COMPLEX, dtype_name_params, flushes_to_zero
from tests.common.debug import assert_all_examples, check_can_generate_examples, find_any, minimal
from tests.common.utils import flaky


def skip_on_missing_unique_values(xp: Any) -> None:
    if not hasattr(xp, 'unique_values'):
        pytest.mark.skip('xp.unique_values() is not required to exist')


def xfail_on_indistinct_nans(xp: Any) -> None:
    """
    xp.unique_value() should return distinct NaNs - if not, tests that (rightly)
    assume such behaviour will likely fail. For example, NumPy 1.22 treats NaNs
    as indistinct, so tests that use this function will be marked as xfail.
    See https://mail.python.org/pipermail/numpy-discussion/2021-August/081995.html
    """
    skip_on_missing_unique_values(xp)
    two_nans = xp.asarray([float('nan'), float('nan')])
    if xp.unique_values(two_nans).size != 2:
        pytest.xfail('NaNs not distinct')


@pytest.mark.parametrize('dtype_name', dtype_name_params)
def test_draw_arrays_from_dtype(xp: Any, xps: Any, dtype_name: str) -> None:
    """Draw arrays from dtypes."""
    dtype = getattr(xp, dtype_name)
    assert_all_examples(xps.arrays(dtype, ()), lambda x: x.dtype == dtype)


@pytest.mark.parametrize('dtype_name', dtype_name_params)
def test_draw_arrays_from_scalar_names(xp: Any, xps: Any, dtype_name: str) -> None:
    """Draw arrays from dtype names."""
    dtype = getattr(xp, dtype_name)
    assert_all_examples(xps.arrays(dtype_name, ()), lambda x: x.dtype == dtype)


@given(data=st.data())
def test_draw_arrays_from_shapes(xp: Any, xps: Any, data: st.DataObject) -> None:
    """Draw arrays from shapes."""
    shape = data.draw(xps.array_shapes())
    x = data.draw(xps.arrays(xp.int8, shape))
    assert x.ndim == len(shape)
    assert x.shape == shape


@given(data=st.data())
def test_draw_arrays_from_int_shapes(xp: Any, xps: Any, data: st.DataObject) -> None:
    """Draw arrays from integers as shapes."""
    size = data.draw(st.integers(0, 10))
    x = data.draw(xps.arrays(xp.int8, size))
    assert x.shape == (size,)


@pytest.mark.parametrize('strat_name', ['scalar_dtypes', 'boolean_dtypes', 'integer_dtypes', 'unsigned_integer_dtypes', 'floating_dtypes', 'real_dtypes', pytest.param('complex_dtypes', marks=pytest.mark.xp_min_version(MIN_VER_FOR_COMPLEX))])
def test_draw_arrays_from_dtype_strategies(xp: Any, xps: Any, strat_name: str) -> None:
    """Draw arrays from dtype strategies."""
    strat_func = getattr(xps, strat_name)
    strat = strat_func()
    find_any(xps.arrays(strat, ()))


@settings(deadline=None)
@given(data=st.data())
def test_draw_arrays_from_dtype_name_strategies(xp: Any, xps: Any, data: st.DataObject) -> None:
    """Draw arrays from dtype name strategies."""
    all_names = ('bool', *REAL_NAMES)
    if xps.api_version > '2021.12':
        all_names += COMPLEX_NAMES
    sample_names = data.draw(st.lists(st.sampled_from(all_names), min_size=1, unique=True))
    find_any(xps.arrays(st.sampled_from(sample_names), ()))


def test_generate_arrays_from_shapes_strategy(xp: Any, xps: Any) -> None:
    """Generate arrays from shapes strategy."""
    find_any(xps.arrays(xp.int8, xps.array_shapes()))


def test_generate_arrays_from_integers_strategy_as_shape(xp: Any, xps: Any) -> None:
    """Generate arrays from integers strategy as shapes strategy."""
    find_any(xps.arrays(xp.int8, st.integers(0, 100)))


def test_generate_arrays_from_zero_dimensions(xp: Any, xps: Any) -> None:
    """Generate arrays from empty shape."""
    assert_all_examples(xps.arrays(xp.int8, ()), lambda x: x.shape == ())


@given(data=st.data())
def test_generate_arrays_from_zero_sided_shapes(xp: Any, xps: Any, data: st.DataObject) -> None:
    """Generate arrays from shapes with at least one 0-sized dimension."""
    shape = data.draw(xps.array_shapes(min_side=0).filter(lambda s: 0 in s))
    arr = data.draw(xps.arrays(xp.int8, shape))
    assert arr.shape == shape


def test_generate_arrays_from_unsigned_ints(xp: Any, xps: Any) -> None:
    """Generate arrays from unsigned integer dtype."""
    assert_all_examples(xps.arrays(xp.uint32, (5, 5)), lambda x: xp.all(x >= 0))
    signed_max = xp.iinfo(xp.int32).max
    find_any(xps.arrays(xp.uint32, (5, 5)), lambda x: xp.any(x > signed_max))


def test_generate_arrays_from_0d_arrays(xp: Any, xps: Any) -> None:
    """Generate arrays from 0d array elements."""
    assert_all_examples(xps.arrays(dtype=xp.uint8, shape=(5, 5), elements=xps.from_dtype(xp.uint8).map(lambda e: xp.asarray(e, dtype=xp.uint8))), lambda x: x.shape == (5, 5))


def test_minimize_arrays_with_default_dtype_shape_strategies(xp: Any, xps: Any) -> None:
    """Strategy with default scalar_dtypes and array_shapes strategies minimize
    to a boolean 1-dimensional array of size 1."""
    smallest = minimal(xps.arrays(xps.scalar_dtypes(), xps.array_shapes()))
    assert smallest.shape == (1,)
    assert smallest.dtype == xp.bool
    assert not xp.any(smallest)


def test_minimize_arrays_with_0d_shape_strategy(xp: Any, xps: Any) -> None:
    """Strategy with shape strategy that can generate empty tuples minimizes to
    0d arrays."""
    smallest = minimal(xps.arrays(xp.int8, xps.array_shapes(min_dims=0)))
    assert smallest.shape == ()


@pytest.mark.parametrize('dtype', dtype_name_params[1:])
def test_minimizes_numeric_arrays(xp: Any, xps: Any, dtype: str) -> None:
    """Strategies with numeric dtypes minimize to zero-filled arrays."""
    smallest = minimal(xps.arrays(dtype, (2, 2)))
    assert xp.all(smallest == 0)


def test_minimize_large_uint_arrays(xp: Any, xps: Any) -> None:
    """Strategy with uint dtype and largely sized shape minimizes to a good
    example."""
    if not hasattr(xp, 'nonzero'):
        pytest.skip('optional API')
    smallest = minimal(xps.arrays(xp.uint8, 100), lambda x: xp.any(x) and (not xp.all(x)))
    assert xp.all(xp.logical_or(smallest == 0, smallest == 1))
    idx = xp.nonzero(smallest)[0]
    assert idx.size in (1, smallest.size - 1)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@flaky(max_runs=50, min_passes=1)
def test_minimize_float_arrays(xp: Any, xps: Any) -> None:
    """Strategy with float dtype minimizes to a good example.

    We filter runtime warnings and expect flaky array generation for
    specifically NumPy - this behaviour may not be required when testing
    with other array libraries.
    """
    smallest = minimal(xps.arrays(xp.float32, 50), lambda x: xp.sum(x) >= 1.0)
    assert xp.sum(smallest) in (1, 50) or all((math.isinf(v) for v in smallest))


def test_minimizes_to_fill(xp: Any, xps: Any) -> None:
    """Strategy with single fill value minimizes to arrays only containing said
    fill value."""
    smallest = minimal(xps.arrays(xp.float32, 10, fill=st.just(3.0)))
    assert xp.all(smallest == 3.0)


def test_generate_unique_arrays(xp: Any, xps: Any) -> None:
    """Generates unique arrays."""
    skip_on_missing_unique_values(xp)
    assert_all_examples(xps.arrays(xp.int8, st.integers(0, 20), unique=True), lambda x: xp.unique_values(x).size == x.size)


def test_cannot_draw_unique_arrays_with_too_small_elements(xp: Any, xps: Any) -> None:
    """Unique strategy with elements strategy range smaller than its size raises
    helpful error."""
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(xps.arrays(xp.int8, 10, elements=st.integers(0, 5), unique=True))


def test_cannot_fill_arrays_with_non_castable_value(xp: Any, xps: Any) -> None:
    """Strategy with fill not castable to dtype raises helpful error."""
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(xps.arrays(xp.int8, 10, fill=st.just('not a castable value'))))


def test_generate_unique_arrays_with_high_collision_elements(xp: Any, xps: Any) -> None:
    """Generates unique arrays with just elements of 0.0 and NaN fill."""

    @given(xps.arrays(dtype=xp.float32, shape=st.integers(0, 20), elements=st.just(0.0), fill=st.just(xp.nan), unique=True))
    def test(x: Any) -> None:
        zero_mask = x == 0.0
        assert xp.sum(xp.astype(zero_mask, xp.uint8)) <= 1
    test()


def test_generate_unique_arrays_using_all_elements(xp: Any, xps: Any) -> None:
    """Unique strategy with elements strategy range equal to its size will only
    generate arrays with one of each possible element."""
    skip_on_missing_unique_values(xp)
    assert_all_examples(xps.arrays(xp.int8, (4,), elements=st.integers(0, 3), unique=True), lambda x: xp.unique_values(x).size == x.size)


def test_may_fill_unique_arrays_with_nan(xp: Any, xps: Any) -> None:
    """Unique strategy with NaN fill can generate arrays holding NaNs."""
    find_any(xps.arrays(dtype=xp.float32, shape=10, elements={'allow_nan': False}, unique=True, fill=st.just(xp.nan)), lambda x: xp.any(xp.isnan(x)))


def test_may_not_fill_unique_array_with_non_nan(xp: Any, xps: Any) -> None:
    """Unique strategy with just fill elements of 0.0 raises helpful error."""
    strat = xps.arrays(dtype=xp.float32, shape=10, elements={'allow_nan': False}, unique=True, fill=st.just(0.0))
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(strat)


def test_floating_point_array() -> None:
    import warnings
    from hypothesis.extra.array_api import make_strategies_namespace
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import numpy.array_api as nxp
    except ModuleNotFoundError:
        import numpy as nxp
    xps = make_strategies_namespace(nxp)
    dtypes = xps.floating_dtypes() | xps.complex_dtypes()
    strat = xps.arrays(dtype=dtypes, shape=10)
    check_can_generate_examples(strat)


@pytest.mark.parametrize('kwargs', [{'elements': st.just(300)}, {'elements': st.nothing(), 'fill': st.just(300)}])
def test_may_not_use_overflowing_integers(xp: Any, xps: Any, kwargs: Dict[str, Any]) -> None:
    """Strategy with elements strategy range outside the dtype's bounds raises
    helpful error."""
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(xps.arrays(dtype=xp.int8, shape=1, **kwargs))


@pytest.mark.parametrize('fill', [False, True])
@pytest.mark.parametrize('dtype, strat', [('float32', st.floats(min_value=10 ** 40, allow_infinity=False)), ('float64', st.floats(min_value=10 ** 40, allow_infinity=False)), pytest.param('complex64', st.complex_numbers(min_magnitude=10 ** 300, allow_infinity=False), marks=pytest.mark.xp_min_version(MIN_VER_FOR_COMPLEX))])
def test_may_not_use_unrepresentable_elements(xp: Any, xps: Any, fill: bool, dtype: str, strat: st.SearchStrategy[Any]) -> None:
    """Strategy with elements not representable by the dtype raises helpful error."""
    if fill:
        kw = {'elements': st.nothing(), 'fill': strat}
    else:
        kw = {'elements': strat}
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(xps.arrays(dtype=dtype, shape=1, **kw))


def test_floats_can_be_constrained(xp: Any, xps: Any) -> None:
    """Strategy with float dtype and specified elements strategy range
    (inclusive) generates arrays with elements inside said range."""
    assert_all_examples(xps.arrays(dtype=xp.float32, shape=10, elements={'min_value': 0, 'max_value': 1}), lambda x: xp.all(x >= 0) and xp.all(x <= 1))


def test_floats_can_be_constrained_excluding_endpoints(xp: Any, xps: Any) -> None:
    """Strategy with float dtype and specified elements strategy range
    (exclusive) generates arrays with elements inside said range."""
    assert_all_examples(xps.arrays(dtype=xp.float32, shape=10, elements={'min_value': 0, 'max_value': 1, 'exclude_min': True, 'exclude_max': True}), lambda x: xp.all(x > 0) and xp.all(x < 1))


def test_is_still_unique_with_nan_fill(xp: Any, xps: Any) -> None:
    """Unique strategy with NaN fill generates unique arrays."""
    skip_on_missing_unique_values(xp)
    xfail_on_indistinct_nans(xp)
    assert_all_examples(xps.arrays(dtype=xp.float32, elements={'allow_nan': False}, shape=10, unique=True, fill=st.just(xp.nan)), lambda x: xp.unique_values(x).size == x.size)


def test_unique_array_with_fill_can_use_all_elements(xp: Any, xps: Any) -> None:
    """Unique strategy with elements range equivalent to its size and NaN fill
    can generate arrays with all possible values."""
    skip_on_missing_unique_values(xp)
    xfail_on_indistinct_nans(xp)
    find_any(xps.arrays(dtype=xp.float32, shape=10, unique=True, elements=st.integers(1, 9), fill=st.just(xp.nan)), lambda x: xp.unique_values(x).size == x.size)


def test_generate_unique_arrays_without_fill(xp: Any, xps: Any) -> None:
    """Generate arrays from unique strategy with no fill.

    Covers the collision-related branches for fully dense unique arrays.
    Choosing 25 of 256 possible values means we're almost certain to see
    collisions thanks to the birthday paradox, but finding unique values should
    still be easy.
    """
    skip_on_missing_unique_values(xp)
    assert_all_examples(xps.arrays(dtype=xp.uint8, shape=25, unique=True, fill=st.nothing()), lambda x: xp.unique_values(x).size == x.size)


def test_efficiently_generate_unique_arrays_using_all_elements(xp: Any, xps: Any) -> None:
    """Unique strategy with elements strategy range equivalent to its size
    generates arrays with all possible values. Generation is not too slow.

    Avoids the birthday paradox with UniqueSampledListStrategy.
    """
    skip_on_missing_unique_values(xp)
    assert_all_examples(xps.arrays(dtype=xp.int8, shape=255, unique=True), lambda x: xp.unique_values(x).size == x.size)


@given(st.data(), st.integers(-100, 100), st.integers(1, 100))
def test_array_element_rewriting(xp: Any, xps: Any, data: st.DataObject, start: int, size: int) -> None:
    """Unique strategy generates arrays with expected elements."""
    x = data.draw(xps.arrays(dtype=xp.int64