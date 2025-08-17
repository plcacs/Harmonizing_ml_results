# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import sys
from functools import reduce
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from hypothesis import (
    HealthCheck,
    Phase,
    assume,
    given,
    note,
    settings,
    strategies as st,
    target,
)
from hypothesis.errors import InvalidArgument, UnsatisfiedAssumption
from hypothesis.extra import numpy as nps
from hypothesis.strategies._internal.lazy import unwrap_strategies

from tests.common.debug import check_can_generate_examples, find_any, minimal
from tests.common.utils import fails_with, flaky

ANY_SHAPE: st.SearchStrategy[Tuple[int, ...]] = nps.array_shapes(min_dims=0, max_dims=32, min_side=0, max_side=32)
ANY_NONZERO_SHAPE: st.SearchStrategy[Tuple[int, ...]] = nps.array_shapes(min_dims=0, max_dims=32, min_side=1, max_side=32)


@given(nps.arrays(float, ()))
def test_empty_dimensions_are_arrays(x: np.ndarray) -> None:
    assert isinstance(x, np.ndarray)
    assert x.dtype.kind == "f"


@given(nps.arrays(float, (1, 0, 1)))
def test_can_handle_zero_dimensions(x: np.ndarray) -> None:
    assert x.shape == (1, 0, 1)


@given(nps.arrays("uint32", (5, 5)))
def test_generates_unsigned_ints(x: np.ndarray) -> None:
    assert (x >= 0).all()


@given(nps.arrays(int, (1,)))
def test_assert_fits_in_machine_size(x: np.ndarray) -> None:
    pass


def test_generates_and_minimizes() -> None:
    assert (minimal(nps.arrays(float, (2, 2))) == np.zeros(shape=(2, 2))).all()


def test_can_minimize_large_arrays() -> None:
    x = minimal(nps.arrays("uint32", 100), lambda x: np.any(x) and not np.all(x))
    assert np.logical_or(x == 0, x == 1).all()
    assert np.count_nonzero(x) in (1, len(x) - 1)


@flaky(max_runs=50, min_passes=1)
def test_can_minimize_float_arrays() -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        x = minimal(nps.arrays(float, 50), lambda t: np.nansum(t) >= 1.0)
        assert x.sum() in (1, 50)


class Foo:
    pass


foos: st.SearchStrategy[Foo] = st.tuples().map(lambda _: Foo())


def test_can_create_arrays_of_composite_types() -> None:
    arr = minimal(nps.arrays(object, 100, elements=foos))
    for x in arr:
        assert isinstance(x, Foo)


@given(st.lists(st.integers()), st.data())
def test_can_create_zero_dim_arrays_of_lists(x: List[int], data: st.DataObject) -> None:
    arr = data.draw(nps.arrays(object, (), elements=st.just(x)))
    assert arr.shape == ()
    assert arr.dtype == np.dtype(object)
    assert arr.item() == x


def test_can_create_arrays_of_tuples() -> None:
    arr = minimal(
        nps.arrays(object, 10, elements=st.tuples(st.integers(), st.integers())),
        lambda x: all(t0 != t1 for t0, t1 in x),
    )
    assert all(a in ((1, 0), (0, 1)) for a in arr)


@given(nps.arrays(object, (2, 2), elements=st.tuples(st.integers())))
def test_does_not_flatten_arrays_of_tuples(arr: np.ndarray) -> None:
    assert isinstance(arr[0][0], tuple)


@given(
    nps.arrays(object, (2, 2), elements=st.lists(st.integers(), min_size=1, max_size=1))
)
def test_does_not_flatten_arrays_of_lists(arr: np.ndarray) -> None:
    assert isinstance(arr[0][0], list)


@given(nps.array_shapes())
def test_can_generate_array_shapes(shape: Tuple[int, ...]) -> None:
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@settings(deadline=None, max_examples=10)
@given(st.integers(0, 10), st.integers(0, 9), st.integers(0), st.integers(0))
def test_minimise_array_shapes(
    min_dims: int, dim_range: int, min_side: int, side_range: int
) -> None:
    smallest = minimal(
        nps.array_shapes(
            min_dims=min_dims,
            max_dims=min_dims + dim_range,
            min_side=min_side,
            max_side=min_side + side_range,
        )
    )
    assert len(smallest) == min_dims
    assert all(k == min_side for k in smallest)


@pytest.mark.parametrize(
    "kwargs", [{"min_side": 100}, {"min_dims": 15}, {"min_dims": 32}]
)
def test_interesting_array_shapes_argument(kwargs: Dict[str, int]) -> None:
    check_can_generate_examples(nps.array_shapes(**kwargs))


@given(nps.scalar_dtypes())
def test_can_generate_scalar_dtypes(dtype: np.dtype) -> None:
    assert isinstance(dtype, np.dtype)


@settings(max_examples=100)
@given(
    nps.nested_dtypes(
        subtype_strategy=st.one_of(
            nps.scalar_dtypes(), nps.byte_string_dtypes(), nps.unicode_string_dtypes()
        )
    )
)
def test_can_generate_compound_dtypes(dtype: np.dtype) -> None:
    assert isinstance(dtype, np.dtype)


@settings(max_examples=100)
@given(
    nps.nested_dtypes(
        subtype_strategy=st.one_of(
            nps.scalar_dtypes(), nps.byte_string_dtypes(), nps.unicode_string_dtypes()
        )
    ).flatmap(lambda dt: nps.arrays(dtype=dt, shape=1))
)
def test_can_generate_data_compound_dtypes(arr: np.ndarray) -> None:
    # This is meant to catch the class of errors which prompted PR #2085
    assert isinstance(arr, np.ndarray)


@given(nps.nested_dtypes())
def test_np_dtype_is_idempotent(dtype: np.dtype) -> None:
    assert dtype == np.dtype(dtype)


def test_minimise_scalar_dtypes() -> None:
    assert minimal(nps.scalar_dtypes()) == np.dtype("bool")


def test_minimise_nested_types() -> None:
    assert minimal(nps.nested_dtypes()) == np.dtype("bool")


def test_minimise_array_strategy() -> None:
    smallest = minimal(
        nps.arrays(
            nps.nested_dtypes(max_itemsize=200),
            nps.array_shapes(max_dims=3, max_side=3),
        )
    )
    assert smallest.dtype == np.dtype("bool")
    assert not smallest.any()


@given(nps.array_dtypes(allow_subarrays=False))
def test_can_turn_off_subarrays(dt: np.dtype) -> None:
    for name in dt.names:
        assert dt.fields[name][0].shape == ()


def test_array_dtypes_may_have_field_titles() -> None:
    find_any(nps.array_dtypes(), lambda dt: len(dt.fields) > len(dt.names))


@pytest.mark.parametrize("byteorder", ["<", ">"])
@given(data=st.data())
def test_can_restrict_endianness(data: st.DataObject, byteorder: str) -> None:
    dtype = data.draw(nps.integer_dtypes(endianness=byteorder, sizes=(16, 32, 64)))
    if byteorder == ("<" if sys.byteorder == "little" else ">"):
        assert dtype.byteorder == "="
    else:
        assert dtype.byteorder == byteorder


@given(nps.integer_dtypes(sizes=8))
def test_can_specify_size_as_an_int(dt: np.dtype) -> None:
    assert dt.itemsize == 1


@given(st.data())
def test_can_draw_arrays_from_scalars(data: st.DataObject) -> None:
    dt = data.draw(nps.scalar_dtypes())
    result = data.draw(nps.arrays(dtype=dt, shape=()))
    assert isinstance(result, np.ndarray)
    assert result.dtype == dt


@given(st.data())
def test_can_cast_for_arrays(data: st.DataObject) -> None:
    # Note: this only passes with castable datatypes, certain dtype
    # combinations will result in an error if numpy is not able to cast them.
    dt_elements = np.dtype(data.draw(st.sampled_from(["bool", "<i2", ">i2"])))
    dt_desired = np.dtype(
        data.draw(st.sampled_from(["<i2", ">i2", "float32", "float64"]))
    )
    result = data.draw(
        nps.arrays(
            dtype=dt_desired, elements=nps.from_dtype(dt_elements), shape=(1, 2, 3)
        )
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == dt_desired


@given(nps.arrays(dtype="int8", shape=st.integers(0, 20), unique=True))
def test_array_values_are_unique(arr: np.ndarray) -> None:
    assert len(set(arr)) == len(arr)


def test_cannot_generate_unique_array_of_too_many_elements() -> None:
    strat = nps.arrays(dtype=int, elements=st.integers(0, 5), shape=10, unique=True)
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(strat)


@given(
    nps.arrays(
        elements=st.just(0.0),
        dtype=float,
        fill=st.just(np.nan),
        shape=st.integers(0, 20),
        unique=True,
    )
)
def test_array_values_are_unique_high_collision(arr: np.ndarray) -> None:
    assert (arr == 0.0).sum() <= 1


@given(nps.arrays(dtype="int8", shape=(4,), elements=st.integers(0, 3), unique=True))
def test_generates_all_values_for_unique_array(arr: np.ndarray) -> None:
    # Ensures that the "reject already-seen element" branch is covered
    assert len(set(arr)) == len(arr)


@given(nps.arrays(dtype="int8", shape=255, unique=True))
def test_efficiently_generates_all_unique_array(arr: np.ndarray) -> None:
    # Avoids the birthday paradox with UniqueSampledListStrategy
    assert len(set(arr)) == len(arr)


@given(st.data(), st.integers(-100, 100), st.integers(1, 100))
def test_array_element_rewriting(data: st.DataObject, start: int, size: int) -> None:
    arr = nps.arrays(
        dtype=np.dtype("int64"),
        shape=size,
        elements=st.integers(start, start + size - 1),
        unique=True,
    )
    assert set(data.draw(arr)) == set(range(start, start + size))


def test_may_fill_with_nan_when_unique_is_set() -> None:
    find_any(
        nps.arrays(
            dtype=float,
            elements=st.floats(allow_nan=False),
            shape=10,
            unique=True,
            fill=st.just(np.nan),
        ),
        lambda x: np.isnan(x).any(),
    )


@given(
    nps.arrays(
        dtype=float,
        elements=st.floats(allow_nan=False),
        shape=10,
        unique=True,
        fill=st.just(np.nan),
    )
)
def test_is_still_unique_with_nan_fill(xs: np.ndarray) -> None:
    assert len(set(xs)) == len(xs)


@fails_with(InvalidArgument)
@given(
    nps.arrays(
        dtype=float,
        elements=st.floats(allow_nan=False),
        shape=10,
        unique=True,
        fill=st.just(0.0),
    )
)
def test_may_not_fill_with_non_nan_when_unique_is_set(arr: np.ndarray) -> None:
    pass


@fails_with(InvalidArgument)
@given(nps.arrays(dtype="U", shape=10, unique=True, fill=st.just("")))
def test_may_not_fill_with_non_nan_when_unique_is_set_and_type_is_not_number(arr: np.ndarray) -> None:
    pass


np_version = tuple(int(x) for x in np.__version__.split(".")[:2])


@pytest.mark.parametrize("fill", [False, True])
# Overflowing elements deprecated upstream in Numpy 1.24 :-)
@fails_with(
    InvalidArgument
    if np_version < (1, 24)
    else (DeprecationWarning if np_version < (2, 0) else OverflowError)
)
@given(st.data())
def test_overflowing_integers_are_deprecated(fill: bool, data: st.DataObject) -> None:
    kw = {"elements": st.just(300)}
    if fill:
        kw = {"elements": st.nothing(), "fill": kw["elements"]}
    arr = data.draw(nps.arrays(dtype="int8", shape=(1,), **kw)
    assert arr[0] == (300 % 256)


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize(
    "dtype,strat",
    [
        ("float16", st.floats(min_value=65520, allow_infinity=False)),
        ("float32", st.floats(min_value=10**40, allow_infinity=False)),
        (
            "complex64",
            st.complex_numbers(min_magnitude=10**300, allow_infinity=False),
        ),
        ("U1", st.text(min_size=2, max_size=2)),
        ("S1", st.binary(min_size=2, max_size=2)),
    ],
)
@fails_with(InvalidArgument)
@given(data=st.data())
def test_unrepresentable_elements_are_deprecated(
    fill: bool, dtype: str, strat: st.SearchStrategy[Any], data: st.DataObject
) -> None:
    if fill:
        kw = {"elements": st.nothing(), "fill": strat}
    else:
        kw = {"elements": strat}
    try:
        arr = data.draw(nps.arrays(dtype=dtype, shape=(1,), **kw))
    except RuntimeWarning:
        assert np_version >= (1, 24), "New overflow-on-cast detection"
        raise InvalidArgument("so the test passes") from None

    try:
        # This is a float or complex number, and has overflowed to infinity,
        # triggering our deprecation for overflow.
        assert np.isinf(arr[0])
    except TypeError:
        # We tried to call isinf on a string.  The string was generated at
        # length two, then truncated by the dtype of size 1 - deprecation
        # again.  If the first character was \0 it is now the empty string.
        assert len(arr[0]) <= 1


@given(nps.arrays(dtype="float16", shape=(1,)))
def test_inferred_floats_do_not_overflow(arr: np.ndarray) -> None:
    pass


@given(nps.arrays(dtype="float16", shape=10, elements={"min_value": 0, "max_value": 1}))
def test_inferred_floats_can_be_constrained_at_low_width(arr: np.ndarray) -> None:
    assert (arr >= 0).all()
    assert (arr <= 1).all()


@given(
    nps.arrays(
        dtype="float16",
        shape=10,
        elements={
            "min_value": 0,
            "max_value": 1,
            "exclude_min": True,
            "exclude_max": True,
        },
    )
)
def test_inferred_floats_can_be_constrained_at_low_width_excluding_endpoints(arr: np.ndarray) -> None:
    assert (arr > 0).all()
    assert (arr < 1).all()


@given(
    nps.arrays(
        dtype="float16",
        shape=10,
        unique=True,
        elements=st.integers(1, 9),
        fill=st.just(np.nan),
    )
)
def test_unique_array_with_fill_can_use_all_elements(arr: np.ndarray) -> None:
    assume(len(set(arr)) == arr.size)


@given(nps.arrays(dtype="uint8", shape=25, unique=True, fill=st.nothing()))
def test_unique_array_without_fill(arr: np.ndarray) -> None:
    # This test covers the collision-related branches for fully dense unique arrays.
    # Choosing 25 of