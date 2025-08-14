from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
    Generic,
    TypeVar,
)

import numpy as np
import pytest

from pandas._libs import (
    lib,
    missing as libmissing,
    ops as libops,
)
from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
    ensure_int32,
    is_bool,
    is_complex,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_float,
    is_integer,
    is_number,
    is_scalar,
    is_scipy_sparse,
    is_timedelta64_dtype,
    is_timedelta64_ns_dtype,
)

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DateOffset,
    DatetimeIndex,
    Index,
    Interval,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    FloatingArray,
    IntegerArray,
)


@pytest.fixture(params=[True, False], ids=str)
def coerce(request: pytest.FixtureRequest) -> bool:
    return request.param


class MockNumpyLikeArray:
    """
    A class which is numpy-like (e.g. Pint's Quantity) but not actually numpy
    """
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def __iter__(self) -> Iterator[Any]:
        iter_values = iter(self._values)

        def it_outer() -> Iterator[Any]:
            yield from iter_values

        return it_outer()

    def __len__(self) -> int:
        return len(self._values)

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    @property
    def ndim(self) -> int:
        return self._values.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._values.dtype

    @property
    def size(self) -> int:
        return self._values.size

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._values.shape


ll_params: List[Tuple[Any, Any, str]] = [
    ([1], True, "list"),
    ([], True, "list-empty"),
    ((1,), True, "tuple"),
    ((), True, "tuple-empty"),
    ({"a": 1}, True, "dict"),
    ({}, True, "dict-empty"),
    ({"a", 1}, "set", "set"),
    (set(), "set", "set-empty"),
    (frozenset({"a", 1}), "set", "frozenset"),
    (frozenset(), "set", "frozenset-empty"),
    (iter([1, 2]), True, "iterator"),
    (iter([]), True, "iterator-empty"),
    ((x for x in [1, 2]), True, "generator"),
    ((_ for _ in []), True, "generator-empty"),
    (Series([1]), True, "Series"),
    (Series([], dtype=object), True, "Series-empty"),
    (Series(["a"]).str, True, "StringMethods"),
    (Series([], dtype="O").str, True, "StringMethods-empty"),
    (Index([1]), True, "Index"),
    (Index([]), True, "Index-empty"),
    (DataFrame([[1]]), True, "DataFrame"),
    (DataFrame(), True, "DataFrame-empty"),
    (np.ndarray((2,) * 1), True, "ndarray-1d"),
    (np.array([]), True, "ndarray-1d-empty"),
    (np.ndarray((2,) * 2), True, "ndarray-2d"),
    (np.array([[]]), True, "ndarray-2d-empty"),
    (np.ndarray((2,) * 3), True, "ndarray-3d"),
    (np.array([[[]]]), True, "ndarray-3d-empty"),
    (np.ndarray((2,) * 4), True, "ndarray-4d"),
    (np.array([[[[]]]]), True, "ndarray-4d-empty"),
    (np.array(2), False, "ndarray-0d"),
    (MockNumpyLikeArray(np.ndarray((2,) * 1)), True, "duck-ndarray-1d"),
    (MockNumpyLikeArray(np.array([])), True, "duck-ndarray-1d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 2)), True, "duck-ndarray-2d"),
    (MockNumpyLikeArray(np.array([[]])), True, "duck-ndarray-2d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 3)), True, "duck-ndarray-3d"),
    (MockNumpyLikeArray(np.array([[[]]])), True, "duck-ndarray-3d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 4)), True, "duck-ndarray-4d"),
    (MockNumpyLikeArray(np.array([[[[]]]])), True, "duck-ndarray-4d-empty"),
    (MockNumpyLikeArray(np.array(2)), False, "duck-ndarray-0d"),
    (1, False, "int"),
    (b"123", False, "bytes"),
    (b"", False, "bytes-empty"),
    ("123", False, "string"),
    ("", False, "string-empty"),
    (str, False, "string-type"),
    (object(), False, "object"),
    (np.nan, False, "NaN"),
    (None, False, "None"),
]
objs, expected, ids = zip(*ll_params)


@pytest.fixture(params=zip(objs, expected), ids=ids)
def maybe_list_like(request: pytest.FixtureRequest) -> Tuple[Any, Any]:
    return request.param


def test_is_list_like(maybe_list_like: Tuple[Any, Any]) -> None:
    obj, expected = maybe_list_like
    expected = True if expected == "set" else expected
    assert inference.is_list_like(obj) == expected


def test_is_list_like_disallow_sets(maybe_list_like: Tuple[Any, Any]) -> None:
    obj, expected = maybe_list_like
    expected = False if expected == "set" else expected
    assert inference.is_list_like(obj, allow_sets=False) == expected


def test_is_list_like_recursion() -> None:
    def list_like() -> None:
        inference.is_list_like([])
        list_like()

    rec_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        with tm.external_error_raised(RecursionError):
            list_like()
    finally:
        sys.setrecursionlimit(rec_limit)


def test_is_list_like_iter_is_none() -> None:
    class NotListLike:
        def __getitem__(self, item: Any) -> "NotListLike":
            return self

        __iter__ = None

    assert not inference.is_list_like(NotListLike())


def test_is_list_like_generic() -> None:
    T = TypeVar("T")

    class MyDataFrame(DataFrame, Generic[T]):
        pass

    tstc = MyDataFrame[int]
    tst = MyDataFrame[int]({"x": [1, 2, 3]})

    assert not inference.is_list_like(tstc)
    assert isinstance(tst, DataFrame)
    assert inference.is_list_like(tst)


def test_is_sequence() -> None:
    is_seq = inference.is_sequence
    assert is_seq((1, 2))
    assert is_seq([1, 2])
    assert not is_seq("abcd")
    assert not is_seq(np.int64)

    class A:
        def __getitem__(self, item: Any) -> int:
            return 1

    assert not is_seq(A())


def test_is_array_like() -> None:
    assert inference.is_array_like(Series([], dtype=object))
    assert inference.is_array_like(Series([1, 2]))
    assert inference.is_array_like(np.array(["a", "b"]))
    assert inference.is_array_like(Index(["2016-01-01"]))
    assert inference.is_array_like(np.array([2, 3]))
    assert inference.is_array_like(MockNumpyLikeArray(np.array([2, 3])))

    class DtypeList(list):
        dtype = "special"

    assert inference.is_array_like(DtypeList())

    assert not inference.is_array_like([1, 2, 3])
    assert not inference.is_array_like(())
    assert not inference.is_array_like("foo")
    assert not inference.is_array_like(123)


@pytest.mark.parametrize(
    "inner",
    [
        [],
        [1],
        (1,),
        (1, 2),
        {"a": 1},
        {1, "a"},
        Series([1]),
        Series([], dtype=object),
        Series(["a"]).str,
        (x for x in range(5)),
    ],
)
@pytest.mark.parametrize("outer", [list, Series, np.array, tuple])
def test_is_nested_list_like_passes(inner: Any, outer: Any) -> None:
    result = outer([inner for _ in range(5)])
    assert inference.is_list_like(result)


@pytest.mark.parametrize(
    "obj",
    [
        "abc",
        [],
        [1],
        (1,),
        ["a"],
        "a",
        {"a"},
        [1, 2, 3],
        Series([1]),
        DataFrame({"A": [1]}),
        ([1, 2] for _ in range(5)),
    ],
)
def test_is_nested_list_like_fails(obj: Any) -> None:
    assert not inference.is_nested_list_like(obj)


@pytest.mark.parametrize("ll", [{}, {"A": 1}, Series([1]), collections.defaultdict()])
def test_is_dict_like_passes(ll: Any) -> None:
    assert inference.is_dict_like(ll)


@pytest.mark.parametrize(
    "ll",
    [
        "1",
        1,
        [1, 2],
        (1, 2),
        range(2),
        Index([1]),
        dict,
        collections.defaultdict,
        Series,
    ],
)
def test_is_dict_like_fails(ll: Any) -> None:
    assert not inference.is_dict_like(ll)


@pytest.mark.parametrize(
    "value, expected_dtype",
    [
        ([2**63], np.uint64),
        ([np.uint64(2**63)], np.uint64),
        ([2, -1], np.int64),
        ([2**63, -1], object),
        ([np.uint8(1)], np.uint8),
        ([np.uint16(1)], np.uint16),
        ([np.uint32(1)], np.uint32),
        ([np.uint64(1)], np.uint64),
        ([np.uint8(2), np.uint16(1)], np.uint16),
        ([np.uint32(2), np.uint16(1)], np.uint32),
        ([np.uint32(2), -1], object),
        ([np.uint32(2), 1], np.uint64),
        ([np.uint32(2), np.int32(1)], object),
    ],
)
def test_maybe_convert_objects_uint(value: List[Any], expected_dtype: np.dtype) -> None:
    arr = np.array(value, dtype=object)
    exp = np.array(value, dtype=expected_dtype)
    tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)


def test_maybe_convert_objects_datetime() -> None:
    arr = np.array(
        [np.datetime64("2000-01-01"), np.timedelta64(1, "s")], dtype=object
    )
    exp = arr.copy()
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    tm.assert_numpy_array_equal(out, exp)

    arr = np.array([pd.NaT, np.timedelta64(1, "s")], dtype=object)
    exp = np.array([np.timedelta64("NaT"), np.timedelta64(1, "s")], dtype="m8[ns]")
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    tm.assert_numpy_array_equal(out, exp)

    arr = np.array([np.timedelta64(1, "s"), np.nan], dtype=object)
    exp = exp[::-1]
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    tm.assert_numpy_array_equal(out, exp)


def test_maybe_convert_objects_dtype_if_all_nat() -> None:
    arr = np.array([pd.NaT, pd.NaT], dtype=object)
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    tm.assert_numpy_array_equal(out, arr)

    out = lib.maybe_convert_objects(
        arr,
        convert_non_numeric=True,
        dtype_if_all_nat=np.dtype("timedelta64[ns]"),
    )
    exp = np.array(["NaT", "NaT"], dtype="timedelta64[ns]")
    tm.assert_numpy_array_equal(out, exp)

    out = lib.maybe_convert_objects(
        arr,
        convert_non_numeric=True,
        dtype_if_all_nat=np.dtype("datetime64[ns]"),
    )
    exp = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
    tm.assert_numpy_array_equal(out, exp)


def test_maybe_convert_objects_dtype_if_all_nat_invalid() -> None:
    arr = np.array([pd.NaT, pd.NaT], dtype=object)

    with pytest.raises(ValueError, match="int64"):
        lib.maybe_convert_objects(
            arr,
            convert_non_numeric=True,
            dtype_if_all_nat=np.dtype("int64"),
        )


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_maybe_convert_objects_datetime_overflow_safe(dtype: str) -> None:
    stamp = datetime(2363, 10, 4)
    if dtype == "timedelta64[ns]":
        stamp = stamp - datetime(1970, 1, 1)
    arr = np.array([stamp], dtype=object)

    out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    if dtype == "datetime64[ns]":
        expected = np.array(["2363-10-04"], dtype="M8[us]")
    else:
        expected = arr
    tm.assert_numpy_array_equal(out, expected)


def test_maybe_convert_objects_mixed_datetimes() -> None:
    ts = Timestamp("now")
    vals = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]

    for data in itertools.permutations(vals):
        data = np.array(list(data), dtype=object)
        expected = DatetimeIndex(data)._data._ndarray
        result = lib.maybe_convert_objects(data, convert_non_numeric=True)
        tm.assert_numpy_array_equal(result, expected)


def test_maybe_convert_objects_timedelta64_nat() -> None:
    obj = np.timedelta64("NaT", "ns")
    arr = np.array([obj], dtype=object)
    assert arr[0] is obj

    result = lib.maybe_convert_objects(arr, convert_non_numeric=True)

    expected = np.array([obj], dtype="m8[ns]")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "exp",
    [
        IntegerArray(np.array([2, 0], dtype="i8"), np.array([False, True])),
        IntegerArray(np.array([2, 0], dtype="int64"), np.array([False, True])),
    ],
)
def test_maybe_convert_objects_nullable_integer(exp: IntegerArray) -> None:
    arr = np.array([2, np.nan], dtype=object)
    result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)

    tm.assert_extension_array_equal(result, exp)


@pytest.mark.parametrize(
    "dtype, val", [("int64", 1), ("uint64", np.iinfo(np.int64).max + 1)]
)
def test_maybe_convert_objects_nullable_none(dtype: str, val: int) -> None:
    arr = np.array([val, None, 3], dtype="object")
    result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
    expected = IntegerArray(
        np.array([val, 0, 3], dtype=dtype), np.array([False, True, False])
    )
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "convert_to_masked_nullable, exp",
    [
        (True