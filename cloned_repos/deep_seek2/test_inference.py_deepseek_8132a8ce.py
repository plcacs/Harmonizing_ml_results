"""
These the test the public routines exposed in types/common.py
related to inference and not otherwise tested in types/test_common.py
"""

import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import Generic, TypeVar, Any, List, Tuple, Union, Optional, Dict, Set, Callable, Sequence, Iterable, Type, cast

import numpy as np
import pytest

from pandas._libs import lib, missing as libmissing, ops as libops
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
from pandas.core.arrays import BooleanArray, FloatingArray, IntegerArray


@pytest.fixture(params=[True, False], ids=str)
def coerce(request: pytest.FixtureRequest) -> bool:
    return request.param


class MockNumpyLikeArray:
    """
    A class which is numpy-like (e.g. Pint's Quantity) but not actually numpy

    The key is that it is not actually a numpy array so
    ``util.is_array(mock_numpy_like_array_instance)`` returns ``False``. Other
    important properties are that the class defines a :meth:`__iter__` method
    (so that ``isinstance(abc.Iterable)`` returns ``True``) and has a
    :meth:`ndim` property, as pandas special-cases 0-dimensional arrays in some
    cases.

    We expect pandas to behave with respect to such duck arrays exactly as
    with real numpy arrays. In particular, a 0-dimensional duck array is *NOT*
    a scalar (`is_scalar(np.array(1)) == False`), but it is not list-like either.
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


# collect all objects to be tested for list-like-ness; use tuples of objects,
# whether they are list-like or not (special casing for sets), and their ID
ll_params: List[Tuple[Any, Union[bool, str], str]] = [
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
    # Series.str will still raise a TypeError if iterated
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
def maybe_list_like(request: pytest.FixtureRequest) -> Tuple[Any, Union[bool, str]]:
    return request.param


def test_is_list_like(maybe_list_like: Tuple[Any, Union[bool, str]]) -> None:
    obj, expected = maybe_list_like
    expected = True if expected == "set" else expected
    assert inference.is_list_like(obj) == expected


def test_is_list_like_disallow_sets(maybe_list_like: Tuple[Any, Union[bool, str]]) -> None:
    obj, expected = maybe_list_like
    expected = False if expected == "set" else expected
    assert inference.is_list_like(obj, allow_sets=False) == expected


def test_is_list_like_recursion() -> None:
    # GH 33721
    # interpreter would crash with SIGABRT
    def list_like() -> None:
        inference.is_list_like([])
        list_like()

    rec_limit = sys.getrecursionlimit()
    try:
        # Limit to avoid stack overflow on Windows CI
        sys.setrecursionlimit(100)
        with tm.external_error_raised(RecursionError):
            list_like()
    finally:
        sys.setrecursionlimit(rec_limit)


def test_is_list_like_iter_is_none() -> None:
    # GH 43373
    # is_list_like was yielding false positives with __iter__ == None
    class NotListLike:
        def __getitem__(self, item: Any) -> "NotListLike":
            return self

        __iter__ = None

    assert not inference.is_list_like(NotListLike())


def test_is_list_like_generic() -> None:
    # GH 49649
    # is_list_like was yielding false positives for Generic classes in python 3.11
    T = TypeVar("T")

    class MyDataFrame(DataFrame, Generic[T]): ...

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
def test_is_nested_list_like_passes(inner: Any, outer: Callable[[Any], Any]) -> None:
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


@pytest.mark.parametrize("has_keys", [True, False])
@pytest.mark.parametrize("has_getitem", [True, False])
@pytest.mark.parametrize("has_contains", [True, False])
def test_is_dict_like_duck_type(has_keys: bool, has_getitem: bool, has_contains: bool) -> None:
    class DictLike:
        def __init__(self, d: Dict[Any, Any]) -> None:
            self.d = d

        if has_keys:

            def keys(self) -> Any:
                return self.d.keys()

        if has_getitem:

            def __getitem__(self, key: Any) -> Any:
                return self.d.__getitem__(key)

        if has_contains:

            def __contains__(self, key: Any) -> bool:
                return self.d.__contains__(key)

    d = DictLike({1: 2})
    result = inference.is_dict_like(d)
    expected = has_keys and has_getitem and has_contains

    assert result is expected


def test_is_file_like() -> None:
    class MockFile:
        pass

    is_file = inference.is_file_like

    data = StringIO("data")
    assert is_file(data)

    # No read / write attributes
    # No iterator attributes
    m = MockFile()
    assert not is_file(m)

    MockFile.write = lambda self: 0

    # Write attribute but not an iterator
    m = MockFile()
    assert not is_file(m)

    # gh-16530: Valid iterator just means we have the
    # __iter__ attribute for our purposes.
    MockFile.__iter__ = lambda self: self

    # Valid write-only file
    m = MockFile()
    assert is_file(m)

    del MockFile.write
    MockFile.read = lambda self: 0

    # Valid read-only file
    m = MockFile()
    assert is_file(m)

    # Iterator but no read / write attributes
    data = [1, 2, 3]
    assert not is_file(data)


test_tuple = collections.namedtuple("test_tuple", ["a", "b", "c"])


@pytest.mark.parametrize("ll", [test_tuple(1, 2, 3)])
def test_is_names_tuple_passes(ll: Any) -> None:
    assert inference.is_named_tuple(ll)


@pytest.mark.parametrize("ll", [(1, 2, 3), "a", Series({"pi": 3.14})])
def test_is_names_tuple_fails(ll: Any) -> None:
    assert not inference.is_named_tuple(ll)


def test_is_hashable() -> None:
    # all new-style classes are hashable by default
    class HashableClass:
        pass

    class UnhashableClass1:
        __hash__ = None

    class UnhashableClass2:
        def __hash__(self) -> None:
            raise TypeError("Not hashable")

    hashable = (1, 3.14, np.float64(3.14), "a", (), (1,), HashableClass())
    not_hashable = ([], UnhashableClass1())
    abc_hashable_not_really_hashable = (([],), UnhashableClass2())

    for i in hashable:
        assert inference.is_hashable(i)
    for i in not_hashable:
        assert not inference.is_hashable(i)
    for i in abc_hashable_not_really_hashable:
        assert not inference.is_hashable(i)

    # numpy.array is no longer collections.abc.Hashable as of
    # https://github.com/numpy/numpy/pull/5326, just test
    # is_hashable()
    assert not inference.is_hashable(np.array([]))


@pytest.mark.parametrize("ll", [re.compile("ad")])
def test_is_re_passes(ll: Any) -> None:
    assert inference.is_re(ll)


@pytest.mark.parametrize("ll", ["x", 2, 3, object()])
def test_is_re_fails(ll: Any) -> None:
    assert not inference.is_re(ll)


@pytest.mark.parametrize(
    "ll", [r"a", "x", r"asdf", re.compile("adsf"), r"\u2233\s*", re.compile(r"")]
)
def test_is_recompilable_passes(ll: Any) -> None:
    assert inference.is_re_compilable(ll)


@pytest.mark.parametrize("ll", [1, [], object()])
def test_is_recompilable_fails(ll: Any) -> None:
    assert not inference.is_re_compilable(ll)


class TestInference:
    @pytest.mark.parametrize(
        "arr",
        [
            np.array(list("abc"), dtype="S1"),
            np.array(list("abc"), dtype="S1").astype(object),
            [b"a", np.nan, b"c"],
        ],
    )
    def test_infer_dtype_bytes(self, arr: np.ndarray) -> None:
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "bytes"

    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("inf"), True),
            (np.inf, True),
            (-np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isposinf_scalar(self, value: Any, expected: bool) -> None:
        # GH 11352
        result = libmissing.isposinf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("-inf"), True),
            (-np.inf, True),
            (np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isneginf_scalar(self, value: Any, expected: bool) -> None:
        result = libmissing.isneginf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (
                True,
                BooleanArray(
                    np.array([True, False], dtype="bool"), np.array([False, True])
                ),
            ),
            (False, np.array([True, np.nan], dtype="object")),
        ],
    )
    def test_maybe_convert_nullable_boolean(
        self, convert_to_masked_nullable: bool, exp: Any
    ) -> None:
        # GH 40687
        arr = np.array([True, np.nan], dtype=object)
        result = libops.maybe_convert_bool(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(BooleanArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    @pytest.mark.parametrize("coerce_numeric", [True, False])
    @pytest.mark.parametrize(
        "infinity", ["inf", "inF", "iNf", "Inf", "iNF", "InF", "INf", "INF"]
    )
    @pytest.mark.parametrize("prefix", ["", "-", "+"])
    def test_maybe_convert_numeric_infinities(
        self,
        coerce_numeric: bool,
        infinity: str,
        prefix: str,
        convert_to_masked_nullable: bool,
    ) -> None:
        # see gh-13274
        result, _ = lib.maybe_convert_numeric(
            np.array([prefix + infinity], dtype=object),
            na_values={"", "NULL", "nan"},
            coerce_numeric=coerce_numeric,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        expected = np.array([np.inf if prefix in ["", "+"] else -np.inf])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_maybe_convert_numeric_infinities_raises(
        self, convert_to_masked_nullable: bool
    ) -> None:
        msg = "Unable to parse string"
        with pytest.raises(ValueError, match=msg):
            lib.maybe_convert_numeric(
                np.array(["foo_inf"], dtype=object),
                na_values={"", "NULL", "nan"},
                coerce_numeric=False,
                convert_to_masked_nullable=convert_to_masked_nullable,
            )

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_maybe_convert_numeric_post_floatify_nan(
        self, coerce: bool, convert_to_masked_nullable: bool
    ) -> None:
        # see gh-13314
        data = np.array(["1.200", "-999.000", "4.500"], dtype=object)
        expected = np.array([1.2, np.nan, 4.5], dtype=np.float64)
        nan_values = {-999, -999.0}

        out = lib.maybe_convert_numeric(
            data,
            nan_values,
            coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable:
            expected = FloatingArray(expected, np.isnan(expected))
            tm.assert_extension_array_equal(expected, FloatingArray(*out))
        else:
            out = out[0]
            tm.assert_numpy_array_equal(out, expected)

    def test_convert_infs(self) -> None:
        arr = np.array(["inf", "inf", "inf"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

        arr = np.array(["-inf", "-inf", "-inf"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

    def test_scientific_no_exponent(self) -> None:
        # See PR 12215
        arr = np.array(["42E", "2E", "99e", "6e"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        assert np.all(np.isnan(result))

    def test_convert_non_hashable(self) -> None:
        # GH13324
        # make sure that we are handing non-hashables
        arr = np.array([[10.0, 2], 1.0, "apple"], dtype=object)
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        tm.assert_numpy_array_equal(result, np.array([np.nan, 1.0, np.nan]))

    def test_convert_numeric_uint64(self) -> None:
        arr = np.array([2**63], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        arr = np.array([str(2**63)], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        arr = np.array([np.uint64(2**63)], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([2**63, np.nan], dtype=object),
            np.array([str(2**63), np.nan], dtype=object),
            np.array([np.nan, 2**63], dtype=object),
            np.array([np.nan, str(2**63)], dtype=object),
        ],
    )
    def test_convert_numeric_uint64_nan(self, coerce: bool, arr: np.ndarray) -> None:
        expected = arr.astype(float) if coerce else arr.copy()
        result, _ = lib.maybe_convert_numeric(arr, set(), coerce_numeric=coerce)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_uint64_nan_values(
        self, coerce: bool, convert_to_masked_nullable: bool
    ) -> None:
        arr = np.array([2**63, 2**63 + 1], dtype=object)
        na_values = {2**63}

        expected = np.array([np.nan, 2**63 + 1], dtype=float) if coerce else arr.copy()
        result = lib.maybe_convert_numeric(
            arr,
            na_values,
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable and coerce:
            expected = IntegerArray(
                np.array([0, 2**63 + 1], dtype="u8"),
                np.array([True, False], dtype="bool"),
            )
            result = IntegerArray(*result)
        else:
            result = result[0]  # discard mask
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "case",
        [
            np.array([2**63, -1], dtype=object),
            np.array([str(2**63), -1], dtype=object),
            np.array([str(2**63), str(-1)], dtype=object),
            np.array([-1, 2**63], dtype=object),
            np.array([-1, str(2**63)], dtype=object),
            np.array([str(-1), str(2**63)], dtype=object),
        ],
    )
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_int64_uint64(
        self, case: np.ndarray, coerce: bool, convert_to_masked_nullable: bool
    ) -> None:
        expected = case.astype(float) if coerce else case.copy()
        result, _ = lib.maybe_convert_numeric(
            case,
            set(),
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )

        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_string_uint64(self, convert_to_masked_nullable: bool) -> None:
        # GH32394
        result = lib.maybe_convert_numeric(
            np.array(["uint64"], dtype=object),
            set(),
            coerce_numeric=True,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable:
            result = FloatingArray(*result)
        else:
            result = result[0]
        assert np.isnan(result)

    @pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
    def test_convert_int_overflow(self, value: int) -> None:
        # see gh-18584
        arr = np.array([value], dtype=object)
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(arr, result)

    @pytest.mark.parametrize("val", [None, np.nan, float("nan")])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_maybe_convert_objects_nat_inference(self, val: Any, dtype: str) -> None:
        dtype = np.dtype(dtype)
        vals = np.array([pd.NaT, val], dtype=object)
        result = lib.maybe_convert_objects(
            vals,
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        assert result.dtype == dtype
        assert np.isnat(result).all()

        result = lib.maybe_convert_objects(
            vals[::-1],
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        assert result.dtype == dtype
        assert np.isnat(result).all()

    @pytest.mark.parametrize(
        "value, expected_dtype",
        [
            # see gh-4471
            ([2**63], np.uint64),
            # NumPy bug: can't compare uint64 to int64, as that
            # results in both casting to float64, so we should
            # make sure that this function is robust against it
            ([np.uint64(2**63)], np.uint64),
            ([2, -1], np.int64),
            ([2**63, -1], object),
            # GH#47294
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
    def test_maybe_convert_objects_uint(self, value: List[Any], expected_dtype: np.dtype) -> None:
        arr = np.array(value, dtype=object)
        exp = np.array(value, dtype=expected_dtype)
        tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)

    def test_maybe_convert_objects_datetime(self) -> None:
        # GH27438
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

        # with convert_non_numeric=True, the nan is a valid NA value for td64
        arr = np.array([np.timedelta64(1, "s"), np.nan], dtype=object)
        exp = exp[::-1]
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat(self) -> None:
        arr = np.array([pd.NaT, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no dtype_if_all_nat passed -> we dont guess
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

    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self) -> None:
        # we accept datetime64[ns], timedelta64[ns], and EADtype
        arr = np.array([pd.NaT, pd.NaT], dtype=object)

        with pytest.raises(ValueError, match="int64"):
            lib.maybe_convert_objects(
                arr,
                convert_non_numeric=True,
                dtype_if_all_nat=np.dtype("int64"),
            )

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_maybe_convert_objects_datetime_overflow_safe(self, dtype: str) -> None:
        stamp = datetime(2363, 10, 4)  # Enterprise-D launch date
        if dtype == "timedelta64[ns]":
            stamp = stamp - datetime(1970, 1, 1)
        arr = np.array([stamp], dtype=object)

        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no OutOfBoundsDatetime/OutOfBoundsTimedeltas
        if dtype == "datetime64[ns]":
            expected = np.array(["2363-10-04"], dtype="M8[us]")
        else:
            expected = arr
        tm.assert_numpy_array_equal(out, expected)

    def test_maybe_convert_objects_mixed_datetimes(self) -> None:
        ts = Timestamp("now")
        vals = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]

        for data in itertools.permutations(vals):
            data = np.array(list(data), dtype=object)
            expected = DatetimeIndex(data)._data._ndarray
            result = lib.maybe_convert_objects(data, convert_non_numeric=True)
            tm.assert_numpy_array_equal(result, expected)

    def test_maybe_convert_objects_timedelta64_nat(self) -> None:
        obj = np.timedelta64("NaT", "ns")
        arr = np.array([obj], dtype=object)
        assert arr[0] is obj

        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)

        expected = np.array([obj], dtype="m8[ns]")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "exp",
        [
            IntegerArray(np.array([2, 0], dtype="i