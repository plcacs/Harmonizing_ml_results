#!/usr/bin/env python3
"""
These the test the public routines exposed in types/common.py
related to inference and not otherwise tested in types/test_common.py
"""
from collections import namedtuple
import collections
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import Any, Callable, Iterator as TypingIterator, List, Optional, Tuple, Union, TypeVar

import numpy as np
import pytest
from pandas._libs import lib, missing as libmissing, ops as libops
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import ensure_int32, is_bool, is_complex, is_datetime64_any_dtype, is_datetime64_dtype, is_datetime64_ns_dtype, is_datetime64tz_dtype, is_float, is_integer, is_number, is_scalar, is_scipy_sparse, is_timedelta64_dtype, is_timedelta64_ns_dtype
import pandas as pd
from pandas import Categorical, DataFrame, DateOffset, DatetimeIndex, Index, Interval, Period, Series, Timedelta, TimedeltaIndex, Timestamp
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

    def __init__(self, values: Any) -> None:
        self._values = values

    def __iter__(self) -> TypingIterator[Any]:
        iter_values = iter(self._values)
        def it_outer() -> TypingIterator[Any]:
            yield from iter_values
        return it_outer()

    def __len__(self) -> int:
        return len(self._values)

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[Any] = None) -> Any:
        return np.asarray(self._values, dtype=dtype)

    @property
    def ndim(self) -> int:
        return self._values.ndim

    @property
    def dtype(self) -> Any:
        return self._values.dtype

    @property
    def size(self) -> int:
        return self._values.size

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._values.shape

ll_params = [
    ([1], True, 'list'),
    ([], True, 'list-empty'),
    ((1,), True, 'tuple'),
    ((), True, 'tuple-empty'),
    ({'a': 1}, True, 'dict'),
    ({}, True, 'dict-empty'),
    ({'a', 1}, 'set', 'set'),
    (set(), 'set', 'set-empty'),
    (frozenset({'a', 1}), 'set', 'frozenset'),
    (frozenset(), 'set', 'frozenset-empty'),
    (iter([1, 2]), True, 'iterator'),
    (iter([]), True, 'iterator-empty'),
    (iter([1, 2]), True, 'generator'),
    ((_ for _ in []), True, 'generator-empty'),
    (Series([1]), True, 'Series'),
    (Series([], dtype=object), True, 'Series-empty'),
    (Series(['a']).str, True, 'StringMethods'),
    (Series([], dtype='O').str, True, 'StringMethods-empty'),
    (Index([1]), True, 'Index'),
    (Index([]), True, 'Index-empty'),
    (DataFrame([[1]]), True, 'DataFrame'),
    (DataFrame(), True, 'DataFrame-empty'),
    (np.ndarray((2,) * 1), True, 'ndarray-1d'),
    (np.array([]), True, 'ndarray-1d-empty'),
    (np.ndarray((2,) * 2), True, 'ndarray-2d'),
    (np.array([[]]), True, 'ndarray-2d-empty'),
    (np.ndarray((2,) * 3), True, 'ndarray-3d'),
    (np.array([[[]]]), True, 'ndarray-3d-empty'),
    (np.ndarray((2,) * 4), True, 'ndarray-4d'),
    (np.array([[[[]]]]), True, 'ndarray-4d-empty'),
    (np.array(2), False, 'ndarray-0d'),
    (MockNumpyLikeArray(np.ndarray((2,) * 1)), True, 'duck-ndarray-1d'),
    (MockNumpyLikeArray(np.array([])), True, 'duck-ndarray-1d-empty'),
    (MockNumpyLikeArray(np.ndarray((2,) * 2)), True, 'duck-ndarray-2d'),
    (MockNumpyLikeArray(np.array([[]])), True, 'duck-ndarray-2d-empty'),
    (MockNumpyLikeArray(np.ndarray((2,) * 3)), True, 'duck-ndarray-3d'),
    (MockNumpyLikeArray(np.array([[[]]])), True, 'duck-ndarray-3d-empty'),
    (MockNumpyLikeArray(np.ndarray((2,) * 4)), True, 'duck-ndarray-4d'),
    (MockNumpyLikeArray(np.array([[[[]]]])), True, 'duck-ndarray-4d-empty'),
    (MockNumpyLikeArray(np.array(2)), False, 'duck-ndarray-0d'),
    (1, False, 'int'),
    (b'123', False, 'bytes'),
    (b'', False, 'bytes-empty'),
    ('123', False, 'string'),
    ('', False, 'string-empty'),
    (str, False, 'string-type'),
    (object(), False, 'object'),
    (np.nan, False, 'NaN'),
    (None, False, 'None')
]
objs, expected, ids = zip(*ll_params)

@pytest.fixture(params=zip(objs, expected), ids=ids)
def maybe_list_like(request: pytest.FixtureRequest) -> Tuple[Any, Union[bool, str]]:
    return request.param

def test_is_list_like(maybe_list_like: Tuple[Any, Union[bool, str]]) -> None:
    obj, expected_val = maybe_list_like
    expected_val = True if expected_val == 'set' else expected_val
    assert inference.is_list_like(obj) == expected_val

def test_is_list_like_disallow_sets(maybe_list_like: Tuple[Any, Union[bool, str]]) -> None:
    obj, expected_val = maybe_list_like
    expected_val = False if expected_val == 'set' else expected_val
    assert inference.is_list_like(obj, allow_sets=False) == expected_val

def test_is_list_like_recursion() -> None:
    def list_like() -> None:
        inference.is_list_like([])
        list_like()
    rec_limit: int = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        with tm.external_error_raised(RecursionError):
            list_like()
    finally:
        sys.setrecursionlimit(rec_limit)

def test_is_list_like_iter_is_none() -> None:
    class NotListLike:
        def __getitem__(self, item: Any) -> Any:
            return self
        __iter__ = None
    assert not inference.is_list_like(NotListLike())

def test_is_list_like_generic() -> None:
    T = TypeVar('T')
    class MyDataFrame(DataFrame, Generic[T]):
        ...
    tstc = MyDataFrame[int]
    tst = MyDataFrame[int]({'x': [1, 2, 3]})
    assert not inference.is_list_like(tstc)
    assert isinstance(tst, DataFrame)
    assert inference.is_list_like(tst)

def test_is_sequence() -> None:
    is_seq = inference.is_sequence
    assert is_seq((1, 2))
    assert is_seq([1, 2])
    assert not is_seq('abcd')
    assert not is_seq(np.int64)
    class A:
        def __getitem__(self, item: Any) -> int:
            return 1
    assert not is_seq(A())

def test_is_array_like() -> None:
    assert inference.is_array_like(Series([], dtype=object))
    assert inference.is_array_like(Series([1, 2]))
    assert inference.is_array_like(np.array(['a', 'b']))
    assert inference.is_array_like(Index(['2016-01-01']))
    assert inference.is_array_like(np.array([2, 3]))
    assert inference.is_array_like(MockNumpyLikeArray(np.array([2, 3])))
    class DtypeList(list):
        dtype: str = 'special'
    assert inference.is_array_like(DtypeList())
    assert not inference.is_array_like([1, 2, 3])
    assert not inference.is_array_like(())
    assert not inference.is_array_like('foo')
    assert not inference.is_array_like(123)

@pytest.mark.parametrize('inner', [[], [1], (1,), (1, 2), {'a': 1}, {1, 'a'}, Series([1]), Series([], dtype=object), Series(['a']).str, (x for x in range(5))])
@pytest.mark.parametrize('outer', [list, Series, np.array, tuple])
def test_is_nested_list_like_passes(inner: Any, outer: Callable[[List[Any]], Any]) -> None:
    result = outer([inner for _ in range(5)])
    assert inference.is_list_like(result)

@pytest.mark.parametrize('obj', ['abc', [], [1], (1,), ['a'], 'a', {'a'}, [1, 2, 3], Series([1]), DataFrame({'A': [1]}), ([1, 2] for _ in range(5))])
def test_is_nested_list_like_fails(obj: Any) -> None:
    assert not inference.is_nested_list_like(obj)

@pytest.mark.parametrize('ll', [{}, {'A': 1}, Series([1]), collections.defaultdict()])
def test_is_dict_like_passes(ll: Any) -> None:
    assert inference.is_dict_like(ll)

@pytest.mark.parametrize('ll', ['1', 1, [1, 2], (1, 2), range(2), Index([1]), dict, collections.defaultdict, Series])
def test_is_dict_like_fails(ll: Any) -> None:
    assert not inference.is_dict_like(ll)

@pytest.mark.parametrize('has_keys', [True, False])
@pytest.mark.parametrize('has_getitem', [True, False])
@pytest.mark.parametrize('has_contains', [True, False])
def test_is_dict_like_duck_type(has_keys: bool, has_getitem: bool, has_contains: bool) -> None:
    class DictLike:
        def __init__(self, d: dict) -> None:
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
    expected: bool = has_keys and has_getitem and has_contains
    assert result is expected

def test_is_file_like() -> None:
    class MockFile:
        pass
    is_file = inference.is_file_like
    data = StringIO('data')
    assert is_file(data)
    m = MockFile()
    assert not is_file(m)
    MockFile.write = lambda self: 0
    m = MockFile()
    assert not is_file(m)
    MockFile.__iter__ = lambda self: self
    m = MockFile()
    assert is_file(m)
    del MockFile.write
    MockFile.read = lambda self: 0
    m = MockFile()
    assert is_file(m)
    data = [1, 2, 3]
    assert not is_file(data)

test_tuple = collections.namedtuple('test_tuple', ['a', 'b', 'c'])

@pytest.mark.parametrize('ll', [test_tuple(1, 2, 3)])
def test_is_names_tuple_passes(ll: Any) -> None:
    assert inference.is_named_tuple(ll)

@pytest.mark.parametrize('ll', [(1, 2, 3), 'a', Series({'pi': 3.14})])
def test_is_names_tuple_fails(ll: Any) -> None:
    assert not inference.is_named_tuple(ll)

def test_is_hashable() -> None:
    class HashableClass:
        pass
    class UnhashableClass1:
        __hash__ = None
    class UnhashableClass2:
        def __hash__(self) -> int:
            raise TypeError('Not hashable')
    hashable = (1, 3.14, np.float64(3.14), 'a', (), (1,), HashableClass())
    not_hashable = ([], UnhashableClass1())
    abc_hashable_not_really_hashable = (([],), UnhashableClass2())
    for i in hashable:
        assert inference.is_hashable(i)
    for i in not_hashable:
        assert not inference.is_hashable(i)
    for i in abc_hashable_not_really_hashable:
        assert not inference.is_hashable(i)
    assert not inference.is_hashable(np.array([]))

@pytest.mark.parametrize('ll', [re.compile('ad')])
def test_is_re_passes(ll: Any) -> None:
    assert inference.is_re(ll)

@pytest.mark.parametrize('ll', ['x', 2, 3, object()])
def test_is_re_fails(ll: Any) -> None:
    assert not inference.is_re(ll)

@pytest.mark.parametrize('ll', ['a', 'x', 'asdf', re.compile('adsf'), '\\u2233\\s*', re.compile('')])
def test_is_recompilable_passes(ll: Any) -> None:
    assert inference.is_re_compilable(ll)

@pytest.mark.parametrize('ll', [1, [], object()])
def test_is_recompilable_fails(ll: Any) -> None:
    assert not inference.is_re_compilable(ll)

class TestInference:
    @pytest.mark.parametrize('arr', [np.array(list('abc'), dtype='S1'),
                                      np.array(list('abc'), dtype='S1').astype(object),
                                      [b'a', np.nan, b'c']])
    def test_infer_dtype_bytes(self, arr: Any) -> None:
        result: str = lib.infer_dtype(arr, skipna=True)
        assert result == 'bytes'

    @pytest.mark.parametrize('value, expected', [(float('inf'), True),
                                                   (np.inf, True),
                                                   (-np.inf, False),
                                                   (1, False),
                                                   ('a', False)])
    def test_isposinf_scalar(self, value: Any, expected: bool) -> None:
        result: bool = libmissing.isposinf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize('value, expected', [(float('-inf'), True),
                                                   (-np.inf, True),
                                                   (np.inf, False),
                                                   (1, False),
                                                   ('a', False)])
    def test_isneginf_scalar(self, value: Any, expected: bool) -> None:
        result: bool = libmissing.isneginf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [
        (True, BooleanArray(np.array([True, False], dtype='bool'), np.array([False, True]))),
        (False, np.array([True, np.nan], dtype='object'))
    ])
    def test_maybe_convert_nullable_boolean(self, convert_to_masked_nullable: bool, exp: Any) -> None:
        arr: np.ndarray = np.array([True, np.nan], dtype=object)
        result: Any = libops.maybe_convert_bool(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(BooleanArray(*result), exp)
        else:
            result_arr = result[0]
            tm.assert_numpy_array_equal(result_arr, exp)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    @pytest.mark.parametrize('coerce_numeric', [True, False])
    @pytest.mark.parametrize('infinity', ['inf', 'inF', 'iNf', 'Inf', 'iNF', 'InF', 'INf', 'INF'])
    @pytest.mark.parametrize('prefix', ['', '-', '+'])
    def test_maybe_convert_numeric_infinities(self, coerce_numeric: bool, infinity: str, prefix: str, convert_to_masked_nullable: bool) -> None:
        result, _ = lib.maybe_convert_numeric(np.array([prefix + infinity], dtype=object),
                                               na_values={'', 'NULL', 'nan'},
                                               coerce_numeric=coerce_numeric,
                                               convert_to_masked_nullable=convert_to_masked_nullable)
        expected: np.ndarray = np.array([np.inf if prefix in ['', '+'] else -np.inf])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_maybe_convert_numeric_infinities_raises(self, convert_to_masked_nullable: bool) -> None:
        msg: str = 'Unable to parse string'
        with pytest.raises(ValueError, match=msg):
            lib.maybe_convert_numeric(np.array(['foo_inf'], dtype=object),
                                      na_values={'', 'NULL', 'nan'},
                                      coerce_numeric=False,
                                      convert_to_masked_nullable=convert_to_masked_nullable)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_maybe_convert_numeric_post_floatify_nan(self, coerce: bool, convert_to_masked_nullable: bool) -> None:
        data: np.ndarray = np.array(['1.200', '-999.000', '4.500'], dtype=object)
        expected: np.ndarray = np.array([1.2, np.nan, 4.5], dtype=np.float64)
        nan_values = {-999, -999.0}
        out, _ = lib.maybe_convert_numeric(data, nan_values, coerce, convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            expected_arr = FloatingArray(expected, np.isnan(expected))
            tm.assert_extension_array_equal(expected_arr, FloatingArray(*out))
        else:
            result_arr = out[0]
            tm.assert_numpy_array_equal(result_arr, expected)

    def test_convert_infs(self) -> None:
        arr: np.ndarray = np.array(['inf', 'inf', 'inf'], dtype='O')
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64
        arr = np.array(['-inf', '-inf', '-inf'], dtype='O')
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

    def test_scientific_no_exponent(self) -> None:
        arr: np.ndarray = np.array(['42E', '2E', '99e', '6e'], dtype='O')
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        assert np.all(np.isnan(result))

    def test_convert_non_hashable(self) -> None:
        arr: np.ndarray = np.array([[10.0, 2], 1.0, 'apple'], dtype=object)
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        tm.assert_numpy_array_equal(result, np.array([np.nan, 1.0, np.nan]))

    def test_convert_numeric_uint64(self) -> None:
        arr: np.ndarray = np.array([2 ** 63], dtype=object)
        exp: np.ndarray = np.array([2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)
        arr = np.array([str(2 ** 63)], dtype=object)
        exp = np.array([2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)
        arr = np.array([np.uint64(2 ** 63)], dtype=object)
        exp = np.array([2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

    @pytest.mark.parametrize('arr', [
        np.array([2 ** 63, np.nan], dtype=object),
        np.array([str(2 ** 63), np.nan], dtype=object),
        np.array([np.nan, 2 ** 63], dtype=object),
        np.array([np.nan, str(2 ** 63)], dtype=object)
    ])
    def test_convert_numeric_uint64_nan(self, coerce: bool, arr: np.ndarray) -> None:
        expected: np.ndarray = arr.astype(float) if coerce else arr.copy()
        result, _ = lib.maybe_convert_numeric(arr, set(), coerce_numeric=coerce)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_uint64_nan_values(self, coerce: bool, convert_to_masked_nullable: bool) -> None:
        arr: np.ndarray = np.array([2 ** 63, 2 ** 63 + 1], dtype=object)
        na_values = {2 ** 63}
        expected: Union[np.ndarray, IntegerArray] = (np.array([np.nan, 2 ** 63 + 1], dtype=float)
                                                     if coerce else arr.copy())
        result = lib.maybe_convert_numeric(arr, na_values, coerce_numeric=coerce, convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable and coerce:
            expected = IntegerArray(np.array([0, 2 ** 63 + 1], dtype='u8'), np.array([True, False], dtype='bool'))
            result = IntegerArray(*result)
        else:
            result = result[0]
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('case', [
        np.array([2 ** 63, -1], dtype=object),
        np.array([str(2 ** 63), -1], dtype=object),
        np.array([str(2 ** 63), str(-1)], dtype=object),
        np.array([-1, 2 ** 63], dtype=object),
        np.array([-1, str(2 ** 63)], dtype=object),
        np.array([str(-1), str(2 ** 63)], dtype=object)
    ])
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_int64_uint64(self, case: np.ndarray, coerce: bool, convert_to_masked_nullable: bool) -> None:
        expected: np.ndarray = case.astype(float) if coerce else case.copy()
        result, _ = lib.maybe_convert_numeric(case, set(), coerce_numeric=coerce, convert_to_masked_nullable=convert_to_masked_nullable)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_string_uint64(self, convert_to_masked_nullable: bool) -> None:
        result = lib.maybe_convert_numeric(np.array(['uint64'], dtype=object), set(), coerce_numeric=True, convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            result = FloatingArray(*result)
        else:
            result = result[0]
        assert np.isnan(result)

    @pytest.mark.parametrize('value', [-2 ** 63 - 1, 2 ** 64])
    def test_convert_int_overflow(self, value: Any) -> None:
        arr: np.ndarray = np.array([value], dtype=object)
        result: np.ndarray = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(arr, result)

    @pytest.mark.parametrize('val', [None, np.nan, float('nan')])
    @pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
    def test_maybe_convert_objects_nat_inference(self, val: Any, dtype: str) -> None:
        dt = np.dtype(dtype)
        vals: np.ndarray = np.array([pd.NaT, val], dtype=object)
        result = lib.maybe_convert_objects(vals, convert_non_numeric=True, dtype_if_all_nat=dt)
        assert result.dtype == dt
        assert np.isnat(result).all()
        result = lib.maybe_convert_objects(vals[::-1], convert_non_numeric=True, dtype_if_all_nat=dt)
        assert result.dtype == dt
        assert np.isnat(result).all()

    @pytest.mark.parametrize('value, expected_dtype', [
        ([2 ** 63], np.uint64),
        ([np.uint64(2 ** 63)], np.uint64),
        ([2, -1], np.int64),
        ([2 ** 63, -1], object),
        ([np.uint8(1)], np.uint8),
        ([np.uint16(1)], np.uint16),
        ([np.uint32(1)], np.uint32),
        ([np.uint64(1)], np.uint64),
        ([np.uint8(2), np.uint16(1)], np.uint16),
        ([np.uint32(2), np.uint16(1)], np.uint32),
        ([np.uint32(2), -1], object),
        ([np.uint32(2), 1], np.uint64),
        ([np.uint32(2), np.int32(1)], object)
    ])
    def test_maybe_convert_objects_uint(self, value: List[Any], expected_dtype: Any) -> None:
        arr: np.ndarray = np.array(value, dtype=object)
        exp: np.ndarray = np.array(value, dtype=expected_dtype)
        tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)

    def test_maybe_convert_objects_datetime(self) -> None:
        arr: np.ndarray = np.array([np.datetime64('2000-01-01'), np.timedelta64(1, 's')], dtype=object)
        exp: np.ndarray = arr.copy()
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)
        arr = np.array([pd.NaT, np.timedelta64(1, 's')], dtype=object)
        exp = np.array([np.timedelta64('NaT'), np.timedelta64(1, 's')], dtype='m8[ns]')
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)
        arr = np.array([np.timedelta64(1, 's'), np.nan], dtype=object)
        exp = exp[::-1]
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat(self) -> None:
        arr: np.ndarray = np.array([pd.NaT, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, arr)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('timedelta64[ns]'))
        exp: np.ndarray = np.array(['NaT', 'NaT'], dtype='timedelta64[ns]')
        tm.assert_numpy_array_equal(out, exp)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('datetime64[ns]'))
        exp = np.array(['NaT', 'NaT'], dtype='datetime64[ns]')
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self) -> None:
        arr: np.ndarray = np.array([pd.NaT, pd.NaT], dtype=object)
        with pytest.raises(ValueError, match='int64'):
            lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('int64'))

    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
    def test_maybe_convert_objects_datetime_overflow_safe(self, dtype: str) -> None:
        stamp: datetime = datetime(2363, 10, 4)
        if dtype == 'timedelta64[ns]':
            stamp = stamp - datetime(1970, 1, 1)
        arr: np.ndarray = np.array([stamp], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        if dtype == 'datetime64[ns]':
            expected: np.ndarray = np.array(['2363-10-04'], dtype='M8[us]')
        else:
            expected = arr
        tm.assert_numpy_array_equal(out, expected)

    def test_maybe_convert_objects_mixed_datetimes(self) -> None:
        ts: Timestamp = Timestamp('now')
        vals: List[Any] = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]
        for data in itertools.permutations(vals):
            data_arr = np.array(list(data), dtype=object)
            expected = DatetimeIndex(data_arr)._data._ndarray
            result = lib.maybe_convert_objects(data_arr, convert_non_numeric=True)
            tm.assert_numpy_array_equal(result, expected)

    def test_maybe_convert_objects_timedelta64_nat(self) -> None:
        obj = np.timedelta64('NaT', 'ns')
        arr: np.ndarray = np.array([obj], dtype=object)
        assert arr[0] is obj
        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        expected: np.ndarray = np.array([obj], dtype='m8[ns]')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('exp', [
        IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True])),
        IntegerArray(np.array([2, 0], dtype='int64'), np.array([False, True]))
    ])
    def test_maybe_convert_objects_nullable_integer(self, exp: Any) -> None:
        arr: np.ndarray = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(result, exp)

    @pytest.mark.parametrize('dtype, val', [('int64', 1), ('uint64', np.iinfo(np.int64).max + 1)])
    def test_maybe_convert_objects_nullable_none(self, dtype: str, val: Any) -> None:
        arr: np.ndarray = np.array([val, None, 3], dtype='object')
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        expected = IntegerArray(np.array([val, 0, 3], dtype=dtype), np.array([False, True, False]))
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [
        (True, IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True]))),
        (False, np.array([2, np.nan], dtype='float64'))
    ])
    def test_maybe_convert_numeric_nullable_integer(self, convert_to_masked_nullable: bool, exp: Any) -> None:
        arr: np.ndarray = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            result_arr = IntegerArray(*result)
            tm.assert_extension_array_equal(result_arr, exp)
        else:
            result_arr = result[0]
            tm.assert_numpy_array_equal(result_arr, exp)

    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [
        (True, FloatingArray(np.array([2.0, 0.0], dtype='float64'), np.array([False, True]))),
        (False, np.array([2.0, np.nan], dtype='float64'))
    ])
    def test_maybe_convert_numeric_floating_array(self, convert_to_masked_nullable: bool, exp: Any) -> None:
        arr: np.ndarray = np.array([2.0, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(FloatingArray(*result), exp)
        else:
            result_arr = result[0]
            tm.assert_numpy_array_equal(result_arr, exp)

    def test_maybe_convert_objects_bool_nan(self) -> None:
        ind: Index = Index([True, False, np.nan], dtype=object)
        exp: np.ndarray = np.array([True, False, np.nan], dtype=object)
        out = lib.maybe_convert_objects(ind.values, safe=1)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_nullable_boolean(self) -> None:
        arr: np.ndarray = np.array([True, False], dtype=object)
        exp: BooleanArray = BooleanArray._from_sequence([True, False], dtype='boolean')
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(out, exp)
        arr = np.array([True, False, pd.NaT], dtype=object)
        exp = np.array([True, False, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_numpy_array_equal(out, exp)

    @pytest.mark.parametrize('val', [None, np.nan])
    def test_maybe_convert_objects_nullable_boolean_na(self, val: Any) -> None:
        arr: np.ndarray = np.array([True, False, val], dtype=object)
        exp = BooleanArray(np.array([True, False, False]), np.array([False, False, True]))
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(out, exp)

    @pytest.mark.parametrize('data0', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
    @pytest.mark.parametrize('data1', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
    def test_maybe_convert_objects_itemsize(self, data0: Any, data1: Any) -> None:
        data: List[Any] = [data0, data1]
        arr: np.ndarray = np.array(data, dtype='object')
        common_kind = np.result_type(type(data0), type(data1)).kind
        kind0 = 'python' if not hasattr(data0, 'dtype') else data0.dtype.kind
        kind1 = 'python' if not hasattr(data1, 'dtype') else data1.dtype.kind
        if kind0 != 'python' and kind1 != 'python':
            kind = common_kind
            itemsize = max(data0.dtype.itemsize, data1.dtype.itemsize)
        elif is_bool(data0) or is_bool(data1):
            kind = 'bool' if is_bool(data0) and is_bool(data1) else 'object'
            itemsize = ''
        elif is_complex(data0) or is_complex(data1):
            kind = common_kind
            itemsize = 16
        else:
            kind = common_kind
            itemsize = 8
        expected = np.array(data, dtype=f'{kind}{itemsize}')
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_dtypes_remain_object_array(self) -> None:
        arr: np.ndarray = np.array([datetime(2015, 1, 1, tzinfo=timezone.utc), 1], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(result, arr)

    @pytest.mark.parametrize('idx', [pd.IntervalIndex.from_breaks(range(5), closed='both'),
                                       pd.period_range('2016-01-01', periods=3, freq='D')])
    def test_maybe_convert_objects_ea(self, idx: Any) -> None:
        result = lib.maybe_convert_objects(np.array(idx, dtype=object), convert_non_numeric=True)
        tm.assert_extension_array_equal(result, idx._data)

class TestTypeInference:
    class Dummy:
        pass

    def test_inferred_dtype_fixture(self, any_skipna_inferred_dtype: Tuple[str, Any]) -> None:
        inferred_dtype, values = any_skipna_inferred_dtype
        assert inferred_dtype == lib.infer_dtype(values, skipna=True)

    def test_length_zero(self, skipna: bool) -> None:
        result: str = lib.infer_dtype(np.array([], dtype='i4'), skipna=skipna)
        assert result == 'integer'
        result = lib.infer_dtype([], skipna=skipna)
        assert result == 'empty'
        arr: np.ndarray = np.array([np.array([], dtype=object), np.array([], dtype=object)])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'empty'

    def test_integers(self) -> None:
        arr: np.ndarray = np.array([1, 2, 3, np.int64(4), np.int32(5)], dtype='O')
        result: str = lib.infer_dtype(arr, skipna=True)
        assert result == 'integer'
        arr = np.array([1, 2, 3, np.int64(4), np.int32(5), 'foo'], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed-integer'
        arr = np.array([1, 2, 3, 4, 5], dtype='i4')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'integer'

    @pytest.mark.parametrize('arr, skipna', [
        ([1, 2, np.nan, np.nan, 3], False),
        ([1, 2, np.nan, np.nan, 3], True),
        ([1, 2, 3, np.int64(4), np.int32(5), np.nan], False),
        ([1, 2, 3, np.int64(4), np.int32(5), np.nan], True)
    ])
    def test_integer_na(self, arr: List[Any], skipna: bool) -> None:
        result: str = lib.infer_dtype(np.array(arr, dtype='O'), skipna=skipna)
        expected: str = 'integer' if skipna else 'integer-na'
        assert result == expected

    def test_infer_dtype_skipna_default(self) -> None:
        arr: np.ndarray = np.array([1, 2, 3, np.nan], dtype=object)
        result: str = lib.infer_dtype(arr)
        assert result == 'integer'

    def test_bools(self) -> None:
        arr: np.ndarray = np.array([True, False, True, True, True], dtype='O')
        result: str = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        arr = np.array([np.bool_(True), np.bool_(False)], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        arr = np.array([True, False, True, 'foo'], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed'
        arr = np.array([True, False, True], dtype=bool)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        arr = np.array([True, np.nan, False], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'boolean'
        result = lib.infer_dtype(arr, skipna=False)
        assert result == 'mixed'

    def test_floats(self) -> None:
        arr: np.ndarray = np.array([1.0, 2.0, 3.0, np.float64(4), np.float32(5)], dtype='O')
        result: str = lib.infer_dtype(arr, skipna=True)
        assert result == 'floating'
        arr = np.array([1, 2, 3, np.float64(4), np.float32(5), 'foo'], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed-integer'
        arr = np.array([1, 2, 3, 4, 5], dtype='f4')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'floating'
        arr = np.array([1, 2, 3, 4, 5], dtype='f8')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'floating'

    def test_decimals(self) -> None:
        arr: np.ndarray = np.array([Decimal(1), Decimal(2), Decimal(3)])
        result: str = lib.infer_dtype(arr, skipna=True)
        assert result == 'decimal'
        arr = np.array([1.0, 2.0, Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed'
        result = lib.infer_dtype(arr[::-1], skipna=True)
        assert result == 'mixed'
        arr = np.array([Decimal(1), Decimal('NaN'), Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'decimal'
        arr = np.array([Decimal(1), np.nan, Decimal(3)], dtype='O')
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'decimal'

    def test_complex(self, skipna: bool) -> None:
        arr: np.ndarray = np.array([1.0, 2.0, 1 + 1j])
        result: str = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'
        arr = np.array([1.0, 2.0, 1 + 1j], dtype='O')
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'mixed'
        result = lib.infer_dtype(arr[::-1], skipna=skipna)
        assert result == 'mixed'
        arr = np.array([1, np.nan, 1 + 1j])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'
        arr = np.array([1.0, np.nan, 1 + 1j], dtype='O')
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'mixed'
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype='O')
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype=np.complex64)
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == 'complex'

    def test_string(self) -> None:
        pass

    def test_unicode(self) -> None:
        arr: List[Any] = ['a', np.nan, 'c']
        result: str = lib.infer_dtype(arr, skipna=False)
        assert result == 'mixed'
        arr = ['a', np.nan, 'c']
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'string'
        arr = ['a', pd.NA, 'c']
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'string'
        arr = ['a', pd.NaT, 'c']
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'mixed'
        arr = ['a', 'c']
        result = lib.infer_dtype(arr, skipna=False)
        assert result == 'string'

    @pytest.mark.parametrize('dtype, missing, skipna, expected', [
        (float, np.nan, False, 'floating'),
        (float, np.nan, True, 'floating'),
        (object, np.nan, False, 'floating'),
        (object, np.nan, True, 'empty'),
        (object, None, False, 'mixed'),
        (object, None, True, 'empty')
    ])
    @pytest.mark.parametrize('box', [Series, np.array])
    def test_object_empty(self, box: Callable[[List[Any], Any], Any], missing: Any, dtype: Any, skipna: bool, expected: str) -> None:
        arr = box([missing, missing], dtype=dtype)
        result: str = lib.infer_dtype(arr, skipna=skipna)
        assert result == expected

    def test_datetime(self) -> None:
        dates: List[datetime] = [datetime(2012, 1, x) for x in range(1, 20)]
        index: Index = Index(dates)
        assert index.inferred_type == 'datetime64'

    def test_infer_dtype_datetime64(self) -> None:
        arr: np.ndarray = np.array([np.datetime64('2011-01-01'), np.datetime64('2011-01-01')], dtype=object)
        assert lib.infer_dtype(arr, skipna=True) == 'datetime64'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    def test_infer_dtype_datetime64_with_na(self, na_value: Any) -> None:
        arr: np.ndarray = np.array([na_value, np.datetime64('2011-01-02')])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime64'
        arr = np.array([na_value, np.datetime64('2011-01-02'), na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime64'

    @pytest.mark.parametrize('arr', [
        np.array([np.timedelta64('nat'), np.datetime64('2011-01-02')], dtype=object),
        np.array([np.datetime64('2011-01-02'), np.timedelta64('nat')], dtype=object),
        np.array([np.datetime64('2011-01-01'), Timestamp('2011-01-02')]),
        np.array([Timestamp('2011-01-02'), np.datetime64('2011-01-01')]),
        np.array([np.nan, Timestamp('2011-01-02'), 1.1]),
        np.array([np.nan, '2011-01-01', Timestamp('2011-01-02')], dtype=object),
        np.array([np.datetime64('nat'), np.timedelta64(1, 'D')], dtype=object),
        np.array([np.timedelta64(1, 'D'), np.datetime64('nat')], dtype=object)
    ])
    def test_infer_datetimelike_dtype_mixed(self, arr: Any) -> None:
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'

    def test_infer_dtype_mixed_integer(self) -> None:
        arr: np.ndarray = np.array([np.nan, Timestamp('2011-01-02'), 1])
        assert lib.infer_dtype(arr, skipna=True) == 'mixed-integer'

    @pytest.mark.parametrize('arr', [
        [Timestamp('2011-01-01'), Timestamp('2011-01-02')],
        [datetime(2011, 1, 1), datetime(2012, 2, 1)],
        [datetime(2011, 1, 1), Timestamp('2011-01-02')]
    ])
    def test_infer_dtype_datetime(self, arr: List[Any]) -> None:
        assert lib.infer_dtype(np.array(arr), skipna=True) == 'datetime'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    @pytest.mark.parametrize('time_stamp', [Timestamp('2011-01-01'), datetime(2011, 1, 1)])
    def test_infer_dtype_datetime_with_na(self, na_value: Any, time_stamp: Union[Timestamp, datetime]) -> None:
        arr: np.ndarray = np.array([na_value, time_stamp])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime'
        arr = np.array([na_value, time_stamp, na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'datetime'

    @pytest.mark.parametrize('arr', [
        np.array([Timedelta('1 days'), Timedelta('2 days')]),
        np.array([np.timedelta64(1, 'D'), np.timedelta64(2, 'D')], dtype=object),
        np.array([timedelta(1), timedelta(2)])
    ])
    def test_infer_dtype_timedelta(self, arr: Any) -> None:
        assert lib.infer_dtype(arr, skipna=True) == 'timedelta'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    @pytest.mark.parametrize('delta', [Timedelta('1 days'), np.timedelta64(1, 'D'), timedelta(1)])
    def test_infer_dtype_timedelta_with_na(self, na_value: Any, delta: Union[Timedelta, np.timedelta64, timedelta]) -> None:
        arr: np.ndarray = np.array([na_value, delta])
        assert lib.infer_dtype(arr, skipna=True) == 'timedelta'
        arr = np.array([na_value, delta, na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'timedelta'

    def test_infer_dtype_period(self) -> None:
        arr: np.ndarray = np.array([Period('2011-01', freq='D'), Period('2011-02', freq='D')])
        assert lib.infer_dtype(arr, skipna=True) == 'period'
        arr = np.array([Period('2011-01', freq='D'), Period('2011-02', freq='M')])
        assert lib.infer_dtype(arr, skipna=True) == 'mixed'

    def test_infer_dtype_period_array(self, index_or_series_or_array: Callable[..., Any], skipna: bool) -> None:
        klass = index_or_series_or_array
        values = klass([Period('2011-01-01', freq='D'), Period('2011-01-02', freq='D'), pd.NaT])
        assert lib.infer_dtype(values, skipna=skipna) == 'period'
        values = klass([Period('2011-01-01', freq='D'), Period('2011-01-02', freq='M'), pd.NaT])
        exp = 'unknown-array' if klass is pd.array else 'mixed'
        assert lib.infer_dtype(values, skipna=skipna) == exp

    def test_infer_dtype_period_mixed(self) -> None:
        arr: np.ndarray = np.array([Period('2011-01', freq='M'), np.datetime64('nat')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([np.datetime64('nat'), Period('2011-01', freq='M')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'

    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    def test_infer_dtype_period_with_na(self, na_value: Any) -> None:
        arr: np.ndarray = np.array([na_value, Period('2011-01', freq='D')])
        assert lib.infer_dtype(arr, skipna=True) == 'period'
        arr = np.array([na_value, Period('2011-01', freq='D'), na_value])
        assert lib.infer_dtype(arr, skipna=True) == 'period'

    def test_infer_dtype_all_nan_nat_like(self) -> None:
        arr: np.ndarray = np.array([np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == 'floating'
        arr = np.array([np.nan, np.nan, None])
        assert lib.infer_dtype(arr, skipna=True) == 'empty'
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([None, np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == 'empty'
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([np.nan, pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([np.nan, pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([None, pd.NaT, None])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime'
        arr = np.array([np.datetime64('nat')])
        assert lib.infer_dtype(arr, skipna=False) == 'datetime64'
        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.datetime64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'datetime64'
            arr = np.array([pd.NaT, n, np.datetime64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'datetime64'
        arr = np.array([np.timedelta64('nat')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'timedelta'
        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.timedelta64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'timedelta'
            arr = np.array([pd.NaT, n, np.timedelta64('nat'), n])
            assert lib.infer_dtype(arr, skipna=False) == 'timedelta'
        arr = np.array([pd.NaT, np.datetime64('nat'), np.timedelta64('nat'), np.nan])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([np.timedelta64('nat'), np.datetime64('nat')], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'

    def test_is_datetimelike_array_all_nan_nat_like(self) -> None:
        arr: np.ndarray = np.array([np.nan, pd.NaT, np.datetime64('nat')])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, pd.NaT, np.timedelta64('nat')])
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, pd.NaT, np.datetime64('nat'), np.timedelta64('nat')])
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, pd.NaT])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)
        arr = np.array([np.nan, np.nan], dtype=object)
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)
        assert lib.is_datetime_with_singletz_array(np.array([Timestamp('20130101', tz='US/Eastern'), Timestamp('20130102', tz='US/Eastern')], dtype=object))
        assert not lib.is_datetime_with_singletz_array(np.array([Timestamp('20130101', tz='US/Eastern'), Timestamp('20130102', tz='CET')], dtype=object))

    @pytest.mark.parametrize('func', [
        'is_datetime_array', 'is_datetime64_array', 'is_bool_array',
        'is_timedelta_or_timedelta64_array', 'is_date_array', 'is_time_array', 'is_interval_array'
    ])
    def test_other_dtypes_for_array(self, func: str) -> None:
        f: Callable[[np.ndarray], bool] = getattr(lib, func)
        arr: np.ndarray = np.array(['foo', 'bar'])
        assert not f(arr)
        assert not f(arr.reshape(2, 1))
        arr = np.array([1, 2])
        assert not f(arr)
        assert not f(arr.reshape(2, 1))

    def test_to_object_array_tuples(self) -> None:
        r: Tuple[int, int] = (5, 6)
        values: List[Tuple[int, int]] = [r]
        lib.to_object_array_tuples(values)
        record = namedtuple('record', 'x y')
        r2 = record(5, 6)
        values = [r2]
        lib.to_object_array_tuples(values)

    def test_object(self) -> None:
        arr: np.ndarray = np.array([None], dtype='O')
        result = lib.infer_dtype(arr, skipna=False)
        assert result == 'mixed'
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'empty'

    def test_to_object_array_width(self) -> None:
        rows: List[List[int]] = [[1, 2, 3], [4, 5, 6]]
        expected: np.ndarray = np.array(rows, dtype=object)
        out = lib.to_object_array(rows)
        tm.assert_numpy_array_equal(out, expected)
        expected = np.array(rows, dtype=object)
        out = lib.to_object_array(rows, min_width=1)
        tm.assert_numpy_array_equal(out, expected)
        expected = np.array([[1, 2, 3, None, None], [4, 5, 6, None, None]], dtype=object)
        out = lib.to_object_array(rows, min_width=5)
        tm.assert_numpy_array_equal(out, expected)

    def test_categorical(self) -> None:
        arr = Categorical(list('abc'))
        result: str = lib.infer_dtype(arr, skipna=True)
        assert result == 'categorical'
        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == 'categorical'
        arr = Categorical(list('abc'), categories=['cegfab'], ordered=True)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'categorical'
        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == 'categorical'

    @pytest.mark.parametrize('asobject', [True, False])
    def test_interval(self, asobject: bool) -> None:
        idx = pd.IntervalIndex.from_breaks(range(5), closed='both')
        if asobject:
            idx = idx.astype(object)
        inferred = lib.infer_dtype(idx, skipna=False)
        assert inferred == 'interval'
        inferred = lib.infer_dtype(idx._data, skipna=False)
        assert inferred == 'interval'
        inferred = lib.infer_dtype(Series(idx, dtype=idx.dtype), skipna=False)
        assert inferred == 'interval'

    @pytest.mark.parametrize('value', [Timestamp(0), Timedelta(0), 0, 0.0])
    def test_interval_mismatched_closed(self, value: Any) -> None:
        first = Interval(value, value, closed='left')
        second = Interval(value, value, closed='right')
        arr: np.ndarray = np.array([first, first], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'interval'
        arr2: np.ndarray = np.array([first, second], dtype=object)
        assert lib.infer_dtype(arr2, skipna=False) == 'mixed'

    def test_interval_mismatched_subtype(self) -> None:
        first = Interval(0, 1, closed='left')
        second = Interval(Timestamp(0), Timestamp(1), closed='left')
        third = Interval(Timedelta(0), Timedelta(1), closed='left')
        arr: np.ndarray = np.array([first, second])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([second, third])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        arr = np.array([first, third])
        assert lib.infer_dtype(arr, skipna=False) == 'mixed'
        flt_interval = Interval(1.5, 2.5, closed='left')
        arr = np.array([first, flt_interval], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == 'interval'

    @pytest.mark.parametrize('data', [['a', 'b', 'c'], ['a', 'b', pd.NA]])
    def test_string_dtype(self, data: List[Any], skipna: bool, index_or_series_or_array: Callable[..., Any], nullable_string_dtype: Any) -> None:
        val = index_or_series_or_array(data, dtype=nullable_string_dtype)
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == 'string'

    @pytest.mark.parametrize('data', [[True, False, True], [True, False, pd.NA]])
    def test_boolean_dtype(self, data: List[Any], skipna: bool, index_or_series_or_array: Callable[..., Any]) -> None:
        val = index_or_series_or_array(data, dtype='boolean')
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == 'boolean'

class TestNumberScalar:
    def test_is_number(self) -> None:
        assert is_number(True)
        assert is_number(1)
        assert is_number(1.1)
        assert is_number(1 + 3j)
        assert is_number(np.int64(1))
        assert is_number(np.float64(1.1))
        assert is_number(np.complex128(1 + 3j))
        assert is_number(np.nan)
        assert not is_number(None)
        assert not is_number('x')
        assert not is_number(datetime(2011, 1, 1))
        assert not is_number(np.datetime64('2011-01-01'))
        assert not is_number(Timestamp('2011-01-01'))
        assert not is_number(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_number(timedelta(1000))
        assert not is_number(Timedelta('1 days'))
        assert not is_number(np.bool_(False))
        assert is_number(np.timedelta64(1, 'D'))

    def test_is_bool(self) -> None:
        assert is_bool(True)
        assert is_bool(False)
        assert is_bool(np.bool_(False))
        assert not is_bool(1)
        assert not is_bool(1.1)
        assert not is_bool(1 + 3j)
        assert not is_bool(np.int64(1))
        assert not is_bool(np.float64(1.1))
        assert not is_bool(np.complex128(1 + 3j))
        assert not is_bool(np.nan)
        assert not is_bool(None)
        assert not is_bool('x')
        assert not is_bool(datetime(2011, 1, 1))
        assert not is_bool(np.datetime64('2011-01-01'))
        assert not is_bool(Timestamp('2011-01-01'))
        assert not is_bool(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_bool(timedelta(1000))
        assert not is_bool(np.timedelta64(1, 'D'))
        assert not is_bool(Timedelta('1 days'))

    def test_is_integer(self) -> None:
        assert is_integer(1)
        assert is_integer(np.int64(1))
        assert not is_integer(True)
        assert not is_integer(1.1)
        assert not is_integer(1 + 3j)
        assert not is_integer(False)
        assert not is_integer(np.bool_(False))
        assert not is_integer(np.float64(1.1))
        assert not is_integer(np.complex128(1 + 3j))
        assert not is_integer(np.nan)
        assert not is_integer(None)
        assert not is_integer('x')
        assert not is_integer(datetime(2011, 1, 1))
        assert not is_integer(np.datetime64('2011-01-01'))
        assert not is_integer(Timestamp('2011-01-01'))
        assert not is_integer(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_integer(timedelta(1000))
        assert not is_integer(Timedelta('1 days'))
        assert not is_integer(np.timedelta64(1, 'D'))

    def test_is_float(self) -> None:
        assert is_float(1.1)
        assert is_float(np.float64(1.1))
        assert is_float(np.nan)
        assert not is_float(True)
        assert not is_float(1)
        assert not is_float(1 + 3j)
        assert not is_float(False)
        assert not is_float(np.bool_(False))
        assert not is_float(np.int64(1))
        assert not is_float(np.complex128(1 + 3j))
        assert not is_float(None)
        assert not is_float('x')
        assert not is_float(datetime(2011, 1, 1))
        assert not is_float(np.datetime64('2011-01-01'))
        assert not is_float(Timestamp('2011-01-01'))
        assert not is_float(Timestamp('2011-01-01', tz='US/Eastern'))
        assert not is_float(timedelta(1000))
        assert not is_float(np.timedelta64(1, 'D'))
        assert not is_float(Timedelta('1 days'))

    def test_is_datetime_dtypes(self) -> None:
        ts = pd.date_range('20130101', periods=3)
        tsa = pd.date_range('20130101', periods=3, tz='US/Eastern')
        msg: str = 'is_datetime64tz_dtype is deprecated'
        assert is_datetime64_dtype('datetime64')
        assert is_datetime64_dtype('datetime64[ns]')
        assert is_datetime64_dtype(ts)
        assert not is_datetime64_dtype(tsa)
        assert not is_datetime64_ns_dtype('datetime64')
        assert is_datetime64_ns_dtype('datetime64[ns]')
        assert is_datetime64_ns_dtype(ts)
        assert is_datetime64_ns_dtype(tsa)
        assert is_datetime64_any_dtype('datetime64')
        assert is_datetime64_any_dtype('datetime64[ns]')
        assert is_datetime64_any_dtype(ts)
        assert is_datetime64_any_dtype(tsa)
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert not is_datetime64tz_dtype('datetime64')
            assert not is_datetime64tz_dtype('datetime64[ns]')
            assert not is_datetime64tz_dtype(ts)
            assert is_datetime64tz_dtype(tsa)

    @pytest.mark.parametrize('tz', ['US/Eastern', 'UTC'])
    def test_is_datetime_dtypes_with_tz(self, tz: str) -> None:
        dtype = f'datetime64[ns, {tz}]'
        assert not is_datetime64_dtype(dtype)
        msg: str = 'is_datetime64tz_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_any_dtype(dtype)

    def test_is_timedelta(self) -> None:
        assert is_timedelta64_dtype('timedelta64')
        assert is_timedelta64_dtype('timedelta64[ns]')
        assert not is_timedelta64_ns_dtype('timedelta64')
        assert is_timedelta64_ns_dtype('timedelta64[ns]')
        tdi = TimedeltaIndex([100000000000000.0, 200000000000000.0], dtype='timedelta64[ns]')
        assert is_timedelta64_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi.astype('timedelta64[ns]'))
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.float64))
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.int64))

class TestIsScalar:
    def test_is_scalar_builtin_scalars(self) -> None:
        assert is_scalar(None)
        assert is_scalar(True)
        assert is_scalar(False)
        assert is_scalar(Fraction())
        assert is_scalar(0.0)
        assert is_scalar(1)
        assert is_scalar(complex(2))
        assert is_scalar(float('NaN'))
        assert is_scalar(np.nan)
        assert is_scalar('foobar')
        assert is_scalar(b'foobar')
        assert is_scalar(datetime(2014, 1, 1))
        assert is_scalar(date(2014, 1, 1))
        assert is_scalar(time(12, 0))
        assert is_scalar(timedelta(hours=1))
        assert is_scalar(pd.NaT)
        assert is_scalar(pd.NA)

    def test_is_scalar_builtin_nonscalars(self) -> None:
        assert not is_scalar({})
        assert not is_scalar([])
        assert not is_scalar([1])
        assert not is_scalar(())
        assert not is_scalar((1,))
        assert not is_scalar(slice(None))
        assert not is_scalar(Ellipsis)

    def test_is_scalar_numpy_array_scalars(self) -> None:
        assert is_scalar(np.int64(1))
        assert is_scalar(np.float64(1.0))
        assert is_scalar(np.int32(1))
        assert is_scalar(np.complex64(2))
        assert is_scalar(np.object_('foobar'))
        assert is_scalar(np.str_('foobar'))
        assert is_scalar(np.bytes_(b'foobar'))
        assert is_scalar(np.datetime64('2014-01-01'))
        assert is_scalar(np.timedelta64(1, 'h'))

    @pytest.mark.parametrize('zerodim', [1, 'foobar', np.datetime64('2014-01-01'), np.timedelta64(1, 'h'), np.datetime64('NaT')])
    def test_is_scalar_numpy_zerodim_arrays(self, zerodim: Any) -> None:
        zerodim = np.array(zerodim)
        assert not is_scalar(zerodim)
        assert is_scalar(lib.item_from_zerodim(zerodim))

    @pytest.mark.parametrize('arr', [np.array([]), np.array([[]])])
    def test_is_scalar_numpy_arrays(self, arr: np.ndarray) -> None:
        assert not is_scalar(arr)
        assert not is_scalar(MockNumpyLikeArray(arr))

    def test_is_scalar_pandas_scalars(self) -> None:
        assert is_scalar(Timestamp('2014-01-01'))
        assert is_scalar(Timedelta(hours=1))
        assert is_scalar(Period('2014-01-01'))
        assert is_scalar(Interval(left=0, right=1))
        assert is_scalar(DateOffset(days=1))
        assert is_scalar(pd.offsets.Minute(3))

    def test_is_scalar_pandas_containers(self) -> None:
        assert not is_scalar(Series(dtype=object))
        assert not is_scalar(Series([1]))
        assert not is_scalar(DataFrame())
        assert not is_scalar(DataFrame([[1]]))
        assert not is_scalar(Index([]))
        assert not is_scalar(Index([1]))
        assert not is_scalar(Categorical([]))
        assert not is_scalar(DatetimeIndex([])._data)
        assert not is_scalar(TimedeltaIndex([])._data)
        assert not is_scalar(DatetimeIndex([])._data.to_period('D'))
        assert not is_scalar(pd.array([1, 2, 3]))

    def test_is_scalar_number(self) -> None:
        class Numeric(Number):
            def __init__(self, value: int) -> None:
                self.value = value
            def __int__(self) -> int:
                return self.value
        num = Numeric(1)
        assert is_scalar(num)

@pytest.mark.parametrize('unit', ['ms', 'us', 'ns'])
def test_datetimeindex_from_empty_datetime64_array(unit: str) -> None:
    idx = DatetimeIndex(np.array([], dtype=f'datetime64[{unit}]'))
    assert len(idx) == 0

def test_nan_to_nat_conversions() -> None:
    df = DataFrame({'A': np.asarray(range(10), dtype='float64'), 'B': Timestamp('20010101')})
    df.iloc[3:6, :] = np.nan
    result = df.loc[4, 'B']
    assert result is pd.NaT
    s = df['B'].copy()
    s[8:9] = np.nan
    assert s[8] is pd.NaT

@pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
@pytest.mark.parametrize('spmatrix', ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'])
def test_is_scipy_sparse(spmatrix: str) -> None:
    sparse = pytest.importorskip('scipy.sparse')
    klass = getattr(sparse, spmatrix + '_matrix')
    assert is_scipy_sparse(klass([[0, 1]]))
    assert not is_scipy_sparse(np.array([1]))

def test_ensure_int32() -> None:
    values = np.arange(10, dtype=np.int32)
    result = ensure_int32(values)
    assert result.dtype == np.int32
    values = np.arange(10, dtype=np.int64)
    result = ensure_int32(values)
    assert result.dtype == np.int32

@pytest.mark.parametrize('right,result', [
    (0, np.uint8),
    (-1, np.int16),
    (300, np.uint16),
    (300.0, np.uint16),
    (300.1, np.float64),
    (np.int16(300), np.int16 if np_version_gt2 else np.uint16)
])
def test_find_result_type_uint_int(right: Any, result: Any) -> None:
    left_dtype = np.dtype('uint8')
    assert find_result_type(left_dtype, right) == result

@pytest.mark.parametrize('right,result', [
    (0, np.int8),
    (-1, np.int8),
    (300, np.int16),
    (300.0, np.int16),
    (300.1, np.float64),
    (np.int16(300), np.int16)
])
def test_find_result_type_int_int(right: Any, result: Any) -> None:
    left_dtype = np.dtype('int8')
    assert find_result_type(left_dtype, right) == result

@pytest.mark.parametrize('right,result', [
    (300.0, np.float64),
    (np.float32(300), np.float32)
])
def test_find_result_type_floats(right: Any, result: Any) -> None:
    left_dtype = np.dtype('float16')
    assert find_result_type(left_dtype, right) == result
