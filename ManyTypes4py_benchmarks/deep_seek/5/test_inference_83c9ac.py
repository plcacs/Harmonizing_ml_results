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
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Generic
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

ll_params: List[Tuple[Any, Union[bool, str], str]] = [
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
    ((x for x in [1, 2]), True, 'generator'),
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
    obj, expected = maybe_list_like
    expected = True if expected == 'set' else expected
    assert inference.is_list_like(obj) == expected

def test_is_list_like_disallow_sets(maybe_list_like: Tuple[Any, Union[bool, str]]) -> None:
    obj, expected = maybe_list_like
    expected = False if expected == 'set' else expected
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
        def __getitem__(self, item: Any) -> 'NotListLike':
            return self
        __iter__ = None
    assert not inference.is_list_like(NotListLike())

def test_is_list_like_generic() -> None:
    T = TypeVar('T')
    class MyDataFrame(DataFrame, Generic[T]):
        pass
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
        dtype = 'special'
    assert inference.is_array_like(DtypeList())
    assert not inference.is_array_like([1, 2, 3])
    assert not inference.is_array_like(())
    assert not inference.is_array_like('foo')
    assert not inference.is_array_like(123)

@pytest.mark.parametrize('inner', [[], [1], (1,), (1, 2), {'a': 1}, {1, 'a'}, Series([1]), Series([], dtype=object), Series(['a']).str, (x for x in range(5))])
@pytest.mark.parametrize('outer', [list, Series, np.array, tuple])
def test_is_nested_list_like_passes(inner: Any, outer: Any) -> None:
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
        def __hash__(self) -> None:
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
    @pytest.mark.parametrize('arr', [np.array(list('abc'), dtype='S1'), np.array(list('abc'), dtype='S1').astype(object), [b'a', np.nan, b'c']])
    def test_infer_dtype_bytes(self, arr: Any) -> None:
        result = lib.infer_dtype(arr, skipna=True)
        assert result == 'bytes'

    @pytest.mark.parametrize('value, expected', [(float('inf'), True), (np.inf, True), (-np.inf, False), (1, False), ('a', False)])
    def test_isposinf_scalar(self, value: Any, expected: bool) -> None:
        result = libmissing.isposinf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize('value, expected', [(float('-inf'), True), (-np.inf, True), (np.inf, False), (1, False), ('a', False)])
    def test_isneginf_scalar(self, value: Any, expected: bool) -> None:
        result = libmissing.isneginf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, BooleanArray(np.array([True, False], dtype='bool'), np.array([False, True]))), (False, np.array([True, np.nan], dtype='object'))])
    def test_maybe_convert_nullable_boolean(self, convert_to_masked_nullable: bool, exp: Any) -> None:
        arr = np.array([True, np.nan], dtype=object)
        result = libops.maybe_convert_bool(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(BooleanArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    @pytest.mark.parametrize('coerce_numeric', [True, False])
    @pytest.mark.parametrize('infinity', ['inf', 'inF', 'iNf', 'Inf', 'iNF', 'InF', 'INf', 'INF'])
    @pytest.mark.parametrize('prefix', ['', '-', '+'])
    def test_maybe_convert_numeric_infinities(self, coerce_numeric: bool, infinity: str, prefix: str, convert_to_masked_nullable: bool) -> None:
        result, _ = lib.maybe_convert_numeric(
            np.array([prefix + infinity], dtype=object),
            na_values={'', 'NULL', 'nan'},
            coerce_numeric=coerce_numeric,
            convert_to_masked_nullable=convert_to_masked_nullable
        )
        expected = np.array([np.inf if prefix in ['', '+'] else -np.inf])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_m