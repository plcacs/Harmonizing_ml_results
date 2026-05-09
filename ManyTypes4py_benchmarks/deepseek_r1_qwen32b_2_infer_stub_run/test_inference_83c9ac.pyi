"""
Stub file for test_inference_83c9ac.py
"""

import pytest
import numpy as np
import pandas as pd
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from io import StringIO
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, ParamSpec

P = ParamSpec('P')

@pytest.fixture(params=[True, False], ids=str)
def coerce(request) -> bool:
    ...

@pytest.fixture(params=zip(objs, expected), ids=ids)
def maybe_list_like(request) -> Tuple[Any, bool]:
    ...

class MockNumpyLikeArray:
    def __init__(self, values: Any) -> None:
        ...
    
    def __iter__(self) -> Iterator[Any]:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __array__(self, dtype: Optional[Any] = None, copy: Optional[Any] = None) -> np.ndarray:
        ...
    
    @property
    def ndim(self) -> int:
        ...
    
    @property
    def dtype(self) -> Any:
        ...
    
    @property
    def size(self) -> int:
        ...
    
    @property
    def shape(self) -> Tuple[int, ...]:
        ...

def test_is_list_like(maybe_list_like: Tuple[Any, bool]) -> None:
    ...

def test_is_list_like_disallow_sets(maybe_list_like: Tuple[Any, bool]) -> None:
    ...

def test_is_list_like_recursion() -> None:
    ...

def test_is_list_like_iter_is_none() -> None:
    ...

def test_is_list_like_generic() -> None:
    ...

def test_is_sequence() -> None:
    ...

def test_is_array_like() -> None:
    ...

@pytest.mark.parametrize('inner', [[], [1], (1,), (1, 2), {'a': 1}, {1, 'a'}, pd.Series([1]), pd.Series([], dtype=object), pd.Series(['a']).str, (x for x in range(5))])
@pytest.mark.parametrize('outer', [list, pd.Series, np.array, tuple])
def test_is_nested_list_like_passes(inner: Any, outer: Any) -> None:
    ...

@pytest.mark.parametrize('obj', ['abc', [], [1], (1,), ['a'], 'a', {'a'}, [1, 2, 3], pd.Series([1]), pd.DataFrame({'A': [1]}), ([1, 2] for _ in range(5))])
def test_is_nested_list_like_fails(obj: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [{}, {'A': 1}, pd.Series([1]), collections.defaultdict()])
def test_is_dict_like_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', ['1', 1, [1, 2], (1, 2), range(2), pd.Index([1]), dict, collections.defaultdict, pd.Series])
def test_is_dict_like_fails(ll: Any) -> None:
    ...

@pytest.mark.parametrize('has_keys', [True, False])
@pytest.mark.parametrize('has_getitem', [True, False])
@pytest.mark.parametrize('has_contains', [True, False])
def test_is_dict_like_duck_type(has_keys: bool, has_getitem: bool, has_contains: bool) -> None:
    ...

def test_is_file_like() -> None:
    ...

@pytest.mark.parametrize('ll', [test_tuple(1, 2, 3)])
def test_is_names_tuple_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [(1, 2, 3), 'a', pd.Series({'pi': 3.14})])
def test_is_names_tuple_fails(ll: Any) -> None:
    ...

def test_is_hashable() -> None:
    ...

@pytest.mark.parametrize('ll', [re.compile('ad')])
def test_is_re_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', ['x', 2, 3, object()])
def test_is_re_fails(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', ['a', 'x', 'asdf', re.compile('adsf'), '\\u2233\\s*', re.compile('')])
def test_is_recompilable_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [1, [], object()])
def test_is_recompilable_fails(ll: Any) -> None:
    ...

class TestInference:
    @pytest.mark.parametrize('arr', [np.array(list('abc'), dtype='S1'), np.array(list('abc'), dtype='S1').astype(object), [b'a', np.nan, b'c']])
    def test_infer_dtype_bytes(self, arr: np.ndarray) -> None:
        ...
    
    @pytest.mark.parametrize('value, expected', [(float('inf'), True), (np.inf, True), (-np.inf, False), (1, False), ('a', False)])
    def test_isposinf_scalar(self, value: Any, expected: bool) -> None:
        ...
    
    @pytest.mark.parametrize('value, expected', [(float('-inf'), True), (-np.inf, True), (np.inf, False), (1, False), ('a', False)])
    def test_isneginf_scalar(self, value: Any, expected: bool) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, pd.BooleanArray(np.array([True, False], dtype='bool'), np.array([False, True]))), (False, np.array([True, np.nan], dtype='object'))])
    def test_maybe_convert_nullable_boolean(self, convert_to_masked_nullable: bool, exp: Any) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    @pytest.mark.parametrize('coerce_numeric', [True, False])
    @pytest.mark.parametrize('infinity', ['inf', 'inF', 'iNf', 'Inf', 'iNF', 'InF', 'INf', 'INF'])
    @pytest.mark.parametrize('prefix', ['', '-', '+'])
    def test_maybe_convert_numeric_infinities(self, coerce_numeric: bool, infinity: str, prefix: str, convert_to_masked_nullable: bool) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_maybe_convert_numeric_infinities_raises(self, convert_to_masked_nullable: bool) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_maybe_convert_numeric_post_floatify_nan(self, coerce: bool, convert_to_masked_nullable: bool) -> None:
        ...
    
    def test_convert_infs(self) -> None:
        ...
    
    def test_scientific_no_exponent(self) -> None:
        ...
    
    def test_convert_non_hashable(self) -> None:
        ...
    
    def test_convert_numeric_uint64(self) -> None:
        ...
    
    @pytest.mark.parametrize('arr', [np.array([2 ** 63, np.nan], dtype=object), np.array([str(2 ** 63), np.nan], dtype=object), np.array([np.nan, 2 ** 63], dtype=object), np.array([np.nan, str(2 ** 63)], dtype=object)])
    def test_convert_numeric_uint64_nan(self, coerce: bool, arr: np.ndarray) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_uint64_nan_values(self, coerce: bool, convert_to_masked_nullable: bool) -> None:
        ...
    
    @pytest.mark.parametrize('case', [np.array([2 ** 63, -1], dtype=object), np.array([str(2 ** 63), -1], dtype=object), np.array([str(2 ** 63), str(-1)], dtype=object), np.array([-1, 2 ** 63], dtype=object), np.array([-1, str(2 ** 63)], dtype=object), np.array([str(-1), str(2 ** 63)], dtype=object)])
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_int64_uint64(self, case: np.ndarray, coerce: bool, convert_to_masked_nullable: bool) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
    def test_convert_numeric_string_uint64(self, convert_to_masked_nullable: bool) -> None:
        ...
    
    @pytest.mark.parametrize('value', [-2 ** 63 - 1, 2 ** 64])
    def test_convert_int_overflow(self, value: int) -> None:
        ...
    
    @pytest.mark.parametrize('val', [None, np.nan, float('nan')])
    @pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
    def test_maybe_convert_objects_nat_inference(self, val: Any, dtype: str) -> None:
        ...
    
    @pytest.mark.parametrize('value, expected_dtype', [([2 ** 63], np.uint64), ([np.uint64(2 ** 63)], np.uint64), ([2, -1], np.int64), ([2 ** 63, -1], object), ([np.uint8(1)], np.uint8), ([np.uint16(1)], np.uint16), ([np.uint32(1)], np.uint32), ([np.uint64(1)], np.uint64), ([np.uint8(2), np.uint16(1)], np.uint16), ([np.uint32(2), np.uint16(1)], np.uint32), ([np.uint32(2), -1], object), ([np.uint32(2), 1], np.uint64), ([np.uint32(2), np.int32(1)], object)])
    def test_maybe_convert_objects_uint(self, value: Any, expected_dtype: Any) -> None:
        ...
    
    def test_maybe_convert_objects_datetime(self) -> None:
        ...
    
    def test_maybe_convert_objects_dtype_if_all_nat(self) -> None:
        ...
    
    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self) -> None:
        ...
    
    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
    def test_maybe_convert_objects_datetime_overflow_safe(self, dtype: str) -> None:
        ...
    
    def test_maybe_convert_objects_mixed_datetimes(self) -> None:
        ...
    
    def test_maybe_convert_objects_timedelta64_nat(self) -> None:
        ...
    
    @pytest.mark.parametrize('exp', [pd.IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True])), pd.IntegerArray(np.array([2, 0], dtype='int64'), np.array([False, True]))])
    def test_maybe_convert_objects_nullable_integer(self, exp: pd.IntegerArray) -> None:
        ...
    
    @pytest.mark.parametrize('dtype, val', [('int64', 1), ('uint64', np.iinfo(np.int64).max + 1)])
    def test_maybe_convert_objects_nullable_none(self, dtype: str, val: Any) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, pd.IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True]))), (False, np.array([2, np.nan], dtype='float64'))])
    def test_maybe_convert_numeric_nullable_integer(self, convert_to_masked_nullable: bool, exp: Any) -> None:
        ...
    
    @pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, pd.FloatingArray(np.array([2.0, 0.0], dtype='float64'), np.array([False, True]))), (False, np.array([2.0, np.nan], dtype='float64'))])
    def test_maybe_convert_numeric_floating_array(self, convert_to_masked_nullable: bool, exp: Any) -> None:
        ...
    
    def test_maybe_convert_objects_bool_nan(self) -> None:
        ...
    
    def test_maybe_convert_objects_nullable_boolean(self) -> None:
        ...
    
    @pytest.mark.parametrize('val', [None, np.nan])
    def test_maybe_convert_objects_nullable_boolean_na(self, val: Any) -> None:
        ...
    
    @pytest.mark.parametrize('data0', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
    @pytest.mark.parametrize('data1', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
    def test_maybe_convert_objects_itemsize(self, data0: Any, data1: Any) -> None:
        ...
    
    def test_mixed_dtypes_remain_object_array(self) -> None:
        ...
    
    @pytest.mark.parametrize('idx', [pd.IntervalIndex.from_breaks(range(5), closed='both'), pd.period_range('2016-01-01', periods=3, freq='D')])
    def test_maybe_convert_objects_ea(self, idx: Any) -> None:
        ...

class TestTypeInference:
    class Dummy:
        pass
    
    def test_inferred_dtype_fixture(self, any_skipna_inferred_dtype: Any) -> None:
        ...
    
    def test_length_zero(self, skipna: bool) -> None:
        ...
    
    def test_integers(self) -> None:
        ...
    
    @pytest.mark.parametrize('arr, skipna', [([1, 2, np.nan, np.nan, 3], False), ([1, 2, np.nan, np.nan, 3], True), ([1, 2, 3, np.int64(4), np.int32(5), np.nan], False), ([1, 2, 3, np.int64(4), np.int32(5), np.nan], True)])
    def test_integer_na(self, arr: Any, skipna: bool) -> None:
        ...
    
    def test_infer_dtype_skipna_default(self) -> None:
        ...
    
    def test_bools(self) -> None:
        ...
    
    def test_floats(self) -> None:
        ...
    
    def test_decimals(self) -> None:
        ...
    
    def test_complex(self, skipna: bool) -> None:
        ...
    
    def test_string(self) -> None:
        ...
    
    def test_unicode(self) -> None:
        ...
    
    @pytest.mark.parametrize('dtype, missing, skipna, expected', [(float, np.nan, False, 'floating'), (float, np.nan, True, 'floating'), (object, np.nan, False, 'floating'), (object, np.nan, True, 'empty'), (object, None, False, 'mixed'), (object, None, True, 'empty')])
    @pytest.mark.parametrize('box', [pd.Series, np.array])
    def test_object_empty(self, box: Any, missing: Any, dtype: Any, skipna: bool, expected: str) -> None:
        ...
    
    def test_datetime(self) -> None:
        ...
    
    def test_infer_dtype_datetime64(self) -> None:
        ...
    
    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    def test_infer_dtype_datetime64_with_na(self, na_value: Any) -> None:
        ...
    
    @pytest.mark.parametrize('arr', [np.array([np.timedelta64('nat'), np.datetime64('2011-01-02')], dtype=object), np.array([np.datetime64('2011-01-02'), np.timedelta64('nat')], dtype=object), np.array([np.datetime64('2011-01-01'), pd.Timestamp('2011-01-02')]), np.array([pd.Timestamp('2011-01-02'), np.datetime64('2011-01-01')]), np.array([np.nan, pd.Timestamp('2011-01-02'), 1.1]), np.array([np.nan, '2011-01-01', pd.Timestamp('2011-01-02')], dtype=object), np.array([np.datetime64('nat'), np.timedelta64(1, 'D')], dtype=object), np.array([np.timedelta64(1, 'D'), np.datetime64('nat')], dtype=object)])
    def test_infer_datetimelike_dtype_mixed(self, arr: np.ndarray) -> None:
        ...
    
    def test_infer_dtype_mixed_integer(self) -> None:
        ...
    
    @pytest.mark.parametrize('arr', [[pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-02')], [datetime(2011, 1, 1), datetime(2012, 2, 1)], [datetime(2011, 1, 1), pd.Timestamp('2011-01-02')]])
    def test_infer_dtype_datetime(self, arr: Any) -> None:
        ...
    
    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    @pytest.mark.parametrize('time_stamp', [pd.Timestamp('2011-01-01'), datetime(2011, 1, 1)])
    def test_infer_dtype_datetime_with_na(self, na_value: Any, time_stamp: Any) -> None:
        ...
    
    @pytest.mark.parametrize('arr', [np.array([pd.Timedelta('1 days'), pd.Timedelta('2 days')]), np.array([np.timedelta64(1, 'D'), np.timedelta64(2, 'D')], dtype=object), np.array([timedelta(1), timedelta(2)])])
    def test_infer_dtype_timedelta(self, arr: np.ndarray) -> None:
        ...
    
    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    @pytest.mark.parametrize('delta', [pd.Timedelta('1 days'), np.timedelta64(1, 'D'), timedelta(1)])
    def test_infer_dtype_timedelta_with_na(self, na_value: Any, delta: Any) -> None:
        ...
    
    def test_infer_dtype_period(self) -> None:
        ...
    
    def test_infer_dtype_period_array(self, index_or_series_or_array: Any, skipna: bool) -> None:
        ...
    
    def test_infer_dtype_period_mixed(self) -> None:
        ...
    
    @pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
    def test_infer_dtype_period_with_na(self, na_value: Any) -> None:
        ...
    
    def test_infer_dtype_all_nan_nat_like(self) -> None:
        ...
    
    def test_is_datetimelike_array_all_nan_nat_like(self) -> None:
        ...
    
    @pytest.mark.parametrize('func', ['is_datetime_array', 'is_datetime64_array', 'is_bool_array', 'is_timedelta_or_timedelta64_array', 'is_date_array', 'is_time_array', 'is_interval_array'])
    def test_other_dtypes_for_array(self, func: str) -> None:
        ...
    
    def test_date(self) -> None:
        ...
    
    @pytest.mark.parametrize('values', [[date(2020, 1, 1), pd.Timestamp('2020-01-01')], [pd.Timestamp('2020-01-01'), date(2020, 1, 1)], [date(2020, 1, 1), pd.NaT], [pd.NaT, date(2020, 1, 1)]])
    def test_infer_dtype_date_order_invariant(self, values: Any, skipna: bool) -> None:
        ...
    
    def test_is_numeric_array(self) -> None:
        ...
    
    def test_is_string_array(self) -> None:
        ...
    
    @pytest.mark.parametrize('func', ['is_bool_array', 'is_date_array', 'is_datetime_array', 'is_datetime64_array', 'is_float_array', 'is_integer_array', 'is_interval_array', 'is_string_array', 'is_time_array', 'is_timedelta_or_timedelta64_array'])
    def test_is_dtype_array_empty_obj(self, func: str) -> None:
        ...
    
    def test_to_object_array_tuples(self) -> None:
        ...
    
    def test_object(self) -> None:
        ...
    
    def test_to_object_array_width(self) -> None:
        ...
    
    def test_categorical(self) -> None:
        ...
    
    @pytest.mark.parametrize('asobject', [True, False])
    def test_interval(self, asobject: bool) -> None:
        ...
    
    @pytest.mark.parametrize('value', [pd.Timestamp(0), pd.Timedelta(0), 0, 0.0])
    def test_interval_mismatched_closed(self, value: Any) -> None:
        ...
    
    def test_interval_mismatched_subtype(self) -> None:
        ...
    
    @pytest.mark.parametrize('data', [['a', 'b', 'c'], ['a', 'b', pd.NA]])
    def test_string_dtype(self, data: Any, skipna: bool, index_or_series_or_array: Any, nullable_string_dtype: Any) -> None:
        ...
    
    @pytest.mark.parametrize('data', [[True, False, True], [True, False, pd.NA]])
    def test_boolean_dtype(self, data: Any, skipna: bool, index_or_series_or_array: Any) -> None:
        ...

class TestNumberScalar:
    def test_is_number(self) -> None:
        ...
    
    def test_is_bool(self) -> None:
        ...
    
    def test_is_integer(self) -> None:
        ...
    
    def test_is_float(self) -> None:
        ...
    
    def test_is_datetime_dtypes(self) -> None:
        ...
    
    @pytest.mark.parametrize('tz', ['US/Eastern', 'UTC'])
    def test_is_datetime_dtypes_with_tz(self, tz: str) -> None:
        ...
    
    def test_is_timedelta(self) -> None:
        ...
    
    def test_is_scalar_builtin_scalars(self) -> None:
        ...
    
    def test_is_scalar_builtin_nonscalars(self) -> None:
        ...
    
    def test_is_scalar_numpy_array_scalars(self) -> None:
        ...
    
    @pytest.mark.parametrize('zerodim', [1, 'foobar', np.datetime64('2014-01-01'), np.timedelta64(1, 'h'), np.datetime64('NaT')])
    def test_is_scalar_numpy_zerodim_arrays(self, zerodim: Any) -> None:
        ...
    
    @pytest.mark.parametrize('arr', [np.array([]), np.array([[]])])
    def test_is_scalar_numpy_arrays(self, arr: np.ndarray) -> None:
        ...
    
    def test_is_scalar_pandas_scalars(self) -> None:
        ...
    
    def test_is_scalar_pandas_containers(self) -> None:
        ...
    
    def test_is_scalar_number(self) -> None:
        ...

@pytest.mark.parametrize('unit', ['ms', 'us', 'ns'])
def test_datetimeindex_from_empty_datetime64_array(unit: str) -> None:
    ...

def test_nan_to_nat_conversions() -> None:
    ...

@pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
@pytest.mark.parametrize('spmatrix', ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'])
def test_is_scipy_sparse(spmatrix: str) -> None:
    ...

def test_ensure_int32() -> None:
    ...

@pytest.mark.parametrize('right,result', [(0, np.uint8), (-1, np.int16), (300, np.uint16), (300.0, np.uint16), (300.1, np.float64), (np.int16(300), np.int16 if np_version_gt2 else np.uint16)])
def test_find_result_type_uint_int(right: Any, result: Any) -> None:
    ...

@pytest.mark.parametrize('right,result', [(300.0, np.float64), (np.float32(300), np.float32)])
def test_find_result_type_floats(right: Any, result: Any) -> None:
    ...