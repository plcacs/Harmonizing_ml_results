from __future__ import annotations
from datetime import datetime, timedelta
import itertools
import numpy as np
import pytest
from pandas.compat import IS64, is_platform_windows
from pandas.compat.numpy import np_version_gt2
import pandas as pd
from pandas._testing import tm
from typing import Any, Dict, Union

class CoercionBase:
    klasses: list[str] = ['index', 'series']
    dtypes: list[str] = ['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64', 'datetime64tz', 'timedelta64', 'period']

    @property
    def method(self) -> Any:
        raise NotImplementedError(self)

class TestSetitemCoercion(CoercionBase):
    method: str = 'setitem'
    klasses: list[str] = []

    def test_setitem_series_no_coercion_from_values_list(self) -> None:
        ...

    def _assert_setitem_index_conversion(self, original_series: pd.Series, loc_key: Any, expected_index: pd.Index, expected_dtype: Any) -> None:
        ...

    def test_setitem_index_object(self, val: Any, exp_dtype: Any) -> None:
        ...

    def test_setitem_index_int64(self, val: Any, exp_dtype: Any) -> None:
        ...

    def test_setitem_index_float64(self, val: Any, exp_dtype: Any, request: Any) -> None:
        ...

    def test_setitem_series_period(self) -> None:
        ...

    def test_setitem_index_complex128(self) -> None:
        ...

    def test_setitem_index_bool(self) -> None:
        ...

    def test_setitem_index_datetime64(self) -> None:
        ...

    def test_setitem_index_datetime64tz(self) -> None:
        ...

    def test_setitem_index_timedelta64(self) -> None:
        ...

    def test_setitem_index_period(self) -> None:
        ...

class TestInsertIndexCoercion(CoercionBase):
    klasses: list[str] = ['index']
    method: str = 'insert'

    def _assert_insert_conversion(self, original: pd.Index, value: Any, expected: pd.Index, expected_dtype: Any) -> None:
        ...

    def test_insert_index_object(self, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        ...

    def test_insert_int_index(self, any_int_numpy_dtype: Any, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        ...

    def test_insert_float_index(self, float_numpy_dtype: Any, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        ...

    def test_insert_index_complex128(self) -> None:
        ...

    def test_insert_index_bool(self) -> None:
        ...

    def test_insert_index_datetime64(self) -> None:
        ...

    def test_insert_index_datetime64tz(self) -> None:
        ...

    def test_insert_index_timedelta64(self) -> None:
        ...

    def test_insert_index_period(self) -> None:
        ...

class TestWhereCoercion(CoercionBase):
    method: str = 'where'
    _cond: np.ndarray = np.array([True, False, True, False])

    def _assert_where_conversion(self, original: pd.Series, cond: pd.Series, values: Any, expected: pd.Series, expected_dtype: Any) -> None:
        ...

    def _construct_exp(self, obj: pd.Series, klass: Any, fill_val: Any, exp_dtype: Any) -> tuple[pd.Series, pd.Series]:
        ...

    def _run_test(self, obj: pd.Series, fill_val: Any, klass: Any, exp_dtype: Any) -> None:
        ...

    def test_where_object(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_where_int64(self, index_or_series: Any, fill_val: Any, exp_dtype: Any, request: Any) -> None:
        ...

    def test_where_float64(self, index_or_series: Any, fill_val: Any, exp_dtype: Any, request: Any) -> None:
        ...

    def test_where_complex128(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_where_series_bool(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_where_datetime64(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_where_index_complex128(self) -> None:
        ...

    def test_where_index_bool(self) -> None:
        ...

    def test_where_series_timedelta64(self) -> None:
        ...

    def test_where_series_period(self) -> None:
        ...

class TestFillnaSeriesCoercion(CoercionBase):
    method: str = 'fillna'

    def test_has_comprehensive_tests(self) -> None:
        ...

    def _assert_fillna_conversion(self, original: pd.Series, value: Any, expected: pd.Series, expected_dtype: Any) -> None:
        ...

    def test_fillna_object(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_fillna_float64(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_fillna_complex128(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_fillna_datetime(self, index_or_series: Any, fill_val: Any, exp_dtype: Any) -> None:
        ...

    def test_fillna_interval(self, index_or_series: Any, fill_val: Any) -> None:
        ...

    def test_fillna_series_int64(self) -> None:
        ...

    def test_fillna_index_int64(self) -> None:
        ...

    def test_fillna_series_bool(self) -> None:
        ...

    def test_fillna_index_bool(self) -> None:
        ...

    def test_fillna_series_timedelta64(self) -> None:
        ...

    def test_fillna_index_timedelta64(self) -> None:
        ...

    def test_fillna_series_period(self, index_or_series: Any, fill_val: Any) -> None:
        ...

    def test_fillna_index_period(self) -> None:
        ...

class TestReplaceSeriesCoercion(CoercionBase):
    klasses: list[str] = ['series']
    method: str = 'replace'
    rep: Dict[str, list[Union[str, int, float, complex, bool, datetime, timedelta]]] = {}
    rep['object'] = ['a', 'b']
    rep['int64'] = [4, 5]
    rep['float64'] = [1.1, 2.2]
    rep['complex128'] = [1 + 1j, 2 + 2j]
    rep['bool'] = [True, False]
    rep['datetime64[ns]'] = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-03')]
    for tz in ['UTC', 'US/Eastern']:
        key = f'datetime64[ns, {tz}]'
        rep[key] = [pd.Timestamp('2011-01-01', tz=tz), pd.Timestamp('2011-01-03', tz=tz)]
    rep['timedelta64[ns]'] = [pd.Timedelta('1 day'), pd.Timedelta('2 day')]

    def test_replace_series(self, how: str, to_key: str, from_key: str, replacer: Union[Dict[Any, Any], pd.Series]) -> None:
        ...

    def test_replace_series_datetime_tz(self, how: str, to_key: str, from_key: str, replacer: Union[Dict[Any, Any], pd.Series], using_infer_string: Any) -> None:
        ...

    def test_replace_series_datetime_datetime(self, how: str, to_key: str, from_key: str, replacer: Union[Dict[Any, Any], pd.Series]) -> None:
        ...

    def test_replace_series_period(self) -> None:
        ...
