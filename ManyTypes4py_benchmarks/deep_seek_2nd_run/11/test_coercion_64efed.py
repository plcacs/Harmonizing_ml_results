from __future__ import annotations
from datetime import datetime, timedelta
import itertools
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import pytest
from pandas.compat import IS64, is_platform_windows
from pandas.compat.numpy import np_version_gt2
import pandas as pd
from pandas import Index, Series, Timestamp, Timedelta, Period, DatetimeIndex, TimedeltaIndex, PeriodIndex
from pandas._testing import tm
from pandas.core.dtypes.dtypes import DatetimeTZDtype, IntervalDtype, PeriodDtype

T = TypeVar('T', bound='CoercionBase')

@pytest.fixture(autouse=True, scope='class')
def check_comprehensiveness(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    cls = request.cls
    combos = itertools.product(cls.klasses, cls.dtypes, [cls.method])

    def has_test(combo: Tuple[str, str, str]) -> bool:
        klass, dtype, method = combo
        cls_funcs = request.node.session.items
        return any((klass in x.name and dtype in x.name and (method in x.name) for x in cls_funcs))
    opts = request.config.option
    if opts.lf or opts.keyword:
        yield
    else:
        for combo in combos:
            if not has_test(combo):
                raise AssertionError(f'test method is not defined: {cls.__name__}, {combo}')
        yield

class CoercionBase:
    klasses: List[str] = ['index', 'series']
    dtypes: List[str] = ['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64', 'datetime64tz', 'timedelta64', 'period']

    @property
    def method(self) -> str:
        raise NotImplementedError(self)

class TestSetitemCoercion(CoercionBase):
    method: str = 'setitem'
    klasses: List[str] = []

    def test_setitem_series_no_coercion_from_values_list(self) -> None:
        ser = pd.Series(['a', 1])
        ser[:] = list(ser.values)
        expected = pd.Series(['a', 1])
        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(self, original_series: Series, loc_key: Any, expected_index: Index, expected_dtype: Any) -> None:
        """test index's coercion triggered by assign key"""
        temp = original_series.copy()
        temp[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        assert temp.index.dtype == expected_dtype
        temp = original_series.copy()
        temp.loc[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize('val,exp_dtype', [('x', object), (5, IndexError), (1.1, object)])
    def test_setitem_index_object(self, val: Any, exp_dtype: Any) -> None:
        obj = pd.Series([1, 2, 3, 4], index=pd.Index(list('abcd'), dtype=object))
        assert obj.index.dtype == object
        exp_index = pd.Index(list('abcd') + [val], dtype=object)
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize('val,exp_dtype', [(5, np.int64), (1.1, np.float64), ('x', object)])
    def test_setitem_index_int64(self, val: Any, exp_dtype: Any) -> None:
        obj = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64
        exp_index = pd.Index([0, 1, 2, 3, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize('val,exp_dtype', [(5, np.float64), (5.1, np.float64), ('x', object)])
    def test_setitem_index_float64(self, val: Any, exp_dtype: Any, request: pytest.FixtureRequest) -> None:
        obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64
        exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_series_period(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_complex128(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_bool(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_datetime64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_datetime64tz(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_timedelta64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_period(self) -> None:
        raise NotImplementedError

class TestInsertIndexCoercion(CoercionBase):
    klasses: List[str] = ['index']
    method: str = 'insert'

    def _assert_insert_conversion(self, original: Index, value: Any, expected: Index, expected_dtype: Any) -> None:
        """test coercion triggered by insert"""
        target = original.copy()
        res = target.insert(1, value)
        tm.assert_index_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, object), (1.1, 1.1, object), (False, False, object), ('x', 'x', object)])
    def test_insert_index_object(self, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        obj = pd.Index(list('abcd'), dtype=object)
        assert obj.dtype == object
        exp = pd.Index(['a', coerced_val, 'b', 'c', 'd'], dtype=object)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, None), (1.1, 1.1, np.float64), (False, False, object), ('x', 'x', object)])
    def test_insert_int_index(self, any_int_numpy_dtype: Any, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        dtype = any_int_numpy_dtype
        obj = pd.Index([1, 2, 3, 4], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        exp = pd.Index([1, coerced_val, 2, 3, 4], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1.0, None), (1.1, 1.1, np.float64), (False, False, object), ('x', 'x', object)])
    def test_insert_float_index(self, float_numpy_dtype: Any, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        dtype = float_numpy_dtype
        obj = pd.Index([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        if np_version_gt2 and dtype == 'float32' and (coerced_val == 1.1):
            coerced_dtype = np.float32
        exp = pd.Index([1.0, coerced_val, 2.0, 3.0, 4.0], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'), (pd.Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]')], ids=['datetime64', 'datetime64tz'])
    @pytest.mark.parametrize('insert_value', [pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-01', tz='Asia/Tokyo'), 1])
    def test_insert_index_datetimes(self, fill_val: Timestamp, exp_dtype: str, insert_value: Any) -> None:
        obj = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04'], tz=fill_val.tz).as_unit('ns')
        assert obj.dtype == exp_dtype
        exp = pd.DatetimeIndex(['2011-01-01', fill_val.date(), '2011-01-02', '2011-01-03', '2011-01-04'], tz=fill_val.tz).as_unit('ns')
        self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)
        if fill_val.tz:
            ts = pd.Timestamp('2012-01-01')
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)
            ts = pd.Timestamp('2012-01-01', tz='Asia/Tokyo')
            result = obj.insert(1, ts)
            expected = obj.insert(1, ts.tz_convert(obj.dtype.tz))
            assert expected.dtype == obj.dtype
            tm.assert_index_equal(result, expected)
        else:
            ts = pd.Timestamp('2012-01-01', tz='Asia/Tokyo')
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)
        item = 1
        result = obj.insert(1, item)
        expected = obj.astype(object).insert(1, item)
        assert expected[1] == item
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)

    def test_insert_index_timedelta64(self) -> None:
        obj = pd.TimedeltaIndex(['1 day', '2 day', '3 day', '4 day'])
        assert obj.dtype == 'timedelta64[ns]'
        exp = pd.TimedeltaIndex(['1 day', '10 day', '2 day', '3 day', '4 day'])
        self._assert_insert_conversion(obj, pd.Timedelta('10 day'), exp, 'timedelta64[ns]')
        for item in [pd.Timestamp('2012-01-01'), 1]:
            result = obj.insert(1, item)
            expected = obj.astype(object).insert(1, item)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(pd.Period('2012-01', freq='M'), '2012-01', 'period[M]'), (pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-01'), object), (1, 1, object), ('x', 'x', object)])
    def test_insert_index_period(self, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        obj = pd.PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M')
        assert obj.dtype == 'period[M]'
        data = [pd.Period('2011-01', freq='M'), coerced_val, pd.Period('2011-02', freq='M'), pd.Period('2011-03', freq='M'), pd.Period('2011-04', freq='M')]
        if isinstance(insert, pd.Period):
            exp = pd.PeriodIndex(data, freq='M')
            self._assert_insert_conversion(obj, insert, exp, coerced_dtype)
            self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)
        else:
            result = obj.insert(0, insert)
            expected = obj.astype(object).insert(0, insert)
            tm.assert_index_equal(result, expected)
            if not isinstance(insert, pd.Timestamp):
                result = obj.insert(0, str(insert))
                expected = obj.astype(object).insert(0, str(insert))
                tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_insert_index_complex128(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_insert_index_bool(self) -> None:
        raise NotImplementedError

class TestWhereCoercion(CoercionBase):
    method: str = 'where'
    _cond: np.ndarray = np.array([True, False, True, False])

    def _assert_where_conversion(self, original: Union[Series, Index], cond: Union[Series, Index, np.ndarray], values: Any, expected: Union[Series, Index], expected_dtype: Any) -> None:
        """test coercion triggered by where"""
        target = original.copy()
        res = target.where(cond, values)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    def _construct_exp(self, obj: Union[Series, Index], klass: Type[Union[Series, Index]], fill_val: Any, exp_dtype: Any) -> Tuple[Union[Series, Index], Union[Series, Index]]:
        if fill_val is True:
            values = klass([True, False, True, True])
        elif isinstance(fill_val, (datetime, np.datetime64)):
            values = pd.date_range(fill_val, periods=4)
        else:
            values = klass((x * fill_val for x in [5, 6, 7, 8]))
        exp = klass([obj[0], values[1], obj[2], values[3]], dtype=exp_dtype)
        return (values, exp)

    def _run_test(self, obj: Union[Series, Index], fill_val: Any, klass: Type[Union[Series, Index]], exp_dtype: Any) -> None:
        cond = klass(self._cond)
        exp = klass([obj[0], fill_val, obj[2], fill_val], dtype=exp_dtype)
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)
        values, exp = self._construct_exp(obj, klass, fill_val, exp_dtype)
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, object)])
    def test_where_object(self, index_or_series: Type[Union[Series, Index]], fill_val: Any, exp_dtype: Any) -> None:
        klass = index_or_series
        obj = klass(list('abcd'), dtype=object)
        assert obj.dtype == object
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_where_int64(self, index_or_series: Type[Union[Series, Index]], fill_val: Any, exp_dtype: Any, request: pytest.FixtureRequest) -> None:
        klass = index_or_series
        obj = klass([1, 2, 3, 4])
        assert obj.dtype == np.int64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val, exp_dtype', [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_where_float64(self, index_or_series: Type[Union[Series, Index]], fill_val: Any, exp_dtype: Any, request: pytest.FixtureRequest) -> None:
        klass = index_or_series
        obj = klass([1.1, 2.2, 3.3, 4.4])
        assert obj.dtype == np.float64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, np.com