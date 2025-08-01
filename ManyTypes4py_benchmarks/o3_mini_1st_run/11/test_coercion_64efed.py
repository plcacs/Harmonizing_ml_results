from __future__ import annotations
from datetime import datetime, timedelta
import itertools
from typing import Any, Generator, Type, Union
import numpy as np
import pytest
from pandas.compat import IS64, is_platform_windows
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm


@pytest.fixture(autouse=True, scope='class')
def check_comprehensiveness(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    cls = request.cls
    combos = itertools.product(cls.klasses, cls.dtypes, [cls.method])

    def has_test(combo: tuple[Any, Any, Any]) -> bool:
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
    klasses: list[str] = ['index', 'series']
    dtypes: list[str] = ['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64', 'datetime64tz', 'timedelta64', 'period']

    @property
    def method(self) -> Any:
        raise NotImplementedError(self)


class TestSetitemCoercion(CoercionBase):
    method: str = 'setitem'
    klasses: list[str] = []

    def test_setitem_series_no_coercion_from_values_list(self) -> None:
        ser: pd.Series = pd.Series(['a', 1])
        ser[:] = list(ser.values)
        expected: pd.Series = pd.Series(['a', 1])
        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(self, original_series: pd.Series, loc_key: Any,
                                           expected_index: pd.Index, expected_dtype: Any) -> None:
        """test index's coercion triggered by assign key"""
        temp: pd.Series = original_series.copy()
        temp[loc_key] = 5
        exp: pd.Series = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        assert temp.index.dtype == expected_dtype
        temp = original_series.copy()
        temp.loc[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize('val,exp_dtype', [('x', object), (5, IndexError), (1.1, object)])
    def test_setitem_index_object(self, val: Any, exp_dtype: Any) -> None:
        obj: pd.Series = pd.Series([1, 2, 3, 4], index=pd.Index(list('abcd'), dtype=object))
        assert obj.index.dtype == object
        exp_index: pd.Index = pd.Index(list('abcd') + [val], dtype=object)
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize('val,exp_dtype', [(5, np.int64), (1.1, np.float64), ('x', object)])
    def test_setitem_index_int64(self, val: Any, exp_dtype: Any) -> None:
        obj: pd.Series = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64
        exp_index: pd.Index = pd.Index([0, 1, 2, 3, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize('val,exp_dtype', [(5, np.float64), (5.1, np.float64), ('x', object)])
    def test_setitem_index_float64(self, val: Any, exp_dtype: Any, request: Any) -> None:
        obj: pd.Series = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64
        exp_index: pd.Index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
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
    klasses: list[str] = ['index']
    method: str = 'insert'

    def _assert_insert_conversion(self, original: pd.Index, value: Any,
                                  expected: pd.Index, expected_dtype: Any) -> None:
        """test coercion triggered by insert"""
        target: pd.Index = original.copy()
        res: pd.Index = target.insert(1, value)
        tm.assert_index_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, object), (1.1, 1.1, object),
                                                                      (False, False, object), ('x', 'x', object)])
    def test_insert_index_object(self, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        obj: pd.Index = pd.Index(list('abcd'), dtype=object)
        assert obj.dtype == object
        exp: pd.Index = pd.Index(['a', coerced_val, 'b', 'c', 'd'], dtype=object)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, None), (1.1, 1.1, np.float64),
                                                                      (False, False, object), ('x', 'x', object)])
    def test_insert_int_index(self, any_int_numpy_dtype: Any, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        dtype: Any = any_int_numpy_dtype
        obj: pd.Index = pd.Index([1, 2, 3, 4], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        exp: pd.Index = pd.Index([1, coerced_val, 2, 3, 4], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1.0, None), (1.1, 1.1, np.float64),
                                                                      (False, False, object), ('x', 'x', object)])
    def test_insert_float_index(self, float_numpy_dtype: Any, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        dtype: Any = float_numpy_dtype
        obj: pd.Index = pd.Index([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        if np_version_gt2 and dtype == 'float32' and (coerced_val == 1.1):
            coerced_dtype = np.float32
        exp: pd.Index = pd.Index([1.0, coerced_val, 2.0, 3.0, 4.0], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'),
                                                      (pd.Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]')],
                             ids=['datetime64', 'datetime64tz'])
    @pytest.mark.parametrize('insert_value', [pd.Timestamp('2012-01-01'),
                                                pd.Timestamp('2012-01-01', tz='Asia/Tokyo'), 1])
    def test_insert_index_datetimes(self, fill_val: pd.Timestamp, exp_dtype: Any,
                                    insert_value: Any) -> None:
        obj: pd.DatetimeIndex = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04'],
                                                  tz=fill_val.tz).as_unit('ns')
        assert obj.dtype == exp_dtype
        exp: pd.DatetimeIndex = pd.DatetimeIndex(['2011-01-01', fill_val.date(), '2011-01-02', '2011-01-03', '2011-01-04'],
                                                  tz=fill_val.tz).as_unit('ns')
        self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)
        if fill_val.tz:
            ts: pd.Timestamp = pd.Timestamp('2012-01-01')
            result: pd.Index = obj.insert(1, ts)
            expected: pd.Index = obj.astype(object).insert(1, ts)
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
        item: Any = 1
        result = obj.insert(1, item)
        expected = obj.astype(object).insert(1, item)
        assert expected[1] == item
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)

    def test_insert_index_timedelta64(self) -> None:
        obj: pd.TimedeltaIndex = pd.TimedeltaIndex(['1 day', '2 day', '3 day', '4 day'])
        assert obj.dtype == 'timedelta64[ns]'
        exp: pd.TimedeltaIndex = pd.TimedeltaIndex(['1 day', '10 day', '2 day', '3 day', '4 day'])
        self._assert_insert_conversion(obj, pd.Timedelta('10 day'), exp, 'timedelta64[ns]')
        for item in [pd.Timestamp('2012-01-01'), 1]:
            result: pd.Index = obj.insert(1, item)
            expected: pd.Index = obj.astype(object).insert(1, item)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [
        (pd.Period('2012-01', freq='M'), '2012-01', 'period[M]'),
        (pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-01'), object),
        (1, 1, object),
        ('x', 'x', object)
    ])
    def test_insert_index_period(self, insert: Any, coerced_val: Any, coerced_dtype: Any) -> None:
        obj: pd.PeriodIndex = pd.PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M')
        assert obj.dtype == 'period[M]'
        data = [pd.Period('2011-01', freq='M'), coerced_val, pd.Period('2011-02', freq='M'),
                pd.Period('2011-03', freq='M'), pd.Period('2011-04', freq='M')]
        if isinstance(insert, pd.Period):
            exp: pd.PeriodIndex = pd.PeriodIndex(data, freq='M')
            self._assert_insert_conversion(obj, insert, exp, coerced_dtype)
            self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)
        else:
            result: pd.Index = obj.insert(0, insert)
            expected: pd.Index = obj.astype(object).insert(0, insert)
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

    def _assert_where_conversion(self, original: Union[pd.Index, pd.Series], cond: Union[pd.Index, np.ndarray],
                                   values: Any, expected: Union[pd.Index, pd.Series], expected_dtype: Any) -> None:
        """test coercion triggered by where"""
        target: Union[pd.Index, pd.Series] = original.copy()
        res: Union[pd.Index, pd.Series] = target.where(cond, values)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    def _construct_exp(self, obj: Union[pd.Index, pd.Series], klass: Type[Union[pd.Index, pd.Series]],
                       fill_val: Any, exp_dtype: Any) -> tuple[Union[pd.Index, pd.Series], Union[pd.Index, pd.Series]]:
        if fill_val is True:
            values = klass([True, False, True, True])
        elif isinstance(fill_val, (datetime, np.datetime64)):
            values = pd.date_range(str(fill_val), periods=4)
        else:
            values = klass((x * fill_val for x in [5, 6, 7, 8]))
        exp = klass([obj[0], values[1], obj[2], values[3]], dtype=exp_dtype)
        return values, exp

    def _run_test(self, obj: Union[pd.Index, pd.Series], fill_val: Any,
                  klass: Type[Union[pd.Index, pd.Series]], exp_dtype: Any) -> None:
        cond = klass(self._cond)
        exp = klass([obj[0], fill_val, obj[2], fill_val], dtype=exp_dtype)
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)
        values, exp = self._construct_exp(obj, klass, fill_val, exp_dtype)
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, object)])
    def test_where_object(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any, exp_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass(list('abcd'), dtype=object)
        assert obj.dtype == object
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_where_int64(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any,
                         exp_dtype: Any, request: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass([1, 2, 3, 4])
        assert obj.dtype == np.int64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val, exp_dtype', [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_where_float64(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any,
                           exp_dtype: Any, request: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass([1.1, 2.2, 3.3, 4.4])
        assert obj.dtype == np.float64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, np.complex128), (1.1, np.complex128), (1 + 1j, np.complex128), (True, object)])
    def test_where_complex128(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any,
                              exp_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, np.bool_)])
    def test_where_series_bool(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any,
                               exp_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass([True, False, True, False])
        assert obj.dtype == np.bool_
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'),
                                                      (pd.Timestamp('2012-01-01', tz='US/Eastern'), object)],
                             ids=['datetime64', 'datetime64tz'])
    def test_where_datetime64(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any,
                              exp_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass(pd.date_range('2011-01-01', periods=4, freq='D')._with_freq(None))
        assert obj.dtype == 'datetime64[ns]'
        fv: Any = fill_val
        if exp_dtype == 'datetime64[ns]':
            for scalar in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
                self._run_test(obj, scalar, klass, exp_dtype)
        else:
            for scalar in [fv, fv.to_pydatetime()]:
                self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_index_complex128(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_index_bool(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_series_timedelta64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_series_period(self) -> None:
        raise NotImplementedError

    @pytest.mark.parametrize('value', [pd.Timedelta(days=9), timedelta(days=9), np.timedelta64(9, 'D')])
    def test_where_index_timedelta64(self, value: Any) -> None:
        tdi: pd.TimedeltaIndex = pd.timedelta_range('1 Day', periods=4)
        cond: np.ndarray = np.array([True, False, False, True])
        expected: pd.TimedeltaIndex = pd.TimedeltaIndex(['1 Day', value, value, '4 Days'])
        result: pd.TimedeltaIndex = tdi.where(cond, value)
        tm.assert_index_equal(result, expected)
        dtnat: np.datetime64 = np.datetime64('NaT', 'ns')
        expected = pd.Index([tdi[0], dtnat, dtnat, tdi[3]], dtype=object)
        assert expected[1] is dtnat
        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)

    def test_where_index_period(self) -> None:
        dti: pd.DatetimeIndex = pd.date_range('2016-01-01', periods=3, freq='QS')
        pi: pd.PeriodIndex = dti.to_period('Q')
        cond: np.ndarray = np.array([False, True, False])
        value: pd.Period = pi[-1] + pi.freq * 10
        expected: pd.PeriodIndex = pd.PeriodIndex([value, pi[1], value])
        result: pd.PeriodIndex = pi.where(cond, value)
        tm.assert_index_equal(result, expected)
        other = np.asarray(pi + pi.freq * 10, dtype=object)
        result = pi.where(cond, other)
        expected = pd.PeriodIndex([other[0], pi[1], other[2]])
        tm.assert_index_equal(result, expected)
        td: pd.Timedelta = pd.Timedelta(days=4)
        expected = pd.Index([td, pi[1], td], dtype=object)
        result = pi.where(cond, td)
        tm.assert_index_equal(result, expected)
        per: pd.Period = pd.Period('2020-04-21', 'D')
        expected = pd.Index([per, pi[1], per], dtype=object)
        result = pi.where(cond, per)
        tm.assert_index_equal(result, expected)


class TestFillnaSeriesCoercion(CoercionBase):
    method: str = 'fillna'

    @pytest.mark.xfail(reason='Test not implemented')
    def test_has_comprehensive_tests(self) -> None:
        raise NotImplementedError

    def _assert_fillna_conversion(self, original: Union[pd.Index, pd.Series],
                                  value: Any, expected: Union[pd.Index, pd.Series],
                                  expected_dtype: Any) -> None:
        """test coercion triggered by fillna"""
        target: Union[pd.Index, pd.Series] = original.copy()
        res: Union[pd.Index, pd.Series] = target.fillna(value)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize('fill_val, fill_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, object)])
    def test_fillna_object(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any, fill_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass(['a', np.nan, 'c', 'd'], dtype=object)
        assert obj.dtype == object
        exp: Union[pd.Index, pd.Series] = klass(['a', fill_val, 'c', 'd'], dtype=object)
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_fillna_float64(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any, fill_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass([1.1, np.nan, 3.3, 4.4])
        assert obj.dtype == np.float64
        exp: Union[pd.Index, pd.Series] = klass([1.1, fill_val, 3.3, 4.4])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [(1, np.complex128), (1.1, np.complex128), (1 + 1j, np.complex128), (True, object)])
    def test_fillna_complex128(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any, fill_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass([1 + 1j, np.nan, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128
        exp: Union[pd.Index, pd.Series] = klass([1 + 1j, fill_val, 3 + 3j, 4 + 4j])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[s]'),
                                                       (pd.Timestamp('2012-01-01', tz='US/Eastern'), object),
                                                       (1, object), ('x', object)],
                             ids=['datetime64', 'datetime64tz', 'object', 'object'])
    def test_fillna_datetime(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any,
                             fill_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        obj: Union[pd.Index, pd.Series] = klass([pd.Timestamp('2011-01-01'), pd.NaT, pd.Timestamp('2011-01-03'), pd.Timestamp('2011-01-04')])
        assert obj.dtype == 'datetime64[s]'
        exp: Union[pd.Index, pd.Series] = klass([pd.Timestamp('2011-01-01'), fill_val, pd.Timestamp('2011-01-03'), pd.Timestamp('2011-01-04')])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val,fill_dtype', [
        (pd.Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[s, US/Eastern]'),
        (pd.Timestamp('2012-01-01'), object),
        (pd.Timestamp('2012-01-01', tz='Asia/Tokyo'), 'datetime64[s, US/Eastern]'),
        (1, object),
        ('x', object)
    ])
    def test_fillna_datetime64tz(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any,
                                 fill_dtype: Any) -> None:
        klass: Type[Union[pd.Index, pd.Series]] = index_or_series
        tz: str = 'US/Eastern'
        obj: Union[pd.Index, pd.Series] = klass([pd.Timestamp('2011-01-01', tz=tz),
                                                  pd.NaT,
                                                  pd.Timestamp('2011-01-03', tz=tz),
                                                  pd.Timestamp('2011-01-04', tz=tz)])
        assert obj.dtype == 'datetime64[s, US/Eastern]'
        if getattr(fill_val, 'tz', None) is None:
            fv: Any = fill_val
        else:
            fv = fill_val.tz_convert(tz)
        exp: Union[pd.Index, pd.Series] = klass([pd.Timestamp('2011-01-01', tz=tz),
                                                  fv,
                                                  pd.Timestamp('2011-01-03', tz=tz),
                                                  pd.Timestamp('2011-01-04', tz=tz)])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize('fill_val', [1, 1.1, 1 + 1j, True, pd.Interval(1, 2, closed='left'),
                                            pd.Timestamp('2012-01-01', tz='US/Eastern'), pd.Timestamp('2012-01-01'),
                                            pd.Timedelta(days=1), pd.Period('2016-01-01', 'D')])
    def test_fillna_interval(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any) -> None:
        ii: pd.IntervalIndex = pd.interval_range(1.0, 5.0, closed='right').insert(1, np.nan)
        assert isinstance(ii.dtype, pd.IntervalDtype)
        obj: Union[pd.Index, pd.Series] = index_or_series(ii)
        exp: Union[pd.Index, pd.Series] = index_or_series([ii[0], fill_val, ii[2], ii[3], ii[4]],)
        fill_dtype: Any = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_series_int64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_int64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_series_bool(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_bool(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_series_timedelta64(self) -> None:
        raise NotImplementedError

    @pytest.mark.parametrize('fill_val', [1, 1.1, 1 + 1j, True, pd.Interval(1, 2, closed='left'),
                                            pd.Timestamp('2012-01-01', tz='US/Eastern'), pd.Timestamp('2012-01-01'),
                                            pd.Timedelta(days=1), pd.Period('2016-01-01', 'W')])
    def test_fillna_series_period(self, index_or_series: Type[Union[pd.Index, pd.Series]], fill_val: Any) -> None:
        pi: pd.PeriodIndex = pd.period_range('2016-01-01', periods=4, freq='D').insert(1, pd.NaT)
        assert isinstance(pi.dtype, pd.PeriodDtype)
        obj: Union[pd.Index, pd.Series] = index_or_series(pi)
        exp: Union[pd.Index, pd.Series] = index_or_series([pi[0], fill_val, pi[2], pi[3], pi[4]], dtype=object)
        fill_dtype: Any = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_timedelta64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_fillna_index_period(self) -> None:
        raise NotImplementedError


class TestReplaceSeriesCoercion(CoercionBase):
    klasses: list[str] = ['series']
    method: str = 'replace'
    rep: dict[str, list[Any]] = {}
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

    @pytest.fixture(params=['dict', 'series'])
    def how(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture(params=['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64[ns]', 
                              'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]', 'timedelta64[ns]'])
    def from_key(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture(params=['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64[ns]', 
                              'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]', 'timedelta64[ns]'],
                    ids=['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64', 'datetime64tz', 'datetime64tz', 'timedelta64'])
    def to_key(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def replacer(self, how: str, from_key: str, to_key: str) -> Any:
        """
        Object we will pass to `Series.replace`
        """
        if how == 'dict':
            replacer_obj = dict(zip(self.rep[from_key], self.rep[to_key]))
        elif how == 'series':
            replacer_obj = pd.Series(self.rep[to_key], index=self.rep[from_key])
        else:
            raise ValueError
        return replacer_obj

    def test_replace_series(self, how: str, to_key: str, from_key: str, replacer: Any) -> None:
        index: pd.Index = pd.Index([3, 4], name='xxx')
        obj: pd.Series = pd.Series(self.rep[from_key], index=index, name='yyy')
        obj = obj.astype(from_key)
        assert obj.dtype == from_key
        if from_key.startswith('datetime') and to_key.startswith('datetime'):
            return
        elif from_key in ['datetime64[ns, US/Eastern]', 'datetime64[ns, UTC]']:
            return
        if from_key == 'float64' and to_key in 'int64' or (from_key == 'complex128' and to_key in ('int64', 'float64')):
            if not IS64 or is_platform_windows():
                pytest.skip(f'32-bit platform buggy: {from_key} -> {to_key}')
            exp: pd.Series = pd.Series(self.rep[to_key], index=index, name='yyy', dtype=from_key)
        else:
            exp = pd.Series(self.rep[to_key], index=index, name='yyy')
        result: pd.Series = obj.replace(replacer)
        tm.assert_series_equal(result, exp, check_dtype=False)

    @pytest.mark.parametrize('to_key', ['timedelta64[ns]', 'bool', 'object', 'complex128', 'float64', 'int64'], indirect=True)
    @pytest.mark.parametrize('from_key', ['datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    def test_replace_series_datetime_tz(self, how: str, to_key: str, from_key: str,
                                        replacer: Any, using_infer_string: bool) -> None:
        index: pd.Index = pd.Index([3, 4], name='xyz')
        obj: pd.Series = pd.Series(self.rep[from_key], index=index, name='yyy').dt.as_unit('ns')
        assert obj.dtype == from_key
        exp: pd.Series = pd.Series(self.rep[to_key], index=index, name='yyy')
        if using_infer_string and to_key == 'object':
            assert exp.dtype == 'string'
        else:
            assert exp.dtype == to_key
        result: pd.Series = obj.replace(replacer)
        tm.assert_series_equal(result, exp, check_dtype=False)

    @pytest.mark.parametrize('to_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    @pytest.mark.parametrize('from_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    def test_replace_series_datetime_datetime(self, how: str, to_key: str, from_key: str,
                                                replacer: Any) -> None:
        index: pd.Index = pd.Index([3, 4], name='xyz')
        obj: pd.Series = pd.Series(self.rep[from_key], index=index, name='yyy').dt.as_unit('ns')
        assert obj.dtype == from_key
        exp: pd.Series = pd.Series(self.rep[to_key], index=index, name='yyy')
        if isinstance(obj.dtype, pd.DatetimeTZDtype) and isinstance(exp.dtype, pd.DatetimeTZDtype):
            exp = exp.astype(obj.dtype)
        elif to_key == from_key:
            exp = exp.dt.as_unit('ns')
        result: pd.Series = obj.replace(replacer)
        tm.assert_series_equal(result, exp, check_dtype=False)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_replace_series_period(self) -> None:
        raise NotImplementedError
