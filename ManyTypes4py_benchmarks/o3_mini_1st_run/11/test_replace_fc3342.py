#!/usr/bin/env python3
from typing import Any, List, Union
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray

class TestSeriesReplace:

    def test_replace_explicit_none(self) -> None:
        ser: pd.Series = pd.Series([0, 0, ''], dtype=object)
        result: pd.Series = ser.replace('', None)
        expected: pd.Series = pd.Series([0, 0, None], dtype=object)
        tm.assert_series_equal(result, expected)
        df: pd.DataFrame = pd.DataFrame(np.zeros((3, 3))).astype({2: object})
        df.iloc[2, 2] = ''
        result = df.replace('', None)
        expected = pd.DataFrame({0: np.zeros(3),
                                 1: np.zeros(3),
                                 2: np.array([0.0, 0.0, None], dtype=object)})
        assert expected.iloc[2, 2] is None
        tm.assert_frame_equal(result, expected)
        ser = pd.Series([10, 20, 30, 'a', 'a', 'b', 'a'])
        result = ser.replace('a', None)
        expected = pd.Series([10, 20, 30, None, None, 'b', None])
        assert expected.iloc[-1] is None
        tm.assert_series_equal(result, expected)

    def test_replace_noop_doesnt_downcast(self) -> None:
        ser: pd.Series = pd.Series([None, None, pd.Timestamp('2021-12-16 17:31')], dtype=object)
        res: pd.Series = ser.replace({np.nan: None})
        tm.assert_series_equal(res, ser)
        assert res.dtype == object
        res = ser.replace(np.nan, None)
        tm.assert_series_equal(res, ser)
        assert res.dtype == object

    def test_replace(self) -> None:
        N: int = 50
        ser: pd.Series = pd.Series(np.random.default_rng(2).standard_normal(N))
        ser[0:4] = np.nan
        ser[6:10] = 0
        return_value = ser.replace([np.nan], -1, inplace=True)
        assert return_value is None
        exp: pd.Series = ser.fillna(-1)
        tm.assert_series_equal(ser, exp)
        rs: pd.Series = ser.replace(0.0, np.nan)
        ser[ser == 0.0] = np.nan
        tm.assert_series_equal(rs, ser)
        ser = pd.Series(np.fabs(np.random.default_rng(2).standard_normal(N)),
                        pd.date_range('2020-01-01', periods=N), dtype=object)
        ser[:5] = np.nan
        ser[6:10] = 'foo'
        ser[20:30] = 'bar'
        rs = ser.replace([np.nan, 'foo', 'bar'], -1)
        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert pd.isna(ser[:5]).all()
        rs = ser.replace({np.nan: -1, 'foo': -2, 'bar': -3})
        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert pd.isna(ser[:5]).all()
        rs2 = ser.replace([np.nan, 'foo', 'bar'], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)
        return_value = ser.replace([np.nan, 'foo', 'bar'], -1, inplace=True)
        assert return_value is None
        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    def test_replace_nan_with_inf(self) -> None:
        ser: pd.Series = pd.Series([np.nan, 0, np.inf])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        ser = pd.Series([np.nan, 0, 'foo', 'bar', np.inf, None, pd.NaT])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        filled: pd.Series = ser.copy()
        filled[4] = 0
        tm.assert_series_equal(ser.replace(np.inf, 0), filled)

    def test_replace_listlike_value_listlike_target(self, datetime_series: pd.Series) -> None:
        ser: pd.Series = pd.Series(datetime_series.index)
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        msg: str = 'Replacement lists must match in length\\. Expecting 3 got 2'
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3], [np.nan, 0])
        result: pd.Series = ser.replace([1, 2], [np.nan, 0])
        tm.assert_series_equal(result, ser)
        ser = pd.Series([0, 1, 2, 3, 4])
        result = ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        tm.assert_series_equal(result, pd.Series([4, 3, 2, 1, 0]))

    def test_replace_gh5319(self) -> None:
        ser: pd.Series = pd.Series([0, np.nan, 2, 3, 4])
        msg: str = "Series.replace must specify either 'value', a dict-like 'to_replace', or dict-like 'regex'"
        with pytest.raises(ValueError, match=msg):
            ser.replace([np.nan])
        with pytest.raises(ValueError, match=msg):
            ser.replace(np.nan)

    def test_replace_datetime64(self) -> None:
        ser: pd.Series = pd.Series(pd.date_range('20130101', periods=5))
        expected: pd.Series = ser.copy()
        expected.loc[2] = pd.Timestamp('20120101')
        result: pd.Series = ser.replace({pd.Timestamp('20130103'): pd.Timestamp('20120101')})
        tm.assert_series_equal(result, expected)
        result = ser.replace(pd.Timestamp('20130103'), pd.Timestamp('20120101'))
        tm.assert_series_equal(result, expected)

    def test_replace_nat_with_tz(self) -> None:
        ts: pd.Timestamp = pd.Timestamp('2015/01/01', tz='UTC')
        s: pd.Series = pd.Series([pd.NaT, pd.Timestamp('2015/01/01', tz='UTC')])
        result: pd.Series = s.replace([np.nan, pd.NaT], pd.Timestamp.min)
        expected: pd.Series = pd.Series([pd.Timestamp.min, ts], dtype=object)
        tm.assert_series_equal(expected, result)

    def test_replace_timedelta_td64(self) -> None:
        tdi: pd.TimedeltaIndex = pd.timedelta_range(0, periods=5)
        ser: pd.Series = pd.Series(tdi)
        result: pd.Series = ser.replace({ser[1]: ser[3]})
        expected: pd.Series = pd.Series([ser[0], ser[3], ser[2], ser[3], ser[4]])
        tm.assert_series_equal(result, expected)

    def test_replace_with_single_list(self) -> None:
        ser: pd.Series = pd.Series([0, 1, 2, 3, 4])
        msg: str = "Series.replace must specify either 'value', a dict-like 'to_replace', or dict-like 'regex'"
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3])
        s: pd.Series = ser.copy()
        with pytest.raises(ValueError, match=msg):
            s.replace([1, 2, 3], inplace=True)

    def test_replace_mixed_types(self) -> None:
        ser: pd.Series = pd.Series(np.arange(5), dtype='int64')

        def check_replace(to_rep: Any, val: Any, expected: pd.Series) -> None:
            sc: pd.Series = ser.copy()
            result: pd.Series = ser.replace(to_rep, val)
            return_value = sc.replace(to_rep, val, inplace=True)
            assert return_value is None
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(expected, sc)
        tr: List[Any] = [3]
        v: List[Any] = [3.0]
        check_replace(tr, v, ser)
        check_replace(tr[0], v[0], ser)
        e: pd.Series = pd.Series([0, 1, 2, 3.5, 4])
        tr, v = ([3], [3.5])
        check_replace(tr, v, e)
        e = pd.Series([0, 1, 2, 3.5, 'a'])
        tr, v = ([3, 4], [3.5, 'a'])
        check_replace(tr, v, e)
        e = pd.Series([0, 1, 2, 3.5, pd.Timestamp('20130101')])
        tr, v = ([3, 4], [3.5, pd.Timestamp('20130101')])
        check_replace(tr, v, e)
        e = pd.Series([0, 1, 2, 3.5, True], dtype='object')
        tr, v = ([3, 4], [3.5, True])
        check_replace(tr, v, e)
        dr: pd.Series = pd.Series(pd.date_range('1/1/2001', '1/10/2001', freq='D'))
        result = dr.astype(object).replace([dr[0], dr[1], dr[2]], [1.0, 2, 'a'])
        expected = pd.Series([1.0, 2, 'a'] + dr[3:].tolist(), dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_bool_with_string_no_op(self) -> None:
        s: pd.Series = pd.Series([True, False, True])
        result: pd.Series = s.replace('fun', 'in-the-sun')
        tm.assert_series_equal(s, result)

    def test_replace_bool_with_string(self) -> None:
        s: pd.Series = pd.Series([True, False, True])
        result: pd.Series = s.replace(True, '2u')
        expected: pd.Series = pd.Series(['2u', False, '2u'])
        tm.assert_series_equal(expected, result)

    def test_replace_bool_with_bool(self) -> None:
        s: pd.Series = pd.Series([True, False, True])
        result: pd.Series = s.replace(True, False)
        expected: pd.Series = pd.Series([False] * len(s))
        tm.assert_series_equal(expected, result)

    def test_replace_with_dict_with_bool_keys(self) -> None:
        s: pd.Series = pd.Series([True, False, True])
        result: pd.Series = s.replace({'asdf': 'asdb', True: 'yes'})
        expected: pd.Series = pd.Series(['yes', False, 'yes'])
        tm.assert_series_equal(result, expected)

    def test_replace_Int_with_na(self, any_int_ea_dtype: Any) -> None:
        result: pd.Series = pd.Series([0, None], dtype=any_int_ea_dtype).replace(0, pd.NA)
        expected: pd.Series = pd.Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
        tm.assert_series_equal(result, expected)
        result = pd.Series([0, 1], dtype=any_int_ea_dtype).replace(0, pd.NA)
        result.replace(1, pd.NA, inplace=True)
        tm.assert_series_equal(result, expected)

    def test_replace2(self) -> None:
        N: int = 50
        ser: pd.Series = pd.Series(np.fabs(np.random.default_rng(2).standard_normal(N)),
                                     pd.date_range('2020-01-01', periods=N), dtype=object)
        ser[:5] = np.nan
        ser[6:10] = 'foo'
        ser[20:30] = 'bar'
        rs: pd.Series = ser.replace([np.nan, 'foo', 'bar'], -1)
        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert pd.isna(ser[:5]).all()
        rs = ser.replace({np.nan: -1, 'foo': -2, 'bar': -3})
        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert pd.isna(ser[:5]).all()
        rs2: pd.Series = ser.replace([np.nan, 'foo', 'bar'], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)
        return_value = ser.replace([np.nan, 'foo', 'bar'], -1, inplace=True)
        assert return_value is None
        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    @pytest.mark.parametrize('inplace', [True, False])
    def test_replace_cascade(self, inplace: bool) -> None:
        ser: pd.Series = pd.Series([1, 2, 3])
        expected: pd.Series = pd.Series([2, 3, 4])
        res = ser.replace([1, 2, 3], [2, 3, 4], inplace=inplace)
        if inplace:
            tm.assert_series_equal(ser, expected)
        else:
            tm.assert_series_equal(res, expected)

    def test_replace_with_dictlike_and_string_dtype(self, nullable_string_dtype: Any) -> None:
        ser: pd.Series = pd.Series(['one', 'two', np.nan], dtype=nullable_string_dtype)
        expected: pd.Series = pd.Series(['1', '2', np.nan], dtype=nullable_string_dtype)
        result: pd.Series = ser.replace({'one': '1', 'two': '2'})
        tm.assert_series_equal(expected, result)

    def test_replace_with_empty_dictlike(self) -> None:
        s: pd.Series = pd.Series(list('abcd'))
        tm.assert_series_equal(s, s.replace({}))
        empty_series: pd.Series = pd.Series([])
        tm.assert_series_equal(s, s.replace(empty_series))

    def test_replace_string_with_number(self) -> None:
        s: pd.Series = pd.Series([1, 2, 3])
        result: pd.Series = s.replace('2', np.nan)
        expected: pd.Series = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_replacer_equals_replacement(self) -> None:
        s: pd.Series = pd.Series(['a', 'b'])
        expected: pd.Series = pd.Series(['b', 'a'])
        result: pd.Series = s.replace({'a': 'b', 'b': 'a'})
        tm.assert_series_equal(expected, result)

    def test_replace_unicode_with_number(self) -> None:
        s: pd.Series = pd.Series([1, 2, 3])
        result: pd.Series = s.replace('2', np.nan)
        expected: pd.Series = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_mixed_types_with_string(self) -> None:
        s: pd.Series = pd.Series([1, 2, 3, '4', 4, 5])
        result: pd.Series = s.replace([2, '4'], np.nan)
        expected: pd.Series = pd.Series([1, np.nan, 3, np.nan, 4, 5], dtype=object)
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize('categorical, numeric', [(['A'], [1]), (['A', 'B'], [1, 2])])
    def test_replace_categorical(self, categorical: List[Any], numeric: List[Any]) -> None:
        ser: pd.Series = pd.Series(pd.Categorical(categorical, categories=['A', 'B']))
        result: pd.Series = ser.cat.rename_categories({'A': 1, 'B': 2})
        expected: pd.Series = pd.Series(numeric).astype('category')
        if 2 not in expected.cat.categories:
            expected = expected.cat.add_categories(2)
        tm.assert_series_equal(expected, result, check_categorical=False)

    def test_replace_categorical_inplace(self) -> None:
        data: List[str] = ['a', 'b', 'c']
        data_exp: List[str] = ['b', 'b', 'c']
        result: pd.Series = pd.Series(data, dtype='category')
        result.replace(to_replace='a', value='b', inplace=True)
        expected: pd.Series = pd.Series(pd.Categorical(data_exp, categories=data))
        tm.assert_series_equal(result, expected)

    def test_replace_categorical_single(self) -> None:
        dti: pd.DatetimeIndex = pd.date_range('2016-01-01', periods=3, tz='US/Pacific')
        s: pd.Series = pd.Series(dti)
        c: pd.Series = s.astype('category')
        expected: pd.Series = c.copy()
        expected = expected.cat.add_categories('foo')
        expected.iloc[2] = 'foo'
        expected = expected.cat.remove_unused_categories()
        assert c.iloc[2] != 'foo'
        result: pd.Series = c.cat.rename_categories({c.values[2]: 'foo'})
        tm.assert_series_equal(expected, result)
        assert c.iloc[2] != 'foo'

    def test_replace_with_no_overflowerror(self) -> None:
        s: pd.Series = pd.Series([0, 1, 2, 3, 4])
        result: pd.Series = s.replace([3], ['100000000000000000000'])
        expected: pd.Series = pd.Series([0, 1, 2, '100000000000000000000', 4])
        tm.assert_series_equal(result, expected)
        s = pd.Series([0, '100000000000000000000', '100000000000000000001'])
        result = s.replace(['100000000000000000000'], [1])
        expected = pd.Series([0, 1, '100000000000000000001'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ser, to_replace, exp', [
        ([1, 2, 3], {1: 2, 2: 3, 3: 4}, [2, 3, 4]),
        (['1', '2', '3'], {'1': '2', '2': '3', '3': '4'}, ['2', '3', '4'])
    ])
    def test_replace_commutative(self, ser: List[Any], to_replace: Any, exp: List[Any]) -> None:
        series: pd.Series = pd.Series(ser)
        expected: pd.Series = pd.Series(exp)
        result: pd.Series = series.replace(to_replace)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ser, exp', [
        ([1, 2, 3], [1, True, 3]),
        (['x', 2, 3], ['x', True, 3])
    ])
    def test_replace_no_cast(self, ser: List[Any], exp: List[Any]) -> None:
        series: pd.Series = pd.Series(ser)
        result: pd.Series = series.replace(2, True)
        expected: pd.Series = pd.Series(exp)
        tm.assert_series_equal(result, expected)

    def test_replace_invalid_to_replace(self) -> None:
        series: pd.Series = pd.Series(['a', 'b', 'c '])
        msg: str = "Expecting 'to_replace' to be either a scalar, array-like, dict or None, got invalid type.*"
        with pytest.raises(TypeError, match=msg):
            series.replace(lambda x: x.strip())

    @pytest.mark.parametrize('frame', [False, True])
    def test_replace_nonbool_regex(self, frame: bool) -> None:
        obj: Union[pd.Series, pd.DataFrame] = pd.Series(['a', 'b', 'c '])
        if frame:
            obj = obj.to_frame()
        msg: str = "'to_replace' must be 'None' if 'regex' is not a bool"
        with pytest.raises(ValueError, match=msg):
            obj.replace(to_replace=['a'], regex='foo')

    @pytest.mark.parametrize('frame', [False, True])
    def test_replace_empty_copy(self, frame: bool) -> None:
        obj: Union[pd.Series, pd.DataFrame] = pd.Series([], dtype=np.float64)
        if frame:
            obj = obj.to_frame()
        res = obj.replace(4, 5, inplace=True)
        assert res is None
        res = obj.replace(4, 5, inplace=False)
        tm.assert_equal(res, obj)
        assert res is not obj

    def test_replace_only_one_dictlike_arg(self, fixed_now_ts: pd.Timestamp) -> None:
        ser: pd.Series = pd.Series([1, 2, 'A', fixed_now_ts, True])
        to_replace: dict = {0: 1, 2: 'A'}
        value: str = 'foo'
        msg: str = 'Series.replace cannot use dict-like to_replace and non-None value'
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)
        to_replace = 1
        value = {0: 'foo', 2: 'bar'}
        msg = 'Series.replace cannot use dict-value and non-None to_replace'
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

    def test_replace_dict_like_with_dict_like(self) -> None:
        s: pd.Series = pd.Series([1, 2, 3, 4, 5])
        to_replace: pd.Series = pd.Series([1])
        value: pd.Series = pd.Series([75])
        msg: str = 'to_replace and value cannot be dict-like for Series.replace'
        with pytest.raises(ValueError, match=msg):
            s.replace(to_replace, value)

    def test_replace_extension_other(self, frame_or_series: Any) -> None:
        obj: Any = frame_or_series(pd.array([1, 2, 3], dtype='Int64'))
        result: Any = obj.replace('', '')
        tm.assert_equal(obj, result)

    def test_replace_with_compiled_regex(self) -> None:
        s: pd.Series = pd.Series(['a', 'b', 'c'])
        regex: Any = re.compile('^a$')
        result: pd.Series = s.replace({regex: 'z'}, regex=True)
        expected: pd.Series = pd.Series(['z', 'b', 'c'])
        tm.assert_series_equal(result, expected)

    def test_pandas_replace_na(self) -> None:
        ser: pd.Series = pd.Series(['AA', 'BB', 'CC', 'DD', 'EE', '', pd.NA, 'AA'], dtype='string')
        regex_mapping: dict = {'AA': 'CC', 'BB': 'CC', 'EE': 'CC', 'CC': 'CC-REPL'}
        result: pd.Series = ser.replace(regex_mapping, regex=True)
        exp: pd.Series = pd.Series(['CC', 'CC', 'CC-REPL', 'DD', 'CC', '', pd.NA, 'CC'], dtype='string')
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize('dtype, input_data, to_replace, expected_data', [
        ('bool', [True, False], {True: False}, [False, False]),
        ('int64', [1, 2], {1: 10, 2: 20}, [10, 20]),
        ('Int64', [1, 2], {1: 10, 2: 20}, [10, 20]),
        ('float64', [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
        ('Float64', [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
        ('string', ['one', 'two'], {'one': '1', 'two': '2'}, ['1', '2']),
        (pd.IntervalDtype('int64'), IntervalArray([pd.Interval(1, 2), pd.Interval(2, 3)]),
         {pd.Interval(1, 2): pd.Interval(10, 20)},
         IntervalArray([pd.Interval(10, 20), pd.Interval(2, 3)])),
        (pd.IntervalDtype('float64'), IntervalArray([pd.Interval(1.0, 2.7), pd.Interval(2.8, 3.1)]),
         {pd.Interval(1.0, 2.7): pd.Interval(10.6, 20.8)},
         IntervalArray([pd.Interval(10.6, 20.8), pd.Interval(2.8, 3.1)])),
        (pd.PeriodDtype('M'), [pd.Period('2020-05', freq='M')],
         {pd.Period('2020-05', freq='M'): pd.Period('2020-06', freq='M')},
         [pd.Period('2020-06', freq='M')])
    ])
    def test_replace_dtype(self, dtype: Any, input_data: Any, to_replace: Any, expected_data: Any) -> None:
        ser: pd.Series = pd.Series(input_data, dtype=dtype)
        result: pd.Series = ser.replace(to_replace)
        expected: pd.Series = pd.Series(expected_data, dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_replace_string_dtype(self) -> None:
        ser: pd.Series = pd.Series(['one', 'two', np.nan], dtype='string')
        res: pd.Series = ser.replace({'one': '1', 'two': '2'})
        expected: pd.Series = pd.Series(['1', '2', np.nan], dtype='string')
        tm.assert_series_equal(res, expected)
        ser2: pd.Series = pd.Series(['A', np.nan], dtype='string')
        res2: pd.Series = ser2.replace('A', 'B')
        expected2: pd.Series = pd.Series(['B', np.nan], dtype='string')
        tm.assert_series_equal(res2, expected2)
        ser3: pd.Series = pd.Series(['A', 'B'], dtype='string')
        res3: pd.Series = ser3.replace('A', pd.NA)
        expected3: pd.Series = pd.Series([pd.NA, 'B'], dtype='string')
        tm.assert_series_equal(res3, expected3)

    def test_replace_string_dtype_list_to_replace(self) -> None:
        ser: pd.Series = pd.Series(['abc', 'def'], dtype='string')
        res: pd.Series = ser.replace(['abc', 'any other string'], 'xyz')
        expected: pd.Series = pd.Series(['xyz', 'def'], dtype='string')
        tm.assert_series_equal(res, expected)

    def test_replace_string_dtype_regex(self) -> None:
        ser: pd.Series = pd.Series(['A', 'B'], dtype='string')
        res: pd.Series = ser.replace('.', 'C', regex=True)
        expected: pd.Series = pd.Series(['C', 'C'], dtype='string')
        tm.assert_series_equal(res, expected)

    def test_replace_nullable_numeric(self) -> None:
        floats: pd.Series = pd.Series([1.0, 2.0, 3.999, 4.4], dtype=pd.Float64Dtype())
        assert floats.replace({1.0: 9}).dtype == floats.dtype
        assert floats.replace(1.0, 9).dtype == floats.dtype
        assert floats.replace({1.0: 9.0}).dtype == floats.dtype
        assert floats.replace(1.0, 9.0).dtype == floats.dtype
        res: pd.Series = floats.replace(to_replace=[1.0, 2.0], value=[9.0, 10.0])
        assert res.dtype == floats.dtype
        ints: pd.Series = pd.Series([1, 2, 3, 4], dtype=pd.Int64Dtype())
        assert ints.replace({1: 9}).dtype == ints.dtype
        assert ints.replace(1, 9).dtype == ints.dtype
        assert ints.replace({1: 9.0}).dtype == ints.dtype
        assert ints.replace(1, 9.0).dtype == ints.dtype
        with pytest.raises(TypeError, match='Invalid value'):
            ints.replace({1: 9.5})
        with pytest.raises(TypeError, match='Invalid value'):
            ints.replace(1, 9.5)

    @pytest.mark.parametrize('regex', [False, True])
    def test_replace_regex_dtype_series(self, regex: bool) -> None:
        series: pd.Series = pd.Series(['0'], dtype=object)
        expected: pd.Series = pd.Series([1], dtype=object)
        result: pd.Series = series.replace(to_replace='0', value=1, regex=regex)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('regex', [False, True])
    def test_replace_regex_dtype_series_string(self, regex: bool) -> None:
        series: pd.Series = pd.Series(['0'], dtype='str')
        expected: pd.Series = pd.Series([1], dtype=object)
        result: pd.Series = series.replace(to_replace='0', value=1, regex=regex)
        tm.assert_series_equal(result, expected)

    def test_replace_different_int_types(self, any_int_numpy_dtype: Any) -> None:
        labs: pd.Series = pd.Series([1, 1, 1, 0, 0, 2, 2, 2], dtype=any_int_numpy_dtype)
        maps: pd.Series = pd.Series([0, 2, 1], dtype=any_int_numpy_dtype)
        map_dict: dict = dict(zip(maps.values, maps.index))
        result: pd.Series = labs.replace(map_dict)
        expected: pd.Series = labs.replace({0: 0, 2: 1, 1: 2})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('val', [2, np.nan, 2.0])
    def test_replace_value_none_dtype_numeric(self, val: Any) -> None:
        ser: pd.Series = pd.Series([1, val])
        result: pd.Series = ser.replace(val, None)
        expected: pd.Series = pd.Series([1, None], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_change_dtype_series(self) -> None:
        df: pd.DataFrame = pd.DataFrame({'Test': ['0.5', True, '0.6']}, dtype=object)
        df['Test'] = df['Test'].replace([True], [np.nan])
        expected: pd.DataFrame = pd.DataFrame({'Test': ['0.5', np.nan, '0.6']}, dtype=object)
        tm.assert_frame_equal(df, expected)
        df = pd.DataFrame({'Test': ['0.5', None, '0.6']}, dtype=object)
        df['Test'] = df['Test'].replace([None], [np.nan])
        tm.assert_frame_equal(df, expected)
        df = pd.DataFrame({'Test': ['0.5', None, '0.6']}, dtype=object)
        df['Test'] = df['Test'].fillna(np.nan)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('dtype', ['object', 'Int64'])
    def test_replace_na_in_obj_column(self, dtype: str) -> None:
        ser: pd.Series = pd.Series([0, 1, pd.NA], dtype=dtype)
        expected: pd.Series = pd.Series([0, 2, pd.NA], dtype=dtype)
        result: pd.Series = ser.replace(to_replace=1, value=2)
        tm.assert_series_equal(result, expected)
        ser.replace(to_replace=1, value=2, inplace=True)
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('val', [0, 0.5])
    def test_replace_numeric_column_with_na(self, val: Union[int, float]) -> None:
        ser: pd.Series = pd.Series([val, 1])
        expected: pd.Series = pd.Series([val, pd.NA])
        result: pd.Series = ser.replace(to_replace=1, value=pd.NA)
        tm.assert_series_equal(result, expected)
        ser.replace(to_replace=1, value=pd.NA, inplace=True)
        tm.assert_series_equal(ser, expected)

    def test_replace_ea_float_with_bool(self) -> None:
        ser: pd.Series = pd.Series([0.0], dtype='Float64')
        expected: pd.Series = ser.copy()
        result: pd.Series = ser.replace(False, 1.0)
        tm.assert_series_equal(result, expected)
        ser = pd.Series([False], dtype='boolean')
        expected = ser.copy()
        result = ser.replace(0.0, True)
        tm.assert_series_equal(result, expected)

    def test_replace_all_NA(self) -> None:
        df: pd.Series = pd.Series([pd.NA, pd.NA])
        result: pd.Series = df.replace({'^#': '$'}, regex=True)
        expected: pd.Series = pd.Series([pd.NA, pd.NA])
        tm.assert_series_equal(result, expected)
