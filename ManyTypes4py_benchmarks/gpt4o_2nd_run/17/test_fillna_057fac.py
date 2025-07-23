from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from pandas import Categorical, DataFrame, DatetimeIndex, NaT, Period, Series, Timedelta, Timestamp, date_range, isna, timedelta_range
import pandas._testing as tm
from pandas.core.arrays import period_array
from typing import Any, Dict, List, Union

class TestSeriesFillNA:

    def test_fillna_nat(self) -> None:
        series: Series = Series([0, 1, 2, NaT._value], dtype='M8[ns]')
        filled: Series = series.ffill()
        filled2: Series = series.fillna(value=series.values[2])
        expected: Series = series.copy()
        expected.iloc[3] = expected.iloc[2]
        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)
        df: DataFrame = DataFrame({'A': series})
        filled = df.ffill()
        filled2 = df.fillna(value=series.values[2])
        expected = DataFrame({'A': expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)
        series = Series([NaT._value, 0, 1, 2], dtype='M8[ns]')
        filled = series.bfill()
        filled2 = series.fillna(value=series[1])
        expected = series.copy()
        expected[0] = expected[1]
        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)
        df = DataFrame({'A': series})
        filled = df.bfill()
        filled2 = df.fillna(value=series[1])
        expected = DataFrame({'A': expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)

    def test_fillna(self) -> None:
        ts: Series = Series([0.0, 1.0, 2.0, 3.0, 4.0], index=date_range('2020-01-01', periods=5))
        tm.assert_series_equal(ts, ts.ffill())
        ts.iloc[2] = np.nan
        exp: Series = Series([0.0, 1.0, 1.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.ffill(), exp)
        exp = Series([0.0, 1.0, 3.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.bfill(), exp)
        exp = Series([0.0, 1.0, 5.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(value=5), exp)

    def test_fillna_nonscalar(self) -> None:
        s1: Series = Series([np.nan])
        s2: Series = Series([1])
        result: Series = s1.fillna(s2)
        expected: Series = Series([1.0])
        tm.assert_series_equal(result, expected)
        result = s1.fillna({})
        tm.assert_series_equal(result, s1)
        result = s1.fillna(Series((), dtype=object))
        tm.assert_series_equal(result, s1)
        result = s2.fillna(s1)
        tm.assert_series_equal(result, s2)
        result = s1.fillna({0: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna({1: 1})
        tm.assert_series_equal(result, Series([np.nan]))
        result = s1.fillna({0: 1, 1: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}))
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}, index=[4, 5]))
        tm.assert_series_equal(result, s1)

    def test_fillna_aligns(self) -> None:
        s1: Series = Series([0, 1, 2], list('abc'))
        s2: Series = Series([0, np.nan, 2], list('bac'))
        result: Series = s2.fillna(s1)
        expected: Series = Series([0, 0, 2.0], list('bac'))
        tm.assert_series_equal(result, expected)

    def test_fillna_limit(self) -> None:
        ser: Series = Series(np.nan, index=[0, 1, 2])
        result: Series = ser.fillna(999, limit=1)
        expected: Series = Series([999, np.nan, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)
        result = ser.fillna(999, limit=2)
        expected = Series([999, 999, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_fillna_dont_cast_strings(self) -> None:
        vals: List[str] = ['0', '1.5', '-0.3']
        for val in vals:
            ser: Series = Series([0, 1, np.nan, np.nan, 4], dtype='float64')
            result: Series = ser.fillna(val)
            expected: Series = Series([0, 1, val, val, 4], dtype='object')
            tm.assert_series_equal(result, expected)

    def test_fillna_consistency(self) -> None:
        ser: Series = Series([Timestamp('20130101'), NaT])
        result: Series = ser.fillna(Timestamp('20130101', tz='US/Eastern'))
        expected: Series = Series([Timestamp('20130101'), Timestamp('2013-01-01', tz='US/Eastern')], dtype='object')
        tm.assert_series_equal(result, expected)
        result = ser.where([True, False], Timestamp('20130101', tz='US/Eastern'))
        tm.assert_series_equal(result, expected)
        result = ser.where([True, False], Timestamp('20130101', tz='US/Eastern'))
        tm.assert_series_equal(result, expected)
        result = ser.fillna('foo')
        expected = Series([Timestamp('20130101'), 'foo'])
        tm.assert_series_equal(result, expected)
        ser2: Series = ser.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            ser2[1] = 'foo'

    def test_timedelta_fillna(self, frame_or_series: Any, unit: str) -> None:
        ser: Series = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130102'), Timestamp('20130103 9:01:01')], dtype=f'M8[{unit}]')
        td: Series = ser.diff()
        obj: Any = frame_or_series(td).copy()
        result: Any = obj.fillna(Timedelta(seconds=0))
        expected: Series = Series([timedelta(0), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        res: Any = obj.fillna(1)
        expected = obj.astype(object).fillna(1)
        tm.assert_equal(res, expected)
        result = obj.fillna(Timedelta(seconds=1))
        expected = Series([timedelta(seconds=1), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        result = obj.fillna(timedelta(days=1, seconds=1))
        expected = Series([timedelta(days=1, seconds=1), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        result = obj.fillna(np.timedelta64(10 ** 9))
        expected = Series([timedelta(seconds=1), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        result = obj.fillna(NaT)
        expected = Series([NaT, timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        td[2] = np.nan
        obj = frame_or_series(td).copy()
        result = obj.ffill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[0] = np.nan
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        td[2] = np.nan
        obj = frame_or_series(td)
        result = obj.bfill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[2] = timedelta(days=1, seconds=9 * 3600 + 60 + 1)
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

    def test_datetime64_fillna(self) -> None:
        ser: Series = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130102'), Timestamp('20130103 9:01:01')])
        ser[2] = np.nan
        result: Series = ser.ffill()
        expected: Series = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130103 9:01:01')])
        tm.assert_series_equal(result, expected)
        result = ser.bfill()
        expected = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130103 9:01:01'), Timestamp('20130103 9:01:01')])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('scalar', [False, True])
    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_datetime64_fillna_mismatched_reso_no_rounding(self, tz: Union[None, str], scalar: bool) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, unit='s', tz=tz)
        item: Timestamp = Timestamp('2016-02-03 04:05:06.789', tz=tz)
        vec: DatetimeIndex = date_range(item, periods=3, unit='ms')
        exp_dtype: str = 'M8[ms]' if tz is None else 'M8[ms, UTC]'
        expected: Series = Series([item, dti[1], dti[2]], dtype=exp_dtype)
        ser: Series = Series(dti)
        ser[0] = NaT
        ser2: Series = ser.copy()
        res: Series = ser.fillna(item)
        res2: Series = ser2.fillna(Series(vec))
        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)

    @pytest.mark.parametrize('scalar', [False, True])
    def test_timedelta64_fillna_mismatched_reso_no_rounding(self, scalar: bool) -> None:
        tdi: Series = date_range('2016-01-01', periods=3, unit='s') - Timestamp('1970-01-01')
        item: Timedelta = Timestamp('2016-02-03 04:05:06.789') - Timestamp('1970-01-01')
        vec: Series = timedelta_range(item, periods=3, unit='ms')
        expected: Series = Series([item, tdi[1], tdi[2]], dtype='m8[ms]')
        ser: Series = Series(tdi)
        ser[0] = NaT
        ser2: Series = ser.copy()
        res: Series = ser.fillna(item)
        res2: Series = ser2.fillna(Series(vec))
        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)

    def test_datetime64_fillna_backfill(self) -> None:
        ser: Series = Series([NaT, NaT, '2013-08-05 15:30:00.000001'], dtype='M8[ns]')
        expected: Series = Series(['2013-08-05 15:30:00.000001', '2013-08-05 15:30:00.000001', '2013-08-05 15:30:00.000001'], dtype='M8[ns]')
        result: Series = ser.bfill()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', ['US/Eastern', 'Asia/Tokyo'])
    def test_datetime64_tz_fillna(self, tz: str, unit: str) -> None:
        ser: Series = Series([Timestamp('2011-01-01 10:00'), NaT, Timestamp('2011-01-03 10:00'), NaT], dtype=f'M8[{unit}]')
        null_loc: Series = Series([False, True, False, True])
        result: Series = ser.fillna(Timestamp('2011-01-02 10:00'))
        expected: Series = Series([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00'), Timestamp('2011-01-03 10:00'), Timestamp('2011-01-02 10:00')], dtype=f'M8[{unit}]')
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('2011-01-02 10:00', tz=tz))
        expected = Series([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz=tz), Timestamp('2011-01-03 10:00'), Timestamp('2011-01-02 10:00', tz=tz)])
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna('AAA')
        expected = Series([Timestamp('2011-01-01 10:00'), 'AAA', Timestamp('2011-01-03 10:00'), 'AAA'], dtype=object)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna({1: Timestamp('2011-01-02 10:00', tz=tz), 3: Timestamp('2011-01-04 10:00')})
        expected = Series([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz=tz), Timestamp('2011-01-03 10:00'), Timestamp('2011-01-04 10:00')])
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna({1: Timestamp('2011-01-02 10:00'), 3: Timestamp('2011-01-04 10:00')})
        expected = Series([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00'), Timestamp('2011-01-03 10:00'), Timestamp('2011-01-04 10:00')], dtype=f'M8[{unit}]')
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        idx: DatetimeIndex = DatetimeIndex(['2011-01-01 10:00', NaT, '2011-01-03 10:00', NaT], tz=tz).as_unit(unit)
        ser = Series(idx)
        assert ser.dtype == f'datetime64[{unit}, {tz}]'
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('2011-01-02 10:00'))
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), Timestamp('2011-01-02 10:00'), Timestamp('2011-01-03 10:00', tz=tz), Timestamp('2011-01-02 10:00')])
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('2011-01-02 10:00', tz=tz))
        idx = DatetimeIndex(['2011-01-01 10:00', '2011-01-02 10:00', '2011-01-03 10:00', '2011-01-02 10:00'], tz=tz).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('2011-01-02 10:00', tz=tz).to_pydatetime())
        idx = DatetimeIndex(['2011-01-01 10:00', '2011-01-02 10:00', '2011-01-03 10:00', '2011-01-02 10:00'], tz=tz).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna('AAA')
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), 'AAA', Timestamp('2011-01-03 10:00', tz=tz), 'AAA'], dtype=object)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna({1: Timestamp('2011-01-02 10:00', tz=tz), 3: Timestamp('2011-01-04 10:00')})
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), Timestamp('2011-01-02 10:00', tz=tz), Timestamp('2011-01-03 10:00', tz=tz), Timestamp('2011-01-04 10:00')])
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna({1: Timestamp('2011-01-02 10:00', tz=tz), 3: Timestamp('2011-01-04 10:00', tz=tz)})
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), Timestamp('2011-01-02 10:00', tz=tz), Timestamp('2011-01-03 10:00', tz=tz), Timestamp('2011-01-04 10:00', tz=tz)]).dt.as_unit(unit)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('20130101'))
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), Timestamp('2013-01-01'), Timestamp('2011-01-03 10:00', tz=tz), Timestamp('2013-01-01')])
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('20130101', tz='US/Pacific'))
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), Timestamp('2013-01-01', tz='US/Pacific').tz_convert(tz), Timestamp('2011-01-03 10:00', tz=tz), Timestamp('2013-01-01', tz='US/Pacific').tz_convert(tz)]).dt.as_unit(unit)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

    def test_fillna_dt64tz_with_method(self) -> None:
        ser: Series = Series([Timestamp('2012-11-11 00:00:00+01:00'), NaT])
        exp: Series = Series([Timestamp('2012-11-11 00:00:00+01:00'), Timestamp('2012-11-11 00:00:00+01:00')])
        tm.assert_series_equal(ser.ffill(), exp)
        ser = Series([NaT, Timestamp('2012-11-11 00:00:00+01:00')])
        exp = Series([Timestamp('2012-11-11 00:00:00+01:00'), Timestamp('2012-11-11 00:00:00+01:00')])
        tm.assert_series_equal(ser.bfill(), exp)

    def test_fillna_pytimedelta(self) -> None:
        ser: Series = Series([np.nan, Timedelta('1 days')], index=['A', 'B'])
        result: Series = ser.fillna(timedelta(1))
        expected: Series = Series(Timedelta('1 days'), index=['A', 'B'])
        tm.assert_series_equal(result, expected)

    def test_fillna_period(self) -> None:
        ser: Series = Series([Period('2011-01', freq='M'), Period('NaT', freq='M')])
        res: Series = ser.fillna(Period('2012-01', freq='M'))
        exp: Series = Series([Period('2011-01', freq='M'), Period('2012-01', freq='M')])
        tm.assert_series_equal(res, exp)
        assert res.dtype == 'Period[M]'

    def test_fillna_dt64_timestamp(self, frame_or_series: Any) -> None:
        ser: Series = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130102'), Timestamp('20130103 9:01:01')])
        ser[2] = np.nan
        obj: Any = frame_or_series(ser)
        result: Any = obj.fillna(Timestamp('20130104'))
        expected: Series = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130104'), Timestamp('20130103 9:01:01')])
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        result = obj.fillna(NaT)
        expected = obj
        tm.assert_equal(result, expected)

    def test_fillna_dt64_non_nao(self) -> None:
        ser: Series = Series([Timestamp('2010-01-01'), NaT, Timestamp('2000-01-01')])
        val: np.datetime64 = np.datetime64('1975-04-05', 'ms')
        result: Series = ser.fillna(val)
        expected: Series = Series([Timestamp('2010-01-01'), Timestamp('1975-04-05'), Timestamp('2000-01-01')])
        tm.assert_series_equal(result, expected)

    def test_fillna_numeric_inplace(self) -> None:
        x: Series = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ['z', 'a', 'b', 'c', 'd'])
        y: Series = x.copy()
        return_value: None = y.fillna(value=0, inplace=True)
        assert return_value is None
        expected: Series = x.fillna(value=0)
        tm.assert_series_equal(y, expected)

    @pytest.mark.parametrize('fill_value, expected_output', [('a', ['a', 'a', 'b', 'a', 'a']), ({1: 'a', 3: 'b', 4: 'b'}, ['a', 'a', 'b', 'b', 'b']), ({1: 'a'}, ['a', 'a', 'b', np.nan, np.nan]), ({1: 'a', 3: 'b'}, ['a', 'a', 'b', 'b', np.nan]), (Series('a'), ['a', np.nan, 'b', np.nan, np.nan]), (Series('a', index=[1]), ['a', 'a', 'b', np.nan, np.nan]), (Series({1: 'a', 3: 'b'}), ['a', 'a', 'b', 'b', np.nan]), (Series(['a', 'b'], index=[3, 4]), ['a', np.nan, 'b', 'a', 'b'])])
    def test_fillna_categorical(self, fill_value: Union[str, Dict[int, str], Series], expected_output: List[Union[str, float]]) -> None:
        data: List[Union[str, float]] = ['a', np.nan, 'b', np.nan, np.nan]
        ser: Series = Series(Categorical(data, categories=['a', 'b']))
        exp: Series = Series(Categorical(expected_output, categories=['a', 'b']))
        result: Series = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize('fill_value, expected_output', [(['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'b', 'd', 'e']), (['b', 'd', 'a', 'd', 'a'], ['a', 'd', 'b', 'd', 'a']), (Categorical(['b', 'd', 'a', 'd', 'a'], categories=['b', 'c', 'd', 'e', 'a']), ['a', 'd', 'b', 'd', 'a'])])
    def test_fillna_categorical_with_new_categories(self, fill_value: Union[List[str], Categorical], expected_output: List[str]) -> None:
        data: List[Union[str, float]] = ['a', np.nan, 'b', np.nan, np.nan]
        ser: Series = Series(Categorical(data, categories=['a', 'b', 'c', 'd', 'e']))
        exp: Series = Series(Categorical(expected_output, categories=['a', 'b', 'c', 'd', 'e']))
        fill_value = Series(fill_value)
        result: Series = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    def test_fillna_categorical_raises(self) -> None:
        data: List[Union[str, float]] = ['a', np.nan, 'b', np.nan, np.nan]
        ser: Series = Series(Categorical(data, categories=['a', 'b']))
        cat: Categorical = ser._values
        msg: str = 'Cannot setitem on a Categorical with a new category'
        with pytest.raises(TypeError, match=msg):
            ser.fillna('d')
        msg2: str = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg2):
            cat.fillna(Series('d'))
        with pytest.raises(TypeError, match=msg):
            ser.fillna({1: 'd', 3: 'a'})
        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(['a', 'b'])
        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(('a', 'b'))
        msg = '"value" parameter must be a scalar, dict or Series, but you passed a "DataFrame"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(DataFrame({1: ['a'], 3: ['b']}))

    @pytest.mark.parametrize('dtype', [float, 'float32', 'float64'])
    @pytest.mark.parametrize('scalar', [True, False])
    def test_fillna_float_casting(self, dtype: Union[type, str], any_real_numpy_dtype: np.dtype, scalar: bool) -> None:
        ser: Series = Series([np.nan, 1.2], dtype=dtype)
        fill_values: Union[Series, float] = Series([2, 2], dtype=any_real_numpy_dtype)
        if scalar:
            fill_values = fill_values.dtype.type(2)
        result: Series = ser.fillna(fill_values)
        expected: Series = Series([2.0, 1.2], dtype=dtype)
        tm.assert_series_equal(result, expected)
        ser = Series([np.nan, 1.2], dtype=dtype)
        mask: np.ndarray = ser.isna().to_numpy()
        ser[mask] = fill_values
        tm.assert_series_equal(ser, expected)
        ser = Series([np.nan, 1.2], dtype=dtype)
        ser.mask(mask, fill_values, inplace=True)
        tm.assert_series_equal(ser, expected)
        ser = Series([np.nan, 1.2], dtype=dtype)
        res: Series = ser.where(~mask, fill_values)
        tm.assert_series_equal(res, expected)

    def test_fillna_f32_upcast_with_dict(self) -> None:
        ser: Series = Series([np.nan, 1.2], dtype=np.float32)
        result: Series = ser.fillna({0: 1})
        expected: Series = Series([1.0, 1.2], dtype=np.float32)
        tm.assert_series_equal(result, expected)

    def test_fillna_listlike_invalid(self) -> None:
        ser: Series = Series(np.random.default_rng(2).integers(-100, 100, 50))
        msg: str = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna([1, 2])
        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna((1, 2))

    def test_fillna_method_and_limit_invalid(self) -> None:
        ser: Series = Series([1, 2, 3, None])
        msg: str = '|'.join(["Cannot specify both 'value' and 'method'\\.", 'Limit must be greater than 0', 'Limit must be an integer'])
        for limit in [-1, 0, 1.0, 2.0]:
            with pytest.raises(ValueError, match=msg):
                ser.fillna(1, limit=limit)

    def test_fillna_datetime64_with_timezone_tzinfo(self) -> None:
        ser: Series = Series(date_range('2020', periods=3, tz='UTC'))
        expected: Series = ser.copy()
        ser[1] = NaT
        result: Series = ser.fillna(datetime(2020, 1, 2, tzinfo=timezone.utc))
        tm.assert_series_equal(result, expected)
        ts: Timestamp = Timestamp('2000-01-01', tz='US/Pacific')
        ser2: Series = Series(ser._values.tz_convert('dateutil/US/Pacific'))
        assert ser2.dtype.kind == 'M'
        result = ser2.fillna(ts)
        expected = Series([ser2[0], ts.tz_convert(ser2.dtype.tz), ser2[2]], dtype=ser2.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input, input_fillna, expected_data, expected_categories', [(['A', 'B', None, 'A'], 'B', ['A', 'B', 'B', 'A'], ['A', 'B']), (['A', 'B', np.nan, 'A'], 'B', ['A', 'B', 'B', 'A'], ['A', 'B'])])
    def test_fillna_categorical_accept_same_type(self, input: List[Union[str, None]], input_fillna: str, expected_data: List[str], expected_categories: List[str]) -> None:
        cat: Categorical = Categorical(input)
        ser: Series = Series(cat).fillna(input_fillna)
        filled: Categorical = cat.fillna(ser)
        result: Categorical = cat.fillna(filled)
        expected: Categorical = Categorical(expected_data, categories=expected_categories)
        tm.assert_categorical_equal(result, expected)

class TestFillnaPad:

    def test_fillna_bug(self) -> None:
        ser: Series = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ['z', 'a', 'b', 'c', 'd'])
        filled: Series = ser.ffill()
        expected: Series = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ser.index)
        tm.assert_series_equal(filled, expected)
        filled = ser.bfill()
        expected = Series([1.0, 1.0, 3.0, 3.0, np.nan], ser.index)
        tm.assert_series_equal(filled, expected)

    def test_ffill_mixed_dtypes_without_missing_data(self) -> None:
        series: Series = Series([datetime(2015, 1, 1, tzinfo=timezone.utc), 1])
        result: Series = series.ffill()
        tm.assert_series_equal(series, result)

    def test_pad_nan(self) -> None:
        x: Series = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ['z', 'a', 'b', 'c', 'd'], dtype=float)
        return_value: None = x.ffill(inplace=True)
        assert return_value is None
        expected: Series = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ['z', 'a', 'b', 'c', 'd'], dtype=float)
        tm.assert_series_equal(x[1:], expected[1:])
        assert np.isnan(x.iloc[0]), np.isnan(expected.iloc[0])

    def test_series_fillna_limit(self) -> None:
        index: np.ndarray = np.arange(10)
        s: Series = Series(np.random.default_rng(2).standard_normal(10), index=index)
        result: Series = s[:2].reindex(index)
        result = result.ffill(limit=5)
        expected: Series = s[:2].reindex(index).ffill()
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)
        result = s[-2:].reindex(index)
        result = result.bfill(limit=5)
        expected = s[-2:].reindex(index).bfill()
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_series_pad_backfill_limit(self) -> None:
        index: np.ndarray = np.arange(10)
        s: Series = Series(np.random.default_rng(2).standard_normal(10), index=index)
        result: Series = s[:2].reindex(index, method='pad', limit=5)
        expected: Series = s[:2].reindex(index).ffill()
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)
        result = s[-2:].reindex(index, method='backfill', limit=5)
        expected = s[-2:].reindex(index).bfill()
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_fillna_int(self) -> None:
        ser: Series = Series(np.random.default_rng(2).integers(-100, 100, 50))
        return_value: None = ser.ffill(inplace=True)
        assert return_value is None
        tm.assert_series_equal(ser.ffill(inplace=False), ser)

    def test_datetime64tz_fillna_round_issue(self) -> None:
        data: Series = Series([NaT, NaT, datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc)])
        filled: Series = data.bfill()
        expected: Series = Series([datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc), datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc), datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc)])
        tm.assert_series_equal(filled, expected)

    def test_fillna_parr(self) -> None:
        dti: DatetimeIndex = date_range(Timestamp.max - Timedelta(nanoseconds=10), periods=5, freq='ns')
        ser: Series = Series(dti.to_period('ns'))
        ser[2] = NaT
        arr: period_array = period_array([Timestamp('2262-04-11 23:47:16.854775797'), Timestamp('2262-04-11 23:47:16.854775798'), Timestamp('2262-04-11 23:47:16.854775798'), Timestamp('2262-04-11 23:47:16.854775800'), Timestamp('2262-04-11 23:47:16.854775801')], freq='ns')
        expected: Series = Series(arr)
        filled: Series = ser.ffill()
        tm.assert_series_equal(filled, expected)

@pytest.mark.parametrize('data, expected_data, method, kwargs', (([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan], 'ffill', {'limit_area': 'inside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan], 'ffill', {'limit_area': 'inside', 'limit': 1}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0], 'ffill', {'limit_area': 'outside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan], 'ffill', {'limit_area': 'outside', 'limit': 1}), ([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], 'ffill', {'limit_area': 'outside', 'limit': 1}), (range(5), range(5), 'ffill', {'limit_area': 'outside', 'limit': 1}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan], 'bfill', {'limit_area': 'inside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan], 'bfill', {'limit_area': 'inside', 'limit': 1}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan], 'bfill', {'limit_area': 'outside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan], 'bfill', {'limit_area': 'outside', 'limit': 1})))
def test_ffill_bfill_limit_area(data: List[Union[float, int]], expected_data: List[float], method: str, kwargs: Dict[str, Union[str, int]]) -> None:
    s: Series = Series(data)
    expected: Series = Series(expected_data)
    result: Series = getattr(s, method)(**kwargs)
    tm.assert_series_equal(result, expected)
