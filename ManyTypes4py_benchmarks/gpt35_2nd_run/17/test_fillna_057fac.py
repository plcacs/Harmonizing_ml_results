from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from pandas import Categorical, DataFrame, DatetimeIndex, NaT, Period, Series, Timedelta, Timestamp, date_range, isna, timedelta_range
import pandas._testing as tm
from pandas.core.arrays import period_array

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

    def test_timedelta_fillna(self, frame_or_series, unit) -> None:
        ser: Series = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130102'), Timestamp('20130103 9:01:01')], dtype=f'M8[{unit}]')
        td: Series = ser.diff()
        obj = frame_or_series(td).copy()
        result: Series = obj.fillna(Timedelta(seconds=0))
        expected: Series = Series([timedelta(0), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)
        res: Series = obj.fillna(1)
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
    def test_datetime64_fillna_mismatched_reso_no_rounding(self, tz, scalar) -> None:
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
    def test_timedelta64_fillna_mismatched_reso_no_rounding(self, scalar) -> None:
        tdi: Series = Series(date_range('2016-01-01', periods=3, unit='s') - Timestamp('1970-01-01'))
        item: Timedelta = Timestamp('2016-02-03 04:05:06.789') - Timestamp('1970-01-01')
        vec: Series = Series(timedelta_range(item, periods=3, unit='ms'))
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
    def test_datetime64_tz_fillna(self, tz, unit) -> None:
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
        ser: Series = Series(idx)
        assert ser.dtype == f'datetime64[{unit}, {tz}]'
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('2011-01-02 10:00'))
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), Timestamp('2011-01-02 10:00'), Timestamp('2011-01-03 10:00', tz=tz), Timestamp('2011-01-02 10:00')])
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('2011-01-02 10:00', tz=tz))
        idx: DatetimeIndex = DatetimeIndex(['2011-01-01 10:00', '2011-01-02 10:00', '2011-01-03 10:00', '2011-01-02 10:00'], tz=tz).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna(Timestamp('2011-01-02 10:00', tz=tz).to_pydatetime())
        idx: DatetimeIndex = DatetimeIndex(['2011-01-01 10:00', '2011-01-02 10:00', '2011-01-03 10:00', '2011-01-02 10:00'], tz=tz).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna('AAA')
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), 'AAA', Timestamp('2011-01-03 10:00', tz=tz), 'AAA'], dtype=object)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)
        result = ser.fillna({1: Timestamp('2011-01-02 10:00', tz=tz), 3: Timestamp('2011-01-04 10:00')})
        expected = Series([Timestamp('2011-01-01 10:00', tz=tz), Timestamp('2011-01-02 10:00', tz=tz), Timestamp('2011-01-03 10:00', tz=tz), Timestamp