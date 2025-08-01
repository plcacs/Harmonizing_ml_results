from datetime import datetime, timezone
import re
import warnings
import zoneinfo
import dateutil
import numpy as np
import pytest
from pandas._libs.tslibs.ccalendar import DAYS, MONTHS
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import Period, PeriodIndex, period_range
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
from typing import Callable, Union, Tuple, Any, Optional, Type

pytestmark = pytest.mark.filterwarnings('ignore:Resampling with a PeriodIndex is deprecated:FutureWarning')

@pytest.fixture
def simple_period_range_series() -> Callable[[str, str, str], Series]:
    """
    Series with period range index and random data for test purposes.
    """

    def _simple_period_range_series(start: str, end: str, freq: str = 'D') -> Series:
        with warnings.catch_warnings():
            msg = '|'.join(['Period with BDay freq', 'PeriodDtype\\[B\\] is deprecated'])
            warnings.filterwarnings('ignore', msg, category=FutureWarning)
            rng = period_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    return _simple_period_range_series

class TestPeriodIndex:

    @pytest.mark.parametrize('freq', ['2D', '1h', '2h'])
    def test_asfreq(self, frame_or_series: Union[DataFrame, Series], freq: str) -> None:
        obj = frame_or_series(range(5), index=period_range('2020-01-01', periods=5))
        expected = obj.to_timestamp().resample(freq).asfreq()
        result = obj.to_timestamp().resample(freq).asfreq()
        tm.assert_almost_equal(result, expected)
        start = obj.index[0].to_timestamp(how='start')
        end = (obj.index[-1] + obj.index.freq).to_timestamp(how='start')
        new_index = date_range(start=start, end=end, freq=freq, inclusive='left')
        expected = obj.to_timestamp().reindex(new_index).to_period(freq)
        result = obj.resample(freq).asfreq()
        tm.assert_almost_equal(result, expected)
        result = obj.resample(freq).asfreq().to_timestamp().to_period()
        tm.assert_almost_equal(result, expected)

    def test_asfreq_fill_value(self) -> None:
        index = period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
        s = Series(range(len(index)), index=index)
        new_index = date_range(s.index[0].to_timestamp(how='start'), s.index[-1].to_timestamp(how='start'), freq='1h')
        expected = s.to_timestamp().reindex(new_index, fill_value=4.0)
        result = s.to_timestamp().resample('1h').asfreq(fill_value=4.0)
        tm.assert_series_equal(result, expected)
        frame = s.to_frame('value')
        new_index = date_range(frame.index[0].to_timestamp(how='start'), frame.index[-1].to_timestamp(how='start'), freq='1h')
        expected = frame.to_timestamp().reindex(new_index, fill_value=3.0)
        result = frame.to_timestamp().resample('1h').asfreq(fill_value=3.0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('freq', ['h', '12h', '2D', 'W'])
    @pytest.mark.parametrize('kwargs', [{'on': 'date'}, {'level': 'd'}])
    def test_selection(self, freq: str, kwargs: dict[str, Any]) -> None:
        index = period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
        rng = np.arange(len(index), dtype=np.int64)
        df = DataFrame({'date': index, 'a': rng}, index=pd.MultiIndex.from_arrays([rng, index], names=['v', 'd']))
        msg = 'Resampling from level= or on= selection with a PeriodIndex is not currently supported, use \\.set_index\\(\\.\\.\\.\\) to explicitly set index'
        with pytest.raises(NotImplementedError, match=msg):
            df.resample(freq, **kwargs)

    @pytest.mark.parametrize('month', MONTHS)
    @pytest.mark.parametrize('meth', ['ffill', 'bfill'])
    @pytest.mark.parametrize('conv', ['start', 'end'])
    @pytest.mark.parametrize(('offset', 'period'), [('D', 'D'), ('B', 'B'), ('ME', 'M'), ('QE', 'Q')])
    def test_annual_upsample_cases(self, offset: str, period: str, conv: str, meth: str, month: str, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        ts = simple_period_range_series('1/1/1990', '12/31/1991', freq=f'Y-{month}')
        warn: Optional[Type[Warning]] = FutureWarning if period == 'B' else None
        msg = 'PeriodDtype\\[B\\] is deprecated'
        if warn is None:
            msg = 'Resampling with a PeriodIndex is deprecated'
            warn = FutureWarning
        with tm.assert_produces_warning(warn, match=msg):
            result = getattr(ts.resample(period, convention=conv), meth)()
            expected = result.to_timestamp(period, how=conv)
            expected = expected.asfreq(offset, meth).to_period()
        tm.assert_series_equal(result, expected)

    def test_basic_downsample(self, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='M')
        result = ts.resample('Y-DEC').mean()
        expected = ts.groupby(ts.index.year).mean()
        expected.index = period_range('1/1/1990', '6/30/1995', freq='Y-DEC')
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(ts.resample('Y-DEC').mean(), result)
        tm.assert_series_equal(ts.resample('Y').mean(), result)

    @pytest.mark.parametrize('rule,expected_error_msg', [('Y-DEC', '<YearEnd: month=12>'), ('Q-MAR', '<QuarterEnd: startingMonth=3>'), ('M', '<MonthEnd>'), ('W-THU', '<Week: weekday=3>')])
    def test_not_subperiod(self, simple_period_range_series: Callable[[str, str, str], Series], rule: str, expected_error_msg: str) -> None:
        ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='W-WED')
        msg = f'Frequency <Week: weekday=2> cannot be resampled to {expected_error_msg}, as they are not sub or super periods'
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts.resample(rule).mean()

    @pytest.mark.parametrize('freq', ['D', '2D'])
    def test_basic_upsample(self, freq: str, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='M')
        result = ts.resample('Y-DEC').mean()
        msg = "The 'convention' keyword in Series.resample is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            resampled = result.resample(freq, convention='end').ffill()
        expected = result.to_timestamp(freq, how='end')
        expected = expected.asfreq(freq, 'ffill').to_period(freq)
        tm.assert_series_equal(resampled, expected)

    def test_upsample_with_limit(self) -> None:
        rng = period_range('1/1/2000', periods=5, freq='Y')
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        msg = "The 'convention' keyword in Series.resample is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ts.resample('M', convention='end').ffill(limit=2)
        expected = ts.asfreq('M').reindex(result.index, method='ffill', limit=2)
        tm.assert_series_equal(result, expected)

    def test_annual_upsample(self, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        ts = simple_period_range_series('1/1/1990', '12/31/1995', freq='Y-DEC')
        df = DataFrame({'a': ts})
        rdf = df.resample('D').ffill()
        exp = df['a'].resample('D').ffill()
        tm.assert_series_equal(rdf['a'], exp)

    def test_annual_upsample2(self) -> None:
        rng = period_range('2000', '2003', freq='Y-DEC')
        ts = Series([1, 2, 3, 4], index=rng)
        result = ts.resample('M').ffill()
        ex_index = period_range('2000-01', '2003-12', freq='M')
        expected = ts.asfreq('M', how='start').reindex(ex_index, method='ffill')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('month', MONTHS)
    @pytest.mark.parametrize('convention', ['start', 'end'])
    @pytest.mark.parametrize(('offset', 'period'), [('D', 'D'), ('B', 'B'), ('ME', 'M')])
    def test_quarterly_upsample(self, month: str, offset: str, period: str, convention: str, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        freq = f'Q-{month}'
        ts = simple_period_range_series('1/1/1990', '12/31/1995', freq=freq)
        warn: Optional[Type[Warning]] = FutureWarning if period == 'B' else None
        msg = 'PeriodDtype\\[B\\] is deprecated'
        if warn is None:
            msg = 'Resampling with a PeriodIndex is deprecated'
            warn = FutureWarning
        with tm.assert_produces_warning(warn, match=msg):
            result = ts.resample(period, convention=convention).ffill()
            expected = result.to_timestamp(period, how=convention)
            expected = expected.asfreq(offset, 'ffill').to_period()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('target', ['D', 'B'])
    @pytest.mark.parametrize('convention', ['start', 'end'])
    def test_monthly_upsample(self, target: str, convention: str, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        ts = simple_period_range_series('1/1/1990', '12/31/1995', freq='M')
        warn: Optional[Type[Warning]] = None if target == 'D' else FutureWarning
        msg = 'PeriodDtype\\[B\\] is deprecated'
        if warn is None:
            msg = 'Resampling with a PeriodIndex is deprecated'
            warn = FutureWarning
        with tm.assert_produces_warning(warn, match=msg):
            result = ts.resample(target, convention=convention).ffill()
            expected = result.to_timestamp(target, how=convention)
            expected = expected.asfreq(target, 'ffill').to_period()
        tm.assert_series_equal(result, expected)

    def test_resample_basic(self) -> None:
        s = Series(range(100), index=date_range('20130101', freq='s', periods=100, name='idx'), dtype='float')
        s[10:30] = np.nan
        index = PeriodIndex([Period('2013-01-01 00:00', 'min'), Period('2013-01-01 00:01', 'min')], name='idx')
        expected = Series([34.5, 79.5], index=index)
        result = s.to_period().resample('min').mean()
        tm.assert_series_equal(result, expected)
        result2 = s.resample('min').mean().to_period()
        tm.assert_series_equal(result2, expected)

    @pytest.mark.parametrize('freq,expected_vals', [('M', [31, 29, 31, 9]), ('2M', [31 + 29, 31 + 9])])
    def test_resample_count(self, freq: str, expected_vals: list[int], ) -> None:
        series = Series(1, index=period_range(start='2000', periods=100))
        result = series.resample(freq).count()
        expected_index = period_range(start='2000', freq=freq, periods=len(expected_vals))
        expected = Series(expected_vals, index=expected_index)
        tm.assert_series_equal(result, expected)

    def test_resample_same_freq(self, resample_method: str) -> None:
        series = Series(range(3), index=period_range(start='2000', periods=3, freq='M'))
        expected = series
        result = getattr(series.resample('M'), resample_method)()
        tm.assert_series_equal(result, expected)

    def test_resample_incompat_freq(self) -> None:
        msg = 'Frequency <MonthEnd> cannot be resampled to <Week: weekday=6>, as they are not sub or super periods'
        pi = period_range(start='2000', periods=3, freq='M')
        ser = Series(range(3), index=pi)
        rs = ser.resample('W')
        with pytest.raises(IncompatibleFrequency, match=msg):
            rs.mean()

    @pytest.mark.parametrize('tz', [zoneinfo.ZoneInfo('America/Los_Angeles'), dateutil.tz.gettz('America/Los_Angeles')])
    def test_with_local_timezone(self, tz: zoneinfo.ZoneInfo | dateutil.tz.tzfile, ) -> None:
        local_timezone: Union[zoneinfo.ZoneInfo, dateutil.tz.tzfile] = tz
        start = datetime(year=2013, month=11, day=1, hour=0, minute=0, tzinfo=timezone.utc)
        end = datetime(year=2013, month=11, day=2, hour=0, minute=0, tzinfo=timezone.utc)
        index = date_range(start, end, freq='h', name='idx')
        series = Series(1, index=index)
        series = series.tz_convert(local_timezone)
        msg = 'Converting to PeriodArray/Index representation will drop timezone'
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = series.resample('D').mean().to_period()
        expected_index = period_range(start=start, end=end, freq='D', name='idx') - offsets.Day()
        expected = Series(1.0, index=expected_index)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', [zoneinfo.ZoneInfo('America/Los_Angeles'), dateutil.tz.gettz('America/Los_Angeles')])
    def test_resample_with_tz(self, tz: zoneinfo.ZoneInfo | dateutil.tz.tzfile, unit: Any) -> None:
        dti = date_range('2017-01-01', periods=48, freq='h', tz=tz, unit=unit)
        ser = Series(2, index=dti)
        result = ser.resample('D').mean()
        exp_dti = pd.DatetimeIndex(['2017-01-01', '2017-01-02'], tz=tz, freq='D').as_unit(unit)
        expected = Series(2.0, index=exp_dti)
        tm.assert_series_equal(result, expected)

    def test_resample_nonexistent_time_bin_edge(self) -> None:
        index = date_range('2017-03-12', '2017-03-12 1:45:00', freq='15min')
        s = Series(np.zeros(len(index)), index=index)
        expected = s.tz_localize('US/Pacific')
        expected.index = pd.DatetimeIndex(expected.index, freq='900s')
        result = expected.resample('900s').mean()
        tm.assert_series_equal(result, expected)

    def test_resample_nonexistent_time_bin_edge2(self) -> None:
        index = date_range(start='2017-10-10', end='2017-10-20', freq='1h')
        index = index.tz_localize('UTC').tz_convert('America/Sao_Paulo')
        df = DataFrame(data=list(range(len(index))), index=index)
        result = df.groupby(pd.Grouper(freq='1D')).count()
        expected = date_range(start='2017-10-09', end='2017-10-20', freq='D', tz='America/Sao_Paulo', nonexistent='shift_forward', inclusive='left')
        tm.assert_index_equal(result.index, expected)

    def test_resample_ambiguous_time_bin_edge(self) -> None:
        idx = date_range('2014-10-25 22:00:00', '2014-10-26 00:30:00', freq='30min', tz='Europe/London')
        expected = Series(np.zeros(len(idx)), index=idx)
        result = expected.resample('30min').mean()
        tm.assert_series_equal(result, expected)

    def test_fill_method_and_how_upsample(self) -> None:
        s = Series(np.arange(9, dtype='int64'), index=date_range('2010-01-01', periods=9, freq='QE'))
        last = s.resample('ME').ffill()
        both = s.resample('ME').ffill().resample('ME').last().astype('int64')
        tm.assert_series_equal(last, both)

    @pytest.mark.parametrize('day', DAYS)
    @pytest.mark.parametrize('target', ['D', 'B'])
    @pytest.mark.parametrize('convention', ['start', 'end'])
    def test_weekly_upsample(self, day: str, target: str, convention: str, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        freq = f'W-{day}'
        ts = simple_period_range_series('1/1/1990', '12/31/1995', freq=freq)
        warn: Optional[Type[Warning]] = None if target == 'D' else FutureWarning
        msg = 'PeriodDtype\\[B\\] is deprecated'
        if warn is None:
            msg = 'Resampling with a PeriodIndex is deprecated'
            warn = FutureWarning
        with tm.assert_produces_warning(warn, match=msg):
            result = ts.resample(target, convention=convention).ffill()
            expected = result.to_timestamp(target, how=convention)
            expected = expected.asfreq(target, 'ffill').to_period()
        tm.assert_series_equal(result, expected)

    def test_resample_to_timestamps(self, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        ts = simple_period_range_series('1/1/1990', '12/31/1995', freq='M')
        result = ts.resample('Y-DEC').mean().to_timestamp()
        expected = ts.resample('Y-DEC').mean().to_timestamp(how='start')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('month', MONTHS)
    def test_resample_to_quarterly(self, simple_period_range_series: Callable[[str, str, str], Series], month: str) -> None:
        ts = simple_period_range_series('1990', '1992', freq=f'Y-{month}')
        quar_ts = ts.resample(f'Q-{month}').ffill()
        stamps = ts.to_timestamp('D', how='start')
        qdates = period_range(ts.index[0].asfreq('D', 'start'), ts.index[-1].asfreq('D', 'end'), freq=f'Q-{month}')
        expected = stamps.reindex(qdates.to_timestamp('D', 's'), method='ffill')
        expected.index = qdates
        tm.assert_series_equal(quar_ts, expected)

    @pytest.mark.parametrize('how', ['start', 'end'])
    def test_resample_to_quarterly_start_end(self, simple_period_range_series: Callable[[str, str, str], Series], how: str) -> None:
        ts = simple_period_range_series('1990', '1992', freq='Y-JUN')
        msg = "The 'convention' keyword in Series.resample is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ts.resample('Q-MAR', convention=how).ffill()
        expected = ts.asfreq('Q-MAR', how=how)
        expected = expected.reindex(result.index, method='ffill')
        tm.assert_series_equal(result, expected)

    def test_resample_fill_missing(self) -> None:
        rng = PeriodIndex([2000, 2005, 2007, 2009], freq='Y')
        s = Series(np.random.default_rng(2).standard_normal(4), index=rng)
        stamps = s.to_timestamp()
        filled = s.resample('Y').ffill()
        expected = stamps.resample('YE').ffill().to_period('Y')
        tm.assert_series_equal(filled, expected)

    def test_cant_fill_missing_dups(self) -> None:
        rng = PeriodIndex([2000, 2005, 2005, 2007, 2007], freq='Y')
        s = Series(np.random.default_rng(2).standard_normal(5), index=rng)
        msg = 'Reindexing only valid with uniquely valued Index objects'
        with pytest.raises(InvalidIndexError, match=msg):
            s.resample('Y').ffill()

    def test_resample_5minute(self) -> None:
        rng = period_range('1/1/2000', '1/5/2000', freq='min')
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        expected = ts.to_timestamp().resample('5min').mean()
        result = ts.resample('5min').mean().to_timestamp()
        tm.assert_series_equal(result, expected)
        expected = expected.to_period('5min')
        result = ts.resample('5min').mean()
        tm.assert_series_equal(result, expected)
        result = ts.resample('5min').mean().to_timestamp().to_period()
        tm.assert_series_equal(result, expected)

    def test_upsample_daily_business_daily(self, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        ts = simple_period_range_series('1/1/2000', '2/1/2000', freq='B')
        result = ts.resample('D').asfreq()
        expected = ts.asfreq('D').reindex(period_range('1/3/2000', '2/1/2000'))
        tm.assert_series_equal(result, expected)
        ts = simple_period_range_series('1/1/2000', '2/1/2000')
        msg = "The 'convention' keyword in Series.resample is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ts.resample('h', convention='s').asfreq()
        exp_rng = period_range('1/1/2000', '2/1/2000 23:00', freq='h')
        expected = ts.asfreq('h', how='s').reindex(exp_rng)
        tm.assert_series_equal(result, expected)

    def test_resample_irregular_sparse(self) -> None:
        dr = date_range(start='1/1/2012', freq='5min', periods=1000)
        s = Series(np.array(100), index=dr)
        subset = s[:'2012-01-04 06:55']
        result = subset.resample('10min').apply(len)
        expected = s.resample('10min').apply(len).loc[result.index]
        tm.assert_series_equal(result, expected)

    def test_resample_weekly_all_na(self) -> None:
        rng = date_range('1/1/2000', periods=10, freq='W-WED')
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        result = ts.resample('W-THU').asfreq()
        assert result.isna().all()
        result = ts.resample('W-THU').asfreq().ffill()[:-1]
        expected = ts.asfreq('W-THU').ffill()
        tm.assert_series_equal(result, expected)

    def test_resample_tz_localized(self, unit: Any) -> None:
        dr = date_range(start='2012-4-13', end='2012-5-1', unit=unit)
        ts = Series(range(len(dr)), index=dr)
        ts_utc = ts.tz_localize('UTC')
        ts_local = ts_utc.tz_convert('America/Los_Angeles')
        result = ts_local.resample('W').mean()
        ts_local_naive = ts_local.copy()
        ts_local_naive.index = ts_local_naive.index.tz_localize(None)
        exp = ts_local_naive.resample('W').mean().tz_localize('America/Los_Angeles')
        exp.index = pd.DatetimeIndex(exp.index, freq='W')
        tm.assert_series_equal(result, exp)
        result = ts_local.resample('D').mean()

    def test_resample_tz_localized2(self) -> None:
        idx = date_range('2001-09-20 15:59', '2001-09-20 16:00', freq='min', tz='Australia/Sydney')
        s = Series([1, 2], index=idx)
        result = s.resample('D', closed='right', label='right').mean()
        ex_index = date_range('2001-09-21', periods=1, freq='D', tz='Australia/Sydney')
        expected = Series([1.5], index=ex_index)
        tm.assert_series_equal(result, expected)
        msg = 'Converting to PeriodArray/Index representation will drop timezone '
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = s.resample('D').mean().to_period()
        ex_index = period_range('2001-09-20', periods=1, freq='D')
        expected = Series([1.5], index=ex_index)
        tm.assert_series_equal(result, expected)

    def test_resample_tz_localized3(self) -> None:
        rng = date_range('1/1/2011', periods=20000, freq='h')
        rng = rng.tz_localize('EST')
        ts = DataFrame(index=rng)
        ts['first'] = np.random.default_rng(2).standard_normal(len(rng))
        ts['second'] = np.cumsum(np.random.default_rng(2).standard_normal(len(rng)))
        expected = DataFrame({'first': ts.resample('YE').sum()['first'], 'second': ts.resample('YE').mean()['second']}, columns=['first', 'second'])
        result = ts.resample('YE').agg({'first': 'sum', 'second': 'mean'}).reindex(columns=['first', 'second'])
        tm.assert_frame_equal(result, expected)

    def test_closed_left_corner(self) -> None:
        s = Series(np.random.default_rng(2).standard_normal(21), index=date_range(start='1/1/2012 9:30', freq='1min', periods=21))
        s.iloc[0] = np.nan
        result = s.resample('10min', closed='left', label='right').mean()
        exp = s[1:].resample('10min', closed='left', label='right').mean()
        tm.assert_series_equal(result, exp)
        result = s.resample('10min', closed='left', label='left').mean()
        exp = s[1:].resample('10min', closed='left', label='left').mean()
        ex_index = date_range(start='1/1/2012 9:30', freq='10min', periods=3)
        tm.assert_index_equal(result.index, ex_index)
        tm.assert_series_equal(result, exp)

    def test_quarterly_resampling(self) -> None:
        rng = period_range('2000Q1', periods=10, freq='Q-DEC')
        ts = Series(np.arange(10), index=rng)
        result = ts.resample('Y').mean()
        exp = ts.to_timestamp().resample('YE').mean().to_period()
        tm.assert_series_equal(result, exp)

    def test_resample_weekly_bug_1726(self) -> None:
        ind = date_range(start='8/6/2012', end='8/26/2012', freq='D')
        n = len(ind)
        data = [[x] * 5 for x in range(n)]
        df = DataFrame(data, columns=['open', 'high', 'low', 'close', 'vol'], index=ind)
        df.resample('W-MON', closed='left', label='left').first()

    def test_resample_with_dst_time_change(self) -> None:
        index = pd.DatetimeIndex([1457537600000000000, 1458059600000000000]).tz_localize('UTC').tz_convert('America/Chicago')
        df = DataFrame([1, 2], index=index)
        result = df.resample('12h', closed='right', label='right').last().ffill()
        expected_index_values = [
            '2016-03-09 12:00:00-06:00',
            '2016-03-10 00:00:00-06:00',
            '2016-03-10 12:00:00-06:00',
            '2016-03-11 00:00:00-06:00',
            '2016-03-11 12:00:00-06:00',
            '2016-03-12 00:00:00-06:00',
            '2016-03-12 12:00:00-06:00',
            '2016-03-13 00:00:00-06:00',
            '2016-03-13 13:00:00-05:00',
            '2016-03-14 01:00:00-05:00',
            '2016-03-14 13:00:00-05:00',
            '2016-03-15 01:00:00-05:00',
            '2016-03-15 13:00:00-05:00'
        ]
        index = pd.to_datetime(expected_index_values, utc=True).tz_convert('America/Chicago').as_unit(index.unit)
        index = pd.DatetimeIndex(index, freq='12h')
        expected = DataFrame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], index=index)
        tm.assert_frame_equal(result, expected)

    def test_resample_bms_2752(self) -> None:
        timeseries = Series(index=pd.bdate_range('20000101', '20000201'), dtype=np.float64)
        res1 = timeseries.resample('BMS').mean()
        res2 = timeseries.resample('BMS').mean().resample('B').mean()
        assert res1.index[0] == Timestamp('20000103')
        assert res1.index[0] == res2.index[0]

    @pytest.mark.xfail(reason='Commented out for more than 3 years. Should this work?')
    def test_monthly_convention_span(self) -> None:
        rng = period_range('2000-01', periods=3, freq='ME')
        ts = Series(np.arange(3), index=rng)
        exp_index = period_range('2000-01-01', '2000-03-31', freq='D')
        expected = ts.asfreq('D', how='end').reindex(exp_index)
        expected = expected.fillna(method='bfill')
        result = ts.resample('D').mean()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('from_freq, to_freq', [('D', 'ME'), ('QE', 'YE'), ('ME', 'QE'), ('D', 'W')])
    def test_default_right_closed_label(self, from_freq: str, to_freq: str, frame_or_series: Union[DataFrame, Series]) -> None:
        idx = date_range(start='8/15/2012', periods=100, freq=from_freq)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), idx)
        resampled = df.resample(to_freq).mean()
        tm.assert_frame_equal(resampled, df.resample(to_freq, closed='right', label='right').mean())

    @pytest.mark.parametrize('from_freq, to_freq', [('D', 'MS'), ('QE', 'YS'), ('ME', 'QS'), ('h', 'D'), ('min', 'h')])
    def test_default_left_closed_label(self, from_freq: str, to_freq: str, frame_or_series: Union[DataFrame, Series]) -> None:
        idx = date_range(start='8/15/2012', periods=100, freq=from_freq)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), idx)
        resampled = df.resample(to_freq).mean()
        tm.assert_frame_equal(resampled, df.resample(to_freq, closed='left', label='left').mean())

    def test_all_values_single_bin(self, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        index = period_range(start='2012-01-01', end='2012-12-31', freq='M')
        ser = Series(np.random.default_rng(2).standard_normal(len(index)), index=index)
        result = ser.resample('Y').mean()
        tm.assert_almost_equal(result.iloc[0], ser.mean())

    def test_evenly_divisible_with_no_extra_bins(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((9, 3)), index=date_range('2000-1-1', periods=9))
        result = df.resample('5D').mean()
        expected = pd.concat([df.iloc[0:5].mean(), df.iloc[5:].mean()], axis=1).T
        expected.index = pd.DatetimeIndex([Timestamp('2000-1-1'), Timestamp('2000-1-6')], dtype='M8[ns]', freq='5D')
        tm.assert_frame_equal(result, expected)

    def test_evenly_divisible_with_no_extra_bins2(self) -> None:
        index = date_range(start='2001-5-4', periods=28)
        df = DataFrame([{'REST_KEY': 1, 'DLY_TRN_QT': 80, 'DLY_SLS_AMT': 90, 'COOP_DLY_TRN_QT': 30, 'COOP_DLY_SLS_AMT': 20}] * 28 + [{'REST_KEY': 2, 'DLY_TRN_QT': 70, 'DLY_SLS_AMT': 10, 'COOP_DLY_TRN_QT': 50, 'COOP_DLY_SLS_AMT': 20}] * 28, index=index.append(index)).sort_index()
        index = date_range('2001-5-4', periods=4, freq='7D')
        expected = DataFrame([{'REST_KEY': 14, 'DLY_TRN_QT': 14, 'DLY_SLS_AMT': 14, 'COOP_DLY_TRN_QT': 14, 'COOP_DLY_SLS_AMT': 14}] * 4, index=index)
        result = df.resample('7D').count()
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([{'REST_KEY': 21, 'DLY_TRN_QT': 1050, 'DLY_SLS_AMT': 700, 'COOP_DLY_TRN_QT': 560, 'COOP_DLY_SLS_AMT': 280}] * 4, index=index)
        result = df.resample('7D').sum()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('freq, period_mult', [('h', 24), ('12h', 2)])
    def test_upsampling_ohlc(self, freq: str, period_mult: int) -> None:
        pi = period_range(start='2000', freq='D', periods=10)
        s = Series(range(len(pi)), index=pi)
        expected = s.to_timestamp().resample(freq).ohlc().to_period(freq)
        new_index = period_range(start='2000', freq=freq, periods=period_mult * len(pi))
        expected = expected.reindex(new_index)
        result = s.resample(freq).ohlc()
        tm.assert_frame_equal(result, expected)
        result = s.resample(freq).ohlc().to_timestamp().to_period()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('periods, values', [
        ([pd.NaT, '1970-01-01 00:00:00', pd.NaT, '1970-01-01 00:00:02', '1970-01-01 00:00:03'], [2, 3, 5, 7, 11]),
        ([pd.NaT, pd.NaT, '1970-01-01 00:00:00', pd.NaT, pd.NaT, pd.NaT, '1970-01-01 00:00:02', '1970-01-01 00:00:03', pd.NaT, pd.NaT], [1, 2, 3, 5, 6, 8, 7, 11, 12, 13])
    ])
    @pytest.mark.parametrize('freq, expected_values', [('1s', [3, np.nan, 7, 11]), ('2s', [3, (7 + 11) / 2]), ('3s', [(3 + 7) / 2, 11])])
    def test_resample_with_nat(self, periods: list[Optional[str]], values: list[int], freq: str, expected_values: list[float]) -> None:
        index = PeriodIndex(periods, freq='s')
        frame = DataFrame(values, index=index)
        expected_index = period_range('1970-01-01 00:00:00', periods=len(expected_values), freq=freq)
        expected = DataFrame(expected_values, index=expected_index)
        msg = 'Resampling with a PeriodIndex is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = frame.resample(freq)
        result = rs.mean()
        tm.assert_frame_equal(result, expected)

    def test_resample_with_only_nat(self) -> None:
        pi = PeriodIndex([pd.NaT] * 3, freq='s')
        frame = DataFrame([2, 3, 5], index=pi, columns=['a'])
        expected_index = PeriodIndex(data=[], freq=pi.freq)
        expected = DataFrame(index=expected_index, columns=['a'], dtype='float64')
        result = frame.resample('1s').mean()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('start,end,start_freq,end_freq,offset', [
        ('19910905', '19910909 03:00', 'h', '24h', '10h'),
        ('19910905', '19910909 12:00', 'h', '24h', '10h'),
        ('19910905', '19910909 23:00', 'h', '24h', '10h'),
        ('19910905 10:00', '19910909', 'h', '24h', '10h'),
        ('19910905 10:00', '19910909 10:00', 'h', '24h', '10h'),
        ('19910905', '19910909 10:00', 'h', '24h', '10h'),
        ('19910905 12:00', '19910909', 'h', '24h', '10h'),
        ('19910905 12:00', '19910909 03:00', 'h', '24h', '10h'),
        ('19910905 12:00', '19910909 12:00', 'h', '24h', '10h'),
        ('19910905 12:00', '19910909 12:00', 'h', '24h', '34h'),
        ('19910905 12:00', '19910909 12:00', 'h', '17h', '10h'),
        ('19910905 12:00', '19910909 12:00', 'h', '17h', '3h'),
        ('19910905', '19910913 06:00', '2h', '24h', '10h'),
        ('19910905', '19910905 01:39', 'Min', '5Min', '3Min'),
        ('19910905', '19910905 03:18', '2Min', '5Min', '3Min')
    ])
    def test_resample_with_offset(self, start: str, end: str, start_freq: str, end_freq: str, offset: str, simple_period_range_series: Callable[[str, str, str], Series]) -> None:
        pi = period_range(start, end, freq=start_freq)
        ser = Series(np.arange(len(pi)), index=pi)
        msg = 'Resampling with a PeriodIndex is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = ser.resample(end_freq, offset=offset)
        result = rs.mean()
        result = result.to_timestamp(end_freq)
        expected = ser.to_timestamp().resample(end_freq, offset=offset).mean()
        tm.assert_series_equal(result, expected)

    def test_resample_with_offset_month(self) -> None:
        pi = period_range('19910905 12:00', '19910909 1:00', freq='h')
        ser = Series(np.arange(len(pi)), index=pi)
        msg = 'Resampling with a PeriodIndex is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = ser.resample('M', offset='3h')
        result = rs.mean()
        result = result.to_timestamp('M')
        expected = ser.to_timestamp().resample('ME', offset='3h').mean()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('first,last,freq,freq_to_offset,exp_first,exp_last', [
        ('19910905', '19910909', 'D', 'D', '19910905', '19910909'),
        ('19910905 00:00', '19920406 06:00', 'D', 'D', '19910905', '19920406'),
        ('19910905 06:00', '19920406 06:00', 'h', 'h', '19910905 06:00', '19920406 06:00'),
        ('19910906', '19920406', 'M', 'ME', '1991-09', '1992-04'),
        ('19910831', '19920430', 'M', 'ME', '1991-08', '1992-04'),
        ('1991-08', '1992-04', 'M', 'ME', '1991-08', '1992-04')
    ])
    def test_get_period_range_edges(self, first: str, last: str, freq: str, freq_to_offset: str, exp_first: str, exp_last: str) -> None:
        first_p = Period(first)
        last_p = Period(last)
        exp_first_p = Period(exp_first, freq=freq)
        exp_last_p = Period(exp_last, freq=freq)
        freq_offset = pd.tseries.frequencies.to_offset(freq_to_offset)
        result = _get_period_range_edges(first_p, last_p, freq_offset)
        expected = (exp_first_p, exp_last_p)
        assert result == expected

    def test_sum_min_count(self) -> None:
        index = date_range(start='2018', freq='ME', periods=6)
        data = np.ones(6)
        data[3:6] = np.nan
        s = Series(data, index).to_period()
        msg = 'Resampling with a PeriodIndex is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = s.resample('Q')
        result = rs.sum(min_count=1)
        expected = Series([3.0, np.nan], index=PeriodIndex(['2018Q1', '2018Q2'], freq='Q-DEC'))
        tm.assert_series_equal(result, expected)

    def test_resample_t_l_deprecated(self) -> None:
        msg_t = 'Invalid frequency: T'
        msg_l = 'Invalid frequency: L'
        with pytest.raises(ValueError, match=msg_l):
            period_range('2020-01-01 00:00:00 00:00', '2020-01-01 00:00:00 00:01', freq='L')
        rng_l = period_range('2020-01-01 00:00:00 00:00', '2020-01-01 00:00:00 00:01', freq='ms')
        ser = Series(np.arange(len(rng_l)), index=rng_l)
        with pytest.raises(ValueError, match=msg_t):
            ser.resample('T').mean()

    @pytest.mark.parametrize('freq, freq_depr, freq_depr_res', [
        ('2Q', '2q', '2y'),
        ('2M', '2m', '2q')
    ])
    def test_resample_lowercase_frequency_raises(self, freq: str, freq_depr: str, freq_depr_res: str, frame_or_series: Union[DataFrame, Series]) -> None:
        msg = f'Invalid frequency: {freq_depr}'
        with pytest.raises(ValueError, match=msg):
            period_range('2020-01-01', '2020-08-01', freq=freq_depr)
        msg = f'Invalid frequency: {freq_depr_res}'
        rng = period_range('2020-01-01', '2020-08-01', freq=freq)
        ser = Series(np.arange(len(rng)), index=rng)
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq_depr_res).mean()

    @pytest.mark.parametrize('offset', [offsets.MonthBegin(), offsets.BYearBegin(2), offsets.BusinessHour(2)])
    def test_asfreq_invalid_period_offset(self, offset: offsets.DateOffset, frame_or_series: Union[DataFrame, Series]) -> None:
        msg = re.escape(f'{offset} is not supported as period frequency')
        obj = frame_or_series(range(5), index=period_range('2020-01-01', periods=5))
        with pytest.raises(ValueError, match=msg):
            obj.asfreq(freq=offset)

@pytest.mark.parametrize('freq', ['2ME', '2QE', '2QE-FEB', '2YE', '2YE-MAR', '2me', '2qe', '2ye-mar'])
def test_resample_frequency_ME_QE_YE_raises(frame_or_series: Union[DataFrame, Series], freq: str) -> None:
    msg = f'{freq[1:]} is not supported as period frequency'
    obj = frame_or_series(range(5), index=period_range('2020-01-01', periods=5))
    msg = f'Invalid frequency: {freq}'
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq)

def test_corner_cases_period(simple_period_range_series: Callable[[str, str, str], Series]) -> None:
    len0pts = simple_period_range_series('2007-01', '2010-05', freq='M')[:0]
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = len0pts.resample('Y-DEC').mean()
    assert len(result) == 0

@pytest.mark.parametrize('freq', ['2BME', '2CBME', '2SME', '2BQE-FEB', '2BYE-MAR'])
def test_resample_frequency_invalid_freq(frame_or_series: Union[DataFrame, Series], freq: str) -> None:
    msg = f'Invalid frequency: {freq}'
    obj = frame_or_series(range(5), index=period_range('2020-01-01', periods=5))
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq)
