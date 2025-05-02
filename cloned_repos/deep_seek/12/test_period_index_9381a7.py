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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pandas._libs.tslibs.offsets import BaseOffset
from pandas.core.indexes.base import Index
from pandas.core.resample import Resampler

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
    def test_asfreq(self, frame_or_series: Callable, freq: str) -> None:
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
    def test_selection(self, freq: str, kwargs: Dict[str, str]) -> None:
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
    def test_annual_upsample_cases(self, offset: str, period: str, conv: str, meth: str, month: str, simple_period_range_series: Callable) -> None:
        ts = simple_period_range_series('1/1/1990', '12/31/1991', freq=f'Y-{month}')
        warn = FutureWarning if period == 'B' else None
        msg = 'PeriodDtype\\[B\\] is deprecated'
        if warn is None:
            msg = 'Resampling with a PeriodIndex is deprecated'
            warn = FutureWarning
        with tm.assert_produces_warning(warn, match=msg):
            result = getattr(ts.resample(period, convention=conv), meth)()
            expected = result.to_timestamp(period, how=conv)
            expected = expected.asfreq(offset, meth).to_period()
        tm.assert_series_equal(result, expected)

    def test_basic_downsample(self, simple_period_range_series: Callable) -> None:
        ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='M')
        result = ts.resample('Y-DEC').mean()
        expected = ts.groupby(ts.index.year).mean()
        expected.index = period_range('1/1/1990', '6/30/1995', freq='Y-DEC')
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(ts.resample('Y-DEC').mean(), result)
        tm.assert_series_equal(ts.resample('Y').mean(), result)

    @pytest.mark.parametrize('rule,expected_error_msg', [('Y-DEC', '<YearEnd: month=12>'), ('Q-MAR', '<QuarterEnd: startingMonth=3>'), ('M', '<MonthEnd>'), ('W-THU', '<Week: weekday=3>')])
    def test_not_subperiod(self, simple_period_range_series: Callable, rule: str, expected_error_msg: str) -> None:
        ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='W-WED')
        msg = f'Frequency <Week: weekday=2> cannot be resampled to {expected_error_msg}, as they are not sub or super periods'
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts.resample(rule).mean()

    @pytest.mark.parametrize('freq', ['D', '2D'])
    def test_basic_upsample(self, freq: str, simple_period_range_series: Callable) -> None:
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

    def test_annual_upsample(self, simple_period_range_series: Callable) -> None:
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
    def test_quarterly_upsample(self, month: str, offset: str, period: str, convention: str, simple_period_range_series: Callable) -> None:
        freq = f'Q-{month}'
        ts = simple_period_range_series('1/1/1990', '12/31/1995', freq=freq)
        warn = FutureWarning if period == 'B' else None
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
    def test_monthly_upsample(self, target: str, convention: str, simple_period_range_series: Callable) -> None:
        ts = simple_period_range_series('1/1/1990', '12/31/1995', freq='M')
        warn = None if target == 'D' else FutureWarning
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
    def test_resample_count(self, freq: str, expected_vals: List[int]) -> None:
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
    def test_with_local_timezone(self, tz: Union[zoneinfo.ZoneInfo, dateutil.tz.tzfile]) -> None:
        local_timezone = tz
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
    def test_resample_with_tz(self, tz: Union[zoneinfo.ZoneInfo, dateutil.tz.tzfile], unit: str) -> None:
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

    def test_resample_ambiguous_time_bin_edge(self) ->