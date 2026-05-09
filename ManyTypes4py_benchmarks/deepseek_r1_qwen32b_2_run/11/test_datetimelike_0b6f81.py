"""Test cases for time series specific (freq conversion, etc)"""
from datetime import date, datetime, time, timedelta
import pickle
import numpy as np
import pytest
from pandas._libs.tslibs import BaseOffset, to_offset
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import DataFrame, Index, NaT, Series, concat, isna, to_datetime
import pandas._testing as tm
from pandas.core.indexes.datetimes import DatetimeIndex, bdate_range, date_range
from pandas.core.indexes.period import Period, PeriodIndex, period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props
from pandas.tseries.offsets import WeekOfMonth
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas.plotting._matplotlib.converter as conv
from typing import Any, List, Tuple, Optional, Union, Dict, Callable

class TestTSPlot:

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ts_plot_with_tz(self, tz_aware_fixture: Any) -> None:
        tz = tz_aware_fixture
        index = date_range('1/1/2011', periods=2, freq='h', tz=tz)
        ts = Series([188.5, 328.25], index=index)
        _check_plot_works(ts.plot)
        ax = ts.plot()
        xdata = next(iter(ax.get_lines())).get_xdata()
        assert (xdata[0].hour, xdata[0].minute) == (0, 0)
        assert (xdata[-1].hour, xdata[-1].minute) == (1, 0)

    def test_fontsize_set_correctly(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        df.plot(fontsize=2, ax=ax)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            assert label.get_fontsize() == 2

    def test_frame_inferred(self, idx: DatetimeIndex, df: DataFrame) -> None:
        _check_plot_works(df.plot)
        idx = idx[0:4].union(idx[6:])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
        _check_plot_works(df2.plot)

    def test_frame_inferred_n_gt_1(self, idx: DatetimeIndex, df: DataFrame) -> None:
        _check_plot_works(df.plot)

    def test_is_error_nozeroindex(self, a: DataFrame) -> None:
        _check_plot_works(a.plot, xerr=a)
        _check_plot_works(a.plot, yerr=a)

    def test_nonnumeric_exclude(self, df: DataFrame) -> None:
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        assert len(ax.get_lines()) == 1

    def test_nonnumeric_exclude_error(self, df: DataFrame) -> None:
        msg = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            df['A'].plot()

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'M', 'Q', 'Y'])
    def test_tsplot_period(self, freq: str, ser: Series) -> None:
        _, ax = plt.subplots()
        _check_plot_works(ser.plot, ax=ax)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_tsplot_datetime(self, freq: str, ser: Series) -> None:
        _, ax = plt.subplots()
        _check_plot_works(ser.plot, ax=ax)

    def test_tsplot(self, ts: Series) -> None:
        _, ax = plt.subplots()
        ts.plot(style='k', ax=ax)
        color = (0.0, 0.0, 0.0, 1)
        assert color == ax.get_lines()[0].get_color()

    @pytest.mark.parametrize('index', [None, date_range('2020-01-01', periods=10)])
    def test_both_style_and_color(self, index: Optional[DatetimeIndex], ts: Series) -> None:
        msg = "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
        with pytest.raises(ValueError, match=msg):
            ts.plot(style='b-', color='#000099')

    @pytest.mark.parametrize('freq', ['ms', 'us'])
    def test_high_freq(self, freq: str, ser: Series) -> None:
        _, ax = plt.subplots()
        _check_plot_works(ser.plot, ax=ax)

    def test_get_datevalue(self) -> None:
        assert conv.get_datevalue(None, 'D') is None
        assert conv.get_datevalue(1987, 'Y') == 1987
        assert conv.get_datevalue(Period(1987, 'Y'), 'M') == Period('1987-12', 'M').ordinal
        assert conv.get_datevalue('1/1/1987', 'D') == Period('1987-1-1', 'D').ordinal

    @pytest.mark.parametrize('freq, expected_string', [['YE-DEC', 't = 2014  y = 1.000000'], ['D', 't = 2014-01-01  y = 1.000000']])
    def test_ts_plot_format_coord(self, freq: str, expected_string: str, ser: Series) -> None:
        _, ax = plt.subplots()
        ser.plot(ax=ax)
        first_line = ax.get_lines()[0]
        first_x = first_line.get_xdata()[0].ordinal
        first_y = first_line.get_ydata()[0]
        assert expected_string == ax.format_coord(first_x, first_y)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'M', 'Q', 'Y'])
    def test_line_plot_period_series(self, freq: str, ser: Series) -> None:
        _check_plot_works(ser.plot, ser.index.freq)

    @pytest.mark.parametrize('frqncy', ['1s', '3s', '5min', '7h', '4D', '8W', '11M', '3Y'])
    def test_line_plot_period_mlt_series(self, frqncy: str, s: Series) -> None:
        _check_plot_works(s.plot, s.index.freq.rule_code)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_line_plot_datetime_series(self, freq: str, ser: Series) -> None:
        _check_plot_works(ser.plot, ser.index.freq.rule_code)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE', 'YE'])
    def test_line_plot_period_frame(self, freq: str, df: DataFrame) -> None:
        _check_plot_works(df.plot, df.index.freq)

    @pytest.mark.parametrize('frqncy', ['1s', '3s', '5min', '7h', '4D', '8W', '11M', '3Y'])
    def test_line_plot_period_mlt_frame(self, frqncy: str, df: DataFrame) -> None:
        freq = df.index.freq.rule_code
        _check_plot_works(df.plot, freq)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_line_plot_datetime_frame(self, freq: str, df: DataFrame) -> None:
        freq = PeriodDtype(df.index.freq)._freqstr
        freq = df.index.to_period(freq).freq
        _check_plot_works(df.plot, freq)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_line_plot_inferred_freq(self, freq: str, ser: Series) -> None:
        ser = Series(ser.values, Index(np.asarray(ser.index)))
        _check_plot_works(ser.plot, ser.index.inferred_freq)
        ser = ser.iloc[[0, 3, 5, 6]]
        _check_plot_works(ser.plot)

    def test_fake_inferred_business(self) -> None:
        _, ax = plt.subplots()
        rng = date_range('2001-1-1', '2001-1-10')
        ts = Series(range(len(rng)), index=rng)
        ts = concat([ts[:3], ts[5:]])
        ts.plot(ax=ax)
        assert not hasattr(ax, 'freq')

    def test_plot_offset_freq(self, ser: Series) -> None:
        _check_plot_works(ser.plot)

    def test_plot_offset_freq_business(self, dr: DatetimeIndex, ser: Series) -> None:
        _check_plot_works(ser.plot)

    def test_plot_multiple_inferred_freq(self, dr: DatetimeIndex, ser: Series) -> None:
        _check_plot_works(ser.plot)

    def test_irreg_hf(self, df: DataFrame, irreg: DataFrame) -> None:
        _, ax = plt.subplots()
        irreg.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - [sec, sec * 2, sec]) < 1e-08).all()

    def test_irreg_hf_object(self, df2: DataFrame) -> None:
        _, ax = plt.subplots()
        df2.index = df2.index.astype(object)
        df2.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - sec) < 1e-08).all()

    def test_irregular_datetime64_repr_bug(self, ser: Series) -> None:
        ser = ser.iloc[[0, 1, 2, 7]]
        _, ax = plt.subplots()
        ret = ser.plot(ax=ax)
        assert ret is not None
        for rs, xp in zip(ax.get_lines()[0].get_xdata(), ser.index):
            assert rs == xp

    def test_business_freq(self, bts: Series) -> None:
        msg = 'PeriodDtype\\[B\\] is deprecated'
        dt = bts.index[0].to_timestamp()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            bts.index = period_range(start=dt, periods=len(bts), freq='B')
        _, ax = plt.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == bts.index[0].ordinal
        idx = ax.get_lines()[0].get_xdata()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert PeriodIndex(data=idx).freqstr == 'B'

    def test_business_freq_convert(self, bts: Series, ts: Series) -> None:
        _, ax = plt.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == ts.index[0].ordinal
        idx = ax.get_lines()[0].get_xdata()
        assert PeriodIndex(data=idx).freqstr == 'M'

    def test_freq_with_no_period_alias(self, freq: WeekOfMonth, bts: Series) -> None:
        _, ax = plt.subplots()
        bts.plot(ax=ax)
        idx = ax.get_lines()[0].get_xdata()
        msg = 'freq not specified and cannot be inferred'
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(data=idx)

    def test_nonzero_base(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        df.plot(ax=ax)
        rs = ax.get_lines()[0].get_xdata()
        assert not Index(rs).is_normalized

    def test_dataframe(self, bts: DataFrame) -> None:
        _, ax = plt.subplots()
        bts.plot(ax=ax)
        idx = ax.get_lines()[0].get_xdata()
        tm.assert_index_equal(bts.index.to_period(), PeriodIndex(idx))

    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    @pytest.mark.parametrize('obj', [Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)), DataFrame({'a': Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)), 'b': Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)) + 1})])
    def test_axis_limits(self, obj: Union[Series, DataFrame]) -> None:
        _, ax = plt.subplots()
        obj.plot(ax=ax)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - 5, xlim[1] + 10)
        result = ax.get_xlim()
        assert result[0] == xlim[0] - 5
        assert result[1] == xlim[1] + 10
        expected = (Period('1/1/2000', ax.freq), Period('4/1/2000', ax.freq))
        ax.set_xlim('1/1/2000', '4/1/2000')
        result = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal
        expected = (Period('1/1/2000', ax.freq), Period('4/1/2000', ax.freq))
        ax.set_xlim(datetime(2000, 1, 1), datetime(2000, 4, 1))
        result = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal

    def test_get_finder(self) -> None:
        assert conv.get_finder(to_offset('B')) == conv._daily_finder
        assert conv.get_finder(to_offset('D')) == conv._daily_finder
        assert conv.get_finder(to_offset('ME')) == conv._monthly_finder
        assert conv.get_finder(to_offset('QE')) == conv._quarterly_finder
        assert conv.get_finder(to_offset('YE')) == conv._annual_finder
        assert conv.get_finder(to_offset('W')) == conv._daily_finder

    def test_finder_daily(self) -> None:
        day_lst = [10, 40, 252, 400, 950, 2750, 10000]
        msg = 'Period with BDay freq is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            xpl1 = xpl2 = [Period('1999-1-1', freq='B').ordinal] * len(day_lst)
        rs1 = []
        rs2 = []
        for n in day_lst:
            rng = bdate_range('1999-1-1', periods=n)
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = plt.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])
            vmin, vmax = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            plt.close(ax.get_figure())
        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_quarterly(self) -> None:
        yrs = [3.5, 11]
        xpl1 = xpl2 = [Period('1988Q1').ordinal] * len(yrs)
        rs1 = []
        rs2 = []
        for n in yrs:
            rng = period_range('1987Q2', periods=int(n * 4), freq='Q')
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = plt.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])
            vmin, vmax = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            plt.close(ax.get_figure())
        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly(self) -> None:
        yrs = [1.15, 2.5, 4, 11]
        xpl1 = xpl2 = [Period('Jan 1988').ordinal] * len(yrs)
        rs1 = []
        rs2 = []
        for n in yrs:
            rng = period_range('1987Q2', periods=int(n * 12), freq='M')
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = plt.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])
            vmin, vmax = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            plt.close(ax.get_figure())
        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly_long(self) -> None:
        rng = period_range('1988Q1', periods=24 * 12, freq='M')
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = plt.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period('1989Q1', 'M').ordinal
        assert rs == xp

    def test_finder_annual(self) -> None:
        xp = [1987, 1988, 1990, 1990, 1995, 2020, 2070, 2170]
        xp = [Period(x, freq='Y').ordinal for x in xp]
        rs = []
        for nyears in [5, 10, 19, 49, 99, 199, 599, 1001]:
            rng = period_range('1987', periods=nyears, freq='Y')
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = plt.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs.append(xaxis.get_majorticklocs()[0])
            plt.close(ax.get_figure())
        assert rs == xp

    @pytest.mark.slow
    def test_finder_minutely(self) -> None:
        nminutes = 1 * 24 * 60
        rng = date_range('1/1/1999', freq='Min', periods=nminutes)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = plt.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period('1/1/1999', freq='Min').ordinal
        assert rs == xp

    def test_finder_hourly(self) -> None:
        nhours = 23
        rng = date_range('1/1/1999', freq='h', periods=nhours)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = plt.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period('1/1/1999', freq='h').ordinal
        assert rs == xp

    def test_gaps(self, ts: Series) -> None:
        ts.iloc[5:7] = np.nan
        _, ax = plt.subplots()
        ts.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[5:7, 1].all()

    def test_gaps_irregular(self, ts: Series) -> None:
        ts = ts.iloc[[0, 1, 2, 5, 7, 9, 12, 15, 20]]
        ts.iloc[2:5] = np.nan
        _, ax = plt.subplots()
        ax = ts.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[2:5, 1].all()

    def test_gaps_non_ts(self, ser: Series) -> None:
        idx = [0, 1, 2, 5, 7, 9, 12, 15, 20]
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser.iloc[2:5] = np.nan
        _, ax = plt.subplots()
        ser.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[2:5, 1].all()

    def test_gap_upsample(self, low: Series, idxh: DatetimeIndex, s: Series) -> None:
        low.iloc[5:7] = np.nan
        _, ax = plt.subplots()
        low.plot(ax=ax)
        s.plot(secondary_y=True)
        lines = ax.get_lines()
        assert len(lines) == 1
        assert len(ax.right_ax.get_lines()) == 1
        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[5:7, 1].all()

    def test_secondary_y(self, ser: Series) -> None:
        fig, _ = plt.subplots()
        ax = ser.plot(secondary_y=True)
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        axes = fig.get_axes()
        line = ax.get_lines()[0]
        xp = Series(line.get_ydata(), line.get_xdata())
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == 'right'
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_yaxis(self, ser2: Series) -> None:
        _, ax2 = plt.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == 'left'

    def test_secondary_both(self, ser: Series, ser2: Series) -> None:
        ax = ser2.plot()
        ax2 = ser.plot(secondary_y=True)
        assert ax.get_yaxis().get_visible()
        assert not hasattr(ax, 'left_ax')
        assert hasattr(ax, 'right_ax')
        assert hasattr(ax2, 'left_ax')
        assert not hasattr(ax2, 'right_ax')

    def test_secondary_y_ts(self, ser: Series) -> None:
        fig, _ = plt.subplots()
        ax = ser.plot(secondary_y=True)
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        axes = fig.get_axes()
        line = ax.get_lines()[0]
        xp = Series(line.get_ydata(), line.get_xdata()).to_timestamp()
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == 'right'
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_ts_yaxis(self, ser2: Series) -> None:
        _, ax2 = plt.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == 'left'

    def test_secondary_y_ts_visible(self, ser2: Series) -> None:
        ax = ser2.plot()
        assert ax.get_yaxis().get_visible()

    def test_secondary_kde(self, ser: Series) -> None:
        pytest.importorskip('scipy')
        fig, ax = plt.subplots()
        ax = ser.plot(secondary_y=True, kind='density', ax=ax)
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        axes = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == 'right'

    def test_secondary_bar(self, ser: Series) -> None:
        fig, ax = plt.subplots()
        ser.plot(secondary_y=True, kind='bar', ax=ax)
        axes = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == 'right'

    def test_secondary_frame(self, df: DataFrame) -> None:
        axes = df.plot(secondary_y=['a', 'c'], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == 'right'
        assert axes[1].get_yaxis().get_ticks_position() == 'left'
        assert axes[2].get_yaxis().get_ticks_position() == 'right'

    def test_secondary_bar_frame(self, df: DataFrame) -> None:
        axes = df.plot(kind='bar', secondary_y=['a', 'c'], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == 'right'
        assert axes[1].get_yaxis().get_ticks_position() == 'left'
        assert axes[2].get_yaxis().get_ticks_position() == 'right'

    def test_mixed_freq_regular_first(self, s1: Series, s2: Series) -> None:
        _, ax = plt.subplots()
        s1.plot(ax=ax)
        ax2 = s2.plot(style='g', ax=ax)
        lines = ax2.get_lines()
        msg = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())
            tm.assert_index_equal(idx1, s1.index.to_period('B'))
            tm.assert_index_equal(idx2, s2.index.to_period('B'))
            left, right = ax2.get_xlim()
            pidx = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first(self, s1: Series, s2: Series) -> None:
        _, ax = plt.subplots()
        s2.plot(style='g', ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, 'freq')
        lines = ax.get_lines()
        x1 = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2 = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_regular_first_df(self, s1: DataFrame, s2: DataFrame) -> None:
        _, ax = plt.subplots()
        s1.plot(ax=ax)
        ax2 = s2.plot(style='g', ax=ax)
        lines = ax2.get_lines()
        msg = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())
            assert idx1.equals(s1.index.to_period('B'))
            assert idx2.equals(s2.index.to_period('B'))
            left, right = ax2.get_xlim()
            pidx = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first_df(self, s1: DataFrame, s2: DataFrame) -> None:
        _, ax = plt.subplots()
        s2.plot(style='g', ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, 'freq')
        lines = ax.get_lines()
        x1 = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2 = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_hf_first(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'D'

    def test_mixed_freq_alignment(self, ts: Series, ts2: Series) -> None:
        _, ax = plt.subplots()
        ax = ts.plot(ax=ax)
        ts2.plot(style='r', ax=ax)
        assert ax.lines[0].get_xdata()[0] == ax.lines[1].get_xdata()[0]

    def test_mixed_freq_lf_first(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        low.plot(legend=True, ax=ax)
        high.plot(legend=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'D'
        leg = ax.get_legend()
        assert len(leg.texts) == 2
        plt.close(ax.get_figure())

    def test_mixed_freq_lf_first_hourly(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'min'

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_mixed_freq_irreg_period(self, ts: Series, irreg: Series, ps: Series) -> None:
        msg = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            irreg.plot()
            ps.plot()

    def test_mixed_freq_shared_ax(self, idx1: DatetimeIndex, idx2: DatetimeIndex, s1: Series, s2: Series) -> None:
        _, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)
        assert ax1.freq == 'M'
        assert ax2.freq == 'M'
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_mixed_freq_shared_ax_twin_x(self, idx1: DatetimeIndex, idx2: DatetimeIndex, s1: Series, s2: Series) -> None:
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    @pytest.mark.xfail(reason='TODO (GH14330, GH14322)')
    def test_mixed_freq_shared_ax_twin_x_irregular_first(self, idx1: DatetimeIndex, idx2: DatetimeIndex, s1: Series, s2: Series) -> None:
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        s2.plot(ax=ax1)
        s1.plot(ax=ax2)
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_nat_handling(self) -> None:
        _, ax = plt.subplots()
        dti = DatetimeIndex(['2015-01-01', NaT, '2015-01-03'])
        s = Series(range(len(dti)), dti)
        s.plot(ax=ax)
        xdata = ax.get_lines()[0].get_xdata()
        assert s.index.min() <= Series(xdata).min()
        assert Series(xdata).max() <= s.index.max()

    def test_to_weekly_resampling_disallow_how_kwd(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        high.plot(ax=ax)
        msg = "'how' is not a valid keyword for plotting functions. If plotting multiple objects on shared axes, resample manually first."
        with pytest.raises(ValueError, match=msg):
            low.plot(ax=ax, how='foo')

    def test_to_weekly_resampling(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == high.index.freq

    def test_from_weekly_resampling(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        expected_h = high.index.to_period().asi8.astype(np.float64)
        expected_l = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == high.index.freq
            xdata = line.get_xdata(orig=False)
            if len(xdata) == 12:
                tm.assert_numpy_array_equal(xdata, expected_l)
            else:
                tm.assert_numpy_array_equal(xdata, expected_h)

    @pytest.mark.parametrize('kind1, kind2', [('line', 'area'), ('area', 'line')])
    def test_from_resampling_area_line_mixed(self, kind1: str, kind2: str, high: DataFrame, low: DataFrame) -> None:
        _, ax = plt.subplots()
        low.plot(kind=kind1, stacked=True, ax=ax)
        high.plot(kind=kind2, stacked=True, ax=ax)
        expected_x = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[i]
            assert PeriodIndex(line.get_xdata()).freq == high.index.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += low[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)
        expected_x = high.index.to_period().asi8.astype(np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[3 + i]
            assert PeriodIndex(data=line.get_xdata()).freq == high.index.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

    @pytest.mark.parametrize('kind1, kind2', [('line', 'area'), ('area', 'line')])
    def test_from_resampling_area_line_mixed_high_to_low(self, kind1: str, kind2: str, high: DataFrame, low: DataFrame) -> None:
        _, ax = plt.subplots()
        high.plot(kind=kind1, stacked=True, ax=ax)
        low.plot(kind=kind2, stacked=True, ax=ax)
        expected_x = high.index.to_period().asi8.astype(np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[i]
            assert PeriodIndex(data=line.get_xdata()).freq == high.index.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)
        expected_x = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            lines = ax.lines[3 + i]
            assert PeriodIndex(data=lines.get_xdata()).freq == high.index.freq
            tm.assert_numpy_array_equal(lines.get_xdata(orig=False), expected_x)
            expected_y += low[i].values
            tm.assert_numpy_array_equal(lines.get_ydata(orig=False), expected_y)

    def test_mixed_freq_second_millisecond(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'ms'

    def test_mixed_freq_second_millisecond_low_to_high(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'ms'

    def test_irreg_dtypes(self, df: DataFrame) -> None:
        _check_plot_works(df.plot)

    def test_irreg_dtypes_dt64(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        _check_plot_works(df.plot, ax=ax)

    def test_time(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        df.plot(ax=ax)
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime('%H:%M:%S')
                else:
                    xp = time(h, m, s).strftime('%H:%M')
                assert xp == rs

    def test_time_change_xlim(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        df.plot(ax=ax)
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime('%H:%M:%S')
                else:
                    xp = time(h, m, s).strftime('%H:%M')
                assert xp == rs
        ax.set_xlim('1:30', '5:00')
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime('%H:%M:%S')
                else:
                    xp = time(h, m, s).strftime('%H:%M')
                assert xp == rs

    def test_time_musec(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        ax = df.plot(ax=ax)
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            us = round((_tick - int(_tick)) * 1000000.0)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if us % 1000 != 0:
                    xp = time(h, m, s, us).strftime('%H:%M:%S.%f')
                elif us // 1000 != 0:
                    xp = time(h, m, s, us).strftime('%H:%M:%S.%f')[:-3]
                elif s != 0:
                    xp = time(h, m, s, us).strftime('%H:%M:%S')
                else:
                    xp = time(h, m, s, us).strftime('%H:%M')
                assert xp == rs

    def test_secondary_upsample(self, high: Series, low: Series) -> None:
        _, ax = plt.subplots()
        low.plot(ax=ax)
        ax = high.plot(secondary_y=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == 'D'
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        for line in ax.left_ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == 'D'

    def test_secondary_legend(self, df: DataFrame) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        df.plot(secondary_y=['A', 'B'], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == 'A (right)'
        assert leg.get_texts()[1].get_text() == 'B (right)'
        assert leg.get_texts()[2].get_text() == 'C'
        assert leg.get_texts()[3].get_text() == 'D'
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    def test_secondary_legend_right(self, df: DataFrame) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        df.plot(secondary_y=['A', 'C'], mark_right=False, ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == 'A'
        assert leg.get_texts()[1].get_text() == 'B'
        assert leg.get_texts()[2].get_text() == 'C'
        assert leg.get_texts()[3].get_text() == 'D'

    def test_secondary_legend_bar(self, df: DataFrame) -> None:
        fig, ax = plt.subplots()
        df.plot(kind='bar', secondary_y=['A'], ax=ax)
        leg = ax.get_legend()
        assert leg.get_texts()[0].get_text() == 'A (right)'
        assert leg.get_texts()[1].get_text() == 'B'

    def test_secondary_legend_bar_right(self, df: DataFrame) -> None:
        fig, ax = plt.subplots()
        df.plot(kind='bar', secondary_y=['A'], mark_right=False, ax=ax)
        leg = ax.get_legend()
        assert leg.get_texts()[0].get_text() == 'A'
        assert leg.get_texts()[1].get_text() == 'B'

    def test_secondary_legend_multi_col(self, df: DataFrame) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        df.plot(secondary_y=['C', 'D'], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    def test_secondary_legend_nonts(self, df: DataFrame) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax = df.plot(secondary_y=['A', 'B'], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    def test_secondary_legend_nonts_multi_col(self, df: DataFrame) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax = df.plot(secondary_y=['C', 'D'], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_format_date_axis(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        ax = df.plot(ax=ax)
        xaxis = ax.get_xaxis()
        for line in xaxis.get_ticklabels():
            if len(line.get_text()) > 0:
                assert line.get_rotation() == 30

    def test_ax_plot(self, x: DatetimeIndex, y: List[float]) -> None:
        _, ax = plt.subplots()
        lines = ax.plot(x, y, label='Y')
        tm.assert_index_equal(DatetimeIndex(lines[0].get_xdata()), x)

    def test_mpl_nopandas(self, dates: List[date], values1: np.ndarray, values2: np.ndarray) -> None:
        _, ax = plt.subplots()
        line1, line2 = ax.plot([x.toordinal() for x in dates], values1, '-', [x.toordinal() for x in dates], values2, '-', linewidth=4)
        exp = np.array([x.toordinal() for x in dates], dtype=np.float64)
        tm.assert_numpy_array_equal(line1.get_xydata()[:, 0], exp)
        tm.assert_numpy_array_equal(line2.get_xydata()[:, 0], exp)

    def test_irregular_ts_shared_ax_xlim(self, ts: Series, ts_irregular: Series) -> None:
        _, ax = plt.subplots()
        ts_irregular[:5].plot(ax=ax)
        ts_irregular[5:].plot(ax=ax)
        left, right = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), '', ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), '', ax)

    def test_secondary_y_non_ts_xlim(self, s1: Series, s2: Series) -> None:
        _, ax = plt.subplots()
        s1.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()
        assert left_before >= left_after
        assert right_before < right_after

    def test_secondary_y_regular_ts_xlim(self, s1: Series, s2: Series) -> None:
        _, ax = plt.subplots()
        s1.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()
        assert left_before == left_after
        assert right_before == right_after

    def test_secondary_y_mixed_freq_ts_xlim(self, ts: Series) -> None:
        _, ax = plt.subplots()
        ts.plot(ax=ax)
        ts.resample('D').mean().plot(secondary_y=True, ax=ax)
        left, right = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts.index.min(), '', ax)
        assert right >= conv.DatetimeConverter.convert(ts.index.max(), '', ax)

    def test_secondary_y_irregular_ts_xlim(self, ts_irregular: Series) -> None:
        _, ax = plt.subplots()
        ts_irregular[:5].plot(ax=ax)
        ts_irregular[5:].plot(secondary_y=True, ax=ax)
        ts_irregular[:5].plot(ax=ax)
        left, right = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), '', ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), '', ax)

    def test_plot_outofbounds_datetime(self) -> None:
        values = [date(1677, 1, 1), date(1677, 1, 2)]
        _, ax = plt.subplots()
        ax.plot(values)
        values = [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]
        ax.plot(values)

    def test_format_timedelta_ticks_narrow(self, df: DataFrame) -> None:
        expected_labels = [f'00:00:00.0000000{i:0>2d}' for i in np.arange(10)]
        _, ax = plt.subplots()
        df.plot(fontsize=2, ax=ax)
        plt.draw()
        labels = ax.get_xticklabels()
        result_labels = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_format_timedelta_ticks_wide(self, df: DataFrame) -> None:
        expected_labels = ['00:00:00', '1 days 03:46:40', '2 days 07:33:20', '3 days 11:20:00', '4 days 15:06:40', '5 days 18:53:20', '6 days 22:40:00', '8 days 02:26:40', '9 days 06:13:20']
        _, ax = plt.subplots()
        ax = df.plot(fontsize=2, ax=ax)
        plt.draw()
        labels = ax.get_xticklabels()
        result_labels = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_timedelta_plot(self, s: Series) -> None:
        _, ax = plt.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_long_period(self, s: Series) -> None:
        _, ax = plt.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_short_period(self, s: Series) -> None:
        _, ax = plt.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_hist(self, x: DatetimeIndex) -> None:
        w1 = np.arange(0, 1, 0.1)
        w2 = np.arange(0, 1, 0.1)[::-1]
        _, ax = plt.subplots()
        ax.hist([x, x], weights=[w1, w2])

    def test_overlapping_datetime(self, s1: Series, s2: Series) -> None:
        s1.plot()
        s2.plot()
        s1.plot()

    @pytest.mark.xfail(reason='GH9053 matplotlib does not use ax.xaxis.converter')
    def test_add_matplotlib_datetime64(self, s: Series) -> None:
        ax = s.plot()
        with tm.assert_produces_warning(DeprecationWarning):
            ax.plot(s.index, s.values, color='g')
        l1, l2 = ax.lines
        tm.assert_numpy_array_equal(l1.get_xydata(), l2.get_xydata())

    def test_matplotlib_scatter_datetime64(self, df: DataFrame) -> None:
        _, ax = plt.subplots()
        ax.scatter(x='time', y='y', data=df)
        plt.draw()
        label = ax.get_xticklabels()[0]
        expected = '2018-01-01'
        assert label.get_text() == expected

    def test_check_xticks_rot(self, df: DataFrame) -> None:
        axes = df.plot(x='x', y='y')
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_irregular(self, df: DataFrame) -> None:
        axes = df.plot(x='x', y='y')
        _check_ticks_props(axes, xrot=30)

    def test_check_xticks_rot_use_idx(self, df: DataFrame) -> None:
        axes = df.set_index('x').plot(y='y', use_index=True)
        _check_ticks_props(axes, xrot=30)
        axes = df.set_index('x').plot(y='y', use_index=False)
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_sharex(self, df: DataFrame) -> None:
        axes = df.plot(x='x', y='y', subplots=True, sharex=True)
        _check_ticks_props(axes, xrot=30)
        axes = df.plot(x='x', y='y', subplots=True, sharex=False)
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize('idx', [date_range('2020-01-01', periods=5), date_range('2020-01-01', periods=5, tz='UTC'), timedelta_range('1 day', periods=5, freq='D'), period_range('2020-01-01', periods=5, freq='D'), Index([date(2000, 1, i) for i in [1, 3, 6, 20, 22]], dtype=object), range(5)])
    def test_pickle_fig(self, temp_file: str, frame_or_series: Union[DataFrame, Series], idx: Union[DatetimeIndex, PeriodIndex, Index, List[int]]) -> None:
        df = frame_or_series(range(5), index=idx)
        fig, ax = plt.subplots(1, 1)
        df.plot(ax=ax)
        with open(temp_file, 'wb') as path:
            pickle.dump(fig, path)

def _check_plot_works(f: Callable, freq: Optional[BaseOffset] = None, series: Optional[Series] = None, *args: Any, **kwargs: Any) -> None:
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(211)
    orig_ax = kwargs.pop('ax', plt.gca())
    orig_axfreq = getattr(orig_ax, 'freq', None)
    ret = f(*args, **kwargs)
    assert ret is not None
    ax = kwargs.pop('ax', plt.gca())
    if series is not None:
        dfreq = series.index.freq
        if isinstance(dfreq, BaseOffset):
            dfreq = dfreq.rule_code
        if orig_axfreq is None:
            assert ax.freq == dfreq
    if freq is not None and orig_axfreq is None:
        assert to_offset(ax.freq, is_period=True) == freq
    ax = fig.add_subplot(212)
    kwargs['ax'] = ax
    ret = f(*args, **kwargs)
    assert ret is not None