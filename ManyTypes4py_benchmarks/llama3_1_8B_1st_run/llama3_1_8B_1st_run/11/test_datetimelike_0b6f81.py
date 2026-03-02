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
mpl = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')
import pandas.plotting._matplotlib.converter as conv

class TestTSPlot:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ts_plot_with_tz(self, tz_aware_fixture: DatetimeIndex) -> None:
        """Test plotting a time series with a timezone."""
        tz: DatetimeIndex = tz_aware_fixture
        index: DatetimeIndex = date_range('1/1/2011', periods=2, freq='h', tz=tz)
        ts: Series = Series([188.5, 328.25], index=index)
        _check_plot_works(ts.plot)
        ax: plt.Axes = ts.plot()
        xdata: np.ndarray = next(iter(ax.get_lines())).get_xdata()
        assert (xdata[0].hour, xdata[0].minute) == (0, 0)
        assert (xdata[-1].hour, xdata[-1].minute) == (1, 0)

    def test_fontsize_set_correctly(self) -> None:
        """Test setting the font size correctly."""
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 9)), index=range(10))
        _, ax: plt.Axes = mpl.pyplot.subplots()
        df.plot(fontsize=2, ax=ax)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            assert label.get_fontsize() == 2

    def test_frame_inferred(self) -> None:
        """Test inferring the frame from the index."""
        idx: DatetimeIndex = date_range('1/1/1987', freq='MS', periods=10)
        idx: DatetimeIndex = DatetimeIndex(idx.values, freq=None)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
        _check_plot_works(df.plot)
        idx: Index = idx[0:4].union(idx[6:])
        df2: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
        _check_plot_works(df2.plot)

    def test_frame_inferred_n_gt_1(self) -> None:
        """Test inferring the frame from the index when n > 1."""
        idx: DatetimeIndex = date_range('2008-1-1 00:15:00', freq='15min', periods=10)
        idx: DatetimeIndex = DatetimeIndex(idx.values, freq=None)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
        _check_plot_works(df.plot)

    def test_is_error_nozeroindex(self) -> None:
        """Test that there is no error when the index is not zero."""
        i: np.ndarray = np.array([1, 2, 3])
        a: DataFrame = DataFrame(i, index=i)
        _check_plot_works(a.plot, xerr=a)
        _check_plot_works(a.plot, yerr=a)

    def test_nonnumeric_exclude(self) -> None:
        """Test excluding non-numeric data."""
        idx: DatetimeIndex = date_range('1/1/1987', freq='YE', periods=3)
        df: DataFrame = DataFrame({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]}, idx)
        fig, ax: plt.Axes = mpl.pyplot.subplots()
        df.plot(ax=ax)
        assert len(ax.get_lines()) == 1

    def test_nonnumeric_exclude_error(self) -> None:
        """Test that there is an error when trying to plot non-numeric data."""
        idx: DatetimeIndex = date_range('1/1/1987', freq='YE', periods=3)
        df: DataFrame = DataFrame({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]}, idx)
        msg: str = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            df['A'].plot()

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'M', 'Q', 'Y'])
    def test_tsplot_period(self, freq: str) -> None:
        """Test plotting a period series."""
        idx: PeriodIndex = period_range('12/31/1999', freq=freq, periods=10)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_tsplot_datetime(self, freq: str) -> None:
        """Test plotting a datetime series."""
        idx: DatetimeIndex = date_range('12/31/1999', freq=freq, periods=10)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    def test_tsplot(self) -> None:
        """Test plotting a time series."""
        ts: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ts.plot(style='k', ax=ax)
        color: tuple = (0.0, 0.0, 0.0, 1)
        assert color == ax.get_lines()[0].get_color()

    @pytest.mark.parametrize('index', [None, date_range('2020-01-01', periods=10)])
    def test_both_style_and_color(self, index: Index | None) -> None:
        """Test that both style and color cannot be used."""
        ts: Series = Series(np.arange(10, dtype=np.float64), index=index)
        msg: str = "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
        with pytest.raises(ValueError, match=msg):
            ts.plot(style='b-', color='#000099')

    @pytest.mark.parametrize('freq', ['ms', 'us'])
    def test_high_freq(self, freq: str) -> None:
        """Test plotting a high-frequency series."""
        _, ax: plt.Axes = mpl.pyplot.subplots()
        rng: DatetimeIndex = date_range('1/1/2012', periods=10, freq=freq)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _check_plot_works(ser.plot, ax=ax)

    def test_get_datevalue(self) -> None:
        """Test getting the date value."""
        assert conv.get_datevalue(None, 'D') is None
        assert conv.get_datevalue(1987, 'Y') == 1987
        assert conv.get_datevalue(Period(1987, 'Y'), 'M') == Period('1987-12', 'M').ordinal
        assert conv.get_datevalue('1/1/1987', 'D') == Period('1987-1-1', 'D').ordinal

    @pytest.mark.parametrize('freq, expected_string', [['YE-DEC', 't = 2014  y = 1.000000'], ['D', 't = 2014-01-01  y = 1.000000']])
    def test_ts_plot_format_coord(self, freq: str, expected_string: str) -> None:
        """Test the format coordinate."""
        ser: Series = Series(1, index=date_range('2014-01-01', periods=3, freq=freq))
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        first_line: plt.Line2D = ax.get_lines()[0]
        first_x: int = int(first_line.get_xdata()[0].ordinal)
        first_y: float = first_line.get_ydata()[0]
        assert expected_string == ax.format_coord(first_x, first_y)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'M', 'Q', 'Y'])
    def test_line_plot_period_series(self, freq: str) -> None:
        """Test plotting a period series."""
        idx: PeriodIndex = period_range('12/31/1999', freq=freq, periods=10)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq)

    @pytest.mark.parametrize('frqncy', ['1s', '3s', '5min', '7h', '4D', '8W', '11M', '3Y'])
    def test_line_plot_period_mlt_series(self, frqncy: str) -> None:
        """Test plotting a period series with multiple frequencies."""
        idx: PeriodIndex = period_range('12/31/1999', freq=frqncy, periods=10)
        s: Series = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(s.plot, s.index.freq.rule_code)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_line_plot_datetime_series(self, freq: str) -> None:
        """Test plotting a datetime series."""
        idx: DatetimeIndex = date_range('12/31/1999', freq=freq, periods=10)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq.rule_code)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE', 'YE'])
    def test_line_plot_period_frame(self, freq: str) -> None:
        """Test plotting a period frame."""
        idx: DatetimeIndex = date_range('12/31/1999', freq=freq, periods=10)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx, columns=['A', 'B', 'C'])
        _check_plot_works(df.plot, df.index.freq)

    @pytest.mark.parametrize('frqncy', ['1s', '3s', '5min', '7h', '4D', '8W', '11M', '3Y'])
    def test_line_plot_period_mlt_frame(self, frqncy: str) -> None:
        """Test plotting a period frame with multiple frequencies."""
        idx: PeriodIndex = period_range('12/31/1999', freq=frqncy, periods=10)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx, columns=['A', 'B', 'C'])
        freq: str = df.index.freq.rule_code
        _check_plot_works(df.plot, freq)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_line_plot_datetime_frame(self, freq: str) -> None:
        """Test plotting a datetime frame."""
        idx: DatetimeIndex = date_range('12/31/1999', freq=freq, periods=10)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx, columns=['A', 'B', 'C'])
        freq: str = PeriodDtype(df.index.freq)._freqstr
        freq: str = df.index.to_period(freq).freq
        _check_plot_works(df.plot, freq)

    @pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
    def test_line_plot_inferred_freq(self, freq: str) -> None:
        """Test plotting a series with inferred frequency."""
        idx: DatetimeIndex = date_range('12/31/1999', freq=freq, periods=10)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser: Series = Series(ser.values, Index(np.asarray(ser.index)))
        _check_plot_works(ser.plot, ser.index.inferred_freq)
        ser: Series = ser.iloc[[0, 3, 5, 6]]
        _check_plot_works(ser.plot)

    def test_fake_inferred_business(self) -> None:
        """Test plotting a series with a fake inferred business frequency."""
        _, ax: plt.Axes = mpl.pyplot.subplots()
        rng: DatetimeIndex = date_range('2001-1-1', '2001-1-10')
        ts: Series = Series(range(len(rng)), index=rng)
        ts: Series = concat([ts[:3], ts[5:]])
        ts.plot(ax=ax)
        assert not hasattr(ax, 'freq')

    def test_plot_offset_freq(self) -> None:
        """Test plotting a series with an offset frequency."""
        ser: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        _check_plot_works(ser.plot)

    def test_plot_offset_freq_business(self) -> None:
        """Test plotting a series with a business frequency."""
        dr: DatetimeIndex = date_range('2023-01-01', freq='BQS', periods=10)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        _check_plot_works(ser.plot)

    def test_plot_multiple_inferred_freq(self) -> None:
        """Test plotting a series with multiple inferred frequencies."""
        dr: Index = Index([datetime(2000, 1, 1), datetime(2000, 1, 6), datetime(2000, 1, 11)])
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        _check_plot_works(ser.plot)

    def test_irreg_hf(self) -> None:
        """Test plotting an irregular high-frequency series."""
        idx: DatetimeIndex = date_range('2012-6-22 21:59:51', freq='s', periods=10)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx)
        irreg: DataFrame = df.iloc[[0, 1, 3, 4]]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        irreg.plot(ax=ax)
        diffs: Series = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        sec: float = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - [sec, sec * 2, sec]) < 1e-08).all()

    def test_irreg_hf_object(self) -> None:
        """Test plotting an irregular high-frequency series with object dtype."""
        idx: DatetimeIndex = date_range('2012-6-22 21:59:51', freq='s', periods=10)
        df2: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        df2.index = df2.index.astype(object)
        df2.plot(ax=ax)
        diffs: Series = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        sec: float = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - sec) < 1e-08).all()

    def test_irregular_datetime64_repr_bug(self) -> None:
        """Test plotting an irregular datetime64 series."""
        ser: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        ser: Series = ser.iloc[[0, 1, 2, 7]]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ret: plt.Axes = ser.plot(ax=ax)
        assert ret is not None
        for rs, xp in zip(ax.get_lines()[0].get_xdata(), ser.index):
            assert rs == xp

    def test_business_freq(self) -> None:
        """Test plotting a series with a business frequency."""
        bts: Series = Series(range(5), period_range('2020-01-01', periods=5))
        msg: str = 'Period with BDay freq is deprecated'
        dt: datetime = bts.index[0].to_timestamp()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            bts.index = period_range(start=dt, periods=len(bts), freq='B')
        _, ax: plt.Axes = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == bts.index[0].ordinal
        idx: PeriodIndex = ax.get_lines()[0].get_xdata()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert PeriodIndex(data=idx).freqstr == 'B'

    def test_business_freq_convert(self) -> None:
        """Test converting a series with a business frequency."""
        bts: Series = Series(np.arange(50, dtype=np.float64), index=date_range('2020-01-01', periods=50, freq='B')).asfreq('BME')
        ts: Series = bts.to_period('M')
        _, ax: plt.Axes = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == ts.index[0].ordinal
        idx: PeriodIndex = ax.get_lines()[0].get_xdata()
        assert PeriodIndex(data=idx).freqstr == 'M'

    def test_freq_with_no_period_alias(self) -> None:
        """Test plotting a series with a frequency that has no period alias."""
        freq: BaseOffset = WeekOfMonth()
        bts: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)).asfreq(freq)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        idx: PeriodIndex = ax.get_lines()[0].get_xdata()
        msg: str = 'freq not specified and cannot be inferred'
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(data=idx)

    def test_nonzero_base(self) -> None:
        """Test plotting a series with a non-zero base."""
        idx: DatetimeIndex = date_range('2012-12-20', periods=24, freq='h') + timedelta(minutes=30)
        df: DataFrame = DataFrame(np.arange(24), index=idx)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        df.plot(ax=ax)
        rs: np.ndarray = ax.get_lines()[0].get_xdata()
        assert not Index(rs).is_normalized

    def test_dataframe(self) -> None:
        """Test plotting a DataFrame."""
        bts: DataFrame = DataFrame({'a': Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))})
        _, ax: plt.Axes = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        idx: PeriodIndex = ax.get_lines()[0].get_xdata()
        tm.assert_index_equal(bts.index.to_period(), PeriodIndex(idx))

    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    @pytest.mark.parametrize('obj', [Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)), DataFrame({'a': Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)), 'b': Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)) + 1})])
    def test_axis_limits(self, obj: Series | DataFrame) -> None:
        """Test setting the axis limits."""
        _, ax: plt.Axes = mpl.pyplot.subplots()
        obj.plot(ax=ax)
        xlim: tuple = ax.get_xlim()
        ax.set_xlim(xlim[0] - 5, xlim[1] + 10)
        result: tuple = ax.get_xlim()
        assert result[0] == xlim[0] - 5
        assert result[1] == xlim[1] + 10
        expected: tuple = (Period('1/1/2000', ax.freq), Period('4/1/2000', ax.freq))
        ax.set_xlim('1/1/2000', '4/1/2000')
        result: tuple = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal
        expected: tuple = (Period('1/1/2000', ax.freq), Period('4/1/2000', ax.freq))
        ax.set_xlim(datetime(2000, 1, 1), datetime(2000, 4, 1))
        result: tuple = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal

    def test_get_finder(self) -> None:
        """Test getting the finder."""
        assert conv.get_finder(to_offset('B')) == conv._daily_finder
        assert conv.get_finder(to_offset('D')) == conv._daily_finder
        assert conv.get_finder(to_offset('ME')) == conv._monthly_finder
        assert conv.get_finder(to_offset('QE')) == conv._quarterly_finder
        assert conv.get_finder(to_offset('YE')) == conv._annual_finder
        assert conv.get_finder(to_offset('W')) == conv._daily_finder

    def test_finder_daily(self) -> None:
        """Test the daily finder."""
        day_lst: list = [10, 40, 252, 400, 950, 2750, 10000]
        msg: str = 'Period with BDay freq is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            xpl1: list = [Period('1999-1-1', freq='B').ordinal] * len(day_lst)
            xpl2: list = [Period('1999-1-1', freq='B').ordinal] * len(day_lst)
        rs1: list = []
        rs2: list = []
        for n in day_lst:
            rng: DatetimeIndex = bdate_range('1999-1-1', periods=n)
            ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax: plt.Axes = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis: plt.TickHelper = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])
            vmin, vmax: float = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())
        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_quarterly(self) -> None:
        """Test the quarterly finder."""
        yrs: list = [3.5, 11]
        xpl1: list = [Period('1988Q1').ordinal] * len(yrs)
        xpl2: list = [Period('1988Q1').ordinal] * len(yrs)
        rs1: list = []
        rs2: list = []
        for n in yrs:
            rng: PeriodIndex = period_range('1987Q2', periods=int(n * 4), freq='Q')
            ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax: plt.Axes = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis: plt.TickHelper = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])
            vmin, vmax: float = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())
        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly(self) -> None:
        """Test the monthly finder."""
        yrs: list = [1.15, 2.5, 4, 11]
        xpl1: list = [Period('Jan 1988').ordinal] * len(yrs)
        xpl2: list = [Period('Jan 1988').ordinal] * len(yrs)
        rs1: list = []
        rs2: list = []
        for n in yrs:
            rng: PeriodIndex = period_range('1987Q2', periods=int(n * 12), freq='M')
            ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax: plt.Axes = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis: plt.TickHelper = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])
            vmin, vmax: float = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())
        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly_long(self) -> None:
        """Test the monthly finder with a long period."""
        rng: PeriodIndex = period_range('1988Q1', periods=24 * 12, freq='M')
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis: plt.TickHelper = ax.get_xaxis()
        rs: float = xaxis.get_majorticklocs()[0]
        xp: float = Period('1989Q1', 'M').ordinal
        assert rs == xp

    def test_finder_annual(self) -> None:
        """Test the annual finder."""
        xp: list = [1987, 1988, 1990, 1990, 1995, 2020, 2070, 2170]
        xp: list = [Period(x, freq='Y').ordinal for x in xp]
        rs: list = []
        for nyears in [5, 10, 19, 49, 99, 199, 599, 1001]:
            rng: PeriodIndex = period_range('1987', periods=nyears, freq='Y')
            ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax: plt.Axes = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis: plt.TickHelper = ax.get_xaxis()
            rs.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())
        assert rs == xp

    @pytest.mark.slow
    def test_finder_minutely(self) -> None:
        """Test the minutely finder."""
        nminutes: int = 1 * 24 * 60
        rng: DatetimeIndex = date_range('1/1/1999', freq='Min', periods=nminutes)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis: plt.TickHelper = ax.get_xaxis()
        rs: float = xaxis.get_majorticklocs()[0]
        xp: float = Period('1/1/1999', freq='Min').ordinal
        assert rs == xp

    def test_finder_hourly(self) -> None:
        """Test the hourly finder."""
        nhours: int = 23
        rng: DatetimeIndex = date_range('1/1/1999', freq='h', periods=nhours)
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis: plt.TickHelper = ax.get_xaxis()
        rs: float = xaxis.get_majorticklocs()[0]
        xp: float = Period('1/1/1999', freq='h').ordinal
        assert rs == xp

    def test_gaps(self) -> None:
        """Test plotting a series with gaps."""
        ts: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        ts.iloc[5:7] = np.nan
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ts.plot(ax=ax)
        lines: list = ax.get_lines()
        assert len(lines) == 1
        line: plt.Line2D = lines[0]
        data: np.ma.MaskedArray = np.ma.MaskedArray(line.get_xydata(), mask=isna(line.get_xydata()), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask: np.ndarray = data.mask
        assert mask[5:7, 1].all()

    def test_gaps_irregular(self) -> None:
        """Test plotting an irregular series with gaps."""
        ts: Series = Series(np.arange(30, dtype=np.float64), index=date_range('2020-01-01', periods=30))
        ts: Series = ts.iloc[[0, 1, 2, 5, 7, 9, 12, 15, 20]]
        ts.iloc[2:5] = np.nan
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ax: plt.Axes = ts.plot(ax=ax)
        lines: list = ax.get_lines()
        assert len(lines) == 1
        line: plt.Line2D = lines[0]
        data: np.ma.MaskedArray = np.ma.MaskedArray(line.get_xydata(), mask=isna(line.get_xydata()), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask: np.ndarray = data.mask
        assert mask[2:5, 1].all()

    def test_gaps_non_ts(self) -> None:
        """Test plotting a non-time series with gaps."""
        idx: list = [0, 1, 2, 5, 7, 9, 12, 15, 20]
        ser: Series = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser.iloc[2:5] = np.nan
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        lines: list = ax.get_lines()
        assert len(lines) == 1
        line: plt.Line2D = lines[0]
        data: np.ma.MaskedArray = np.ma.MaskedArray(line.get_xydata(), mask=isna(line.get_xydata()), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask: np.ndarray = data.mask
        assert mask[2:5, 1].all()

    def test_gap_upsample(self) -> None:
        """Test up-sampling a series with gaps."""
        low: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
        low.iloc[5:7] = np.nan
        _, ax: plt.Axes = mpl.pyplot.subplots()
        low.plot(ax=ax)
        idxh: DatetimeIndex = date_range(low.index[0], low.index[-1], freq='12h')
        s: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        s.plot(secondary_y=True)
        lines: list = ax.get_lines()
        assert len(lines) == 1
        assert len(ax.right_ax.get_lines()) == 1
        line: plt.Line2D = lines[0]
        data: np.ma.MaskedArray = np.ma.MaskedArray(line.get_xydata(), mask=isna(line.get_xydata()), fill_value=np.nan)
        assert isinstance(data, np.ma.core.MaskedArray)
        mask: np.ndarray = data.mask
        assert mask[5:7, 1].all()

    def test_secondary_y(self) -> None:
        """Test plotting a secondary y-axis."""
        ser: Series = Series(np.random.default_rng(2).standard_normal(10))
        fig, _ = mpl.pyplot.subplots()
        ax: plt.Axes = ser.plot(secondary_y=True)
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        axes: list = fig.get_axes()
        line: plt.Line2D = ax.get_lines()[0]
        xp: Series = Series(line.get_ydata(), line.get_xdata())
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == 'right'
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_yaxis(self) -> None:
        """Test plotting a secondary y-axis for a y-axis."""
        Series(np.random.default_rng(2).standard_normal(10))
        ser2: Series = Series(np.random.default_rng(2).standard_normal(10))
        _, ax2: plt.Axes = mpl.pyplot.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == 'left'

    def test_secondary_both(self) -> None:
        """Test plotting a secondary y-axis for both axes."""
        ser: Series = Series(np.random.default_rng(2).standard_normal(10))
        ser2: Series = Series(np.random.default_rng(2).standard_normal(10))
        ax: plt.Axes = ser2.plot()
        ax2: plt.Axes = ser.plot(secondary_y=True)
        assert ax.get_yaxis().get_visible()
        assert not hasattr(ax, 'left_ax')
        assert hasattr(ax, 'right_ax')
        assert hasattr(ax2, 'left_ax')
        assert not hasattr(ax2, 'right_ax')

    def test_secondary_y_ts(self) -> None:
        """Test plotting a secondary y-axis for a time series."""
        idx: DatetimeIndex = date_range('1/1/2000', periods=10)
        ser: Series = Series(np.random.default_rng(2).standard_normal(10), idx)
        fig, _ = mpl.pyplot.subplots()
        ax: plt.Axes = ser.plot(secondary_y=True)
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        axes: list = fig.get_axes()
        line: plt.Line2D = ax.get_lines()[0]
        xp: Series = Series(line.get_ydata(), line.get_xdata()).to_timestamp()
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == 'right'
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_ts_yaxis(self) -> None:
        """Test plotting a secondary y-axis for a time series y-axis."""
        idx: DatetimeIndex = date_range('1/1/2000', periods=10)
        ser2: Series = Series(np.random.default_rng(2).standard_normal(10), idx)
        _, ax2: plt.Axes = mpl.pyplot.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == 'left'

    def test_secondary_y_ts_visible(self) -> None:
        """Test plotting a secondary y-axis for a visible time series."""
        idx: DatetimeIndex = date_range('1/1/2000', periods=10)
        ser2: Series = Series(np.random.default_rng(2).standard_normal(10), idx)
        ax: plt.Axes = ser2.plot()
        assert ax.get_yaxis().get_visible()

    def test_secondary_kde(self) -> None:
        """Test plotting a secondary y-axis for a kernel density estimate."""
        pytest.importorskip('scipy')
        ser: Series = Series(np.random.default_rng(2).standard_normal(10))
        fig, ax: plt.Axes = mpl.pyplot.subplots()
        ax: plt.Axes = ser.plot(secondary_y=True, kind='density', ax=ax)
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        axes: list = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == 'right'

    def test_secondary_bar(self) -> None:
        """Test plotting a secondary y-axis for a bar plot."""
        ser: Series = Series(np.random.default_rng(2).standard_normal(10))
        fig, ax: plt.Axes = mpl.pyplot.subplots()
        ser.plot(secondary_y=True, kind='bar', ax=ax)
        axes: list = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == 'right'

    def test_secondary_frame(self) -> None:
        """Test plotting a secondary y-axis for a DataFrame."""
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a', 'b', 'c'])
        axes: list = df.plot(secondary_y=['a', 'c'], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == 'right'
        assert axes[1].get_yaxis().get_ticks_position() == 'left'
        assert axes[2].get_yaxis().get_ticks_position() == 'right'

    def test_secondary_bar_frame(self) -> None:
        """Test plotting a secondary y-axis for a bar plot DataFrame."""
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a', 'b', 'c'])
        axes: list = df.plot(kind='bar', secondary_y=['a', 'c'], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == 'right'
        assert axes[1].get_yaxis().get_ticks_position() == 'left'
        assert axes[2].get_yaxis().get_ticks_position() == 'right'

    def test_mixed_freq_regular_first(self) -> None:
        """Test plotting a mixed-frequency series with a regular first series."""
        s1: Series = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20, freq='B'))
        s2: Series = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        ax2: plt.Axes = s2.plot(style='g', ax=ax)
        lines: list = ax2.get_lines()
        msg: str = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1: PeriodIndex = PeriodIndex(lines[0].get_xdata())
            idx2: PeriodIndex = PeriodIndex(lines[1].get_xdata())
            tm.assert_index_equal(idx1, s1.index.to_period('B'))
            tm.assert_index_equal(idx2, s2.index.to_period('B'))
            left, right: float = ax2.get_xlim()
            pidx: PeriodIndex = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first(self) -> None:
        """Test plotting a mixed-frequency series with an irregular first series."""
        s1: Series = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20))
        s2: Series = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        s2.plot(style='g', ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, 'freq')
        lines: list = ax.get_lines()
        x1: np.ndarray = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2: np.ndarray = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_regular_first_df(self) -> None:
        """Test plotting a mixed-frequency DataFrame with a regular first series."""
        s1: Series = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20, freq='B')).to_frame()
        s2: Series = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        ax2: plt.Axes = s2.plot(style='g', ax=ax)
        lines: list = ax2.get_lines()
        msg: str = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1: PeriodIndex = PeriodIndex(lines[0].get_xdata())
            idx2: PeriodIndex = PeriodIndex(lines[1].get_xdata())
            assert idx1.equals(s1.index.to_period('B'))
            assert idx2.equals(s2.index.to_period('B'))
            left, right: float = ax2.get_xlim()
            pidx: PeriodIndex = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first_df(self) -> None:
        """Test plotting a mixed-frequency DataFrame with an irregular first series."""
        s1: Series = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20)).to_frame()
        s2: Series = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        s2.plot(style='g', ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, 'freq')
        lines: list = ax.get_lines()
        x1: np.ndarray = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2: np.ndarray = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_hf_first(self) -> None:
        """Test plotting a mixed-frequency series with a high-frequency first series."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=365, freq='D')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'D'

    def test_mixed_freq_alignment(self) -> None:
        """Test aligning a mixed-frequency series."""
        ts_ind: DatetimeIndex = date_range('2012-01-01 13:00', '2012-01-02', freq='h')
        ts_data: np.ndarray = np.random.default_rng(2).standard_normal(12)
        ts: Series = Series(ts_data, index=ts_ind)
        ts2: Series = ts.asfreq('min').interpolate()
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ax: plt.Axes = ts.plot(ax=ax)
        ts2.plot(style='r', ax=ax)
        assert ax.lines[0].get_xdata()[0] == ax.lines[1].get_xdata()[0]

    def test_mixed_freq_lf_first(self) -> None:
        """Test plotting a mixed-frequency series with a low-frequency first series."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=365, freq='D')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        low.plot(legend=True, ax=ax)
        high.plot(legend=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'D'
        leg: plt.Legend = ax.get_legend()
        assert len(leg.texts) == 2
        mpl.pyplot.close(ax.get_figure())

    def test_mixed_freq_lf_first_hourly(self) -> None:
        """Test plotting a mixed-frequency series with a low-frequency first series and hourly frequency."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=240, freq='min')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=4, freq='h')
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'min'

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_mixed_freq_irreg_period(self) -> None:
        """Test plotting a mixed-frequency series with an irregular period."""
        ts: Series = Series(np.arange(30, dtype=np.float64), index=date_range('2020-01-01', periods=30))
        irreg: Series = ts.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 29]]
        msg: str = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rng: PeriodIndex = period_range('1/3/2000', periods=30, freq='B')
        ps: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        irreg.plot(ax=ax)
        ps.plot(ax=ax)

    def test_mixed_freq_shared_ax(self) -> None:
        """Test plotting a mixed-frequency series on a shared axis."""
        idx1: DatetimeIndex = date_range('2015-01-01', periods=3, freq='ME')
        idx2: DatetimeIndex = idx1[:1].union(idx1[2:])
        s1: Series = Series(range(len(idx1)), idx1)
        s2: Series = Series(range(len(idx2)), idx2)
        _, (ax1, ax2): plt.Axes = mpl.pyplot.subplots(nrows=2, sharex=True)
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)
        assert ax1.freq == 'M'
        assert ax2.freq == 'M'
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_mixed_freq_shared_ax_twin_x(self) -> None:
        """Test plotting a mixed-frequency series on a shared axis with a twin x-axis."""
        idx1: DatetimeIndex = date_range('2015-01-01', periods=3, freq='ME')
        idx2: DatetimeIndex = idx1[:1].union(idx1[2:])
        s1: Series = Series(range(len(idx1)), idx1)
        s2: Series = Series(range(len(idx2)), idx2)
        _, ax1: plt.Axes = mpl.pyplot.subplots()
        ax2: plt.Axes = ax1.twinx()
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    @pytest.mark.xfail(reason='TODO (GH14330, GH14322)')
    def test_mixed_freq_shared_ax_twin_x_irregular_first(self) -> None:
        """Test plotting a mixed-frequency series on a shared axis with a twin x-axis and an irregular first series."""
        idx1: DatetimeIndex = date_range('2015-01-01', periods=3, freq='ME')
        idx2: DatetimeIndex = idx1[:1].union(idx1[2:])
        s1: Series = Series(range(len(idx1)), idx1)
        s2: Series = Series(range(len(idx2)), idx2)
        _, ax1: plt.Axes = mpl.pyplot.subplots()
        ax2: plt.Axes = ax1.twinx()
        s2.plot(ax=ax1)
        s1.plot(ax=ax2)
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_nat_handling(self) -> None:
        """Test handling NaT values."""
        _, ax: plt.Axes = mpl.pyplot.subplots()
        dti: DatetimeIndex = DatetimeIndex(['2015-01-01', NaT, '2015-01-03'])
        s: Series = Series(range(len(dti)), dti)
        s.plot(ax=ax)
        xdata: np.ndarray = ax.get_lines()[0].get_xdata()
        assert s.index.min() <= Series(xdata).min()
        assert Series(xdata).max() <= s.index.max()

    def test_to_weekly_resampling_disallow_how_kwd(self) -> None:
        """Test resampling to weekly frequency."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=52, freq='W')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        high.plot(ax=ax)
        msg: str = "'how' is not a valid keyword for plotting functions. If plotting multiple objects on shared axes, resample manually first."
        with pytest.raises(ValueError, match=msg):
            low.plot(ax=ax, how='foo')

    def test_to_weekly_resampling(self) -> None:
        """Test resampling to weekly frequency."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=52, freq='W')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq

    def test_from_weekly_resampling(self) -> None:
        """Test resampling from weekly frequency."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=52, freq='W')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        expected_h: np.ndarray = idxh.to_period().asi8.astype(np.float64)
        expected_l: np.ndarray = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            xdata: np.ndarray = line.get_xdata(orig=False)
            if len(xdata) == 12:
                tm.assert_numpy_array_equal(xdata, expected_l)
            else:
                tm.assert_numpy_array_equal(xdata, expected_h)

    @pytest.mark.parametrize('kind1, kind2', [('line', 'area'), ('area', 'line')])
    def test_from_resampling_area_line_mixed(self, kind1: str, kind2: str) -> None:
        """Test resampling and plotting a mixed-frequency series."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=52, freq='W')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: DataFrame = DataFrame(np.random.default_rng(2).random((len(idxh), 3)), index=idxh, columns=[0, 1, 2])
        low: DataFrame = DataFrame(np.random.default_rng(2).random((len(idxl), 3)), index=idxl, columns=[0, 1, 2])
        _, ax: plt.Axes = mpl.pyplot.subplots()
        low.plot(kind=kind1, stacked=True, ax=ax)
        high.plot(kind=kind2, stacked=True, ax=ax)
        expected_x: np.ndarray = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
        expected_y: np.ndarray = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line: plt.Line2D = ax.lines[i]
            assert PeriodIndex(line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += low[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)
        expected_x: np.ndarray = idxh.to_period().asi8.astype(np.float64)
        expected_y: np.ndarray = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line: plt.Line2D = ax.lines[3 + i]
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

    @pytest.mark.parametrize('kind1, kind2', [('line', 'area'), ('area', 'line')])
    def test_from_resampling_area_line_mixed_high_to_low(self, kind1: str, kind2: str) -> None:
        """Test resampling and plotting a mixed-frequency series with high-to-low frequency."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=52, freq='W')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: DataFrame = DataFrame(np.random.default_rng(2).random((len(idxh), 3)), index=idxh, columns=[0, 1, 2])
        low: DataFrame = DataFrame(np.random.default_rng(2).random((len(idxl), 3)), index=idxl, columns=[0, 1, 2])
        _, ax: plt.Axes = mpl.pyplot.subplots()
        high.plot(kind=kind1, stacked=True, ax=ax)
        low.plot(kind=kind2, stacked=True, ax=ax)
        expected_x: np.ndarray = idxh.to_period().asi8.astype(np.float64)
        expected_y: np.ndarray = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line: plt.Line2D = ax.lines[i]
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)
        expected_x: np.ndarray = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
        expected_y: np.ndarray = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            lines: list = ax.lines[3 + i]
            assert PeriodIndex(data=lines.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(lines.get_xdata(orig=False), expected_x)
            expected_y += low[i].values
            tm.assert_numpy_array_equal(lines.get_ydata(orig=False), expected_y)

    def test_mixed_freq_second_millisecond(self) -> None:
        """Test plotting a mixed-frequency series with a second and millisecond frequency."""
        idxh: DatetimeIndex = date_range('2014-07-01 09:00', freq='s', periods=5)
        idxl: DatetimeIndex = date_range('2014-07-01 09:00', freq='100ms', periods=50)
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'ms'

    def test_mixed_freq_second_millisecond_low_to_high(self) -> None:
        """Test plotting a mixed-frequency series with a second and millisecond frequency, low-to-high."""
        idxh: DatetimeIndex = date_range('2014-07-01 09:00', freq='s', periods=5)
        idxl: DatetimeIndex = date_range('2014-07-01 09:00', freq='100ms', periods=50)
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == 'ms'

    def test_irreg_dtypes(self) -> None:
        """Test plotting a series with irregular dtypes."""
        idx: list = [date(2000, 1, 1), date(2000, 1, 5), date(2000, 1, 20)]
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), Index(idx, dtype=object))
        _check_plot_works(df.plot)

    def test_irreg_dtypes_dt64(self) -> None:
        """Test plotting a series with irregular dt64 values."""
        idx: list = date_range('1/1/2000', periods=10)
        idx: list = idx[[0, 2, 5, 9]].astype(object)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), idx)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        _check_plot_works(df.plot, ax=ax)

    def test_time(self) -> None:
        """Test plotting a series with time values."""
        t: datetime = datetime(1, 1, 1, 3, 30, 0)
        deltas: np.ndarray = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        ts: np.ndarray = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
        df: DataFrame = DataFrame({'a': np.random.default_rng(2).standard_normal(len(ts)), 'b': np.random.default_rng(2).standard_normal(len(ts))}, index=ts)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        df.plot(ax=ax)
        ticks: np.ndarray = ax.get_xticks()
        labels: list = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            us = round((_tick - int(_tick)) * 1000000.0)
            h, m = divmod(m, 60)
            rs: str = _label.get_text()
            if len(rs) > 0:
                if us % 1000 != 0:
                    xp: str = time(h, m, s, us).strftime('%H:%M:%S.%f')
                elif us // 1000 != 0:
                    xp: str = time(h, m, s, us).strftime('%H:%M:%S.%f')[:-3]
                elif s != 0:
                    xp: str = time(h, m, s, us).strftime('%H:%M:%S')
                else:
                    xp: str = time(h, m, s, us).strftime('%H:%M')
                assert xp == rs

    def test_secondary_upsample(self) -> None:
        """Test up-sampling a series for a secondary y-axis."""
        idxh: DatetimeIndex = date_range('1/1/1999', periods=365, freq='D')
        idxl: DatetimeIndex = date_range('1/1/1999', periods=12, freq='ME')
        high: Series = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low: Series = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        low.plot(ax=ax)
        ax: plt.Axes = high.plot(secondary_y=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == 'D'
        assert hasattr(ax, 'left_ax')
        assert not hasattr(ax, 'right_ax')
        for line in ax.left_ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == 'D'

    def test_secondary_legend(self) -> None:
        """Test plotting a secondary y-axis legend."""
        fig: plt.Figure = mpl.pyplot.figure()
        ax: plt.Axes = fig.add_subplot(211)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df.plot(secondary_y=['A', 'B'], ax=ax)
        leg: plt.Legend = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == 'A (right)'
        assert leg.get_texts()[1].get_text() == 'B (right)'
        assert leg.get_texts()[2].get_text() == 'C'
        assert leg.get_texts()[3].get_text() == 'D'
        assert ax.right_ax.get_legend() is None
        colors: set = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    def test_secondary_legend_right(self) -> None:
        """Test plotting a secondary y-axis legend with mark_right=False."""
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        fig: plt.Figure = mpl.pyplot.figure()
        ax: plt.Axes = fig.add_subplot(211)
        df.plot(secondary_y=['A', 'C'], mark_right=False, ax=ax)
        leg: plt.Legend = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == 'A'
        assert leg.get_texts()[1].get_text() == 'B'
        assert leg.get_texts()[2].get_text() == 'C'
        assert leg.get_texts()[3].get_text() == 'D'

    def test_secondary_legend_bar(self) -> None:
        """Test plotting a secondary y-axis legend for a bar plot."""
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        fig, ax: plt.Axes = mpl.pyplot.subplots()
        df.plot(kind='bar', secondary_y=['A'], ax=ax)
        leg: plt.Legend = ax.get_legend()
        assert leg.get_texts()[0].get_text() == 'A (right)'
        assert leg.get_texts()[1].get_text() == 'B'

    def test_secondary_legend_bar_right(self) -> None:
        """Test plotting a secondary y-axis legend for a bar plot with mark_right=False."""
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        fig, ax: plt.Axes = mpl.pyplot.subplots()
        df.plot(kind='bar', secondary_y=['A'], mark_right=False, ax=ax)
        leg: plt.Legend = ax.get_legend()
        assert leg.get_texts()[0].get_text() == 'A'
        assert leg.get_texts()[1].get_text() == 'B'

    def test_secondary_legend_multi_col(self) -> None:
        """Test plotting a secondary y-axis legend with multiple columns."""
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        fig: plt.Figure = mpl.pyplot.figure()
        ax: plt.Axes = fig.add_subplot(211)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        ax: plt.Axes = df.plot(secondary_y=['C', 'D'], ax=ax)
        leg: plt.Legend = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors: set = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    def test_secondary_legend_nonts(self) -> None:
        """Test plotting a secondary y-axis legend for a non-time series."""
        df: DataFrame = DataFrame(1.1 * np.arange(40).reshape((10, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(10)], dtype=object))
        fig: plt.Figure = mpl.pyplot.figure()
        ax: plt.Axes = fig.add_subplot(211)
        ax: plt.Axes = df.plot(secondary_y=['A', 'B'], ax=ax)
        leg: plt.Legend = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors: set = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    def test_secondary_legend_nonts_multi_col(self) -> None:
        """Test plotting a secondary y-axis legend with multiple columns for a non-time series."""
        df: DataFrame = DataFrame(1.1 * np.arange(40).reshape((10, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(10)], dtype=object))
        fig: plt.Figure = mpl.pyplot.figure()
        ax: plt.Axes = fig.add_subplot(211)
        ax: plt.Axes = df.plot(secondary_y=['C', 'D'], ax=ax)
        leg: plt.Legend = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors: set = set()
        for line in leg.get_lines():
            colors.add(line.get_color())
        assert len(colors) == 4

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_format_date_axis(self) -> None:
        """Test formatting the date axis."""
        rng: DatetimeIndex = date_range('1/1/2012', periods=12, freq='ME')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ax: plt.Axes = df.plot(ax=ax)
        xaxis: plt.TickHelper = ax.get_xaxis()
        for line in xaxis.get_ticklabels():
            if len(line.get_text()) > 0:
                assert line.get_rotation() == 30

    def test_ax_plot(self) -> None:
        """Test plotting on an axis."""
        x: DatetimeIndex = date_range(start='2012-01-02', periods=10, freq='D')
        y: list = list(range(len(x)))
        _, ax: plt.Axes = mpl.pyplot.subplots()
        lines: list = ax.plot(x, y, label='Y')
        tm.assert_index_equal(DatetimeIndex(lines[0].get_xdata()), x)

    def test_mpl_nopandas(self) -> None:
        """Test plotting without pandas."""
        dates: list = [date(2008, 12, 31), date(2009, 1, 31)]
        values1: np.ndarray = np.arange(10.0, 11.0, 0.5)
        values2: np.ndarray = np.arange(11.0, 12.0, 0.5)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        line1, line2: plt.Line2D = ax.plot([x.toordinal() for x in dates], values1, '-', [x.toordinal() for x in dates], values2, '-', linewidth=4)
        exp: np.ndarray = np.array([x.toordinal() for x in dates], dtype=np.float64)
        tm.assert_numpy_array_equal(line1.get_xydata()[:, 0], exp)
        tm.assert_numpy_array_equal(line2.get_xydata()[:, 0], exp)

    def test_irregular_ts_shared_ax_xlim(self) -> None:
        """Test plotting an irregular time series on a shared axis."""
        ts: Series = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20))
        ts_irregular: Series = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ts_irregular[:5].plot(ax=ax)
        ts_irregular[5:].plot(ax=ax)
        left, right: float = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), '', ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), '', ax)

    def test_secondary_y_non_ts_xlim(self) -> None:
        """Test plotting a secondary y-axis for a non-time series."""
        index_1: list = [1, 2, 3, 4]
        index_2: list = [5, 6, 7, 8]
        s1: Series = Series(1, index=index_1)
        s2: Series = Series(2, index=index_2)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        left_before, right_before: float = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after: float = ax.get_xlim()
        assert left_before >= left_after
        assert right_before < right_after

    def test_secondary_y_regular_ts_xlim(self) -> None:
        """Test plotting a secondary y-axis for a regular time series."""
        index_1: DatetimeIndex = date_range(start='2000-01-01', periods=4, freq='D')
        index_2: DatetimeIndex = date_range(start='2000-01-05', periods=4, freq='D')
        s1: Series = Series(1, index=index_1)
        s2: Series = Series(2, index=index_2)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        left_before, right_before: float = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after: float = ax.get_xlim()
        assert left_before >= left_after
        assert right_before < right_after

    def test_secondary_y_mixed_freq_ts_xlim(self) -> None:
        """Test plotting a secondary y-axis for a mixed-frequency time series."""
        rng: DatetimeIndex = date_range('2000-01-01', periods=10, freq='min')
        ts: Series = Series(1, index=rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ts.plot(ax=ax)
        left_before, right_before: float = ax.get_xlim()
        ts.resample('D').mean().plot(secondary_y=True, ax=ax)
        left_after, right_after: float = ax.get_xlim()
        assert left_before == left_after
        assert right_before == right_after

    def test_secondary_y_irregular_ts_xlim(self) -> None:
        """Test plotting a secondary y-axis for an irregular time series."""
        ts: Series = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20))
        ts_irregular: Series = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ts_irregular[:5].plot(ax=ax)
        ts_irregular[5:].plot(secondary_y=True, ax=ax)
        ts_irregular[:5].plot(ax=ax)
        left, right: float = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), '', ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), '', ax)

    def test_plot_outofbounds_datetime(self) -> None:
        """Test plotting out-of-bounds datetime values."""
        values: list = [date(1677, 1, 1), date(1677, 1, 2)]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ax.plot(values)
        values: list = [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]
        ax.plot(values)

    def test_format_timedelta_ticks_narrow(self) -> None:
        """Test formatting timedelta ticks with narrow format."""
        expected_labels: list = [f'00:00:00.0000000{i:0>2d}' for i in np.arange(10)]
        rng: timedelta_range = timedelta_range('0', periods=10, freq='ns')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        df.plot(fontsize=2, ax=ax)
        mpl.pyplot.draw()
        labels: list = ax.get_xticklabels()
        result_labels: list = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_format_timedelta_ticks_wide(self) -> None:
        """Test formatting timedelta ticks with wide format."""
        expected_labels: list = ['00:00:00', '1 days 03:46:40', '2 days 07:33:20', '3 days 11:20:00', '4 days 15:06:40', '5 days 18:53:20', '6 days 22:40:00', '8 days 02:26:40', '9 days 06:13:20']
        rng: timedelta_range = timedelta_range('0', periods=10, freq='1 D')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ax = df.plot(fontsize=2, ax=ax)
        mpl.pyplot.draw()
        labels: list = ax.get_xticklabels()
        result_labels: list = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_timedelta_plot(self) -> None:
        """Test plotting a timedelta series."""
        s: Series = Series(range(5), timedelta_range('1day', periods=5))
        _, ax: plt.Axes = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_long_period(self) -> None:
        """Test plotting a timedelta series with a long period."""
        index: timedelta_range = timedelta_range('1 day 2 hr 30 min 10 s', periods=10, freq='1 D')
        s: Series = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_short_period(self) -> None:
        """Test plotting a timedelta series with a short period."""
        index: timedelta_range = timedelta_range('1 day 2 hr 30 min 10 s', periods=10, freq='1 ns')
        s: Series = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        _, ax: plt.Axes = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_hist(self) -> None:
        """Test plotting a histogram."""
        rng: DatetimeIndex = date_range('1/1/2011', periods=10, freq='h')
        x: DatetimeIndex = rng
        w1: np.ndarray = np.arange(0, 1, 0.1)
        w2: np.ndarray = np.arange(0, 1, 0.1)[::-1]
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ax.hist([x, x], weights=[w1, w2])

    def test_overlapping_datetime(self) -> None:
        """Test plotting overlapping datetime values."""
        s1: Series = Series([1, 2, 3], index=[datetime(1995, 12, 31), datetime(2000, 12, 31), datetime(2005, 12, 31)])
        s2: Series = Series([1, 2, 3], index=[datetime(1997, 12, 31), datetime(2003, 12, 31), datetime(2008, 12, 31)])
        _, ax: plt.Axes = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        s2.plot(ax=ax)
        s1.plot(ax=ax)

    @pytest.mark.xfail(reason='GH9053 matplotlib does not use ax.xaxis.converter')
    def test_add_matplotlib_datetime64(self) -> None:
        """Test adding matplotlib datetime64 values."""
        s: Series = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1970-01-02', periods=10))
        ax: plt.Axes = s.plot()
        with tm.assert_produces_warning(DeprecationWarning):
            ax.plot(s.index, s.values, color='g')
        l1, l2: plt.Line2D = ax.lines
        tm.assert_numpy_array_equal(l1.get_xydata(), l2.get_xydata())

    def test_matplotlib_scatter_datetime64(self) -> None:
        """Test plotting a scatter plot with matplotlib datetime64 values."""
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['x', 'y'])
        df['time'] = date_range('2018-01-01', periods=10, freq='D')
        _, ax: plt.Axes = mpl.pyplot.subplots()
        ax.scatter(x='time', y='y', data=df)
        mpl.pyplot.draw()
        label: plt.Text = ax.get_xticklabels()[0]
        expected: str = '2018-01-01'
        assert label.get_text() == expected

    def test_check_xticks_rot(self) -> None:
        """Test checking x-axis tick rotation."""
        x: DatetimeIndex = to_datetime(['2020-05-01', '2020-05-02', '2020-05-03'])
        df: DataFrame = DataFrame({'x': x, 'y': [1, 2, 3]})
        axes: list = df.plot(x='x', y='y')
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_irregular(self) -> None:
        """Test checking x-axis tick rotation for irregular datetime values."""
        x: DatetimeIndex = to_datetime(['2020-05-01', '2020-05-02', '2020-05-04'])
        df: DataFrame = DataFrame({'x': x, 'y': [1, 2, 3]})
        axes: list = df.plot(x='x', y='y')
        _check_ticks_props(axes, xrot=30)

    def test_check_xticks_rot_use_idx(self) -> None:
        """Test checking x-axis tick rotation when using the index."""
        x: DatetimeIndex = to_datetime(['2020-05-01', '2020-05-02', '2020-05-04'])
        df: DataFrame = DataFrame({'x': x, 'y': [1, 2, 3]})
        axes: list = df.set_index('x').plot(y='y', use_index=True)
        _check_ticks_props(axes, xrot=30)
        axes: list = df.set_index('x').plot(y='y', use_index=False)
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_sharex(self) -> None:
        """Test checking x-axis tick rotation when sharing an x-axis."""
        x: DatetimeIndex = to_datetime(['2020-05-01', '2020-05-02', '2020-05-04'])
        df: DataFrame = DataFrame({'x': x, 'y': [1, 2, 3]})
        axes: list = df.plot(x='x', y='y', subplots=True, sharex=True)
        _check_ticks_props(axes, xrot=30)
        axes: list = df.plot(x='x', y='y', subplots=True, sharex=False)
        _check_ticks_props(axes, xrot=0)

def _check_plot_works(f: callable, freq: str | None = None, series: Series | None = None, *args, **kwargs) -> None:
    fig: plt.Figure = plt.gcf()
    fig.clf()
    ax: plt.Axes = fig.add_subplot(211)
    orig_axfreq: str | None = getattr(orig_ax, 'freq', None)
    ret: plt.Axes = f(*args, **kwargs)
    assert ret is not None
    ax: plt.Axes = kwargs.pop('ax', plt.gca())
    if series is not None:
        dfreq: str = series.index.freq
        if isinstance(dfreq, BaseOffset):
            dfreq: str = dfreq.rule_code
        if orig_axfreq is None:
            assert ax.freq == dfreq
    if freq is not None and orig_axfreq is None:
        assert to_offset(ax.freq, is_period=True) == freq
    ax: plt.Axes = fig.add_subplot(212)
    kwargs['ax'] = ax
    ret: plt.Axes = f(*args, **kwargs)
    assert ret is not None
