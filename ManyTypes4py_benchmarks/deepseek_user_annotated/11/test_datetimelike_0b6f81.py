"""Test cases for time series specific (freq conversion, etc)"""

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pandas._libs.tslibs import (
    BaseOffset,
    to_offset,
)

from pandas.core.dtypes.dtypes import PeriodDtype

from pandas import (
    DataFrame,
    Index,
    NaT,
    Series,
    concat,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    bdate_range,
    date_range,
)
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props

from pandas.tseries.offsets import WeekOfMonth

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

import pandas.plotting._matplotlib.converter as conv


class TestTSPlot:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_ts_plot_with_tz(self, tz_aware_fixture: Any) -> None:
        # GH2877, GH17173, GH31205, GH31580
        tz = tz_aware_fixture
        index = date_range("1/1/2011", periods=2, freq="h", tz=tz)
        ts = Series([188.5, 328.25], index=index)
        _check_plot_works(ts.plot)
        ax = ts.plot()
        xdata = next(iter(ax.get_lines())).get_xdata()
        # Check first and last points' labels are correct
        assert (xdata[0].hour, xdata[0].minute) == (0, 0)
        assert (xdata[-1].hour, xdata[-1].minute) == (1, 0)

    def test_fontsize_set_correctly(self) -> None:
        # For issue #8765
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)), index=range(10)
        )
        _, ax = mpl.pyplot.subplots()
        df.plot(fontsize=2, ax=ax)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            assert label.get_fontsize() == 2

    def test_frame_inferred(self) -> None:
        # inferred freq
        idx = date_range("1/1/1987", freq="MS", periods=10)
        idx = DatetimeIndex(idx.values, freq=None)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df.plot)

        # axes freq
        idx = idx[0:4].union(idx[6:])
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df2.plot)

    def test_frame_inferred_n_gt_1(self) -> None:
        # N > 1
        idx = date_range("2008-1-1 00:15:00", freq="15min", periods=10)
        idx = DatetimeIndex(idx.values, freq=None)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df.plot)

    def test_is_error_nozeroindex(self) -> None:
        # GH11858
        i = np.array([1, 2, 3])
        a = DataFrame(i, index=i)
        _check_plot_works(a.plot, xerr=a)
        _check_plot_works(a.plot, yerr=a)

    def test_nonnumeric_exclude(self) -> None:
        idx = date_range("1/1/1987", freq="YE", periods=3)
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)

        fig, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)  # it works
        assert len(ax.get_lines()) == 1  # B was plotted

    def test_nonnumeric_exclude_error(self) -> None:
        idx = date_range("1/1/1987", freq="YE", periods=3)
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            df["A"].plot()

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    def test_tsplot_period(self, freq: str) -> None:
        idx = period_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_tsplot_datetime(self, freq: str) -> None:
        idx = date_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    def test_tsplot(self) -> None:
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        _, ax = mpl.pyplot.subplots()
        ts.plot(style="k", ax=ax)
        color = (0.0, 0.0, 0.0, 1)
        assert color == ax.get_lines()[0].get_color()

    @pytest.mark.parametrize("index", [None, date_range("2020-01-01", periods=10)])
    def test_both_style_and_color(self, index: Optional[DatetimeIndex]) -> None:
        ts = Series(np.arange(10, dtype=np.float64), index=index)
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' "
            "keyword argument. Please use one or the other or pass 'style' "
            "without a color symbol"
        )
        with pytest.raises(ValueError, match=msg):
            ts.plot(style="b-", color="#000099")

    @pytest.mark.parametrize("freq", ["ms", "us"])
    def test_high_freq(self, freq: str) -> None:
        _, ax = mpl.pyplot.subplots()
        rng = date_range("1/1/2012", periods=10, freq=freq)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _check_plot_works(ser.plot, ax=ax)

    def test_get_datevalue(self) -> None:
        assert conv.get_datevalue(None, "D") is None
        assert conv.get_datevalue(1987, "Y") == 1987
        assert (
            conv.get_datevalue(Period(1987, "Y"), "M") == Period("1987-12", "M").ordinal
        )
        assert conv.get_datevalue("1/1/1987", "D") == Period("1987-1-1", "D").ordinal

    @pytest.mark.parametrize(
        "freq, expected_string",
        [["YE-DEC", "t = 2014  y = 1.000000"], ["D", "t = 2014-01-01  y = 1.000000"]],
    )
    def test_ts_plot_format_coord(self, freq: str, expected_string: str) -> None:
        ser = Series(1, index=date_range("2014-01-01", periods=3, freq=freq))
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        first_line = ax.get_lines()[0]
        first_x = first_line.get_xdata()[0].ordinal
        first_y = first_line.get_ydata()[0]
        assert expected_string == ax.format_coord(first_x, first_y)

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    def test_line_plot_period_series(self, freq: str) -> None:
        idx = period_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq)

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    def test_line_plot_period_mlt_series(self, frqncy: str) -> None:
        # test period index line plot for series with multiples (`mlt`) of the
        # frequency (`frqncy`) rule code. tests resolution of issue #14763
        idx = period_range("12/31/1999", freq=frqncy, periods=10)
        s = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(s.plot, s.index.freq.rule_code)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_datetime_series(self, freq: str) -> None:
        idx = date_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq.rule_code)

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "ME", "QE", "YE"])
    def test_line_plot_period_frame(self, freq: str) -> None:
        idx = date_range("12/31/1999", freq=freq, periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        _check_plot_works(df.plot, df.index.freq)

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    def test_line_plot_period_mlt_frame(self, frqncy: str) -> None:
        # test period index line plot for DataFrames with multiples (`mlt`)
        # of the frequency (`frqncy`) rule code. tests resolution of issue
        # #14763
        idx = period_range("12/31/1999", freq=frqncy, periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        freq = df.index.freq.rule_code
        _check_plot_works(df.plot, freq)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_datetime_frame(self, freq: str) -> None:
        idx = date_range("12/31/1999", freq=freq, periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        freq = PeriodDtype(df.index.freq)._freqstr
        freq = df.index.to_period(freq).freq
        _check_plot_works(df.plot, freq)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_inferred_freq(self, freq: str) -> None:
        idx = date_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser = Series(ser.values, Index(np.asarray(ser.index)))
        _check_plot_works(ser.plot, ser.index.inferred_freq)

        ser = ser.iloc[[0, 3, 5, 6]]
        _check_plot_works(ser.plot)

    def test_fake_inferred_business(self) -> None:
        _, ax = mpl.pyplot.subplots()
        rng = date_range("2001-1-1", "2001-1-10")
        ts = Series(range(len(rng)), index=rng)
        ts = concat([ts[:3], ts[5:]])
        ts.plot(ax=ax)
        assert not hasattr(ax, "freq")

    def test_plot_offset_freq(self) -> None:
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        _check_plot_works(ser.plot)

    def test_plot_offset_freq_business(self) -> None:
        dr = date_range("2023-01-01", freq="BQS", periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), dr)
        _check_plot_works(ser.plot)

    def test_plot_multiple_inferred_freq(self) -> None:
        dr = Index([datetime(2000, 1, 1), datetime(2000, 1, 6), datetime(2000, 1, 11)])
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), dr)
        _check_plot_works(ser.plot)

    def test_irreg_hf(self) -> None:
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )

        irreg = df.iloc[[0, 1, 3, 4]]
        _, ax = mpl.pyplot.subplots()
        irreg.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()

        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - [sec, sec * 2, sec]) < 1e-8).all()

    def test_irreg_hf_object(self) -> None:
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )
        _, ax = mpl.pyplot.subplots()
        df2.index = df2.index.astype(object)
        df2.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - sec) < 1e-8).all()

    def test_irregular_datetime64_repr_bug(self) -> None:
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        ser = ser.iloc[[0, 1, 2, 7]]

        _, ax = mpl.pyplot.subplots()

        ret = ser.plot(ax=ax)
        assert ret is not None

        for rs, xp in zip(ax.get_lines()[0].get_xdata(), ser.index):
            assert rs == xp

    def test_business_freq(self) -> None:
        bts = Series(range(5), period_range("2020-01-01", periods=5))
        msg = r"PeriodDtype\[B\] is deprecated"
        dt = bts.index[0].to_timestamp()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            bts.index = period_range(start=dt, periods=len(bts), freq="B")
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == bts.index[0].ordinal
        idx = ax.get_lines()[0].get_xdata()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert PeriodIndex(data=idx).freqstr == "B"

    def test_business_freq_convert(self) -> None:
        bts = Series(
            np.arange