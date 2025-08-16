"""Test cases for Series.plot"""

from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    period_range,
    plotting,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_colors,
    _check_grid_settings,
    _check_has_errorbars,
    _check_legend_labels,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    _unpack_cycler,
    get_y_axis,
)

from pandas.tseries.offsets import CustomBusinessDay

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

from pandas.plotting._matplotlib.converter import DatetimeConverter
from pandas.plotting._matplotlib.style import get_standard_colors


@pytest.fixture
def ts() -> Series:
    return Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10),
        name="ts",
    )


@pytest.fixture
def series() -> Series:
    return Series(
        range(10), dtype=np.float64, name="series", index=[f"i_{i}" for i in range(10)]
    )


class TestSeriesPlots:
    @pytest.mark.slow
    @pytest.mark.parametrize("kwargs", [{"label": "foo"}, {"use_index": False}])
    def test_plot(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        _check_plot_works(ts.plot, **kwargs)

    @pytest.mark.slow
    def test_plot_tick_props(self, ts: Series) -> None:
        axes = _check_plot_works(ts.plot, rot=0)
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "scale, exp_scale",
        [
            [{"logy": True}, {"yaxis": "log"}],
            [{"logx": True}, {"xaxis": "log"}],
            [{"loglog": True}, {"xaxis": "log", "yaxis": "log"}],
        ],
    )
    def test_plot_scales(self, ts: Series, scale: Dict[str, bool], exp_scale: Dict[str, str]) -> None:
        ax = _check_plot_works(ts.plot, style=".", **scale)
        _check_ax_scales(ax, **exp_scale)

    @pytest.mark.slow
    def test_plot_ts_bar(self, ts: Series) -> None:
        _check_plot_works(ts[:10].plot.bar)

    @pytest.mark.slow
    def test_plot_ts_area_stacked(self, ts: Series) -> None:
        _check_plot_works(ts.plot.area, stacked=False)

    def test_plot_iseries(self) -> None:
        ser = Series(range(5), period_range("2020-01-01", periods=5))
        _check_plot_works(ser.plot)

    @pytest.mark.parametrize(
        "kind",
        [
            "line",
            "bar",
            "barh",
            pytest.param("kde", marks=td.skip_if_no("scipy")),
            "hist",
            "box",
        ],
    )
    def test_plot_series_kinds(self, series: Series, kind: str) -> None:
        _check_plot_works(series[:5].plot, kind=kind)

    def test_plot_series_barh(self, series: Series) -> None:
        _check_plot_works(series[:10].plot.barh)

    def test_plot_series_bar_ax(self) -> None:
        ax = _check_plot_works(
            Series(np.random.default_rng(2).standard_normal(10)).plot.bar, color="black"
        )
        _check_colors([ax.patches[0]], facecolors=["black"])

    @pytest.mark.parametrize("kwargs", [{}, {"layout": (-1, 1)}, {"layout": (1, -1)}])
    def test_plot_6951(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        # GH 6951
        ax = _check_plot_works(ts.plot, subplots=True, **kwargs)
        _check_axes_shape(ax, axes_num=1, layout=(1, 1))

    def test_plot_figsize_and_title(self, series: Series) -> None:
        # figsize and title
        _, ax = mpl.pyplot.subplots()
        ax = series.plot(title="Test", figsize=(16, 8), ax=ax)
        _check_text_labels(ax.title, "Test")
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))

    def test_dont_modify_rcParams(self) -> None:
        # GH 8242
        key = "axes.prop_cycle"
        colors = mpl.pyplot.rcParams[key]
        _, ax = mpl.pyplot.subplots()
        Series([1, 2, 3]).plot(ax=ax)
        assert colors == mpl.pyplot.rcParams[key]

    @pytest.mark.parametrize("kwargs", [{}, {"secondary_y": True}])
    def test_ts_line_lim(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax, **kwargs)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]

    def test_ts_area_lim(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_area_lim_xcompat(self, ts: Series) -> None:
        # GH 7471
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=30)

    def test_ts_tz_area_lim_xcompat(self, ts: Series) -> None:
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_tz_area_lim_xcompat_secondary_y(self, ts: Series) -> None:
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_area_sharey_dont_overwrite(self, ts: Series) -> None:
        # GH37942
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)

        abs(ts).plot(ax=ax1, kind="area")
        abs(ts).plot(ax=ax2, kind="area")

        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    def test_label(self) -> None:
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(label="LABEL", legend=True, ax=ax)
        _check_legend_labels(ax, labels=["LABEL"])

    def test_label_none(self) -> None:
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=[""])

    def test_label_ser_name(self) -> None:
        s = Series([1, 2], name="NAME")
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=["NAME"])

    def test_label_ser_name_override(self) -> None:
        s = Series([1, 2], name="NAME")
        # override the default
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, label="LABEL", ax=ax)
        _check_legend_labels(ax, labels=["LABEL"])

    def test_label_ser_name_override_dont_draw(self) -> None:
        s = Series([1, 2], name="NAME")
        # Add lebel info, but don't draw
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=False, label="LABEL", ax=ax)
        assert ax.get_legend() is None  # Hasn't been drawn
        ax.legend()  # draw it
        _check_legend_labels(ax, labels=["LABEL"])

    def test_boolean(self) -> None:
        # GH 23719
        s = Series([False, False, True])
        _check_plot_works(s.plot, include_bool=True)

        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            _check_plot_works(s.plot)

    @pytest.mark.parametrize("index", [None, date_range("2020-01-01", periods=4)])
    def test_line_area_nan_series(self, index: Optional[pd.DatetimeIndex]) -> None:
        values = [1, 2, np.nan, 3]
        d = Series(values, index=index)
        ax = _check_plot_works(d.plot)
        masked = ax.lines[0].get_ydata()
        # remove nan for comparison purpose
        exp = np.array([1, 2, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
        tm.assert_numpy_array_equal(masked.mask, np.array([False, False, True, False]))

        expected = np.array([1, 2, 0, 3], dtype=np.float64)
        ax = _check_plot_works(d.plot, stacked=True)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area, stacked=False)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)

    def test_line_use_index_false(self) -> None:
        s = Series([1, 2, 3], index=["a", "b", "c"])
        s.index.name = "The Index"
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(use_index=False, ax=ax)
        label = ax.get_xlabel()
        assert label == ""

    def test_line_use_index_false_diff_var(self) -> None:
        s = Series([1, 2, 3], index=["a", "b", "c"])
        s.index.name = "The Index"
        _, ax = mpl.pyplot.subplots()
        ax2 = s.plot.bar(use_index=False, ax=ax)
        label2 = ax2.get_xlabel()
        assert label2 == ""

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize("axis, meth", [("yaxis", "bar"), ("xaxis", "barh")])
    def test_bar_log(self, axis: str, meth: str) -> None:
        expected = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

        _, ax = mpl.pyplot.subplots()
        ax = getattr(Series([200, 500]).plot, meth)(log=True, ax=ax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize(
        "axis, kind, res_meth",
        [["yaxis", "bar", "get_ylim"], ["xaxis", "barh", "get_xlim"]],
    )
    def test_bar_log_kind_bar(self, axis: str, kind: str, res_meth: str) -> None:
        # GH 9905
        expected = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])

        _, ax = mpl.pyplot.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind=kind, ax=ax)
        ymin = 0.0007943282347242822
        ymax = 0.12589254117941673
        res = getattr(ax, res_meth)()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    def test_bar_ignore_index(self) -> None:
        df = Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.bar(use_index=False, ax=ax)
        _check_text_labels(ax.get_xticklabels(), ["0", "1", "2", "3"])

    def test_bar_user_colors(self) -> None:
        s = Series([1, 2, 3, 4])
        ax = s.plot.bar(color=["red", "blue", "blue", "red"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_rotation_default(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # Default rot 0
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(ax=ax)
        _check_ticks_props(axes, xrot=0)

    def test_rotation_30(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(rot=30, ax=ax)
        _check_ticks_props(axes, xrot=30)

    def test_irregular_datetime(self) -> None:
        rng = date_range("1/1/2000", "1/15/2000")
        rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        xp = DatetimeConverter.convert(datetime(1999, 1, 1), "", ax)
        ax.set_xlim("1/1/1999", "1/1/2001")
        assert xp == ax.get_xlim()[0]
        _check_ticks_props(ax, xrot=30)

    def test_unsorted_index_xlim(self) -> None:
        ser = Series(
            [0.0, 1.0, np.nan, 3.0, 4.0, 5.0,