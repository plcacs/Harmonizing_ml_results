"""Test cases for .hist method"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_colors,
    _check_legend_labels,
    _check_patches_all_filled,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    get_x_axis,
    get_y_axis,
)

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from pandas.plotting._matplotlib.hist import _grouped_hist


@pytest.fixture
def ts() -> Series:
    return Series(
        np.arange(30, dtype=np.float64),
        index=date_range("2020-01-01", periods=30, freq="B"),
        name="ts",
    )


class TestSeriesPlots:
    @pytest.mark.parametrize("kwargs", [{}, {"grid": False}, {"figsize": (8, 10)}])
    def test_hist_legacy_kwargs(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        _check_plot_works(ts.hist, **kwargs)

    @pytest.mark.parametrize("kwargs", [{}, {"bins": 5}])
    def test_hist_legacy_kwargs_warning(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(ts.hist, by=ts.index.month, **kwargs)

    def test_hist_legacy_ax(self, ts: Series) -> None:
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, default_axes=True)

    def test_hist_legacy_ax_and_fig(self, ts: Series) -> None:
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, figure=fig, default_axes=True)

    def test_hist_legacy_fig(self, ts: Series) -> None:
        fig, _ = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, figure=fig, default_axes=True)

    def test_hist_legacy_multi_ax(self, ts: Series) -> None:
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2)
        _check_plot_works(ts.hist, figure=fig, ax=ax1, default_axes=True)
        _check_plot_works(ts.hist, figure=fig, ax=ax2, default_axes=True)

    def test_hist_legacy_by_fig_error(self, ts: Series) -> None:
        fig, _ = mpl.pyplot.subplots(1, 1)
        msg = (
            "Cannot pass 'figure' when using the 'by' argument, since a new 'Figure' "
            "instance will be created"
        )
        with pytest.raises(ValueError, match=msg):
            ts.hist(by=ts.index, figure=fig)

    def test_hist_bins_legacy(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        ax = df.hist(bins=2)[0][0]
        assert len(ax.patches) == 2

    def test_hist_layout(self, hist_df: DataFrame) -> None:
        df = hist_df
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=(1, 1))

        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=[1, 1])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, layout, axes_num, res_layout",
        [
            ["gender", (2, 1), 2, (2, 1)],
            ["gender", (3, -1), 2, (3, 1)],
            ["category", (4, 1), 4, (4, 1)],
            ["category", (2, -1), 4, (2, 2)],
            ["category", (3, -1), 4, (3, 2)],
            ["category", (-1, 4), 4, (1, 4)],
            ["classroom", (2, 2), 3, (2, 2)],
        ],
    )
    def test_hist_layout_with_by(
        self,
        hist_df: DataFrame,
        by: str,
        layout: Tuple[int, int],
        axes_num: int,
        res_layout: Tuple[int, int],
    ) -> None:
        df = hist_df

        # _check_plot_works adds an `ax` kwarg to the method call
        # so we get a warning about an axis being cleared, even
        # though we don't explicitly pass one, see GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.height.hist, by=getattr(df, by), layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=res_layout)

    def test_hist_layout_with_by_shape(self, hist_df: DataFrame) -> None:
        df = hist_df

        axes = df.height.hist(by=df.category, layout=(4, 2), figsize=(12, 7))
        _check_axes_shape(axes, axes_num=4, layout=(4, 2), figsize=(12, 7))

    def test_hist_no_overlap(self) -> None:
        x = Series(np.random.default_rng(2).standard_normal(2))
        y = Series(np.random.default_rng(2).standard_normal(2))
        plt.subplot(121)
        x.hist()
        plt.subplot(122)
        y.hist()
        fig = plt.gcf()
        axes = fig.axes
        assert len(axes) == 2

    def test_hist_by_no_extra_plots(self, hist_df: DataFrame) -> None:
        df = hist_df
        df.height.hist(by=df.gender)
        assert len(mpl.pyplot.get_fignums()) == 1

    def test_plot_fails_when_ax_differs_from_figure(self, ts: Series) -> None:
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        ax1 = fig1.add_subplot(111)
        msg = "passed axis not bound to passed figure"
        with pytest.raises(AssertionError, match=msg):
            ts.hist(ax=ax1, figure=fig2)

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        # GH23992 Verify functioning of histtype argument
        ser = Series(np.random.default_rng(2).integers(1, 10))
        ax = ser.hist(histtype=histtype)
        _check_patches_all_filled(ax, filled=expected)

    @pytest.mark.parametrize(
        "by, expected_axes_num, expected_layout", [(None, 1, (1, 1)), ("b", 2, (1, 2))]
    )
    def test_hist_with_legend(
        self, by: Optional[str], expected_axes_num: int, expected_layout: Tuple[int, int]
    ) -> None:
        # GH 6279 - Series histogram can have a legend
        index = 5 * ["1"] + 5 * ["2"]
        s = Series(np.random.default_rng(2).standard_normal(10), index=index, name="a")
        s.index.name = "b"

        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(s.hist, default_axes=True, legend=True, by=by)
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        _check_legend_labels(axes, "a")

    @pytest.mark.parametrize("by", [None, "b"])
    def test_hist_with_legend_raises(self, by: Optional[str]) -> None:
        # GH 6279 - Series histogram with legend and label raises
        index = 5 * ["1"] + 5 * ["2"]
        s = Series(np.random.default_rng(2).standard_normal(10), index=index, name="a")
        s.index.name = "b"

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            s.hist(legend=True, by=by, label="c")

    def test_hist_kwargs(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 5
        _check_text_labels(ax.yaxis.get_label(), "Frequency")

    def test_hist_kwargs_horizontal(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(orientation="horizontal", ax=ax)
        _check_text_labels(ax.xaxis.get_label(), "Frequency")

    def test_hist_kwargs_align(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(align="left", stacked=True, ax=ax)

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_hist_kde(self, ts: Series) -> None:
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        # ticks are values, thus ticklabels are blank
        _check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [""] * len(ylabels))

    def test_hist_kde_plot_works(self, ts: Series) -> None:
        pytest.importorskip("scipy")
        _check_plot_works(ts.plot.kde)

    def test_hist_kde_density_works(self, ts: Series) -> None:
        pytest.importorskip("scipy")
        _check_plot_works(ts.plot.density)

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_hist_kde_logy(self, ts: Series) -> None:
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.kde(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [""] * len(ylabels))

    def test_hist_kde_color_bins(self, ts: Series) -> None:
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(logy=True, bins=10, color="b", ax=ax)
        _check_ax_scales(ax, yaxis="log")
        assert len(ax.patches) == 10
        _check_colors(ax.patches, facecolors=["b"] * 10)

    def test_hist_kde_color(self, ts: Series) -> None:
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.kde(logy=True, color="r", ax=ax)
        _check_ax_scales(ax, yaxis="log")
        lines = ax.get_lines()
        assert len(lines) == 1
        _check_colors(lines, ["r"])


class TestDataFramePlots:
    @pytest.mark.slow
    def test_hist_df_legacy(self, hist_df: DataFrame) -> None:
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(hist_df.hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout(self) -> None:
        # make sure layout is handled
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, grid=False)
        _check_axes_shape(axes, axes_num=3, layout=(2, 2))
        assert not axes[1, 1].get_visible()

        _check_plot_works(df[[2]].hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout2(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
        _check_plot_works(df.hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout3(self) -> None:
        # make sure layout is handled
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, layout=(4, 2))
        _check_axes_shape(axes, axes_num=6, layout=(4, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs", [{"sharex": True, "sharey": True}, {"figsize": (8, 10)}, {"bins": 5}]
    )
    def test_hist_df_legacy_layout_kwargs(self, kwargs: Dict[str, Any]) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # make sure sharex, sharey is handled
        # handle figsize arg
        # check bins argument
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.hist, **kwargs)

    @pytest.mark.slow
    def test_hist_df_legacy_layout_labelsize_rot(
        self, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        # make sure xlabelsize and xrot are handled
        obj = frame_or_series(range(10))
        xf, yf = 20, 18
        xrot, yrot = 30, 40
        axes = obj.hist(xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
        _check_ticks_props(axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)

    @pytest.mark.slow
    def test_hist_df_legacy_rectangles(self) -> None:
        ser = Series(range(10))
        ax = ser.hist(cumulative=True, bins=4, density=True)
        # height of last bin (index 5) must be 1.0
        rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

    @pytest.mark.slow
    def test_hist_df_legacy_scale(self) -> None:
        ser = Series(range(10))
        ax = ser.hist(log=True)
        # scale of y must be 'log'
        _check_ax_scales(ax, yaxis="log")

    @pytest.mark.slow
    def test_hist_df_legacy_external_error(self) -> None:
        ser = Series(range(10))
        # propagate attr exception from matplotlib.Axes.hist
        with tm.external_error_raised(AttributeError):
            ser.hist(foo="bar")

    def test_hist_non_numerical_or_datetime_raises(self) -> None:
        # gh-10444, GH32590
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(