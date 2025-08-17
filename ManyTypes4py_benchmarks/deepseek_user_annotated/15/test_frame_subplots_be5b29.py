"""Test cases for DataFrame.plot"""

import string
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_box_return_type,
    _check_legend_labels,
    _check_ticks_props,
    _check_visible,
    _flatten_visible,
)

from pandas.io.formats.printing import pprint_thing

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
Axes = plt.Axes
Figure = plt.Figure


class TestDataFramePlotsSubplots:
    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots(self, kind: str) -> None:
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(kind=kind, subplots=True, sharex=True, legend=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        assert axes.shape == (3,)

        for ax, column in zip(axes, df.columns):
            _check_legend_labels(ax, labels=[pprint_thing(column)])

        for ax in axes[:-2]:
            _check_visible(ax.xaxis)  # xaxis must be visible for grid
            _check_visible(ax.get_xticklabels(), visible=False)
            if kind != "bar":
                # change https://github.com/pandas-dev/pandas/issues/26714
                _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())

        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots_no_share_x(self, kind: str) -> None:
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(kind=kind, subplots=True, sharex=False)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots_no_legend(self, kind: str) -> None:
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(kind=kind, subplots=True, legend=False)
        for ax in axes:
            assert ax.get_legend() is None

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries(self, kind: str) -> None:
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)

        axes = df.plot(kind=kind, subplots=True, sharex=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))

        for ax in axes[:-2]:
            # GH 7801
            _check_visible(ax.xaxis)  # xaxis must be visible for grid
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())

        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries_rot(self, kind: str) -> None:
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        axes = df.plot(kind=kind, subplots=True, sharex=False, rot=45, fontsize=7)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())
            _check_ticks_props(ax, xlabelsize=7, xrot=45, ylabelsize=7)

    @pytest.mark.parametrize(
        "col", ["numeric", "timedelta", "datetime_no_tz", "datetime_all_tz"]
    )
    def test_subplots_timeseries_y_axis(self, col: str) -> None:
        # GH16953
        data = {
            "numeric": np.array([1, 2, 5]),
            "timedelta": [
                pd.Timedelta(-10, unit="s"),
                pd.Timedelta(10, unit="m"),
                pd.Timedelta(10, unit="h"),
            ],
            "datetime_no_tz": [
                pd.to_datetime("2017-08-01 00:00:00"),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
            "datetime_all_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00", utc=True),
                pd.to_datetime("2017-08-02 00:00:00", utc=True),
            ],
            "text": ["This", "should", "fail"],
        }
        testdata = DataFrame(data)

        ax = testdata.plot(y=col)
        result = ax.get_lines()[0].get_data()[1]
        expected = testdata[col].values
        assert (result == expected).all()

    def test_subplots_timeseries_y_text_error(self) -> None:
        # GH16953
        data = {
            "numeric": np.array([1, 2, 5]),
            "text": ["This", "should", "fail"],
        }
        testdata = DataFrame(data)
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            testdata.plot(y="text")

    @pytest.mark.xfail(reason="not support for period, categorical, datetime_mixed_tz")
    def test_subplots_timeseries_y_axis_not_supported(self) -> None:
        """
        This test will fail for:
            period:
                since period isn't yet implemented in ``select_dtypes``
                and because it will need a custom value converter +
                tick formatter (as was done for x-axis plots)

            categorical:
                 because it will need a custom value converter +
                 tick formatter (also doesn't work for x-axis, as of now)

            datetime_mixed_tz:
                because of the way how pandas handles ``Series`` of
                ``datetime`` objects with different timezone,
                generally converting ``datetime`` objects in a tz-aware
                form could help with this problem
        """
        data = {
            "numeric": np.array([1, 2, 5]),
            "period": [
                pd.Period("2017-08-01 00:00:00", freq="h"),
                pd.Period("2017-08-01 02:00", freq="h"),
                pd.Period("2017-08-02 00:00:00", freq="h"),
            ],
            "categorical": pd.Categorical(
                ["c", "b", "a"], categories=["a", "b", "c"], ordered=False
            ),
            "datetime_mixed_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
        }
        testdata = DataFrame(data)
        ax_period = testdata.plot(x="numeric", y="period")
        assert (
            ax_period.get_lines()[0].get_data()[1] == testdata["period"].values
        ).all()
        ax_categorical = testdata.plot(x="numeric", y="categorical")
        assert (
            ax_categorical.get_lines()[0].get_data()[1]
            == testdata["categorical"].values
        ).all()
        ax_datetime_mixed_tz = testdata.plot(x="numeric", y="datetime_mixed_tz")
        assert (
            ax_datetime_mixed_tz.get_lines()[0].get_data()[1]
            == testdata["datetime_mixed_tz"].values
        ).all()

    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 2), (2, 2)],
            [(-1, 2), (2, 2)],
            [(2, -1), (2, 2)],
            [(1, 4), (1, 4)],
            [(-1, 4), (1, 4)],
            [(4, -1), (4, 1)],
        ],
    )
    def test_subplots_layout_multi_column(
        self, layout: Tuple[int, int], exp_layout: Tuple[int, int]
    ) -> None:
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(subplots=True, layout=layout)
        _check_axes_shape(axes, axes_num=3, layout=exp_layout)
        assert axes.shape == exp_layout

    def test_subplots_layout_multi_column_error(self) -> None:
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        msg = "Layout of 1x1 must be larger than required size 3"

        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(1, 1))

        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(-1, -1))

    @pytest.mark.parametrize(
        "kwargs, expected_axes_num, expected_layout, expected_shape",
        [
            ({}, 1, (1, 1), (1,)),
            ({"layout": (3, 3)}, 1, (3, 3), (3, 3)),
        ],
    )
    def test_subplots_layout_single_column(
        self,
        kwargs: Dict[str, Any],
        expected_axes_num: int,
        expected_layout: Tuple[int, int],
        expected_shape: Tuple[int, ...],
    ) -> None:
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(subplots=True, **kwargs)
        _check_axes_shape(
            axes,
            axes_num=expected_axes_num,
            layout=expected_layout,
        )
        assert axes.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize("idx", [range(5), date_range("1/1/2000", periods=5)])
    def test_subplots_warnings(self, idx: Union[range, pd.DatetimeIndex]) -> None:
        # GH 9464
        with tm.assert_produces_warning(None):
            df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=idx)
            df.plot(subplots=True, layout=(3, 2))

    def test_subplots_multiple_axes(self) -> None:
        # GH 5353, 6970, GH 7069
        fig, axes = mpl.pyplot.subplots(2, 3)
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        returned = df.plot(subplots=True, ax=axes[0], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        # draw on second row
        returned = df.plot(subplots=True, ax=axes[1], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        _check_axes_shape(axes, axes_num=6, layout=(2, 3))

    def test_subplots_multiple_axes_error(self) -> None:
        # GH 5353, 6970, GH 7069
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        msg = "The number of passed axes must be 3, the same as the output plot"
        _, axes = mpl.pyplot.subplots(2, 3)

        with pytest.raises(ValueError, match=msg):
            # pass different number of axes from required
            df.plot(subplots=True, ax=axes)

    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 1), (2, 2)],
            [(2, -1), (2, 2)],
            [(-1, 2), (2, 2)],
        ],
    )
    def test_subplots_multiple_axes_2_dim(
        self, layout: Tuple[int, int], exp_layout: Tuple[int, int]
    ) -> None:
        # GH 5353, 6970, GH 7069
        # pass 2-dim axes and invalid layout
        # invalid layout should not affect to input and return value
        # (show warning is tested in
        # TestDataFrameGroupByPlots.test_grouped_box_multiple_axes
        _, axes = mpl.pyplot.subplots(2, 2)
        df = DataFrame(
            np.random.default_rng(2).random((10, 4)),
            index=list(string.ascii_letters[:10]),
        )
        with tm.assert_produces_warning(UserWarning, match="layout keyword is ignored"):
            returned = df.plot(
                subplots=True, ax=axes, layout=layout, sharex=False, sharey=False
            )
            _check_axes_shape(returned, axes_num=4, layout=exp_layout)
            assert returned.shape == (4,)

    def test_subplots_multiple_axes_single_col(self) -> None:
        # GH 5353, 6970, GH 7069
        # single column
        _, axes = mpl.pyplot.subplots(1, 1)
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(subplots=True, ax=[axes], sharex=False, sharey=False)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        assert axes.shape == (1,)

    def test_subplots_ts_share_axes(self) -> None:
        # GH 3964
        _, axes = mpl.pyplot.subplots(3, 3, sharex=True, sharey=True)
        mpl.pyplot.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)),
            index=date_range(start="2014-07-01", freq="ME", periods=10),
        )
        for i, ax in enumerate(axes.ravel()):
            df[i].plot(ax=ax, fontsize=5)

        # Rows other than bottom should not be visible
        for ax in axes[0:-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=False)

        # Bottom row should be visible
        for ax in axes[-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=True)

        # First column should be visible
        for ax in axes[[0, 1, 2], [0]].