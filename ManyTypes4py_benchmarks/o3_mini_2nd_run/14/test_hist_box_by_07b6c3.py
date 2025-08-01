import re
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (_check_axes_shape, _check_plot_works, get_x_axis, get_y_axis)
from matplotlib.axes import Axes

pytest.importorskip("matplotlib")


@pytest.fixture
def hist_df() -> DataFrame:
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((30, 2)),
        columns=["A", "B"]
    )
    df["C"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    df["D"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    return df


class TestHistWithBy:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [
            ("C", "A", ["a", "b", "c"], [["A"]] * 3),
            ("C", ["A", "B"], ["a", "b", "c"], [["A", "B"]] * 3),
            ("C", None, ["a", "b", "c"], [["A", "B"]] * 3),
            (["C", "D"], "A", ["(a, a)", "(b, b)", "(c, c)"], [["A"]] * 3),
            (["C", "D"], ["A", "B"], ["(a, a)", "(b, b)", "(c, c)"], [["A", "B"]] * 3),
            (["C", "D"], None, ["(a, a)", "(b, b)", "(c, c)"], [["A", "B"]] * 3),
        ],
    )
    def test_hist_plot_by_argument(
        self,
        by: Union[str, List[str]],
        column: Optional[Union[str, List[str]]],
        titles: List[str],
        legends: List[List[str]],
        hist_df: DataFrame,
    ) -> None:
        axes: List[Axes] = _check_plot_works(hist_df.plot.hist, column=column, by=by, default_axes=True)
        result_titles: List[str] = [ax.get_title() for ax in axes]
        result_legends: List[List[str]] = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]
        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [
            (0, "A", ["a", "b", "c"], [["A"]] * 3),
            (0, None, ["a", "b", "c"], [["A", "B"]] * 3),
            ([0, "D"], "A", ["(a, a)", "(b, b)", "(c, c)"], [["A"]] * 3),
        ],
    )
    def test_hist_plot_by_0(
        self,
        by: Union[int, List[Any]],
        column: Optional[Union[str, List[str]]],
        titles: List[str],
        legends: List[List[str]],
        hist_df: DataFrame,
    ) -> None:
        df: DataFrame = hist_df.copy()
        df = df.rename(columns={"C": 0})
        axes: List[Axes] = _check_plot_works(df.plot.hist, default_axes=True, column=column, by=by)
        result_titles: List[str] = [ax.get_title() for ax in axes]
        result_legends: List[List[str]] = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]
        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),
            ([], ["A", "B"]),
            ((), None),
            ((), ["A", "B"]),
        ],
    )
    def test_hist_plot_empty_list_string_tuple_by(
        self,
        by: Union[List[Any], Tuple[Any, ...]],
        column: Optional[Union[str, List[str]]],
        hist_df: DataFrame,
    ) -> None:
        msg: str = "No group keys passed"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.hist, default_axes=True, column=column, by=by)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [
            (["C"], "A", (2, 2), 3),
            ("C", "A", (2, 2), 3),
            (["C"], ["A"], (1, 3), 3),
            ("C", None, (3, 1), 3),
            ("C", ["A", "B"], (3, 1), 3),
            (["C", "D"], "A", (9, 1), 3),
            (["C", "D"], "A", (3, 3), 3),
            (["C", "D"], ["A"], (5, 2), 3),
            (["C", "D"], ["A", "B"], (9, 1), 3),
            (["C", "D"], None, (9, 1), 3),
            (["C", "D"], ["A", "B"], (5, 2), 3),
        ],
    )
    def test_hist_plot_layout_with_by(
        self,
        by: Union[str, List[str]],
        column: Optional[Union[str, List[str]]],
        layout: Tuple[int, int],
        axes_num: int,
        hist_df: DataFrame,
    ) -> None:
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes: List[Axes] = _check_plot_works(hist_df.plot.hist, column=column, by=by, layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [
            ("larger than required size", ["C", "D"], (1, 1)),
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),
            ("At least one dimension of layout must be positive", "C", (-1, -1)),
        ],
    )
    def test_hist_plot_invalid_layout_with_by_raises(
        self,
        msg: str,
        by: Union[str, List[str]],
        layout: Any,
        hist_df: DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            hist_df.plot.hist(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.slow
    def test_axis_share_x_with_by(self, hist_df: DataFrame) -> None:
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharex=True)
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)
        assert get_x_axis(ax3).joined(ax1, ax3)
        assert get_x_axis(ax3).joined(ax2, ax3)
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)
        assert not get_y_axis(ax3).joined(ax1, ax3)
        assert not get_y_axis(ax3).joined(ax2, ax3)

    @pytest.mark.slow
    def test_axis_share_y_with_by(self, hist_df: DataFrame) -> None:
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharey=True)
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)
        assert get_y_axis(ax3).joined(ax1, ax3)
        assert get_y_axis(ax3).joined(ax2, ax3)
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)
        assert not get_x_axis(ax3).joined(ax1, ax3)
        assert not get_x_axis(ax3).joined(ax2, ax3)

    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(
        self, figsize: Tuple[int, int], hist_df: DataFrame
    ) -> None:
        axes: List[Axes] = hist_df.plot.hist(column="A", by="C", figsize=figsize)
        _check_axes_shape(axes, axes_num=3, figsize=figsize)


class TestBoxWithBy:
    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            ("C", "A", ["A"], [["a", "b", "c"]]),
            (["C", "D"], "A", ["A"], [["(a, a)", "(b, b)", "(c, c)"]]),
            ("C", ["A", "B"], ["A", "B"], [["a", "b", "c"]] * 2),
            (["C", "D"], ["A", "B"], ["A", "B"], [["(a, a)", "(b, b)", "(c, c)"]] * 2),
            (["C"], None, ["A", "B"], [["a", "b", "c"]] * 2),
        ],
    )
    def test_box_plot_by_argument(
        self,
        by: Union[str, List[str]],
        column: Optional[Union[str, List[str]]],
        titles: List[str],
        xticklabels: List[List[str]],
        hist_df: DataFrame,
    ) -> None:
        axes: List[Axes] = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)
        result_titles: List[str] = [ax.get_title() for ax in axes]
        result_xticklabels: List[List[str]] = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]
        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            (0, "A", ["A"], [["a", "b", "c"]]),
            ([0, "D"], "A", ["A"], [["(a, a)", "(b, b)", "(c, c)"]]),
            (0, None, ["A", "B"], [["a", "b", "c"]] * 2),
        ],
    )
    def test_box_plot_by_0(
        self,
        by: Union[int, List[Any]],
        column: Optional[Union[str, List[str]]],
        titles: List[str],
        xticklabels: List[List[str]],
        hist_df: DataFrame,
    ) -> None:
        df: DataFrame = hist_df.copy()
        df = df.rename(columns={"C": 0})
        axes: List[Axes] = _check_plot_works(df.plot.box, default_axes=True, column=column, by=by)
        result_titles: List[str] = [ax.get_title() for ax in axes]
        result_xticklabels: List[List[str]] = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]
        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),
            ((), "A"),
            ([], None),
            ((), ["A", "B"]),
        ],
    )
    def test_box_plot_with_none_empty_list_by(
        self,
        by: Union[List[Any], Tuple[Any, ...]],
        column: Optional[Union[str, List[str]]],
        hist_df: DataFrame,
    ) -> None:
        msg: str = "No group keys passed"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [
            (["C"], "A", (1, 1), 1),
            ("C", "A", (1, 1), 1),
            ("C", None, (2, 1), 2),
            ("C", ["A", "B"], (1, 2), 2),
            (["C", "D"], "A", (1, 1), 1),
            (["C", "D"], None, (1, 2), 2),
        ],
    )
    def test_box_plot_layout_with_by(
        self,
        by: Union[str, List[str]],
        column: Optional[Union[str, List[str]]],
        layout: Tuple[int, int],
        axes_num: int,
        hist_df: DataFrame,
    ) -> None:
        axes: List[Axes] = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by, layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [
            ("larger than required size", ["C", "D"], (1, 1)),
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),
            ("At least one dimension of layout must be positive", "C", (-1, -1)),
        ],
    )
    def test_box_plot_invalid_layout_with_by_raises(
        self,
        msg: str,
        by: Union[str, List[str]],
        layout: Any,
        hist_df: DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            hist_df.plot.box(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(
        self, figsize: Tuple[int, int], hist_df: DataFrame
    ) -> None:
        axes: List[Axes] = hist_df.plot.box(column="A", by="C", figsize=figsize)
        _check_axes_shape(axes, axes_num=1, figsize=figsize)