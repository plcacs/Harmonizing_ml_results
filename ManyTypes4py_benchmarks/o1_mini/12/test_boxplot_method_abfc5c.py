"""Test cases for .boxplot method"""
from __future__ import annotations
import itertools
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    plotting,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_box_return_type,
    _check_plot_works,
    _check_ticks_props,
    _check_visible,
)
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

if Version(mpl.__version__) < Version("3.10"):
    verts: List[Dict[str, bool]] = [{"vert": False}, {"vert": True}]
else:
    verts = [{"orientation": "horizontal"}, {"orientation": "vertical"}]


@pytest.fixture(params=verts)
def vert(request: pytest.FixtureRequest) -> Dict[str, Union[bool, str]]:
    return request.param


def _check_ax_limits(col: Series, ax: Axes) -> None:
    y_min, y_max = ax.get_ylim()
    assert y_min <= col.min()
    assert y_max >= col.max()


class TestDataFramePlots:
    def test_stacked_boxplot_set_axis(self) -> None:
        n: int = 30
        df: DataFrame = DataFrame(
            {
                "Clinical": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Confirmed": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Discarded": np.random.default_rng(2).choice([0, 1, 2, 3], n),
            },
            index=np.arange(0, n),
        )
        ax: Axes = df.plot(kind="bar", stacked=True)
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == df.index.to_list()
        ax.set_xticks(np.arange(0, n, 10))
        plt.draw()
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == list(np.arange(0, n, 10))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs, warn",
        [
            ({"return_type": "dict"}, None),
            ({"column": ["one", "two"]}, None),
            ({"column": ["one", "two"], "by": "indic"}, UserWarning),
            ({"column": ["one"], "by": ["indic", "indic2"]}, None),
            ({"by": "indic"}, UserWarning),
            ({"by": ["indic", "indic2"]}, UserWarning),
            ({"notch": 1}, None),
            ({"by": "indic", "notch": 1}, UserWarning),
        ],
    )
    def test_boxplot_legacy1(
        self, kwargs: Dict[str, Any], warn: Optional[pytest.Warnings]
    ) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        df["indic"] = ["foo", "bar"] * 3
        df["indic2"] = ["foo", "bar", "foo"] * 2
        with tm.assert_produces_warning(warn, check_stacklevel=False):
            _check_plot_works(df.boxplot, **kwargs)

    def test_boxplot_legacy1_series(self) -> None:
        ser: Series = Series(np.random.default_rng(2).standard_normal(6))
        _check_plot_works(plotting._core.boxplot, data=ser, return_type="dict")

    def test_boxplot_legacy2(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 2)),
            columns=["Col1", "Col2"],
        )
        df["X"] = Series(["A"] * 5 + ["B"] * 5)
        df["Y"] = Series(["A"] * 10)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.boxplot, by="X")

    def test_boxplot_legacy2_with_ax(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 2)),
            columns=["Col1", "Col2"],
        )
        df["X"] = Series(["A"] * 5 + ["B"] * 5)
        df["Y"] = Series(["A"] * 10)
        _, ax = plt.subplots()
        axes: Axes = df.boxplot("Col1", by="X", ax=ax)
        ax_axes: Axes = ax.axes
        assert ax_axes is axes

    def test_boxplot_legacy2_with_ax_return_type(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 2)),
            columns=["Col1", "Col2"],
        )
        df["X"] = Series(["A"] * 5 + ["B"] * 5)
        df["Y"] = Series(["A"] * 10)
        fig, ax = plt.subplots()
        axes: Dict[str, Axes] = df.groupby("Y").boxplot(ax=ax, return_type="axes")
        ax_axes: Axes = ax.axes
        assert ax_axes is axes["A"]

    def test_boxplot_legacy2_with_multi_col(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 2)),
            columns=["Col1", "Col2"],
        )
        df["X"] = Series(["A"] * 5 + ["B"] * 5)
        df["Y"] = Series(["A"] * 10)
        fig, ax = plt.subplots()
        msg: str = "the figure containing the passed axes is being cleared"
        with tm.assert_produces_warning(UserWarning, match=msg):
            axes: Dict[str, Axes] = df.boxplot(
                column=["Col1", "Col2"], by="X", ax=ax, return_type="axes"
            )
        assert axes["Col1"].get_figure() is fig

    def test_boxplot_legacy2_by_none(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 2)),
            columns=["Col1", "Col2"],
        )
        df["X"] = Series(["A"] * 5 + ["B"] * 5)
        df["Y"] = Series(["A"] * 10)
        _, ax = plt.subplots()
        d: Dict[str, List[Any]] = df.boxplot(ax=ax, return_type="dict")
        lines: List[Any] = list(itertools.chain.from_iterable(d.values()))
        assert len(ax.get_lines()) == len(lines)

    def test_boxplot_return_type_none(self, hist_df: DataFrame) -> None:
        result: Axes = hist_df.boxplot()
        assert isinstance(result, mpl.axes.Axes)

    def test_boxplot_return_type_legacy1(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        msg: str = "return_type must be {'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(return_type="NOT_A_TYPE")
        result: Axes = df.boxplot()
        _check_box_return_type(result, "axes")

    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_boxplot_return_type_legacy_return_type(
        self, return_type: str
    ) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type=return_type)
        _check_box_return_type(result, return_type)

    def test_boxplot_axis_limits(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df.copy()
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        height_ax, weight_ax = df.boxplot(["height", "weight"], by="category")
        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        assert weight_ax._sharey == height_ax

    def test_boxplot_axis_limits_two_rows(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df.copy()
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        p = df.boxplot(["height", "weight", "age"], by="category")
        height_ax, weight_ax, age_ax = (p[0, 0], p[0, 1], p[1, 0])
        dummy_ax: Axes = p[1, 1]
        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        _check_ax_limits(df["age"], age_ax)
        assert weight_ax._sharey == height_ax
        assert age_ax._sharey == height_ax
        assert dummy_ax._sharey is None

    def test_boxplot_empty_column(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((20, 4)))
        df.loc[:, 0] = np.nan
        _check_plot_works(df.boxplot, return_type="axes")

    def test_figsize(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        result: Axes = df.boxplot(return_type="axes", figsize=(12, 8))
        assert result.figure.bbox_inches.width == 12
        assert result.figure.bbox_inches.height == 8

    def test_fontsize(self) -> None:
        df: DataFrame = DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        _check_ticks_props(
            df.boxplot("a", fontsize=16), xlabelsize=16, ylabelsize=16
        )

    def test_boxplot_numeric_data(self) -> None:
        df: DataFrame = DataFrame(
            {
                "a": date_range("2012-01-01", periods=10),
                "b": np.random.default_rng(2).standard_normal(10),
                "c": np.random.default_rng(2).standard_normal(10) + 2,
                "d": date_range("2012-01-01", periods=10).astype(str),
                "e": date_range("2012-01-01", periods=10, tz="UTC"),
                "f": timedelta_range("1 days", periods=10),
            }
        )
        ax: Axes = df.plot(kind="box")
        assert [x.get_text() for x in ax.get_xticklabels()] == ["b", "c"]

    @pytest.mark.parametrize(
        "colors_kwd, expected",
        [
            (
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
            ),
            ({"boxes": "r"}, {"boxes": "r"}),
            (
                "r",
                {"boxes": "r", "whiskers": "r", "medians": "r", "caps": "r"},
            ),
        ],
    )
    def test_color_kwd(
        self, colors_kwd: Union[Dict[str, str], str], expected: Dict[str, str]
    ) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)))
        result: Dict[str, List[Any]] = df.boxplot(color=colors_kwd, return_type="dict")
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "scheme,expected",
        [
            (
                "dark_background",
                {
                    "boxes": "#8dd3c7",
                    "whiskers": "#8dd3c7",
                    "medians": "#bfbbd9",
                    "caps": "#8dd3c7",
                },
            ),
            (
                "default",
                {
                    "boxes": "#1f77b4",
                    "whiskers": "#1f77b4",
                    "medians": "#2ca02c",
                    "caps": "#1f77b4",
                },
            ),
        ],
    )
    def test_colors_in_theme(
        self, scheme: str, expected: Dict[str, str]
    ) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)))
        plt.style.use(scheme)
        result: Dict[str, List[Any]] = df.plot.box(return_type="dict")
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "dict_colors, msg",
        [
            (
                {"boxes": "r", "invalid_key": "r"},
                "invalid key 'invalid_key'",
            )
        ],
    )
    def test_color_kwd_errors(
        self, dict_colors: Dict[str, str], msg: str
    ) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)))
        with pytest.raises(ValueError, match=msg):
            df.boxplot(color=dict_colors, return_type="dict")

    @pytest.mark.parametrize(
        "props, expected",
        [
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    def test_specified_props_kwd(
        self, props: str, expected: str
    ) -> None:
        df: DataFrame = DataFrame(
            {k: np.random.default_rng(2).random(10) for k in "ABC"}
        )
        kwd: Dict[str, Dict[str, str]] = {props: {"color": "C1"}}
        result: Dict[str, List[Any]] = df.boxplot(return_type="dict", **kwd)
        assert result[expected][0].get_color() == "C1"

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_plot_xlabel_ylabel(
        self, vert: Dict[str, Union[bool, str]]
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(
                    ["group1", "group2"], 10
                ),
            }
        )
        xlabel: str = "x"
        ylabel: str = "y"
        ax: Axes = df.plot(kind="box", xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_plot_box(
        self, vert: Dict[str, Union[bool, str]]
    ) -> None:
        rng = np.random.default_rng(2)
        df1: DataFrame = DataFrame(
            rng.integers(0, 100, size=(10, 4)), columns=list("ABCD")
        )
        df2: DataFrame = DataFrame(
            rng.integers(0, 100, size=(10, 4)), columns=list("ABCD")
        )
        xlabel: str = "x"
        ylabel: str = "y"
        _, axs = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
        df1.plot.box(ax=axs[0], xlabel=xlabel, ylabel=ylabel, **vert)
        df2.plot.box(ax=axs[1], xlabel=xlabel, ylabel=ylabel, **vert)
        for ax in axs:
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_xlabel_ylabel(
        self, vert: Dict[str, Union[bool, str]]
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(
                    ["group1", "group2"], 10
                ),
            }
        )
        xlabel: str = "x"
        ylabel: str = "y"
        ax: Axes = df.boxplot(
            xlabel=xlabel, ylabel=ylabel, **vert
        )
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_group_xlabel_ylabel(
        self, vert: Dict[str, Union[bool, str]]
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(
                    ["group1", "group2"], 10
                ),
            }
        )
        xlabel: str = "x"
        ylabel: str = "y"
        ax: Union[Axes, Dict[Any, Axes]] = df.boxplot(
            by="group", xlabel=xlabel, ylabel=ylabel, **vert
        )
        if isinstance(ax, dict):
            for subplot in ax.values():
                assert subplot.get_xlabel() == xlabel
                assert subplot.get_ylabel() == ylabel
        else:
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_group_no_xlabel_ylabel(
        self,
        vert: Dict[str, Union[bool, str]],
        request: pytest.FixtureRequest,
    ) -> None:
        if (
            Version(mpl.__version__) >= Version("3.10")
            and vert == {"orientation": "horizontal"}
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"{vert} fails starting with matplotlib 3.10"
                )
            )
        df: DataFrame = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(
                    ["group1", "group2"], 10
                ),
            }
        )
        ax: Union[Axes, Dict[Any, Axes]] = df.boxplot(
            by="group", **vert
        )
        if isinstance(ax, dict):
            for subplot in ax.values():
                target_label: str = (
                    subplot.get_xlabel()
                    if vert in [{"vert": True}, {"orientation": "vertical"}]
                    else subplot.get_ylabel()
                )
                assert target_label == pprint_thing(["group"])
        else:
            target_label = (
                ax.get_xlabel()
                if vert in [{"vert": True}, {"orientation": "vertical"}]
                else ax.get_ylabel()
            )
            assert target_label == pprint_thing(["group"])


class TestDataFrameGroupByPlots:
    def test_boxplot_legacy1(self, hist_df: DataFrame) -> None:
        grouped = hist_df.groupby(by="gender")
        with tm.assert_produces_warning(
            UserWarning, check_stacklevel=False
        ):
            axes: Dict[str, Axes] = _check_plot_works(
                grouped.boxplot, return_type="axes"
            )
        _check_axes_shape(list(axes.values()), axes_num=2, layout=(1, 2))

    def test_boxplot_legacy1_return_type(self, hist_df: DataFrame) -> None:
        grouped = hist_df.groupby(by="gender")
        axes: Axes = _check_plot_works(
            grouped.boxplot, subplots=False, return_type="axes"
        )
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_boxplot_legacy2(self) -> None:
        tuples: List[Tuple[str, int]] = list(zip(string.ascii_letters[:10], range(10)))
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        grouped = df.groupby(level=1)
        with tm.assert_produces_warning(
            UserWarning, check_stacklevel=False
        ):
            axes: Dict[str, Axes] = _check_plot_works(
                grouped.boxplot, return_type="axes"
            )
        _check_axes_shape(
            list(axes.values()), axes_num=10, layout=(4, 3)
        )

    @pytest.mark.slow
    def test_boxplot_legacy2_return_type(self) -> None:
        tuples: List[Tuple[str, int]] = list(zip(string.ascii_letters[:10], range(10)))
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        grouped = df.groupby(level=1)
        axes: Axes = _check_plot_works(
            grouped.boxplot, subplots=False, return_type="axes"
        )
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_grouped_plot_fignums(self) -> None:
        n: int = 10
        weight: Series = Series(
            np.random.default_rng(2).normal(166, 20, size=n)
        )
        height: Series = Series(
            np.random.default_rng(2).normal(60, 10, size=n)
        )
        gender: np.ndarray = np.random.default_rng(2).choice(
            ["male", "female"], size=n
        )
        df: DataFrame = DataFrame(
            {"height": height, "weight": weight, "gender": gender}
        )
        gb = df.groupby("gender")
        res = gb.plot()
        assert len(mpl.pyplot.get_fignums()) == 2
        assert len(res) == 2
        plt.close("all")
        res = gb.boxplot(return_type="axes")
        assert len(mpl.pyplot.get_fignums()) == 1
        assert len(res) == 2

    def test_grouped_plot_fignums_excluded_col(self) -> None:
        n: int = 10
        weight: Series = Series(
            np.random.default_rng(2).normal(166, 20, size=n)
        )
        height: Series = Series(
            np.random.default_rng(2).normal(60, 10, size=n)
        )
        gender: np.ndarray = np.random.default_rng(2).choice(
            ["male", "female"], size=n
        )
        df: DataFrame = DataFrame(
            {"height": height, "weight": weight, "gender": gender}
        )
        df.groupby("gender").hist()

    @pytest.mark.slow
    def test_grouped_box_return_type(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        result: np.ndarray = df.boxplot(by="gender")
        assert isinstance(result, np.ndarray)
        _check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    def test_grouped_box_return_type_groupby(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        result: Dict[str, Axes] = df.groupby("gender").boxplot(
            return_type="dict"
        )
        _check_box_return_type(
            result, "dict", expected_keys=["Male", "Female"]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "return_type", ["dict", "axes", "both"]
    )
    def test_grouped_box_return_type_arg(
        self, hist_df: DataFrame, return_type: str
    ) -> None:
        df: DataFrame = hist_df
        returned = df.groupby("classroom").boxplot(
            return_type=return_type
        )
        _check_box_return_type(
            returned, return_type, expected_keys=["A", "B", "C"]
        )
        returned = df.boxplot(by="classroom", return_type=return_type)
        _check_box_return_type(
            returned, return_type, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "return_type", ["dict", "axes", "both"]
    )
    def test_grouped_box_return_type_arg_duplcate_cats(
        self, return_type: str
    ) -> None:
        columns2: List[str] = ["X", "B", "C", "D", "A"]
        df2: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((6, 5)),
            columns=columns2,
        )
        categories2: List[str] = ["A", "B"]
        df2["category"] = categories2 * 3
        returned = df2.groupby("category").boxplot(
            return_type=return_type
        )
        _check_box_return_type(
            returned, return_type, expected_keys=categories2
        )
        returned = df2.boxplot(by="category", return_type=return_type)
        _check_box_return_type(
            returned, return_type, expected_keys=columns2
        )

    @pytest.mark.slow
    def test_grouped_box_layout_too_small(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        msg: str = "Layout of 1x1 must be larger than required size 2"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(1, 1))

    @pytest.mark.slow
    def test_grouped_box_layout_needs_by(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        msg: str = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(
                column=["height", "weight", "category"],
                layout=(2, 1),
                return_type="dict",
            )

    @pytest.mark.slow
    def test_grouped_box_layout_positive_layout(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        msg: str = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(
                column=["weight", "height"],
                by=df.gender,
                layout=(-1, -1),
            )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "gb_key, axes_num, rows",
        [
            ("gender", 2, 1),
            ("category", 4, 2),
            ("classroom", 3, 2),
        ],
    )
    def test_grouped_box_layout_positive_layout_axes(
        self, hist_df: DataFrame, gb_key: str, axes_num: int, rows: int
    ) -> None:
        df: DataFrame = hist_df
        with tm.assert_produces_warning(
            UserWarning, check_stacklevel=False
        ):
            _check_plot_works(
                df.groupby(gb_key).boxplot,
                column="height",
                return_type="dict",
            )
        _check_axes_shape(
            mpl.pyplot.gcf().axes, axes_num=axes_num, layout=(rows, 2)
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "col, visible",
        [
            ("height", False),
            ("weight", True),
            ("category", True),
        ],
    )
    def test_grouped_box_layout_visible(
        self, hist_df: DataFrame, col: str, visible: bool
    ) -> None:
        df: DataFrame = hist_df
        axes: Dict[str, Axes] = df.boxplot(
            column=["height", "weight", "category"],
            by="gender",
            return_type="axes",
        )
        _check_axes_shape(
            mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2)
        )
        ax: Axes = axes[col]
        _check_visible(ax.get_xticklabels(), visible=visible)
        _check_visible([ax.xaxis.get_label()], visible=visible)

    @pytest.mark.slow
    def test_grouped_box_layout_shape(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        df.groupby("classroom").boxplot(
            column=["height", "weight", "category"], return_type="dict"
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize("cols", [2, -1])
    def test_grouped_box_layout_works(
        self, hist_df: DataFrame, cols: int
    ) -> None:
        df: DataFrame = hist_df
        with tm.assert_produces_warning(
            UserWarning, check_stacklevel=False
        ):
            _check_plot_works(
                df.groupby("category").boxplot,
                column="height",
                layout=(3, cols),
                return_type="dict",
            )
        _check_axes_shape(
            mpl.pyplot.gcf().axes, axes_num=4, layout=(3, 2)
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("rows, res", [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_rows(
        self, hist_df: DataFrame, rows: int, res: int
    ) -> None:
        df: DataFrame = hist_df
        df.boxplot(
            column=["height", "weight", "category"],
            by="gender",
            layout=(rows, 1),
        )
        _check_axes_shape(
            mpl.pyplot.gcf().axes, axes_num=3, layout=(res, 1)
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("cols, res", [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_cols_groupby(
        self, hist_df: DataFrame, cols: int, res: int
    ) -> None:
        df: DataFrame = hist_df
        df.groupby("classroom").boxplot(
            column=["height", "weight", "category"],
            layout=(1, cols),
            return_type="dict",
        )
        _check_axes_shape(
            mpl.pyplot.gcf().axes, axes_num=3, layout=(1, res)
        )

    @pytest.mark.slow
    def test_grouped_box_multiple_axes(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        with tm.assert_produces_warning(
            UserWarning, match="sharex and sharey"
        ):
            _, axes = plt.subplots(2, 2)
            df.groupby("category").boxplot(
                column="height", return_type="axes", ax=axes
            )
            _check_axes_shape(
                mpl.pyplot.gcf().axes, axes_num=4, layout=(2, 2)
            )

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_on_fig(
        self, hist_df: DataFrame
    ) -> None:
        df: DataFrame = hist_df
        fig, axes = plt.subplots(2, 3)
        with tm.assert_produces_warning(
            UserWarning, match="sharex and sharey"
        ):
            returned = df.boxplot(
                column=["height", "weight", "category"],
                by="gender",
                return_type="axes",
                ax=axes[0],
            )
        returned_array: np.ndarray = np.array(list(returned.values()))
        _check_axes_shape(returned_array, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned_array, axes[0])
        assert returned_array[0].figure is fig
        with tm.assert_produces_warning(
            UserWarning, match="sharex and sharey"
        ):
            returned = df.groupby("classroom").boxplot(
                column=["height", "weight", "category"],
                return_type="axes",
                ax=axes[1],
            )
        returned_array = np.array(list(returned.values()))
        _check_axes_shape(returned_array, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned_array, axes[1])
        assert returned_array[0].figure is fig

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_ax_error(
        self, hist_df: DataFrame
    ) -> None:
        df: DataFrame = hist_df
        msg: str = "The number of passed axes must be 3, the same as the output plot"
        _, axes = plt.subplots(2, 3)
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                UserWarning, match="sharex and sharey"
            ):
                df.groupby("classroom").boxplot(ax=axes)

    def test_fontsize_grouped_box(self) -> None:
        df: DataFrame = DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
        )
        _check_ticks_props(
            df.boxplot("a", by="b", fontsize=16),
            xlabelsize=16,
            ylabelsize=16,
        )

    @pytest.mark.parametrize(
        "col, expected_xticklabel",
        [
            (
                "v",
                ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"],
            ),
            (
                ["v"],
                ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"],
            ),
            (
                "v1",
                ["(a, v1)", "(b, v1)", "(c, v1)", "(d, v1)", "(e, v1)"],
            ),
            (
                ["v", "v1"],
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
            (
                None,
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
        ],
    )
    def test_groupby_boxplot_subplots_false(
        self,
        col: Union[str, List[str], None],
        expected_xticklabel: List[str],
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "cat": np.random.default_rng(2).choice(list("abcde"), 100),
                "v": np.random.default_rng(2).random(100),
                "v1": np.random.default_rng(2).random(100),
            }
        )
        grouped = df.groupby("cat")
        axes: Axes = _check_plot_works(
            grouped.boxplot,
            subplots=False,
            column=col,
            return_type="axes",
        )
        result_xticklabel: List[str] = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

    def test_groupby_boxplot_object(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df.astype("object")
        grouped = df.groupby("gender")
        msg: str = "boxplot method requires numerical columns, nothing to plot"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(grouped.boxplot, subplots=False)

    def test_boxplot_multiindex_column(self) -> None:
        arrays: List[List[str]] = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples: List[Tuple[str, str]] = list(zip(*arrays))
        index: MultiIndex = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((3, 8)),
            index=["A", "B", "C"],
            columns=index,
        )
        col: List[Tuple[str, str]] = [("bar", "one"), ("bar", "two")]
        axes: Axes = _check_plot_works(
            df.boxplot, column=col, return_type="axes"
        )
        expected_xticklabel: List[str] = ["(bar, one)", "(bar, two)"]
        result_xticklabel: List[str] = [
            x.get_text() for x in axes.get_xticklabels()
        ]
        assert expected_xticklabel == result_xticklabel

    @pytest.mark.parametrize(
        "group",
        ["X", ["X", "Y"]],
    )
    def test_boxplot_multi_groupby_groups(
        self, group: Union[str, List[str]]
    ) -> None:
        rows: int = 20
        df: DataFrame = DataFrame(
            np.random.default_rng(12).normal(size=(rows, 2)),
            columns=["Col1", "Col2"],
        )
        df["X"] = Series(np.repeat(["A", "B"], int(rows / 2)))
        df["Y"] = Series(np.tile(["C", "D"], int(rows / 2)))
        grouped = df.groupby(group)
        _check_plot_works(
            df.boxplot, by=group, default_axes=True
        )
        _check_plot_works(
            df.plot.box, by=group, default_axes=True
        )
        _check_plot_works(
            grouped.boxplot, default_axes=True
        )
