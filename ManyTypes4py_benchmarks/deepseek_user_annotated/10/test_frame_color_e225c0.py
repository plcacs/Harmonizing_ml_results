"""Test cases for DataFrame.plot"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_colors,
    _check_plot_works,
    _unpack_cycler,
)
from pandas.util.version import Version

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
cm = pytest.importorskip("matplotlib.cm")


def _check_colors_box(
    bp: Dict[str, Any],
    box_c: Union[str, Tuple[float, float, float]],
    whiskers_c: Union[str, Tuple[float, float, float]],
    medians_c: Union[str, Tuple[float, float, float]],
    caps_c: Union[str, Tuple[float, float, float]] = "k",
    fliers_c: Optional[Union[str, Tuple[float, float, float]]] = None,
) -> None:
    if fliers_c is None:
        fliers_c = "k"
    _check_colors(bp["boxes"], linecolors=[box_c] * len(bp["boxes"]))
    _check_colors(bp["whiskers"], linecolors=[whiskers_c] * len(bp["whiskers"]))
    _check_colors(bp["medians"], linecolors=[medians_c] * len(bp["medians"]))
    _check_colors(bp["fliers"], linecolors=[fliers_c] * len(bp["fliers"]))
    _check_colors(bp["caps"], linecolors=[caps_c] * len(bp["caps"]))


class TestDataFrameColor:
    @pytest.mark.parametrize("color", list(range(10)))
    def test_mpl2_color_cycle_str(self, color: int) -> None:
        # GH 15516
        color = f"C{color}"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        _check_plot_works(df.plot, color=color)

    def test_color_single_series_list(self) -> None:
        # GH 3486
        df = DataFrame({"A": [1, 2, 3]})
        _check_plot_works(df.plot, color=["red"])

    @pytest.mark.parametrize("color", [(1, 0, 0), (1, 0, 0, 0.5)])
    def test_rgb_tuple_color(self, color: Tuple[float, ...]) -> None:
        # GH 16695
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        _check_plot_works(df.plot, x="x", y="y", color=color)

    def test_color_empty_string(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        with pytest.raises(ValueError, match="Invalid color argument:"):
            df.plot(color="")

    def test_color_and_style_arguments(self) -> None:
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        # passing both 'color' and 'style' arguments should be allowed
        # if there is no color symbol in the style strings:
        ax = df.plot(color=["red", "black"], style=["-", "--"])
        # check that the linestyles are correctly set:
        linestyle = [line.get_linestyle() for line in ax.lines]
        assert linestyle == ["-", "--"]
        # check that the colors are correctly set:
        color = [line.get_color() for line in ax.lines]
        assert color == ["red", "black"]
        # passing both 'color' and 'style' arguments should not be allowed
        # if there is a color symbol in the style strings:
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(color=["red", "black"], style=["k-", "r--"])

    @pytest.mark.parametrize(
        "color, expected",
        [
            ("green", ["green"] * 4),
            (["yellow", "red", "green", "blue"], ["yellow", "red", "green", "blue"]),
        ],
    )
    def test_color_and_marker(self, color: Union[str, List[str]], expected: List[str]) -> None:
        # GH 21003
        df = DataFrame(np.random.default_rng(2).random((7, 4)))
        ax = df.plot(color=color, style="d--")
        # check colors
        result = [i.get_color() for i in ax.lines]
        assert result == expected
        # check markers and linestyles
        assert all(i.get_linestyle() == "--" for i in ax.lines)
        assert all(i.get_marker() == "d" for i in ax.lines)

    def test_color_and_style(self) -> None:
        color = {"g": "black", "h": "brown"}
        style = {"g": "-", "h": "--"}
        expected_color = ["black", "brown"]
        expected_style = ["-", "--"]
        df = DataFrame({"g": [1, 2], "h": [2, 3]}, index=[1, 2])
        ax = df.plot.line(color=color, style=style)
        color = [i.get_color() for i in ax.lines]
        style = [i.get_linestyle() for i in ax.lines]
        assert color == expected_color
        assert style == expected_style

    def test_bar_colors(self) -> None:
        default_colors = _unpack_cycler(plt.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar()
        _check_colors(ax.patches[::5], facecolors=default_colors[:5])

    def test_bar_colors_custom(self) -> None:
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(color=custom_colors)
        _check_colors(ax.patches[::5], facecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_bar_colors_cmap(self, colormap: Union[str, Any]) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        ax = df.plot.bar(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::5], facecolors=rgba_colors)

    def test_bar_colors_single_col(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.bar(color="DodgerBlue")
        _check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

    def test_bar_colors_green(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind="bar", color="green")
        _check_colors(ax.patches[::5], facecolors=["green"] * 5)

    def test_bar_user_colors(self) -> None:
        df = DataFrame(
            {"A": range(4), "B": range(1, 5), "color": ["red", "blue", "blue", "red"]}
        )
        # This should *only* work when `y` is specified, else
        # we use one color per column
        ax = df.plot.bar(y="A", color=df["color"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self) -> None:
        # addressing issue #10611, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax1 = df.plot.scatter(x="A label", y="B label")
        ax2 = df.plot.scatter(x="A label", y="B label", c="C label")

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
        assert vis1 == vis2

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
        assert vis1 == vis2

        assert (
            ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()
        )

    def test_if_hexbin_xaxis_label_is_visible(self) -> None:
        # addressing issue #10678, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax = df.plot.hexbin("A label", "B label", gridsize=12)
        assert all(vis.get_visible() for vis in ax.xaxis.get_minorticklabels())
        assert all(vis.get_visible() for vis in ax.xaxis.get_majorticklabels())
        assert ax.xaxis.get_label().get_visible()

    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self) -> None:
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        fig, axes = plt.subplots(1, 2)
        df.plot.scatter("A label", "B label", c="C label", ax=axes[0])
        df.plot.scatter("A label", "B label", c="C label", ax=axes[1])
        plt.tight_layout()

        points = np.array([ax.get_position().get_points() for ax in fig.axes])
        axes_x_coords = points[:, :, 0]
        parent_distance = axes_x_coords[1, :] - axes_x_coords[0, :]
        colorbar_distance = axes_x_coords[3, :] - axes_x_coords[2, :]
        assert np.isclose(parent_distance, colorbar_distance, atol=1e-7).all()

    @pytest.mark.parametrize("cmap", [None, "Greys"])
    def test_scatter_with_c_column_name_with_colors(self, cmap: Optional[str]) -> None:
        # https://github.com/pandas-dev/pandas/issues/34316

        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        df["species"] = ["r", "r", "g", "g", "b"]
        if cmap is not None:
            with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
                ax = df.plot.scatter(x=0, y=1, cmap=cmap, c="species")
        else:
            ax = df.plot.scatter(x=0, y=1, c="species", cmap=cmap)

        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 3  # r/g/b
        assert (
            np.unique(ax.collections[0].get_facecolor(), axis=0)
            == np.array(
                [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
            )  # r/g/b
        ).all()
        assert ax.collections[0].colorbar is None

    def test_scatter_with_c_column_name_without_colors(self) -> None:
        # Given
        colors = ["NY", "MD", "MA", "CA"]
        color_count = 4  # 4 unique colors

        # When
        df = DataFrame(
            {
                "dataX": range(100),
                "dataY": range(100),
                "color": (colors[i % len(colors)] for i in range(100)),
            }
        )

        # Then
        ax = df.plot.scatter("dataX", "dataY", c="color")
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count

        # Given
        colors = ["r", "g", "not-a-color"]
        color_count = 3
        # Also, since not all are mpl-colors, points matching 'r' or 'g'
        # are not necessarily red or green

        # When
        df = DataFrame(
            {
                "dataX": range(100),
                "dataY": range(100),
                "color": (colors[i % len(colors)] for i in range(100)),
            }
        )

        # Then
        ax = df.plot.scatter("dataX", "dataY", c="color")
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count

    def test_scatter_colors(self) -> None:
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        with pytest.raises(TypeError, match="Specify exactly one of `c` and `color`"):
            df.plot.scatter(x="a", y="b", c="c", color="green")

    def test_scatter_colors_not_raising_warnings(self) -> None:
        # GH-53908. Do not raise UserWarning: No data for colormapping
        # provided via 'c'. Parameters 'cmap' will be ignored
        df = DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        with tm.assert_produces_warning(None):
            ax = df.plot.scatter(x="x", y="y", c="b")
            assert (
                len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 1
            )  # blue
            assert (
                np.unique(ax.collections[0].get_facecolor(), axis=0)
                == np.array([[0.0, 0.0, 1.0, 1.0]])
            ).all()  # blue

    def test_scatter_colors_default(self) -> None:
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        ax = df.plot.scatter(x="a", y="b", c="c")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array(mpl.colors.ColorConverter.to_rgba(default_colors[0])),
        )

    def test_scatter_colors_white(self) -> None:
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        ax = df.plot.scatter(x="a", y="b", color="white")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array([1, 1, 1, 1], dtype=np.float64),
        )

    def test_scatter_colorbar_different_cmap(self) -> None:
        # GH 33389
        df = DataFrame({"x": [1, 2, 3], "y": [1, 3, 2], "c": [1, 2, 3]})
        df["x2"] = df["x"] + 1

        _, ax = plt.subplots()
        df.plot("x", "y", c="c", kind="scatter", cmap="cividis", ax=ax)
        df.plot("x2", "y", c="c", kind="scatter", cmap="magma", ax=ax)

        assert ax.collections[0].cmap.name == "cividis"
        assert ax.collections[1].cmap.name == "magma"

    def test_line_colors(self) -> None:
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        ax