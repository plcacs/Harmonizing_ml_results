#!/usr/bin/env python3
"""Test cases for DataFrame.plot"""
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import _check_colors, _check_plot_works, _unpack_cycler
from pandas.util.version import Version

mpl = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')
cm = pytest.importorskip('matplotlib.cm')


def _check_colors_box(
    bp: Dict[str, List[Any]],
    box_c: Any,
    whiskers_c: Any,
    medians_c: Any,
    caps_c: str = 'k',
    fliers_c: Optional[Any] = None
) -> None:
    if fliers_c is None:
        fliers_c = 'k'
    _check_colors(bp['boxes'], linecolors=[box_c] * len(bp['boxes']))
    _check_colors(bp['whiskers'], linecolors=[whiskers_c] * len(bp['whiskers']))
    _check_colors(bp['medians'], linecolors=[medians_c] * len(bp['medians']))
    _check_colors(bp['fliers'], linecolors=[fliers_c] * len(bp['fliers']))
    _check_colors(bp['caps'], linecolors=[caps_c] * len(bp['caps']))


class TestDataFrameColor:

    @pytest.mark.parametrize('color', list(range(10)))
    def test_mpl2_color_cycle_str(self, color: int) -> None:
        color = f'C{color}'
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
        _check_plot_works(df.plot, color=color)

    def test_color_single_series_list(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3]})
        _check_plot_works(df.plot, color=['red'])

    @pytest.mark.parametrize('color', [(1, 0, 0), (1, 0, 0, 0.5)])
    def test_rgb_tuple_color(self, color: Union[Tuple[float, float, float], Tuple[float, float, float, float]]) -> None:
        df: DataFrame = DataFrame({'x': [1, 2], 'y': [3, 4]})
        _check_plot_works(df.plot, x='x', y='y', color=color)

    def test_color_empty_string(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        with pytest.raises(ValueError, match='Invalid color argument:'):
            df.plot(color='')

    def test_color_and_style_arguments(self) -> None:
        df: DataFrame = DataFrame({'x': [1, 2], 'y': [3, 4]})
        ax = df.plot(color=['red', 'black'], style=['-', '--'])
        linestyle: List[Any] = [line.get_linestyle() for line in ax.lines]
        assert linestyle == ['-', '--']
        color = [line.get_color() for line in ax.lines]
        assert color == ['red', 'black']
        msg: str = ("Cannot pass 'style' string with a color symbol and 'color' keyword argument. "
                    "Please use one or the other or pass 'style' without a color symbol")
        with pytest.raises(ValueError, match=msg):
            df.plot(color=['red', 'black'], style=['k-', 'r--'])

    @pytest.mark.parametrize('color, expected', [
        ('green', ['green'] * 4),
        (['yellow', 'red', 'green', 'blue'], ['yellow', 'red', 'green', 'blue'])
    ])
    def test_color_and_marker(self, color: Union[str, List[str]], expected: List[str]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((7, 4)))
        ax = df.plot(color=color, style='d--')
        result: List[Any] = [i.get_color() for i in ax.lines]
        assert result == expected
        assert all((i.get_linestyle() == '--' for i in ax.lines))
        assert all((i.get_marker() == 'd' for i in ax.lines))

    def test_color_and_style(self) -> None:
        color: Dict[str, str] = {'g': 'black', 'h': 'brown'}
        style: Dict[str, str] = {'g': '-', 'h': '--'}
        expected_color: List[str] = ['black', 'brown']
        expected_style: List[str] = ['-', '--']
        df: DataFrame = DataFrame({'g': [1, 2], 'h': [2, 3]}, index=[1, 2])
        ax = df.plot.line(color=color, style=style)
        color_result = [i.get_color() for i in ax.lines]
        style_result = [i.get_linestyle() for i in ax.lines]
        assert color_result == expected_color
        assert style_result == expected_style

    def test_bar_colors(self) -> None:
        default_colors: List[Any] = _unpack_cycler(plt.rcParams)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar()
        _check_colors(ax.patches[::5], facecolors=default_colors[:5])

    def test_bar_colors_custom(self) -> None:
        custom_colors: str = 'rgcby'
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(color=custom_colors)
        _check_colors(ax.patches[::5], facecolors=custom_colors)

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_bar_colors_cmap(self, colormap: Union[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(colormap=colormap)
        rgba_colors: List[Any] = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::5], facecolors=rgba_colors)

    def test_bar_colors_single_col(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.bar(color='DodgerBlue')
        _check_colors([ax.patches[0]], facecolors=['DodgerBlue'])

    def test_bar_colors_green(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind='bar', color='green')
        _check_colors(ax.patches[::5], facecolors=['green'] * 5)

    def test_bar_user_colors(self) -> None:
        df: DataFrame = DataFrame({'A': list(range(4)), 'B': list(range(1, 5)), 'color': ['red', 'blue', 'blue', 'red']})
        ax = df.plot.bar(y='A', color=df['color'])
        result: List[Any] = [p.get_facecolor() for p in ax.patches]
        expected: List[Tuple[float, float, float, float]] = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0)
        ]
        assert result == expected

    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self) -> None:
        random_array: np.ndarray = np.random.default_rng(2).random((10, 3))
        df: DataFrame = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
        ax1 = df.plot.scatter(x='A label', y='B label')
        ax2 = df.plot.scatter(x='A label', y='B label', c='C label')
        vis1: List[bool] = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
        vis2: List[bool] = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
        assert vis1 == vis2
        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
        assert vis1 == vis2
        assert ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()

    def test_if_hexbin_xaxis_label_is_visible(self) -> None:
        random_array: np.ndarray = np.random.default_rng(2).random((10, 3))
        df: DataFrame = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
        ax = df.plot.hexbin('A label', 'B label', gridsize=12)
        assert all((vis.get_visible() for vis in ax.xaxis.get_minorticklabels()))
        assert all((vis.get_visible() for vis in ax.xaxis.get_majorticklabels()))
        assert ax.xaxis.get_label().get_visible()

    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self) -> None:
        random_array: np.ndarray = np.random.default_rng(2).random((10, 3))
        df: DataFrame = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
        fig, axes = plt.subplots(1, 2)
        df.plot.scatter('A label', 'B label', c='C label', ax=axes[0])
        df.plot.scatter('A label', 'B label', c='C label', ax=axes[1])
        plt.tight_layout()
        points: np.ndarray = np.array([ax.get_position().get_points() for ax in fig.axes])
        axes_x_coords: np.ndarray = points[:, :, 0]
        parent_distance: np.ndarray = axes_x_coords[1, :] - axes_x_coords[0, :]
        colorbar_distance: np.ndarray = axes_x_coords[3, :] - axes_x_coords[2, :]
        assert np.isclose(parent_distance, colorbar_distance, atol=1e-07).all()

    @pytest.mark.parametrize('cmap', [None, 'Greys'])
    def test_scatter_with_c_column_name_with_colors(self, cmap: Optional[Union[str, Any]]) -> None:
        df: DataFrame = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=['length', 'width']
        )
        df['species'] = ['r', 'r', 'g', 'g', 'b']
        if cmap is not None:
            with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
                ax = df.plot.scatter(x=0, y=1, cmap=cmap, c='species')
        else:
            ax = df.plot.scatter(x=0, y=1, c='species', cmap=cmap)
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 3
        expected_colors = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.5, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0]
        ])
        assert (np.unique(ax.collections[0].get_facecolor(), axis=0) == expected_colors).all()
        assert ax.collections[0].colorbar is None

    def test_scatter_with_c_column_name_without_colors(self) -> None:
        colors: List[str] = ['NY', 'MD', 'MA', 'CA']
        color_count: int = 4
        df: DataFrame = DataFrame({
            'dataX': list(range(100)),
            'dataY': list(range(100)),
            'color': (colors[i % len(colors)] for i in range(100))
        })
        ax = df.plot.scatter('dataX', 'dataY', c='color')
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count
        colors = ['r', 'g', 'not-a-color']
        color_count = 3
        df = DataFrame({
            'dataX': list(range(100)),
            'dataY': list(range(100)),
            'color': (colors[i % len(colors)] for i in range(100))
        })
        ax = df.plot.scatter('dataX', 'dataY', c='color')
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count

    def test_scatter_colors(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
        with pytest.raises(TypeError, match='Specify exactly one of `c` and `color`'):
            df.plot.scatter(x='a', y='b', c='c', color='green')

    def test_scatter_colors_not_raising_warnings(self) -> None:
        df: DataFrame = DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
        with tm.assert_produces_warning(None):
            ax = df.plot.scatter(x='x', y='y', c='b')
            assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 1
            assert (np.unique(ax.collections[0].get_facecolor(), axis=0) ==
                    np.array([[0.0, 0.0, 1.0, 1.0]])).all()

    def test_scatter_colors_default(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
        default_colors: List[Any] = _unpack_cycler(mpl.pyplot.rcParams)
        ax = df.plot.scatter(x='a', y='b', c='c')
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array(mpl.colors.ColorConverter.to_rgba(default_colors[0]))
        )

    def test_scatter_colors_white(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
        ax = df.plot.scatter(x='a', y='b', color='white')
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array([1, 1, 1, 1], dtype=np.float64)
        )

    def test_scatter_colorbar_different_cmap(self) -> None:
        df: DataFrame = DataFrame({'x': [1, 2, 3], 'y': [1, 3, 2], 'c': [1, 2, 3]})
        df['x2'] = df['x'] + 1
        _, ax = plt.subplots()
        df.plot('x', 'y', c='c', kind='scatter', cmap='cividis', ax=ax)
        df.plot('x2', 'y', c='c', kind='scatter', cmap='magma', ax=ax)
        assert ax.collections[0].cmap.name == 'cividis'
        assert ax.collections[1].cmap.name == 'magma'

    def test_line_colors(self) -> None:
        custom_colors: str = 'rgcby'
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)
        plt.close('all')
        ax2 = df.plot(color=custom_colors)
        lines2 = ax2.get_lines()
        for l1, l2 in zip(ax.get_lines(), lines2):
            assert l1.get_color() == l2.get_color()

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_line_colors_cmap(self, colormap: Union[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(colormap=colormap)
        rgba_colors: List[Any] = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    def test_line_colors_single_col(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot(color='DodgerBlue')
        _check_colors(ax.lines, linecolors=['DodgerBlue'])

    def test_line_colors_single_color(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(color='red')
        _check_colors(ax.get_lines(), linecolors=['red'] * 5)

    def test_line_colors_hex(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors: List[str] = ['#FF0000', '#0000FF', '#FFFF00', '#000000', '#FFFFFF']
        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    def test_dont_modify_colors(self) -> None:
        colors: List[str] = ['r', 'g', 'b']
        DataFrame(np.random.default_rng(2).random((10, 2))).plot(color=colors)
        assert len(colors) == 3

    def test_line_colors_and_styles_subplots(self) -> None:
        default_colors: List[Any] = _unpack_cycler(mpl.pyplot.rcParams)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize('color', ['k', 'green'])
    def test_line_colors_and_styles_subplots_single_color_str(self, color: str) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(subplots=True, color=color)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[color])

    @pytest.mark.parametrize('color', ['rgcby', list('rgcby')])
    def test_line_colors_and_styles_subplots_custom_colors(self, color: Union[str, List[str]]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(color=color, subplots=True)
        for ax, c in zip(axes, list(color) if isinstance(color, str) else color):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_colormap_hex(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors: List[str] = ['#FF0000', '#0000FF', '#FFFF00', '#000000', '#FFFFFF']
        axes = df.plot(color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize('cmap', ['jet', cm.jet])
    def test_line_colors_and_styles_subplots_colormap_subplot(self, cmap: Union[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        rgba_colors: List[Any] = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        axes = df.plot(colormap=cmap, subplots=True)
        for ax, c in zip(axes, rgba_colors):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_single_col(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.loc[:, [0]].plot(color='DodgerBlue', subplots=True)
        _check_colors(axes[0].lines, linecolors=['DodgerBlue'])

    def test_line_colors_and_styles_subplots_single_char(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(style='r', subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=['r'])

    def test_line_colors_and_styles_subplots_list_styles(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        styles: List[str] = list('rgcby')
        axes = df.plot(style=styles, subplots=True)
        for ax, c in zip(axes, styles):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_area_colors(self) -> None:
        custom_colors: str = 'rgcby'
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax = df.plot.area(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)
        poly: List[Any] = [o for o in ax.get_children() if isinstance(o, mpl.collections.PolyCollection)]
        _check_colors(poly, facecolors=custom_colors)
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=custom_colors)
        for h in handles:
            assert h.get_alpha() is None

    def test_area_colors_poly(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax = df.plot.area(colormap='jet')
        jet_colors: List[Any] = [mpl.cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        poly: List[Any] = [o for o in ax.get_children() if isinstance(o, mpl.collections.PolyCollection)]
        _check_colors(poly, facecolors=jet_colors)
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=jet_colors)
        for h in handles:
            assert h.get_alpha() is None

    def test_area_colors_stacked_false(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 5)))
        jet_colors: List[Any] = [mpl.cm.jet(n) for n in np.linspace(0, 1, len(df))]
        ax = df.plot.area(colormap=mpl.cm.jet, stacked=False)
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        poly: List[Any] = [o for o in ax.get_children() if isinstance(o, mpl.collections.PolyCollection)]
        jet_with_alpha: List[Tuple[float, float, float, float]] = [(c[0], c[1], c[2], 0.5) for c in jet_colors]
        _check_colors(poly, facecolors=jet_with_alpha)
        handles, _ = ax.get_legend_handles_labels()
        linecolors = jet_with_alpha
        _check_colors(handles[:len(jet_colors)], linecolors=linecolors)
        for h in handles:
            assert h.get_alpha() == 0.5

    def test_hist_colors(self) -> None:
        default_colors: List[Any] = _unpack_cycler(plt.rcParams)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.hist()
        _check_colors(ax.patches[::10], facecolors=default_colors[:5])

    def test_hist_colors_single_custom(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors: str = 'rgcby'
        ax = df.plot.hist(color=custom_colors)
        _check_colors(ax.patches[::10], facecolors=custom_colors)

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_hist_colors_cmap(self, colormap: Union[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.hist(colormap=colormap)
        rgba_colors: List[Any] = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::10], facecolors=rgba_colors)

    def test_hist_colors_single_col(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.hist(color='DodgerBlue')
        _check_colors([ax.patches[0]], facecolors=['DodgerBlue'])

    def test_hist_colors_single_color(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind='hist', color='green')
        _check_colors(ax.patches[::10], facecolors=['green'] * 5)

    def test_kde_colors(self) -> None:
        pytest.importorskip('scipy')
        custom_colors: str = 'rgcby'
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax = df.plot.kde(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_kde_colors_cmap(self, colormap: Union[str, Any]) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.kde(colormap=colormap)
        rgba_colors: List[Any] = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    def test_kde_colors_and_styles_subplots(self) -> None:
        pytest.importorskip('scipy')
        default_colors: List[Any] = _unpack_cycler(plt.rcParams)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(kind='kde', subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize('colormap', ['k', 'red'])
    def test_kde_colors_and_styles_subplots_single_col_str(self, colormap: str) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(kind='kde', color=colormap, subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[colormap])

    def test_kde_colors_and_styles_subplots_custom_color(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors: str = 'rgcby'
        axes = df.plot(kind='kde', color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_kde_colors_and_styles_subplots_cmap(self, colormap: Union[str, Any]) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        rgba_colors: List[Any] = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        axes = df.plot(kind='kde', colormap=colormap, subplots=True)
        for ax, c in zip(axes, rgba_colors):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_kde_colors_and_styles_subplots_single_col(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.loc[:, [0]].plot(kind='kde', color='DodgerBlue', subplots=True)
        _check_colors(axes[0].lines, linecolors=['DodgerBlue'])

    def test_kde_colors_and_styles_subplots_single_char(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(kind='kde', style='r', subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=['r'])

    def test_kde_colors_and_styles_subplots_list(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        styles: List[str] = list('rgcby')
        axes = df.plot(kind='kde', style=styles, subplots=True)
        for ax, c in zip(axes, styles):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_boxplot_colors(self) -> None:
        default_colors: List[Any] = _unpack_cycler(plt.rcParams)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp: Dict[str, List[Any]] = df.plot.box(return_type='dict')
        _check_colors_box(bp, default_colors[0], default_colors[0], default_colors[2], default_colors[0])

    def test_boxplot_colors_dict_colors(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        dict_colors: Dict[str, str] = {
            'boxes': '#572923',
            'whiskers': '#982042',
            'medians': '#804823',
            'caps': '#123456'
        }
        bp: Dict[str, List[Any]] = df.plot.box(color=dict_colors, sym='r+', return_type='dict')
        _check_colors_box(bp, dict_colors['boxes'], dict_colors['whiskers'], dict_colors['medians'], dict_colors['caps'], 'r')

    def test_boxplot_colors_default_color(self) -> None:
        default_colors: List[Any] = _unpack_cycler(plt.rcParams)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        dict_colors: Dict[str, str] = {'whiskers': 'c', 'medians': 'm'}
        bp: Dict[str, List[Any]] = df.plot.box(color=dict_colors, return_type='dict')
        _check_colors_box(bp, default_colors[0], 'c', 'm', default_colors[0])

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_boxplot_colors_cmap(self, colormap: Union[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp: Dict[str, List[Any]] = df.plot.box(colormap=colormap, return_type='dict')
        jet_colors: List[Any] = [cm.jet(n) for n in np.linspace(0, 1, 3)]
        _check_colors_box(bp, jet_colors[0], jet_colors[0], jet_colors[2], jet_colors[0])

    def test_boxplot_colors_single(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp: Dict[str, List[Any]] = df.plot.box(color='DodgerBlue', return_type='dict')
        _check_colors_box(bp, 'DodgerBlue', 'DodgerBlue', 'DodgerBlue', 'DodgerBlue')

    def test_boxplot_colors_tuple(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp: Dict[str, List[Any]] = df.plot.box(color=(0, 1, 0), sym='#123456', return_type='dict')
        _check_colors_box(bp, (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), '#123456')

    def test_boxplot_colors_invalid(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        msg: str = re.escape("color dict contains invalid key 'xxxx'. The key must be either ['boxes', 'whiskers', 'medians', 'caps']")
        with pytest.raises(ValueError, match=msg):
            df.plot.box(color={'boxes': 'red', 'xxxx': 'blue'})

    def test_default_color_cycle(self) -> None:
        import cycler
        colors: List[str] = list('rgbk')
        plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', colors)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        ax = df.plot()
        expected: List[Any] = _unpack_cycler(plt.rcParams)[:3]
        _check_colors(ax.get_lines(), linecolors=expected)

    def test_no_color_bar(self) -> None:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).uniform(size=20),
            'B': np.random.default_rng(2).uniform(size=20),
            'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)
        })
        ax = df.plot.hexbin(x='A', y='B', colorbar=None)
        assert ax.collections[0].colorbar is None

    def test_mixing_cmap_and_colormap_raises(self) -> None:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).uniform(size=20),
            'B': np.random.default_rng(2).uniform(size=20),
            'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)
        })
        msg: str = 'Only specify one of `cmap` and `colormap`'
        with pytest.raises(TypeError, match=msg):
            df.plot.hexbin(x='A', y='B', cmap='YlGn', colormap='BuGn')

    def test_passed_bar_colors(self) -> None:
        color_tuples: List[Tuple[float, float, float, float]] = [
            (0.9, 0, 0, 1),
            (0, 0.9, 0, 1),
            (0, 0, 0.9, 1)
        ]
        colormap = mpl.colors.ListedColormap(color_tuples)
        barplot = DataFrame([[1, 2, 3]]).plot(kind='bar', cmap=colormap)
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_rcParams_bar_colors(self) -> None:
        color_tuples: List[Tuple[float, float, float, float]] = [
            (0.9, 0, 0, 1),
            (0, 0.9, 0, 1),
            (0, 0, 0.9, 1)
        ]
        with mpl.rc_context(rc={'axes.prop_cycle': mpl.cycler('color', color_tuples)}):
            barplot = DataFrame([[1, 2, 3]]).plot(kind='bar')
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_colors_of_columns_with_same_name(self) -> None:
        df: DataFrame = DataFrame({'b': [0, 1, 0], 'a': [1, 2, 3]})
        df1: DataFrame = DataFrame({'a': [2, 4, 6]})
        df_concat: DataFrame = pd.concat([df, df1], axis=1)
        result = df_concat.plot()
        if Version(mpl.__version__) < Version('3.7'):
            handles = result.get_legend().legendHandles
        else:
            handles = result.get_legend().legend_handles
        for legend_item, line in zip(handles, result.lines):
            assert legend_item.get_color() == line.get_color()

    def test_invalid_colormap(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 2)), columns=['A', 'B'])
        msg: str = '(is not a valid value)|(is not a known colormap)'
        with pytest.raises((ValueError, KeyError), match=msg):
            df.plot(colormap='invalid_colormap')

    def test_dataframe_none_color(self) -> None:
        df: DataFrame = DataFrame([[1, 2, 3]])
        ax = df.plot(color=None)
        expected: List[Any] = _unpack_cycler(plt.rcParams)[:3]
        _check_colors(ax.get_lines(), linecolors=expected)
