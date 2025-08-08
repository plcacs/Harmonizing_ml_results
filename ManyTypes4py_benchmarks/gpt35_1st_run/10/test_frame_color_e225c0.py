import re
from typing import List, Tuple, Dict
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def _check_colors_box(bp: Dict[str, List], box_c: str, whiskers_c: str, medians_c: str, caps_c: str = 'k', fliers_c: str = None) -> None:
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
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
        _check_plot_works(df.plot, color=color)

    def test_color_single_series_list(self) -> None:
        df = DataFrame({'A': [1, 2, 3]})
        _check_plot_works(df.plot, color=['red'])

    @pytest.mark.parametrize('color', [(1, 0, 0), (1, 0, 0, 0.5)])
    def test_rgb_tuple_color(self, color: Tuple) -> None:
        df = DataFrame({'x': [1, 2], 'y': [3, 4]})
        _check_plot_works(df.plot, x='x', y='y', color=color)

    def test_color_empty_string(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        with pytest.raises(ValueError, match='Invalid color argument:'):
            df.plot(color='')

    def test_color_and_style_arguments(self) -> None:
        df = DataFrame({'x': [1, 2], 'y': [3, 4]})
        ax = df.plot(color=['red', 'black'], style=['-', '--'])
        linestyle = [line.get_linestyle() for line in ax.lines]
        assert linestyle == ['-', '--']
        color = [line.get_color() for line in ax.lines]
        assert color == ['red', 'black']
        msg = "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
        with pytest.raises(ValueError, match=msg):
            df.plot(color=['red', 'black'], style=['k-', 'r--'])

    @pytest.mark.parametrize('color, expected', [('green', ['green'] * 4), (['yellow', 'red', 'green', 'blue'], ['yellow', 'red', 'green', 'blue'])])
    def test_color_and_marker(self, color: str, expected: List[str]) -> None:
        df = DataFrame(np.random.default_rng(2).random((7, 4)))
        ax = df.plot(color=color, style='d--')
        result = [i.get_color() for i in ax.lines]
        assert result == expected
        assert all((i.get_linestyle() == '--' for i in ax.lines))
        assert all((i.get_marker() == 'd' for i in ax.lines))

    def test_color_and_style(self) -> None:
        color = {'g': 'black', 'h': 'brown'}
        style = {'g': '-', 'h': '--'}
        expected_color = ['black', 'brown']
        expected_style = ['-', '--']
        df = DataFrame({'g': [1, 2], 'h': [2, 3]}, index=[1, 2])
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
        custom_colors = 'rgcby'
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(color=custom_colors)
        _check_colors(ax.patches[::5], facecolors=custom_colors)

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_bar_colors_cmap(self, colormap: str) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::5], facecolors=rgba_colors)

    def test_bar_colors_single_col(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.bar(color='DodgerBlue')
        _check_colors([ax.patches[0]], facecolors=['DodgerBlue'])

    def test_bar_colors_green(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind='bar', color='green')
        _check_colors(ax.patches[::5], facecolors=['green'] * 5)

    def test_bar_user_colors(self) -> None:
        df = DataFrame({'A': range(4), 'B': range(1, 5), 'color': ['red', 'blue', 'blue', 'red']})
        ax = df.plot.bar(y='A', color=df['color'])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
        assert result == expected

    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self) -> None:
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
        ax1 = df.plot.scatter(x='A label', y='B label')
        ax2 = df.plot.scatter(x='A label', y='B label', c='C label')
        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
        assert vis1 == vis2
        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
        assert vis1 == vis2
        assert ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()

    def test_if_hexbin_xaxis_label_is_visible(self) -> None:
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
        ax = df.plot.hexbin('A label', 'B label', gridsize=12)
        assert all((vis.get_visible() for vis in ax.xaxis.get_minorticklabels()))
        assert all((vis.get_visible() for vis in ax.xaxis.get_majorticklabels()))
        assert ax.xaxis.get_label().get_visible()

    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self) -> None:
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
        fig, axes = plt.subplots(1, 2)
        df.plot.scatter('A label', 'B label', c='C label', ax=axes[0])
        df.plot.scatter('A label', 'B label', c='C label', ax=axes[1])
        plt.tight_layout()
        points = np.array([ax.get_position().get_points() for ax in fig.axes])
        axes_x_coords = points[:, :, 0]
        parent_distance = axes_x_coords[1, :] - axes_x_coords[0, :]
        colorbar_distance = axes_x_coords[3, :] - axes_x_coords[2, :]
        assert np.isclose(parent_distance, colorbar_distance, atol=1e-07).all()

    @pytest.mark.parametrize('cmap', [None, 'Greys'])
    def test_scatter_with_c_column_name_with_colors(self, cmap: str) -> None:
        df = DataFrame([[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]], columns=['length', 'width'])
        df['species'] = ['r', 'r', 'g', 'g', 'b']
        if cmap is not None:
            with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
                ax = df.plot.scatter(x=0, y=1, cmap=cmap, c='species')
        else:
            ax = df.plot.scatter(x=0, y=1, c='species', cmap=cmap)
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 3
        assert (np.unique(ax.collections[0].get_facecolor(), axis=0) == np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])).all()
        assert ax.collections[0].colorbar is None

    def test_scatter_with_c_column_name_without_colors(self) -> None:
        colors = ['NY', 'MD', 'MA', 'CA']
        color_count = 4
        df = DataFrame({'dataX': range(100), 'dataY': range(100), 'color': (colors[i % len(colors)] for i in range(100))})
        ax = df.plot.scatter('dataX', 'dataY', c='color')
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count
        colors = ['r', 'g', 'not-a-color']
        color_count = 3
        df = DataFrame({'dataX': range(100), 'dataY': range(100), 'color': (colors[i % len(colors)] for i in range(100))})
        ax = df.plot.scatter('dataX', 'dataY', c='color')
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count

    def test_scatter_colors(self) -> None:
        df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
        with pytest.raises(TypeError, match='Specify exactly one of `c` and `color`'):
            df.plot.scatter(x='a', y='b', c='c', color='green')

    def test_scatter_colors_not_raising_warnings(self) -> None:
        df = DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
        with tm.assert_produces_warning(None):
            ax = df.plot.scatter(x='x', y='y', c='b')
            assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 1
            assert (np.unique(ax.collections[0].get_facecolor(), axis=0) == np.array([[0.0, 0.0, 1.0, 1.0]])).all()

    def test_scatter_colors_default(self) -> None:
        df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)
        ax = df.plot.scatter(x='a', y='b', c='c')
        tm.assert_numpy_array_equal(ax.collections[0].get_facecolor()[0], np.array(mpl.colors.ColorConverter.to_rgba(default_colors[0])))

    def test_scatter_colors_white(self) -> None:
        df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
        ax = df.plot.scatter(x='a', y='b', color='white')
        tm.assert_numpy_array_equal(ax.collections[0].get_facecolor()[0], np.array([1, 1, 1, 1], dtype=np.float64))

    def test_scatter_colorbar_different_cmap(self) -> None:
        df = DataFrame({'x': [1, 2, 3], 'y': [1, 3, 2], 'c': [1, 2, 3]})
        df['x2'] = df['x'] + 1
        _, ax = plt.subplots()
        df.plot('x', 'y', c='c', kind='scatter', cmap='cividis', ax=ax)
        df.plot('x2', 'y', c='c', kind='scatter', cmap='magma', ax=ax)
        assert ax.collections[0].cmap.name == 'cividis'
        assert ax.collections[1].cmap.name == 'magma'

    def test_line_colors(self) -> None:
        custom_colors = 'rgcby'
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)
        plt.close('all')
        ax2 = df.plot(color=custom_colors)
        lines2 = ax2.get_lines()
        for l1, l2 in zip(ax.get_lines(), lines2):
            assert l1.get_color() == l2.get_color()

    @pytest.mark.parametrize('colormap', ['jet', cm.jet])
    def test_line_colors_cmap(self, colormap: str) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    def test_line_colors_single_col(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot(color='DodgerBlue')
        _check_colors(ax.lines, linecolors=['DodgerBlue'])

    def test_line_colors_single_color(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(color='red')
        _check_colors(ax.get_lines(), linecolors=['red'] * 5)

    def test_line_colors_hex(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = ['#FF0000', '#0000FF', '#FFFF00', '#000000', '#FFFFFF']
        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    def test_dont_modify_colors(self) -> None:
        colors = ['r', 'g', 'b']
        DataFrame(np.random.default_rng(2).random((10, 2))).plot(color=colors)
        assert len(colors) == 3

    def test_line_colors_and_styles_subplots(self) -> None:
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize('color', ['k', 'green'])
    def test_line_colors_and_styles_subplots_single_color_str(self, color: str) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(subplots=True, color=color)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[color])

    @pytest.mark.parametrize('color', ['rgcby', list('rgcby')])
    def test_line_colors_and_styles_subplots_custom_colors(self, color: str) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(color=color, subplots=True)
        for ax, c in zip(axes, list(color)):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_colormap_hex(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = ['#FF0000', '#0000FF', '#FFFF00', '#000000', '#FFFFFF']
        axes = df.plot(color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize('cmap', ['jet', cm.jet])
    def test_line_colors_and_styles_subplots_colormap_subplot(self, cmap: str) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        axes = df.plot(colormap=cmap, subplots=True)
        for ax, c in zip(axes, rgba_colors):
