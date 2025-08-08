import re
from typing import List, Tuple
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import _check_axes_shape, _check_plot_works, get_x_axis, get_y_axis
pytest.importorskip('matplotlib')

@pytest.fixture
def hist_df() -> DataFrame:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((30, 2)), columns=['A', 'B'])
    df['C'] = np.random.default_rng(2).choice(['a', 'b', 'c'], 30)
    df['D'] = np.random.default_rng(2).choice(['a', 'b', 'c'], 30)
    return df

class TestHistWithBy:

    @pytest.mark.slow
    @pytest.mark.parametrize('by, column, titles, legends', [('C', 'A', List[str], List[List[str]]), ('C', List[str], List[str], List[List[str]]), ('C', None, List[str], List[List[str]]), (List[str], 'A', List[str], List[List[str]]), (List[str], List[str], List[str], List[List[str]]), (List[str], None, List[str], List[List[str]])])
    def test_hist_plot_by_argument(self, by, column, titles, legends, hist_df) -> None:
        axes = _check_plot_works(hist_df.plot.hist, column=column, by=by, default_axes=True)
        result_titles = [ax.get_title() for ax in axes]
        result_legends = [[legend.get_text() for legend in ax.get_legend().texts] for ax in axes]
        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize('by, column, titles, legends', [(int, 'A', List[str], List[List[str]]), (int, None, List[str], List[List[str]]), (List[int], 'A', List[str], List[List[str]])])
    def test_hist_plot_by_0(self, by, column, titles, legends, hist_df) -> None:
        df = hist_df.copy()
        df = df.rename(columns={'C': 0})
        axes = _check_plot_works(df.plot.hist, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_legends = [[legend.get_text() for legend in ax.get_legend().texts] for ax in axes]
        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize('by, column', [(List, List[str]), (Tuple, None), (Tuple, List[str])])
    def test_hist_plot_empty_list_string_tuple_by(self, by, column, hist_df) -> None:
        msg = 'No group keys passed'
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.hist, default_axes=True, column=column, by=by)

    @pytest.mark.slow
    @pytest.mark.parametrize('by, column, layout, axes_num', [(List[str], 'A', Tuple[int, int], int), ('C', 'A', Tuple[int, int], int), (List[str], List[str], Tuple[int, int], int), ('C', None, Tuple[int, int], int), ('C', List[str], Tuple[int, int], int), (List[str], 'A', Tuple[int, int], int), (List[str], 'A', Tuple[int, int], int), (List[str], List[str], Tuple[int, int], int), (List[str], List[str], Tuple[int, int], int), (List[str], None, Tuple[int, int], int), (List[str], List[str], Tuple[int, int], int)])
    def test_hist_plot_layout_with_by(self, by, column, layout, axes_num, hist_df) -> None:
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(hist_df.plot.hist, column=column, by=by, layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize('msg, by, layout', [('larger than required size', List[str], Tuple[int, int]), (re.escape('Layout must be a tuple of (rows, columns)'), 'C', Tuple[int]), ('At least one dimension of layout must be positive', 'C', Tuple[int])])
    def test_hist_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df) -> None:
        with pytest.raises(ValueError, match=msg):
            hist_df.plot.hist(column=['A', 'B'], by=by, layout=layout)

    @pytest.mark.slow
    def test_axis_share_x_with_by(self, hist_df) -> None:
        ax1, ax2, ax3 = hist_df.plot.hist(column='A', by='C', sharex=True)
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)
        assert get_x_axis(ax3).joined(ax1, ax3)
        assert get_x_axis(ax3).joined(ax2, ax3)
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)
        assert not get_y_axis(ax3).joined(ax1, ax3)
        assert not get_y_axis(ax3).joined(ax2, ax3)

    @pytest.mark.slow
    def test_axis_share_y_with_by(self, hist_df) -> None:
        ax1, ax2, ax3 = hist_df.plot.hist(column='A', by='C', sharey=True)
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)
        assert get_y_axis(ax3).joined(ax1, ax3)
        assert get_y_axis(ax3).joined(ax2, ax3)
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)
        assert not get_x_axis(ax3).joined(ax1, ax3)
        assert not get_x_axis(ax3).joined(ax2, ax3)

    @pytest.mark.parametrize('figsize', [Tuple[int, int], Tuple[int, int]])
    def test_figure_shape_hist_with_by(self, figsize, hist_df) -> None:
        axes = hist_df.plot.hist(column='A', by='C', figsize=figsize)
        _check_axes_shape(axes, axes_num=3, figsize=figsize)

class TestBoxWithBy:

    @pytest.mark.parametrize('by, column, titles, xticklabels', [('C', 'A', List[str], List[List[str]]), (List[str], 'A', List[str], List[List[str]]), ('C', List[str], List[str], List[List[str]]), (List[str], List[str], List[str], List[List[str]]), (List[str], None, List[str], List[List[str]])])
    def test_box_plot_by_argument(self, by, column, titles, xticklabels, hist_df) -> None:
        axes = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [[label.get_text() for label in ax.get_xticklabels()] for ax in axes]
        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize('by, column, titles, xticklabels', [(int, 'A', List[str], List[List[str]]), (List[int], 'A', List[str], List[List[str]]), (int, None, List[str], List[List[str]])])
    def test_box_plot_by_0(self, by, column, titles, xticklabels, hist_df) -> None:
        df = hist_df.copy()
        df = df.rename(columns={'C': 0})
        axes = _check_plot_works(df.plot.box, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [[label.get_text() for label in ax.get_xticklabels()] for ax in axes]
        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize('by, column', [(List, List[str]), (Tuple, 'A'), (List, None), (Tuple, List[str])])
    def test_box_plot_with_none_empty_list_by(self, by, column, hist_df) -> None:
        msg = 'No group keys passed'
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)

    @pytest.mark.slow
    @pytest.mark.parametrize('by, column, layout, axes_num', [(List[str], 'A', Tuple[int, int], int), ('C', 'A', Tuple[int, int], int), ('C', None, Tuple[int, int], int), ('C', List[str], Tuple[int, int], int), (List[str], 'A', Tuple[int, int], int), (List[str], None, Tuple[int, int], int)])
    def test_box_plot_layout_with_by(self, by, column, layout, axes_num, hist_df) -> None:
        axes = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by, layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize('msg, by, layout', [('larger than required size', List[str], Tuple[int, int]), (re.escape('Layout must be a tuple of (rows, columns)'), 'C', Tuple[int]), ('At least one dimension of layout must be positive', 'C', Tuple[int])])
    def test_box_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df) -> None:
        with pytest.raises(ValueError, match=msg):
            hist_df.plot.box(column=['A', 'B'], by=by, layout=layout)

    @pytest.mark.parametrize('figsize', [Tuple[int, int], Tuple[int, int]])
    def test_figure_shape_hist_with_by(self, figsize, hist_df) -> None:
        axes = hist_df.plot.box(column='A', by='C', figsize=figsize)
        _check_axes_shape(axes, axes_num=1, figsize=figsize)
