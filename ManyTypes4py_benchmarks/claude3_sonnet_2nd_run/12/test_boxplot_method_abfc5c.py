"""Test cases for .boxplot method"""
from __future__ import annotations
import itertools
import string
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series, date_range, plotting, timedelta_range
import pandas._testing as tm
from pandas.tests.plotting.common import _check_axes_shape, _check_box_return_type, _check_plot_works, _check_ticks_props, _check_visible
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
mpl = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')

def _check_ax_limits(col: Series, ax: mpl.axes.Axes) -> None:
    y_min, y_max = ax.get_ylim()
    assert y_min <= col.min()
    assert y_max >= col.max()

if Version(mpl.__version__) < Version('3.10'):
    verts: List[Dict[str, Union[bool, str]]] = [{'vert': False}, {'vert': True}]
else:
    verts = [{'orientation': 'horizontal'}, {'orientation': 'vertical'}]

@pytest.fixture(params=verts)
def vert(request: pytest.FixtureRequest) -> Dict[str, Union[bool, str]]:
    return request.param

class TestDataFramePlots:

    def test_stacked_boxplot_set_axis(self) -> None:
        n = 30
        df = DataFrame({'Clinical': np.random.default_rng(2).choice([0, 1, 2, 3], n), 'Confirmed': np.random.default_rng(2).choice([0, 1, 2, 3], n), 'Discarded': np.random.default_rng(2).choice([0, 1, 2, 3], n)}, index=np.arange(0, n))
        ax = df.plot(kind='bar', stacked=True)
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == df.index.to_list()
        ax.set_xticks(np.arange(0, n, 10))
        plt.draw()
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == list(np.arange(0, n, 10))

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs, warn', [[{'return_type': 'dict'}, None], [{'column': ['one', 'two']}, None], [{'column': ['one', 'two'], 'by': 'indic'}, UserWarning], [{'column': ['one'], 'by': ['indic', 'indic2']}, None], [{'by': 'indic'}, UserWarning], [{'by': ['indic', 'indic2']}, UserWarning], [{'notch': 1}, None], [{'by': 'indic', 'notch': 1}, UserWarning]])
    def test_boxplot_legacy1(self, kwargs: Dict[str, Any], warn: Optional[type]) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
        df['indic'] = ['foo', 'bar'] * 3
        df['indic2'] = ['foo', 'bar', 'foo'] * 2
        with tm.assert_produces_warning(warn, check_stacklevel=False):
            _check_plot_works(df.boxplot, **kwargs)

    def test_boxplot_legacy1_series(self) -> None:
        ser = Series(np.random.default_rng(2).standard_normal(6))
        _check_plot_works(plotting._core.boxplot, data=ser, return_type='dict')

    def test_boxplot_legacy2(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.boxplot, by='X')

    def test_boxplot_legacy2_with_ax(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        _, ax = mpl.pyplot.subplots()
        axes = df.boxplot('Col1', by='X', ax=ax)
        ax_axes = ax.axes
        assert ax_axes is axes

    def test_boxplot_legacy2_with_ax_return_type(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        fig, ax = mpl.pyplot.subplots()
        axes = df.groupby('Y').boxplot(ax=ax, return_type='axes')
        ax_axes = ax.axes
        assert ax_axes is axes['A']

    def test_boxplot_legacy2_with_multi_col(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        fig, ax = mpl.pyplot.subplots()
        msg = 'the figure containing the passed axes is being cleared'
        with tm.assert_produces_warning(UserWarning, match=msg):
            axes = df.boxplot(column=['Col1', 'Col2'], by='X', ax=ax, return_type='axes')
        assert axes['Col1'].get_figure() is fig

    def test_boxplot_legacy2_by_none(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        _, ax = mpl.pyplot.subplots()
        d = df.boxplot(ax=ax, return_type='dict')
        lines = list(itertools.chain.from_iterable(d.values()))
        assert len(ax.get_lines()) == len(lines)

    def test_boxplot_return_type_none(self, hist_df: DataFrame) -> None:
        result = hist_df.boxplot()
        assert isinstance(result, mpl.pyplot.Axes)

    def test_boxplot_return_type_legacy(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
        msg = "return_type must be {'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(return_type='NOT_A_TYPE')
        result = df.boxplot()
        _check_box_return_type(result, 'axes')

    @pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
    def test_boxplot_return_type_legacy_return_type(self, return_type: str) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type=return_type)
        _check_box_return_type(result, return_type)

    def test_boxplot_axis_limits(self, hist_df: DataFrame) -> None:
        df = hist_df.copy()
        df['age'] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        height_ax, weight_ax = df.boxplot(['height', 'weight'], by='category')
        _check_ax_limits(df['height'], height_ax)
        _check_ax_limits(df['weight'], weight_ax)
        assert weight_ax._sharey == height_ax

    def test_boxplot_axis_limits_two_rows(self, hist_df: DataFrame) -> None:
        df = hist_df.copy()
        df['age'] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        p = df.boxplot(['height', 'weight', 'age'], by='category')
        height_ax, weight_ax, age_ax = (p[0, 0], p[0, 1], p[1, 0])
        dummy_ax = p[1, 1]
        _check_ax_limits(df['height'], height_ax)
        _check_ax_limits(df['weight'], weight_ax)
        _check_ax_limits(df['age'], age_ax)
        assert weight_ax._sharey == height_ax
        assert age_ax._sharey == height_ax
        assert dummy_ax._sharey is None

    def test_boxplot_empty_column(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)))
        df.loc[:, 0] = np.nan
        _check_plot_works(df.boxplot, return_type='axes')

    def test_figsize(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 5)), columns=['A', 'B', 'C', 'D', 'E'])
        result = df.boxplot(return_type='axes', figsize=(12, 8))
        assert result.figure.bbox_inches.width == 12
        assert result.figure.bbox_inches.height == 8

    def test_fontsize(self) -> None:
        df = DataFrame({'a': [1, 2, 3, 4, 5, 6]})
        _check_ticks_props(df.boxplot('a', fontsize=16), xlabelsize=16, ylabelsize=16)

    def test_boxplot_numeric_data(self) -> None:
        df = DataFrame({'a': date_range('2012-01-01', periods=10), 'b': np.random.default_rng(2).standard_normal(10), 'c': np.random.default_rng(2).standard_normal(10) + 2, 'd': date_range('2012-01-01', periods=10).astype(str), 'e': date_range('2012-01-01', periods=10, tz='UTC'), 'f': timedelta_range('1 days', periods=10)})
        ax = df.plot(kind='box')
        assert [x.get_text() for x in ax.get_xticklabels()] == ['b', 'c']

    @pytest.mark.parametrize('colors_kwd, expected', [({'boxes': 'r', 'whiskers': 'b', 'medians': 'g', 'caps': 'c'}, {'boxes': 'r', 'whiskers': 'b', 'medians': 'g', 'caps': 'c'}), ({'boxes': 'r'}, {'boxes': 'r'}), ('r', {'boxes': 'r', 'whiskers': 'r', 'medians': 'r', 'caps': 'r'})])
    def test_color_kwd(self, colors_kwd: Union[str, Dict[str, str]], expected: Dict[str, str]) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        result = df.boxplot(color=colors_kwd, return_type='dict')
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize('scheme,expected', [('dark_background', {'boxes': '#8dd3c7', 'whiskers': '#8dd3c7', 'medians': '#bfbbd9', 'caps': '#8dd3c7'}), ('default', {'boxes': '#1f77b4', 'whiskers': '#1f77b4', 'medians': '#2ca02c', 'caps': '#1f77b4'})])
    def test_colors_in_theme(self, scheme: str, expected: Dict[str, str]) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        plt.style.use(scheme)
        result = df.plot.box(return_type='dict')
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize('dict_colors, msg', [({'boxes': 'r', 'invalid_key': 'r'}, "invalid key 'invalid_key'")])
    def test_color_kwd_errors(self, dict_colors: Dict[str, str], msg: str) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        with pytest.raises(ValueError, match=msg):
            df.boxplot(color=dict_colors, return_type='dict')

    @pytest.mark.parametrize('props, expected', [('boxprops', 'boxes'), ('whiskerprops', 'whiskers'), ('capprops', 'caps'), ('medianprops', 'medians')])
    def test_specified_props_kwd(self, props: str, expected: str) -> None:
        df = DataFrame({k: np.random.default_rng(2).random(10) for k in 'ABC'})
        kwd = {props: {'color': 'C1'}}
        result = df.boxplot(return_type='dict', **kwd)
        assert result[expected][0].get_color() == 'C1'

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_plot_xlabel_ylabel(self, vert: Dict[str, Union[bool, str]]) -> None:
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        xlabel, ylabel = ('x', 'y')
        ax = df.plot(kind='box', xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_plot_box(self, vert: Dict[str, Union[bool, str]]) -> None:
        rng = np.random.default_rng(2)
        df1 = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list('ABCD'))
        df2 = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list('ABCD'))
        xlabel, ylabel = ('x', 'y')
        _, axs = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
        df1.plot.box(ax=axs[0], xlabel=xlabel, ylabel=ylabel, **vert)
        df2.plot.box(ax=axs[1], xlabel=xlabel, ylabel=ylabel, **vert)
        for ax in axs:
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_boxplot_xlabel_ylabel(self, vert: Dict[str, Union[bool, str]]) -> None:
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        xlabel, ylabel = ('x', 'y')
        ax = df.boxplot(xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_boxplot_group_xlabel_ylabel(self, vert: Dict[str, Union[bool, str]]) -> None:
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        xlabel, ylabel = ('x', 'y')
        ax = df.boxplot(by='group', xlabel=xlabel, ylabel=ylabel, **vert)
        for subplot in ax:
            assert subplot.get_xlabel() == xlabel
            assert subplot.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_boxplot_group_no_xlabel_ylabel(self, vert: Dict[str, Union[bool, str]], request: pytest.FixtureRequest) -> None:
        if Version(mpl.__version__) >= Version('3.10') and vert == {'orientation': 'horizontal'}:
            request.applymarker(pytest.mark.xfail(reason=f'{vert} fails starting with matplotlib 3.10'))
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        ax = df.boxplot(by='group', **vert)
        for subplot in ax:
            target_label = subplot.get_xlabel() if vert == {'vert': True} or vert == {'orientation': 'vertical'} else subplot.get_ylabel()
            assert target_label == pprint_thing(['group'])

class TestDataFrameGroupByPlots:

    def test_boxplot_legacy1(self, hist_df: DataFrame) -> None:
        grouped = hist_df.groupby(by='gender')
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(grouped.boxplot, return_type='axes')
        _check_axes_shape(list(axes.values), axes_num=2, layout=(1, 2))

    def test_boxplot_legacy1_return_type(self, hist_df: DataFrame) -> None:
        grouped = hist_df.groupby(by='gender')
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type='axes')
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_boxplot_legacy2(self) -> None:
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=MultiIndex.from_tuples(tuples))
        grouped = df.groupby(level=1)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(grouped.boxplot, return_type='axes')
        _check_axes_shape(list(axes.values), axes_num=10, layout=(4, 3))

    @pytest.mark.slow
    def test_boxplot_legacy2_return_type(self) -> None:
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=MultiIndex.from_tuples(tuples))
        grouped = df.groupby(level=1)
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type='axes')
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_grouped_plot_fignums(self) -> None:
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(['male', 'female'], size=n)
        df = DataFrame({'height': height, 'weight': weight, 'gender': gender})
        gb = df.groupby('gender')
        res = gb.plot()
        assert len(mpl.pyplot.get_fignums()) == 2
        assert len(res) == 2
        plt.close('all')
        res = gb.boxplot(return_type='axes')
        assert len(mpl.pyplot.get_fignums()) == 1
        assert len(res) == 2

    def test_grouped_plot_fignums_excluded_col(self) -> None:
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(['male', 'female'], size=n)
        df = DataFrame({'height': height, 'weight': weight, 'gender': gender})
        df.groupby('gender').hist()

    @pytest.mark.slow
    def test_grouped_box_return_type(self, hist_df: DataFrame) -> None:
        df = hist_df
        result = df.boxplot(by='gender')
        assert isinstance(result, np.ndarray)
        _check_box_return_type(result, None, expected_keys=['height', 'weight', 'category'])

    @pytest.mark.slow
    def test_grouped_box_return_type_groupby(self, hist_df: DataFrame) -> None:
        df = hist_df
        result = df.groupby('gender').boxplot(return_type='dict')
        _check_box_return_type(result, 'dict', expected_keys=['Male', 'Female'])

    @pytest.mark.slow
    @pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
    def test_grouped_box_return_type_arg(self, hist_df: DataFrame, return_type: str) -> None:
        df = hist_df
        returned = df.groupby('classroom').boxplot(return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=['A', 'B', 'C'])
        returned = df.boxplot(by='classroom', return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=['height', 'weight', 'category'])

    @pytest.mark.slow
    @pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
    def test_grouped_box_return_type_arg_duplcate_cats(self, return_type: str) -> None:
        columns2 = 'X B C D A'.split()
        df2 = DataFrame(np.random.default_rng(2).standard_normal((6, 5)), columns=columns2)
        categories2 = 'A B'.split()
        df2['category'] = categories2 * 3
        returned = df2.groupby('category').boxplot(return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=categories2)
        returned = df2.boxplot(by='category', return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=columns2)

    @pytest.mark.slow
    def test_grouped_box_layout_too_small(self, hist_df: DataFrame) -> None:
        df = hist_df
        msg = 'Layout of 1x1 must be larger than required size 2'
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=['weight', 'height'], by=df.gender, layout=(1, 1))

    @pytest.mark.slow
    def test_grouped_box_layout_needs_by(self, hist_df: DataFrame) -> None:
        df = hist_df
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=['height', 'weight', 'category'], layout=(2, 1), return_type='dict')

    @pytest.mark.slow
    def test_grouped_box_layout_positive_layout(self, hist_df: DataFrame) -> None:
        df = hist_df
        msg = 'At least one dimension of layout must be positive'
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=['weight', 'height'], by=df.gender, layout=(-1, -1))

    @pytest.mark.slow
    @pytest.mark.parametrize('gb_key, axes_num, rows', [['gender', 2, 1], ['category', 4, 2], ['classroom', 3, 2]])
    def test_grouped_box_layout_positive_layout_axes(self, hist_df: DataFrame, gb_key: str, axes_num: int, rows: int) -> None:
        df = hist_df
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.groupby(gb_key).boxplot, column='height', return_type='dict')
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=axes_num, layout=(rows, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize('col, visible', [['height', False], ['weight', True], ['category', True]])
    def test_grouped_box_layout_visible(self, hist_df: DataFrame, col: str, visible: bool) -> None:
        df = hist_df
        axes = df.boxplot(column=['height', 'weight', 'category'], by='gender', return_type='axes')
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))
        ax = axes[col]
        _check_visible(ax.get_xticklabels(), visible=visible)
        _check_visible([ax.xaxis.get_label()], visible=visible)

    @pytest.mark.slow
    def test_grouped_box_layout_shape(self, hist_df: DataFrame) -> None:
        df = hist_df
        df.groupby('classroom').boxplot(column=['height', 'weight', 'category'], return_type='dict')
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize('cols', [2, -1])
    def test_grouped_box_layout_works(self, hist_df: DataFrame, cols: int) -> None:
        df = hist_df
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.groupby('category').boxplot, column='height', layout=(3, cols), return_type='dict')
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(3, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize('rows, res', [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_rows(self, hist_df: DataFrame, rows: int, res: int) -> None:
        df = hist_df
        df.boxplot(column=['height', 'weight', 'category'], by='gender', layout=(rows, 1))
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(res, 1))

    @pytest.mark.slow
    @pytest.mark.parametrize('cols, res', [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_cols_groupby(self, hist_df: DataFrame, cols: int, res: int) -> None:
        df = hist_df
        df.groupby('classroom').boxplot(column=['height', 'weight', 'category'], layout=(1, cols), return_type='dict')
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(1, res))

    @pytest.mark.slow
    def test_grouped_box_multiple_axes(self, hist_df: DataFrame) -> None:
        df = hist_df
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            _, axes = mpl.pyplot.subplots(2, 2)
            df.groupby('category').boxplot(column='height', return_type='axes', ax=axes)
            _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(2, 2))

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_on_fig(self, hist_df: DataFrame) -> None:
        df = hist_df
        fig, axes = mpl.pyplot.subplots(2, 3)
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            returned = df.boxplot(column=['height', 'weight', 'category'], by='gender', return_type='axes', ax=axes[0])
        returned = np.array(list(returned.values()))
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[0])
        assert returned[0].figure is fig
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            returned = df.groupby('classroom').boxplot(column=['height', 'weight', 'category'], return_type='axes', ax=axes[1])
        returned = np.array(list(returned.values()))
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[1])
        assert returned[0].figure is fig

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_ax_error(self, hist_df: DataFrame) -> None:
        df = hist_df
        msg = 'The number of passed axes must be 3, the same as the output plot'
        _, axes = mpl.pyplot.subplots(2, 3)
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
                axes = df.groupby('classroom').boxplot(ax=axes)

    def test_fontsize(self) -> None:
        df = DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [0, 0, 0, 1, 1, 1]})
        _check_ticks_props(df.boxplot('a', by='b', fontsize=16), xlabelsize=16, ylabelsize=16)

    @pytest.mark.parametrize('col, expected_xticklabel', [('v', ['(a, v)', '(b, v)', '(c, v)', '(d, v)', '(e, v)']), (['v'], ['(a, v)', '(b, v)', '(c, v)', '(d, v)', '(e, v)']), ('v1', ['(a, v1)', '(b, v1)', '(c, v1)', '(d, v1)', '(e, v1)']), (['v', 'v1'], ['(a, v)', '(a, v1)', '(b, v)', '(b, v1)', '(c, v)', '(c, v1)', '(d, v)', '(d, v1)', '(e, v)', '(e, v1)']), (None, ['(a, v)', '(a, v1)', '(b, v)', '(b, v1)', '(c, v)', '(c, v1)', '(d, v)', '(d, v1)', '(e, v)', '(e, v1)'])])
    def test_groupby_boxplot_subplots_false(self, col: Optional[Union[str, List[str]]], expected_xticklabel: List[str]) -> None:
        df = DataFrame({'cat': np.random.default_rng(2).choice(list('abcde'), 100), 'v': np.random.default_rng(2).random(100), 'v1': np.random.default_rng(2).random(100)})
        grouped = df.groupby('cat')
        axes = _check_plot_works(grouped.boxplot, subplots=False, column=col, return_type='axes')
        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

    def test_groupby_boxplot_object(self, hist_df: DataFrame) -> None:
        df = hist_df.astype('object')
        grouped = df.groupby('gender')
        msg = 'boxplot method requires numerical columns, nothing to plot'
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(grouped.boxplot, subplots=False)

    def test_boxplot_multiindex_column(self) -> None:
        arrays: List[List[str]] = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 8)), index=['A', 'B', 'C'], columns=index)
        col = [('bar', 'one'), ('bar', 'two')]
        axes = _check_plot_works(df.boxplot, column=col, return_type='axes')
        expected_xticklabel = ['(bar, one)', '(bar, two)']
        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

    @pytest.mark.parametrize('group', ['X', ['X', 'Y']])
    def test_boxplot_multi_groupby_groups(self, group: Union[str, List[str]]) -> None:
        rows = 20
        df = DataFrame(np.random.default_rng(12).normal(size=(rows, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(np.repeat(['A', 'B'], int(rows / 2)))
        df['Y'] = Series(np.tile(['C', 'D'], int(rows / 2)))
        grouped = df.groupby(group)
        _check_plot_works(df.boxplot, by=group, default_axes=True)
        _check_plot_works(df.plot.box, by=group, default_axes=True)
        _check_plot_works(grouped.boxplot, default_axes=True)
