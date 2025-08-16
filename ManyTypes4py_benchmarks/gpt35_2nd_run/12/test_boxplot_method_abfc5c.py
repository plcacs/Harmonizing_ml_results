from __future__ import annotations
import itertools
import string
import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series, date_range, plotting, timedelta_range
import pandas._testing as tm
from pandas.tests.plotting.common import _check_axes_shape, _check_box_return_type, _check_plot_works, _check_ticks_props, _check_visible
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
mpl = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')

def _check_ax_limits(col: Series, ax: mpl.axes.Axes):
    y_min, y_max = ax.get_ylim()
    assert y_min <= col.min()
    assert y_max >= col.max()

if Version(mpl.__version__) < Version('3.10'):
    verts: list[dict[str, bool]] = [{'vert': False}, {'vert': True}]
else:
    verts: list[dict[str, str]] = [{'orientation': 'horizontal'}, {'orientation': 'vertical'}]

@pytest.fixture(params=verts)
def vert(request) -> dict[str, bool]:
    return request.param

class TestDataFramePlots:

    def test_stacked_boxplot_set_axis(self):
        n: int = 30
        df: DataFrame = DataFrame({'Clinical': np.random.default_rng(2).choice([0, 1, 2, 3], n), 'Confirmed': np.random.default_rng(2).choice([0, 1, 2, 3], n), 'Discarded': np.random.default_rng(2).choice([0, 1, 2, 3], n)}, index=np.arange(0, n))
        ax: mpl.axes.Axes = df.plot(kind='bar', stacked=True)
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == df.index.to_list()
        ax.set_xticks(np.arange(0, n, 10))
        plt.draw()
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == list(np.arange(0, n, 10))

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs, warn', [[{'return_type': 'dict'}, None], [{'column': ['one', 'two']}, None], [{'column': ['one', 'two'], 'by': 'indic'}, UserWarning], [{'column': ['one'], 'by': ['indic', 'indic2']}, None], [{'by': 'indic'}, UserWarning], [{'by': ['indic', 'indic2']}, UserWarning], [{'notch': 1}, None], [{'by': 'indic', 'notch': 1}, UserWarning]])
    def test_boxplot_legacy1(self, kwargs: dict, warn: UserWarning):
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
        df['indic'] = ['foo', 'bar'] * 3
        df['indic2'] = ['foo', 'bar', 'foo'] * 2
        with tm.assert_produces_warning(warn, check_stacklevel=False):
            _check_plot_works(df.boxplot, **kwargs)

    def test_boxplot_legacy1_series(self):
        ser: Series = Series(np.random.default_rng(2).standard_normal(6))
        _check_plot_works(plotting._core.boxplot, data=ser, return_type='dict')

    def test_boxplot_legacy2(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.boxplot, by='X')

    def test_boxplot_legacy2_with_ax(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        _, ax = mpl.pyplot.subplots()
        axes = df.boxplot('Col1', by='X', ax=ax)
        ax_axes = ax.axes
        assert ax_axes is axes

    def test_boxplot_legacy2_with_ax_return_type(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        fig, ax = mpl.pyplot.subplots()
        axes = df.groupby('Y').boxplot(ax=ax, return_type='axes')
        ax_axes = ax.axes
        assert ax_axes is axes['A']

    def test_boxplot_legacy2_with_multi_col(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        fig, ax = mpl.pyplot.subplots()
        msg: str = 'the figure containing the passed axes is being cleared'
        with tm.assert_produces_warning(UserWarning, match=msg):
            axes = df.boxplot(column=['Col1', 'Col2'], by='X', ax=ax, return_type='axes')
        assert axes['Col1'].get_figure() is fig

    def test_boxplot_legacy2_by_none(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
        df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        df['Y'] = Series(['A'] * 10)
        _, ax = mpl.pyplot.subplots()
        d: dict = df.boxplot(ax=ax, return_type='dict')
        lines: list = list(itertools.chain.from_iterable(d.values()))
        assert len(ax.get_lines()) == len(lines)

    def test_boxplot_return_type_none(self, hist_df: DataFrame):
        result: mpl.axes.Axes = hist_df.boxplot()
        assert isinstance(result, mpl.axes.Axes)

    def test_boxplot_return_type_legacy(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
        msg: str = "return_type must be {'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(return_type='NOT_A_TYPE')
        result: mpl.axes.Axes = df.boxplot()
        _check_box_return_type(result, 'axes')

    @pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
    def test_boxplot_return_type_legacy_return_type(self, return_type: str):
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type=return_type)
        _check_box_return_type(result, return_type)

    def test_boxplot_axis_limits(self, hist_df: DataFrame):
        df: DataFrame = hist_df.copy()
        df['age'] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        height_ax, weight_ax = df.boxplot(['height', 'weight'], by='category')
        _check_ax_limits(df['height'], height_ax)
        _check_ax_limits(df['weight'], weight_ax)
        assert weight_ax._sharey == height_ax

    def test_boxplot_axis_limits_two_rows(self, hist_df: DataFrame):
        df: DataFrame = hist_df.copy()
        df['age'] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        p: np.ndarray = df.boxplot(['height', 'weight', 'age'], by='category')
        height_ax, weight_ax, age_ax = (p[0, 0], p[0, 1], p[1, 0])
        dummy_ax = p[1, 1]
        _check_ax_limits(df['height'], height_ax)
        _check_ax_limits(df['weight'], weight_ax)
        _check_ax_limits(df['age'], age_ax)
        assert weight_ax._sharey == height_ax
        assert age_ax._sharey == height_ax
        assert dummy_ax._sharey is None

    def test_boxplot_empty_column(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((20, 4)))
        df.loc[:, 0] = np.nan
        _check_plot_works(df.boxplot, return_type='axes')

    def test_figsize(self):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 5)), columns=['A', 'B', 'C', 'D', 'E'])
        result: mpl.axes.Axes = df.boxplot(return_type='axes', figsize=(12, 8))
        assert result.figure.bbox_inches.width == 12
        assert result.figure.bbox_inches.height == 8

    def test_fontsize(self):
        df: DataFrame = DataFrame({'a': [1, 2, 3, 4, 5, 6]})
        _check_ticks_props(df.boxplot('a', fontsize=16), xlabelsize=16, ylabelsize=16)

    def test_boxplot_numeric_data(self):
        df: DataFrame = DataFrame({'a': date_range('2012-01-01', periods=10), 'b': np.random.default_rng(2).standard_normal(10), 'c': np.random.default_rng(2).standard_normal(10) + 2, 'd': date_range('2012-01-01', periods=10).astype(str), 'e': date_range('2012-01-01', periods=10, tz='UTC'), 'f': timedelta_range('1 days', periods=10)})
        ax: mpl.axes.Axes = df.plot(kind='box')
        assert [x.get_text() for x in ax.get_xticklabels()] == ['b', 'c']

    @pytest.mark.parametrize('colors_kwd, expected', [({'boxes': 'r', 'whiskers': 'b', 'medians': 'g', 'caps': 'c'}, {'boxes': 'r', 'whiskers': 'b', 'medians': 'g', 'caps': 'c'}), ({'boxes': 'r'}, {'boxes': 'r'}), ('r', {'boxes': 'r', 'whiskers': 'r', 'medians': 'r', 'caps': 'r'})])
    def test_color_kwd(self, colors_kwd: dict, expected: dict):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)))
        result: dict = df.boxplot(color=colors_kwd, return_type='dict')
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize('scheme,expected', [('dark_background', {'boxes': '#8dd3c7', 'whiskers': '#8dd3c7', 'medians': '#bfbbd9', 'caps': '#8dd3c7'}), ('default', {'boxes': '#1f77b4', 'whiskers': '#1f77b4', 'medians': '#2ca02c', 'caps': '#1f77b4'})])
    def test_colors_in_theme(self, scheme: str, expected: dict):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)))
        plt.style.use(scheme)
        result: dict = df.plot.box(return_type='dict')
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize('dict_colors, msg', [({'boxes': 'r', 'invalid_key': 'r'}, "invalid key 'invalid_key'")])
    def test_color_kwd_errors(self, dict_colors: dict, msg: str):
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)))
        with pytest.raises(ValueError, match=msg):
            df.boxplot(color=dict_colors, return_type='dict')

    @pytest.mark.parametrize('props, expected', [('boxprops', 'boxes'), ('whiskerprops', 'whiskers'), ('capprops', 'caps'), ('medianprops', 'medians')])
    def test_specified_props_kwd(self, props: str, expected: str):
        df: DataFrame = DataFrame({k: np.random.default_rng(2).random(10) for k in 'ABC'})
        kwd: dict = {props: {'color': 'C1'}}
        result: dict = df.boxplot(return_type='dict', **kwd)
        assert result[expected][0].get_color() == 'C1'

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_plot_xlabel_ylabel(self, vert: dict):
        df: DataFrame = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        xlabel, ylabel = ('x', 'y')
        ax: mpl.axes.Axes = df.plot(kind='box', xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_plot_box(self, vert: dict):
        rng: np.random.Generator = np.random.default_rng(2)
        df1: DataFrame = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list('ABCD'))
        df2: DataFrame = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list('ABCD'))
        xlabel, ylabel = ('x', 'y')
        _, axs = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
        df1.plot.box(ax=axs[0], xlabel=xlabel, ylabel=ylabel, **vert)
        df2.plot.box(ax=axs[1], xlabel=xlabel, ylabel=ylabel, **vert)
        for ax in axs:
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_boxplot_xlabel_ylabel(self, vert: dict):
        df: DataFrame = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        xlabel, ylabel = ('x', 'y')
        ax: mpl.axes.Axes = df.boxplot(xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_boxplot_group_xlabel_ylabel(self, vert: dict):
        df: DataFrame = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        xlabel, ylabel = ('x', 'y')
        ax: mpl.axes.Axes = df.boxplot(by='group', xlabel=xlabel, ylabel=ylabel, **vert)
        for subplot in ax:
            assert subplot.get_xlabel() == xlabel
            assert subplot.get_ylabel() == ylabel

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    def test_boxplot_group_no_xlabel_ylabel(self, vert: dict, request: pytest.FixtureRequest):
        if Version(mpl.__version__) >= Version('3.10') and vert == {'orientation': 'horizontal'}:
            request.applymarker(pytest.mark.xfail(reason=f'{vert} fails starting with matplotlib 3.10'))
        df: DataFrame = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
        ax: np.ndarray = df.boxplot(by='group', **vert)
        for subplot in ax:
            target_label = subplot.get_xlabel() if vert == {'vert': True} or vert == {'orientation': 'vertical'} else subplot.get_ylabel()
            assert target_label == pprint_thing(['group'])

class TestDataFrameGroupByPlots:

    def test_boxplot_legacy1(self, hist_df: DataFrame):
        grouped: DataFrame = hist_df.groupby(by='gender')
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes: dict = _check_plot_works(grouped.boxplot, return_type='axes')
        _check_axes_shape(list(axes.values), axes_num=2, layout=(1, 2))

    def test_boxplot_legacy1_return_type(self, hist_df: DataFrame):
        grouped: DataFrame = hist_df.groupby(by='gender')
        axes: np.ndarray = _check_plot_works(grouped.boxplot, subplots=False, return_type='axes')
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_boxplot_legacy2(self):
        tuples: list = zip(string.ascii_letters[:10], range(10))
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)), index=MultiIndex.from_tuples(tuples))
        grouped: DataFrame = df.groupby(level=1)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes: dict = _check_plot_works(grouped.boxplot, return_type='axes')
        _check_axes_shape(list(axes.values), axes_num=10, layout=(4, 3))

    @pytest.mark.slow
    def