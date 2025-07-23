"""Test cases for Series.plot"""
from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame, Series, date_range, period_range, plotting
import pandas._testing as tm
from pandas.tests.plotting.common import _check_ax_scales, _check_axes_shape, _check_colors, _check_grid_settings, _check_has_errorbars, _check_legend_labels, _check_plot_works, _check_text_labels, _check_ticks_props, _unpack_cycler, get_y_axis
from pandas.tseries.offsets import CustomBusinessDay
mpl = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')
from pandas.plotting._matplotlib.converter import DatetimeConverter
from pandas.plotting._matplotlib.style import get_standard_colors

@pytest.fixture
def ts() -> Series:
    return Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')

@pytest.fixture
def series() -> Series:
    return Series(range(10), dtype=np.float64, name='series', index=[f'i_{i}' for i in range(10)])

class TestSeriesPlots:

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [{'label': 'foo'}, {'use_index': False}])
    def test_plot(self, ts: Series, kwargs: dict) -> None:
        _check_plot_works(ts.plot, **kwargs)

    @pytest.mark.slow
    def test_plot_tick_props(self, ts: Series) -> None:
        axes = _check_plot_works(ts.plot, rot=0)
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.slow
    @pytest.mark.parametrize('scale, exp_scale', [[{'logy': True}, {'yaxis': 'log'}], [{'logx': True}, {'xaxis': 'log'}], [{'loglog': True}, {'xaxis': 'log', 'yaxis': 'log'}]])
    def test_plot_scales(self, ts: Series, scale: dict, exp_scale: dict) -> None:
        ax = _check_plot_works(ts.plot, style='.', **scale)
        _check_ax_scales(ax, **exp_scale)

    @pytest.mark.slow
    def test_plot_ts_bar(self, ts: Series) -> None:
        _check_plot_works(ts[:10].plot.bar)

    @pytest.mark.slow
    def test_plot_ts_area_stacked(self, ts: Series) -> None:
        _check_plot_works(ts.plot.area, stacked=False)

    def test_plot_iseries(self) -> None:
        ser = Series(range(5), period_range('2020-01-01', periods=5))
        _check_plot_works(ser.plot)

    @pytest.mark.parametrize('kind', ['line', 'bar', 'barh', pytest.param('kde', marks=td.skip_if_no('scipy')), 'hist', 'box'])
    def test_plot_series_kinds(self, series: Series, kind: str) -> None:
        _check_plot_works(series[:5].plot, kind=kind)

    def test_plot_series_barh(self, series: Series) -> None:
        _check_plot_works(series[:10].plot.barh)

    def test_plot_series_bar_ax(self) -> None:
        ax = _check_plot_works(Series(np.random.default_rng(2).standard_normal(10)).plot.bar, color='black')
        _check_colors([ax.patches[0]], facecolors=['black'])

    @pytest.mark.parametrize('kwargs', [{}, {'layout': (-1, 1)}, {'layout': (1, -1)}])
    def test_plot_6951(self, ts: Series, kwargs: dict) -> None:
        ax = _check_plot_works(ts.plot, subplots=True, **kwargs)
        _check_axes_shape(ax, axes_num=1, layout=(1, 1))

    def test_plot_figsize_and_title(self, series: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = series.plot(title='Test', figsize=(16, 8), ax=ax)
        _check_text_labels(ax.title, 'Test')
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))

    def test_dont_modify_rcParams(self) -> None:
        key = 'axes.prop_cycle'
        colors = mpl.pyplot.rcParams[key]
        _, ax = mpl.pyplot.subplots()
        Series([1, 2, 3]).plot(ax=ax)
        assert colors == mpl.pyplot.rcParams[key]

    @pytest.mark.parametrize('kwargs', [{}, {'secondary_y': True}])
    def test_ts_line_lim(self, ts: Series, kwargs: dict) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax, **kwargs)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]

    def test_ts_area_lim(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_area_lim_xcompat(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=30)

    def test_ts_tz_area_lim_xcompat(self, ts: Series) -> None:
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize('GMT').tz_convert('CET')
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_tz_area_lim_xcompat_secondary_y(self, ts: Series) -> None:
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize('GMT').tz_convert('CET')
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_area_sharey_dont_overwrite(self, ts: Series) -> None:
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)
        abs(ts).plot(ax=ax1, kind='area')
        abs(ts).plot(ax=ax2, kind='area')
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    def test_label(self) -> None:
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(label='LABEL', legend=True, ax=ax)
        _check_legend_labels(ax, labels=['LABEL'])

    def test_label_none(self) -> None:
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=[''])

    def test_label_ser_name(self) -> None:
        s = Series([1, 2], name='NAME')
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=['NAME'])

    def test_label_ser_name_override(self) -> None:
        s = Series([1, 2], name='NAME')
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, label='LABEL', ax=ax)
        _check_legend_labels(ax, labels=['LABEL'])

    def test_label_ser_name_override_dont_draw(self) -> None:
        s = Series([1, 2], name='NAME')
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=False, label='LABEL', ax=ax)
        assert ax.get_legend() is None
        ax.legend()
        _check_legend_labels(ax, labels=['LABEL'])

    def test_boolean(self) -> None:
        s = Series([False, False, True])
        _check_plot_works(s.plot, include_bool=True)
        msg = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            _check_plot_works(s.plot)

    @pytest.mark.parametrize('index', [None, date_range('2020-01-01', periods=4)])
    def test_line_area_nan_series(self, index: pd.DatetimeIndex) -> None:
        values = [1, 2, np.nan, 3]
        d = Series(values, index=index)
        ax = _check_plot_works(d.plot)
        masked = ax.lines[0].get_ydata()
        exp = np.array([1, 2, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
        tm.assert_numpy_array_equal(masked.mask, np.array([False, False, True, False]))
        expected = np.array([1, 2, 0, 3], dtype=np.float64)
        ax = _check_plot_works(d.plot, stacked=True)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area, stacked=False)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)

    def test_line_use_index_false(self) -> None:
        s = Series([1, 2, 3], index=['a', 'b', 'c'])
        s.index.name = 'The Index'
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(use_index=False, ax=ax)
        label = ax.get_xlabel()
        assert label == ''

    def test_line_use_index_false_diff_var(self) -> None:
        s = Series([1, 2, 3], index=['a', 'b', 'c'])
        s.index.name = 'The Index'
        _, ax = mpl.pyplot.subplots()
        ax2 = s.plot.bar(use_index=False, ax=ax)
        label2 = ax2.get_xlabel()
        assert label2 == ''

    @pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
    @pytest.mark.parametrize('axis, meth', [('yaxis', 'bar'), ('xaxis', 'barh')])
    def test_bar_log(self, axis: str, meth: str) -> None:
        expected = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
        _, ax = mpl.pyplot.subplots()
        ax = getattr(Series([200, 500]).plot, meth)(log=True, ax=ax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    @pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
    @pytest.mark.parametrize('axis, kind, res_meth', [['yaxis', 'bar', 'get_ylim'], ['xaxis', 'barh', 'get_xlim']])
    def test_bar_log_kind_bar(self, axis: str, kind: str, res_meth: str) -> None:
        expected = np.array([1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
        _, ax = mpl.pyplot.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind=kind, ax=ax)
        ymin = 0.0007943282347242822
        ymax = 0.12589254117941673
        res = getattr(ax, res_meth)()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    def test_bar_ignore_index(self) -> None:
        df = Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.bar(use_index=False, ax=ax)
        _check_text_labels(ax.get_xticklabels(), ['0', '1', '2', '3'])

    def test_bar_user_colors(self) -> None:
        s = Series([1, 2, 3, 4])
        ax = s.plot.bar(color=['red', 'blue', 'blue', 'red'])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
        assert result == expected

    def test_rotation_default(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(ax=ax)
        _check_ticks_props(axes, xrot=0)

    def test_rotation_30(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(rot=30, ax=ax)
        _check_ticks_props(axes, xrot=30)

    def test_irregular_datetime(self) -> None:
        rng = date_range('1/1/2000', '1/15/2000')
        rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        xp = DatetimeConverter.convert(datetime(1999, 1, 1), '', ax)
        ax.set_xlim('1/1/1999', '1/1/2001')
        assert xp == ax.get_xlim()[0]
        _check_ticks_props(ax, xrot=30)

    def test_unsorted_index_xlim(self) -> None:
        ser = Series([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0], index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0])
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data(orig=False)[0])
        assert xmax >= np.nanmax(lines[0].get_data(orig=False)[0])

    def test_pie_series(self) -> None:
        series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
        ax = _check_plot_works(series.plot.pie)
        _check_text_labels(ax.texts, series.index)
        assert ax.get_ylabel() == ''

    def test_pie_arrow_type(self) -> None:
        pytest.importorskip('pyarrow')
        ser = Series([1, 2, 3, 4], dtype='int32[pyarrow]')
        _check_plot_works(ser.plot.pie)

    def test_pie_series_no_label(self) -> None:
        series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
        ax = _check_plot_works(series.plot.pie, labels=None)
        _check_text_labels(ax.texts, [''] * 5)

    def test_pie_series_less_colors_than_elements(self) -> None:
        series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
        color_args = ['r', 'g', 'b']
        ax = _check_plot_works(series.plot.pie, colors=color_args)
        color_expected = ['r', 'g', 'b', 'r', 'g']
        _check_colors(ax.patches, facecolors=color_expected)

    def test_pie_series_labels_and_colors(self) -> None:
        series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
        labels = ['A', 'B', 'C', 'D', 'E']
        color_args = ['r', 'g', 'b', 'c', 'm']
        ax = _check_plot_works(series.plot.pie, labels=labels, colors=color_args)
        _check_text_labels(ax.texts, labels)
        _check_colors(ax.patches, facecolors=color_args)

    def test_pie_series_autopct_and_fontsize(self) -> None:
        series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
        color_args = ['r', 'g', 'b', 'c', 'm']
        ax = _check_plot_works(series.plot.pie, colors=color_args, autopct='%.2f', fontsize=7)
        pcts = [f'{s * 100:.2f}' for s in series.values / series.sum()]
        expected_texts = list(chain.from_iterable(zip(series.index, pcts)))
        _check_text_labels(ax.texts, expected_texts)
        for t in ax.texts:
            assert t.get_fontsize() == 7

    def test_pie_series_negative_raises(self) -> None:
        series = Series([1, 2, 0, 4, -1], index=['a', 'b', 'c', 'd', 'e'])
        with pytest.raises(ValueError, match="pie plot doesn't allow negative values"):
            series.plot.pie()

    def test_pie_series_nan(self) -> None:
        series = Series([1, 2, np.nan, 4], index=['a', 'b', 'c', 'd'], name='YLABEL')
        ax = _check_plot_works(series.plot.pie)
        _check_text_labels(ax.texts, ['a', 'b', '', 'd'])

    def test_pie_nan(self) -> None:
        s = Series([1, np.nan, 1, 1])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot.pie(legend=True, ax=ax)
        expected = ['0', '', '2', '3']
        result = [x.get_text() for x in ax.texts]
        assert result == expected

    def test_df_series_secondary_legend(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=list('abc'))
        s = Series(np.random.default_rng(2).standard_normal(10), name='x')
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        _check_legend_labels(ax, labels=['a', 'b', 'c', 'x (right)'])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=list('abc'))
        s = Series(np.random.default_rng(2).standard_normal(10), name='x')
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        expected = ['a (right)', 'b (right)', 'c (right)', 'x (right)']
        _check_legend_labels(ax.left_ax, labels=expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both_with_axis_2(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=list('abc'))
        s = Series(np.random.default_rng(2).standard_normal(10), name='x')
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(secondary_y=True, mark_right=False, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        expected = ['a', 'b', 'c', 'x (right)']
        _check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    @pytest.mark.parametrize('input_logy, expected_scale', [(True, 'log'), ('sym', 'symlog')])
    @pytest.mark.parametrize('secondary_kwarg', [{}, {'secondary_y': True}])
    def test_secondary_logy(self, input_logy: bool | str, expected_scale: str, secondary_kwarg: dict) -> None:
        s1 = Series(np.random.default_rng(2).standard_normal(10))
        ax1 = s1.plot(logy=input_logy, **secondary_kwarg)
        assert ax1.get_yscale() == expected_scale

    def test_plot_fails_with_dupe_color_and_style(self) -> None:
        x = Series(np.random.default_rng(2).standard_normal(2))
        _, ax = mpl.pyplot.subplots()
        msg = "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
        with pytest.raises(ValueError, match=msg):
            x.plot(style='k--', color='k', ax=ax)

    @pytest.mark.parametrize('bw_method, ind', [['scott', 20], [None, 20], [None, np.int_(20)], [0.5, np.linspace(-100, 100, 20)]])
    def test_kde_kwargs(self, ts: Series, bw_method: str | float | None, ind: int | np.ndarray) -> None:
        pytest.importorskip('scipy')
        _check_plot_works(ts.plot.kde, bw_method=bw_method, ind=ind)

    @pytest.mark.parametrize('bw_method, ind, weights', [['scott', 20, None], [None, 20, None], [None, np.int_(20), None], [0.5, np.linspace(-100, 100, 20), None], ['scott', 40, np.linspace(0.0, 2.0, 50)]])
    def test_kde_kwargs_weights(self, bw_method: str | float | None, ind: int | np.ndarray, weights: np.ndarray | None) -> None:
        pytest.importorskip('scipy')
        s = Series(np.random.default_rng(2).uniform(size=50))
        _check_plot_works(s.plot.kde, bw_method=bw_method, ind=ind, weights=weights)

    def test_density_kwargs(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        sample_points = np.linspace(-100, 100, 20)
        _check_plot_works(ts.plot.density, bw_method=0.5, ind=sample_points)

    def test_kde_kwargs_check_axes(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        _, ax = mpl.pyplot.subplots()
        sample_points = np.linspace(-100, 100, 20)
        ax = ts.plot.kde(logy=True, bw_method=0.5, ind=sample_points, ax=ax)
        _check_ax_scales(ax, yaxis='log')
        _check_text_labels(ax.yaxis.get_label(), 'Density')

    def test_kde_missing_vals(self) -> None:
        pytest.importorskip('scipy')
        s = Series(np.random.default_rng(2).uniform(size=50))
        s[0] = np.nan
        axes = _check_plot_works(s.plot.kde)
        assert any(~np.isnan(axes.lines[0].get_xdata()))

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_boxplot_series(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.box(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis='log')
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [ts.name])
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [''] * len(ylabels))

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds)
    def test_kind_kwarg(self, kind: str) -> None:
        pytest.importorskip('scipy')
        s = Series(range(3))
        _, ax = mpl.pyplot.subplots()
        s.plot(kind=kind, ax=ax)
        mpl.pyplot.close()

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds)
    def test_kind_attr(self, kind: str) -> None:
        pytest.importorskip('scipy')
        s = Series(range(3))
        _, ax = mpl.pyplot.subplots()
        getattr(s.plot, kind)()
        mpl.pyplot.close()

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_invalid_plot_data(self, kind: str) -> None:
        s = Series(list('abcd'))
        _, ax = mpl.pyplot.subplots()
        msg = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            s.plot(kind=kind, ax=ax)

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_valid_object_plot(self, kind: str) -> None:
        pytest.importorskip('scipy')
        s = Series(range(10), dtype=object)
        _check_plot_works(s.plot, kind=kind)

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_partially_invalid_plot_data(self, kind: str) -> None:
        s = Series(['a', 'b', 1.0, 2])
        _, ax = mpl.pyplot.subplots()
        msg = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            s.plot(kind=kind, ax=ax)

    def test_invalid_kind(self) -> None:
        s = Series([1, 2])
        with pytest.raises(ValueError, match='invalid_kind is not a valid plot kind'):
            s.plot(kind='invalid_kind')

    def test_dup_datetime_index_plot(self) -> None:
        dr1 = date_range('1/1/2009', periods=4)
        dr2 = date_range('1/2/2009', periods=4)
        index = dr1.append(dr2)
        values = np.random.default_rng(2).standard_normal(index.size)
        s = Series(values, index=index)
        _check_plot_works(s.plot)

    def test_errorbar_asymmetrical(self) -> None:
        s = Series(np.arange(10), name='x')
        err = np.random.default_rng(2).random((2, 10))
        ax = s.plot(yerr=err, xerr=err)
        result = np.vstack([i.vertices[:, 1] for i in ax.collections[1].get_paths()])
        expected = err.T * np.array([-1, 1]) + s.to_numpy().reshape(-1, 1)
        tm.assert_numpy_array_equal(result, expected)

    def test_errorbar_asymmetrical_error(self) -> None:
        s = Series(np.arange(10), name='x')
        msg = f'Asymmetrical error bars should be provided with the shape \\(2, {len(s)}\\)'
        with pytest.raises(ValueError, match=msg):
            s.plot(yerr=np.random.default_rng(2).random((2, 11)))

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['line', 'bar'])
    @pytest.mark.parametrize('yerr', [Series(np.abs(np.random.default_rng(2).standard_normal(10))), np.abs(np.random.default_rng(2).standard_normal(10)), list(np.abs(np.random.default_rng(2).standard_normal(10))), DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=['x', 'y'])])
    def test_errorbar_plot(self, kind: str, yerr: Series | np.ndarray | list | DataFrame) -> None:
        s = Series(np.arange(10), name='x')
        ax = _check_plot_works(s.plot, yerr=yerr, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_yerr_0(self) -> None:
        s = Series(np.arange(10), name='x')
        s_err = np.abs(np.random.default_rng(2).standard_normal(10))
        ax = _check_plot_works(s.plot, xerr=s_err)
        _check_has_errorbars(ax, xerr=1, yerr=0)

    @pytest.mark.slow
    @pytest.mark.parametrize('yerr', [Series(np.abs(np.random.default_rng(2).standard_normal(12))), DataFrame(np.abs(np.random.default_rng(2).standard_normal((12, 2))), columns=['x', 'y'])])
    def test_errorbar_plot_ts(self, yerr: Series | DataFrame) -> None:
        ix = date_range('1/1/2000', '1/1/2001', freq='ME')
        ts = Series(np.arange(12), index=ix, name='x')
        yerr.index = ix
        ax = _check_plot_works(ts.plot, yerr=yerr)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr_shape(self) -> None:
        s = Series(np.arange(10), name='x')
        with tm.external_error_raised(ValueError):
            s.plot(yerr=np.arange(11))

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr(self) -> None:
        s = Series(np.arange(10), name='x')
        s_err = ['zzz'] * 10
        with tm.external_error_raised(TypeError):
            s.plot(yerr=s_err)

    @pytest.mark.slow
    def test_table_true(self, series: Series) -> None:
        _check_plot_works(series.plot, table=True)

    @pytest.mark.slow
    def test_table_self(self, series: Series) -> None:
        _check_plot_works(series.plot, table=series)

    @pytest.mark.slow
    def test_series_grid_settings(self) -> None:
        pytest.importorskip('scipy')
        _check_grid_settings(Series([1, 2, 3]), plotting.PlotAccessor._series_kinds + plotting.PlotAccessor._common_kinds)

    @pytest.mark.parametrize('c', ['r', 'red', 'green', '#FF0000'])
    def test_standard_colors(self, c: str) -> None:
        result = get_standard_colors(1, color=c)
        assert result == [c]
        result = get_standard_colors(1, color=[c])
        assert result == [c]
        result = get_standard_colors(3, color=c)
        assert result == [c] * 3
        result = get_standard_colors(3, color=[c])
        assert result == [c] * 3

    def test_standard_colors_all(self) -> None:
        for c in mpl.colors.cnames:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]
            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]
            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3
            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3
        for c in mpl.colors.ColorConverter.colors:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]
            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]
            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3
            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

    def test_series_plot_color_kwargs(self) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1).plot(color='green', ax=ax)
        _check_colors(ax.get_lines(), linecolors=['green'])

    def test_time_series_plot_color_kwargs(self) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1, index=date_range('1/1/2000', periods=12)).plot(color='green', ax=ax)
        _check_colors(ax.get_lines(), linecolors=['green'])

    def test_time_series_plot_color_with_empty_kwargs(self) -> None:
        def_colors = _unpack_cycler(mpl.rcParams)
        index = date_range('1/1/2000', periods=12)
        s = Series(np.arange(1, 13), index=index)
        ncolors = 3
        _, ax = mpl.pyplot.subplots()
        for i in range(ncolors):
            ax = s.plot(ax=ax)
        _check_colors(ax.get_lines(), linecolors=def_colors[:ncolors])

    def test_xticklabels(self) -> None:
        s = Series(np.arange(10), index=[f'P{i:02d}' for i in range(10)])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(xticks=[0, 3, 5, 9], ax=ax)
        exp = [f'P{i:02d}' for i in [0, 3, 5, 9]]
        _check_text_labels(ax.get_xticklabels(), exp)

    def test_xtick_barPlot(self) -> None:
        s = Series(range(10), index=[f'P{i:02d}' for i in range(10)])
        ax = s.plot.bar(xticks=range(0, 11, 2))
        exp = np.array(list(range(0, 11, 2)))
        tm.assert_numpy_array_equal(exp, ax.get_xticks())

    def test_custom_business_day_freq(self) -> None:
        s = Series(range(100, 121), index=pd.bdate_range(start='2014-05-01', end='2014-06-01', freq=CustomBusinessDay(holidays=['2014-05-26'])))
        _check_plot_works(s.plot)

    @pytest.mark.xfail(reason='GH#24426, see also github.com/pandas-dev/pandas/commit/ef1bd69fa42bbed5d09dd17f08c44fc8bfc2b685#r61470674')
    def test_plot_accessor_updates_on_inplace(self) -> None:
        ser = Series([1, 2, 3, 4])
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        before = ax.xaxis.get_ticklocs()
        ser.drop([0, 1], inplace=True)
        _, ax = mpl.pyplot.subplots()
        after = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(before, after)

    @pytest.mark.parametrize('kind', ['line', 'area'])
    def test_plot_xlim_for_series(self, kind: str) -> None:
        s = Series([2, 3])
        _, ax = mpl.pyplot.subplots()
        s.plot(kind=kind, ax=ax)
        xlims = ax.get_xlim()
        assert xlims[0] < 0
        assert xlims[1] > 1

    def test_plot_no_rows(self) -> None:
        df = Series(dtype=int)
        assert df.empty
        ax = df.plot()
        assert len(ax.get_lines()) == 1
        line = ax.get_lines()[0]
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self) -> None:
        df = Series(['a', 'b', 'c'])
        with pytest.raises(TypeError, match='no numeric data to plot'):
            df.plot()

    @pytest.mark.parametrize('data, index', [([1, 2, 3, 4], [3, 2, 1, 0]), ([10, 50, 20, 30], [1910, 1920, 1980, 1950])])
    def test_plot_order(self, data: list, index: list) -> None:
        ser = Series(data=data, index=index)
        ax = ser.plot(kind='bar')
        expected = ser.tolist()
        result = [patch.get_bbox().ymax for patch in sorted(ax.patches, key=lambda patch: patch.get_bbox().xmax)]
        assert expected == result

    def test_style_single_ok(self) -> None:
        s = Series([1, 2])
        ax = s.plot(style='s', color='C3')
        assert ax.lines[0].get_color() == 'C3'

    @pytest.mark.parametrize('index_name, old_label, new_label', [(None, '', 'new'), ('old', 'old', 'new'), (None, '', '')])
    @pytest.mark.parametrize('kind', ['line', 'area', 'bar', 'barh', 'hist'])
    def test_xlabel_ylabel_series(self, kind: str, index_name: str | None, old_label: str, new_label: str) -> None:
        ser = Series([1, 2, 3, 4])
        ser.index.name = index_name
        ax = ser.plot(kind=kind)
        if kind == 'barh':
            assert ax.get_xlabel() == ''
            assert ax.get_ylabel() == old_label
        elif kind == 'hist':
            assert ax.get_xlabel() == ''
            assert ax.get_ylabel() == 'Frequency'
        else:
            assert ax.get_ylabel() == ''
            assert ax.get_xlabel() == old_label
        ax = ser.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        assert ax.get_ylabel() == new_label
        assert ax.get_xlabel() == new_label

    @pytest.mark.parametrize('index', [pd.timedelta_range(start=0, periods=2, freq='D'), [pd.Timedelta(days=1), pd.Timedelta(days=2)]])
    def test_timedelta_index(self, index: pd.TimedeltaIndex | list) -> None:
        xlims = (3, 1)
        ax = Series([1, 2], index=index).plot(xlim=xlims)
        assert ax.get_xlim() == (3, 1)

    def test_series_none_color(self) -> None:
        series = Series([1, 2, 3])
        ax = series.plot(color=None)
        expected = _unpack_cycler(mpl.pyplot.rcParams)[:1]
        _check_colors(ax.get_lines(), linecolors=expected)

    @pytest.mark.slow
    def test_plot_no_warning(self, ts: Series) -> None:
        with tm.assert_produces_warning(False):
            _ = ts.plot()

    def test_secondary_y_subplot_axis_labels(self) -> None:
        s1 = Series([5, 7, 6, 8, 7], index=[1, 2, 3, 4, 5])
        s2 = Series([6, 4, 5, 3, 4], index=[1, 2, 3, 4, 5])
        ax = plt.subplot(2, 1, 1)
        s1.plot(ax=ax)
        s2.plot(ax=ax, secondary_y=True)
        ax2 = plt.subplot(2, 1, 2)
        s1.plot(ax=ax2)
        assert len(ax.xaxis.get_minor_ticks()) == 0
        assert len(ax.get_xticklabels()) > 0
