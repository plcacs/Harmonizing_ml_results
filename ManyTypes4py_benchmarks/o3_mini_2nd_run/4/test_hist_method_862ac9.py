#!/usr/bin/env python3
"""
Test cases for .hist method with type annotations
"""
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas import DataFrame, Index, Series, date_range, to_datetime
import pandas._testing as tm
from pandas.tests.plotting.common import (_check_ax_scales, _check_axes_shape,
                                          _check_colors, _check_legend_labels,
                                          _check_patches_all_filled,
                                          _check_plot_works, _check_text_labels,
                                          _check_ticks_props, get_x_axis,
                                          get_y_axis)
mpl = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')
from pandas.plotting._matplotlib.hist import _grouped_hist


@pytest.fixture
def ts() -> Series:
    return Series(np.arange(30, dtype=np.float64),
                  index=date_range('2020-01-01', periods=30, freq='B'),
                  name='ts')


class TestSeriesPlots:

    @pytest.mark.parametrize('kwargs', [{}, {'grid': False}, {'figsize': (8, 10)}])
    def test_hist_legacy_kwargs(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        _check_plot_works(ts.hist, **kwargs)

    @pytest.mark.parametrize('kwargs', [{}, {'bins': 5}])
    def test_hist_legacy_kwargs_warning(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(ts.hist, by=ts.index.month, **kwargs)

    def test_hist_legacy_ax(self, ts: Series) -> None:
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, default_axes=True)

    def test_hist_legacy_ax_and_fig(self, ts: Series) -> None:
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, figure=fig, default_axes=True)

    def test_hist_legacy_fig(self, ts: Series) -> None:
        fig, _ = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, figure=fig, default_axes=True)

    def test_hist_legacy_multi_ax(self, ts: Series) -> None:
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2)
        _check_plot_works(ts.hist, figure=fig, ax=ax1, default_axes=True)
        _check_plot_works(ts.hist, figure=fig, ax=ax2, default_axes=True)

    def test_hist_legacy_by_fig_error(self, ts: Series) -> None:
        fig, _ = mpl.pyplot.subplots(1, 1)
        msg: str = "Cannot pass 'figure' when using the 'by' argument, since a new 'Figure' instance will be created"
        with pytest.raises(ValueError, match=msg):
            ts.hist(by=ts.index, figure=fig)

    def test_hist_bins_legacy(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        ax = df.hist(bins=2)[0][0]
        assert len(ax.patches) == 2

    def test_hist_layout(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        msg: str = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=(1, 1))
        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=[1, 1])

    @pytest.mark.slow
    @pytest.mark.parametrize('by, layout, axes_num, res_layout',
                             [['gender', (2, 1), 2, (2, 1)],
                              ['gender', (3, -1), 2, (3, 1)],
                              ['category', (4, 1), 4, (4, 1)],
                              ['category', (2, -1), 4, (2, 2)],
                              ['category', (3, -1), 4, (3, 2)],
                              ['category', (-1, 4), 4, (1, 4)],
                              ['classroom', (2, 2), 3, (2, 2)]])
    def test_hist_layout_with_by(self, hist_df: DataFrame, by: Union[str, Any],
                                 layout: Tuple[int, int], axes_num: int,
                                 res_layout: Tuple[int, int]) -> None:
        df: DataFrame = hist_df
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.height.hist, by=getattr(df, by), layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=res_layout)

    def test_hist_layout_with_by_shape(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        axes = df.height.hist(by=df.category, layout=(4, 2), figsize=(12, 7))
        _check_axes_shape(axes, axes_num=4, layout=(4, 2), figsize=(12, 7))

    def test_hist_no_overlap(self) -> None:
        x: Series = Series(np.random.default_rng(2).standard_normal(2))
        y: Series = Series(np.random.default_rng(2).standard_normal(2))
        plt.subplot(121)
        x.hist()
        plt.subplot(122)
        y.hist()
        fig = plt.gcf()
        axes = fig.axes
        assert len(axes) == 2

    def test_hist_by_no_extra_plots(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        df.height.hist(by=df.gender)
        assert len(mpl.pyplot.get_fignums()) == 1

    def test_plot_fails_when_ax_differs_from_figure(self, ts: Series) -> None:
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        ax1 = fig1.add_subplot(111)
        msg: str = 'passed axis not bound to passed figure'
        with pytest.raises(AssertionError, match=msg):
            ts.hist(ax=ax1, figure=fig2)

    @pytest.mark.parametrize('histtype, expected', [('bar', True), ('barstacked', True), ('step', False), ('stepfilled', True)])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        ser: Series = Series(np.random.default_rng(2).integers(1, 10))
        ax = ser.hist(histtype=histtype)
        _check_patches_all_filled(ax, filled=expected)

    @pytest.mark.parametrize('by, expected_axes_num, expected_layout', [(None, 1, (1, 1)), ('b', 2, (1, 2))])
    def test_hist_with_legend(self, by: Optional[str], expected_axes_num: int, expected_layout: Tuple[int, int]) -> None:
        index: List[str] = 5 * ['1'] + 5 * ['2']
        s: Series = Series(np.random.default_rng(2).standard_normal(10), index=index, name='a')
        s.index.name = 'b'
        axes = _check_plot_works(s.hist, default_axes=True, legend=True, by=by)
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        _check_legend_labels(axes, 'a')

    @pytest.mark.parametrize('by', [None, 'b'])
    def test_hist_with_legend_raises(self, by: Optional[str]) -> None:
        index: List[str] = 5 * ['1'] + 5 * ['2']
        s: Series = Series(np.random.default_rng(2).standard_normal(10), index=index, name='a')
        s.index.name = 'b'
        with pytest.raises(ValueError, match='Cannot use both legend and label'):
            s.hist(legend=True, by=by, label='c')

    def test_hist_kwargs(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 5
        _check_text_labels(ax.yaxis.get_label(), 'Frequency')

    def test_hist_kwargs_horizontal(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(orientation='horizontal', ax=ax)
        _check_text_labels(ax.xaxis.get_label(), 'Frequency')

    def test_hist_kwargs_align(self, ts: Series) -> None:
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(align='left', stacked=True, ax=ax)

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_hist_kde(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis='log')
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [''] * len(xlabels))
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [''] * len(ylabels))

    def test_hist_kde_plot_works(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        _check_plot_works(ts.plot.kde)

    def test_hist_kde_density_works(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        _check_plot_works(ts.plot.density)

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_hist_kde_logy(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.kde(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis='log')
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [''] * len(xlabels))
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [''] * len(ylabels))

    def test_hist_kde_color_bins(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(logy=True, bins=10, color='b', ax=ax)
        _check_ax_scales(ax, yaxis='log')
        assert len(ax.patches) == 10
        _check_colors(ax.patches, facecolors=['b'] * 10)

    def test_hist_kde_color(self, ts: Series) -> None:
        pytest.importorskip('scipy')
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.kde(logy=True, color='r', ax=ax)
        _check_ax_scales(ax, yaxis='log')
        lines = ax.get_lines()
        assert len(lines) == 1
        _check_colors(lines, ['r'])


class TestDataFramePlots:

    @pytest.mark.slow
    def test_hist_df_legacy(self, hist_df: DataFrame) -> None:
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(hist_df.hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000,
                                                                819331200000000000, size=10, dtype=np.int64))
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, grid=False)
        _check_axes_shape(axes, axes_num=3, layout=(2, 2))
        assert not axes[1, 1].get_visible()
        _check_plot_works(df[[2]].hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout2(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
        _check_plot_works(df.hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout3(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(np.random.default_rng(2).integers(812419200000000000,
                                                                819331200000000000, size=10, dtype=np.int64))
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, layout=(4, 2))
        _check_axes_shape(axes, axes_num=6, layout=(4, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [{'sharex': True, 'sharey': True}, {'figsize': (8, 10)}, {'bins': 5}])
    def test_hist_df_legacy_layout_kwargs(self, kwargs: Dict[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(np.random.default_rng(2).integers(812419200000000000,
                                                                819331200000000000, size=10, dtype=np.int64))
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.hist, **kwargs)

    @pytest.mark.slow
    def test_hist_df_legacy_layout_labelsize_rot(self, frame_or_series: Callable[[Any], Union[DataFrame, Series]]) -> None:
        obj = frame_or_series(range(10))
        xf, yf = (20, 18)
        xrot, yrot = (30, 40)
        axes = obj.hist(xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
        _check_ticks_props(axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)

    @pytest.mark.slow
    def test_hist_df_legacy_rectangles(self) -> None:
        ser: Series = Series(range(10))
        ax = ser.hist(cumulative=True, bins=4, density=True)
        rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

    @pytest.mark.slow
    def test_hist_df_legacy_scale(self) -> None:
        ser: Series = Series(range(10))
        ax = ser.hist(log=True)
        _check_ax_scales(ax, yaxis='log')

    @pytest.mark.slow
    def test_hist_df_legacy_external_error(self) -> None:
        ser: Series = Series(range(10))
        with tm.external_error_raised(AttributeError):
            ser.hist(foo='bar')

    def test_hist_non_numerical_or_datetime_raises(self) -> None:
        df: DataFrame = DataFrame({
            'a': np.random.default_rng(2).random(10),
            'b': np.random.default_rng(2).integers(0, 10, 10),
            'c': to_datetime(np.random.default_rng(2).integers(1582800000000000000,
                                                                 1583500000000000000, 10, dtype=np.int64)),
            'd': to_datetime(np.random.default_rng(2).integers(1582800000000000000,
                                                                 1583500000000000000, 10, dtype=np.int64), utc=True)
        })
        df_o: DataFrame = df.astype(object)
        msg: str = 'hist method requires numerical or datetime columns, nothing to plot.'
        with pytest.raises(ValueError, match=msg):
            df_o.hist()

    @pytest.mark.parametrize('layout_test', (
        {'layout': None, 'expected_size': (2, 2)},
        {'layout': (2, 2), 'expected_size': (2, 2)},
        {'layout': (4, 1), 'expected_size': (4, 1)},
        {'layout': (1, 4), 'expected_size': (1, 4)},
        {'layout': (3, 3), 'expected_size': (3, 3)},
        {'layout': (-1, 4), 'expected_size': (1, 4)},
        {'layout': (4, -1), 'expected_size': (4, 1)},
        {'layout': (-1, 2), 'expected_size': (2, 2)},
        {'layout': (2, -1), 'expected_size': (2, 2)}
    ))
    def test_hist_layout(self, layout_test: Dict[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000,
                                                                819331200000000000, size=10, dtype=np.int64))
        axes = df.hist(layout=layout_test['layout'])
        expected: Tuple[int, int] = layout_test['expected_size']
        _check_axes_shape(axes, axes_num=3, layout=expected)

    def test_hist_layout_error(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000,
                                                                819331200000000000, size=10, dtype=np.int64))
        msg1: str = 'Layout of 1x1 must be larger than required size 3'
        with pytest.raises(ValueError, match=msg1):
            df.hist(layout=(1, 1))
        msg2: str = re.escape('Layout must be a tuple of (rows, columns)')
        with pytest.raises(ValueError, match=msg2):
            df.hist(layout=(1,))
        msg3: str = 'At least one dimension of layout must be positive'
        with pytest.raises(ValueError, match=msg3):
            df.hist(layout=(-1, -1))

    def test_tight_layout(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000,
                                                                819331200000000000, size=10, dtype=np.int64))
        _check_plot_works(df.hist, default_axes=True)
        mpl.pyplot.tight_layout()

    def test_hist_subplot_xrot(self) -> None:
        df: DataFrame = DataFrame({'length': [1.5, 0.5, 1.2, 0.9, 3],
                                   'animal': ['pig', 'rabbit', 'pig', 'pig', 'rabbit']})
        axes = _check_plot_works(df.hist, default_axes=True, column='length', by='animal', bins=5, xrot=0)
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize('column, expected',
                             [(None, ['width', 'length', 'height']),
                              (['length', 'width', 'height'], ['length', 'width', 'height'])])
    def test_hist_column_order_unchanged(self, column: Optional[Union[List[str], str]], expected: List[str]) -> None:
        df: DataFrame = DataFrame({
            'width': [0.7, 0.2, 0.15, 0.2, 1.1],
            'length': [1.5, 0.5, 1.2, 0.9, 3],
            'height': [3, 0.5, 3.4, 2, 1]
        }, index=['pig', 'rabbit', 'duck', 'chicken', 'horse'])
        axes = _check_plot_works(df.hist, default_axes=True, column=column, layout=(1, 3))
        result: List[str] = [axes[0, i].get_title() for i in range(3)]
        assert result == expected

    @pytest.mark.parametrize('histtype, expected', [('bar', True), ('barstacked', True), ('step', False), ('stepfilled', True)])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).integers(1, 10, size=(10, 2)), columns=['a', 'b'])
        ax = df.hist(histtype=histtype)
        _check_patches_all_filled(ax, filled=expected)

    @pytest.mark.parametrize('by', [None, 'c'])
    @pytest.mark.parametrize('column', [None, 'b'])
    def test_hist_with_legend(self, by: Optional[str], column: Optional[Union[str, List[str]]]) -> None:
        expected_axes_num: int = 1 if by is None and column is not None else 2
        expected_layout: Tuple[int, int] = (1, expected_axes_num)
        expected_labels: Union[str, List[Any]] = column if column is not None else ['a', 'b']
        if by is not None:
            expected_labels = [expected_labels] * 2  # type: ignore
        index: Index = Index(5 * ['1'] + 5 * ['2'], name='c')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index, columns=['a', 'b'])
        axes = _check_plot_works(df.hist, default_axes=True, legend=True, by=by, column=column)
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        if by is None and column is None:
            axes = axes[0]
        if isinstance(expected_labels, list):
            for expected_label, ax in zip(expected_labels, axes):
                _check_legend_labels(ax, expected_label)
        else:
            _check_legend_labels(axes[0], expected_labels)

    @pytest.mark.parametrize('by', [None, 'c'])
    @pytest.mark.parametrize('column', [None, 'b'])
    def test_hist_with_legend_raises(self, by: Optional[str], column: Optional[Union[str, List[str]]]) -> None:
        index: Index = Index(5 * ['1'] + 5 * ['2'], name='c')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index, columns=['a', 'b'])
        with pytest.raises(ValueError, match='Cannot use both legend and label'):
            df.hist(legend=True, by=by, column=column, label='d')

    def test_hist_df_kwargs(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 10

    def test_hist_df_with_nonnumerics(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'B', 'C', 'D'])
        df['E'] = ['x', 'y'] * 5
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 20

    def test_hist_df_with_nonnumerics_no_bins(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'B', 'C', 'D'])
        df['E'] = ['x', 'y'] * 5
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(ax=ax)
        assert len(ax.patches) == 40

    def test_hist_secondary_legend(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=list('abcd'))
        _, ax = mpl.pyplot.subplots()
        ax = df['a'].plot.hist(legend=True, ax=ax)
        df['b'].plot.hist(ax=ax, legend=True, secondary_y=True)
        _check_legend_labels(ax, labels=['a', 'b (right)'])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_hist_secondary_secondary(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=list('abcd'))
        _, ax = mpl.pyplot.subplots()
        ax = df['a'].plot.hist(legend=True, secondary_y=True, ax=ax)
        df['b'].plot.hist(ax=ax, legend=True, secondary_y=True)
        _check_legend_labels(ax.left_ax, labels=['a (right)', 'b (right)'])
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_hist_secondary_primary(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=list('abcd'))
        _, ax = mpl.pyplot.subplots()
        ax = df['a'].plot.hist(legend=True, secondary_y=True, ax=ax)
        df['b'].plot.hist(ax=ax, legend=True)
        _check_legend_labels(ax.left_ax, labels=['a (right)', 'b'])
        assert ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_hist_with_nans_and_weights(self) -> None:
        df: DataFrame = DataFrame([[np.nan, 0.2, 0.3],
                                   [0.4, np.nan, np.nan],
                                   [0.7, 0.8, 0.9]], columns=list('abc'))
        weights: np.ndarray = np.array([0.25, 0.3, 0.45])
        no_nan_df: DataFrame = DataFrame([[0.4, 0.2, 0.3],
                                          [0.7, 0.8, 0.9]], columns=list('abc'))
        no_nan_weights: np.ndarray = np.array([[0.3, 0.25, 0.25],
                                               [0.45, 0.45, 0.45]])
        _, ax0 = mpl.pyplot.subplots()
        df.plot.hist(ax=ax0, weights=weights)
        rects = [x for x in ax0.get_children() if isinstance(x, mpl.patches.Rectangle)]
        heights = [rect.get_height() for rect in rects]
        _, ax1 = mpl.pyplot.subplots()
        no_nan_df.plot.hist(ax=ax1, weights=no_nan_weights)
        no_nan_rects = [x for x in ax1.get_children() if isinstance(x, mpl.patches.Rectangle)]
        no_nan_heights = [rect.get_height() for rect in no_nan_rects]
        assert all((h0 == h1 for h0, h1 in zip(heights, no_nan_heights)))
        idxerror_weights: np.ndarray = np.array([[0.3, 0.25],
                                                  [0.45, 0.45]])
        msg: str = 'weights must have the same shape as data, or be a single column'
        _, ax2 = mpl.pyplot.subplots()
        with pytest.raises(ValueError, match=msg):
            no_nan_df.plot.hist(ax=ax2, weights=idxerror_weights)


class TestDataFrameGroupByPlots:

    def test_grouped_hist_legacy(self) -> None:
        rs = np.random.default_rng(10)
        df: DataFrame = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
        df['B'] = to_datetime(rs.integers(812419200000000000,
                                            819331200000000000, size=10, dtype=np.int64))
        df['C'] = rs.integers(0, 4, 10)
        df['D'] = ['X'] * 10
        axes = _grouped_hist(df.A, by=df.C)
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    def test_grouped_hist_legacy_axes_shape_no_col(self) -> None:
        rs = np.random.default_rng(10)
        df: DataFrame = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
        df['B'] = to_datetime(rs.integers(812419200000000000,
                                            819331200000000000, size=10, dtype=np.int64))
        df['C'] = rs.integers(0, 4, 10)
        df['D'] = ['X'] * 10
        axes = df.hist(by=df.C)
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    def test_grouped_hist_legacy_single_key(self) -> None:
        rs = np.random.default_rng(2)
        df: DataFrame = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
        df['B'] = to_datetime(rs.integers(812419200000000000,
                                            819331200000000000, size=10, dtype=np.int64))
        df['C'] = rs.integers(0, 4, 10)
        df['D'] = ['X'] * 10
        axes = df.hist(by='D', rot=30)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        _check_ticks_props(axes, xrot=30)

    def test_grouped_hist_legacy_grouped_hist_kwargs(self) -> None:
        rs = np.random.default_rng(2)
        df: DataFrame = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
        df['B'] = to_datetime(rs.integers(812419200000000000,
                                            819331200000000000, size=10, dtype=np.int64))
        df['C'] = rs.integers(0, 4, 10)
        xf, yf = (20, 18)
        xrot, yrot = (30, 40)
        axes = _grouped_hist(df.A, by=df.C, cumulative=True, bins=4, xlabelsize=xf,
                               xrot=xrot, ylabelsize=yf, yrot=yrot, density=True)
        for ax in axes.ravel():
            rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
            height = rects[-1].get_height()
            tm.assert_almost_equal(height, 1.0)
        _check_ticks_props(axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)

    def test_grouped_hist_legacy_grouped_hist(self) -> None:
        rs = np.random.default_rng(2)
        df: DataFrame = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
        df['B'] = to_datetime(rs.integers(812419200000000000,
                                            819331200000000000, size=10, dtype=np.int64))
        df['C'] = rs.integers(0, 4, 10)
        df['D'] = ['X'] * 10
        axes = _grouped_hist(df.A, by=df.C, log=True)
        _check_ax_scales(axes, yaxis='log')

    def test_grouped_hist_legacy_external_err(self) -> None:
        rs = np.random.default_rng(2)
        df: DataFrame = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
        df['B'] = to_datetime(rs.integers(812419200000000000,
                                            819331200000000000, size=10, dtype=np.int64))
        df['C'] = rs.integers(0, 4, 10)
        df['D'] = ['X'] * 10
        with tm.external_error_raised(AttributeError):
            _grouped_hist(df.A, by=df.C, foo='bar')

    def test_grouped_hist_legacy_figsize_err(self) -> None:
        rs = np.random.default_rng(2)
        df: DataFrame = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
        df['B'] = to_datetime(rs.integers(812419200000000000,
                                            819331200000000000, size=10, dtype=np.int64))
        df['C'] = rs.integers(0, 4, 10)
        df['D'] = ['X'] * 10
        msg: str = 'Specify figure size by tuple instead'
        with pytest.raises(ValueError, match=msg):
            df.hist(by='C', figsize='default')

    def test_grouped_hist_legacy2(self) -> None:
        n: int = 10
        weight: Series = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height: Series = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender_int: np.ndarray = np.random.default_rng(2).choice([0, 1], size=n)
        df_int: DataFrame = DataFrame({'height': height, 'weight': weight, 'gender': gender_int})
        gb = df_int.groupby('gender')
        axes = gb.hist()
        assert len(axes) == 2
        assert len(mpl.pyplot.get_fignums()) == 2

    @pytest.mark.slow
    @pytest.mark.parametrize('msg, plot_col, by_col, layout',
                             [['Layout of 1x1 must be larger than required size 2', 'weight', 'gender', (1, 1)],
                              ['Layout of 1x3 must be larger than required size 4', 'height', 'category', (1, 3)],
                              ['At least one dimension of layout must be positive', 'height', 'category', (-1, -1)]])
    def test_grouped_hist_layout_error(self, hist_df: DataFrame, msg: str,
                                       plot_col: Union[str, List[str]],
                                       by_col: str, layout: Tuple[int, int]) -> None:
        df: DataFrame = hist_df
        with pytest.raises(ValueError, match=msg):
            df.hist(column=plot_col, by=getattr(df, by_col), layout=layout)

    @pytest.mark.slow
    def test_grouped_hist_layout_warning(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, column='height', by=df.gender, layout=(2, 1))
        _check_axes_shape(axes, axes_num=2, layout=(2, 1))

    @pytest.mark.slow
    @pytest.mark.parametrize('layout, check_layout, figsize',
                             [((4, 1), (4, 1), None),
                              ((-1, 1), (4, 1), None),
                              ((4, 2), (4, 2), (12, 8))])
    def test_grouped_hist_layout_figsize(self, hist_df: DataFrame, layout: Tuple[int, int],
                                         check_layout: Tuple[int, int],
                                         figsize: Optional[Tuple[int, int]]) -> None:
        df: DataFrame = hist_df
        axes = df.hist(column='height', by=df.category, layout=layout, figsize=figsize)
        _check_axes_shape(axes, axes_num=4, layout=check_layout, figsize=figsize)

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs, axes_num, layout',
                             [[{'by': 'gender', 'layout': (3, 5)}, 2, (3, 5)],
                              [{'column': ['height', 'weight', 'category']}, 3, (2, 2)]])
    def test_grouped_hist_layout_axes(self, hist_df: DataFrame, kwargs: Dict[str, Any],
                                      axes_num: int, layout: Tuple[int, int]) -> None:
        df: DataFrame = hist_df
        axes = df.hist(**kwargs)
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    def test_grouped_hist_multiple_axes(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        fig, axes = mpl.pyplot.subplots(2, 3)
        returned = df.hist(column=['height', 'weight', 'category'], ax=axes[0])
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[0])
        assert returned[0].figure is fig

    def test_grouped_hist_multiple_axes_no_cols(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        fig, axes = mpl.pyplot.subplots(2, 3)
        returned = df.hist(by='classroom', ax=axes[1])
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[1])
        assert returned[0].figure is fig

    def test_grouped_hist_multiple_axes_error(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        fig, axes = mpl.pyplot.subplots(2, 3)
        msg: str = 'The number of passed axes must be 1, the same as the output plot'
        with pytest.raises(ValueError, match=msg):
            df.hist(column='height', ax=axes)

    def test_axis_share_x(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        ax1, ax2 = df.hist(column='height', by=df.gender, sharex=True)
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)

    def test_axis_share_y(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        ax1, ax2 = df.hist(column='height', by=df.gender, sharey=True)
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)

    def test_axis_share_xy(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        ax1, ax2 = df.hist(column='height', by=df.gender, sharex=True, sharey=True)
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    @pytest.mark.parametrize('histtype, expected', [('bar', True), ('barstacked', True), ('step', False), ('stepfilled', True)])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).integers(1, 10, size=(10, 2)), columns=['a', 'b'])
        ax = df.hist(by='a', histtype=histtype)
        _check_patches_all_filled(ax, filled=expected)
