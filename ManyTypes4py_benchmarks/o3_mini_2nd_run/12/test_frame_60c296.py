from datetime import date, datetime
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, PeriodIndex, Series, bdate_range, date_range, option_context, plotting
import pandas._testing as tm
from pandas.tests.plotting.common import (_check_ax_scales, _check_axes_shape, _check_box_return_type, 
                                            _check_colors, _check_data, _check_grid_settings, _check_has_errorbars, 
                                            _check_legend_labels, _check_plot_works, _check_text_labels, _check_ticks_props, 
                                            _check_visible, get_y_axis)
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Union, Optional

class TestDataFramePlots:

    @pytest.mark.slow
    def test_plot(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        _check_plot_works(df.plot, grid=False)

    @pytest.mark.slow
    def test_plot_subplots(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        axes: Any = _check_plot_works(df.plot, default_axes=True, subplots=True)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    @pytest.mark.slow
    def test_plot_subplots_negative_layout(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        axes: Any = _check_plot_works(df.plot, default_axes=True, subplots=True, layout=(-1, 2))
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    @pytest.mark.slow
    def test_plot_subplots_use_index(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        axes: Any = _check_plot_works(df.plot, default_axes=True, subplots=True, use_index=False)
        _check_ticks_props(axes, xrot=0)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    @pytest.mark.slow
    def test_plot_invalid_arg(self) -> None:
        df: DataFrame = DataFrame({'x': [1, 2], 'y': [3, 4]})
        msg: str = "'Line2D' object has no property 'blarg'"
        with pytest.raises(AttributeError, match=msg):
            df.plot.line(blarg=True)

    @pytest.mark.slow
    def test_plot_tick_props(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        ax: Any = _check_plot_works(df.plot, use_index=True)
        _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [
        {'yticks': [1, 5, 10]},
        {'xticks': [1, 5, 10]},
        {'ylim': (-100, 100), 'xlim': (-100, 100)},
        {'default_axes': True, 'subplots': True, 'title': 'blah'}
    ])
    def test_plot_other_args(self, kwargs: Dict[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        _check_plot_works(df.plot, **kwargs)

    @pytest.mark.slow
    def test_plot_visible_ax(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        axes: Any = df.plot(subplots=True, title='blah')
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        for ax in axes[:2]:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible([ax.xaxis.get_label()], visible=False)
        for ax in [axes[2]]:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible([ax.xaxis.get_label()])
            _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    def test_plot_title(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        _check_plot_works(df.plot, title='blah')

    @pytest.mark.slow
    def test_plot_multiindex(self) -> None:
        tuples: Any = zip(string.ascii_letters[:10], range(10))
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)), index=MultiIndex.from_tuples(list(tuples)))
        ax: Any = _check_plot_works(df.plot, use_index=True)
        _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    def test_plot_multiindex_unicode(self) -> None:
        index: MultiIndex = MultiIndex.from_tuples(
            [('α', 0), ('α', 1), ('β', 2), ('β', 3), ('γ', 4), ('γ', 5), ('δ', 6), ('δ', 7)],
            names=['i0', 'i1']
        )
        columns: MultiIndex = MultiIndex.from_tuples(
            [('bar', 'Δ'), ('bar', 'Ε')],
            names=['c0', 'c1']
        )
        df: DataFrame = DataFrame(np.random.default_rng(2).integers(0, 10, (8, 2)), columns=columns, index=index)
        _check_plot_works(df.plot, title='Σ')

    @pytest.mark.slow
    @pytest.mark.parametrize('layout', [None, (-1, 1)])
    def test_plot_single_column_bar(self, layout: Optional[Union[Tuple[int, int], None]]) -> None:
        df: DataFrame = DataFrame({'x': np.random.default_rng(2).random(10)})
        axes: Any = _check_plot_works(df.plot.bar, subplots=True, layout=layout)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_plot_passed_ax(self) -> None:
        df: DataFrame = DataFrame({'x': np.random.default_rng(2).random(10)})
        _, ax = mpl.pyplot.subplots()
        axes: Any = df.plot.bar(subplots=True, ax=ax)
        assert len(axes) == 1
        result: Any = ax.axes
        assert result is axes[0]

    @pytest.mark.parametrize('cols, x, y', [
        [list('ABCDE'), 'A', 'B'],
        [['A', 'B'], 'A', 'B'],
        [['C', 'A'], 'C', 'A'],
        [['A', 'C'], 'A', 'C'],
        [['B', 'C'], 'B', 'C'],
        [['A', 'D'], 'A', 'D'],
        [['A', 'E'], 'A', 'E']
    ])
    def test_nullable_int_plot(self, cols: List[Any], x: Any, y: Any) -> None:
        dates: List[Optional[str]] = ['2008', '2009', None, '2011', '2012']
        df: DataFrame = DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 2, 3, 4, 5],
            'C': np.array([7, 5, np.nan, 3, 2], dtype=object),
            'D': pd.to_datetime(dates, format='%Y').view('i8'),
            'E': pd.to_datetime(dates, format='%Y', utc=True).view('i8')
        })
        _check_plot_works(df[cols].plot, x=x, y=y)

    @pytest.mark.slow
    @pytest.mark.parametrize('plot', ['line', 'bar', 'hist', 'pie'])
    def test_integer_array_plot_series(self, plot: str) -> None:
        arr = pd.array([1, 2, 3, 4], dtype='UInt32')
        s: Series = Series(arr)
        _check_plot_works(getattr(s.plot, plot))

    @pytest.mark.slow
    @pytest.mark.parametrize('plot, kwargs', [
        ['line', {}],
        ['bar', {}],
        ['hist', {}],
        ['pie', {'y': 'y'}],
        ['scatter', {'x': 'x', 'y': 'y'}],
        ['hexbin', {'x': 'x', 'y': 'y'}]
    ])
    def test_integer_array_plot_df(self, plot: str, kwargs: Dict[str, Any]) -> None:
        arr = pd.array([1, 2, 3, 4], dtype='UInt32')
        df: DataFrame = DataFrame({'x': arr, 'y': arr})
        _check_plot_works(getattr(df.plot, plot), **kwargs)

    def test_nonnumeric_exclude(self) -> None:
        df: DataFrame = DataFrame({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]})
        ax: Any = df.plot()
        assert len(ax.get_lines()) == 1

    def test_implicit_label(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
        ax: Any = df.plot(x='a', y='b')
        _check_text_labels(ax.xaxis.get_label(), 'a')

    def test_donot_overwrite_index_name(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((2, 2)), columns=['a', 'b'])
        df.index.name = 'NAME'
        df.plot(y='b', label='LABEL')
        assert df.index.name == 'NAME'

    def test_plot_xy(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=5, freq='B'))
        _check_data(df.plot(x=0, y=1), df.set_index('A')['B'].plot())
        _check_data(df.plot(x=0), df.set_index('A').plot())
        _check_data(df.plot(y=0), df.B.plot())
        _check_data(df.plot(x='A', y='B'), df.set_index('A').B.plot())
        _check_data(df.plot(x='A'), df.set_index('A').plot())
        _check_data(df.plot(y='B'), df.B.plot())

    def test_plot_xy_int_cols(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=5, freq='B'))
        df.columns = np.arange(1, len(df.columns) + 1)
        _check_data(df.plot(x=1, y=2), df.set_index(1)[2].plot())
        _check_data(df.plot(x=1), df.set_index(1).plot())
        _check_data(df.plot(y=1), df[1].plot())

    def test_plot_xy_figsize_and_title(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=5, freq='B'))
        ax: Any = df.plot(x=1, y=2, title='Test', figsize=(16, 8))
        _check_text_labels(ax.title, 'Test')
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16.0, 8.0))

    @pytest.mark.parametrize('input_log, expected_log', [(True, 'log'), ('sym', 'symlog')])
    def test_logscales(self, input_log: Union[bool, str], expected_log: str) -> None:
        df: DataFrame = DataFrame({'a': np.arange(100)}, index=np.arange(100))
        ax: Any = df.plot(logy=input_log)
        _check_ax_scales(ax, yaxis=expected_log)
        assert ax.get_yscale() == expected_log
        ax = df.plot(logx=input_log)
        _check_ax_scales(ax, xaxis=expected_log)
        assert ax.get_xscale() == expected_log
        ax = df.plot(loglog=input_log)
        _check_ax_scales(ax, xaxis=expected_log, yaxis=expected_log)
        assert ax.get_xscale() == expected_log
        assert ax.get_yscale() == expected_log

    @pytest.mark.parametrize('input_param', ['logx', 'logy', 'loglog'])
    def test_invalid_logscale(self, input_param: str) -> None:
        df: DataFrame = DataFrame({'a': np.arange(100)}, index=np.arange(100))
        msg: str = f"keyword '{input_param}' should be bool, None, or 'sym', not 'sm'"
        with pytest.raises(ValueError, match=msg):
            df.plot(**{input_param: 'sm'})
        msg2: str = f"PiePlot ignores the '{input_param}' keyword"
        with tm.assert_produces_warning(UserWarning, match=msg2):
            df.plot.pie(subplots=True, **{input_param: True})

    def test_xcompat(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        ax: Any = df.plot(x_compat=True)
        lines: List[Any] = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        plotting.plot_params['xaxis.compat'] = True
        ax: Any = df.plot()
        lines: List[Any] = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params_x_compat(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        plotting.plot_params['x_compat'] = False
        ax: Any = df.plot()
        lines: List[Any] = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        msg: str = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)

    def test_xcompat_plot_params_context_manager(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        with plotting.plot_params.use('x_compat', True):
            ax: Any = df.plot()
            lines: List[Any] = ax.get_lines()
            assert not isinstance(lines[0].get_xdata(), PeriodIndex)
            _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_period(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        ax: Any = df.plot()
        lines: List[Any] = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        msg: str = 'PeriodDtype\\[B\\] is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)
        _check_ticks_props(ax, xrot=0)

    def test_period_compat(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((21, 2)),
                                  index=bdate_range(datetime(2000, 1, 1), datetime(2000, 1, 31)),
                                  columns=['a', 'b'])
        df.plot()
        mpl.pyplot.axhline(y=0)

    @pytest.mark.parametrize('index_dtype', [np.int64, np.float64])
    def test_unsorted_index(self, index_dtype: Any) -> None:
        df: DataFrame = DataFrame({'y': np.arange(100)}, index=Index(np.arange(99, -1, -1), dtype=index_dtype), dtype=np.int64)
        ax: Any = df.plot()
        lines: Any = ax.get_lines()[0]
        rs: Series = Series(lines.get_xydata()[:, 1], index=lines.get_xydata()[:, 0], dtype=np.int64, name='y')
        tm.assert_series_equal(rs, df.y, check_index_type=False)

    @pytest.mark.parametrize('df', [
        DataFrame({'y': [0.0, 1.0, 2.0, 3.0]}, index=[1.0, 0.0, 3.0, 2.0]),
        DataFrame({'y': [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0]}, index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0])
    ])
    def test_unsorted_index_lims(self, df: DataFrame) -> None:
        ax: Any = df.plot()
        xmin, xmax = ax.get_xlim()
        lines: List[Any] = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        assert xmax >= np.nanmax(lines[0].get_data()[0])

    def test_unsorted_index_lims_x_y(self) -> None:
        df: DataFrame = DataFrame({'y': [0.0, 1.0, 2.0, 3.0], 'z': [91.0, 90.0, 93.0, 92.0]})
        ax: Any = df.plot(x='z', y='y')
        xmin, xmax = ax.get_xlim()
        lines: List[Any] = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        assert xmax >= np.nanmax(lines[0].get_data()[0])

    def test_negative_log(self) -> None:
        df: DataFrame = -DataFrame(np.random.default_rng(2).random((6, 4)),
                                   index=list(string.ascii_letters[:6]),
                                   columns=['x', 'y', 'z', 'four'])
        msg: str = 'Log-y scales are not supported in area plot'
        with pytest.raises(ValueError, match=msg):
            df.plot.area(logy=True)
        with pytest.raises(ValueError, match=msg):
            df.plot.area(loglog=True)

    def _compare_stacked_y_cood(self, normal_lines: List[Any], stacked_lines: List[Any]) -> None:
        base: np.ndarray = np.zeros(len(normal_lines[0].get_data()[1]))
        for nl, sl in zip(normal_lines, stacked_lines):
            base += nl.get_data()[1]
            sy: np.ndarray = sl.get_data()[1]
            tm.assert_numpy_array_equal(base, sy)

    @pytest.mark.parametrize('kind', ['line', 'area'])
    @pytest.mark.parametrize('mult', [1, -1])
    def test_line_area_stacked(self, kind: str, mult: int) -> None:
        df: DataFrame = mult * DataFrame(np.random.default_rng(2).random((6, 4)), columns=['w', 'x', 'y', 'z'])
        ax1: Any = _check_plot_works(df.plot, kind=kind, stacked=False)
        ax2: Any = _check_plot_works(df.plot, kind=kind, stacked=True)
        self._compare_stacked_y_cood(ax1.lines, ax2.lines)

    @pytest.mark.parametrize('kind', ['line', 'area'])
    def test_line_area_stacked_sep_df(self, kind: str) -> None:
        sep_df: DataFrame = DataFrame({
            'w': np.random.default_rng(2).random(6),
            'x': np.random.default_rng(2).random(6),
            'y': -np.random.default_rng(2).random(6),
            'z': -np.random.default_rng(2).random(6)
        })
        ax1: Any = _check_plot_works(sep_df.plot, kind=kind, stacked=False)
        ax2: Any = _check_plot_works(sep_df.plot, kind=kind, stacked=True)
        self._compare_stacked_y_cood(ax1.lines[:2], ax2.lines[:2])
        self._compare_stacked_y_cood(ax1.lines[2:], ax2.lines[2:])

    def test_line_area_stacked_mixed(self) -> None:
        mixed_df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)),
                                        index=list(string.ascii_letters[:6]),
                                        columns=['w', 'x', 'y', 'z'])
        _check_plot_works(mixed_df.plot, stacked=False)
        msg: str = ("When stacked is True, each column must be either all positive or all negative. "
                    "Column 'w' contains both positive and negative values")
        with pytest.raises(ValueError, match=msg):
            mixed_df.plot(stacked=True)

    @pytest.mark.parametrize('kind', ['line', 'area'])
    def test_line_area_stacked_positive_idx(self, kind: str) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((6, 4)), columns=['w', 'x', 'y', 'z'])
        df2: DataFrame = df.set_index(df.index + 1)
        _check_plot_works(df2.plot, kind=kind, logx=True, stacked=True)

    @pytest.mark.parametrize('idx', [range(4), date_range('2023-01-1', freq='D', periods=4)])
    def test_line_area_nan_df(self, idx: Union[range, pd.DatetimeIndex]) -> None:
        values1: List[Optional[float]] = [1, 2, np.nan, 3]
        values2: List[Optional[float]] = [3, np.nan, 2, 1]
        df: DataFrame = DataFrame({'a': values1, 'b': values2}, index=idx)
        ax: Any = _check_plot_works(df.plot)
        masked1: Any = ax.lines[0].get_ydata()
        masked2: Any = ax.lines[1].get_ydata()
        exp: np.ndarray = np.array([1, 2, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked1.data, 2), exp)
        exp = np.array([3, 2, 1], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked2.data, 1), exp)
        tm.assert_numpy_array_equal(masked1.mask, np.array([False, False, True, False]))
        tm.assert_numpy_array_equal(masked2.mask, np.array([False, True, False, False]))

    @pytest.mark.parametrize('idx', [range(4), date_range('2023-01-1', freq='D', periods=4)])
    def test_line_area_nan_df_stacked(self, idx: Union[range, pd.DatetimeIndex]) -> None:
        values1: List[Optional[float]] = [1, 2, np.nan, 3]
        values2: List[Optional[float]] = [3, np.nan, 2, 1]
        df: DataFrame = DataFrame({'a': values1, 'b': values2}, index=idx)
        expected1: np.ndarray = np.array([1, 2, 0, 3], dtype=np.float64)
        expected2: np.ndarray = np.array([3, 0, 2, 1], dtype=np.float64)
        ax: Any = _check_plot_works(df.plot, stacked=True)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)

    @pytest.mark.parametrize('idx', [range(4), date_range('2023-01-1', freq='D', periods=4)])
    @pytest.mark.parametrize('kwargs', [{}, {'stacked': False}])
    def test_line_area_nan_df_stacked_area(self, idx: Union[range, pd.DatetimeIndex], kwargs: Dict[str, Any]) -> None:
        values1: List[Optional[float]] = [1, 2, np.nan, 3]
        values2: List[Optional[float]] = [3, np.nan, 2, 1]
        df: DataFrame = DataFrame({'a': values1, 'b': values2}, index=idx)
        expected1: np.ndarray = np.array([1, 2, 0, 3], dtype=np.float64)
        expected2: np.ndarray = np.array([3, 0, 2, 1], dtype=np.float64)
        ax: Any = _check_plot_works(df.plot.area, **kwargs)
        if kwargs:
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)
        else:
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)
        ax = _check_plot_works(df.plot.area, stacked=False)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)

    @pytest.mark.parametrize('kwargs', [{}, {'secondary_y': True}])
    def test_line_lim(self, kwargs: Dict[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((6, 3)), columns=['x', 'y', 'z'])
        ax: Any = df.plot(**kwargs)
        xmin, xmax = ax.get_xlim()
        lines: List[Any] = ax.get_lines()
        assert xmin <= lines[0].get_data()[0][0]
        assert xmax >= lines[0].get_data()[0][-1]

    def test_line_lim_subplots(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((6, 3)), columns=['x', 'y', 'z'])
        axes: Any = df.plot(secondary_y=True, subplots=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        for ax in axes:
            assert hasattr(ax, 'left_ax')
            assert not hasattr(ax, 'right_ax')
            xmin, xmax = ax.get_xlim()
            lines: List[Any] = ax.get_lines()
            assert xmin <= lines[0].get_data()[0][0]
            assert xmax >= lines[0].get_data()[0][-1]

    @pytest.mark.xfail(strict=False, reason='2020-12-01 this has been failing periodically on the ymin==0 assertion for a week or so.')
    @pytest.mark.parametrize('stacked', [True, False])
    def test_area_lim(self, stacked: bool) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((6, 4)), columns=['x', 'y', 'z', 'four'])
        neg_df: DataFrame = -df
        ax: Any = _check_plot_works(df.plot.area, stacked=stacked)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lines: List[Any] = ax.get_lines()
        assert xmin <= lines[0].get_data()[0][0]
        assert xmax >= lines[0].get_data()[0][-1]
        assert ymin == 0
        ax = _check_plot_works(neg_df.plot.area, stacked=stacked)
        ymin, ymax = ax.get_ylim()
        assert ymax == 0

    def test_area_sharey_dont_overwrite(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((4, 2)), columns=['x', 'y'])
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)
        df.plot(ax=ax1, kind='area')
        df.plot(ax=ax2, kind='area')
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    @pytest.mark.parametrize('stacked', [True, False])
    def test_bar_linewidth(self, stacked: bool) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax: Any = df.plot.bar(stacked=stacked, linewidth=2)
        for r in ax.patches:
            assert r.get_linewidth() == 2

    def test_bar_linewidth_subplots(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes: Any = df.plot.bar(linewidth=2, subplots=True)
        _check_axes_shape(axes, axes_num=5, layout=(5, 1))
        for ax in axes:
            for r in ax.patches:
                assert r.get_linewidth() == 2

    @pytest.mark.parametrize('meth, dim', [('bar', 'get_width'), ('barh', 'get_height')])
    @pytest.mark.parametrize('stacked', [True, False])
    def test_bar_barwidth(self, meth: str, dim: str, stacked: bool) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        width: float = 0.9
        ax: Any = getattr(df.plot, meth)(stacked=stacked, width=width)
        for r in ax.patches:
            if not stacked:
                assert getattr(r, dim)() == width / len(df.columns)
            else:
                assert getattr(r, dim)() == width

    @pytest.mark.parametrize('meth, dim', [('bar', 'get_width'), ('barh', 'get_height')])
    def test_barh_barwidth_subplots(self, meth: str, dim: str) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        width: float = 0.9
        axes: Any = getattr(df.plot, meth)(width=width, subplots=True)
        for ax in axes:
            for r in ax.patches:
                assert getattr(r, dim)() == width

    def test_bar_bottom_left_bottom(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax: Any = df.plot.bar(stacked=False, bottom=1)
        result: List[float] = [p.get_y() for p in ax.patches]
        assert result == [1] * 25
        ax = df.plot.bar(stacked=True, bottom=[-1, -2, -3, -4, -5])
        result = [p.get_y() for p in ax.patches[:5]]
        assert result == [-1, -2, -3, -4, -5]

    def test_bar_bottom_left_left(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax: Any = df.plot.barh(stacked=False, left=np.array([1, 1, 1, 1, 1]))
        result: List[float] = [p.get_x() for p in ax.patches]
        assert result == [1] * 25
        ax = df.plot.barh(stacked=True, left=[1, 2, 3, 4, 5])
        result = [p.get_x() for p in ax.patches[:5]]
        assert result == [1, 2, 3, 4, 5]

    def test_bar_bottom_left_subplots(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 5)))
        axes: Any = df.plot.bar(subplots=True, bottom=-1)
        for ax in axes:
            result: List[float] = [p.get_y() for p in ax.patches]
            assert result == [-1] * 5
        axes = df.plot.barh(subplots=True, left=np.array([1, 1, 1, 1, 1]))
        for ax in axes:
            result = [p.get_x() for p in ax.patches]
            assert result == [1] * 5

    def test_bar_nan(self) -> None:
        df: DataFrame = DataFrame({'A': [10, np.nan, 20], 'B': [5, 10, 20], 'C': [1, 2, 3]})
        ax: Any = df.plot.bar()
        expected: List[float] = [10, 0, 20, 5, 10, 20, 1, 2, 3]
        result: List[float] = [p.get_height() for p in ax.patches]
        assert result == expected

    def test_bar_nan_stacked(self) -> None:
        df: DataFrame = DataFrame({'A': [10, np.nan, 20], 'B': [5, 10, 20], 'C': [1, 2, 3]})
        ax: Any = df.plot.bar(stacked=True)
        expected: List[float] = [10, 0, 20, 5, 10, 20, 1, 2, 3]
        result: List[float] = [p.get_height() for p in ax.patches]
        assert result == expected
        result = [p.get_y() for p in ax.patches]
        expected = [0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 15.0, 10.0, 40.0]
        assert result == expected

    def test_bar_stacked_label_position_with_zero_height(self) -> None:
        df: DataFrame = DataFrame({'A': [3, 0, 1], 'B': [0, 2, 4], 'C': [5, 0, 2]})
        ax: Any = df.plot.bar(stacked=True)
        ax.bar_label(ax.containers[-1])
        expected: List[float] = [8.0, 2.0, 7.0]
        result: List[float] = [text.xy[1] for text in ax.texts]
        tm.assert_almost_equal(result, expected)
        plt.close('all')

    @pytest.mark.parametrize('idx', [Index, pd.CategoricalIndex])
    def test_bar_categorical(self, idx: Any) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 5)),
                                  index=idx(list('ABCDEF')),
                                  columns=idx(list('abcde')))
        ax: Any = df.plot.bar()
        ticks: np.ndarray = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
        assert ax.get_xlim() == (-0.5, 5.5)
        assert ax.patches[0].get_x() == -0.25
        assert ax.patches[-1].get_x() == 5.15
        ax = df.plot.bar(stacked=True)
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
        assert ax.get_xlim() == (-0.5, 5.5)
        assert ax.patches[0].get_x() == -0.25
        assert ax.patches[-1].get_x() == 4.75

    @pytest.mark.parametrize('x, y', [('x', 'y'), (1, 2)])
    def test_plot_scatter(self, x: Union[str, int], y: Union[str, int]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)),
                                  index=list(string.ascii_letters[:6]),
                                  columns=['x', 'y', 'z', 'four'])
        _check_plot_works(df.plot.scatter, x=x, y=y)

    def test_plot_scatter_error(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)),
                                  index=list(string.ascii_letters[:6]),
                                  columns=['x', 'y', 'z', 'four'])
        msg: str = re.escape("scatter() missing 1 required positional argument: 'y'")
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(x='x')
        msg = re.escape("scatter() missing 1 required positional argument: 'x'")
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(y='y')

    def test_plot_scatter_shape(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)),
                                  index=list(string.ascii_letters[:6]),
                                  columns=['x', 'y', 'z', 'four'])
        axes: Any = df.plot(x='x', y='y', kind='scatter', subplots=True)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_raise_error_on_datetime_time_data(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal(10), columns=['a'])
        df['dtime'] = date_range(start='2014-01-01', freq='h', periods=10).time
        msg: str = "must be a string or a (real )?number, not 'datetime.time'"
        with pytest.raises(TypeError, match=msg):
            df.plot(kind='scatter', x='dtime', y='a')

    @pytest.mark.parametrize('x, y', [('dates', 'vals'), (0, 1)])
    def test_scatterplot_datetime_data(self, x: Union[str, int], y: Union[str, int]) -> None:
        dates: Any = date_range(start=date(2019, 1, 1), periods=12, freq='W')
        vals: np.ndarray = np.random.default_rng(2).normal(0, 1, len(dates))
        df: DataFrame = DataFrame({'dates': dates, 'vals': vals})
        _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
    @pytest.mark.parametrize('x, y', [('a', 'b'), (0, 1)])
    @pytest.mark.parametrize('b_col', [[2, 3, 4], ['a', 'b', 'c']])
    def test_scatterplot_object_data(self, b_col: List[Any], x: Union[str, int], y: Union[str, int], infer_string: bool) -> None:
        with option_context('future.infer_string', infer_string):
            df: DataFrame = DataFrame({'a': ['A', 'B', 'C'], 'b': b_col})
            _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize('ordered', [True, False])
    @pytest.mark.parametrize('categories', (['setosa', 'versicolor', 'virginica'], ['versicolor', 'virginica', 'setosa']))
    def test_scatterplot_color_by_categorical(self, ordered: bool, categories: List[str]) -> None:
        df: DataFrame = DataFrame([[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
                                  columns=['length', 'width'])
        df['species'] = pd.Categorical(['setosa', 'setosa', 'virginica', 'virginica', 'versicolor'],
                                       ordered=ordered, categories=categories)
        ax: Any = df.plot.scatter(x=0, y=1, c='species')
        colorbar_collection, = ax.collections
        colorbar = colorbar_collection.colorbar
        expected_ticks: np.ndarray = np.array([0.5, 1.5, 2.5])
        result_ticks: np.ndarray = colorbar.get_ticks()
        tm.assert_numpy_array_equal(result_ticks, expected_ticks)
        expected_boundaries: np.ndarray = np.array([0.0, 1.0, 2.0, 3.0])
        result_boundaries: np.ndarray = colorbar._boundaries
        tm.assert_numpy_array_equal(result_boundaries, expected_boundaries)
        expected_yticklabels: List[str] = categories
        result_yticklabels: List[str] = [i.get_text() for i in colorbar.ax.get_ymajorticklabels()]
        assert all((i == j for i, j in zip(result_yticklabels, expected_yticklabels)))

    @pytest.mark.parametrize('x, y', [('x', 'y'), ('y', 'x'), ('y', 'y')])
    def test_plot_scatter_with_categorical_data(self, x: Union[str, int], y: Union[str, int]) -> None:
        df: DataFrame = DataFrame({'x': [1, 2, 3, 4], 'y': pd.Categorical(['a', 'b', 'a', 'c'])})
        _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize('x, y, c', [('x', 'y', 'z'), (0, 1, 2)])
    def test_plot_scatter_with_c(self, x: Union[str, int], y: Union[str, int], c: Union[str, int]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).integers(low=0, high=100, size=(6, 4)),
                                  index=list(string.ascii_letters[:6]), columns=['x', 'y', 'z', 'four'])
        ax: Any = df.plot.scatter(x=x, y=y, c=c)
        assert ax.collections[0].cmap.name == 'Greys'
        assert ax.collections[0].colorbar.ax.get_ylabel() == 'z'

    def test_plot_scatter_with_c_props(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).integers(low=0, high=100, size=(6, 4)),
                                  index=list(string.ascii_letters[:6]), columns=['x', 'y', 'z', 'four'])
        cm: str = 'cubehelix'
        ax: Any = df.plot.scatter(x='x', y='y', c='z', colormap=cm)
        assert ax.collections[0].cmap.name == cm
        ax = df.plot.scatter(x='x', y='y', c='z', colorbar=False)
        assert ax.collections[0].colorbar is None
        ax = df.plot.scatter(x=0, y=1, c='red')
        assert ax.collections[0].colorbar is None
        _check_colors(ax.collections, facecolors=['r'])

    def test_plot_scatter_with_c_array(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4]})
        red_rgba: List[float] = [1.0, 0.0, 0.0, 1.0]
        green_rgba: List[float] = [0.0, 1.0, 0.0, 1.0]
        rgba_array: np.ndarray = np.array([red_rgba, green_rgba])
        ax: Any = df.plot.scatter(x='A', y='B', c=rgba_array)
        tm.assert_numpy_array_equal(ax.collections[0].get_facecolor(), rgba_array)
        float_array: np.ndarray = np.array([0.0, 1.0])
        df.plot.scatter(x='A', y='B', c=float_array, cmap='spring')

    def test_plot_scatter_with_s(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)) * 100, columns=['a', 'b', 'c'])
        ax: Any = df.plot.scatter(x='a', y='b', s='c')
        tm.assert_numpy_array_equal(df['c'].values, right=ax.collections[0].get_sizes())

    def test_plot_scatter_with_norm(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)) * 100, columns=['a', 'b', 'c'])
        norm: Any = mpl.colors.LogNorm()
        ax: Any = df.plot.scatter(x='a', y='b', c='c', norm=norm)
        assert ax.collections[0].norm is norm

    def test_plot_scatter_without_norm(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)) * 100, columns=['a', 'b', 'c'])
        ax: Any = df.plot.scatter(x='a', y='b', c='c')
        plot_norm: Any = ax.collections[0].norm
        color_min_max: Tuple[float, float] = (df.c.min(), df.c.max())
        default_norm: Any = mpl.colors.Normalize(*color_min_max)
        for value in df.c:
            assert plot_norm(value) == default_norm(value)

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [{}, {'legend': False}, {'default_axes': True, 'subplots': True}, {'stacked': True}])
    def test_plot_bar(self, kwargs: Dict[str, Any]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)),
                                  index=list(string.ascii_letters[:6]),
                                  columns=['one', 'two', 'three', 'four'])
        _check_plot_works(df.plot.bar, **kwargs)

    @pytest.mark.slow
    def test_plot_bar_int_col(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 15)),
                                  index=list(string.ascii_letters[:10]),
                                  columns=range(15))
        _check_plot_works(df.plot.bar)

    @pytest.mark.slow
    def test_plot_bar_ticks(self) -> None:
        df: DataFrame = DataFrame({'a': [0, 1], 'b': [1, 0]})
        ax: Any = _check_plot_works(df.plot.bar)
        _check_ticks_props(ax, xrot=90)
        ax = df.plot.bar(rot=35, fontsize=10)
        _check_ticks_props(ax, xrot=35, xlabelsize=10, ylabelsize=10)

    @pytest.mark.slow
    def test_plot_barh_ticks(self) -> None:
        df: DataFrame = DataFrame({'a': [0, 1], 'b': [1, 0]})
        ax: Any = _check_plot_works(df.plot.barh)
        _check_ticks_props(ax, yrot=0)
        ax = df.plot.barh(rot=55, fontsize=11)
        _check_ticks_props(ax, yrot=55, ylabelsize=11, xlabelsize=11)

    def test_boxplot(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        numeric_cols: Any = df._get_numeric_data().columns
        labels: List[str] = [pprint_thing(c) for c in numeric_cols]
        ax: Any = _check_plot_works(df.plot.box)
        _check_text_labels(ax.get_xticklabels(), labels)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), np.arange(1, len(numeric_cols) + 1))
        assert len(ax.lines) == 7 * len(numeric_cols)

    def test_boxplot_series(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        series: Series = df['height']
        axes: Any = series.plot.box(rot=40)
        _check_ticks_props(axes, xrot=40, yrot=0)
        _check_plot_works(series.plot.box)

    def test_boxplot_series_positions(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        positions: np.ndarray = np.array([1, 6, 7])
        ax: Any = df.plot.box(positions=positions)
        numeric_cols: Any = df._get_numeric_data().columns
        labels: List[str] = [pprint_thing(c) for c in numeric_cols]
        _check_text_labels(ax.get_xticklabels(), labels)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), positions)
        assert len(ax.lines) == 7 * len(numeric_cols)

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    @pytest.mark.xfail(Version(mpl.__version__) >= Version('3.10'), reason='Fails starting with matplotlib 3.10')
    def test_boxplot_vertical(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        numeric_cols: Any = df._get_numeric_data().columns
        labels: List[str] = [pprint_thing(c) for c in numeric_cols]
        kwargs: Dict[str, Any] = {'vert': False} if Version(mpl.__version__) < Version('3.10') else {'orientation': 'horizontal'}
        ax: Any = df.plot.box(rot=50, fontsize=8, **kwargs)
        _check_ticks_props(ax, xrot=0, yrot=50, ylabelsize=8)
        _check_text_labels(ax.get_yticklabels(), labels)
        assert len(ax.lines) == 7 * len(numeric_cols)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.xfail(Version(mpl.__version__) >= Version('3.10'), reason='Fails starting with matplotlib version 3.10')
    def test_boxplot_vertical_subplots(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        numeric_cols: Any = df._get_numeric_data().columns
        labels: List[str] = [pprint_thing(c) for c in numeric_cols]
        kwargs: Dict[str, Any] = {'vert': False} if Version(mpl.__version__) < Version('3.10') else {'orientation': 'horizontal'}
        axes: Any = _check_plot_works(df.plot.box, default_axes=True, subplots=True, logx=True, **kwargs)
        _check_axes_shape(axes, axes_num=3, layout=(1, 3))
        _check_ax_scales(axes, xaxis='log')
        for ax, label in zip(axes, labels):
            _check_text_labels(ax.get_yticklabels(), [label])
            assert len(ax.lines) == 7

    @pytest.mark.filterwarnings('ignore:set_ticklabels:UserWarning')
    @pytest.mark.xfail(Version(mpl.__version__) >= Version('3.10'), reason='Fails starting with matplotlib 3.10')
    def test_boxplot_vertical_positions(self, hist_df: DataFrame) -> None:
        df: DataFrame = hist_df
        numeric_cols: Any = df._get_numeric_data().columns
        labels: List[str] = [pprint_thing(c) for c in numeric_cols]
        positions: np.ndarray = np.array([3, 2, 8])
        kwargs: Dict[str, Any] = {'vert': False} if Version(mpl.__version__) < Version('3.10') else {'orientation': 'horizontal'}
        ax: Any = df.plot.box(positions=positions, **kwargs)
        _check_text_labels(ax.get_yticklabels(), labels)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), positions)
        assert len(ax.lines) == 7 * len(numeric_cols)

    def test_boxplot_return_type_invalid(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)),
                                  index=list(string.ascii_letters[:6]),
                                  columns=['one', 'two', 'three', 'four'])
        msg: str = "return_type must be {None, 'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.plot.box(return_type='not_a_type')

    @pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
    def test_boxplot_return_type_invalid_type(self, return_type: str) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)),
                                  index=list(string.ascii_letters[:6]),
                                  columns=['one', 'two', 'three', 'four'])
        result: Any = df.plot.box(return_type=return_type)
        _check_box_return_type(result, return_type)

    def test_kde_df(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        ax: Any = _check_plot_works(df.plot, kind='kde')
        expected: List[str] = [pprint_thing(c) for c in df.columns]
        _check_legend_labels(ax, labels=expected)
        _check_ticks_props(ax, xrot=0)

    def test_kde_df_rot(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        ax: Any = df.plot(kind='kde', rot=20, fontsize=5)
        _check_ticks_props(ax, xrot=20, xlabelsize=5, ylabelsize=5)

    def test_kde_df_subplots(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        axes: Any = _check_plot_works(df.plot, default_axes=True, kind='kde', subplots=True)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    def test_kde_df_logy(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        axes: Any = df.plot(kind='kde', logy=True, subplots=True)
        _check_ax_scales(axes, yaxis='log')

    def test_kde_missing_vals(self) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame(np.random.default_rng(2).uniform(size=(100, 4)))
        df.loc[0, 0] = np.nan
        _check_plot_works(df.plot, kind='kde')

    def test_hist_df(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((100, 4)))
        ax: Any = _check_plot_works(df.plot.hist)
        expected: List[str] = [pprint_thing(c) for c in df.columns]
        _check_legend_labels(ax, labels=expected)
        axes: Any = _check_plot_works(df.plot.hist, default_axes=True, subplots=True, logy=True)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))
        _check_ax_scales(axes, yaxis='log')

    def test_hist_df_series(self) -> None:
        series: Series = Series(np.random.default_rng(2).random(10))
        axes: Any = series.plot.hist(rot=40)
        _check_ticks_props(axes, xrot=40, yrot=0)

    def test_hist_df_series_cumulative_density(self) -> None:
        series: Series = Series(np.random.default_rng(2).random(10))
        ax: Any = series.plot.hist(cumulative=True, bins=4, density=True)
        rects: List[Any] = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

    def test_hist_df_series_cumulative(self) -> None:
        series: Series = Series(np.random.default_rng(2).random(10))
        ax: Any = series.plot.hist(cumulative=True, bins=4)
        rects: List[Any] = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        tm.assert_almost_equal(rects[-2].get_height(), 10.0)

    def test_hist_df_orientation(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        axes: Any = df.plot.hist(rot=50, fontsize=8, orientation='horizontal')
        _check_ticks_props(axes, xrot=0, yrot=50, ylabelsize=8)

    @pytest.mark.parametrize('weight_shape', [(100,), (100, 2)])
    def test_hist_weights(self, weight_shape: Tuple[int, ...]) -> None:
        weights: np.ndarray = 0.1 * np.ones(shape=weight_shape)
        df: DataFrame = DataFrame(dict(zip(['A', 'B'], np.random.default_rng(2).standard_normal((2, 100)))))
        ax1: Any = _check_plot_works(df.plot, kind='hist', weights=weights)
        ax2: Any = _check_plot_works(df.plot, kind='hist')
        patch_height_with_weights: List[float] = [patch.get_height() for patch in ax1.patches]
        expected_patch_height: List[float] = [0.1 * patch.get_height() for patch in ax2.patches]
        tm.assert_almost_equal(patch_height_with_weights, expected_patch_height)

    def _check_box_coord(self, patches: List[Any], 
                         expected_y: Optional[np.ndarray] = None, 
                         expected_h: Optional[np.ndarray] = None, 
                         expected_x: Optional[np.ndarray] = None, 
                         expected_w: Optional[np.ndarray] = None) -> None:
        result_y: np.ndarray = np.array([p.get_y() for p in patches])
        result_height: np.ndarray = np.array([p.get_height() for p in patches])
        result_x: np.ndarray = np.array([p.get_x() for p in patches])
        result_width: np.ndarray = np.array([p.get_width() for p in patches])
        if expected_y is not None:
            tm.assert_numpy_array_equal(result_y, expected_y, check_dtype=False)
        if expected_h is not None:
            tm.assert_numpy_array_equal(result_height, expected_h, check_dtype=False)
        if expected_x is not None:
            tm.assert_numpy_array_equal(result_x, expected_x, check_dtype=False)
        if expected_w is not None:
            tm.assert_numpy_array_equal(result_width, expected_w, check_dtype=False)

    @pytest.mark.parametrize('data', [
        {'A': np.repeat(np.array([1, 2, 3, 4, 5]), np.array([10, 9, 8, 7, 6])),
         'B': np.repeat(np.array([1, 2, 3, 4, 5]), np.array([8, 8, 8, 8, 8])),
         'C': np.repeat(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]))},
        {'A': np.repeat(np.array([np.nan, 1, 2, 3, 4, 5]), np.array([3, 10, 9, 8, 7, 6])),
         'B': np.repeat(np.array([1, np.nan, 2, 3, 4, 5]), np.array([8, 3, 8, 8, 8, 8])),
         'C': np.repeat(np.array([1, 2, 3, np.nan, 4, 5]), np.array([6, 7, 8, 3, 9, 10]))}
    ])
    def test_hist_df_coord(self, data: Dict[str, Any]) -> None:
        df: DataFrame = DataFrame(data)
        ax: Any = df.plot.hist(bins=5)
        self._check_box_coord(ax.patches[:5], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([10, 9, 8, 7, 6]))
        self._check_box_coord(ax.patches[5:10], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([8, 8, 8, 8, 8]))
        self._check_box_coord(ax.patches[10:], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([6, 7, 8, 9, 10]))
        ax = df.plot.hist(bins=5, stacked=True)
        self._check_box_coord(ax.patches[:5], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([10, 9, 8, 7, 6]))
        self._check_box_coord(ax.patches[5:10], expected_y=np.array([10, 9, 8, 7, 6]), expected_h=np.array([8, 8, 8, 8, 8]))
        self._check_box_coord(ax.patches[10:], expected_y=np.array([18, 17, 16, 15, 14]), expected_h=np.array([6, 7, 8, 9, 10]))
        axes: Any = df.plot.hist(bins=5, stacked=True, subplots=True)
        self._check_box_coord(axes[0].patches, expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([10, 9, 8, 7, 6]))
        self._check_box_coord(axes[1].patches, expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([8, 8, 8, 8, 8]))
        self._check_box_coord(axes[2].patches, expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([6, 7, 8, 9, 10]))
        ax = df.plot.hist(bins=5, orientation='horizontal')
        self._check_box_coord(ax.patches[:5], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([10, 9, 8, 7, 6]))
        self._check_box_coord(ax.patches[5:10], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([8, 8, 8, 8, 8]))
        self._check_box_coord(ax.patches[10:], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([6, 7, 8, 9, 10]))
        ax = df.plot.hist(bins=5, stacked=True, orientation='horizontal')
        self._check_box_coord(ax.patches[:5], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([10, 9, 8, 7, 6]))
        self._check_box_coord(ax.patches[5:10], expected_x=np.array([10, 9, 8, 7, 6]), expected_w=np.array([8, 8, 8, 8, 8]))
        self._check_box_coord(ax.patches[10:], expected_x=np.array([18, 17, 16, 15, 14]), expected_w=np.array([6, 7, 8, 9, 10]))
        axes = df.plot.hist(bins=5, stacked=True, subplots=True, orientation='horizontal')
        self._check_box_coord(axes[0].patches, expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([10, 9, 8, 7, 6]))
        self._check_box_coord(axes[1].patches, expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([8, 8, 8, 8, 8]))
        self._check_box_coord(axes[2].patches, expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([6, 7, 8, 9, 10]))

    def test_plot_int_columns(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((100, 4))).cumsum()
        _check_plot_works(df.plot, legend=True)

    @pytest.mark.parametrize('markers', [{0: '^', 1: '+', 2: 'o'}, {0: '^', 1: '+'}, ['^', '+', 'o'], ['^', '+']])
    def test_style_by_column(self, markers: Union[Dict[int, str], List[str]]) -> None:
        fig = plt.gcf()
        fig.clf()
        fig.add_subplot(111)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
        ax: Any = df.plot(style=markers)
        for idx, line in enumerate(ax.get_lines()[:len(markers)]):
            assert line.get_marker() == markers[idx]

    def test_line_label_none(self) -> None:
        s: Series = Series([1, 2])
        ax: Any = s.plot()
        assert ax.get_legend() is None
        ax = s.plot(legend=True)
        assert ax.get_legend().get_texts()[0].get_text() == ''

    @pytest.mark.parametrize('props, expected', [('boxprops', 'boxes'),
                                                   ('whiskerprops', 'whiskers'),
                                                   ('capprops', 'caps'),
                                                   ('medianprops', 'medians')])
    def test_specified_props_kwd_plot_box(self, props: str, expected: str) -> None:
        df: DataFrame = DataFrame({k: np.random.default_rng(2).random(100) for k in 'ABC'})
        kwd: Dict[str, Any] = {props: {'color': 'C1'}}
        result: Any = df.plot.box(return_type='dict', **kwd)
        assert result[expected][0].get_color() == 'C1'

    def test_unordered_ts(self) -> None:
        index: List[date] = [date(2012, 10, 1), date(2012, 9, 1), date(2012, 8, 1)]
        values: List[float] = [3.0, 2.0, 1.0]
        df: DataFrame = DataFrame(np.array(values), index=index, columns=['test'])
        ax: Any = df.plot()
        xticks: Any = ax.lines[0].get_xdata()
        tm.assert_numpy_array_equal(xticks, np.array(index, dtype=object))
        ydata: Any = ax.lines[0].get_ydata()
        tm.assert_numpy_array_equal(ydata, np.array(values))
        xticks = ax.xaxis.get_ticklabels()
        xlocs: List[float] = [x.get_position()[0] for x in xticks]
        assert Index(xlocs).is_monotonic_increasing
        xlabels: List[str] = [x.get_text() for x in xticks]
        assert pd.to_datetime(xlabels, format='%Y-%m-%d').is_monotonic_increasing

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_kind_both_ways(self, kind: str) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame({'x': [1, 2, 3]})
        df.plot(kind=kind)
        getattr(df.plot, kind)()

    @pytest.mark.parametrize('kind', ['scatter', 'hexbin'])
    def test_kind_both_ways_x_y(self, kind: str) -> None:
        pytest.importorskip('scipy')
        df: DataFrame = DataFrame({'x': [1, 2, 3]})
        df.plot('x', 'x', kind=kind)
        getattr(df.plot, kind)('x', 'x')

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_all_invalid_plot_data(self, kind: str) -> None:
        df: DataFrame = DataFrame(list('abcd'))
        msg: str = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            df.plot(kind=kind)

    @pytest.mark.parametrize('kind', list(plotting.PlotAccessor._common_kinds) + ['area'])
    def test_partially_invalid_plot_data_numeric(self, kind: str) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), dtype=object)
        df[np.random.default_rng(2).random(df.shape[0]) > 0.5] = 'a'
        msg: str = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            df.plot(kind=kind)

    def test_invalid_kind(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        msg: str = 'invalid_plot_kind is not a valid plot kind'
        with pytest.raises(ValueError, match=msg):
            df.plot(kind='invalid_plot_kind')

    @pytest.mark.parametrize('x,y,lbl', [
        (['B', 'C'], 'A', 'a'),
        (['A'], ['B', 'C'], ['b', 'c'])
    ])
    def test_invalid_xy_args(self, x: Any, y: Any, lbl: Any) -> None:
        df: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        with pytest.raises(ValueError, match='x must be a label or position'):
            df.plot(x=x, y=y, label=lbl)

    def test_bad_label(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        msg: str = 'label should be list-like and same length as y'
        with pytest.raises(ValueError, match=msg):
            df.plot(x='A', y=['B', 'C'], label='bad_label')

    @pytest.mark.parametrize('x,y', [('A', 'B'), (['A'], 'B')])
    def test_invalid_xy_args_dup_cols(self, x: Any, y: Any) -> None:
        df: DataFrame = DataFrame([[1, 3, 5], [2, 4, 6]], columns=list('AAB'))
        with pytest.raises(ValueError, match='x must be a label or position'):
            df.plot(x=x, y=y)

    @pytest.mark.parametrize('x,y,lbl,colors', [
        ('A', ['B'], ['b'], ['red']),
        ('A', ['B', 'C'], ['b', 'c'], ['red', 'blue']),
        (0, [1, 2], ['bokeh', 'cython'], ['green', 'yellow'])
    ])
    def test_y_listlike(self, x: Any, y: Any, lbl: Any, colors: List[str]) -> None:
        df: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        _check_plot_works(df.plot, x='A', y=y, label=lbl)
        ax: Any = df.plot(x=x, y=y, label=lbl, color=colors)
        assert len(ax.lines) == len(y)
        _check_colors(ax.get_lines(), linecolors=colors)

    @pytest.mark.parametrize('x,y,colnames', [(0, 1, ['A', 'B']), (1, 0, [0, 1])])
    def test_xy_args_integer(self, x: Any, y: Any, colnames: List[Any]) -> None:
        df: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4]})
        df.columns = colnames
        _check_plot_works(df.plot, x=x, y=y)

    def test_hexbin_basic(self) -> None:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).uniform(size=20),
            'B': np.random.default_rng(2).uniform(size=20),
            'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)
        })
        ax: Any = df.plot.hexbin(x='A', y='B', gridsize=10)
        assert len(ax.collections) == 1

    def test_hexbin_basic_subplots(self) -> None:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).uniform(size=20),
            'B': np.random.default_rng(2).uniform(size=20),
            'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)
        })
        axes: Any = df.plot.hexbin(x='A', y='B', subplots=True)
        assert len(axes[0].figure.axes) == 2
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.parametrize('reduce_C', [None, np.std])
    def test_hexbin_with_c(self, reduce_C: Optional[callable]) -> None:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).uniform(size=20),
            'B': np.random.default_rng(2).uniform(size=20),
            'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)
        })
        ax: Any = df.plot.hexbin(x='A', y='B', C='C', reduce_C_function=reduce_C)
        assert len(ax.collections) == 1

    @pytest.mark.parametrize('kwargs, expected', [
        ({}, 'BuGn'),
        ({'colormap': 'cubehelix'}, 'cubehelix'),
        ({'cmap': 'YlGn'}, 'YlGn')
    ])
    def test_hexbin_cmap(self, kwargs: Dict[str, Any], expected: str) -> None:
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).uniform(size=20),
            'B': np.random.default_rng(2).uniform(size=20),
            'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)
        })
        ax: Any = df.plot.hexbin(x='A', y='B', **kwargs)
        assert ax.collections[0].cmap.name == expected

    def test_pie_df_err(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 3)),
                                  columns=['X', 'Y', 'Z'],
                                  index=['a', 'b', 'c', 'd', 'e'])
        msg: str = "pie requires either y column or 'subplots=True'"
        with pytest.raises(ValueError, match=msg):
            df.plot.pie()

    @pytest.mark.parametrize('y', ['Y', 2])
    def test_pie_df(self, y: Union[str, int]) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 3)),
                                  columns=['X', 'Y', 'Z'],
                                  index=['a', 'b', 'c', 'd', 'e'])
        ax: Any = _check_plot_works(df.plot.pie, y=y)
        _check_text_labels(ax.texts, df.index)

    def test_pie_df_subplots(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 3)),
                                  columns=['X', 'Y', 'Z'],
                                  index=['a', 'b', 'c', 'd', 'e'])
        axes: Any = _check_plot_works(df.plot.pie, default_axes=True, subplots=True)
        assert len(axes) == len(df.columns)
        for ax in axes:
            _check_text_labels(ax.texts, df.index)
        for ax, ylabel in zip(axes, df.columns):
            assert ax.get_ylabel() == ''

    def test_pie_df_labels_colors(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 3)),
                                  columns=['X', 'Y', 'Z'],
                                  index=['a', 'b', 'c', 'd', 'e'])
        labels: List[str] = ['A', 'B', 'C', 'D', 'E']
        color_args: List[str] = ['r', 'g', 'b', 'c', 'm']
        axes: Any = _check_plot_works(df.plot.pie, default_axes=True, subplots=True, labels=labels, colors=color_args)
        assert len(axes) == len(df.columns)
        for ax in axes:
            _check_text_labels(ax.texts, labels)
            _check_colors(ax.patches, facecolors=color_args)

    def test_pie_df_nan(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((4, 4)))
        for i in range(4):
            df.iloc[i, i] = np.nan
        _, axes = mpl.pyplot.subplots(ncols=4)
        kwargs: Dict[str, Any] = {'normalize': True}
        with tm.assert_produces_warning(None):
            df.plot.pie(subplots=True, ax=axes, legend=True, **kwargs)
        base_expected: List[str] = ['0', '1', '2', '3']
        for i, ax in enumerate(axes):
            expected: List[str] = list(base_expected)
            expected[i] = ''
            result: List[str] = [x.get_text() for x in ax.texts]
            assert result == expected
            result_labels: List[str] = [x.get_text() for x in ax.get_legend().get_texts()]
            expected_labels: List[str] = base_expected[:i] + base_expected[i + 1:]
            assert result_labels == expected_labels

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [{'logy': True}, {'logx': True, 'logy': True}, {'loglog': True}])
    def test_errorbar_plot(self, kwargs: Dict[str, Any]) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        d_err: Dict[str, Any] = {'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4}
        df_err: DataFrame = DataFrame(d_err)
        ax: Any = _check_plot_works(df.plot, yerr=df_err, **kwargs)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    def test_errorbar_plot_bar(self) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        d_err: Dict[str, Any] = {'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4}
        df_err: DataFrame = DataFrame(d_err)
        ax: Any = _check_plot_works((df + 1).plot, yerr=df_err, xerr=df_err, kind='bar', log=True)
        _check_has_errorbars(ax, xerr=2, yerr=2)

    @pytest.mark.slow
    def test_errorbar_plot_yerr_array(self) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        ax: Any = _check_plot_works(df['y'].plot, yerr=np.ones(12) * 0.4)
        _check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(df.plot, yerr=np.ones((2, 12)) * 0.4)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize('yerr', ['yerr', '誤差'])
    def test_errorbar_plot_column_name(self, yerr: str) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        df[yerr] = np.ones(12) * 0.2
        ax: Any = _check_plot_works(df.plot, yerr=yerr)
        _check_has_errorbars(ax, xerr=0, yerr=2)
        ax = _check_plot_works(df.plot, y='y', x='x', yerr=yerr)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_external_valueerror(self) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        with tm.external_error_raised(ValueError):
            df.plot(yerr=np.random.default_rng(2).standard_normal(11))

    @pytest.mark.slow
    def test_errorbar_plot_external_typeerror(self) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        df_err: DataFrame = DataFrame({'x': ['zzz'] * 12, 'y': ['zzz'] * 12})
        with tm.external_error_raised(TypeError):
            df.plot(yerr=df_err)

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['line', 'bar', 'barh'])
    @pytest.mark.parametrize('y_err', [
        Series(np.ones(12) * 0.2, name='x'),
        DataFrame({'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4})
    ])
    def test_errorbar_plot_different_yerr(self, kind: str, y_err: Union[Series, DataFrame]) -> None:
        df: DataFrame = DataFrame({'x': np.arange(12), 'y': np.arange(12, 0, -1)})
        ax: Any = _check_plot_works(df.plot, yerr=y_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['line', 'bar', 'barh'])
    @pytest.mark.parametrize('y_err, x_err', [
        (DataFrame({'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4}),
         DataFrame({'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4})),
        (Series(np.ones(12) * 0.2, name='x'),
         Series(np.ones(12) * 0.2, name='x')),
        (0.2, 0.2)
    ])
    def test_errorbar_plot_different_yerr_xerr(self, kind: str, y_err: Union[Series, DataFrame, float], x_err: Union[Series, DataFrame, float]) -> None:
        df: DataFrame = DataFrame({'x': np.arange(12), 'y': np.arange(12, 0, -1)})
        ax: Any = _check_plot_works(df.plot, yerr=y_err, xerr=x_err, kind=kind)
        _check_has_errorbars(ax, xerr=2, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['line', 'bar', 'barh'])
    def test_errorbar_plot_different_yerr_xerr_subplots(self, kind: str) -> None:
        df: DataFrame = DataFrame({'x': np.arange(12), 'y': np.arange(12, 0, -1)})
        df_err: DataFrame = DataFrame({'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4})
        axes: Any = _check_plot_works(df.plot, default_axes=True, yerr=df_err, xerr=df_err, subplots=True, kind=kind)
        _check_has_errorbars(axes, xerr=1, yerr=1)

    @pytest.mark.xfail(reason='Iterator is consumed', raises=ValueError)
    def test_errorbar_plot_iterator(self) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        ax: Any = _check_plot_works(df.plot, yerr=itertools.repeat(0.1, len(df)))
        _check_has_errorbars(ax, xerr=0, yerr=2)

    def test_errorbar_with_integer_column_names(self) -> None:
        df: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))))
        df_err: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))))
        ax: Any = _check_plot_works(df.plot, yerr=df_err)
        _check_has_errorbars(ax, xerr=0, yerr=2)
        ax = _check_plot_works(df.plot, y=0, yerr=1)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['line', 'bar'])
    def test_errorbar_with_partial_columns_kind(self, kind: str) -> None:
        df: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 3))))
        df_err: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=[0, 2])
        ax: Any = _check_plot_works(df.plot, yerr=df_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    def test_errorbar_with_partial_columns_dti(self) -> None:
        df: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 3))))
        df_err: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=[0, 2])
        ix: Any = date_range('1/1/2000', periods=10, freq='ME')
        df.set_index(ix, inplace=True)
        df_err.set_index(ix, inplace=True)
        ax: Any = _check_plot_works(df.plot, yerr=df_err, kind='line')
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize('err_box', [lambda x: x, DataFrame])
    def test_errorbar_with_partial_columns_box(self, err_box: Any) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        df: DataFrame = DataFrame(d)
        err: Any = err_box({'x': np.ones(12) * 0.2, 'z': np.ones(12) * 0.4})
        ax: Any = _check_plot_works(df.plot, yerr=err)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.parametrize('kind', ['line', 'bar', 'barh'])
    def test_errorbar_timeseries(self, kind: str) -> None:
        d: Dict[str, Any] = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
        d_err: Dict[str, Any] = {'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4}
        ix: Any = date_range('1/1/2000', '1/1/2001', freq='ME')
        tdf: DataFrame = DataFrame(d, index=ix)
        tdf_err: DataFrame = DataFrame(d_err, index=ix)
        ax: Any = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)
        ax = _check_plot_works(tdf.plot, yerr=d_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)
        ax = _check_plot_works(tdf.plot, y='y', yerr=tdf_err['x'], kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(tdf.plot, y='y', yerr='x', kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)
        axes: Any = _check_plot_works(tdf.plot, default_axes=True, kind=kind, yerr=tdf_err, subplots=True)
        _check_has_errorbars(axes, xerr=0, yerr=1)

    def test_errorbar_asymmetrical(self) -> None:
        err: np.ndarray = np.random.default_rng(2).random((3, 2, 5))
        df: DataFrame = DataFrame(np.arange(15).reshape(3, 5)).T
        ax: Any = df.plot(yerr=err, xerr=err / 2)
        yerr_0_0: np.ndarray = ax.collections[1].get_paths()[0].vertices[:, 1]
        expected_0_0: np.ndarray = err[0, :, 0] * np.array([-1, 1])
        tm.assert_almost_equal(yerr_0_0, expected_0_0)
        msg: str = re.escape('Asymmetrical error bars should be provided with the shape (3, 2, 5)')
        with pytest.raises(ValueError, match=msg):
            df.plot(yerr=err.T)

    def test_table(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 3)),
                                  index=list(string.ascii_letters[:10]))
        _check_plot_works(df.plot, table=True)
        _check_plot_works(df.plot, table=df)
        with tm.assert_produces_warning(None):
            ax: Any = df.plot()
            assert len(ax.tables) == 0
            plotting.table(ax, df.T)
            assert len(ax.tables) == 1

    def test_errorbar_scatter(self) -> None:
        df: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((5, 2))),
                                  index=range(5), columns=['x', 'y'])
        df_err: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((5, 2))) / 5,
                                      index=range(5), columns=['x', 'y'])
        ax: Any = _check_plot_works(df.plot.scatter, x='x', y='y')
        _check_has_errorbars(ax, xerr=0, yerr=0)
        ax = _check_plot_works(df.plot.scatter, x='x', y='y', xerr=df_err)
        _check_has_errorbars(ax, xerr=1, yerr=0)
        ax = _check_plot_works(df.plot.scatter, x='x', y='y', yerr=df_err)
        _check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(df.plot.scatter, x='x', y='y', xerr=df_err, yerr=df_err)
        _check_has_errorbars(ax, xerr=1, yerr=1)

    def test_errorbar_scatter_color(self) -> None:
        def _check_errorbar_color(containers: List[Any], expected: str, has_err: str = 'has_xerr') -> None:
            lines: List[Any] = []
            errs = next((c.lines for c in ax.containers if getattr(c, has_err, False)))
            for el in errs:
                if is_list_like(el):
                    lines.extend(el)
                else:
                    lines.append(el)
            err_lines = [x for x in lines if x in ax.collections]
            _check_colors(err_lines, linecolors=np.array([expected] * len(err_lines)))
        df: DataFrame = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 5))),
                                  columns=['a', 'b', 'c', 'd', 'e'])
        ax: Any = df.plot.scatter(x='a', y='b', xerr='d', yerr='e', c='red')
        _check_has_errorbars(ax, xerr=1, yerr=1)
        _check_errorbar_color(ax.containers, 'red', has_err='has_xerr')
        _check_errorbar_color(ax.containers, 'red', has_err='has_yerr')
        ax = df.plot.scatter(x='a', y='b', yerr='e', color='green')
        _check_has_errorbars(ax, xerr=0, yerr=1)
        _check_errorbar_color(ax.containers, 'green', has_err='has_yerr')

    def test_scatter_unknown_colormap(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 2, 3], 'b': 4})
        with pytest.raises((ValueError, KeyError), match="'unknown' is not a"):
            df.plot(x='a', y='b', colormap='unknown', kind='scatter')

    def test_sharex_and_ax(self) -> None:
        gs, axes = _generate_4_axes_via_gridspec()
        df: DataFrame = DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [1, 2, 3, 4, 5, 6],
            'c': [1, 2, 3, 4, 5, 6],
            'd': [1, 2, 3, 4, 5, 6]
        })

        def _check(axes: List[Any]) -> None:
            for ax in axes:
                assert len(ax.lines) == 1
                _check_visible(ax.get_yticklabels(), visible=True)
            for ax in [axes[0], axes[2]]:
                _check_visible(ax.get_xticklabels(), visible=False)
                _check_visible(ax.get_xticklabels(minor=True), visible=False)
            for ax in [axes[1], axes[3]]:
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
        for ax in axes:
            df.plot(x='a', y='b', title='title', ax=ax, sharex=True)
        gs.tight_layout(plt.gcf())
        _check(axes)
        plt.close('all')
        gs, axes = _generate_4_axes_via_gridspec()
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        _check(axes)

    def test_sharex_false_and_ax(self) -> None:
        df: DataFrame = DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [1, 2, 3, 4, 5, 6],
            'c': [1, 2, 3, 4, 5, 6],
            'd': [1, 2, 3, 4, 5, 6]
        })
        gs, axes = _generate_4_axes_via_gridspec()
        for ax in axes:
            df.plot(x='a', y='b', title='title', ax=ax)
        gs.tight_layout(plt.gcf())
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    def test_sharey_and_ax(self) -> None:
        gs, axes = _generate_4_axes_via_gridspec()
        df: DataFrame = DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [1, 2, 3, 4, 5, 6],
            'c': [1, 2, 3, 4, 5, 6],
            'd': [1, 2, 3, 4, 5, 6]
        })

        def _check(axes: List[Any]) -> None:
            for ax in axes:
                assert len(ax.lines) == 1
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
            for ax in [axes[0], axes[1]]:
                _check_visible(ax.get_yticklabels(), visible=True)
            for ax in [axes[2], axes[3]]:
                _check_visible(ax.get_yticklabels(), visible=False)
        for ax in axes:
            df.plot(x='a', y='b', title='title', ax=ax, sharey=True)
        gs.tight_layout(plt.gcf())
        _check(axes)
        plt.close('all')
        gs, axes = _generate_4_axes_via_gridspec()
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            axes = df.plot(subplots=True, ax=axes, sharey=True)
        gs.tight_layout(plt.gcf())
        _check(axes)

    def test_sharey_and_ax_tight(self) -> None:
        df: DataFrame = DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [1, 2, 3, 4, 5, 6],
            'c': [1, 2, 3, 4, 5, 6],
            'd': [1, 2, 3, 4, 5, 6]
        })
        gs, axes = _generate_4_axes_via_gridspec()
        for ax in axes:
            df.plot(x='a', y='b', title='title', ax=ax)
        gs.tight_layout(plt.gcf())
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._all_kinds)
    def test_memory_leak(self, kind: str) -> None:
        pytest.importorskip('scipy')
        args: Dict[str, Any] = {}
        if kind in ['hexbin', 'scatter', 'pie']:
            df: DataFrame = DataFrame({
                'A': np.random.default_rng(2).uniform(size=20),
                'B': np.random.default_rng(2).uniform(size=20),
                'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)
            })
            args = {'x': 'A', 'y': 'B'}
        elif kind == 'area':
            df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                      columns=Index(list('ABCD'), dtype=object),
                                      index=date_range('2000-01-01', periods=10, freq='B')).abs()
        else:
            df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                      columns=Index(list('ABCD'), dtype=object),
                                      index=date_range('2000-01-01', periods=10, freq='B'))
        ref = weakref.ref(df.plot(kind=kind, **args))
        plt.close('all')
        gc.collect()
        assert ref() is None

    def test_df_gridspec_patterns_vert_horiz(self) -> None:
        ts: Series = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1/1/2000', periods=10))
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)),
                                  index=ts.index, columns=list('AB'))

        def _get_vertical_grid() -> Tuple[Any, List[Any]]:
            gs = mpl.gridspec.GridSpec(3, 1)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:2, :])
            ax2 = fig.add_subplot(gs[2, :])
            return (ax1, ax2)

        def _get_horizontal_grid() -> Tuple[Any, List[Any]]:
            gs = mpl.gridspec.GridSpec(1, 3)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:, :2])
            ax2 = fig.add_subplot(gs[:, 2])
            return (ax1, ax2)
        for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
            ax1 = ts.plot(ax=ax1)
            assert len(ax1.lines) == 1
            ax2 = df.plot(ax=ax2)
            assert len(ax2.lines) == 2
            for ax in [ax1, ax2]:
                _check_visible(ax.get_yticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
            plt.close('all')
        for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
            axes: Any = df.plot(subplots=True, ax=[ax1, ax2])
            assert len(ax1.lines) == 1
            assert len(ax2.lines) == 1
            for ax in axes:
                _check_visible(ax.get_yticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
            plt.close('all')
        ax1, ax2 = _get_vertical_grid()
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1
        for ax in [ax1, ax2]:
            _check_visible(ax.get_yticklabels(), visible=True)
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)
        plt.close('all')
        ax1, ax2 = _get_horizontal_grid()
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1
        _check_visible(axes[0].get_yticklabels(), visible=True)
        _check_visible(axes[1].get_yticklabels(), visible=False)
        for ax in [ax1, ax2]:
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close('all')

    def test_df_gridspec_patterns_boxed(self) -> None:
        ts: Series = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1/1/2000', periods=10))
        def _get_boxed_grid() -> Tuple[Any, Any, Any, Any]:
            gs = mpl.gridspec.GridSpec(3, 3)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:2, :2])
            ax2 = fig.add_subplot(gs[:2, 2])
            ax3 = fig.add_subplot(gs[2, :2])
            ax4 = fig.add_subplot(gs[2, 2])
            return (ax1, ax2, ax3, ax4)
        axes: Tuple[Any, Any, Any, Any] = _get_boxed_grid()
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  index=ts.index, columns=list('ABCD'))
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close('all')
        axes = _get_boxed_grid()
        with tm.assert_produces_warning(UserWarning, match='sharex and sharey'):
            axes = df.plot(subplots=True, ax=axes, sharex=True, sharey=True)
        for ax in axes:
            assert len(ax.lines) == 1
        for ax in [axes[0], axes[2]]:
            _check_visible(ax.get_yticklabels(), visible=True)
        for ax in [axes[1], axes[3]]:
            _check_visible(ax.get_yticklabels(), visible=False)
        for ax in [axes[0], axes[1]]:
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
        for ax in [axes[2], axes[3]]:
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close('all')

    def test_df_grid_settings(self) -> None:
        _check_grid_settings(DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]}),
                             plotting.PlotAccessor._dataframe_kinds,
                             kws={'x': 'a', 'y': 'b'})

    def test_plain_axes(self) -> None:
        fig, ax = mpl.pyplot.subplots()
        fig.add_axes([0.2, 0.2, 0.2, 0.2])
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)

    def test_plain_axes_df(self) -> None:
        df: DataFrame = DataFrame({'a': np.random.default_rng(2).standard_normal(8),
                                   'b': np.random.default_rng(2).standard_normal(8)})
        fig = mpl.pyplot.figure()
        ax = fig.add_axes((0, 0, 1, 1))
        df.plot(kind='scatter', ax=ax, x='a', y='b', c='a', cmap='hsv')

    def test_plain_axes_make_axes_locatable(self) -> None:
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        Series(np.random.default_rng(2).random(10)).plot(ax=cax)

    def test_plain_axes_make_inset_axes(self) -> None:
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        iax = inset_axes(ax, width='30%', height=1.0, loc=3)
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        Series(np.random.default_rng(2).random(10)).plot(ax=iax)

    @pytest.mark.parametrize('method', ['line', 'barh', 'bar'])
    def test_secondary_axis_font_size(self, method: str) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((15, 2)),
                                  columns=list('AB')).assign(C=lambda df: df.B.cumsum()).assign(D=lambda df: df.C * 1.1)
        fontsize: int = 20
        sy: List[str] = ['C', 'D']
        kwargs: Dict[str, Any] = {'secondary_y': sy, 'fontsize': fontsize, 'mark_right': True}
        ax: Any = getattr(df.plot, method)(**kwargs)
        _check_ticks_props(axes=ax.right_ax, ylabelsize=fontsize)

    def test_x_string_values_ticks(self) -> None:
        df: DataFrame = DataFrame({'sales': [3, 2, 3],
                                   'visits': [20, 42, 28],
                                   'day': ['Monday', 'Tuesday', 'Wednesday']})
        ax: Any = df.plot.area(x='day')
        ax.set_xlim(-1, 3)
        xticklabels: List[str] = [t.get_text() for t in ax.get_xticklabels()]
        labels_position: Dict[str, float] = dict(zip(xticklabels, ax.get_xticks()))
        assert labels_position['Monday'] == 0.0
        assert labels_position['Tuesday'] == 1.0
        assert labels_position['Wednesday'] == 2.0

    def test_x_multiindex_values_ticks(self) -> None:
        index: MultiIndex = MultiIndex.from_product([[2012, 2013], [1, 2]])
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 2)),
                                  columns=['A', 'B'],
                                  index=index)
        ax: Any = df.plot()
        ax.set_xlim(-1, 4)
        xticklabels: List[str] = [t.get_text() for t in ax.get_xticklabels()]
        labels_position: Dict[str, float] = dict(zip(xticklabels, ax.get_xticks()))
        assert labels_position['(2012, 1)'] == 0.0
        assert labels_position['(2012, 2)'] == 1.0
        assert labels_position['(2013, 1)'] == 2.0
        assert labels_position['(2013, 2)'] == 3.0

    @pytest.mark.parametrize('kind', ['line', 'area'])
    def test_xlim_plot_line(self, kind: str) -> None:
        df: DataFrame = DataFrame([2, 4], index=[1, 2])
        ax: Any = df.plot(kind=kind)
        xlims: Tuple[float, float] = ax.get_xlim()
        assert xlims[0] < 1
        assert xlims[1] > 2

    def test_xlim_plot_line_correctly_in_mixed_plot_type(self) -> None:
        fig, ax = mpl.pyplot.subplots()
        indexes: List[str] = ['k1', 'k2', 'k3', 'k4']
        df: DataFrame = DataFrame({
            's1': [1000, 2000, 1500, 2000],
            's2': [900, 1400, 2000, 3000],
            's3': [1500, 1500, 1600, 1200],
            'secondary_y': [1, 3, 4, 3]
        }, index=indexes)
        df[['s1', 's2', 's3']].plot.bar(ax=ax, stacked=False)
        df[['secondary_y']].plot(ax=ax, secondary_y=True)
        xlims = ax.get_xlim()
        assert xlims[0] < 0
        assert xlims[1] > 3
        xticklabels: List[str] = [t.get_text() for t in ax.get_xticklabels()]
        assert xticklabels == indexes

    def test_plot_no_rows(self) -> None:
        df: DataFrame = DataFrame(columns=['foo'], dtype=int)
        assert df.empty
        ax: Any = df.plot()
        assert len(ax.get_lines()) == 1
        line: Any = ax.get_lines()[0]
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self) -> None:
        df: DataFrame = DataFrame(['a', 'b', 'c'])
        with pytest.raises(TypeError, match='no numeric data to plot'):
            df.plot()

    @pytest.mark.parametrize('kind', ('line', 'bar', 'barh', 'hist', 'kde', 'density', 'area', 'pie'))
    def test_group_subplot(self, kind: str) -> None:
        pytest.importorskip('scipy')
        d: Dict[str, Any] = {'a': np.arange(10), 'b': np.arange(10) + 1,
                             'c': np.arange(10) + 1, 'd': np.arange(10),
                             'e': np.arange(10)}
        df: DataFrame = DataFrame(d)
        axes: Any = df.plot(subplots=[('b', 'e'), ('c', 'd')], kind=kind)
        assert len(axes) == 3
        expected_labels: Tuple[List[str], List[str], List[str]] = (['b', 'e'], ['c', 'd'], ['a'])
        for ax, labels in zip(axes, expected_labels):
            if kind != 'pie':
                _check_legend_labels(ax, labels=labels)
            if kind == 'line':
                assert len(ax.lines) == len(labels)

    def test_group_subplot_series_notimplemented(self) -> None:
        ser: Series = Series(range(1))
        msg: str = 'An iterable subplots for a Series'
        with pytest.raises(NotImplementedError, match=msg):
            ser.plot(subplots=[('a',)])

    def test_group_subplot_multiindex_notimplemented(self) -> None:
        df: DataFrame = DataFrame(np.eye(2), columns=MultiIndex.from_tuples([(0, 1), (1, 2)]))
        msg: str = 'An iterable subplots for a DataFrame with a MultiIndex'
        with pytest.raises(NotImplementedError, match=msg):
            df.plot(subplots=[(0, 1)])

    def test_group_subplot_nonunique_cols_notimplemented(self) -> None:
        df: DataFrame = DataFrame(np.eye(2), columns=['a', 'a'])
        msg: str = 'An iterable subplots for a DataFrame with non-unique'
        with pytest.raises(NotImplementedError, match=msg):
            df.plot(subplots=[('a',)])

    @pytest.mark.parametrize('subplots, expected_msg', [
        (123, 'subplots should be a bool or an iterable'),
        ('a', 'each entry should be a list/tuple'),
        ((1,), 'each entry should be a list/tuple'),
        (('a',), 'each entry should be a list/tuple')
    ])
    def test_group_subplot_bad_input(self, subplots: Any, expected_msg: str) -> None:
        d: Dict[str, Any] = {'a': np.arange(10), 'b': np.arange(10)}
        df: DataFrame = DataFrame(d)
        with pytest.raises(ValueError, match=expected_msg):
            df.plot(subplots=subplots)

    def test_group_subplot_invalid_column_name(self) -> None:
        d: Dict[str, Any] = {'a': np.arange(10), 'b': np.arange(10)}
        df: DataFrame = DataFrame(d)
        if Version(np.__version__) < Version('2.0.0'):
            with pytest.raises(ValueError, match="Column label\\(s\\) \\['bad_name'\\]"):
                df.plot(subplots=[('a', 'bad_name')])
        else:
            with pytest.raises(ValueError, match="Column label\\(s\\) \\[np\\.str\\_\\('bad_name'\\)\\]"):
                df.plot(subplots=[('a', 'bad_name')])

    def test_group_subplot_duplicated_column(self) -> None:
        d: Dict[str, Any] = {'a': np.arange(10), 'b': np.arange(10), 'c': np.arange(10)}
        df: DataFrame = DataFrame(d)
        with pytest.raises(ValueError, match='should be in only one subplot'):
            df.plot(subplots=[('a', 'b'), ('a', 'c')])

    @pytest.mark.parametrize('kind', ('box', 'scatter', 'hexbin'))
    def test_group_subplot_invalid_kind(self, kind: str) -> None:
        d: Dict[str, Any] = {'a': np.arange(10), 'b': np.arange(10)}
        df: DataFrame = DataFrame(d)
        with pytest.raises(ValueError, match='When subplots is an iterable, kind must be one of'):
            df.plot(subplots=[('a', 'b')], kind=kind)

    @pytest.mark.parametrize('index_name, old_label, new_label', [
        (None, '', 'new'),
        ('old', 'old', 'new'),
        (None, '', ''),
        (None, '', 1),
        (None, '', [1, 2])
    ])
    @pytest.mark.parametrize('kind', ['line', 'area', 'bar'])
    def test_xlabel_ylabel_dataframe_single_plot(self, kind: str, index_name: Optional[str], old_label: Any, new_label: Any) -> None:
        df: DataFrame = DataFrame([[1, 2], [2, 5]], columns=['Type A', 'Type B'])
        df.index.name = index_name
        ax: Any = df.plot(kind=kind)
        assert ax.get_xlabel() == old_label
        assert ax.get_ylabel() == ''
        ax = df.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        assert ax.get_ylabel() == str(new_label)
        assert ax.get_xlabel() == str(new_label)

    @pytest.mark.parametrize('xlabel, ylabel', [(None, None), ('X Label', None), (None, 'Y Label'), ('X Label', 'Y Label')])
    @pytest.mark.parametrize('kind', ['scatter', 'hexbin'])
    def test_xlabel_ylabel_dataframe_plane_plot(self, kind: str, xlabel: Optional[str], ylabel: Optional[str]) -> None:
        xcol: str = 'Type A'
        ycol: str = 'Type B'
        df: DataFrame = DataFrame([[1, 2], [2, 5]], columns=[xcol, ycol])
        ax: Any = df.plot(kind=kind, x=xcol, y=ycol, xlabel=xlabel, ylabel=ylabel)
        assert ax.get_xlabel() == (xcol if xlabel is None else xlabel)
        assert ax.get_ylabel() == (ycol if ylabel is None else ylabel)

    @pytest.mark.parametrize('secondary_y', (False, True))
    def test_secondary_y(self, secondary_y: bool) -> None:
        ax_df: Any = DataFrame([0]).plot(secondary_y=secondary_y, ylabel='Y', ylim=(0, 100), yticks=[99])
        for ax in ax_df.figure.axes:
            if ax.yaxis.get_visible():
                assert ax.get_ylabel() == 'Y'
                assert ax.get_ylim() == (0, 100)
                assert ax.get_yticks()[0] == 99

    @pytest.mark.slow
    def test_plot_no_warning(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                  columns=Index(list('ABCD'), dtype=object),
                                  index=date_range('2000-01-01', periods=10, freq='B'))
        with tm.assert_produces_warning(False):
            _ = df.plot()
            _ = df.T.plot()

    @pytest.mark.parametrize('freq', ['h', '7h', '60min', '120min', '3M'])
    def test_plot_period_index_makes_no_right_shift(self, freq: str) -> None:
        idx: pd.PeriodIndex = pd.period_range('01/01/2000', freq=freq, periods=4)
        df: DataFrame = DataFrame(np.array([0, 1, 0, 1]), index=idx, columns=['A'])
        expected: Any = idx.values
        ax: Any = df.plot()
        result: Any = ax.get_lines()[0].get_xdata()
        assert all((str(result[i]) == str(expected[i]) for i in range(4)))

    def test_plot_display_xlabel_and_xticks(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['a', 'b'])
        ax: Any = df.plot.hexbin(x='a', y='b')
        _check_visible([ax.xaxis.get_label()], visible=True)
        _check_visible(ax.get_xticklabels(), visible=True)

def _generate_4_axes_via_gridspec() -> Tuple[mpl.gridspec.GridSpec, List[Any]]:
    gs = mpl.gridspec.GridSpec(2, 2)
    ax_tl = plt.subplot(gs[0, 0])
    ax_ll = plt.subplot(gs[1, 0])
    ax_tr = plt.subplot(gs[0, 1])
    ax_lr = plt.subplot(gs[1, 1])
    return (gs, [ax_tl, ax_ll, ax_tr, ax_lr])