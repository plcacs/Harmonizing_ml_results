from datetime import date, datetime
import gc
import itertools
import re
import string
import weakref
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, PeriodIndex, Series, bdate_range, date_range, option_context, plotting
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_box_return_type,
    _check_colors,
    _check_data,
    _check_grid_settings,
    _check_has_errorbars,
    _check_legend_labels,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    _check_visible,
    get_y_axis,
)
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from numpy.typing import NDArray

class TestDataFramePlots:
    @pytest.mark.slow
    def test_plot(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        )
        _check_plot_works(df.plot, grid=False)

    @pytest.mark.slow
    def test_plot_subplots(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        )
        axes = _check_plot_works(df.plot, default_axes=True, subplots=True)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    @pytest.mark.slow
    def test_plot_subplots_negative_layout(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        )
        axes = _check_plot_works(df.plot, default_axes=True, subplots=True, layout=(-1, 2))
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    @pytest.mark.slow
    def test_plot_subplots_use_index(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        )
        axes = _check_plot_works(df.plot, default_axes=True, subplots=True, use_index=False)
        _check_ticks_props(axes, xrot=0)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    @pytest.mark.slow
    def test_plot_invalid_arg(self) -> None:
        df = DataFrame({'x': [1, 2], 'y': [3, 4]})
        msg = "'Line2D' object has no property 'blarg'"
        with pytest.raises(AttributeError, match=msg):
            df.plot.line(blarg=True)

    @pytest.mark.slow
    def test_plot_tick_props(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        ax = _check_plot_works(df.plot, use_index=True)
        _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [{'yticks': [1, 5, 10]}, {'xticks': [1, 5, 10]}, {'ylim': (-100, 100), 'xlim': (-100, 100)}, {'default_axes': True, 'subplots': True, 'title': 'blah'}])
    def test_plot_other_args(self, kwargs: Dict[str, Any]) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        _check_plot_works(df.plot, **kwargs)

    @pytest.mark.slow
    def test_plot_visible_ax(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        axes = df.plot(subplots=True, title='blah')
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
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        _check_plot_works(df.plot, title='blah')

    @pytest.mark.slow
    def test_plot_multiindex(self) -> None:
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=MultiIndex.from_tuples(tuples))
        ax = _check_plot_works(df.plot, use_index=True)
        _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    def test_plot_multiindex_unicode(self) -> None:
        index = MultiIndex.from_tuples([('α', 0), ('α', 1), ('β', 2), ('β', 3), ('γ', 4), ('γ', 5), ('δ', 6), ('δ', 7)], names=['i0', 'i1'])
        columns = MultiIndex.from_tuples([('bar', 'Δ'), ('bar', 'Ε')], names=['c0', 'c1'])
        df = DataFrame(np.random.default_rng(2).integers(0, 10, (8, 2)), columns=columns, index=index)
        _check_plot_works(df.plot, title='Σ')

    @pytest.mark.slow
    @pytest.mark.parametrize('layout', [None, (-1, 1)])
    def test_plot_single_column_bar(self, layout: Optional[Tuple[int, int]]) -> None:
        df = DataFrame({'x': np.random.default_rng(2).random(10)})
        axes = _check_plot_works(df.plot.bar, subplots=True, layout=layout)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_plot_passed_ax(self) -> None:
        df = DataFrame({'x': np.random.default_rng(2).random(10)})
        _, ax = mpl.pyplot.subplots()
        axes = df.plot.bar(subplots=True, ax=ax)
        assert len(axes) == 1
        result = ax.axes
        assert result is axes[0]

    @pytest.mark.parametrize('cols, x, y', [[list('ABCDE'), 'A', 'B'], [['A', 'B'], 'A', 'B'], [['C', 'A'], 'C', 'A'], [['A', 'C'], 'A', 'C'], [['B', 'C'], 'B', 'C'], [['A', 'D'], 'A', 'D'], [['A', 'E'], 'A', 'E']])
    def test_nullable_int_plot(self, cols: List[str], x: str, y: str) -> None:
        dates = ['2008', '2009', None, '2011', '2012']
        df = DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 2, 3, 4, 5], 'C': np.array([7, 5, np.nan, 3, 2], dtype=object), 'D': pd.to_datetime(dates, format='%Y').view('i8'), 'E': pd.to_datetime(dates, format='%Y', utc=True).view('i8')})
        _check_plot_works(df[cols].plot, x=x, y=y)

    @pytest.mark.slow
    @pytest.mark.parametrize('plot', ['line', 'bar', 'hist', 'pie'])
    def test_integer_array_plot_series(self, plot: str) -> None:
        arr = pd.array([1, 2, 3, 4], dtype='UInt32')
        s = Series(arr)
        _check_plot_works(getattr(s.plot, plot))

    @pytest.mark.slow
    @pytest.mark.parametrize('plot, kwargs', [['line', {}], ['bar', {}], ['hist', {}], ['pie', {'y': 'y'}], ['scatter', {'x': 'x', 'y': 'y'}], ['hexbin', {'x': 'x', 'y': 'y'}]])
    def test_integer_array_plot_df(self, plot: str, kwargs: Dict[str, str]) -> None:
        arr = pd.array([1, 2, 3, 4], dtype='UInt32')
        df = DataFrame({'x': arr, 'y': arr})
        _check_plot_works(getattr(df.plot, plot), **kwargs)

    def test_nonnumeric_exclude(self) -> None:
        df = DataFrame({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]})
        ax = df.plot()
        assert len(ax.get_lines()) == 1

    def test_implicit_label(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
        ax = df.plot(x='a', y='b')
        _check_text_labels(ax.xaxis.get_label(), 'a')

    def test_donot_overwrite_index_name(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)), columns=['a', 'b'])
        df.index.name = 'NAME'
        df.plot(y='b', label='LABEL')
        assert df.index.name == 'NAME'

    def test_plot_xy(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=5, freq='B'))
        _check_data(df.plot(x=0, y=1), df.set_index('A')['B'].plot())
        _check_data(df.plot(x=0), df.set_index('A').plot())
        _check_data(df.plot(y=0), df.B.plot())
        _check_data(df.plot(x='A', y='B'), df.set_index('A').B.plot())
        _check_data(df.plot(x='A'), df.set_index('A').plot())
        _check_data(df.plot(y='B'), df.B.plot())

    def test_plot_xy_int_cols(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=5, freq='B'))
        df.columns = np.arange(1, len(df.columns) + 1)
        _check_data(df.plot(x=1, y=2), df.set_index(1)[2].plot())
        _check_data(df.plot(x=1), df.set_index(1).plot())
        _check_data(df.plot(y=1), df[1].plot())

    def test_plot_xy_figsize_and_title(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=5, freq='B'))
        ax = df.plot(x=1, y=2, title='Test', figsize=(16, 8))
        _check_text_labels(ax.title, 'Test')
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16.0, 8.0))

    @pytest.mark.parametrize('input_log, expected_log', [(True, 'log'), ('sym', 'symlog')])
    def test_logscales(self, input_log: Union[bool, str], expected_log: str) -> None:
        df = DataFrame({'a': np.arange(100)}, index=np.arange(100))
        ax = df.plot(logy=input_log)
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
        df = DataFrame({'a': np.arange(100)}, index=np.arange(100))
        msg = f"keyword '{input_param}' should be bool, None, or 'sym', not 'sm'"
        with pytest.raises(ValueError, match=msg):
            df.plot(**{input_param: 'sm'})
        msg = f"PiePlot ignores the '{input_param}' keyword"
        with tm.assert_produces_warning(UserWarning, match=msg):
            df.plot.pie(subplots=True, **{input_param: True})

    def test_xcompat(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        ax = df.plot(x_compat=True)
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        plotting.plot_params['xaxis.compat'] = True
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params_x_compat(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        plotting.plot_params['x_compat'] = False
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        msg = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)

    def test_xcompat_plot_params_context_manager(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        with plotting.plot_params.use('x_compat', True):
            ax = df.plot()
            lines = ax.get_lines()
            assert not isinstance(lines[0].get_xdata(), PeriodIndex)
            _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_period(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        msg = 'PeriodDtype\\[B\\] is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)
        _check_ticks_props(ax, xrot=0