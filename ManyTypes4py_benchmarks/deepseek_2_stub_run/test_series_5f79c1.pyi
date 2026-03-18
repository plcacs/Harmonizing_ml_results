```python
import datetime
from typing import Any, Literal, Sequence, Union

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.tseries.offsets import CustomBusinessDay

def _check_plot_works(
    func: Any,
    *args: Any,
    **kwargs: Any
) -> Any: ...

def _check_ax_scales(
    ax: Any,
    **kwargs: Any
) -> None: ...

def _check_axes_shape(
    ax: Any,
    axes_num: int,
    layout: tuple[int, int]
) -> None: ...

def _check_colors(
    artists: Any,
    facecolors: Any = None,
    linecolors: Any = None
) -> None: ...

def _check_grid_settings(
    obj: Any,
    kinds: list[str]
) -> None: ...

def _check_has_errorbars(
    ax: Any,
    xerr: int,
    yerr: int
) -> None: ...

def _check_legend_labels(
    ax: Any,
    labels: list[str]
) -> None: ...

def _check_text_labels(
    labels: Any,
    expected: Any
) -> None: ...

def _check_ticks_props(
    axes: Any,
    xrot: Any = None,
    yrot: Any = None
) -> None: ...

def _unpack_cycler(
    rc_params: Any
) -> list[Any]: ...

def get_y_axis(
    ax: Any
) -> Any: ...

def get_standard_colors(
    num_colors: int,
    color: Any = None
) -> list[Any]: ...

class TestSeriesPlots:
    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [{'label': 'foo'}, {'use_index': False}])
    def test_plot(
        self,
        ts: Series,
        kwargs: dict[str, Any]
    ) -> None: ...

    @pytest.mark.slow
    def test_plot_tick_props(
        self,
        ts: Series
    ) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('scale, exp_scale', [[{'logy': True}, {'yaxis': 'log'}], [{'logx': True}, {'xaxis': 'log'}], [{'loglog': True}, {'xaxis': 'log', 'yaxis': 'log'}]])
    def test_plot_scales(
        self,
        ts: Series,
        scale: dict[str, bool],
        exp_scale: dict[str, str]
    ) -> None: ...

    @pytest.mark.slow
    def test_plot_ts_bar(
        self,
        ts: Series
    ) -> None: ...

    @pytest.mark.slow
    def test_plot_ts_area_stacked(
        self,
        ts: Series
    ) -> None: ...

    def test_plot_iseries(self) -> None: ...

    @pytest.mark.parametrize('kind', ['line', 'bar', 'barh', pytest.param('kde', marks=td.skip_if_no('scipy')), 'hist', 'box'])
    def test_plot_series_kinds(
        self,
        series: Series,
        kind: str
    ) -> None: ...

    def test_plot_series_barh(
        self,
        series: Series
    ) -> None: ...

    def test_plot_series_bar_ax(self) -> None: ...

    @pytest.mark.parametrize('kwargs', [{}, {'layout': (-1, 1)}, {'layout': (1, -1)}])
    def test_plot_6951(
        self,
        ts: Series,
        kwargs: dict[str, Any]
    ) -> None: ...

    def test_plot_figsize_and_title(
        self,
        series: Series
    ) -> None: ...

    def test_dont_modify_rcParams(self) -> None: ...

    @pytest.mark.parametrize('kwargs', [{}, {'secondary_y': True}])
    def test_ts_line_lim(
        self,
        ts: Series,
        kwargs: dict[str, Any]
    ) -> None: ...

    def test_ts_area_lim(
        self,
        ts: Series
    ) -> None: ...

    def test_ts_area_lim_xcompat(
        self,
        ts: Series
    ) -> None: ...

    def test_ts_tz_area_lim_xcompat(
        self,
        ts: Series
    ) -> None: ...

    def test_ts_tz_area_lim_xcompat_secondary_y(
        self,
        ts: Series
    ) -> None: ...

    def test_area_sharey_dont_overwrite(
        self,
        ts: Series
    ) -> None: ...

    def test_label(self) -> None: ...

    def test_label_none(self) -> None: ...

    def test_label_ser_name(self) -> None: ...

    def test_label_ser_name_override(self) -> None: ...

    def test_label_ser_name_override_dont_draw(self) -> None: ...

    def test_boolean(self) -> None: ...

    @pytest.mark.parametrize('index', [None, date_range('2020-01-01', periods=4)])
    def test_line_area_nan_series(
        self,
        index: Any
    ) -> None: ...

    def test_line_use_index_false(self) -> None: ...

    def test_line_use_index_false_diff_var(self) -> None: ...

    @pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
    @pytest.mark.parametrize('axis, meth', [('yaxis', 'bar'), ('xaxis', 'barh')])
    def test_bar_log(
        self,
        axis: str,
        meth: str
    ) -> None: ...

    @pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
    @pytest.mark.parametrize('axis, kind, res_meth', [['yaxis', 'bar', 'get_ylim'], ['xaxis', 'barh', 'get_xlim']])
    def test_bar_log_kind_bar(
        self,
        axis: str,
        kind: str,
        res_meth: str
    ) -> None: ...

    def test_bar_ignore_index(self) -> None: ...

    def test_bar_user_colors(self) -> None: ...

    def test_rotation_default(self) -> None: ...

    def test_rotation_30(self) -> None: ...

    def test_irregular_datetime(self) -> None: ...

    def test_unsorted_index_xlim(self) -> None: ...

    def test_pie_series(self) -> None: ...

    def test_pie_arrow_type(self) -> None: ...

    def test_pie_series_no_label(self) -> None: ...

    def test_pie_series_less_colors_than_elements(self) -> None: ...

    def test_pie_series_labels_and_colors(self) -> None: ...

    def test_pie_series_autopct_and_fontsize(self) -> None: ...

    def test_pie_series_negative_raises(self) -> None: ...

    def test_pie_series_nan(self) -> None: ...

    def test_pie_nan(self) -> None: ...

    def test_df_series_secondary_legend(self) -> None: ...

    def test_df_series_secondary_legend_both(self) -> None: ...

    def test_df_series_secondary_legend_both_with_axis_2(self) -> None: ...

    @pytest.mark.parametrize('input_logy, expected_scale', [(True, 'log'), ('sym', 'symlog')])
    @pytest.mark.parametrize('secondary_kwarg', [{}, {'secondary_y': True}])
    def test_secondary_logy(
        self,
        input_logy: Union[bool, str],
        expected_scale: str,
        secondary_kwarg: dict[str, Any]
    ) -> None: ...

    def test_plot_fails_with_dupe_color_and_style(self) -> None: ...

    @pytest.mark.parametrize('bw_method, ind', [['scott', 20], [None, 20], [None, np.int_(20)], [0.5, np.linspace(-100, 100, 20)]])
    def test_kde_kwargs(
        self,
        ts: Series,
        bw_method: Any,
        ind: Any
    ) -> None: ...

    @pytest.mark.parametrize('bw_method, ind, weights', [['scott', 20, None], [None, 20, None], [None, np.int_(20), None], [0.5, np.linspace(-100, 100, 20), None], ['scott', 40, np.linspace(0.0, 2.0, 50)]])
    def test_kde_kwargs_weights(
        self,
        bw_method: Any,
        ind: Any,
        weights: Any
    ) -> None: ...

    def test_density_kwargs(
        self,
        ts: Series
    ) -> None: ...

    def test_kde_kwargs_check_axes(
        self,
        ts: Series
    ) -> None: ...

    def test_kde_missing_vals(self) -> None: ...

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_boxplot_series(
        self,
        ts: Series
    ) -> None: ...

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds)
    def test_kind_kwarg(
        self,
        kind: str
    ) -> None: ...

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds)
    def test_kind_attr(
        self,
        kind: str
    ) -> None: ...

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_invalid_plot_data(
        self,
        kind: str
    ) -> None: ...

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_valid_object_plot(
        self,
        kind: str
    ) -> None: ...

    @pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
    def test_partially_invalid_plot_data(
        self,
        kind: str
    ) -> None: ...

    def test_invalid_kind(self) -> None: ...

    def test_dup_datetime_index_plot(self) -> None: ...

    def test_errorbar_asymmetrical(self) -> None: ...

    def test_errorbar_asymmetrical_error(self) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['line', 'bar'])
    @pytest.mark.parametrize('yerr', [Series(np.abs(np.random.default_rng(2).standard_normal(10))), np.abs(np.random.default_rng(2).standard_normal(10)), list(np.abs(np.random.default_rng(2).standard_normal(10))), DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=['x', 'y'])])
    def test_errorbar_plot(
        self,
        kind: str,
        yerr: Any
    ) -> None: ...

    @pytest.mark.slow
    def test_errorbar_plot_yerr_0(self) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('yerr', [Series(np.abs(np.random.default_rng(2).standard_normal(12))), DataFrame(np.abs(np.random.default_rng(2).standard_normal((12, 2))), columns=['x', 'y'])])
    def test_errorbar_plot_ts(
        self,
        yerr: Any
    ) -> None: ...

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr_shape(self) -> None: ...

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr(self) -> None: ...

    @pytest.mark.slow
    def test_table_true(
        self,
        series: Series
    ) -> None: ...

    @pytest.mark.slow
    def test_table_self(
        self,
        series: Series
    ) -> None: ...

    @pytest.mark.slow
    def test_series_grid_settings(self) -> None: ...

    @pytest.mark.parametrize('c', ['r', 'red', 'green', '#FF0000'])
    def test_standard_colors(
        self,
        c: str
    ) -> None: ...

    def test_standard_colors_all(self) -> None: ...

    def test_series_plot_color_kwargs(self) -> None: ...

    def test_time_series_plot_color_kwargs(self) -> None: ...

    def test_time_series_plot_color_with_empty_kwargs(self) -> None: ...

    def test_xticklabels(self) -> None: ...

    def test_xtick_barPlot(self) -> None: ...

    def test_custom_business_day_freq(self) -> None: ...

    @pytest.mark.xfail(reason='GH#24426, see also github.com/pandas-dev/pandas/commit/ef1bd69fa42bbed5d09dd17f08c44fc8bfc2b685#r61470674')
    def test_plot_accessor_updates_on_inplace(self) -> None: ...

    @pytest.mark.parametrize('kind', ['line', 'area'])
    def test_plot_xlim_for_series(
        self,
        kind: str
    ) -> None: ...

    def test_plot_no_rows(self) -> None: ...

    def test_plot_no_numeric_data(self) -> None: ...

    @pytest.mark.parametrize('data, index', [([1, 2, 3, 4], [3, 2, 1, 0]), ([10, 50, 20, 30], [1910, 1920, 1980, 1950])])
    def test_plot_order(
        self,
        data: list[int],
        index: list[int]
    ) -> None: ...

    def test_style_single_ok(self) -> None: ...

    @pytest.mark.parametrize('index_name, old_label, new_label', [(None, '', 'new'), ('old', 'old', 'new'), (None, '', '')])
    @pytest.mark.parametrize('kind', ['line', 'area', 'bar', 'barh', 'hist'])
    def test_xlabel_ylabel_series(
        self,
        kind: str,
        index_name: Union[str, None],
        old_label: str,
        new_label: str
    ) -> None: ...

    @pytest.mark.parametrize('index', [pd.timedelta_range(start=0, periods=2, freq='D'), [pd.Timedelta(days=1), pd.Timedelta(days=2)]])
    def test_timedelta_index(
        self,
        index: Any
    ) -> None: ...

    def test_series_none_color(self) -> None: ...

    @pytest.mark.slow
    def test_plot_no_warning(
        self,
        ts: Series
    ) -> None: ...

    def test_secondary_y_subplot_axis_labels(self) -> None: ...

# Pytest fixtures
@pytest.fixture
def ts() -> Series: ...

@pytest.fixture
def series() -> Series: ...

# Module-level imports from test code (preserved as they appear in public API)
np_version_gte1p24: bool = ...
is_platform_linux: Any = ...
td: Any = ...
plotting: Any = ...
mpl: Any = ...
plt: Any = ...
DatetimeConverter: Any = ...
```