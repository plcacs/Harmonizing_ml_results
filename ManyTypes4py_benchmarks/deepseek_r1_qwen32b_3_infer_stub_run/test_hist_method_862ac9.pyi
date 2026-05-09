"""Test cases for .hist method"""

from matplotlib.pyplot import Figure
from numpy import float64
from pandas import DataFrame, Index, Series, DatetimeIndex
from pandas._testing import tm
from pytest import fixture, mark
from typing import Any, Dict, List, Optional, Tuple, Union

@fixture
def ts() -> Series:
    ...

class TestSeriesPlots:
    @mark.parametrize('kwargs', [Dict[str, Any], Dict[str, Any]])
    def test_hist_legacy_kwargs(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        ...

    @mark.parametrize('kwargs', [Dict[str, Any], Dict[str, Any]])
    def test_hist_legacy_kwargs_warning(self, ts: Series, kwargs: Dict[str, Any]) -> None:
        ...

    def test_hist_legacy_ax(self, ts: Series) -> None:
        ...

    def test_hist_legacy_ax_and_fig(self, ts: Series) -> None:
        ...

    def test_hist_legacy_fig(self, ts: Series) -> None:
        ...

    def test_hist_legacy_multi_ax(self, ts: Series) -> None:
        ...

    def test_hist_legacy_by_fig_error(self, ts: Series) -> None:
        ...

    def test_hist_bins_legacy(self) -> None:
        ...

    def test_hist_layout(self, hist_df: DataFrame) -> None:
        ...

    @mark.slow
    @mark.parametrize('by, layout, axes_num, res_layout', List[Tuple[Union[str, None], Tuple[int, int], int, Tuple[int, int]]])
    def test_hist_layout_with_by(self, hist_df: DataFrame, by: Union[str, None], layout: Tuple[int, int], axes_num: int, res_layout: Tuple[int, int]) -> None:
        ...

    def test_hist_layout_with_by_shape(self, hist_df: DataFrame) -> None:
        ...

    def test_hist_no_overlap(self) -> None:
        ...

    def test_hist_by_no_extra_plots(self, hist_df: DataFrame) -> None:
        ...

    def test_plot_fails_when_ax_differs_from_figure(self, ts: Series) -> None:
        ...

    @mark.parametrize('histtype, expected', [('bar', bool), ('barstacked', bool), ('step', bool), ('stepfilled', bool)])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        ...

    @mark.parametrize('by, expected_axes_num, expected_layout', [(None, int, Tuple[int, int]), ('b', int, Tuple[int, int])])
    def test_hist_with_legend(self, by: Union[str, None], expected_axes_num: int, expected_layout: Tuple[int, int]) -> None:
        ...

    @mark.parametrize('by', [None, 'b'])
    def test_hist_with_legend_raises(self, by: Union[str, None]) -> None:
        ...

    def test_hist_kwargs(self, ts: Series) -> None:
        ...

    def test_hist_kwargs_horizontal(self, ts: Series) -> None:
        ...

    def test_hist_kwargs_align(self, ts: Series) -> None:
        ...

    @mark.xfail(reason='Api changed in 3.6.0')
    def test_hist_kde(self, ts: Series) -> None:
        ...

    def test_hist_kde_plot_works(self, ts: Series) -> None:
        ...

    def test_hist_kde_density_works(self, ts: Series) -> None:
        ...

    @mark.xfail(reason='Api changed in 3.6.0')
    def test_hist_kde_logy(self, ts: Series) -> None:
        ...

    def test_hist_kde_color_bins(self, ts: Series) -> None:
        ...

    def test_hist_kde_color(self, ts: Series) -> None:
        ...

class TestDataFramePlots:
    @mark.slow
    def test_hist_df_legacy(self, hist_df: DataFrame) -> None:
        ...

    @mark.slow
    def test_hist_df_legacy_layout(self) -> None:
        ...

    @mark.slow
    def test_hist_df_legacy_layout2(self) -> None:
        ...

    @mark.slow
    def test_hist_df_legacy_layout3(self) -> None:
        ...

    @mark.slow
    @mark.parametrize('kwargs', [{'sharex': bool, 'sharey': bool}, {'figsize': Tuple[int, int]}, {'bins': int}])
    def test_hist_df_legacy_layout_kwargs(self, kwargs: Dict[str, Any]) -> None:
        ...

    @mark.slow
    def test_hist_df_legacy_layout_labelsize_rot(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    @mark.slow
    def test_hist_df_legacy_rectangles(self) -> None:
        ...

    @mark.slow
    def test_hist_df_legacy_scale(self) -> None:
        ...

    @mark.slow
    def test_hist_df_legacy_external_error(self) -> None:
        ...

    def test_hist_non_numerical_or_datetime_raises(self) -> None:
        ...

    @mark.parametrize('layout_test', List[Dict[str, Union[Tuple[int, int], Tuple[int, int]]]])
    def test_hist_layout(self, layout_test: Dict[str, Union[Tuple[int, int], Tuple[int, int]]]) -> None:
        ...

    def test_hist_layout_error(self) -> None:
        ...

    def test_tight_layout(self) -> None:
        ...

    def test_hist_subplot_xrot(self) -> None:
        ...

    @mark.parametrize('column, expected', [(None, List[str]), (List[str], List[str])])
    def test_hist_column_order_unchanged(self, column: Union[List[str], None], expected: List[str]) -> None:
        ...

    @mark.parametrize('histtype, expected', [('bar', bool), ('barstacked', bool), ('step', bool), ('stepfilled', bool)])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        ...

    @mark.parametrize('by', [None, 'c'])
    @mark.parametrize('column', [None, 'b'])
    def test_hist_with_legend(self, by: Union[str, None], column: Union[str, None]) -> None:
        ...

    @mark.parametrize('by', [None, 'c'])
    @mark.parametrize('column', [None, 'b'])
    def test_hist_with_legend_raises(self, by: Union[str, None], column: Union[str, None]) -> None:
        ...

    def test_hist_df_kwargs(self) -> None:
        ...

    def test_hist_df_with_nonnumerics(self) -> None:
        ...

    def test_hist_df_with_nonnumerics_no_bins(self) -> None:
        ...

    def test_hist_secondary_legend(self) -> None:
        ...

    def test_hist_secondary_secondary(self) -> None:
        ...

    def test_hist_secondary_primary(self) -> None:
        ...

    def test_hist_with_nans_and_weights(self) -> None:
        ...

class TestDataFrameGroupByPlots:
    def test_grouped_hist_legacy(self) -> None:
        ...

    def test_grouped_hist_legacy_axes_shape_no_col(self) -> None:
        ...

    def test_grouped_hist_legacy_single_key(self) -> None:
        ...

    def test_grouped_hist_legacy_grouped_hist_kwargs(self) -> None:
        ...

    def test_grouped_hist_legacy_grouped_hist(self) -> None:
        ...

    def test_grouped_hist_legacy_external_err(self) -> None:
        ...

    def test_grouped_hist_legacy_figsize_err(self) -> None:
        ...

    def test_grouped_hist_legacy2(self) -> None:
        ...

    @mark.slow
    @mark.parametrize('msg, plot_col, by_col, layout', List[Tuple[str, str, str, Tuple[int, int]]])
    def test_grouped_hist_layout_error(self, hist_df: DataFrame, msg: str, plot_col: str, by_col: str, layout: Tuple[int, int]) -> None:
        ...

    @mark.slow
    def test_grouped_hist_layout_warning(self, hist_df: DataFrame) -> None:
        ...

    @mark.slow
    @mark.parametrize('layout, check_layout, figsize', List[Tuple[Union[Tuple[int, int], None], Tuple[int, int], Optional[Tuple[int, int]]]])
    def test_grouped_hist_layout_figsize(self, hist_df: DataFrame, layout: Union[Tuple[int, int], None], check_layout: Tuple[int, int], figsize: Optional[Tuple[int, int]]) -> None:
        ...

    @mark.slow
    @mark.parametrize('kwargs', [Dict[str, Any], Dict[str, Any]])
    def test_grouped_hist_layout_by_warning(self, hist_df: DataFrame, kwargs: Dict[str, Any]) -> None:
        ...

    @mark.slow
    @mark.parametrize('kwargs, axes_num, layout', List[Tuple[Dict[str, Any], int, Tuple[int, int]]])
    def test_grouped_hist_layout_axes(self, hist_df: DataFrame, kwargs: Dict[str, Any], axes_num: int, layout: Tuple[int, int]) -> None:
        ...

    def test_grouped_hist_multiple_axes(self, hist_df: DataFrame) -> None:
        ...

    def test_grouped_hist_multiple_axes_no_cols(self, hist_df: DataFrame) -> None:
        ...

    def test_grouped_hist_multiple_axes_error(self, hist_df: DataFrame) -> None:
        ...

    def test_axis_share_x(self, hist_df: DataFrame) -> None:
        ...

    def test_axis_share_y(self, hist_df: DataFrame) -> None:
        ...

    def test_axis_share_xy(self, hist_df: DataFrame) -> None:
        ...

    @mark.parametrize('histtype, expected', [('bar', bool), ('barstacked', bool), ('step', bool), ('stepfilled', bool)])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        ...