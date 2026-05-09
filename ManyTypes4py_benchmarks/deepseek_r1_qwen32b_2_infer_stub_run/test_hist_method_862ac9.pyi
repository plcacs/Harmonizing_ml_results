"""Test cases for .hist method"""

import pytest
from pandas import Series, DataFrame
from matplotlib.pyplot import Figure, Axes

mpl: Figure
plt: Figure

@pytest.fixture
def ts() -> Series[float]:
    ...

class TestSeriesPlots:
    def test_hist_legacy_kwargs(self, ts: Series[float], kwargs: dict) -> None:
        ...

    def test_hist_legacy_kwargs_warning(self, ts: Series[float], kwargs: dict) -> None:
        ...

    def test_hist_legacy_ax(self, ts: Series[float]) -> None:
        ...

    def test_hist_legacy_ax_and_fig(self, ts: Series[float]) -> None:
        ...

    def test_hist_legacy_fig(self, ts: Series[float]) -> None:
        ...

    def test_hist_legacy_multi_ax(self, ts: Series[float]) -> None:
        ...

    def test_hist_legacy_by_fig_error(self, ts: Series[float]) -> None:
        ...

    def test_hist_bins_legacy(self) -> None:
        ...

    def test_hist_layout(self, hist_df: DataFrame) -> None:
        ...

    @pytest.mark.slow
    @pytest.mark.parametrize('by, layout, axes_num, res_layout', [[str, tuple, int, tuple]] * 7)
    def test_hist_layout_with_by(self, hist_df: DataFrame, by: str, layout: tuple, axes_num: int, res_layout: tuple) -> None:
        ...

    def test_hist_layout_with_by_shape(self, hist_df: DataFrame) -> None:
        ...

    def test_hist_no_overlap(self) -> None:
        ...

    def test_hist_by_no_extra_plots(self, hist_df: DataFrame) -> None:
        ...

    def test_plot_fails_when_ax_differs_from_figure(self, ts: Series[float]) -> None:
        ...

    @pytest.mark.parametrize('histtype, expected', [[str, bool]] * 4)
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        ...

    @pytest.mark.parametrize('by, expected_axes_num, expected_layout', [[str, int, tuple]] * 2)
    def test_hist_with_legend(self, by: str, expected_axes_num: int, expected_layout: tuple) -> None:
        ...

    @pytest.mark.parametrize('by', [None, str])
    def test_hist_with_legend_raises(self, by: str | None) -> None:
        ...

    def test_hist_kwargs(self, ts: Series[float]) -> None:
        ...

    def test_hist_kwargs_horizontal(self, ts: Series[float]) -> None:
        ...

    def test_hist_kwargs_align(self, ts: Series[float]) -> None:
        ...

    @pytest.mark.xfail
    def test_hist_kde(self, ts: Series[float]) -> None:
        ...

    def test_hist_kde_plot_works(self, ts: Series[float]) -> None:
        ...

    def test_hist_kde_density_works(self, ts: Series[float]) -> None:
        ...

    @pytest.mark.xfail
    def test_hist_kde_logy(self, ts: Series[float]) -> None:
        ...

    def test_hist_kde_color_bins(self, ts: Series[float]) -> None:
        ...

    def test_hist_kde_color(self, ts: Series[float]) -> None:
        ...

class TestDataFramePlots:
    @pytest.mark.slow
    def test_hist_df_legacy(self, hist_df: DataFrame) -> None:
        ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout(self) -> None:
        ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout2(self) -> None:
        ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout3(self) -> None:
        ...

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [{}, {'grid': bool}, {'figsize': tuple}, {'bins': int}])
    def test_hist_df_legacy_layout_kwargs(self, kwargs: dict) -> None:
        ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout_labelsize_rot(self, frame_or_series: Series | DataFrame) -> None:
        ...

    @pytest.mark.slow
    def test_hist_df_legacy_rectangles(self) -> None:
        ...

    @pytest.mark.slow
    def test_hist_df_legacy_scale(self) -> None:
        ...

    @pytest.mark.slow
    def test_hist_df_legacy_external_error(self) -> None:
        ...

    def test_hist_non_numerical_or_datetime_raises(self) -> None:
        ...

    @pytest.mark.parametrize('layout_test', [dict] * 9)
    def test_hist_layout(self, layout_test: dict) -> None:
        ...

    def test_hist_layout_error(self) -> None:
        ...

    def test_tight_layout(self) -> None:
        ...

    def test_hist_subplot_xrot(self) -> None:
        ...

    @pytest.mark.parametrize('column, expected', [[str, list]] * 2)
    def test_hist_column_order_unchanged(self, column: str | None, expected: list) -> None:
        ...

    @pytest.mark.parametrize('histtype, expected', [[str, bool]] * 4)
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        ...

    @pytest.mark.parametrize('by', [None, str])
    @pytest.mark.parametrize('column', [None, str])
    def test_hist_with_legend(self, by: str | None, column: str | None) -> None:
        ...

    @pytest.mark.parametrize('by', [None, str])
    @pytest.mark.parametrize('column', [None, str])
    def test_hist_with_legend_raises(self, by: str | None, column: str | None) -> None:
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

    @pytest.mark.slow
    @pytest.mark.parametrize('msg, plot_col, by_col, layout', [[str, str, str, tuple]] * 3)
    def test_grouped_hist_layout_error(self, hist_df: DataFrame, msg: str, plot_col: str, by_col: str, layout: tuple) -> None:
        ...

    @pytest.mark.slow
    def test_grouped_hist_layout_warning(self, hist_df: DataFrame) -> None:
        ...

    @pytest.mark.slow
    @pytest.mark.parametrize('layout, check_layout, figsize', [[tuple, tuple, tuple]] * 3)
    def test_grouped_hist_layout_figsize(self, hist_df: DataFrame, layout: tuple, check_layout: tuple, figsize: tuple | None) -> None:
        ...

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [dict] * 2)
    def test_grouped_hist_layout_by_warning(self, hist_df: DataFrame, kwargs: dict) -> None:
        ...

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs, axes_num, layout', [[dict, int, tuple]] * 2)
    def test_grouped_hist_layout_axes(self, hist_df: DataFrame, kwargs: dict, axes_num: int, layout: tuple) -> None:
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

    @pytest.mark.parametrize('histtype, expected', [[str, bool]] * 4)
    def test_histtype_argument(self, histtype: str, expected: bool) -> None:
        ...