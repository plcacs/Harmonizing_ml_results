from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Index, Series, date_range, to_datetime
import pandas._testing as tm
from pandas.plotting._matplotlib.hist import _grouped_hist
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_colors,
    _check_legend_labels,
    _check_patches_all_filled,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    get_x_axis,
    get_y_axis,
)

mpl: Any
plt: Any

@pytest.fixture
def ts() -> Series: ...

class TestSeriesPlots:
    @pytest.mark.parametrize('kwargs', [{}, {'grid': False}, {'figsize': (8, 10)}])
    def test_hist_legacy_kwargs(self, ts: Series, kwargs: Dict[str, Any]) -> None: ...

    @pytest.mark.parametrize('kwargs', [{}, {'bins': 5}])
    def test_hist_legacy_kwargs_warning(self, ts: Series, kwargs: Dict[str, Any]) -> None: ...

    def test_hist_legacy_ax(self, ts: Series) -> None: ...

    def test_hist_legacy_ax_and_fig(self, ts: Series) -> None: ...

    def test_hist_legacy_fig(self, ts: Series) -> None: ...

    def test_hist_legacy_multi_ax(self, ts: Series) -> None: ...

    def test_hist_legacy_by_fig_error(self, ts: Series) -> None: ...

    def test_hist_bins_legacy(self) -> None: ...

    def test_hist_layout(self, hist_df: DataFrame) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('by, layout, axes_num, res_layout', [
        ['gender', (2, 1), 2, (2, 1)],
        ['gender', (3, -1), 2, (3, 1)],
        ['category', (4, 1), 4, (4, 1)],
        ['category', (2, -1), 4, (2, 2)],
        ['category', (3, -1), 4, (3, 2)],
        ['category', (-1, 4), 4, (1, 4)],
        ['classroom', (2, 2), 3, (2, 2)],
    ])
    def test_hist_layout_with_by(
        self,
        hist_df: DataFrame,
        by: str,
        layout: Tuple[int, int],
        axes_num: int,
        res_layout: Tuple[int, int],
    ) -> None: ...

    def test_hist_layout_with_by_shape(self, hist_df: DataFrame) -> None: ...

    def test_hist_no_overlap(self) -> None: ...

    def test_hist_by_no_extra_plots(self, hist_df: DataFrame) -> None: ...

    def test_plot_fails_when_ax_differs_from_figure(self, ts: Series) -> None: ...

    @pytest.mark.parametrize('histtype, expected', [
        ('bar', True),
        ('barstacked', True),
        ('step', False),
        ('stepfilled', True),
    ])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None: ...

    @pytest.mark.parametrize('by, expected_axes_num, expected_layout', [
        (None, 1, (1, 1)),
        ('b', 2, (1, 2)),
    ])
    def test_hist_with_legend(
        self,
        by: Optional[str],
        expected_axes_num: int,
        expected_layout: Tuple[int, int],
    ) -> None: ...

    @pytest.mark.parametrize('by', [None, 'b'])
    def test_hist_with_legend_raises(self, by: Optional[str]) -> None: ...

    def test_hist_kwargs(self, ts: Series) -> None: ...

    def test_hist_kwargs_horizontal(self, ts: Series) -> None: ...

    def test_hist_kwargs_align(self, ts: Series) -> None: ...

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_hist_kde(self, ts: Series) -> None: ...

    def test_hist_kde_plot_works(self, ts: Series) -> None: ...

    def test_hist_kde_density_works(self, ts: Series) -> None: ...

    @pytest.mark.xfail(reason='Api changed in 3.6.0')
    def test_hist_kde_logy(self, ts: Series) -> None: ...

    def test_hist_kde_color_bins(self, ts: Series) -> None: ...

    def test_hist_kde_color(self, ts: Series) -> None: ...

class TestDataFramePlots:
    @pytest.mark.slow
    def test_hist_df_legacy(self, hist_df: DataFrame) -> None: ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout(self) -> None: ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout2(self) -> None: ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout3(self) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [
        {'sharex': True, 'sharey': True},
        {'figsize': (8, 10)},
        {'bins': 5},
    ])
    def test_hist_df_legacy_layout_kwargs(self, kwargs: Dict[str, Any]) -> None: ...

    @pytest.mark.slow
    def test_hist_df_legacy_layout_labelsize_rot(self, frame_or_series: Union[DataFrame, Series]) -> None: ...

    @pytest.mark.slow
    def test_hist_df_legacy_rectangles(self) -> None: ...

    @pytest.mark.slow
    def test_hist_df_legacy_scale(self) -> None: ...

    @pytest.mark.slow
    def test_hist_df_legacy_external_error(self) -> None: ...

    def test_hist_non_numerical_or_datetime_raises(self) -> None: ...

    @pytest.mark.parametrize('layout_test', [
        {'layout': None, 'expected_size': (2, 2)},
        {'layout': (2, 2), 'expected_size': (2, 2)},
        {'layout': (4, 1), 'expected_size': (4, 1)},
        {'layout': (1, 4), 'expected_size': (1, 4)},
        {'layout': (3, 3), 'expected_size': (3, 3)},
        {'layout': (-1, 4), 'expected_size': (1, 4)},
        {'layout': (4, -1), 'expected_size': (4, 1)},
        {'layout': (-1, 2), 'expected_size': (2, 2)},
        {'layout': (2, -1), 'expected_size': (2, 2)},
    ])
    def test_hist_layout(self, layout_test: Dict[str, Any]) -> None: ...

    def test_hist_layout_error(self) -> None: ...

    def test_tight_layout(self) -> None: ...

    def test_hist_subplot_xrot(self) -> None: ...

    @pytest.mark.parametrize('column, expected', [
        (None, ['width', 'length', 'height']),
        (['length', 'width', 'height'], ['length', 'width', 'height']),
    ])
    def test_hist_column_order_unchanged(self, column: Optional[List[str]], expected: List[str]) -> None: ...

    @pytest.mark.parametrize('histtype, expected', [
        ('bar', True),
        ('barstacked', True),
        ('step', False),
        ('stepfilled', True),
    ])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None: ...

    @pytest.mark.parametrize('by', [None, 'c'])
    @pytest.mark.parametrize('column', [None, 'b'])
    def test_hist_with_legend(self, by: Optional[str], column: Optional[str]) -> None: ...

    @pytest.mark.parametrize('by', [None, 'c'])
    @pytest.mark.parametrize('column', [None, 'b'])
    def test_hist_with_legend_raises(self, by: Optional[str], column: Optional[str]) -> None: ...

    def test_hist_df_kwargs(self) -> None: ...

    def test_hist_df_with_nonnumerics(self) -> None: ...

    def test_hist_df_with_nonnumerics_no_bins(self) -> None: ...

    def test_hist_secondary_legend(self) -> None: ...

    def test_hist_secondary_secondary(self) -> None: ...

    def test_hist_secondary_primary(self) -> None: ...

    def test_hist_with_nans_and_weights(self) -> None: ...

class TestDataFrameGroupByPlots:
    def test_grouped_hist_legacy(self) -> None: ...

    def test_grouped_hist_legacy_axes_shape_no_col(self) -> None: ...

    def test_grouped_hist_legacy_single_key(self) -> None: ...

    def test_grouped_hist_legacy_grouped_hist_kwargs(self) -> None: ...

    def test_grouped_hist_legacy_grouped_hist(self) -> None: ...

    def test_grouped_hist_legacy_external_err(self) -> None: ...

    def test_grouped_hist_legacy_figsize_err(self) -> None: ...

    def test_grouped_hist_legacy2(self) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('msg, plot_col, by_col, layout', [
        ['Layout of 1x1 must be larger than required size 2', 'weight', 'gender', (1, 1)],
        ['Layout of 1x3 must be larger than required size 4', 'height', 'category', (1, 3)],
        ['At least one dimension of layout must be positive', 'height', 'category', (-1, -1)],
    ])
    def test_grouped_hist_layout_error(
        self,
        hist_df: DataFrame,
        msg: str,
        plot_col: str,
        by_col: str,
        layout: Tuple[int, int],
    ) -> None: ...

    @pytest.mark.slow
    def test_grouped_hist_layout_warning(self, hist_df: DataFrame) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('layout, check_layout, figsize', [
        [(4, 1), (4, 1), None],
        [(-1, 1), (4, 1), None],
        [(4, 2), (4, 2), (12, 8)],
    ])
    def test_grouped_hist_layout_figsize(
        self,
        hist_df: DataFrame,
        layout: Tuple[int, int],
        check_layout: Tuple[int, int],
        figsize: Optional[Tuple[int, int]],
    ) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs', [
        {},
        {'column': 'height', 'layout': (2, 2)},
    ])
    def test_grouped_hist_layout_by_warning(self, hist_df: DataFrame, kwargs: Dict[str, Any]) -> None: ...

    @pytest.mark.slow
    @pytest.mark.parametrize('kwargs, axes_num, layout', [
        [{'by': 'gender', 'layout': (3, 5)}, 2, (3, 5)],
        [{'column': ['height', 'weight', 'category']}, 3, (2, 2)],
    ])
    def test_grouped_hist_layout_axes(
        self,
        hist_df: DataFrame,
        kwargs: Dict[str, Any],
        axes_num: int,
        layout: Tuple[int, int],
    ) -> None: ...

    def test_grouped_hist_multiple_axes(self, hist_df: DataFrame) -> None: ...

    def test_grouped_hist_multiple_axes_no_cols(self, hist_df: DataFrame) -> None: ...

    def test_grouped_hist_multiple_axes_error(self, hist_df: DataFrame) -> None: ...

    def test_axis_share_x(self, hist_df: DataFrame) -> None: ...

    def test_axis_share_y(self, hist_df: DataFrame) -> None: ...

    def test_axis_share_xy(self, hist_df: DataFrame) -> None: ...

    @pytest.mark.parametrize('histtype, expected', [
        ('bar', True),
        ('barstacked', True),
        ('step', False),
        ('stepfilled', True),
    ])
    def test_histtype_argument(self, histtype: str, expected: bool) -> None: ...