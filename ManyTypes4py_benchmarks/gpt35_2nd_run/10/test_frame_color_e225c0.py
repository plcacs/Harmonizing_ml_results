import re
from typing import List, Tuple
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def _check_colors_box(bp: dict, box_c: str, whiskers_c: str, medians_c: str, caps_c: str = 'k', fliers_c: str = None) -> None:
    if fliers_c is None:
        fliers_c = 'k'
    _check_colors(bp['boxes'], linecolors=[box_c] * len(bp['boxes']))
    _check_colors(bp['whiskers'], linecolors=[whiskers_c] * len(bp['whiskers']))
    _check_colors(bp['medians'], linecolors=[medians_c] * len(bp['medians']))
    _check_colors(bp['fliers'], linecolors=[fliers_c] * len(bp['fliers']))
    _check_colors(bp['caps'], linecolors=[caps_c] * len(bp['caps'])

class TestDataFrameColor:

    def test_mpl2_color_cycle_str(self, color: int) -> None:
    
    def test_color_single_series_list(self) -> None:
    
    def test_rgb_tuple_color(self, color: Tuple[float, float, float, float]) -> None:
    
    def test_color_empty_string(self) -> None:
    
    def test_color_and_style_arguments(self) -> None:
    
    def test_color_and_marker(self, color: str, expected: List[str]) -> None:
    
    def test_color_and_style(self) -> None:
    
    def test_bar_colors(self) -> None:
    
    def test_bar_colors_custom(self) -> None:
    
    def test_bar_colors_cmap(self, colormap: str) -> None:
    
    def test_bar_colors_single_col(self) -> None:
    
    def test_bar_colors_green(self) -> None:
    
    def test_bar_user_colors(self) -> None:
    
    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self) -> None:
    
    def test_if_hexbin_xaxis_label_is_visible(self) -> None:
    
    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self) -> None:
    
    def test_scatter_with_c_column_name_with_colors(self, cmap: str) -> None:
    
    def test_scatter_with_c_column_name_without_colors(self) -> None:
    
    def test_scatter_colors(self) -> None:
    
    def test_scatter_colors_not_raising_warnings(self) -> None:
    
    def test_scatter_colors_default(self) -> None:
    
    def test_scatter_colors_white(self) -> None:
    
    def test_scatter_colorbar_different_cmap(self) -> None:
    
    def test_line_colors(self) -> None:
    
    def test_line_colors_cmap(self, colormap: str) -> None:
    
    def test_line_colors_single_col(self) -> None:
    
    def test_line_colors_single_color(self) -> None:
    
    def test_line_colors_hex(self) -> None:
    
    def test_dont_modify_colors(self) -> None:
    
    def test_line_colors_and_styles_subplots(self) -> None:
    
    def test_line_colors_and_styles_subplots_single_color_str(self, color: str) -> None:
    
    def test_line_colors_and_styles_subplots_custom_colors(self, color: str) -> None:
    
    def test_line_colors_and_styles_subplots_colormap_hex(self) -> None:
    
    def test_line_colors_and_styles_subplots_colormap_subplot(self, cmap: str) -> None:
    
    def test_line_colors_and_styles_subplots_single_col(self) -> None:
    
    def test_line_colors_and_styles_subplots_single_char(self) -> None:
    
    def test_line_colors_and_styles_subplots_list_styles(self) -> None:
    
    def test_area_colors(self) -> None:
    
    def test_area_colors_poly(self) -> None:
    
    def test_area_colors_stacked_false(self) -> None:
    
    def test_hist_colors(self) -> None:
    
    def test_hist_colors_single_custom(self) -> None:
    
    def test_hist_colors_cmap(self, colormap: str) -> None:
    
    def test_hist_colors_single_col(self) -> None:
    
    def test_hist_colors_single_color(self) -> None:
    
    def test_kde_colors(self) -> None:
    
    def test_kde_colors_cmap(self, colormap: str) -> None:
    
    def test_kde_colors_and_styles_subplots(self) -> None:
    
    def test_kde_colors_and_styles_subplots_single_col_str(self, colormap: str) -> None:
    
    def test_kde_colors_and_styles_subplots_custom_color(self) -> None:
    
    def test_kde_colors_and_styles_subplots_cmap(self, colormap: str) -> None:
    
    def test_kde_colors_and_styles_subplots_single_col(self) -> None:
    
    def test_kde_colors_and_styles_subplots_single_char(self) -> None:
    
    def test_kde_colors_and_styles_subplots_list(self) -> None:
    
    def test_boxplot_colors(self) -> None:
    
    def test_boxplot_colors_dict_colors(self) -> None:
    
    def test_boxplot_colors_default_color(self) -> None:
    
    def test_boxplot_colors_cmap(self, colormap: str) -> None:
    
    def test_boxplot_colors_single(self) -> None:
    
    def test_boxplot_colors_tuple(self) -> None:
    
    def test_boxplot_colors_invalid(self) -> None:
    
    def test_default_color_cycle(self) -> None:
    
    def test_no_color_bar(self) -> None:
    
    def test_mixing_cmap_and_colormap_raises(self) -> None:
    
    def test_passed_bar_colors(self) -> None:
    
    def test_rcParams_bar_colors(self) -> None:
    
    def test_colors_of_columns_with_same_name(self) -> None:
    
    def test_invalid_colormap(self) -> None:
    
    def test_dataframe_none_color(self) -> None:
