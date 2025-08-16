from typing import List, Tuple, Union
import pandas as pd
import numpy as np

def bar_grad(a: Union[str, None] = None, b: Union[str, None] = None, c: Union[str, None] = None, d: Union[str, None] = None) -> List[Tuple[str, str]]:
def no_bar() -> List[Tuple[str, str]]:
def bar_to(x: float, color: str = '#d65f5f') -> List[Tuple[str, str]]:
def bar_from_to(x: float, y: float, color: str = '#d65f5f') -> List[Tuple[str, str]]:
def test_align_positive_cases(df_pos: pd.DataFrame, align: Union[str, float, np.ufunc], exp: List[List[Tuple[str, str]]]):
def test_align_negative_cases(df_neg: pd.DataFrame, align: Union[str, float, np.ufunc], exp: List[List[Tuple[str, str]]]):
def test_align_mixed_cases(df_mix: pd.DataFrame, align: Union[str, float, np.ufunc], exp: List[List[Tuple[str, str]]], nans: bool):
def test_align_axis(align: Union[str, int], exp: dict, axis: str):
def test_vmin_vmax_clipping(df_pos: pd.DataFrame, df_neg: pd.DataFrame, df_mix: pd.DataFrame, values: str, vmin: float, vmax: float, nullify: Union[None, str], align: str):
def test_vmin_vmax_widening(df_pos: pd.DataFrame, df_neg: pd.DataFrame, df_mix: pd.DataFrame, values: str, vmin: float, vmax: float, nullify: Union[None, str], align: str):
def test_numerics():
def test_colors_mixed(align: str, exp: List[Tuple[str, str]]):
def test_bar_align_height():
def test_bar_value_error_raises():
def test_bar_color_and_cmap_error_raises():
def test_bar_invalid_color_type_error_raises():
def test_styler_bar_with_NA_values():
def test_style_bar_with_pyarrow_NA_values():
