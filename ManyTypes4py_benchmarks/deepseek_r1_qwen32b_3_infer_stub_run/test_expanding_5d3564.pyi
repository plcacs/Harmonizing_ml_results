from pandas import DataFrame, Series, Index, MultiIndex, DatetimeIndex
from typing import Union, Optional, Tuple, Callable, List, Any, Dict
import numpy as np

def test_doc_string() -> None:
    ...

def test_constructor(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

def test_constructor_invalid(frame_or_series: Union[DataFrame, Series], w: Union[float, str, np.ndarray]) -> None:
    ...

def test_empty_df_expanding(expander: Union[int, str]) -> None:
    ...

def test_missing_minp_zero() -> None:
    ...

def test_expanding() -> None:
    ...

def test_expanding_count_with_min_periods(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

def test_expanding_count_default_min_periods_with_null_values(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('df,expected,min_periods', [...])
def test_iter_expanding_dataframe(df: DataFrame, expected: List[DataFrame], min_periods: int) -> None:
    ...

@pytest.mark.parametrize('ser,expected,min_periods', [...])
def test_iter_expanding_series(ser: Series, expected: List[Series], min_periods: int) -> None:
    ...

def test_center_invalid() -> None:
    ...

def test_expanding_sem(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('method', ['skew', 'kurt'])
def test_expanding_skew_kurt_numerical_stability(method: str) -> None:
    ...

@pytest.mark.parametrize('window', [1, 3, 10, 20])
@pytest.mark.parametrize('method', ['min', 'max', 'average'])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('test_data', ['default', 'duplicates', 'nans'])
def test_rank(window: int, method: str, pct: bool, ascending: bool, test_data: str) -> None:
    ...

def test_expanding_corr(series: Series) -> None:
    ...

def test_expanding_count(series: Series) -> None:
    ...

def test_expanding_quantile(series: Series) -> None:
    ...

def test_expanding_cov(series: Series) -> None:
    ...

def test_expanding_cov_pairwise(frame: DataFrame) -> None:
    ...

def test_expanding_corr_pairwise(frame: DataFrame) -> None:
    ...

@pytest.mark.parametrize('func,static_comp', [...])
def test_expanding_func(func: str, static_comp: Callable, frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('func,static_comp', [...])
def test_expanding_min_periods(func: str, static_comp: Callable) -> None:
    ...

def test_expanding_apply(engine_and_raw: Tuple[str, bool], frame_or_series: Union[DataFrame, Series]) -> None:
    ...

def test_expanding_min_periods_apply(engine_and_raw: Tuple[str, bool]) -> None:
    ...

@pytest.mark.parametrize('f', [...])
def test_moment_functions_zero_length_pairwise(f: Callable) -> None:
    ...

@pytest.mark.parametrize('f', [...])
def test_moment_functions_zero_length(f: Callable) -> None:
    ...

def test_expanding_apply_empty_series(engine_and_raw: Tuple[str, bool]) -> None:
    ...

def test_expanding_apply_min_periods_0(engine_and_raw: Tuple[str, bool]) -> None:
    ...

def test_expanding_cov_diff_index() -> None:
    ...

def test_expanding_corr_diff_index() -> None:
    ...

def test_expanding_cov_pairwise_diff_length() -> None:
    ...

def test_expanding_corr_pairwise_diff_length() -> None:
    ...

@pytest.mark.parametrize('values,method,expected', [...])
def test_expanding_first_last(values: List[float], method: str, expected: List[float]) -> None:
    ...

@pytest.mark.parametrize('values,method,expected', [...])
def test_expanding_first_last_no_minp(values: List[float], method: str, expected: List[float]) -> None:
    ...

def test_expanding_apply_args_kwargs(engine_and_raw: Tuple[str, bool]) -> None:
    ...

@pytest.mark.parametrize('kernel', [...])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_numeric_only_frame(kernel: str, numeric_only: bool) -> None:
    ...

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
def test_numeric_only_corr_cov_frame(kernel: str, use_arg: bool, numeric_only: bool) -> None:
    ...

@pytest.mark.parametrize('kernel', [...])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_numeric_only_series(kernel: str, numeric_only: bool, dtype: type) -> None:
    ...

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
@pytest.mark.parametrize('dtype', [int, object])
def test_numeric_only_corr_cov_series(kernel: str, use_arg: bool, numeric_only: bool, dtype: type) -> None:
    ...