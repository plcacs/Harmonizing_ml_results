import numpy as np
import pytest
from pandas import DataFrame, Series, MultiIndex
from pandas.compat import IS64
from pandas.core.algorithms import safe_sort
from typing import Any, Callable, Optional, Union, List, Tuple, Dict, Any, ParamSpec, Concatenate

P = ParamSpec('P')

@pytest.fixture(params=[DataFrame, ...])
def pairwise_frames(request: pytest.FixtureRequest) -> DataFrame:
    ...

@pytest.fixture
def pairwise_target_frame() -> DataFrame:
    ...

@pytest.fixture
def pairwise_other_frame() -> DataFrame:
    ...

def test_rolling_cov(series: Series) -> None:
    ...

def test_rolling_corr(series: Series) -> None:
    ...

def test_rolling_corr_bias_correction() -> None:
    ...

@pytest.mark.parametrize('func', ['cov', 'corr'])
def test_rolling_pairwise_cov_corr(func: str, frame: DataFrame) -> None:
    ...

@pytest.mark.parametrize('method', ['corr', 'cov'])
def test_flex_binary_frame(method: str, frame: DataFrame) -> None:
    ...

@pytest.mark.parametrize('window', range(7))
def test_rolling_corr_with_zero_variance(window: int) -> None:
    ...

def test_corr_sanity() -> None:
    ...

def test_rolling_cov_diff_length() -> None:
    ...

def test_rolling_corr_diff_length() -> None:
    ...

@pytest.mark.parametrize('f', [Callable[[DataFrame], DataFrame], ...])
def test_rolling_functions_window_non_shrinkage_binary(f: Callable[[DataFrame], DataFrame]) -> None:
    ...

@pytest.mark.parametrize('f', [Callable[[DataFrame], DataFrame], ...])
def test_moment_functions_zero_length_pairwise(f: Callable[[DataFrame], DataFrame]) -> None:
    ...

class TestPairwise:
    @pytest.mark.parametrize('f', [Callable[[DataFrame], DataFrame], ...])
    def test_no_flex(self, pairwise_frames: DataFrame, pairwise_target_frame: DataFrame, f: Callable[[DataFrame], DataFrame]) -> None:
        ...

    @pytest.mark.parametrize('f', [Callable[[DataFrame], DataFrame], ...])
    def test_pairwise_with_self(self, pairwise_frames: DataFrame, pairwise_target_frame: DataFrame, f: Callable[[DataFrame], DataFrame]) -> None:
        ...

    @pytest.mark.parametrize('f', [Callable[[DataFrame], DataFrame], ...])
    def test_no_pairwise_with_self(self, pairwise_frames: DataFrame, pairwise_target_frame: DataFrame, f: Callable[[DataFrame], DataFrame]) -> None:
        ...

    @pytest.mark.parametrize('f', [Callable[[DataFrame, DataFrame], DataFrame], ...])
    def test_pairwise_with_other(self, pairwise_frames: DataFrame, pairwise_target_frame: DataFrame, pairwise_other_frame: DataFrame, f: Callable[[DataFrame, DataFrame], DataFrame]) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:RuntimeWarning')
    @pytest.mark.parametrize('f', [Callable[[DataFrame, DataFrame], DataFrame], ...])
    def test_no_pairwise_with_other(self, pairwise_frames: DataFrame, pairwise_other_frame: DataFrame, f: Callable[[DataFrame, DataFrame], DataFrame]) -> None:
        ...

    @pytest.mark.parametrize('f', [Callable[[DataFrame, Series], DataFrame], ...])
    def test_pairwise_with_series(self, pairwise_frames: DataFrame, pairwise_target_frame: DataFrame, f: Callable[[DataFrame, Series], DataFrame]) -> None:
        ...

    def test_corr_freq_memory_error(self) -> None:
        ...

    def test_cov_mulittindex(self) -> None:
        ...

    def test_multindex_columns_pairwise_func(self) -> None:
        ...