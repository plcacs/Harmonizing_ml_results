from datetime import datetime as Timestamp
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from pandas import NA, NaT, Series
from pandas._libs.algos import Infinity, NegInfinity
from pandas.api.types import CategoricalDtype

@pytest.fixture
def ser() -> Series:
    ...

@pytest.fixture
def results(request: Any) -> Tuple[str, np.ndarray]:
    ...

@pytest.fixture
def dtype(request: Any) -> str:
    ...

def expected_dtype(dtype: str, method: str, pct: bool = False) -> str:
    ...

class TestSeriesRank:
    def test_rank(self, datetime_series: Series) -> None:
        ...

    def test_rank_categorical(self) -> None:
        ...

    def test_rank_nullable_integer(self) -> None:
        ...

    def test_rank_signature(self) -> None:
        ...

    def test_rank_tie_methods(self, ser: Series, results: Tuple[str, np.ndarray], dtype: str, using_infer_string: bool) -> None:
        ...

    def test_rank_tie_methods_on_infs_nans(self, rank_method: str, na_option: str, ascending: bool, dtype: str, na_value: Any, pos_inf: Any, neg_inf: Any) -> None:
        ...

    def test_rank_desc_mix_nans_infs(self) -> None:
        ...

    def test_rank_methods_series(self, rank_method: str, op: Any, value: Union[int, float]) -> None:
        ...

    def test_rank_dense_method(self, dtype: str, ser: List[float], exp: List[float]) -> None:
        ...

    def test_rank_descending(self, ser: Series, results: Tuple[str, np.ndarray], dtype: str, using_infer_string: bool) -> None:
        ...

    def test_rank_int(self, ser: Series, results: Tuple[str, np.ndarray]) -> None:
        ...

    def test_rank_object_bug(self) -> None:
        ...

    def test_rank_modify_inplace(self) -> None:
        ...

    def test_rank_ea_small_values(self) -> None:
        ...

@pytest.mark.parametrize('ser, exp', [...])
def test_rank_dense_pct(dtype: str, ser: List[float], exp: List[float]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [...])
def test_rank_min_pct(dtype: str, ser: List[float], exp: List[float]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [...])
def test_rank_max_pct(dtype: str, ser: List[float], exp: List[float]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [...])
def test_rank_average_pct(dtype: str, ser: List[float], exp: List[float]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [...])
def test_rank_first_pct(dtype: str, ser: List[float], exp: List[float]) -> None:
    ...

@pytest.mark.single_cpu
def test_pct_max_many_rows() -> None:
    ...