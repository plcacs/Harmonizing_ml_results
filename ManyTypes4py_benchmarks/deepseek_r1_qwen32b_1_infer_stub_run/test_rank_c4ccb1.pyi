import pytest
import numpy as np
from pandas import Series
from pandas._libs.algos import Infinity, NegInfinity
from pandas.api.types import CategoricalDtype

@pytest.fixture
def ser() -> Series[np.float64]:
    ...

@pytest.fixture
def results() -> tuple[str, np.ndarray[np.float64]]:
    ...

@pytest.fixture
def dtype() -> str:
    ...

def expected_dtype(dtype: str, method: str, pct: bool = False) -> str:
    ...

class TestSeriesRank:
    def test_rank(self, datetime_series: Series[Any]) -> None:
        ...

    def test_rank_categorical(self) -> None:
        ...

    def test_rank_nullable_integer(self) -> None:
        ...

    def test_rank_signature(self, s: Series[Any]) -> None:
        ...

    def test_rank_tie_methods(self, ser: Series[Any], results: tuple[str, np.ndarray[np.float64]], dtype: str, using_infer_string: bool) -> None:
        ...

    def test_rank_tie_methods_on_infs_nans(self, rank_method: str, na_option: str, ascending: bool, dtype: str, na_value: object, pos_inf: object, neg_inf: object) -> None:
        ...

    def test_rank_desc_mix_nans_infs(self) -> None:
        ...

    def test_rank_methods_series(self, rank_method: str, op: object, value: float) -> None:
        ...

    @pytest.mark.parametrize('ser, exp', [])
    def test_rank_dense_method(self, dtype: str, ser: list[np.float64], exp: list[np.float64]) -> None:
        ...

    def test_rank_descending(self, ser: Series[Any], results: tuple[str, np.ndarray[np.float64]], dtype: str, using_infer_string: bool) -> None:
        ...

    def test_rank_int(self, ser: Series[Any], results: tuple[str, np.ndarray[np.float64]]) -> None:
        ...

    def test_rank_object_bug(self) -> None:
        ...

    def test_rank_modify_inplace(self) -> None:
        ...

    def test_rank_ea_small_values(self) -> None:
        ...

@pytest.mark.parametrize('ser, exp', [])
def test_rank_dense_pct(dtype: str, ser: list[np.float64], exp: list[np.float64]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [])
def test_rank_min_pct(dtype: str, ser: list[np.float64], exp: list[np.float64]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [])
def test_rank_max_pct(dtype: str, ser: list[np.float64], exp: list[np.float64]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [])
def test_rank_average_pct(dtype: str, ser: list[np.float64], exp: list[np.float64]) -> None:
    ...

@pytest.mark.parametrize('ser, exp', [])
def test_rank_first_pct(dtype: str, ser: list[np.float64], exp: list[np.float64]) -> None:
    ...

@pytest.mark.single_cpu
def test_pct_max_many_rows() -> None:
    ...