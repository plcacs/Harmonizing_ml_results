from typing import Any, Literal, Optional, Union
from typing_extensions import TypeAlias
from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import Infinity, NegInfinity
import pandas.util._test_decorators as td
from pandas import NA, NaT, Series, Timestamp
from pandas.api.types import CategoricalDtype
import pandas._testing as tm

@pytest.fixture
def ser() -> Series:
    ...

@pytest.fixture(params: list[list[Union[str, np.ndarray]]], ids: Any = ...)
def results(request: pytest.FixtureRequest) -> list[Union[str, np.ndarray]]:
    ...

@pytest.fixture(params: list[str])
def dtype(request: pytest.FixtureRequest) -> str:
    ...

def expected_dtype(dtype: str, method: str, pct: bool = ...) -> str:
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

    def test_rank_tie_methods(
        self,
        ser: Series,
        results: list[Union[str, np.ndarray]],
        dtype: str,
        using_infer_string: bool
    ) -> None:
        ...

    @pytest.mark.parametrize("na_option", ...)
    @pytest.mark.parametrize("dtype, na_value, pos_inf, neg_inf", ...)
    def test_rank_tie_methods_on_infs_nans(
        self,
        rank_method: str,
        na_option: str,
        ascending: bool,
        dtype: str,
        na_value: Any,
        pos_inf: Union[Infinity, float],
        neg_inf: Union[NegInfinity, float]
    ) -> None:
        ...

    def test_rank_desc_mix_nans_infs(self) -> None:
        ...

    @pytest.mark.parametrize("op, value", ...)
    def test_rank_methods_series(
        self,
        rank_method: str,
        op: operator.add | operator.mul,
        value: Union[int, float]
    ) -> None:
        ...

    @pytest.mark.parametrize("ser, exp", ...)
    def test_rank_dense_method(
        self,
        dtype: str,
        ser: list[Union[int, float]],
        exp: list[int]
    ) -> None:
        ...

    def test_rank_descending(
        self,
        ser: Series,
        results: list[Union[str, np.ndarray]],
        dtype: str,
        using_infer_string: bool
    ) -> None:
        ...

    def test_rank_int(
        self,
        ser: Series,
        results: list[Union[str, np.ndarray]]
    ) -> None:
        ...

    def test_rank_object_bug(self) -> None:
        ...

    def test_rank_modify_inplace(self) -> None:
        ...

    def test_rank_ea_small_values(self) -> None:
        ...

@pytest.mark.parametrize("ser, exp", ...)
def test_rank_dense_pct(
    dtype: str,
    ser: list[Union[int, float]],
    exp: list[float]
) -> None:
    ...

@pytest.mark.parametrize("ser, exp", ...)
def test_rank_min_pct(
    dtype: str,
    ser: list[Union[int, float]],
    exp: list[float]
) -> None:
    ...

@pytest.mark.parametrize("ser, exp", ...)
def test_rank_max_pct(
    dtype: str,
    ser: list[Union[int, float]],
    exp: list[float]
) -> None:
    ...

@pytest.mark.parametrize("ser, exp", ...)
def test_rank_average_pct(
    dtype: str,
    ser: list[Union[int, float]],
    exp: list[float]
) -> None:
    ...

@pytest.mark.parametrize("ser, exp", ...)
def test_rank_first_pct(
    dtype: str,
    ser: list[Union[int, float]],
    exp: list[float]
) -> None:
    ...

@pytest.mark.single_cpu
def test_pct_max_many_rows() -> None:
    ...