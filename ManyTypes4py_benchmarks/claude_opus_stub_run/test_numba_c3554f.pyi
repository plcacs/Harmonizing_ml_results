from typing import Any

import numpy as np
import pytest
from pandas import DataFrame, Series

pytestmark: list[Any]
numba: Any

@pytest.fixture
def method(request: pytest.FixtureRequest) -> str: ...

@pytest.fixture
def arithmetic_numba_supported_operators(request: pytest.FixtureRequest) -> list[Any]: ...

@pytest.fixture
def roll_frame() -> DataFrame: ...

class TestEngine:
    def test_numba_vs_cython_apply(
        self,
        jit: bool,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        center: bool,
        step: int | None,
    ) -> None: ...
    def test_apply_numba_with_kwargs(self, roll_frame: DataFrame) -> None: ...
    def test_numba_min_periods(self) -> None: ...
    def test_numba_vs_cython_rolling_methods(
        self,
        data: DataFrame | Series,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: list[Any],
        step: int | None,
    ) -> None: ...
    def test_numba_vs_cython_expanding_methods(
        self,
        data: DataFrame | Series,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: list[Any],
    ) -> None: ...
    def test_cache_apply(
        self,
        jit: bool,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        step: int | None,
    ) -> None: ...
    def test_dont_cache_args(
        self,
        window: str,
        window_kwargs: dict[str, Any],
        nogil: bool,
        parallel: bool,
        nopython: bool,
        method: str,
    ) -> None: ...
    def test_dont_cache_engine_kwargs(self) -> None: ...

class TestEWM:
    def test_invalid_engine(self, grouper: Any, method: str) -> None: ...
    def test_invalid_engine_kwargs(self, grouper: Any, method: str) -> None: ...
    def test_cython_vs_numba(
        self,
        grouper: str,
        method: str,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        ignore_na: bool,
        adjust: bool,
    ) -> None: ...
    def test_cython_vs_numba_times(
        self,
        grouper: str,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        ignore_na: bool,
    ) -> None: ...

def test_use_global_config() -> None: ...
def test_invalid_kwargs_nopython() -> None: ...

class TestTableMethod:
    def test_table_series_valueerror(self) -> None: ...
    def test_table_method_rolling_methods(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: list[Any],
        step: int | None,
    ) -> None: ...
    def test_table_method_rolling_apply(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        step: int | None,
    ) -> None: ...
    def test_table_method_rolling_apply_col_order(self) -> None: ...
    def test_table_method_rolling_weighted_mean(self, step: int | None) -> None: ...
    def test_table_method_expanding_apply(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool,
    ) -> None: ...
    def test_table_method_expanding_methods(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: list[Any],
    ) -> None: ...
    def test_table_method_ewm(
        self,
        data: np.ndarray,
        method: str,
        nogil: bool,
        parallel: bool,
        nopython: bool,
    ) -> None: ...

def test_npfunc_no_warnings() -> None: ...