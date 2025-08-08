from typing import Any, Tuple, Dict, List

def method(request: Any) -> str:
    return request.param

def arithmetic_numba_supported_operators(request: Any) -> Tuple[str, Dict]:
    return request.param

def roll_frame() -> DataFrame:
    return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40})

def test_numba_vs_cython_apply(self, jit: bool, nogil: Any, parallel: Any, nopython: Any, center: Any, step: Any) -> None:

def test_apply_numba_with_kwargs(self, roll_frame: DataFrame) -> None:

def test_numba_min_periods(self) -> None:

def test_numba_vs_cython_rolling_methods(self, data: Any, nogil: Any, parallel: Any, nopython: Any, arithmetic_numba_supported_operators: Tuple[str, Dict], step: Any) -> None:

def test_numba_vs_cython_expanding_methods(self, data: Any, nogil: Any, parallel: Any, nopython: Any, arithmetic_numba_supported_operators: Tuple[str, Dict]) -> None:

def test_cache_apply(self, jit: bool, nogil: Any, parallel: Any, nopython: Any, step: Any) -> None:

def test_dont_cache_args(self, window: str, window_kwargs: Dict, nogil: Any, parallel: Any, nopython: Any, method: str) -> None:

def test_dont_cache_engine_kwargs(self) -> None:

def test_invalid_engine(self, grouper: Any, method: str) -> None:

def test_invalid_engine_kwargs(self, grouper: Any, method: str) -> None:

def test_cython_vs_numba(self, grouper: Any, method: str, nogil: Any, parallel: Any, nopython: Any, ignore_na: Any, adjust: Any) -> None:

def test_cython_vs_numba_times(self, grouper: Any, nogil: Any, parallel: Any, nopython: Any, ignore_na: Any) -> None:

def test_use_global_config() -> None:

def test_invalid_kwargs_nopython() -> None:

def test_table_series_valueerror() -> None:

def test_table_method_rolling_methods(self, nogil: Any, parallel: Any, nopython: Any, arithmetic_numba_supported_operators: Tuple[str, Dict], step: Any) -> None:

def test_table_method_rolling_apply(self, nogil: Any, parallel: Any, nopython: Any, step: Any) -> None:

def test_table_method_rolling_apply_col_order() -> None:

def test_table_method_rolling_weighted_mean(self, step: Any) -> None:

def test_table_method_expanding_apply(self, nogil: Any, parallel: Any, nopython: Any) -> None:

def test_table_method_expanding_methods(self, nogil: Any, parallel: Any, nopython: Any, arithmetic_numba_supported_operators: Tuple[str, Dict]) -> None:

def test_table_method_ewm(self, data: Any, method: str, nogil: Any, parallel: Any, nopython: Any) -> None:

def test_npfunc_no_warnings() -> None:
