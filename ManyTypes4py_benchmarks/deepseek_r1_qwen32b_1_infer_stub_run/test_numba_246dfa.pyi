import numpy as np
import pytest
from pandas import DataFrame, Index, NamedAgg, Series
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
from pandas.util.version import Version

pytestmark = list[pytest.Mark]

def test_correct_function_signature() -> None:
    ...

def test_check_nopython_kwargs() -> None:
    ...

def test_numba_vs_cython(jit: bool, frame_or_series: type[DataFrame] | type[Series], nogil: bool, parallel: bool, nopython: bool, as_index: bool) -> None:
    ...

def test_cache(jit: bool, frame_or_series: type[DataFrame] | type[Series], nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_use_global_config() -> None:
    ...

def test_multifunc_numba_vs_cython_frame(agg_kwargs: dict) -> None:
    ...

def test_multifunc_numba_vs_cython_frame_noskipna(func: str) -> None:
    ...

def test_multifunc_numba_udf_frame(agg_kwargs: dict, expected_func: str | list[str]) -> None:
    ...

def test_multifunc_numba_vs_cython_series(agg_kwargs: dict) -> None:
    ...

def test_multifunc_numba_kwarg_propagation(data: DataFrame | Series, agg_kwargs: dict) -> None:
    ...

def test_args_not_cached() -> None:
    ...

def test_index_data_correctly_passed() -> None:
    ...

def test_engine_kwargs_not_cached() -> None:
    ...

def test_multiindex_one_key(nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_multiindex_multi_key_not_supported(nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_multilabel_numba_vs_cython(numba_supported_reductions: tuple[str, dict]) -> None:
    ...

def test_multilabel_udf_numba_vs_cython() -> None:
    ...