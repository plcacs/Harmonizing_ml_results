import numpy as np
from pandas import DataFrame, Series
from pandas.core.indexes.base import Index
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas.core.window import Rolling
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def test_correct_function_signature() -> None:
    ...

def test_check_nopython_kwargs() -> None:
    ...

def test_numba_vs_cython(jit: bool, frame_or_series: type[Union[DataFrame, Series]], nogil: bool, parallel: bool, nopython: bool, as_index: bool) -> None:
    ...

def test_cache(jit: bool, frame_or_series: type[Union[DataFrame, Series]], nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_use_global_config() -> None:
    ...

def test_multifunc_numba_vs_cython_frame(agg_kwargs: Dict[str, Any]) -> None:
    ...

def test_multifunc_numba_vs_cython_frame_noskipna(func: str) -> None:
    ...

def test_multifunc_numba_udf_frame(agg_kwargs: Dict[str, Any], expected_func: Union[str, List[str]]) -> None:
    ...

def test_multifunc_numba_vs_cython_series(agg_kwargs: Dict[str, Any]) -> None:
    ...

def test_multifunc_numba_kwarg_propagation(data: Union[Series, DataFrame], agg_kwargs: Dict[str, Any]) -> None:
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

def test_multilabel_numba_vs_cython(numba_supported_reductions: Tuple[str, Dict[str, Any]]) -> None:
    ...

def test_multilabel_udf_numba_vs_cython() -> None:
    ...