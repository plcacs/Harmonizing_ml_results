import pytest
from pandas import DataFrame, Series
from pandas.errors import NumbaUtilError
from pandas.compat import is_platform_arm
from pandas.util.version import Version
from typing import Any, Union, List, Dict, Optional, Callable, Type, Tuple
from numpy.typing import ArrayLike
from pandas.core.indexes.base import Index
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

def test_correct_function_signature() -> None:
    ...

def test_check_nopython_kwargs() -> None:
    ...

def test_numba_vs_cython(jit: bool, frame_or_series: Union[Type[DataFrame], Type[Series]], nogil: bool, parallel: bool, nopython: bool, as_index: bool) -> None:
    ...

def test_cache(jit: bool, frame_or_series: Union[Type[DataFrame], Type[Series]], nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_use_global_config() -> None:
    ...

def test_string_cython_vs_numba(agg_func: Union[List[str], str, Dict[str, Union[List[str], str]]], numba_supported_reductions: Tuple[Any, Dict[str, Any]]) -> None:
    ...

def test_args_not_cached() -> None:
    ...

def test_index_data_correctly_passed() -> None:
    ...

def test_index_order_consistency_preserved() -> None:
    ...

def test_engine_kwargs_not_cached() -> None:
    ...

def test_multiindex_one_key(nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_multiindex_multi_key_not_supported(nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_multilabel_numba_vs_cython(numba_supported_reductions: Tuple[Callable[[ArrayLike, Index], ArrayLike], Dict[str, Any]]) -> None:
    ...

def test_multilabel_udf_numba_vs_cython() -> None:
    ...