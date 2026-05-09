import numpy as np
import pytest
from pandas import DataFrame, Series
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
from pandas._testing import tm
from pandas.util.version import Version
from typing import List, Tuple, Callable, Dict, Optional

pytestmark: List[pytest.MarkDecorator] = ...

def test_correct_function_signature() -> None:
    ...

def test_check_nopython_kwargs() -> None:
    ...

def test_numba_vs_cython(jit: bool, frame_or_series: type[DataFrame | Series], nogil: bool, parallel: bool, nopython: bool, as_index: Optional[bool] = None) -> None:
    ...

def test_cache(jit: bool, frame_or_series: type[DataFrame | Series], nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_use_global_config() -> None:
    ...

def test_string_cython_vs_numba(agg_func: List[str] | str | Dict[str, List[str] | str], numba_supported_reductions: Tuple[Callable, Dict]) -> None:
    ...

def test_args_not_cached() -> None:
    ...

def test_index_data_correctly_passed() -> None:
    ...

def test_index_order_consistency_preserved() -> None:
    ...

def test_engine_kwargs_not_cached(nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_multiindex_one_key(nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_multiindex_multi_key_not_supported(nogil: bool, parallel: bool, nopython: bool) -> None:
    ...

def test_multilabel_numba_vs_cython(numba_supported_reductions: Tuple[Callable, Dict]) -> None:
    ...

def test_multilabel_udf_numba_vs_cython() -> None:
    ...