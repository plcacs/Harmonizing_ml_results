import numpy as np
import pytest
from pandas import DataFrame, Series
from pandas.errors import NumbaUtilError
from typing import Any, Callable, Dict, List, Union

pytestmark: List[Any] = ...

def test_correct_function_signature() -> None: ...

def test_check_nopython_kwargs() -> None: ...

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_numba_vs_cython(
    jit: bool,
    frame_or_series: Any,
    nogil: bool,
    parallel: bool,
    nopython: bool,
    as_index: bool
) -> None: ...

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_cache(
    jit: bool,
    frame_or_series: Any,
    nogil: bool,
    parallel: bool,
    nopython: bool
) -> None: ...

def test_use_global_config() -> None: ...

@pytest.mark.parametrize('agg_func', [['min', 'max'], 'min', {'B': ['min', 'max'], 'C': 'sum'}])
def test_string_cython_vs_numba(
    agg_func: Union[List[str], str, Dict[str, Union[List[str], str]]],
    numba_supported_reductions: Any
) -> None: ...

def test_args_not_cached() -> None: ...

def test_index_data_correctly_passed() -> None: ...

def test_index_order_consistency_preserved() -> None: ...

def test_engine_kwargs_not_cached() -> None: ...

@pytest.mark.filterwarnings('ignore')
def test_multiindex_one_key(
    nogil: bool,
    parallel: bool,
    nopython: bool
) -> None: ...

def test_multiindex_multi_key_not_supported(
    nogil: bool,
    parallel: bool,
    nopython: bool
) -> None: ...

def test_multilabel_numba_vs_cython(
    numba_supported_reductions: Any
) -> None: ...

def test_multilabel_udf_numba_vs_cython() -> None: ...