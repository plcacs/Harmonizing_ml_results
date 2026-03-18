```python
import pytest
from pandas import DataFrame, Index, NamedAgg, Series
from pandas.errors import NumbaUtilError
from typing import Any

pytestmark: list[Any] = ...

def test_correct_function_signature() -> None: ...

def test_check_nopython_kwargs() -> None: ...

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_numba_vs_cython(
    jit: bool,
    frame_or_series: Any,
    nogil: Any,
    parallel: Any,
    nopython: Any,
    as_index: Any
) -> None: ...

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_cache(
    jit: bool,
    frame_or_series: Any,
    nogil: Any,
    parallel: Any,
    nopython: Any
) -> None: ...

def test_use_global_config() -> None: ...

@pytest.mark.parametrize('agg_kwargs', [{'func': ['min', 'max']}, {'func': 'min'}, {'func': {1: ['min', 'max'], 2: 'sum'}}, {'bmin': NamedAgg(column=1, aggfunc='min')}])
def test_multifunc_numba_vs_cython_frame(agg_kwargs: dict[str, Any]) -> None: ...

@pytest.mark.parametrize('func', ['sum', 'mean', 'var', 'std', 'min', 'max'])
def test_multifunc_numba_vs_cython_frame_noskipna(func: str) -> None: ...

@pytest.mark.parametrize('agg_kwargs,expected_func', [({'func': lambda values, index: values.sum()}, 'sum'), pytest.param({'func': [lambda values, index: values.sum(), lambda values, index: values.min()]}, ['sum', 'min'], marks=pytest.mark.xfail(reason="This doesn't work yet! Fails in nopython pipeline!"))])
def test_multifunc_numba_udf_frame(agg_kwargs: dict[str, Any], expected_func: Any) -> None: ...

@pytest.mark.parametrize('agg_kwargs', [{'func': ['min', 'max']}, {'func': 'min'}, {'min_val': 'min', 'max_val': 'max'}])
def test_multifunc_numba_vs_cython_series(agg_kwargs: dict[str, Any]) -> None: ...

@pytest.mark.single_cpu
@pytest.mark.parametrize('data,agg_kwargs', [(Series([1.0, 2.0, 3.0, 4.0, 5.0]), {'func': ['min', 'max']}), (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {'func': 'min'}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': ['min', 'max']}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': 'min'}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': {1: ['min', 'max'], 2: 'sum'}}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'min_col': NamedAgg(column=1, aggfunc='min')})])
def test_multifunc_numba_kwarg_propagation(data: Any, agg_kwargs: dict[str, Any]) -> None: ...

def test_args_not_cached() -> None: ...

def test_index_data_correctly_passed() -> None: ...

def test_engine_kwargs_not_cached() -> None: ...

@pytest.mark.filterwarnings('ignore')
def test_multiindex_one_key(nogil: Any, parallel: Any, nopython: Any) -> None: ...

def test_multiindex_multi_key_not_supported(nogil: Any, parallel: Any, nopython: Any) -> None: ...

def test_multilabel_numba_vs_cython(numba_supported_reductions: Any) -> None: ...

def test_multilabel_udf_numba_vs_cython() -> None: ...
```