from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TypeVar,
    overload,
)
import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Series,
    MultiIndex,
    DatetimeIndex,
    BusinessDay,
    Index,
    UUID,
)
from pandas.api.indexers import (
    BaseIndexer,
    FixedForwardWindowIndexer,
    ExpandingIndexer,
    FixedWindowIndexer,
    VariableOffsetWindowIndexer,
)

T = TypeVar("T", DataFrame, Series)

def test_bad_get_window_bounds_signature() -> None:
    ...

def test_expanding_indexer() -> None:
    ...

def test_indexer_constructor_arg() -> None:
    ...

def test_indexer_accepts_rolling_args() -> None:
    ...

@pytest.mark.parametrize(
    'func,np_func,expected,np_kwargs',
    [('count', len, List[float], Dict[str, Any]), ...]
)
def test_rolling_forward_window(
    frame_or_series: Callable[..., T],
    func: str,
    np_func: Callable[..., Any],
    expected: Union[List[float], T],
    np_kwargs: Dict[str, Any],
    step: int,
) -> None:
    ...

@pytest.mark.parametrize('end_value,values', [(1, [0.0, 1, 1, 3, 2]), (-1, [0.0, 1, 0, 3, 1])])
@pytest.mark.parametrize('func,args', [('median', []), ('quantile', [0.5])])
def test_indexer_quantile_sum(
    end_value: int,
    values: List[float],
    func: str,
    args: List[Any],
) -> None:
    ...

@pytest.mark.parametrize('group_keys', [(1,), (1, 2), (2, 1), (1, 1, 2), (1, 2, 1), (1, 1, 2, 2), (1, 2, 3, 2, 3), (1, 1, 2) * 4, (1, 2, 3) * 5])
@pytest.mark.parametrize('window_size', [1, 2, 3, 4, 5, 8, 20])
def test_rolling_groupby_with_fixed_forward_many(
    group_keys: Tuple[int, ...],
    window_size: int,
) -> None:
    ...

def test_unequal_start_end_bounds() -> None:
    ...

def test_unequal_bounds_to_object() -> None:
    ...

class CustomIndexer(BaseIndexer):
    def __init__(self, window_size: int, use_expanding: List[bool]) -> None:
        ...

    def get_window_bounds(
        self,
        num_values: int,
        min_periods: Optional[int],
        center: bool,
        closed: str,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

@pytest.mark.parametrize('func,expected', [('cov', List[float]), ('corr', List[float])])
def test_rolling_forward_cov_corr(
    func: str,
    expected: Union[List[float], Series],
) -> None:
    ...

@pytest.mark.parametrize('closed,expected_data', [['right', List[float]], ['left', List[float]]])
def test_non_fixed_variable_window_indexer(
    closed: str,
    expected_data: Union[List[float], DataFrame],
) -> None:
    ...

def test_variableoffsetwindowindexer_not_dti() -> None:
    ...

def test_variableoffsetwindowindexer_not_offset() -> None:
    ...

def test_fixed_forward_indexer_count(step: int) -> None:
    ...

def test_rolling_groupby_with_fixed_forward_specific(
    df: DataFrame,
    window_size: int,
    expected: Series,
) -> None:
    ...

@pytest.mark.parametrize('indexer_class', [FixedWindowIndexer, FixedForwardWindowIndexer, ExpandingIndexer])
@pytest.mark.parametrize('window_size', [1, 2, 12])
@pytest.mark.parametrize('df_data', [{'a': List[int], 'b': List[Any]}, ...])
def test_indexers_are_reusable_after_groupby_rolling(
    indexer_class: type[BaseIndexer],
    window_size: int,
    df_data: Dict[str, List[Any]],
) -> None:
    ...

@pytest.mark.parametrize('window_size,num_values,expected_start,expected_end', [(1,1,[0],[1]), ...])
def test_fixed_forward_indexer_bounds(
    window_size: int,
    num_values: int,
    expected_start: np.ndarray,
    expected_end: np.ndarray,
    step: int,
) -> None:
    ...