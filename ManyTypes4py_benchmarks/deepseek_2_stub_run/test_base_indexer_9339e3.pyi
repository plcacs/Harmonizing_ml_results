```python
import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series
from pandas.api.indexers import BaseIndexer, FixedForwardWindowIndexer
from pandas.core.indexers.objects import ExpandingIndexer, FixedWindowIndexer, VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
from typing import Any, List, Tuple, Union

def test_bad_get_window_bounds_signature() -> None: ...

def test_expanding_indexer() -> None: ...

def test_indexer_constructor_arg() -> None: ...

def test_indexer_accepts_rolling_args() -> None: ...

def test_rolling_forward_window(
    frame_or_series: Any,
    func: str,
    np_func: Any,
    expected: List[float],
    np_kwargs: dict,
    step: int
) -> None: ...

def test_rolling_forward_skewness(
    frame_or_series: Any,
    step: int
) -> None: ...

def test_rolling_forward_cov_corr(
    func: str,
    expected: List[float]
) -> None: ...

def test_non_fixed_variable_window_indexer(
    closed: str,
    expected_data: List[float]
) -> None: ...

def test_variableoffsetwindowindexer_not_dti() -> None: ...

def test_variableoffsetwindowindexer_not_offset() -> None: ...

def test_fixed_forward_indexer_count(
    step: int
) -> None: ...

def test_indexer_quantile_sum(
    end_value: int,
    values: List[float],
    func: str,
    args: List[Any]
) -> None: ...

def test_indexers_are_reusable_after_groupby_rolling(
    indexer_class: Any,
    window_size: int,
    df_data: dict
) -> None: ...

def test_fixed_forward_indexer_bounds(
    window_size: int,
    num_values: int,
    expected_start: Any,
    expected_end: Any,
    step: int
) -> None: ...

def test_rolling_groupby_with_fixed_forward_specific(
    df: DataFrame,
    window_size: int,
    expected: Series
) -> None: ...

def test_rolling_groupby_with_fixed_forward_many(
    group_keys: Tuple[int, ...],
    window_size: int
) -> None: ...

def test_unequal_start_end_bounds() -> None: ...

def test_unequal_bounds_to_object() -> None: ...
```