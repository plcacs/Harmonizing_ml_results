```python
import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, Timestamp
from typing import Any, Literal, Union, Sequence, Optional
import pytest

def test_first_last_nth(df: DataFrame) -> None: ...

@pytest.mark.parametrize
def test_first_last_with_na_object(
    method: str,
    nulls_fixture: Any
) -> None: ...

@pytest.mark.parametrize
def test_nth_with_na_object(
    index: int,
    nulls_fixture: Any
) -> None: ...

@pytest.mark.parametrize
def test_first_last_with_None(method: str) -> None: ...

@pytest.mark.parametrize
@pytest.mark.parametrize
def test_first_last_with_None_expanded(
    method: str,
    df: DataFrame,
    expected: DataFrame
) -> None: ...

def test_first_last_nth_dtypes() -> None: ...

def test_first_last_nth_dtypes2() -> None: ...

def test_first_last_nth_nan_dtype() -> None: ...

def test_first_strings_timestamps() -> None: ...

def test_nth() -> None: ...

def test_nth2() -> None: ...

def test_nth3() -> None: ...

def test_nth4() -> None: ...

def test_nth5() -> None: ...

def test_nth_bdays(unit: Any) -> None: ...

def test_nth_multi_grouper(three_group: DataFrame) -> None: ...

@pytest.mark.parametrize
def test_first_last_tz(
    data: dict,
    expected_first: dict,
    expected_last: dict
) -> None: ...

@pytest.mark.parametrize
def test_first_last_tz_multi_column(
    method: str,
    ts: Timestamp,
    alpha: str,
    unit: Any
) -> None: ...

@pytest.mark.parametrize
@pytest.mark.parametrize
def test_first_last_extension_array_keeps_dtype(
    values: Any,
    function: str
) -> None: ...

def test_nth_multi_index_as_expected() -> None: ...

@pytest.mark.parametrize
@pytest.mark.parametrize
def test_groupby_head_tail(
    op: str,
    n: int,
    expected_rows: list[int],
    columns: Optional[list[str]],
    as_index: bool
) -> None: ...

def test_group_selection_cache() -> None: ...

def test_nth_empty() -> None: ...

def test_nth_column_order() -> None: ...

@pytest.mark.parametrize
def test_nth_nan_in_grouper(dropna: Optional[str]) -> None: ...

@pytest.mark.parametrize
def test_nth_nan_in_grouper_series(dropna: Optional[str]) -> None: ...

def test_first_categorical_and_datetime_data_nat() -> None: ...

def test_first_multi_key_groupby_categorical() -> None: ...

@pytest.mark.parametrize
def test_groupby_last_first_nth_with_none(
    method: str,
    nulls_fixture: Any
) -> None: ...

@pytest.mark.parametrize
def test_slice(
    slice_test_df: DataFrame,
    slice_test_grouped: Any,
    arg: Any,
    expected_rows: list[int]
) -> None: ...

def test_nth_indexed(
    slice_test_df: DataFrame,
    slice_test_grouped: Any
) -> None: ...

def test_invalid_argument(slice_test_grouped: Any) -> None: ...

def test_negative_step(slice_test_grouped: Any) -> None: ...

def test_np_ints(
    slice_test_df: DataFrame,
    slice_test_grouped: Any
) -> None: ...

def test_groupby_nth_interval() -> None: ...

@pytest.mark.filterwarnings
def test_head_tail_dropna_true() -> None: ...

def test_head_tail_dropna_false() -> None: ...

@pytest.mark.parametrize
@pytest.mark.parametrize
def test_nth_after_selection(
    selection: Union[str, list[str]],
    dropna: Optional[str]
) -> None: ...

@pytest.mark.parametrize
def test_groupby_nth_int_like_precision(data: tuple) -> None: ...
```