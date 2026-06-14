from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pytest
from pandas import DataFrame, Grouper, Series

ByArg = Union[
    str,
    List[str],
    List[int],
    Grouper,
    Callable[[Any], Any],
    List[Series],
    np.ndarray,
    Dict[int, int],
    Series,
]

def by(request: pytest.FixtureRequest) -> ByArg: ...
def groupby_series(request: pytest.FixtureRequest) -> bool: ...
def df_with_string_col() -> DataFrame: ...
def df_with_datetime_col() -> DataFrame: ...
def df_with_cat_col() -> DataFrame: ...

def _call_and_check(
    klass: Optional[Union[type[BaseException], Tuple[type[BaseException], ...]]],
    msg: str,
    how: Literal["method", "agg", "transform"],
    gb: Any,
    groupby_func: Union[str, Callable[..., Any]],
    args: Tuple[Any, ...],
    warn_msg: str = ...,
) -> None: ...

def test_groupby_raises_string(
    how: Literal["method", "agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    groupby_func: str,
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

def test_groupby_raises_string_udf(
    how: Literal["agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    df_with_string_col: DataFrame,
) -> None: ...

def test_groupby_raises_string_np(
    how: Literal["agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

def test_groupby_raises_datetime(
    how: Literal["method", "agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    groupby_func: str,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_datetime_udf(
    how: Literal["agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_datetime_np(
    how: Literal["agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_timedelta(
    func: Literal["prod", "cumprod", "skew", "kurt", "var"],
) -> None: ...

def test_groupby_raises_category(
    how: Literal["method", "agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    groupby_func: str,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_udf(
    how: Literal["agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_np(
    how: Literal["agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_on_category(
    how: Literal["method", "agg", "transform"],
    by: ByArg,
    groupby_series: bool,
    groupby_func: str,
    observed: bool,
    df_with_cat_col: DataFrame,
) -> None: ...