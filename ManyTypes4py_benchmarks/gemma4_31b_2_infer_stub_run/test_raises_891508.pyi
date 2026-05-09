import datetime
import numpy as np
import pytest
from typing import Any, Callable, Union, Optional, Type, Sequence
from pandas import Categorical, DataFrame, Grouper, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

def by(request: Any) -> Any: ...

def groupby_series(request: Any) -> bool: ...

def df_with_string_col() -> DataFrame: ...

def df_with_datetime_col() -> DataFrame: ...

def df_with_cat_col() -> DataFrame: ...

def _call_and_check(
    klass: Optional[Union[Type[BaseException], Sequence[Type[BaseException]]]],
    msg: str,
    how: str,
    gb: Union[DataFrameGroupBy, SeriesGroupBy],
    groupby_func: Union[str, Callable],
    args: Sequence[Any],
    warn_msg: str = '',
) -> None: ...

def test_groupby_raises_string(
    how: str,
    by: Any,
    groupby_series: bool,
    groupby_func: str,
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

def test_groupby_raises_string_udf(
    how: str,
    by: Any,
    groupby_series: bool,
    df_with_string_col: DataFrame,
) -> None: ...

def test_groupby_raises_string_np(
    how: str,
    by: Any,
    groupby_series: bool,
    groupby_func_np: Callable,
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

def test_groupby_raises_datetime(
    how: str,
    by: Any,
    groupby_series: bool,
    groupby_func: str,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_datetime_udf(
    how: str,
    by: Any,
    groupby_series: bool,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_datetime_np(
    how: str,
    by: Any,
    groupby_series: bool,
    groupby_func_np: Callable,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_timedelta(func: str) -> None: ...

def test_groupby_raises_category(
    how: str,
    by: Any,
    groupby_series: bool,
    groupby_func: str,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_udf(
    how: str,
    by: Any,
    groupby_series: bool,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_np(
    how: str,
    by: Any,
    groupby_series: bool,
    groupby_func_np: Callable,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_on_category(
    how: str,
    by: Any,
    groupby_series: bool,
    groupby_func: str,
    observed: bool,
    df_with_cat_col: DataFrame,
) -> None: ...