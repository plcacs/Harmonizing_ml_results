import datetime
from typing import Any, Callable, Union

import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series

@pytest.fixture
def by(request: pytest.FixtureRequest) -> Any: ...

@pytest.fixture
def groupby_series(request: pytest.FixtureRequest) -> bool: ...

@pytest.fixture
def df_with_string_col() -> DataFrame: ...

@pytest.fixture
def df_with_datetime_col() -> DataFrame: ...

@pytest.fixture
def df_with_cat_col() -> DataFrame: ...

def _call_and_check(
    klass: Union[type[Exception], tuple[type[Exception], ...], None],
    msg: str,
    how: str,
    gb: Any,
    groupby_func: Union[str, Callable[..., Any]],
    args: tuple[Any, ...],
    warn_msg: str = ...,
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
    groupby_func_np: Callable[..., Any],
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
    groupby_func_np: Callable[..., Any],
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
    groupby_func_np: Callable[..., Any],
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