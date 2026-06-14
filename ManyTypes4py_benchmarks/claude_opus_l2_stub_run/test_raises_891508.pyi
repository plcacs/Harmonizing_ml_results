import datetime
import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series

@pytest.fixture(params=['a', ['a'], ['a', 'b'], Grouper(key='a'), lambda x: x % 2, [0, 0, 0, 1, 2, 2, 2, 3, 3], np.array([0, 0, 0, 1, 2, 2, 2, 3, 3]), dict(zip(range(9), [0, 0, 0, 1, 2, 2, 2, 3, 3])), Series([1, 1, 1, 1, 1, 2, 2, 2, 2]), [Series([1, 1, 1, 1, 1, 2, 2, 2, 2]), Series([3, 3, 4, 4, 4, 4, 4, 3, 3])]])
def by(request: pytest.FixtureRequest) -> object: ...

@pytest.fixture(params=[True, False])
def groupby_series(request: pytest.FixtureRequest) -> bool: ...

@pytest.fixture
def df_with_string_col() -> DataFrame: ...

@pytest.fixture
def df_with_datetime_col() -> DataFrame: ...

@pytest.fixture
def df_with_cat_col() -> DataFrame: ...

def _call_and_check(
    klass: type[Exception] | tuple[type[Exception], ...] | None,
    msg: str,
    how: str,
    gb: object,
    groupby_func: object,
    args: tuple[object, ...] | list[object],
    warn_msg: str = ...,
) -> None: ...

def test_groupby_raises_string(
    how: str,
    by: object,
    groupby_series: bool,
    groupby_func: str,
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

def test_groupby_raises_string_udf(
    how: str,
    by: object,
    groupby_series: bool,
    df_with_string_col: DataFrame,
) -> None: ...

def test_groupby_raises_string_np(
    how: str,
    by: object,
    groupby_series: bool,
    groupby_func_np: np.ufunc,
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

def test_groupby_raises_datetime(
    how: str,
    by: object,
    groupby_series: bool,
    groupby_func: str,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_datetime_udf(
    how: str,
    by: object,
    groupby_series: bool,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_datetime_np(
    how: str,
    by: object,
    groupby_series: bool,
    groupby_func_np: np.ufunc,
    df_with_datetime_col: DataFrame,
) -> None: ...

def test_groupby_raises_timedelta(func: str) -> None: ...

def test_groupby_raises_category(
    how: str,
    by: object,
    groupby_series: bool,
    groupby_func: str,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_udf(
    how: str,
    by: object,
    groupby_series: bool,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_np(
    how: str,
    by: object,
    groupby_series: bool,
    groupby_func_np: np.ufunc,
    df_with_cat_col: DataFrame,
) -> None: ...

def test_groupby_raises_category_on_category(
    how: str,
    by: object,
    groupby_series: bool,
    groupby_func: str,
    observed: bool,
    df_with_cat_col: DataFrame,
) -> None: ...