import datetime
import re
import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

@pytest.fixture
def by(request) -> Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series]:
    ...

@pytest.fixture
def groupby_series(request) -> bool:
    ...

@pytest.fixture
def df_with_string_col() -> DataFrame:
    ...

@pytest.fixture
def df_with_datetime_col() -> DataFrame:
    ...

@pytest.fixture
def df_with_cat_col() -> DataFrame:
    ...

def _call_and_check(
    klass: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]],
    msg: str,
    how: str,
    gb: Union[DataFrame, Series],
    groupby_func: Union[str, Callable],
    args: Tuple,
    warn_msg: str = ''
) -> None:
    ...

@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_string(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    groupby_func: str,
    df_with_string_col: DataFrame,
    using_infer_string: bool
) -> None:
    ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
def test_groupby_raises_string_udf(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    df_with_string_col: DataFrame
) -> None:
    ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
@pytest.mark.parametrize('groupby_func_np', [np.sum, np.mean])
def test_groupby_raises_string_np(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    groupby_func_np: Union[np.sum, np.mean],
    df_with_string_col: DataFrame,
    using_infer_string: bool
) -> None:
    ...

@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_datetime(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    groupby_func: str,
    df_with_datetime_col: DataFrame
) -> None:
    ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
def test_groupby_raises_datetime_udf(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    df_with_datetime_col: DataFrame
) -> None:
    ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
@pytest.mark.parametrize('groupby_func_np', [np.sum, np.mean])
def test_groupby_raises_datetime_np(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    groupby_func_np: Union[np.sum, np.mean],
    df_with_datetime_col: DataFrame
) -> None:
    ...

@pytest.mark.parametrize('func', ['prod', 'cumprod', 'skew', 'kurt', 'var'])
def test_groupby_raises_timedelta(func: str) -> None:
    ...

@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_category(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    groupby_func: str,
    df_with_cat_col: DataFrame
) -> None:
    ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
def test_groupby_raises_category_udf(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    df_with_cat_col: DataFrame
) -> None:
    ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
@pytest.mark.parametrize('groupby_func_np', [np.sum, np.mean])
def test_groupby_raises_category_np(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    groupby_func_np: Union[np.sum, np.mean],
    df_with_cat_col: DataFrame
) -> None:
    ...

@pytest.mark.filterwarnings('ignore:`groups` by one element list returns scalar is deprecated')
@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_category_on_category(
    how: str,
    by: Union[str, List[str], Grouper, Callable, np.ndarray, Dict[int, int], Series],
    groupby_series: bool,
    groupby_func: str,
    observed: bool,
    df_with_cat_col: DataFrame
) -> None:
    ...