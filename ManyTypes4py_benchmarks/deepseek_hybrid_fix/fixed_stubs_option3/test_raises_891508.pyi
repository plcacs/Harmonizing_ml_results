import datetime
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

def by(request: Any) -> Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]]: ...

def groupby_series(request: Any) -> bool: ...

def df_with_string_col() -> DataFrame: ...

def df_with_datetime_col() -> DataFrame: ...

def df_with_cat_col() -> DataFrame: ...

def _call_and_check(
    klass: Optional[Union[Type, Tuple[Type, ...]]],
    msg: str,
    how: Literal["method", "agg", "transform"],
    gb: Any,
    groupby_func: Any,
    args: Union[Tuple[Any, ...], List[Any]],
    warn_msg: str = "",
) -> None: ...

@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_string(
    how: Literal["method", "agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    groupby_func: str,
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_string_udf(
    how: Literal["agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    df_with_string_col: DataFrame,
) -> None: ...

@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_string_np(
    how: Literal["agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None: ...

@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_datetime(
    how: Literal["method", "agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    groupby_func: str,
    df_with_datetime_col: DataFrame,
) -> None: ...

@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_datetime_udf(
    how: Literal["agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    df_with_datetime_col: DataFrame,
) -> None: ...

@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_datetime_np(
    how: Literal["agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_datetime_col: DataFrame,
) -> None: ...

@pytest.mark.parametrize("func", ["prod", "cumprod", "skew", "kurt", "var"])
def test_groupby_raises_timedelta(func: str) -> None: ...

@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category(
    how: Literal["method", "agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    groupby_func: str,
    df_with_cat_col: DataFrame,
) -> None: ...

@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_category_udf(
    how: Literal["agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    df_with_cat_col: DataFrame,
) -> None: ...

@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_category_np(
    how: Literal["agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_cat_col: DataFrame,
) -> None: ...

@pytest.mark.filterwarnings("ignore:`groups` by one element list returns scalar is deprecated")
@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category_on_category(
    how: Literal["method", "agg", "transform"],
    by: Union[str, List[str], List[int], Grouper, Callable[[Any], Any], np.ndarray, Dict[int, int], Series, List[Series]],
    groupby_series: bool,
    groupby_func: str,
    observed: bool,
    df_with_cat_col: DataFrame,
) -> None: ...