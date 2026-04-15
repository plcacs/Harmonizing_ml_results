import datetime
from typing import Any, Callable, Literal, Optional, Tuple, Union
import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series
import pandas._testing as tm

@pytest.fixture
def by(
    request: pytest.FixtureRequest,
) -> Union[
    str,
    list[str],
    list[int],
    Grouper,
    Callable[[Any], Any],
    list[int],
    np.ndarray,
    dict[int, int],
    Series,
    list[Series],
]:
    ...

@pytest.fixture
def groupby_series(request: pytest.FixtureRequest) -> bool:
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
    klass: Optional[Union[type, Tuple[type, ...]]],
    msg: str,
    how: Literal["method", "agg", "transform"],
    gb: Any,
    groupby_func: Union[str, Callable[..., Any]],
    args: tuple[Any, ...],
    warn_msg: str = "",
) -> None:
    ...

@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_string(
    how: Literal["method", "agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    groupby_func: str,
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None:
    ...

@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_string_udf(
    how: Literal["agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    df_with_string_col: DataFrame,
) -> None:
    ...

@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_string_np(
    how: Literal["agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_string_col: DataFrame,
    using_infer_string: bool,
) -> None:
    ...

@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_datetime(
    how: Literal["method", "agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    groupby_func: str,
    df_with_datetime_col: DataFrame,
) -> None:
    ...

@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_datetime_udf(
    how: Literal["agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    df_with_datetime_col: DataFrame,
) -> None:
    ...

@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_datetime_np(
    how: Literal["agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_datetime_col: DataFrame,
) -> None:
    ...

@pytest.mark.parametrize("func", ["prod", "cumprod", "skew", "kurt", "var"])
def test_groupby_raises_timedelta(func: str) -> None:
    ...

@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category(
    how: Literal["method", "agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    groupby_func: str,
    df_with_cat_col: DataFrame,
) -> None:
    ...

@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_category_udf(
    how: Literal["agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    df_with_cat_col: DataFrame,
) -> None:
    ...

@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_category_np(
    how: Literal["agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    groupby_func_np: Callable[..., Any],
    df_with_cat_col: DataFrame,
) -> None:
    ...

@pytest.mark.filterwarnings("ignore:`groups` by one element list returns scalar is deprecated")
@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category_on_category(
    how: Literal["method", "agg", "transform"],
    by: Union[
        str,
        list[str],
        list[int],
        Grouper,
        Callable[[Any], Any],
        list[int],
        np.ndarray,
        dict[int, int],
        Series,
        list[Series],
    ],
    groupby_series: bool,
    groupby_func: str,
    observed: bool,
    df_with_cat_col: DataFrame,
) -> None:
    ...