import datetime
import re
import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series
from pandas._testing import tm
from pandas.tests.groupby import get_groupby_method_args

@pytest.fixture
def by() -> object: ...

@pytest.fixture
def groupby_series() -> bool: ...

@pytest.fixture
def df_with_string_col() -> DataFrame: ...

@pytest.fixture
def df_with_datetime_col() -> DataFrame: ...

@pytest.fixture
def df_with_cat_col() -> DataFrame: ...

def _call_and_check(klass: type, msg: str, how: str, gb: object, groupby_func: str, args: tuple, warn_msg: str = '') -> None: ...

@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_string(how: str, by: object, groupby_series: bool, groupby_func: str, df_with_string_col: DataFrame, using_infer_string: bool) -> None: ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
def test_groupby_raises_string_udf(how: str, by: object, groupby_series: bool, df_with_string_col: DataFrame) -> None: ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
@pytest.mark.parametrize('groupby_func_np', [np.sum, np.mean])
def test_groupby_raises_string_np(how: str, by: object, groupby_series: bool, groupby_func_np: object, df_with_string_col: DataFrame, using_infer_string: bool) -> None: ...

@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_datetime(how: str, by: object, groupby_series: bool, groupby_func: str, df_with_datetime_col: DataFrame) -> None: ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
def test_groupby_raises_datetime_udf(how: str, by: object, groupby_series: bool, df_with_datetime_col: DataFrame) -> None: ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
@pytest.mark.parametrize('groupby_func_np', [np.sum, np.mean])
def test_groupby_raises_datetime_np(how: str, by: object, groupby_series: bool, groupby_func_np: object, df_with_datetime_col: DataFrame) -> None: ...

@pytest.mark.parametrize('func', ['prod', 'cumprod', 'skew', 'kurt', 'var'])
def test_groupby_raises_timedelta(func: str) -> None: ...

@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_category(how: str, by: object, groupby_series: bool, groupby_func: str, df_with_cat_col: DataFrame) -> None: ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
def test_groupby_raises_category_udf(how: str, by: object, groupby_series: bool, df_with_cat_col: DataFrame) -> None: ...

@pytest.mark.parametrize('how', ['agg', 'transform'])
@pytest.mark.parametrize('groupby_func_np', [np.sum, np.mean])
def test_groupby_raises_category_np(how: str, by: object, groupby_series: bool, groupby_func_np: object, df_with_cat_col: DataFrame) -> None: ...

@pytest.mark.filterwarnings('ignore:`groups` by one element list returns scalar is deprecated')
@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_category_on_category(how: str, by: object, groupby_series: bool, groupby_func: str, observed: bool, df_with_cat_col: DataFrame) -> None: ...