from typing import Any, List, Tuple, Union

import datetime
import re
import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

def _call_and_check(klass: Union[None, Tuple[type, str]], msg: str, how: str, gb: Any, groupby_func: str, args: Tuple = (), warn_msg: str = ''):
    ...

def test_groupby_raises_string(how: str, by: Any, groupby_series: bool, groupby_func: str, df_with_string_col: DataFrame, using_infer_string: bool):
    ...

def test_groupby_raises_string_udf(how: str, by: Any, groupby_series: bool, df_with_string_col: DataFrame):
    ...

def test_groupby_raises_string_np(how: str, by: Any, groupby_series: bool, groupby_func_np: np.ufunc, df_with_string_col: DataFrame, using_infer_string: bool):
    ...

def test_groupby_raises_datetime(how: str, by: Any, groupby_series: bool, groupby_func: str, df_with_datetime_col: DataFrame):
    ...

def test_groupby_raises_datetime_udf(how: str, by: Any, groupby_series: bool, df_with_datetime_col: DataFrame):
    ...

def test_groupby_raises_datetime_np(how: str, by: Any, groupby_series: bool, groupby_func_np: np.ufunc, df_with_datetime_col: DataFrame):
    ...

def test_groupby_raises_timedelta(func: str):
    ...

def test_groupby_raises_category(how: str, by: Any, groupby_series: bool, groupby_func: str, df_with_cat_col: DataFrame):
    ...

def test_groupby_raises_category_udf(how: str, by: Any, groupby_series: bool, df_with_cat_col: DataFrame):
    ...

def test_groupby_raises_category_np(how: str, by: Any, groupby_series: bool, groupby_func_np: np.ufunc, df_with_cat_col: DataFrame):
    ...

def test_groupby_raises_category_on_category(how: str, by: Any, groupby_series: bool, groupby_func: str, observed: bool, df_with_cat_col: DataFrame):
    ...
