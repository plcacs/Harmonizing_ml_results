import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import JSONArray, JSONDtype, make_data
from typing import Any, Callable

@pytest.fixture
def dtype() -> JSONDtype:
    return JSONDtype()

@pytest.fixture
def data() -> JSONArray:
    """Length-100 PeriodArray for semantics test."""
    data = make_data()
    while len(data[0]) == len(data[1]):
        data = make_data()
    return JSONArray(data)

@pytest.fixture
def data_missing() -> JSONArray:
    """Length 2 array with [NA, Valid]"""
    return JSONArray([{}, {'a': 10}])

@pytest.fixture
def data_for_sorting() -> JSONArray:
    return JSONArray([{'b': 1}, {'c': 4}, {'a': 2, 'c': 3}])

@pytest.fixture
def data_missing_for_sorting() -> JSONArray:
    return JSONArray([{'b': 1}, {}, {'a': 4}])

@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    return operator.eq

@pytest.fixture
def data_for_grouping() -> JSONArray:
    return JSONArray([{'b': 1}, {'b': 1}, {}, {}, {'a': 0, 'c': 2}, {'a': 0, 'c': 2}, {'b': 1}, {'c': 2}])

class TestJSONArray(base.ExtensionTests):
    # ...

    def test_custom_asserts(self) -> None:
        data = JSONArray([collections.UserDict({'a': 1}), collections.UserDict({'b': 2}), collections.UserDict({'c': 3})])
        a = pd.Series(data)
        custom_assert_series_equal(a, a)
        custom_assert_frame_equal(a.to_frame(), a.to_frame())
        b = pd.Series(data.take([0, 0, 1]))
        msg = 'Series are different'
        with pytest.raises(AssertionError, match=msg):
            custom_assert_series_equal(a, b)
        with pytest.raises(AssertionError, match=msg):
            custom_assert_frame_equal(a.to_frame(), b.to_frame())

def custom_assert_series_equal(left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any) -> None:
    if left.dtype.name == 'json':
        assert left.dtype == right.dtype
        left = pd.Series(JSONArray(left.values.astype(object)), index=left.index, name=left.name)
        right = pd.Series(JSONArray(right.values.astype(object)), index=right.index, name=right.name)
    tm.assert_series_equal(left, right, *args, **kwargs)

def custom_assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
    obj_type = kwargs.get('obj', 'DataFrame')
    tm.assert_index_equal(left.columns, right.columns, exact=kwargs.get('check_column_type', 'equiv'), check_names=kwargs.get('check_names', True), check_exact=kwargs.get('check_exact', False), check_categorical=kwargs.get('check_categorical', True), obj=f'{obj_type}.columns')
    jsons = (left.dtypes == 'json').index
    for col in jsons:
        custom_assert_series_equal(left[col], right[col], *args, **kwargs)
    left = left.drop(columns=jsons)
    right = right.drop(columns=jsons)
    tm.assert_frame_equal(left, right, *args, **kwargs)
