import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort

class BaseMethodsTests:
    """Various Series and DataFrame methods."""

    def test_hash_pandas_object(self, data: pd.Series) -> None:
        # ...
    @pytest.mark.parametrize('dropna', [True, False])
    def test_value_counts(self, all_data: pd.Series, dropna: bool) -> None:
        # ...
    @pytest.mark.parametrize('na_position', ['last', 'first'])
    def test_nargsort(self, data_missing_for_sorting: pd.Series, na_position: str) -> None:
        # ...
    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_argmin_argmax(self, data_for_sorting: pd.Series, data_missing_for_sorting: pd.Series, na_value: pd.NA) -> None:
        # ...
    @pytest.mark.parametrize('box', [pd.Series, lambda x: x])
    @pytest.mark.parametrize('method', [lambda x: x.unique(), pd.unique])
    def test_unique(self, data: pd.Series, box: callable, method: callable) -> None:
        # ...
    def test_factorize(self, data_for_grouping: pd.Series) -> None:
        # ...
    def test_fillna_limit_frame(self, data_missing: pd.Series) -> None:
        # ...
    def test_shift_zero_copies(self, data: pd.Series) -> None:
        # ...
    def test_searchsorted(self, data_for_sorting: pd.Series, as_series: bool) -> None:
        # ...
    def test_where_series(self, data: pd.Series, na_value: pd.NA, as_frame: bool) -> None:
        # ...
    @pytest.mark.parametrize('repeats', [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data: pd.Series, repeats: int | list, as_series: bool, use_numpy: bool) -> None:
        # ...
    @pytest.mark.parametrize('repeats, kwargs, error, msg', [(2, {'axis': 1}, ValueError, 'axis'), (-1, {}, ValueError, 'negative'), ([1, 2], {}, ValueError, 'shape'), (2, {'foo': 'bar'}, TypeError, "'foo'")])
    def test_repeat_raises(self, data: pd.Series, repeats: int | list, kwargs: dict, error: type, msg: str, use_numpy: bool) -> None:
        # ...
    def test_delete(self, data: pd.Series) -> None:
        # ...
    def test_insert(self, data: pd.Series) -> None:
        # ...
    def test_equals(self, data: pd.Series, na_value: pd.NA, as_series: bool, box: callable) -> None:
        # ...
