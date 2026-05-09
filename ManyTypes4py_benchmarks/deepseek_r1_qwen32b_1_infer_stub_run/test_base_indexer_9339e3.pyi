import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series
from pandas._testing import tm
from pandas.api.indexers import BaseIndexer, FixedForwardWindowIndexer
from pandas.core.indexers.objects import ExpandingIndexer, FixedWindowIndexer, VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
from pytest import MarkDecorator

def test_bad_get_window_bounds_signature() -> None:
    ...

def test_expanding_indexer() -> None:
    ...

def test_indexer_constructor_arg() -> None:
    ...

def test_indexer_accepts_rolling_args() -> None:
    ...

@pytest.mark.parametrize('func,np_func,expected,np_kwargs', [('count', len, [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, np.nan], {}), ('min', np.min, [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0, np.nan], {}), ('max', np.max, [2.0, 3.0, 4.0, 100.0, 100.0, 100.0, 8.0, 9.0, 9.0, np.nan], {}), ('std', np.std, [1.0, 1.0, 1.0, 55.71654452, 54.85739087, 53.9845657, 1.0, 1.0, 0.70710678, np.nan], {'ddof': 1}), ('var', np.var, [1.0, 1.0, 1.0, 3104.333333, 3009.333333, 2914.333333, 1.0, 1.0, 0.5, np.nan], {'ddof': 1}), ('median', np.median, [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 8.5, np.nan], {})])
def test_rolling_forward_window(frame_or_series: Union[DataFrame, Series], func: str, np_func: callable, expected: list, np_kwargs: dict, step: int) -> None:
    ...

def test_rolling_forward_skewness(frame_or_series: Union[DataFrame, Series], step: int) -> None:
    ...

@pytest.mark.parametrize('func,expected', [('cov', [2.0, 2.0, 2.0, 97.0, 2.0, -93.0, 2.0, 2.0, np.nan, np.nan]), ('corr', [1.0, 1.0, 1.0, 0.8704775290207161, 0.018229084250926637, -0.861357304646493, 1.0, 1.0, np.nan, np.nan])])
def test_rolling_forward_cov_corr(func: str, expected: list) -> None:
    ...

@pytest.mark.parametrize('closed,expected_data', [['right', [0.0, 1.0, 2.0, 3.0, 7.0, 12.0, 6.0, 7.0, 8.0, 9.0]], ['left', [0.0, 0.0, 1.0, 2.0, 5.0, 9.0, 5.0, 6.0, 7.0, 8.0]]])
def test_non_fixed_variable_window_indexer(closed: str, expected_data: list) -> None:
    ...

def test_variableoffsetwindowindexer_not_dti() -> None:
    ...

def test_variableoffsetwindowindexer_not_offset() -> None:
    ...

def test_fixed_forward_indexer_count(step: int) -> None:
    ...

@pytest.mark.parametrize(('end_value', 'values'), [(1, [0.0, 1, 1, 3, 2]), (-1, [0.0, 1, 0, 3, 1])])
@pytest.mark.parametrize(('func', 'args'), [('median', []), ('quantile', [0.5])])
def test_indexer_quantile_sum(end_value: int, values: list, func: str, args: list) -> None:
    ...

@pytest.mark.parametrize('indexer_class', [FixedWindowIndexer, FixedForwardWindowIndexer, ExpandingIndexer])
@pytest.mark.parametrize('window_size', [1, 2, 12])
@pytest.mark.parametrize('df_data', [{'a': [1, 1], 'b': [0, 1]}, {'a': [1, 2], 'b': [0, 1]}, {'a': [1] * 16, 'b': [np.nan, 1, 2, np.nan] + list(range(4, 16))}])
def test_indexers_are_reusable_after_groupby_rolling(indexer_class: type, window_size: int, df_data: dict) -> None:
    ...

@pytest.mark.parametrize('window_size, num_values, expected_start, expected_end', [(1, 1, [0], [1]), (1, 2, [0, 1], [1, 2]), (2, 1, [0], [1]), (2, 2, [0, 1], [2, 2]), (5, 12, range(12), list(range(5, 12)) + [12] * 5), (12, 5, range(5), [5] * 5), (0, 0, np.array([]), np.array([])), (1, 0, np.array([]), np.array([])), (0, 1, [0], [0])])
def test_fixed_forward_indexer_bounds(window_size: int, num_values: int, expected_start: list, expected_end: list, step: int) -> None:
    ...

def test_rolling_groupby_with_fixed_forward_specific(df: DataFrame, window_size: int, expected: Series) -> None:
    ...

def test_rolling_groupby_with_fixed_forward_many(group_keys: tuple, window_size: int) -> None:
    ...

def test_unequal_start_end_bounds() -> None:
    ...

def test_unequal_bounds_to_object() -> None:
    ...