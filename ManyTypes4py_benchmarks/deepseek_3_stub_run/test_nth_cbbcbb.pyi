import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, Timestamp
from typing import Any, Literal, Union, Optional, List, Tuple
import pandas._testing as tm

def test_first_last_nth(df: DataFrame) -> None: ...

@pytest.mark.parametrize('method', ['first', 'last'])
def test_first_last_with_na_object(
    method: Literal['first', 'last'], 
    nulls_fixture: Any
) -> None: ...

@pytest.mark.parametrize('index', [0, -1])
def test_nth_with_na_object(
    index: Literal[0, -1], 
    nulls_fixture: Any
) -> None: ...

@pytest.mark.parametrize('method', ['first', 'last'])
def test_first_last_with_None(method: Literal['first', 'last']) -> None: ...

@pytest.mark.parametrize('method', ['first', 'last'])
@pytest.mark.parametrize('df, expected', [
    (
        DataFrame({'id': 'a', 'value': [None, 'foo', np.nan]}), 
        DataFrame({'value': ['foo']}, index=Index(['a'], name='id'))
    ), 
    (
        DataFrame({'id': 'a', 'value': [np.nan]}, dtype=object), 
        DataFrame({'value': [None]}, index=Index(['a'], name='id'))
    )
])
def test_first_last_with_None_expanded(
    method: Literal['first', 'last'], 
    df: DataFrame, 
    expected: DataFrame
) -> None: ...

def test_first_last_nth_dtypes() -> None: ...

def test_first_last_nth_dtypes2() -> None: ...

def test_first_last_nth_nan_dtype() -> None: ...

def test_first_strings_timestamps() -> None: ...

def test_nth() -> None: ...

def test_nth2() -> None: ...

def test_nth3() -> None: ...

def test_nth4() -> None: ...

def test_nth5() -> None: ...

def test_nth_bdays(unit: str) -> None: ...

def test_nth_multi_grouper(three_group: DataFrame) -> None: ...

@pytest.mark.parametrize('data, expected_first, expected_last', [
    (
        {'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}, 
        {'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}, 
        {'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}
    ), 
    (
        {
            'id': ['A', 'B', 'A'], 
            'time': [
                Timestamp('2012-01-01 13:00:00', tz='America/New_York'), 
                Timestamp('2012-02-01 14:00:00', tz='US/Central'), 
                Timestamp('2012-03-01 12:00:00', tz='Europe/London')
            ], 
            'foo': [1, 2, 3]
        }, 
        {
            'id': ['A', 'B'], 
            'time': [
                Timestamp('2012-01-01 13:00:00', tz='America/New_York'), 
                Timestamp('2012-02-01 14:00:00', tz='US/Central')
            ], 
            'foo': [1, 2]
        }, 
        {
            'id': ['A', 'B'], 
            'time': [
                Timestamp('2012-03-01 12:00:00', tz='Europe/London'), 
                Timestamp('2012-02-01 14:00:00', tz='US/Central')
            ], 
            'foo': [3, 2]
        }
    )
])
def test_first_last_tz(
    data: dict[str, Any], 
    expected_first: dict[str, Any], 
    expected_last: dict[str, Any]
) -> None: ...

@pytest.mark.parametrize('method, ts, alpha', [
    ['first', Timestamp('2013-01-01', tz='US/Eastern'), 'a'], 
    ['last', Timestamp('2013-01-02', tz='US/Eastern'), 'b']
])
def test_first_last_tz_multi_column(
    method: Literal['first', 'last'], 
    ts: Timestamp, 
    alpha: str, 
    unit: str
) -> None: ...

@pytest.mark.parametrize('values', [
    pd.array([True, False], dtype='boolean'), 
    pd.array([1, 2], dtype='Int64'), 
    pd.to_datetime(['2020-01-01', '2020-02-01']), 
    pd.to_timedelta([1, 2], unit='D')
])
@pytest.mark.parametrize('function', ['first', 'last', 'min', 'max'])
def test_first_last_extension_array_keeps_dtype(
    values: Union[pd.arrays.BooleanArray, pd.arrays.IntegerArray, pd.DatetimeIndex, pd.TimedeltaIndex], 
    function: Literal['first', 'last', 'min', 'max']
) -> None: ...

def test_nth_multi_index_as_expected() -> None: ...

@pytest.mark.parametrize('op, n, expected_rows', [
    ('head', -1, [0]), 
    ('head', 0, []), 
    ('head', 1, [0, 2]), 
    ('head', 7, [0, 1, 2]), 
    ('tail', -1, [1]), 
    ('tail', 0, []), 
    ('tail', 1, [1, 2]), 
    ('tail', 7, [0, 1, 2])
])
@pytest.mark.parametrize('columns', [None, [], ['A'], ['B'], ['A', 'B']])
def test_groupby_head_tail(
    op: Literal['head', 'tail'], 
    n: int, 
    expected_rows: List[int], 
    columns: Optional[List[str]], 
    as_index: bool
) -> None: ...

def test_group_selection_cache() -> None: ...

def test_nth_empty() -> None: ...

def test_nth_column_order() -> None: ...

@pytest.mark.parametrize('dropna', [None, 'any', 'all'])
def test_nth_nan_in_grouper(dropna: Optional[Literal['any', 'all']]) -> None: ...

@pytest.mark.parametrize('dropna', [None, 'any', 'all'])
def test_nth_nan_in_grouper_series(dropna: Optional[Literal['any', 'all']]) -> None: ...

def test_first_categorical_and_datetime_data_nat() -> None: ...

def test_first_multi_key_groupby_categorical() -> None: ...

@pytest.mark.parametrize('method', ['first', 'last', 'nth'])
def test_groupby_last_first_nth_with_none(
    method: Literal['first', 'last', 'nth'], 
    nulls_fixture: Any
) -> None: ...

@pytest.mark.parametrize('arg, expected_rows', [
    [slice(None, 3, 2), [0, 1, 4, 5]], 
    [slice(None, -2), [0, 2, 5]], 
    [[slice(None, 2), slice(-2, None)], [0, 1, 2, 3, 4, 6, 7]], 
    [[0, 1, slice(-2, None)], [0, 1, 2, 3, 4, 6, 7]]
])
def test_slice(
    slice_test_df: DataFrame, 
    slice_test_grouped: Any, 
    arg: Union[slice, List[Union[int, slice]]], 
    expected_rows: List[int]
) -> None: ...

def test_nth_indexed(slice_test_df: DataFrame, slice_test_grouped: Any) -> None: ...

def test_invalid_argument(slice_test_grouped: Any) -> None: ...

def test_negative_step(slice_test_grouped: Any) -> None: ...

def test_np_ints(slice_test_df: DataFrame, slice_test_grouped: Any) -> None: ...

def test_groupby_nth_interval() -> None: ...

@pytest.mark.filterwarnings('ignore:invalid value encountered in remainder:RuntimeWarning')
def test_head_tail_dropna_true() -> None: ...

def test_head_tail_dropna_false() -> None: ...

@pytest.mark.parametrize('selection', ('b', ['b'], ['b', 'c']))
@pytest.mark.parametrize('dropna', ['any', 'all', None])
def test_nth_after_selection(
    selection: Union[str, List[str]], 
    dropna: Optional[Literal['any', 'all']]
) -> None: ...

@pytest.mark.parametrize('data', [
    (Timestamp('2011-01-15 12:50:28.502376'), Timestamp('2011-01-20 12:50:28.593448')), 
    (24650000000000001, 24650000000000002)
])
def test_groupby_nth_int_like_precision(
    data: Tuple[Union[Timestamp, int], Union[Timestamp, int]]
) -> None: ...