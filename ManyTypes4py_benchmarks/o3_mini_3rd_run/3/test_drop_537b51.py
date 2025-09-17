from typing import Any, List, Tuple, Union, Optional, Type
import pytest
from pandas import Index, Series
import pandas._testing as tm
from pandas.api.types import is_bool_dtype

@pytest.mark.parametrize(
    'data, index, drop_labels, axis, expected_data, expected_index',
    [
        ([1, 2], ['one', 'two'], ['two'], 0, [1], ['one']),
        ([1, 2], ['one', 'two'], ['two'], 'rows', [1], ['one']),
        ([1, 1, 2], ['one', 'two', 'one'], ['two'], 0, [1, 2], ['one', 'one']),
        ([1, 1, 2], ['one', 'two', 'one'], 'two', 0, [1, 2], ['one', 'one']),
        ([1, 1, 2], ['one', 'two', 'one'], ['one'], 0, [1], ['two']),
        ([1, 1, 2], ['one', 'two', 'one'], 'one', 0, [1], ['two'])
    ]
)
def test_drop_unique_and_non_unique_index(
    data: List[Any],
    index: List[Any],
    drop_labels: Union[Any, List[Any]],
    axis: Union[int, str],
    expected_data: List[Any],
    expected_index: List[Any]
) -> None:
    ser: Series = Series(data=data, index=index)
    result: Series = ser.drop(drop_labels, axis=axis)
    expected: Series = Series(data=expected_data, index=expected_index)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'drop_labels, axis, error_type, error_desc',
    [
        ('bc', 0, KeyError, 'not found in axis'),
        (('a',), 0, KeyError, 'not found in axis'),
        ('one', 'columns', ValueError, 'No axis named columns')
    ]
)
def test_drop_exception_raised(
    drop_labels: Union[str, Tuple[str, ...]],
    axis: Union[int, str],
    error_type: Type[Exception],
    error_desc: str
) -> None:
    ser: Series = Series(range(3), index=list('abc'))
    with pytest.raises(error_type, match=error_desc):
        ser.drop(drop_labels, axis=axis)

def test_drop_with_ignore_errors() -> None:
    ser: Series = Series(range(3), index=list('abc'))
    result: Series = ser.drop('bc', errors='ignore')
    tm.assert_series_equal(result, ser)
    
    result = ser.drop(['a', 'd'], errors='ignore')
    expected: Series = ser.iloc[1:]
    tm.assert_series_equal(result, expected)
    
    ser = Series([2, 3], index=[True, False])
    assert is_bool_dtype(ser.index)
    assert ser.index.dtype == bool
    result = ser.drop(True)
    expected = Series([3], index=[False])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('index', [[1, 2, 3], [1, 1, 3]])
@pytest.mark.parametrize('drop_labels', [[], [1], [3]])
def test_drop_empty_list(index: List[Any], drop_labels: List[Any]) -> None:
    expected_index: List[Any] = [i for i in index if i not in drop_labels]
    series: Series = Series(index=index, dtype=object).drop(drop_labels)
    expected: Series = Series(index=expected_index, dtype=object)
    tm.assert_series_equal(series, expected)

@pytest.mark.parametrize(
    'data, index, drop_labels',
    [
        (None, [1, 2, 3], [1, 4]),
        (None, [1, 2, 2], [1, 4]),
        ([2, 3], [0, 1], [False, True])
    ]
)
def test_drop_non_empty_list(
    data: Optional[List[Any]],
    index: List[Any],
    drop_labels: List[Any]
) -> None:
    dtype: Any = object if data is None else None
    ser: Series = Series(data=data, index=index, dtype=dtype)
    with pytest.raises(KeyError, match='not found in axis'):
        ser.drop(drop_labels)

def test_drop_index_ea_dtype(any_numeric_ea_dtype: Any) -> None:
    df: Series = Series(100, index=Index([1, 2, 2], dtype=any_numeric_ea_dtype))
    idx: Index = Index([df.index[1]])
    result: Series = df.drop(idx)
    expected: Series = Series(100, index=Index([1], dtype=any_numeric_ea_dtype))
    tm.assert_series_equal(result, expected)