import pytest
from pandas import Index, Series
import pandas._testing as tm
from pandas.api.types import is_bool_dtype

@pytest.mark.parametrize('data, index, drop_labels, axis, expected_data, expected_index', [([1, 2], ['one', 'two'], ['two'], 0, [1], ['one']), ([1, 2], ['one', 'two'], ['two'], 'rows', [1], ['one']), ([1, 1, 2], ['one', 'two', 'one'], ['two'], 0, [1, 2], ['one', 'one']), ([1, 1, 2], ['one', 'two', 'one'], 'two', 0, [1, 2], ['one', 'one']), ([1, 1, 2], ['one', 'two', 'one'], ['one'], 0, [1], ['two']), ([1, 1, 2], ['one', 'two', 'one'], 'one', 0, [1], ['two'])])
def test_drop_unique_and_non_unique_index(data: Union[list[str], typing.Sequence[int], str], index: Union[list[str], typing.Sequence[int], str], axis: Union[pandas.DataFrame, int, list[str]], drop_labels: Union[pandas.DataFrame, int, list[str]], expected_data: Union[int, pandas.DataFrame, typing.Sequence[int]], expected_index: Union[int, pandas.DataFrame, typing.Sequence[int]]) -> None:
    ser = Series(data=data, index=index)
    result = ser.drop(drop_labels, axis=axis)
    expected = Series(data=expected_data, index=expected_index)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('drop_labels, axis, error_type, error_desc', [('bc', 0, KeyError, 'not found in axis'), (('a',), 0, KeyError, 'not found in axis'), ('one', 'columns', ValueError, 'No axis named columns')])
def test_drop_exception_raised(drop_labels: Union[mypy.types.FunctionLike, mypy.types.Overloaded, typing.Callable], axis: Union[mypy.types.FunctionLike, mypy.types.Overloaded, typing.Callable], error_type: Union[str, typing.Type, ValueError], error_desc: Union[str, typing.Type, ValueError]) -> None:
    ser = Series(range(3), index=list('abc'))
    with pytest.raises(error_type, match=error_desc):
        ser.drop(drop_labels, axis=axis)

def test_drop_with_ignore_errors() -> None:
    ser = Series(range(3), index=list('abc'))
    result = ser.drop('bc', errors='ignore')
    tm.assert_series_equal(result, ser)
    result = ser.drop(['a', 'd'], errors='ignore')
    expected = ser.iloc[1:]
    tm.assert_series_equal(result, expected)
    ser = Series([2, 3], index=[True, False])
    assert is_bool_dtype(ser.index)
    assert ser.index.dtype == bool
    result = ser.drop(True)
    expected = Series([3], index=[False])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('index', [[1, 2, 3], [1, 1, 3]])
@pytest.mark.parametrize('drop_labels', [[], [1], [3]])
def test_drop_empty_list(index: Union[tuple[int], Series], drop_labels: Union[pandas.core.indexes.api.Index, str]) -> None:
    expected_index = [i for i in index if i not in drop_labels]
    series = Series(index=index, dtype=object).drop(drop_labels)
    expected = Series(index=expected_index, dtype=object)
    tm.assert_series_equal(series, expected)

@pytest.mark.parametrize('data, index, drop_labels', [(None, [1, 2, 3], [1, 4]), (None, [1, 2, 2], [1, 4]), ([2, 3], [0, 1], [False, True])])
def test_drop_non_empty_list(data: Union[pandas.DataFrame, int, typing.Sequence[typing.Mapping]], index: Union[pandas.DataFrame, float, Series], drop_labels: Union[pandas.DataFrame, list[str]]) -> None:
    dtype = object if data is None else None
    ser = Series(data=data, index=index, dtype=dtype)
    with pytest.raises(KeyError, match='not found in axis'):
        ser.drop(drop_labels)

def test_drop_index_ea_dtype(any_numeric_ea_dtype: Union[bool, typing.Iterable[typing.Any]]) -> None:
    df = Series(100, index=Index([1, 2, 2], dtype=any_numeric_ea_dtype))
    idx = Index([df.index[1]])
    result = df.drop(idx)
    expected = Series(100, index=Index([1], dtype=any_numeric_ea_dtype))
    tm.assert_series_equal(result, expected)