import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
from typing import Any, Dict, List, Optional, Tuple, Union

@pytest.mark.parametrize('dropna, tuples, outputs', [(True, [['A', 'B'], ['B', 'A']], {'c': [13.0, 123.23], 'd': [13.0, 123.0], 'e': [13.0, 1.0]}), (False, [['A', 'B'], ['A', np.nan], ['B', 'A']], {'c': [13.0, 12.3, 123.23], 'd': [13.0, 233.0, 123.0], 'e': [13.0, 12.0, 1.0]})])
def test_groupby_dropna_multi_index_dataframe_nan_in_one_group(dropna: bool, tuples: List[List[Union[str, float]], outputs: Dict[str, List[float]], nulls_fixture: Any) -> None:
    df_list = [['A', 'B', 12, 12, 12], ['A', nulls_fixture, 12.3, 233.0, 12], ['B', 'A', 123.23, 123, 1], ['A', 'B', 1, 1, 1.0]]
    df = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd', 'e'])
    grouped = df.groupby(['a', 'b'], dropna=dropna).sum()
    mi = pd.MultiIndex.from_tuples(tuples, names=list('ab'))
    if not dropna:
        mi = mi.set_levels(['A', 'B', np.nan], level='b')
    expected = pd.DataFrame(outputs, index=mi)
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize('dropna, tuples, outputs', [(True, [['A', 'B'], ['B', 'A']], {'c': [12.0, 123.23], 'd': [12.0, 123.0], 'e': [12.0, 1.0]}), (False, [['A', 'B'], ['A', np.nan], ['B', 'A'], [np.nan, 'B']], {'c': [12.0, 13.3, 123.23, 1.0], 'd': [12.0, 234.0, 123.0, 1.0], 'e': [12.0, 13.0, 1.0, 1.0]})])
def test_groupby_dropna_multi_index_dataframe_nan_in_two_groups(dropna: bool, tuples: List[List[Union[str, float]]], outputs: Dict[str, List[float]], nulls_fixture: Any, nulls_fixture2: Any) -> None:
    df_list = [['A', 'B', 12, 12, 12], ['A', nulls_fixture, 12.3, 233.0, 12], ['B', 'A', 123.23, 123, 1], [nulls_fixture2, 'B', 1, 1, 1.0], ['A', nulls_fixture2, 1, 1, 1.0]]
    df = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd', 'e'])
    grouped = df.groupby(['a', 'b'], dropna=dropna).sum()
    mi = pd.MultiIndex.from_tuples(tuples, names=list('ab'))
    if not dropna:
        mi = mi.set_levels([['A', 'B', np.nan], ['A', 'B', np.nan]])
    expected = pd.DataFrame(outputs, index=mi)
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize('dropna, idx, outputs', [(True, ['A', 'B'], {'b': [123.23, 13.0], 'c': [123.0, 13.0], 'd': [1.0, 13.0]}), (False, ['A', 'B', np.nan], {'b': [123.23, 13.0, 12.3], 'c': [123.0, 13.0, 233.0], 'd': [1.0, 13.0, 12.0]})])
def test_groupby_dropna_normal_index_dataframe(dropna: bool, idx: List[Union[str, float]], outputs: Dict[str, List[float]]) -> None:
    df_list = [['B', 12, 12, 12], [None, 12.3, 233.0, 12], ['A', 123.23, 123, 1], ['B', 1, 1, 1.0]]
    df = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd'])
    grouped = df.groupby('a', dropna=dropna).sum()
    expected = pd.DataFrame(outputs, index=pd.Index(idx, name='a'))
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize('dropna, idx, expected', [(True, ['a', 'a', 'b', np.nan], pd.Series([3, 3], index=['a', 'b'])), (False, ['a', 'a', 'b', np.nan], pd.Series([3, 3, 3], index=['a', 'b', np.nan]))])
def test_groupby_dropna_series_level(dropna: bool, idx: List[Union[str, float]], expected: pd.Series) -> None:
    ser = pd.Series([1, 2, 3, 3], index=idx)
    result = ser.groupby(level=0, dropna=dropna).sum()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('dropna, expected', [(True, pd.Series([210.0, 350.0], index=['a', 'b'], name='Max Speed')), (False, pd.Series([210.0, 350.0, 20.0], index=['a', 'b', np.nan], name='Max Speed'))])
def test_groupby_dropna_series_by(dropna: bool, expected: pd.Series) -> None:
    ser = pd.Series([390.0, 350.0, 30.0, 20.0], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'], name='Max Speed')
    result = ser.groupby(['a', 'b', 'a', np.nan], dropna=dropna).mean()
    tm.assert_series_equal(result, expected)

def test_grouper_dropna_propagation(dropna: bool) -> None:
    df = pd.DataFrame({'A': [0, 0, 1, None], 'B': [1, 2, 3, None]})
    gb = df.groupby('A', dropna=dropna)
    assert gb._grouper.dropna == dropna

@pytest.mark.parametrize('index', [pd.RangeIndex(0, 4), list('abcd'), pd.MultiIndex.from_product([(1, 2), ('R', 'B')], names=['num', 'col'])])
def test_groupby_dataframe_slice_then_transform(dropna: bool, index: Union[pd.RangeIndex, List[str], pd.MultiIndex]) -> None:
    expected_data = {'B': [2, 2, 1, np.nan if dropna else 1]}
    df = pd.DataFrame({'A': [0, 0, 1, None], 'B': [1, 2, 3, None]}, index=index)
    gb = df.groupby('A', dropna=dropna)
    result = gb.transform(len)
    expected = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)
    result = gb[['B']].transform(len)
    expected = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)
    result = gb['B'].transform(len)
    expected = pd.Series(expected_data['B'], index=index, name='B')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('dropna, tuples, outputs', [(True, [['A', 'B'], ['B', 'A']], {'c': [13.0, 123.23], 'd': [12.0, 123.0], 'e': [1.0, 1.0]}), (False, [['A', 'B'], ['A', np.nan], ['B', 'A']], {'c': [13.0, 12.3, 123.23], 'd': [12.0, 233.0, 123.0], 'e': [1.0, 12.0, 1.0]})])
def test_groupby_dropna_multi_index_dataframe_agg(dropna: bool, tuples: List[List[Union[str, float]]], outputs: Dict[str, List[float]]) -> None:
    df_list = [['A', 'B', 12, 12, 12], ['A', None, 12.3, 233.0, 12], ['B', 'A', 123.23, 123, 1], ['A', 'B', 1, 1, 1.0]]
    df = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd', 'e'])
    agg_dict = {'c': 'sum', 'd': 'max', 'e': 'min'}
    grouped = df.groupby(['a', 'b'], dropna=dropna).agg(agg_dict)
    mi = pd.MultiIndex.from_tuples(tuples, names=list('ab'))
    if not dropna:
        mi = mi.set_levels(['A', 'B', np.nan], level='b')
    expected = pd.DataFrame(outputs, index=mi)
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.arm_slow
@pytest.mark.parametrize('datetime1, datetime2', [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01')), (pd.Timedelta('-2 days'), pd.Timedelta('-1 days')), (pd.Period('2020-01-01'), pd.Period('2020-02-01'))])
@pytest.mark.parametrize('dropna, values', [(True, [12, 3]), (False, [12, 3, 6])])
def test_groupby_dropna_datetime_like_data(dropna: bool, values: List[int], datetime1: Union[pd.Timestamp, pd.Timedelta, pd.Period], datetime2: Union[pd.Timestamp, pd.Timedelta, pd.Period], unique_nulls_fixture: Any, unique_nulls_fixture2: Any) -> None:
    df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6], 'dt': [datetime1, unique_nulls_fixture, datetime2, unique_nulls_fixture2, datetime1, datetime1]})
    if dropna:
        indexes = [datetime1, datetime2]
    else:
        indexes = [datetime1, datetime2, np.nan]
    grouped = df.groupby('dt', dropna=dropna).agg({'values': 'sum'})
    expected = pd.DataFrame({'values': values}, index=pd.Index(indexes, name='dt'))
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize('dropna, data, selected_data, levels', [pytest.param(False, {'groups': ['a', 'a', 'b', np.nan], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0, 0]}, ['a', 'b', np.nan], id='dropna_false_has_nan'), pytest.param(True, {'groups': ['a', 'a', 'b', np.nan], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0]}, None, id='dropna_true_has_nan'), pytest.param(False, {'groups': ['a', 'a', 'b', 'c'], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0, 0]}, None, id='dropna_false_no_nan'), pytest.param(True, {'groups': ['a', 'a', 'b', 'c'], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0, 0]}, None, id='dropna_true_no_nan')])
def test_groupby_apply_with_dropna_for_multi_index(dropna: bool, data: Dict[str, List[Union[str, float, None]]], selected_data: Dict[str, List[int]], levels: Optional[List[Union[str, float]]]) -> None:
    df = pd.DataFrame(data)
    gb = df.groupby('groups', dropna=dropna)
    result = gb.apply(lambda grp: pd.DataFrame({'values': range(len(grp))}))
    mi_tuples = tuple(zip(data['groups'], selected_data['values']))
    mi = pd.MultiIndex.from_tuples(mi_tuples, names=['groups', None])
    if not dropna and levels:
        mi = mi.set_levels(levels, level='groups')
    expected = pd.DataFrame(selected_data, index=mi)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('input_index', [None, ['a'], ['a', 'b']])
@pytest.mark.parametrize('keys', [['a'], ['a', 'b']])
@pytest.mark.parametrize('series', [True, False])
def test_groupby_dropna_with_multiindex_input(input_index: Optional[List[str]], keys: List[str], series: bool) -> None:
    obj = pd.DataFrame({'a': [1, np.nan], 'b': [1, 1], 'c': [2, 3]})
    expected = obj.set_index(keys)
    if series:
        expected = expected['c']
    elif input_index == ['a', 'b'] and keys == ['a']:
        expected = expected[['c']]
    if input_index is not None:
        obj = obj.set_index(input_index)
    gb = obj.groupby(keys, dropna=False)
    if series:
        gb = gb['c']
    result = gb.sum()
    tm.assert_equal(result, expected)

def test_groupby_nan_included() -> None:
    data = {'group': ['g1', np.nan, 'g1', 'g2', np.nan], 'B': [0, 1, 2, 3, 4]}
    df = pd.DataFrame(data)
    grouped = df.groupby('group', dropna=False)
    result = grouped.indices
    dtype = np.intp
    expected = {'g1': np.array([0, 2], dtype=dtype), 'g2': np.array([3], dtype=dtype), np.nan: np.array([1, 4], dtype=dtype)}
    for result_values, expected_values in zip(result.values(), expected.values()):
        tm.assert_numpy_array_equal(result_values, expected_values)
    assert np.isnan(list(result.keys())[2])
    assert list(result.keys())[0:2] == ['g1', 'g2']

def test_groupby_drop_nan_with_multi_index() -> None:
    df = pd.DataFrame([[np.nan, 0, 1]], columns=['a', 'b', 'c'])
    df = df.set_index(['a', 'b'])
    result = df.groupby(['a', 'b'], dropna=False).first()
    expected = df
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('sequence_index', range(3 ** 4))
@pytest.mark.parametrize('dtype', [None, 'UInt8', 'Int8', 'UInt16', 'Int16', 'UInt32', 'Int32', 'UInt64', 'Int64', 'Float32', 'Float64', 'category', 'string', pytest.param('string[pyarrow]', marks=pytest.mark.skipif(pa_version_under10p1, reason='pyarrow is not installed')), 'datetime64[ns]', 'period[D]', 'Sparse[float]'])
@pytest.mark.parametrize('test_series', [True, False])
def test_no_sort_keep_na(sequence_index: int, dtype: Optional[str], test_series: bool, as_index: bool) -> None:
    sequence = ''.join([{0: 'x', 1: 'y', 2: 'z'}[sequence_index // 3 ** k % 3] for k in range(4)])
    if dtype in ('string', 'string[pyarrow]'):
        uniques = {'x': 'x', 'y': 'y', 'z': pd.NA}
    elif dtype in ('datetime64[ns]', 'period[D]'):
        uniques = {'x': '2016-01-01', 'y': '2017-01-01', 'z': pd.NA}
    else:
        uniques = {'x': 1, 'y': 2, 'z': np.nan}
    df = pd.DataFrame({'key': pd.Series([uniques[label] for label in sequence], dtype=dtype), 'a': [0, 1, 2, 3]})
    gb = df.groupby('key', drop