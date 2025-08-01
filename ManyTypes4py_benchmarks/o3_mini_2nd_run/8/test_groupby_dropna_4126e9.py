from typing import Any, Dict, List, Union, Optional, Tuple
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

@pytest.mark.parametrize(
    'dropna, tuples, outputs',
    [
        (
            True,
            [['A', 'B'], ['B', 'A']],
            {'c': [13.0, 123.23], 'd': [13.0, 123.0], 'e': [13.0, 1.0]},
        ),
        (
            False,
            [['A', 'B'], ['A', np.nan], ['B', 'A']],
            {'c': [13.0, 12.3, 123.23], 'd': [13.0, 233.0, 123.0], 'e': [13.0, 12.0, 1.0]},
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_nan_in_one_group(
    dropna: bool, tuples: List[List[Any]], outputs: Dict[str, List[float]], nulls_fixture: Any
) -> None:
    df_list: List[List[Any]] = [['A', 'B', 12, 12, 12],
                                ['A', nulls_fixture, 12.3, 233.0, 12],
                                ['B', 'A', 123.23, 123, 1],
                                ['A', 'B', 1, 1, 1.0]]
    df: pd.DataFrame = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd', 'e'])
    grouped: pd.DataFrame = df.groupby(['a', 'b'], dropna=dropna).sum()
    mi: pd.MultiIndex = pd.MultiIndex.from_tuples([tuple(x) for x in tuples], names=list('ab'))
    if not dropna:
        mi = mi.set_levels(['A', 'B', np.nan], level='b')
    expected: pd.DataFrame = pd.DataFrame(outputs, index=mi)
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize(
    'dropna, tuples, outputs',
    [
        (
            True,
            [['A', 'B'], ['B', 'A']],
            {'c': [12.0, 123.23], 'd': [12.0, 123.0], 'e': [12.0, 1.0]},
        ),
        (
            False,
            [['A', 'B'], ['A', np.nan], ['B', 'A'], [np.nan, 'B']],
            {'c': [12.0, 13.3, 123.23, 1.0], 'd': [12.0, 234.0, 123.0, 1.0], 'e': [12.0, 13.0, 1.0, 1.0]},
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_nan_in_two_groups(
    dropna: bool,
    tuples: List[List[Any]],
    outputs: Dict[str, List[float]],
    nulls_fixture: Any,
    nulls_fixture2: Any,
) -> None:
    df_list: List[List[Any]] = [
        ['A', 'B', 12, 12, 12],
        ['A', nulls_fixture, 12.3, 233.0, 12],
        ['B', 'A', 123.23, 123, 1],
        [nulls_fixture2, 'B', 1, 1, 1.0],
        ['A', nulls_fixture2, 1, 1, 1.0],
    ]
    df: pd.DataFrame = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd', 'e'])
    grouped: pd.DataFrame = df.groupby(['a', 'b'], dropna=dropna).sum()
    mi: pd.MultiIndex = pd.MultiIndex.from_tuples([tuple(x) for x in tuples], names=list('ab'))
    if not dropna:
        mi = mi.set_levels([['A', 'B', np.nan], ['A', 'B', np.nan]])
    expected: pd.DataFrame = pd.DataFrame(outputs, index=mi)
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize(
    'dropna, idx, outputs',
    [
        (
            True,
            ['A', 'B'],
            {'b': [123.23, 13.0], 'c': [123.0, 13.0], 'd': [1.0, 13.0]},
        ),
        (
            False,
            ['A', 'B', np.nan],
            {'b': [123.23, 13.0, 12.3], 'c': [123.0, 13.0, 233.0], 'd': [1.0, 13.0, 12.0]},
        ),
    ],
)
def test_groupby_dropna_normal_index_dataframe(
    dropna: bool, idx: List[Any], outputs: Dict[str, List[Union[float, int]]]
) -> None:
    df_list: List[List[Any]] = [['B', 12, 12, 12],
                                [None, 12.3, 233.0, 12],
                                ['A', 123.23, 123, 1],
                                ['B', 1, 1, 1.0]]
    df: pd.DataFrame = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd'])
    grouped: pd.DataFrame = df.groupby('a', dropna=dropna).sum()
    expected: pd.DataFrame = pd.DataFrame(outputs, index=pd.Index(idx, name='a'))
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize(
    'dropna, idx, expected',
    [
        (True, ['a', 'a', 'b', np.nan], pd.Series([3, 3], index=['a', 'b'])),
        (False, ['a', 'a', 'b', np.nan], pd.Series([3, 3, 3], index=['a', 'b', np.nan])),
    ],
)
def test_groupby_dropna_series_level(
    dropna: bool, idx: List[Any], expected: pd.Series
) -> None:
    ser: pd.Series = pd.Series([1, 2, 3, 3], index=idx)
    result: pd.Series = ser.groupby(level=0, dropna=dropna).sum()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'dropna, expected',
    [
        (True, pd.Series([210.0, 350.0], index=['a', 'b'], name='Max Speed')),
        (False, pd.Series([210.0, 350.0, 20.0], index=['a', 'b', np.nan], name='Max Speed')),
    ],
)
def test_groupby_dropna_series_by(dropna: bool, expected: pd.Series) -> None:
    ser: pd.Series = pd.Series([390.0, 350.0, 30.0, 20.0], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'], name='Max Speed')
    result: pd.Series = ser.groupby(['a', 'b', 'a', np.nan], dropna=dropna).mean()
    tm.assert_series_equal(result, expected)

def test_grouper_dropna_propagation(dropna: bool) -> None:
    df: pd.DataFrame = pd.DataFrame({'A': [0, 0, 1, None], 'B': [1, 2, 3, None]})
    gb = df.groupby('A', dropna=dropna)
    assert gb._grouper.dropna == dropna

@pytest.mark.parametrize(
    'index',
    [
        pd.RangeIndex(0, 4),
        list('abcd'),
        pd.MultiIndex.from_product([(1, 2), ('R', 'B')], names=['num', 'col']),
    ],
)
def test_groupby_dataframe_slice_then_transform(dropna: bool, index: Union[pd.RangeIndex, List[str], pd.MultiIndex]) -> None:
    expected_data: Dict[str, List[Optional[Union[int, float]]]] = {'B': [2, 2, 1, np.nan if dropna else 1]}
    df: pd.DataFrame = pd.DataFrame({'A': [0, 0, 1, None], 'B': [1, 2, 3, None]}, index=index)
    gb = df.groupby('A', dropna=dropna)
    result: pd.DataFrame = gb.transform(len)
    expected: pd.DataFrame = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)
    result = gb[['B']].transform(len)
    expected = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)
    result_series: pd.Series = gb['B'].transform(len)
    expected_series: pd.Series = pd.Series(expected_data['B'], index=index, name='B')
    tm.assert_series_equal(result_series, expected_series)

@pytest.mark.parametrize(
    'dropna, tuples, outputs',
    [
        (
            True,
            [['A', 'B'], ['B', 'A']],
            {'c': [13.0, 123.23], 'd': [12.0, 123.0], 'e': [1.0, 1.0]},
        ),
        (
            False,
            [['A', 'B'], ['A', np.nan], ['B', 'A']],
            {'c': [13.0, 12.3, 123.23], 'd': [12.0, 233.0, 123.0], 'e': [1.0, 12.0, 1.0]},
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_agg(
    dropna: bool, tuples: List[List[Any]], outputs: Dict[str, List[Union[float, int]]]
) -> None:
    df_list: List[List[Any]] = [['A', 'B', 12, 12, 12],
                                ['A', None, 12.3, 233.0, 12],
                                ['B', 'A', 123.23, 123, 1],
                                ['A', 'B', 1, 1, 1.0]]
    df: pd.DataFrame = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd', 'e'])
    agg_dict: Dict[str, str] = {'c': 'sum', 'd': 'max', 'e': 'min'}
    grouped: pd.DataFrame = df.groupby(['a', 'b'], dropna=dropna).agg(agg_dict)
    mi: pd.MultiIndex = pd.MultiIndex.from_tuples([tuple(x) for x in tuples], names=list('ab'))
    if not dropna:
        mi = mi.set_levels(['A', 'B', np.nan], level='b')
    expected: pd.DataFrame = pd.DataFrame(outputs, index=mi)
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.arm_slow
@pytest.mark.parametrize(
    'datetime1, datetime2',
    [
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01')),
        (pd.Timedelta('-2 days'), pd.Timedelta('-1 days')),
        (pd.Period('2020-01-01'), pd.Period('2020-02-01')),
    ],
)
@pytest.mark.parametrize(
    'dropna, values',
    [(True, [12, 3]), (False, [12, 3, 6])],
)
def test_groupby_dropna_datetime_like_data(
    dropna: bool,
    values: List[int],
    datetime1: Union[pd.Timestamp, pd.Timedelta, pd.Period],
    datetime2: Union[pd.Timestamp, pd.Timedelta, pd.Period],
    unique_nulls_fixture: Any,
    unique_nulls_fixture2: Any,
) -> None:
    df: pd.DataFrame = pd.DataFrame({
        'values': [1, 2, 3, 4, 5, 6],
        'dt': [datetime1, unique_nulls_fixture, datetime2, unique_nulls_fixture2, datetime1, datetime1]
    })
    if dropna:
        indexes: List[Any] = [datetime1, datetime2]
    else:
        indexes = [datetime1, datetime2, np.nan]
    grouped: pd.DataFrame = df.groupby('dt', dropna=dropna).agg({'values': 'sum'})
    expected: pd.DataFrame = pd.DataFrame({'values': values}, index=pd.Index(indexes, name='dt'))
    tm.assert_frame_equal(grouped, expected)

@pytest.mark.parametrize(
    'dropna, data, selected_data, levels',
    [
        pytest.param(
            False,
            {'groups': ['a', 'a', 'b', np.nan], 'values': [10, 10, 20, 30]},
            {'values': [0, 1, 0, 0]},
            ['a', 'b', np.nan],
            id='dropna_false_has_nan',
        ),
        pytest.param(
            True,
            {'groups': ['a', 'a', 'b', np.nan], 'values': [10, 10, 20, 30]},
            {'values': [0, 1, 0]},
            None,
            id='dropna_true_has_nan',
        ),
        pytest.param(
            False,
            {'groups': ['a', 'a', 'b', 'c'], 'values': [10, 10, 20, 30]},
            {'values': [0, 1, 0, 0]},
            None,
            id='dropna_false_no_nan',
        ),
        pytest.param(
            True,
            {'groups': ['a', 'a', 'b', 'c'], 'values': [10, 10, 20, 30]},
            {'values': [0, 1, 0, 0]},
            None,
            id='dropna_true_no_nan',
        ),
    ],
)
def test_groupby_apply_with_dropna_for_multi_index(
    dropna: bool, data: Dict[str, List[Any]], selected_data: Dict[str, List[Any]], levels: Optional[List[Any]]
) -> None:
    df: pd.DataFrame = pd.DataFrame(data)
    gb = df.groupby('groups', dropna=dropna)
    result: pd.DataFrame = gb.apply(lambda grp: pd.DataFrame({'values': list(range(len(grp)))}))
    mi_tuples: Tuple[Tuple[Any, Any], ...] = tuple(zip(data['groups'], selected_data['values']))
    mi: pd.MultiIndex = pd.MultiIndex.from_tuples(mi_tuples, names=['groups', None])
    if not dropna and levels:
        mi = mi.set_levels(levels, level='groups')
    expected: pd.DataFrame = pd.DataFrame(selected_data, index=mi)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('input_index', [None, ['a'], ['a', 'b']])
@pytest.mark.parametrize('keys', [['a'], ['a', 'b']])
@pytest.mark.parametrize('series', [True, False])
def test_groupby_dropna_with_multiindex_input(
    input_index: Optional[List[str]], keys: List[str], series: bool
) -> None:
    obj: pd.DataFrame = pd.DataFrame({'a': [1, np.nan], 'b': [1, 1], 'c': [2, 3]})
    expected: Union[pd.DataFrame, pd.Series] = obj.set_index(keys)
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
    data: Dict[str, List[Any]] = {'group': ['g1', np.nan, 'g1', 'g2', np.nan], 'B': [0, 1, 2, 3, 4]}
    df: pd.DataFrame = pd.DataFrame(data)
    grouped = df.groupby('group', dropna=False)
    result: Dict[Any, np.ndarray] = grouped.indices
    dtype = np.intp
    expected: Dict[Any, np.ndarray] = {
        'g1': np.array([0, 2], dtype=dtype),
        'g2': np.array([3], dtype=dtype),
        np.nan: np.array([1, 4], dtype=dtype),
    }
    for result_values, expected_values in zip(result.values(), expected.values()):
        tm.assert_numpy_array_equal(result_values, expected_values)
    assert np.isnan(list(result.keys())[2])
    assert list(result.keys())[0:2] == ['g1', 'g2']

def test_groupby_drop_nan_with_multi_index() -> None:
    df: pd.DataFrame = pd.DataFrame([[np.nan, 0, 1]], columns=['a', 'b', 'c'])
    df = df.set_index(['a', 'b'])
    result: pd.DataFrame = df.groupby(['a', 'b'], dropna=False).first()
    expected: pd.DataFrame = df
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('sequence_index', range(3 ** 4))
@pytest.mark.parametrize(
    'dtype',
    [
        None,
        'UInt8', 'Int8', 'UInt16', 'Int16', 'UInt32', 'Int32', 'UInt64', 'Int64',
        'Float32', 'Float64', 'category', 'string',
        pytest.param('string[pyarrow]', marks=pytest.mark.skipif(pa_version_under10p1, reason='pyarrow is not installed')),
        'datetime64[ns]', 'period[D]', 'Sparse[float]'
    ],
)
@pytest.mark.parametrize('test_series', [True, False])
def test_no_sort_keep_na(
    sequence_index: int, dtype: Optional[str], test_series: bool, as_index: bool
) -> None:
    sequence: str = ''.join([{0: 'x', 1: 'y', 2: 'z'}[sequence_index // 3 ** k % 3] for k in range(4)])
    if dtype in ('string', 'string[pyarrow]'):
        uniques: Dict[str, Any] = {'x': 'x', 'y': 'y', 'z': pd.NA}
    elif dtype in ('datetime64[ns]', 'period[D]'):
        uniques = {'x': '2016-01-01', 'y': '2017-01-01', 'z': pd.NA}
    else:
        uniques = {'x': 1, 'y': 2, 'z': np.nan}
    df: pd.DataFrame = pd.DataFrame({
        'key': pd.Series([uniques[label] for label in sequence], dtype=dtype),
        'a': [0, 1, 2, 3]
    })
    gb = df.groupby('key', dropna=False, sort=False, as_index=as_index, observed=False)
    if test_series:
        gb = gb['a']
    result = gb.sum()
    summed: Dict[str, int] = {}
    for idx, label in enumerate(sequence):
        summed[label] = summed.get(label, 0) + idx
    if dtype == 'category':
        index = pd.CategoricalIndex([uniques[e] for e in summed], df['key'].cat.categories, name='key')
    elif isinstance(dtype, str) and dtype.startswith('Sparse'):
        index = pd.Index(pd.array([uniques[label] for label in summed], dtype=dtype), name='key')
    else:
        index = pd.Index([uniques[label] for label in summed], dtype=dtype, name='key')
    expected = pd.Series(list(summed.values()), index=index, name='a', dtype=None)
    if not test_series:
        expected = expected.to_frame()
    if not as_index:
        expected = expected.reset_index()
        if dtype is not None and isinstance(dtype, str) and dtype.startswith('Sparse'):
            expected['key'] = expected['key'].astype(dtype)
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('test_series', [True, False])
@pytest.mark.parametrize('dtype', [object, None])
def test_null_is_null_for_dtype(
    sort: bool, dtype: Optional[Any], nulls_fixture: Any, nulls_fixture2: Any, test_series: bool
) -> None:
    df: pd.DataFrame = pd.DataFrame({'a': [1, 2]})
    groups: pd.Series = pd.Series([nulls_fixture, nulls_fixture2], dtype=dtype)
    obj: Union[pd.Series, pd.DataFrame] = df['a'] if test_series else df
    gb = obj.groupby(groups, dropna=False, sort=sort)
    result = gb.sum()
    index = pd.Index([na_value_for_dtype(groups.dtype)])
    expected = pd.DataFrame({'a': [3]}, index=index)
    if test_series:
        tm.assert_series_equal(result, expected['a'])
    else:
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'])
def test_categorical_reducers(
    reduction_func: str, observed: bool, sort: bool, as_index: bool, index_kind: str
) -> None:
    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    df: pd.DataFrame = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(20)})
    df_filled = df.copy()
    df_filled['x'] = pd.Categorical(values, categories=[1, 2, 3, 4]).fillna(4)
    if index_kind == 'range':
        keys = ['x']
    elif index_kind == 'single':
        keys = ['x']
        df = df.set_index('x')
        df_filled = df_filled.set_index('x')
    else:
        keys = ['x', 'x2']
        df['x2'] = df['x']
        df = df.set_index(['x', 'x2'])
        df_filled['x2'] = df_filled['x']
        df_filled = df_filled.set_index(['x', 'x2'])
    args = get_groupby_method_args(reduction_func, df)
    args_filled = get_groupby_method_args(reduction_func, df_filled)
    if reduction_func == 'corrwith' and index_kind == 'range':
        args = (args[0].drop(columns=keys),)
        args_filled = (args_filled[0].drop(columns=keys),)
    gb_keepna = df.groupby(keys, dropna=False, observed=observed, sort=sort, as_index=as_index)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            getattr(gb_keepna, reduction_func)(*args)
        return
    gb_filled = df_filled.groupby(keys, observed=observed, sort=sort, as_index=True)
    if reduction_func == 'corrwith':
        warn = FutureWarning
        msg = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn = None
        msg = ''
    with tm.assert_produces_warning(warn, match=msg):
        expected = getattr(gb_filled, reduction_func)(*args_filled).reset_index()
    expected['x'] = expected['x'].cat.remove_categories([4])
    if index_kind == 'multi':
        expected['x2'] = expected['x2'].cat.remove_categories([4])
    if as_index:
        if index_kind == 'multi':
            expected = expected.set_index(['x', 'x2'])
        else:
            expected = expected.set_index('x')
    if reduction_func in ('idxmax', 'idxmin') and index_kind != 'range':
        values_list = expected['y'].values.tolist()
        if index_kind == 'single':
            values_list = [np.nan if e == 4 else e for e in values_list]
            expected['y'] = pd.Categorical(values_list, categories=[1, 2, 3])
        else:
            values_list = [(np.nan, np.nan) if e == (4, 4) else e for e in values_list]
            expected['y'] = values_list
    if reduction_func == 'size':
        expected = expected.rename(columns={0: 'size'})
        if as_index:
            expected = expected['size'].rename(None)
    if reduction_func == 'corrwith':
        warn = FutureWarning
        msg = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn = None
        msg = ''
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(gb_keepna, reduction_func)(*args)
    tm.assert_equal(result, expected)

def test_categorical_transformers(
    transformation_func: str, observed: bool, sort: bool, as_index: bool
) -> None:
    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    df: pd.DataFrame = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(20)})
    args = get_groupby_method_args(transformation_func, df)
    null_group_values = df[df['x'].isnull()]['y']
    if transformation_func == 'cumcount':
        null_group_data = list(range(len(null_group_values)))
    elif transformation_func == 'ngroup':
        if sort:
            if observed:
                na_group = df['x'].nunique(dropna=False) - 1
            else:
                na_group = df['x'].nunique(dropna=False) - 1
        else:
            na_group = df.iloc[:null_group_values.index[0]]['x'].nunique()
        null_group_data = len(null_group_values) * [na_group]
    else:
        null_group_data = getattr(null_group_values, transformation_func)(*args)
    null_group_result = pd.DataFrame({'y': null_group_data})
    gb_keepna = df.groupby('x', dropna=False, observed=observed, sort=sort, as_index=as_index)
    gb_dropna = df.groupby('x', dropna=True, observed=observed, sort=sort)
    result = getattr(gb_keepna, transformation_func)(*args)
    expected = getattr(gb_dropna, transformation_func)(*args)
    for iloc, value in zip(df[df['x'].isnull()].index.tolist(), null_group_result.values.ravel()):
        if expected.ndim == 1:
            expected.iloc[iloc] = value
        else:
            expected.iloc[iloc, 0] = value
    if transformation_func == 'ngroup':
        expected[df['x'].notnull() & expected.ge(na_group)] += 1
    if transformation_func not in ('rank', 'diff', 'pct_change', 'shift'):
        expected = expected.astype('int64')
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('method', ['head', 'tail'])
def test_categorical_head_tail(
    method: str, observed: bool, sort: bool, as_index: bool
) -> None:
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df: pd.DataFrame = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(len(values))})
    gb = df.groupby('x', dropna=False, observed=observed, sort=sort, as_index=as_index)
    result = getattr(gb, method)()
    if method == 'tail':
        values = values[::-1]
    mask = ((values == 1) & ((values == 1).cumsum() <= 5)) | ((values == 2) & ((values == 2).cumsum() <= 5)) | ((values == None) & ((values == None).cumsum() <= 5))
    if method == 'tail':
        mask = mask[::-1]
    expected = df[mask]
    tm.assert_frame_equal(result, expected)

def test_categorical_agg() -> None:
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df: pd.DataFrame = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(len(values))})
    gb = df.groupby('x', dropna=False, observed=False)
    result = gb.agg(lambda x: x.sum())
    expected = gb.sum()
    tm.assert_frame_equal(result, expected)

def test_categorical_transform() -> None:
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df: pd.DataFrame = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(len(values))})
    gb = df.groupby('x', dropna=False, observed=False)
    result = gb.transform(lambda x: x.sum())
    expected = gb.transform('sum')
    tm.assert_frame_equal(result, expected)