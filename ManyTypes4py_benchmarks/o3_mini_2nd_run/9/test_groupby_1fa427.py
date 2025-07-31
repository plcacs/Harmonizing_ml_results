from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Categorical, DataFrame, Grouper, Index, Interval, MultiIndex, RangeIndex, Series, Timedelta, Timestamp, date_range, to_datetime
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
from typing import Any, Callable, List, Union

pytestmark = pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')

def test_repr() -> None:
    result: str = repr(Grouper(key='A', level='B'))
    expected: str = "Grouper(key='A', level='B', sort=False, dropna=True)"
    assert result == expected

def test_groupby_nonobject_dtype(multiindex_dataframe_random_data: Any) -> None:
    key: Any = multiindex_dataframe_random_data.index.codes[0]
    grouped: Any = multiindex_dataframe_random_data.groupby(key)
    result: Any = grouped.sum()
    expected: Any = multiindex_dataframe_random_data.groupby(key.astype('O')).sum()
    assert result.index.dtype == np.int8
    assert expected.index.dtype == np.int64
    tm.assert_frame_equal(result, expected, check_index_type=False)

def test_groupby_nonobject_dtype_mixed() -> None:
    df: DataFrame = DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C': np.random.default_rng(2).standard_normal(8),
        'D': np.array(np.random.default_rng(2).standard_normal(8), dtype='float32')
    })
    df['value'] = range(len(df))

    def max_value(group: DataFrame) -> Any:
        return group.loc[group['value'].idxmax()]
    applied: Any = df.groupby('A').apply(max_value)
    result: Any = applied.dtypes
    expected: Any = df.drop(columns='A').dtypes
    tm.assert_series_equal(result, expected)

def test_pass_args_kwargs(ts: Any) -> None:
    def f(x: Any, q: Any = None, axis: int = 0) -> Any:
        return np.percentile(x, q, axis=axis)
    g: Callable[[Any], Any] = lambda x: np.percentile(x, 80, axis=0)
    ts_grouped: Any = ts.groupby(lambda x: x.month)
    agg_result: Any = ts_grouped.agg(np.percentile, 80, axis=0)
    apply_result: Any = ts_grouped.apply(np.percentile, 80, axis=0)
    trans_result: Any = ts_grouped.transform(np.percentile, 80, axis=0)
    agg_expected: Any = ts_grouped.quantile(0.8)
    trans_expected: Any = ts_grouped.transform(g)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)
    agg_result = ts_grouped.agg(f, q=80)
    apply_result = ts_grouped.apply(f, q=80)
    trans_result = ts_grouped.transform(f, q=80)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

def test_pass_args_kwargs_dataframe(tsframe: Any, as_index: Any) -> None:
    def f(x: Any, q: Any = None, axis: int = 0) -> Any:
        return np.percentile(x, q, axis=axis)
    df_grouped: Any = tsframe.groupby(lambda x: x.month, as_index=as_index)
    agg_result: Any = df_grouped.agg(np.percentile, 80, axis=0)
    apply_result: Any = df_grouped.apply(DataFrame.quantile, 0.8)
    expected: Any = df_grouped.quantile(0.8)
    tm.assert_frame_equal(apply_result, expected, check_names=False)
    tm.assert_frame_equal(agg_result, expected)
    apply_result = df_grouped.apply(DataFrame.quantile, [0.4, 0.8])
    expected_seq: Any = df_grouped.quantile([0.4, 0.8])
    if not as_index:
        apply_result.index = range(4)
        apply_result.insert(loc=0, column='level_0', value=[1, 1, 2, 2])
        apply_result.insert(loc=1, column='level_1', value=[0.4, 0.8, 0.4, 0.8])
    tm.assert_frame_equal(apply_result, expected_seq, check_names=False)
    agg_result = df_grouped.agg(f, q=80)
    apply_result = df_grouped.apply(DataFrame.quantile, q=0.8)
    tm.assert_frame_equal(agg_result, expected)
    tm.assert_frame_equal(apply_result, expected, check_names=False)

def test_len() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), 
                              columns=Index(list('ABCD'), dtype=object), 
                              index=date_range('2000-01-01', periods=10, freq='B'))
    grouped: Any = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    assert len(grouped) == len(df)
    grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    expected: int = len({(x.year, x.month) for x in df.index})
    assert len(grouped) == expected

def test_len_nan_group() -> None:
    df: DataFrame = DataFrame({'a': [np.nan] * 3, 'b': [1, 2, 3]})
    assert len(df.groupby('a')) == 0
    assert len(df.groupby('b')) == 3
    assert len(df.groupby(['a', 'b'])) == 0

def test_groupby_timedelta_median() -> None:
    expected: Series = Series(data=Timedelta('1D'), index=['foo'])
    df: DataFrame = DataFrame({'label': ['foo', 'foo'], 'timedelta': [pd.NaT, Timedelta('1D')]})
    gb: Any = df.groupby('label')['timedelta']
    actual: Any = gb.median()
    tm.assert_series_equal(actual, expected, check_names=False)

@pytest.mark.parametrize('keys', [['a'], ['a', 'b']])
def test_len_categorical(dropna: Any, observed: Any, keys: List[Any]) -> None:
    df: DataFrame = DataFrame({
        'a': Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]), 
        'b': Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]), 
        'c': 1})
    gb: Any = df.groupby(keys, observed=observed, dropna=dropna)
    result: int = len(gb)
    if observed and dropna:
        expected: int = 2
    elif observed and (not dropna):
        expected = 3
    elif len(keys) == 1:
        expected = 3 if dropna else 4
    else:
        expected = 9 if dropna else 16
    assert result == expected, f'{result} vs {expected}'

def test_basic_regression() -> None:
    result: Series = Series([1.0 * x for x in list(range(1, 10)) * 10])
    data: Any = np.random.default_rng(2).random(1100) * 10.0
    groupings: Series = Series(data)
    grouped: Any = result.groupby(groupings)
    grouped.mean()

def test_indices_concatenation_order() -> None:
    def f1(x: DataFrame) -> DataFrame:
        y: DataFrame = x[x.b % 2 == 1] ** 2
        if y.empty:
            multiindex: MultiIndex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=['b', 'c'])
            res: DataFrame = DataFrame(columns=['a'], index=multiindex)
            return res
        else:
            y = y.set_index(['b', 'c'])
            return y

    def f2(x: DataFrame) -> DataFrame:
        y: DataFrame = x[x.b % 2 == 1] ** 2
        if y.empty:
            return DataFrame()
        else:
            y = y.set_index(['b', 'c'])
            return y

    def f3(x: DataFrame) -> DataFrame:
        y: DataFrame = x[x.b % 2 == 1] ** 2
        if y.empty:
            multiindex: MultiIndex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=['foo', 'bar'])
            res: DataFrame = DataFrame(columns=['a', 'b'], index=multiindex)
            return res
        else:
            return y
    df: DataFrame = DataFrame({'a': [1, 2, 2, 2], 'b': range(4), 'c': range(5, 9)})
    df2: DataFrame = DataFrame({'a': [3, 2, 2, 2], 'b': range(4), 'c': range(5, 9)})
    result1: DataFrame = df.groupby('a').apply(f1)
    result2: DataFrame = df2.groupby('a').apply(f1)
    tm.assert_frame_equal(result1, result2)
    msg: str = 'Cannot concat indices that do not have the same number of levels'
    with pytest.raises(AssertionError, match=msg):
        df.groupby('a').apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby('a').apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df.groupby('a').apply(f3)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby('a').apply(f3)

def test_attr_wrapper(ts: Any) -> None:
    grouped: Any = ts.groupby(lambda x: x.weekday())
    result: Any = grouped.std()
    expected: Any = grouped.agg(lambda x: np.std(x, ddof=1))
    tm.assert_series_equal(result, expected)
    result = grouped.describe()
    expected = {name: gp.describe() for name, gp in grouped}
    expected = DataFrame(expected).T
    tm.assert_frame_equal(result, expected)
    result = grouped.dtype
    expected = grouped.agg(lambda x: x.dtype)
    tm.assert_series_equal(result, expected)
    msg: str = "'SeriesGroupBy' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        grouped.foo

def test_frame_groupby(tsframe: Any) -> None:
    grouped: Any = tsframe.groupby(lambda x: x.weekday())
    aggregated: DataFrame = grouped.aggregate('mean')
    assert len(aggregated) == 5
    assert len(aggregated.columns) == 4
    tscopy: DataFrame = tsframe.copy()
    tscopy['weekday'] = [x.weekday() for x in tscopy.index]
    stragged: DataFrame = tscopy.groupby('weekday').aggregate('mean')
    tm.assert_frame_equal(stragged, aggregated, check_names=False)
    grouped = tsframe.head(30).groupby(lambda x: x.weekday())
    transformed: DataFrame = grouped.transform(lambda x: x - x.mean())
    assert len(transformed) == 30
    assert len(transformed.columns) == 4
    transformed = grouped.transform(lambda x: x.mean())
    for weekday, group in grouped:
        mean: Any = group.mean()
        for idx in group.index:
            tm.assert_series_equal(transformed.xs(idx), mean, check_names=False)
    for weekday, group in grouped:
        assert group.index[0].weekday() == weekday
    groups: Any = grouped.groups
    indices: Any = grouped.indices
    for k, v in groups.items():
        samething: Any = tsframe.index.take(indices[k])
        assert (samething == v).all()

def test_frame_set_name_single(df: DataFrame) -> None:
    grouped: Any = df.groupby('A')
    result: DataFrame = grouped.mean(numeric_only=True)
    assert result.index.name == 'A'
    result = df.groupby('A', as_index=False).mean(numeric_only=True)
    assert result.index.name != 'A'
    result = grouped[['C', 'D']].agg('mean')
    assert result.index.name == 'A'
    result = grouped.agg({'C': 'mean', 'D': 'std'})
    assert result.index.name == 'A'
    result = grouped['C'].mean()
    assert result.index.name == 'A'
    result = grouped['C'].agg('mean')
    assert result.index.name == 'A'
    result = grouped['C'].agg(['mean', 'std'])
    assert result.index.name == 'A'
    msg: str = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        grouped['C'].agg({'foo': 'mean', 'bar': 'std'})

def test_multi_func(df: DataFrame) -> None:
    col1: Series = df['A']
    col2: Series = df['B']
    grouped: Any = df.groupby([col1.get, col2.get])
    agged: DataFrame = grouped.mean(numeric_only=True)
    expected: DataFrame = df.groupby(['A', 'B']).mean()
    tm.assert_frame_equal(agged.loc[:, ['C', 'D']], expected.loc[:, ['C', 'D']], check_names=False)
    df = DataFrame({
        'v1': np.random.default_rng(2).standard_normal(6),
        'v2': np.random.default_rng(2).standard_normal(6),
        'k1': np.array(['b', 'b', 'b', 'a', 'a', 'a']),
        'k2': np.array(['1', '1', '1', '2', '2', '2'])
    }, index=['one', 'two', 'three', 'four', 'five', 'six'])
    grouped = df.groupby(['k1', 'k2'])
    grouped.agg('sum')

def test_multi_key_multiple_functions(df: DataFrame) -> None:
    grouped: Any = df.groupby(['A', 'B'])['C']
    agged: DataFrame = grouped.agg(['mean', 'std'])
    expected: DataFrame = DataFrame({'mean': grouped.agg('mean'), 'std': grouped.agg('std')})
    tm.assert_frame_equal(agged, expected)

def test_frame_multi_key_function_list() -> None:
    data: DataFrame = DataFrame({
        'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'],
        'D': np.random.default_rng(2).standard_normal(11),
        'E': np.random.default_rng(2).standard_normal(11),
        'F': np.random.default_rng(2).standard_normal(11)
    })
    grouped: Any = data.groupby(['A', 'B'])
    funcs: List[str] = ['mean', 'std']
    agged: DataFrame = grouped.agg(funcs)
    expected: DataFrame = pd.concat([
        grouped['D'].agg(funcs), grouped['E'].agg(funcs), grouped['F'].agg(funcs)
    ], keys=['D', 'E', 'F'], axis=1)
    assert isinstance(agged.index, MultiIndex)
    assert isinstance(expected.index, MultiIndex)
    tm.assert_frame_equal(agged, expected)

def test_frame_multi_key_function_list_partial_failure(using_infer_string: Any) -> None:
    data: DataFrame = DataFrame({
        'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'],
        'C': ['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny', 'shiny', 'dull', 'shiny', 'shiny', 'shiny'],
        'D': np.random.default_rng(2).standard_normal(11),
        'E': np.random.default_rng(2).standard_normal(11),
        'F': np.random.default_rng(2).standard_normal(11)
    })
    grouped: Any = data.groupby(['A', 'B'])
    funcs: List[str] = ['mean', 'std']
    msg: str = re.escape('agg function failed [how->mean,dtype->')
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg(funcs)

@pytest.mark.parametrize('op', [lambda x: x.sum(), lambda x: x.mean()])
def test_groupby_multiple_columns(df: DataFrame, op: Callable[[Any], Any]) -> None:
    data: DataFrame = df
    grouped: Any = data.groupby(['A', 'B'])
    result1: Any = op(grouped)
    keys: List[Any] = []
    values: List[Any] = []
    for n1, gp1 in data.groupby('A'):
        for n2, gp2 in gp1.groupby('B'):
            keys.append((n1, n2))
            values.append(op(gp2.loc[:, ['C', 'D']]))
    mi: MultiIndex = MultiIndex.from_tuples(keys, names=['A', 'B'])
    expected: DataFrame = pd.concat(values, axis=1).T
    expected.index = mi
    for col in ['C', 'D']:
        result_col: Any = op(grouped[col])
        pivoted: Any = result1[col]
        exp: Any = expected[col]
        tm.assert_series_equal(result_col, exp)
        tm.assert_series_equal(pivoted, exp)
    result: Series = data['C'].groupby([data['A'], data['B']]).mean()
    expected = data.groupby(['A', 'B']).mean()['C']
    tm.assert_series_equal(result, expected)

def test_as_index_select_column() -> None:
    df: DataFrame = DataFrame([[1, 2], [1, 4], [5, 6]], columns=['A', 'B'])
    result: Series = df.groupby('A', as_index=False)['B'].get_group(1)
    expected: Series = Series([2, 4], name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', as_index=False, group_keys=True)['B'].apply(lambda x: x.cumsum())
    expected = Series([2, 6, 6], name='B', index=range(3))
    tm.assert_series_equal(result, expected)

def test_groupby_as_index_select_column_sum_empty_df() -> None:
    df: DataFrame = DataFrame(columns=Index(['A', 'B', 'C'], name='alpha'))
    left: DataFrame = df.groupby(by='A', as_index=False)['B'].sum(numeric_only=False)
    expected: DataFrame = DataFrame(columns=df.columns[:2], index=range(0))
    expected.columns.names = [None]
    tm.assert_frame_equal(left, expected)

def test_ops_not_as_index(reduction_func: str) -> None:
    if reduction_func in ('corrwith', 'nth', 'ngroup'):
        pytest.skip(f'GH 5755: Test not applicable for {reduction_func}')
    df: DataFrame = DataFrame(np.random.default_rng(2).integers(0, 5, size=(100, 2)), columns=['a', 'b'])
    expected: Any = getattr(df.groupby('a'), reduction_func)()
    if reduction_func == 'size':
        expected = expected.rename('size')
    expected = expected.reset_index()
    if reduction_func != 'size':
        expected['a'] = expected['a'].astype(df['a'].dtype)
    g: Any = df.groupby('a', as_index=False)
    result: Any = getattr(g, reduction_func)()
    tm.assert_frame_equal(result, expected)
    result = g.agg(reduction_func)
    tm.assert_frame_equal(result, expected)
    result = getattr(g['b'], reduction_func)()
    tm.assert_frame_equal(result, expected)
    result = g['b'].agg(reduction_func)
    tm.assert_frame_equal(result, expected)

def test_as_index_series_return_frame(df: DataFrame) -> None:
    grouped: Any = df.groupby('A', as_index=False)
    grouped2: Any = df.groupby(['A', 'B'], as_index=False)
    result: DataFrame = grouped['C'].agg('sum')
    expected: DataFrame = grouped.agg('sum').loc[:, ['A', 'C']]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)
    result2: DataFrame = grouped2['C'].agg('sum')
    expected2: DataFrame = grouped2.agg('sum').loc[:, ['A', 'B', 'C']]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)
    result = grouped['C'].sum()
    expected = grouped.sum().loc[:, ['A', 'C']]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)
    result2 = grouped2['C'].sum()
    expected2 = grouped2.sum().loc[:, ['A', 'B', 'C']]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)

def test_as_index_series_column_slice_raises(df: DataFrame) -> None:
    grouped: Any = df.groupby('A', as_index=False)
    msg: str = 'Column\\(s\\) C already selected'
    with pytest.raises(IndexError, match=msg):
        grouped['C'].__getitem__('D')

def test_groupby_as_index_cython(df: DataFrame) -> None:
    data: DataFrame = df
    grouped: Any = data.groupby('A', as_index=False)
    result: DataFrame = grouped.mean(numeric_only=True)
    expected: DataFrame = data.groupby(['A']).mean(numeric_only=True)
    expected.insert(0, 'A', expected.index)
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)
    grouped = data.groupby(['A', 'B'], as_index=False)
    result = grouped.mean()
    expected = data.groupby(['A', 'B']).mean()
    arrays = list(zip(*expected.index.values))
    expected.insert(0, 'A', arrays[0])
    expected.insert(1, 'B', arrays[1])
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)

def test_groupby_as_index_series_scalar(df: DataFrame) -> None:
    grouped: Any = df.groupby(['A', 'B'], as_index=False)
    result: DataFrame = grouped['C'].agg(len)
    expected: DataFrame = grouped.agg(len).loc[:, ['A', 'B', 'C']]
    tm.assert_frame_equal(result, expected)

def test_groupby_multiple_key() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), 
                              columns=Index(list('ABCD'), dtype=object), 
                              index=date_range('2000-01-01', periods=10, freq='B'))
    grouped: Any = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    agged: DataFrame = grouped.sum()
    tm.assert_almost_equal(df.values, agged.values)

def test_groupby_multi_corner(df: DataFrame) -> None:
    df = df.copy()
    df['bad'] = np.nan
    agged: DataFrame = df.groupby(['A', 'B']).mean()
    expected: DataFrame = df.groupby(['A', 'B']).mean()
    expected['bad'] = np.nan
    tm.assert_frame_equal(agged, expected)

def test_raises_on_nuisance(df: DataFrame, using_infer_string: Any) -> None:
    grouped: Any = df.groupby('A')
    msg: str = re.escape('agg function failed [how->mean,dtype->')
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg('mean')
    with pytest.raises(TypeError, match=msg):
        grouped.mean()
    df = df.loc[:, ['A', 'C', 'D']]
    df['E'] = datetime.now()
    grouped = df.groupby('A')
    msg = "datetime64 type does not support operation 'sum'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg('sum')
    with pytest.raises(TypeError, match=msg):
        grouped.sum()

@pytest.mark.parametrize('agg_function', ['max', 'min'])
def test_keep_nuisance_agg(df: DataFrame, agg_function: str) -> None:
    grouped: Any = df.groupby('A')
    result: DataFrame = getattr(grouped, agg_function)()
    expected: DataFrame = result.copy()
    expected.loc['bar', 'B'] = getattr(df.loc[df['A'] == 'bar', 'B'], agg_function)()
    expected.loc['foo', 'B'] = getattr(df.loc[df['A'] == 'foo', 'B'], agg_function)()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('agg_function', ['sum', 'mean', 'prod', 'std', 'var', 'sem', 'median'])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_omit_nuisance_agg(df: DataFrame, agg_function: str, numeric_only: bool, using_infer_string: Any) -> None:
    grouped: Any = df.groupby('A')
    no_drop_nuisance = ('var', 'std', 'sem', 'mean', 'prod', 'median')
    if agg_function in no_drop_nuisance and (not numeric_only):
        if using_infer_string:
            msg: str = f"dtype 'str' does not support operation '{agg_function}'"
            klass = TypeError
        elif agg_function in ('std', 'sem'):
            klass = ValueError
            msg = "could not convert string to float: 'one'"
        else:
            klass = TypeError
            msg = re.escape(f'agg function failed [how->{agg_function},dtype->')
        with pytest.raises(klass, match=msg):
            getattr(grouped, agg_function)(numeric_only=numeric_only)
    else:
        result: Any = getattr(grouped, agg_function)(numeric_only=numeric_only)
        if not numeric_only and agg_function == 'sum':
            columns: Union[List[str], Index] = ['A', 'B', 'C', 'D']
        else:
            columns = ['A', 'C', 'D']
        expected: Any = getattr(df.loc[:, columns].groupby('A'), agg_function)(numeric_only=numeric_only)
        tm.assert_frame_equal(result, expected)

def test_raise_on_nuisance_python_single(df: DataFrame, using_infer_string: Any) -> None:
    grouped: Any = df.groupby('A')
    err = ValueError
    msg: str = 'could not convert'
    if using_infer_string:
        err = TypeError
        msg = "dtype 'str' does not support operation 'skew'"
    with pytest.raises(err, match=msg):
        grouped.skew()

def test_raise_on_nuisance_python_multiple(three_group: Any, using_infer_string: Any) -> None:
    grouped: Any = three_group.groupby(['A', 'B'])
    msg: str = re.escape('agg function failed [how->mean,dtype->')
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg('mean')
    with pytest.raises(TypeError, match=msg):
        grouped.mean()

def test_empty_groups_corner(multiindex_dataframe_random_data: Any) -> None:
    df: DataFrame = DataFrame({
        'k1': np.array(['b', 'b', 'b', 'a', 'a', 'a']),
        'k2': np.array(['1', '1', '1', '2', '2', '2']),
        'k3': ['foo', 'bar'] * 3,
        'v1': np.random.default_rng(2).standard_normal(6),
        'v2': np.random.default_rng(2).standard_normal(6)
    })
    grouped: Any = df.groupby(['k1', 'k2'])
    result: Any = grouped[['v1', 'v2']].agg('mean')
    expected: Any = grouped.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)
    grouped = multiindex_dataframe_random_data[3:5].groupby(level=0)
    agged: Any = grouped.apply(lambda x: x.mean())
    agged_A: Any = grouped['A'].apply('mean')
    tm.assert_series_equal(agged['A'], agged_A)
    assert agged.index.name == 'first'

def test_nonsense_func() -> None:
    df: DataFrame = DataFrame([0])
    msg: str = "unsupported operand type\\(s\\) for \\+: 'int' and 'str'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(lambda x: x + 'foo')

def test_wrap_aggregated_output_multindex(multiindex_dataframe_random_data: Any, using_infer_string: Any) -> None:
    df: DataFrame = multiindex_dataframe_random_data.T
    df['baz', 'two'] = 'peekaboo'
    keys: List[Any] = [np.array([0, 0, 1]), np.array([0, 0, 1])]
    msg: str = re.escape('agg function failed [how->mean,dtype->')
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(keys).agg('mean')
    agged: DataFrame = df.drop(columns=('baz', 'two')).groupby(keys).agg('mean')
    assert isinstance(agged.columns, MultiIndex)

    def aggfun(ser: Series) -> Any:
        if ser.name == ('foo', 'one'):
            raise TypeError('Test error message')
        return ser.sum()
    with pytest.raises(TypeError, match='Test error message'):
        df.groupby(keys).aggregate(aggfun)

def test_groupby_level_apply(multiindex_dataframe_random_data: Any) -> None:
    result: Any = multiindex_dataframe_random_data.groupby(level=0).count()
    assert result.index.name == 'first'
    result = multiindex_dataframe_random_data.groupby(level=1).count()
    assert result.index.name == 'second'
    result = multiindex_dataframe_random_data['A'].groupby(level=0).count()
    assert result.index.name == 'first'

def test_groupby_level_mapper(multiindex_dataframe_random_data: Any) -> None:
    deleveled: DataFrame = multiindex_dataframe_random_data.reset_index()
    mapper0: dict = {'foo': 0, 'bar': 0, 'baz': 1, 'qux': 1}
    mapper1: dict = {'one': 0, 'two': 0, 'three': 1}
    result0: Any = multiindex_dataframe_random_data.groupby(mapper0, level=0).sum()
    result1: Any = multiindex_dataframe_random_data.groupby(mapper1, level=1).sum()
    mapped_level0: np.ndarray = np.array([mapper0.get(x) for x in deleveled['first']], dtype=np.int64)
    mapped_level1: np.ndarray = np.array([mapper1.get(x) for x in deleveled['second']], dtype=np.int64)
    expected0: Any = multiindex_dataframe_random_data.groupby(mapped_level0).sum()
    expected1: Any = multiindex_dataframe_random_data.groupby(mapped_level1).sum()
    expected0.index.name, expected1.index.name = ('first', 'second')
    tm.assert_frame_equal(result0, expected0)
    tm.assert_frame_equal(result1, expected1)

def test_groupby_level_nonmulti() -> None:
    s: Series = Series([1, 2, 3, 10, 4, 5, 20, 6], Index([1, 2, 3, 1, 4, 5, 2, 6], name='foo'))
    expected: Series = Series([11, 22, 3, 4, 5, 6], Index(list(range(1, 7)), name='foo'))
    result: Series = s.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[0]).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=-1).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[-1]).sum()
    tm.assert_series_equal(result, expected)
    msg: str = 'level > 0 or level < -1 only valid with MultiIndex'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=1)
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=-2)
    msg = 'No group keys passed!'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[])
    msg = 'multiple levels only valid with MultiIndex'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 0])
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 1])
    msg = 'level > 0 or level < -1 only valid with MultiIndex'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[1])

def test_groupby_complex() -> None:
    a: Series = Series(data=np.arange(4) * (1 + 2j), index=[0, 0, 1, 1])
    expected: Series = Series((1 + 2j, 5 + 10j), index=Index([0, 1]))
    result: Series = a.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)

def test_groupby_complex_mean() -> None:
    df: DataFrame = DataFrame([{'a': 2, 'b': 1 + 2j}, {'a': 1, 'b': 1 + 1j}, {'a': 1, 'b': 1 + 2j}])
    result: DataFrame = df.groupby('b').mean()
    expected: DataFrame = DataFrame([[1.0], [1.5]], index=Index([1 + 1j, 1 + 2j], name='b'), columns=Index(['a']))
    tm.assert_frame_equal(result, expected)

def test_groupby_complex_numbers() -> None:
    df: DataFrame = DataFrame([{'a': 1, 'b': 1 + 1j}, {'a': 1, 'b': 1 + 2j}, {'a': 4, 'b': 1}])
    expected: DataFrame = DataFrame(np.array([1, 1, 1], dtype=np.int64), index=Index([1 + 1j, 1 + 2j, 1 + 0j], name='b'), columns=Index(['a']))
    result: DataFrame = df.groupby('b', sort=False).count()
    tm.assert_frame_equal(result, expected)
    expected.index = Index([1 + 0j, 1 + 1j, 1 + 2j], name='b')
    result = df.groupby('b', sort=True).count()
    tm.assert_frame_equal(result, expected)

def test_groupby_series_indexed_differently() -> None:
    s1: Series = Series([5.0, -9.0, 4.0, 100.0, -5.0, 55.0, 6.7], index=Index(['a', 'b', 'c', 'd', 'e', 'f', 'g']))
    s2: Series = Series([1.0, 1.0, 4.0, 5.0, 5.0, 7.0], index=Index(['a', 'b', 'd', 'f', 'g', 'h']))
    grouped: Any = s1.groupby(s2)
    agged: Any = grouped.mean()
    exp: Any = s1.groupby(s2.reindex(s1.index).get).mean()
    tm.assert_series_equal(agged, exp)

def test_groupby_with_hier_columns() -> None:
    tuples: List[Any] = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], 
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
    index: MultiIndex = MultiIndex.from_tuples(tuples)
    columns: MultiIndex = MultiIndex.from_tuples([('A', 'cat'), ('B', 'dog'), ('B', 'cat'), ('A', 'dog')])
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), index=index, columns=columns)
    result: Any = df.groupby(level=0).mean()
    tm.assert_index_equal(result.columns, columns)
    result = df.groupby(level=0).agg('mean')
    tm.assert_index_equal(result.columns, columns)
    result = df.groupby(level=0).apply(lambda x: x.mean())
    tm.assert_index_equal(result.columns, columns)
    sorted_columns, _ = columns.sortlevel(0)
    df['A', 'foo'] = 'bar'
    result = df.groupby(level=0).mean(numeric_only=True)
    tm.assert_index_equal(result.columns, df.columns[:-1])

def test_grouping_ndarray(df: DataFrame) -> None:
    grouped: Any = df.groupby(df['A'].values)
    grouped2: Any = df.groupby(df['A'].rename(None))
    result: DataFrame = grouped.sum()
    expected: DataFrame = grouped2.sum()
    tm.assert_frame_equal(result, expected)

def test_groupby_wrong_multi_labels() -> None:
    index: Index = Index([0, 1, 2, 3, 4], name='index')
    data: DataFrame = DataFrame({
        'foo': ['foo1', 'foo1', 'foo2', 'foo1', 'foo3'],
        'bar': ['bar1', 'bar2', 'bar2', 'bar1', 'bar1'],
        'baz': ['baz1', 'baz1', 'baz1', 'baz2', 'baz2'],
        'spam': ['spam2', 'spam3', 'spam2', 'spam1', 'spam1'],
        'data': [20, 30, 40, 50, 60]
    }, index=index)
    grouped: Any = data.groupby(['foo', 'bar', 'baz', 'spam'])
    result: Any = grouped.agg('mean')
    expected: Any = grouped.mean()
    tm.assert_frame_equal(result, expected)

def test_groupby_series_with_name(df: DataFrame) -> None:
    result: Any = df.groupby(df['A']).mean(numeric_only=True)
    result2: Any = df.groupby(df['A'], as_index=False).mean(numeric_only=True)
    assert result.index.name == 'A'
    assert 'A' in result2
    result = df.groupby([df['A'], df['B']]).mean()
    result2 = df.groupby([df['A'], df['B']], as_index=False).mean()
    assert result.index.names == ('A', 'B')
    assert 'A' in result2
    assert 'B' in result2

def test_seriesgroupby_name_attr(df: DataFrame) -> None:
    result: Any = df.groupby('A')['C']
    assert result.count().name == 'C'
    assert result.mean().name == 'C'
    testFunc: Callable[[Any], Any] = lambda x: np.sum(x) * 2
    assert result.agg(testFunc).name == 'C'

def test_consistency_name() -> None:
    df: DataFrame = DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'],
        'C': np.random.default_rng(2).standard_normal(8) + 1.0,
        'D': np.arange(8)
    })
    expected: Series = df.groupby(['A']).B.count()
    result: Series = df.B.groupby(df.A).count()
    tm.assert_series_equal(result, expected)

def test_groupby_name_propagation(df: DataFrame) -> None:
    def summarize(df: DataFrame, name: Any = None) -> Series:
        return Series({'count': 1, 'mean': 2, 'omissions': 3}, name=name)

    def summarize_random_name(df: DataFrame) -> Series:
        return Series({'count': 1, 'mean': 2, 'omissions': 3}, name=df.iloc[0]['C'])
    metrics: Any = df.groupby('A').apply(summarize)
    assert metrics.columns.name is None
    metrics = df.groupby('A').apply(summarize, 'metrics')
    assert metrics.columns.name == 'metrics'
    metrics = df.groupby('A').apply(summarize_random_name)
    assert metrics.columns.name is None

def test_groupby_nonstring_columns() -> None:
    df: DataFrame = DataFrame([np.arange(10) for x in range(10)])
    grouped: Any = df.groupby(0)
    result: DataFrame = grouped.mean()
    expected: DataFrame = df.groupby(df[0]).mean()
    tm.assert_frame_equal(result, expected)

def test_groupby_mixed_type_columns() -> None:
    df: DataFrame = DataFrame([[0, 1, 2]], columns=['A', 'B', 0])
    expected: DataFrame = DataFrame([[1, 2]], columns=['B', 0], index=Index([0], name='A'))
    result: DataFrame = df.groupby('A').first()
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A').sum()
    tm.assert_frame_equal(result, expected)

def test_cython_grouper_series_bug_noncontig() -> None:
    arr: np.ndarray = np.empty((100, 100))
    arr.fill(np.nan)
    obj: Series = Series(arr[:, 0])
    inds: Any = np.tile(range(10), 10)
    result: Any = obj.groupby(inds).agg(Series.median)
    assert result.isna().all()

def test_series_grouper_noncontig_index() -> None:
    index: Index = Index(['a' * 10] * 100)
    values: Series = Series(np.random.default_rng(2).standard_normal(50), index=index[::2])
    labels: Any = np.random.default_rng(2).integers(0, 5, 50)
    grouped: Any = values.groupby(labels)
    f: Callable[[Any], Any] = lambda x: len(set(map(id, x.index)))
    grouped.agg(f)

def test_convert_objects_leave_decimal_alone() -> None:
    s: Series = Series(range(5))
    labels: np.ndarray = np.array(['a', 'b', 'c', 'd', 'e'], dtype='O')

    def convert_fast(x: Any) -> Decimal:
        return Decimal(str(x.mean()))

    def convert_force_pure(x: Any) -> Decimal:
        assert len(x.values.base) > 0
        return Decimal(str(x.mean()))
    grouped: Any = s.groupby(labels)
    result: Any = grouped.agg(convert_fast)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)
    result = grouped.agg(convert_force_pure)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)

def test_groupby_dtype_inference_empty() -> None:
    df: DataFrame = DataFrame({'x': [], 'range': np.arange(0, dtype='int64')})
    assert df['x'].dtype == np.float64
    result: Any = df.groupby('x').first()
    exp_index: Index = Index([], name='x', dtype=np.float64)
    expected: DataFrame = DataFrame({'range': Series([], index=exp_index, dtype='int64')})
    tm.assert_frame_equal(result, expected, by_blocks=True)

def test_groupby_unit64_float_conversion() -> None:
    df: DataFrame = DataFrame({'first': [1], 'second': [1], 'value': [16148277970000000000]})
    result: Series = df.groupby(['first', 'second'])['value'].max()
    expected: Series = Series([16148277970000000000], MultiIndex.from_product([[1], [1]], names=['first', 'second']), name='value')
    tm.assert_series_equal(result, expected)

def test_groupby_list_infer_array_like(df: DataFrame) -> None:
    result: Any = df.groupby(list(df['A'])).mean(numeric_only=True)
    expected: Any = df.groupby(df['A']).mean(numeric_only=True)
    tm.assert_frame_equal(result, expected, check_names=False)
    with pytest.raises(KeyError, match="^'foo'$"):
        df.groupby(list(df['A'][:-1]))
    df = DataFrame({'foo': [0, 1], 'bar': [3, 4], 'val': np.random.default_rng(2).standard_normal(2)})
    result = df.groupby(['foo', 'bar']).mean()
    expected = df.groupby([df['foo'], df['bar']]).mean()[['val']]

def test_groupby_keys_same_size_as_index() -> None:
    freq: str = 's'
    index: Any = date_range(start=Timestamp('2015-09-29T11:34:44-0700'), periods=2, freq=freq)
    df: DataFrame = DataFrame([['A', 10], ['B', 15]], columns=['metric', 'values'], index=index)
    result: Any = df.groupby([Grouper(level=0, freq=freq), 'metric']).mean()
    expected: Any = df.set_index([df.index, 'metric']).astype(float)
    tm.assert_frame_equal(result, expected)

def test_groupby_one_row() -> None:
    msg: str = "^'Z'$"
    df1: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), columns=list('ABCD'))
    with pytest.raises(KeyError, match=msg):
        df1.groupby('Z')
    df2: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((2, 4)), columns=list('ABCD'))
    with pytest.raises(KeyError, match=msg):
        df2.groupby('Z')

def test_groupby_nat_exclude() -> None:
    df: DataFrame = DataFrame({
        'values': np.random.default_rng(2).standard_normal(8),
        'dt': [np.nan, Timestamp('2013-01-01'), np.nan, Timestamp('2013-02-01'),
               np.nan, Timestamp('2013-02-01'), np.nan, Timestamp('2013-01-01')],
        'str': [np.nan, 'a', np.nan, 'a', np.nan, 'a', np.nan, 'b']
    })
    grouped: Any = df.groupby('dt')
    expected: List[RangeIndex] = [RangeIndex(start=1, stop=13, step=6), RangeIndex(start=3, stop=7, step=2)]
    keys: List[Any] = sorted(grouped.groups.keys())
    assert len(keys) == 2
    for k, e in zip(keys, expected):
        tm.assert_index_equal(grouped.groups[k], e)
    tm.assert_frame_equal(grouped._grouper.groupings[0].obj, df)
    assert grouped.ngroups == 2
    expected_indices: dict = {
        Timestamp('2013-01-01 00:00:00'): np.array([1, 7], dtype=np.intp),
        Timestamp('2013-02-01 00:00:00'): np.array([3, 5], dtype=np.intp)
    }
    for k in grouped.indices:
        tm.assert_numpy_array_equal(grouped.indices[k], expected_indices[k])
    tm.assert_frame_equal(grouped.get_group(Timestamp('2013-01-01')), df.iloc[[1, 7]])
    tm.assert_frame_equal(grouped.get_group(Timestamp('2013-02-01')), df.iloc[[3, 5]])
    with pytest.raises(KeyError, match='^NaT$'):
        grouped.get_group(pd.NaT)
    nan_df: DataFrame = DataFrame({'nan': [np.nan, np.nan, np.nan], 'nat': [pd.NaT, pd.NaT, pd.NaT]})
    assert nan_df['nan'].dtype == 'float64'
    assert nan_df['nat'].dtype == 'datetime64[s]'
    for key in ['nan', 'nat']:
        grouped = nan_df.groupby(key)
        assert grouped.groups == {}
        assert grouped.ngroups == 0
        assert grouped.indices == {}
        with pytest.raises(KeyError, match='^nan$'):
            grouped.get_group(np.nan)
        with pytest.raises(KeyError, match='^NaT$'):
            grouped.get_group(pd.NaT)

def test_groupby_two_group_keys_all_nan() -> None:
    df: DataFrame = DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan], 'c': [1, 2]})
    result: Any = df.groupby(['a', 'b']).indices
    assert result == {}

def test_groupby_2d_malformed() -> None:
    d: DataFrame = DataFrame(index=range(2))
    d['group'] = ['g1', 'g2']
    d['zeros'] = [0, 0]
    d['ones'] = [1, 1]
    d['label'] = ['l1', 'l2']
    tmp: DataFrame = d.groupby(['group']).mean(numeric_only=True)
    res_values: np.ndarray = np.array([[0.0, 1.0], [0.0, 1.0]])
    tm.assert_index_equal(tmp.columns, Index(['zeros', 'ones'], dtype=object))
    tm.assert_numpy_array_equal(tmp.values, res_values)

def test_int32_overflow() -> None:
    B: np.ndarray = np.concatenate((np.arange(10000), np.arange(10000), np.arange(5000)))
    A: np.ndarray = np.arange(25000)
    df: DataFrame = DataFrame({'A': A, 'B': B, 'C': A, 'D': B, 'E': np.random.default_rng(2).standard_normal(25000)})
    left: Any = df.groupby(['A', 'B', 'C', 'D']).sum()
    right: Any = df.groupby(['D', 'C', 'B', 'A']).sum()
    assert len(left) == len(right)

def test_groupby_sort_multi() -> None:
    df: DataFrame = DataFrame({'a': ['foo', 'bar', 'baz'], 'b': [3, 2, 1], 'c': [0, 1, 2], 'd': np.random.default_rng(2).standard_normal(3)})
    tups: Any = [tuple(row) for row in df[['a', 'b', 'c']].values]
    tups = com.asarray_tuplesafe(tups)
    result: DataFrame = df.groupby(['a', 'b', 'c'], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[1, 2, 0]])
    tups = [tuple(row) for row in df[['c', 'a', 'b']].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(['c', 'a', 'b'], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups)
    tups = [tuple(x) for x in df[['b', 'c', 'a']].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(['b', 'c', 'a'], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[2, 1, 0]])
    df = DataFrame({'a': [0, 1, 2, 0, 1, 2], 'b': [0, 0, 0, 1, 1, 1], 'd': np.random.default_rng(2).standard_normal(6)})
    grouped: Any = df.groupby(['a', 'b'])['d']
    result = grouped.sum()

    def _check_groupby(df: DataFrame, result: Any, keys: List[str], field: str, f: Callable[[Any], Any]=lambda x: x.sum()) -> None:
        tups = [tuple(row) for row in df[keys].values]
        tups = com.asarray_tuplesafe(tups)
        expected = f(df.groupby(tups)[field])
        for k, v in expected.items():
            assert result[k] == v
    _check_groupby(df, result, ['a', 'b'], 'd')

def test_dont_clobber_name_column() -> None:
    df: DataFrame = DataFrame({'key': ['a', 'a', 'a', 'b', 'b', 'b'], 'name': ['foo', 'bar', 'baz'] * 2})
    result: DataFrame = df.groupby('key', group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, df[['name']])

def test_skip_group_keys() -> None:
    tsf: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), 
                               columns=Index(list('ABCD'), dtype=object), 
                               index=date_range('2000-01-01', periods=10, freq='B'))
    grouped: Any = tsf.groupby(lambda x: x.month, group_keys=False)
    result: DataFrame = grouped.apply(lambda x: x.sort_values(by='A')[:3])
    pieces: List[Any] = [group.sort_values(by='A')[:3] for key, group in grouped]
    expected: DataFrame = pd.concat(pieces)
    tm.assert_frame_equal(result, expected)
    grouped = tsf['A'].groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x.sort_values()[:3])
    pieces = [group.sort_values()[:3] for key, group in grouped]
    expected = pd.concat(pieces)
    tm.assert_series_equal(result, expected)

def test_no_nonsense_name(float_frame: DataFrame) -> None:
    s: Series = float_frame['C'].copy()
    s.name = None
    result: Any = s.groupby(float_frame['A']).agg('sum')
    assert result.name is None

def test_multifunc_sum_bug() -> None:
    x: DataFrame = DataFrame(np.arange(9).reshape(3, 3))
    x['test'] = 0
    x['fl'] = [1.3, 1.5, 1.6]
    grouped: Any = x.groupby('test')
    result: Any = grouped.agg({'fl': 'sum', 2: 'size'})
    assert result['fl'].dtype == np.float64

def test_handle_dict_return_value(df: DataFrame) -> None:
    def f(group: Any) -> dict:
        return {'max': group.max(), 'min': group.min()}

    def g(group: Any) -> Series:
        return Series({'max': group.max(), 'min': group.min()})
    result: Series = df.groupby('A')['C'].apply(f)
    expected: Series = df.groupby('A')['C'].apply(g)
    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('grouper', ['A', ['A', 'B']])
def test_set_group_name(df: DataFrame, grouper: Any) -> None:
    def f(group: Any) -> Any:
        assert group.name is not None
        return group

    def freduce(group: Any) -> Any:
        assert group.name is not None
        return group.sum()

    def freducex(x: Any) -> Any:
        return freduce(x)
    grouped: Any = df.groupby(grouper, group_keys=False)
    grouped.apply(f)
    grouped.aggregate(freduce)
    grouped.aggregate({'C': freduce, 'D': freduce})
    grouped.transform(f)
    grouped['C'].apply(f)
    grouped['C'].aggregate(freduce)
    grouped['C'].aggregate([freduce, freducex])
    grouped['C'].transform(f)

def test_group_name_available_in_inference_pass() -> None:
    df: DataFrame = DataFrame({'a': [0, 0, 1, 1, 2, 2], 'b': np.arange(6)})
    names: List[Any] = []

    def f(group: Any) -> Any:
        names.append(group.name)
        return group.copy()
    df.groupby('a', sort=False, group_keys=False).apply(f)
    expected_names: List[Any] = [0, 1, 2]
    assert names == expected_names

def test_no_dummy_key_names(df: DataFrame) -> None:
    result: Any = df.groupby(df['A'].values).sum()
    assert result.index.name is None
    result2: Any = df.groupby([df['A'].values, df['B'].values]).sum()
    assert result2.index.names == (None, None)

def test_groupby_sort_multiindex_series() -> None:
    index: MultiIndex = MultiIndex(levels=[[1, 2], [1, 2]], codes=[[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]], names=['a', 'b'])
    mseries: Series = Series([0, 1, 2, 3, 4, 5], index=index)
    index = MultiIndex(levels=[[1, 2], [1, 2]], codes=[[0, 0, 1], [1, 0, 0]], names=['a', 'b'])
    mseries_result: Series = Series([0, 2, 4], index=index)
    result: Series = mseries.groupby(level=['a', 'b'], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=['a', 'b'], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())

def test_groupby_reindex_inside_function() -> None:
    periods: int = 1000
    ind: Any = date_range(start='2012/1/1', freq='5min', periods=periods)
    df: DataFrame = DataFrame({'high': np.arange(periods), 'low': np.arange(periods)}, index=ind)

    def agg_before(func: Callable[[Any], Any], fix: bool=False) -> Callable[[DataFrame], Any]:
        def _func(data: DataFrame) -> Any:
            d: DataFrame = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)
        return _func
    grouped: Any = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad: Any = grouped.agg({'high': agg_before(np.max)})
    closure_good: Any = grouped.agg({'high': agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)

def test_groupby_multiindex_missing_pair() -> None:
    df: DataFrame = DataFrame({
        'group1': ['a', 'a', 'a', 'b'],
        'group2': ['c', 'c', 'd', 'c'],
        'value': [1, 1, 1, 5]
    })
    df = df.set_index(['group1', 'group2'])
    df_grouped: Any = df.groupby(level=['group1', 'group2'], sort=True)
    res: DataFrame = df_grouped.agg('sum')
    idx: MultiIndex = MultiIndex.from_tuples([('a', 'c'), ('a', 'd'), ('b', 'c')], names=['group1', 'group2'])
    exp: DataFrame = DataFrame([[2], [1], [5]], index=idx, columns=['value'])
    tm.assert_frame_equal(res, exp)

def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi: MultiIndex = MultiIndex.from_tuples([('a', ''), ('b1', 'c1'), ('b2', 'c2')], names=['b', 'c'])
    lexsorted_df: DataFrame = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df: DataFrame = DataFrame(columns=['a', 'b', 'c', 'd'], data=[[1, 'b1', 'c1', 3], [1, 'b2', 'c2', 4]])
    not_lexsorted_df = not_lexsorted_df.pivot_table(index='a', columns=['b', 'c'], values='d')
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected: DataFrame = lexsorted_df.groupby('a').mean()
    with tm.assert_produces_warning(performance_warning):
        result: DataFrame = not_lexsorted_df.groupby('a').mean()
    tm.assert_frame_equal(expected, result)
    df: DataFrame = DataFrame({'x': ['a', 'a', 'b', 'a'], 'y': [1, 1, 2, 2], 'z': [1, 2, 3, 4]}).set_index(['x', 'y'])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(level=level, sort=sort, group_keys=False).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(level=level, sort=sort, group_keys=False).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)

def test_index_label_overlaps_location() -> None:
    df: DataFrame = DataFrame(list('ABCDE'), index=[2, 0, 2, 1, 1])
    g: Any = df.groupby(list('ababb'))
    actual: DataFrame = g.filter(lambda x: len(x) > 2)
    expected: DataFrame = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser: Series = df[0]
    g = ser.groupby(list('ababb'))
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(list('ababb'))
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(list('ababb'))
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)

def test_transform_doesnt_clobber_ints() -> None:
    n: int = 6
    x: np.ndarray = np.arange(n)
    df: DataFrame = DataFrame({'a': x // 2, 'b': 2.0 * x, 'c': 3.0 * x})
    df2: DataFrame = DataFrame({'a': x // 2 * 1.0, 'b': 2.0 * x, 'c': 3.0 * x})
    gb: Any = df.groupby('a')
    result: DataFrame = gb.transform('mean')
    gb2: Any = df2.groupby('a')
    expected: DataFrame = gb2.transform('mean')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('sort_column', ['ints', 'floats', 'strings', ['ints', 'floats'], ['ints', 'strings']])
@pytest.mark.parametrize('group_column', ['int_groups', 'string_groups', ['int_groups', 'string_groups']])
def test_groupby_preserves_sort(sort_column: Any, group_column: Any) -> None:
    df: DataFrame = DataFrame({
        'int_groups': [3, 1, 0, 1, 0, 3, 3, 3],
        'string_groups': ['z', 'a', 'z', 'a', 'a', 'g', 'g', 'g'],
        'ints': [8, 7, 4, 5, 2, 9, 1, 1],
        'floats': [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5],
        'strings': ['z', 'd', 'a', 'e', 'word', 'word2', '42', '47']
    })
    df = df.sort_values(by=sort_column)
    g: Any = df.groupby(group_column)

    def test_sort(x: DataFrame) -> None:
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))
    g.apply(test_sort)

def test_pivot_table_values_key_error() -> None:
    df: DataFrame = DataFrame({
        'eventDate': date_range(datetime.today(), periods=20, freq='ME').tolist(),
        'thename': range(20)
    })
    df['year'] = df.set_index('eventDate').index.year
    df['month'] = df.set_index('eventDate').index.month
    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(index='year', columns='month', values='badname', aggfunc='count')

@pytest.mark.parametrize('columns', ['C', ['C']])
@pytest.mark.parametrize('keys', [['A'], ['A', 'B']])
@pytest.mark.parametrize('values', [
    [True], [0], [0.0], ['a'], Categorical([0]), [to_datetime(0)],
    date_range(0, 1, 1, tz='US/Eastern'),
    pd.period_range('2016-01-01', periods=3, freq='D'),
    pd.array([0], dtype='Int64'), pd.array([0], dtype='Float64'),
    pd.array([False], dtype='boolean')
], ids=['bool', 'int', 'float', 'str', 'cat', 'dt64', 'dt64tz', 'period', 'Int64', 'Float64', 'boolean'])
@pytest.mark.parametrize('method', ['attr', 'agg', 'apply'])
@pytest.mark.parametrize('op', ['idxmax', 'idxmin', 'min', 'max', 'sum', 'prod', 'skew', 'kurt'])
def test_empty_groupby(columns: Any, keys: List[Any], values: Any, method: str, op: str, dropna: Any, using_infer_string: Any) -> None:
    override_dtype: Any = None
    if isinstance(values, BooleanArray) and op in ['sum', 'prod']:
        override_dtype = 'Int64'
    if isinstance(values[0], bool) and op in ('prod', 'sum'):
        override_dtype = 'int64'
    df: DataFrame = DataFrame({'A': values, 'B': values, 'C': values}, columns=list('ABC'))
    if hasattr(values, 'dtype'):
        assert (df.dtypes == values.dtype).all()
    df = df.iloc[:0]
    gb: Any = df.groupby(keys, group_keys=False, dropna=dropna, observed=False)[columns]

    def get_result(**kwargs: Any) -> Any:
        if method == 'attr':
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    def get_categorical_invalid_expected() -> DataFrame:
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            idx = Index(lev, name=keys[0])
        if using_infer_string:
            columns = Index([], dtype='str')
        else:
            columns = []
        expected = DataFrame([], columns=columns, index=idx)
        return expected
    is_per: bool = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    is_dt64: bool = df.dtypes.iloc[0].kind == 'M'
    is_cat: bool = isinstance(values, Categorical)
    is_str: bool = isinstance(df.dtypes.iloc[0], pd.StringDtype)
    if isinstance(values, Categorical) and (not values.ordered) and (op in ['min', 'max', 'idxmin', 'idxmax']):
        if op in ['min', 'max']:
            msg = f'Cannot perform {op} with non-ordered Categorical'
            klass = TypeError
        else:
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()
        if op in ['min', 'max', 'idxmin', 'idxmax'] and isinstance(columns, list):
            result = get_result(numeric_only=True)
            expected = get_categorical_invalid_expected()
            tm.assert_equal(result, expected)
        return
    if op in ['prod', 'sum', 'skew', 'kurt']:
        if is_dt64 or is_cat or is_per or (is_str and op != 'sum'):
            if is_dt64:
                msg = 'datetime64 type does not support'
            elif is_per:
                msg = 'Period type does not support'
            elif is_str:
                msg = f"dtype 'str' does not support operation '{op}'"
            else:
                msg = 'category type does not support'
            if op in ['skew', 'kurt']:
                msg = '|'.join([msg, f"does not support operation '{op}'"])
            with pytest.raises(TypeError, match=msg):
                get_result()
            if not isinstance(columns, list):
                return
            elif op in ['skew', 'kurt']:
                return
            else:
                result = get_result(numeric_only=True)
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                tm.assert_equal(result, expected)
                return
    result = get_result()
    expected = df.set_index(keys)[columns]
    if op in ['idxmax', 'idxmin']:
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)

def test_empty_groupby_apply_nonunique_columns() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((0, 4)))
    df[3] = df[3].astype(np.int64)
    df.columns = [0, 1, 2, 0]
    gb: Any = df.groupby(df[1], group_keys=False)
    res: DataFrame = gb.apply(lambda x: x)
    assert (res.dtypes == df.drop(columns=1).dtypes).all()

def test_tuple_as_grouping() -> None:
    df: DataFrame = DataFrame({('a', 'b'): [1, 1, 1, 1], 'a': [2, 2, 2, 2], 'b': [2, 2, 2, 2], 'c': [1, 1, 1, 1]})
    with pytest.raises(KeyError, match="\\('a', 'b'\\)"):
        df[['a', 'b', 'c']].groupby(('a', 'b'))
    result: Series = df.groupby(('a', 'b'))['c'].sum()
    expected: Series = Series([4], name='c', index=Index([1], name=('a', 'b')))
    tm.assert_series_equal(result, expected)

def test_tuple_correct_keyerror() -> None:
    df: DataFrame = DataFrame(1, index=range(3), columns=MultiIndex.from_product([[1, 2], [3, 4]]))
    with pytest.raises(KeyError, match='^\\(7, 8\\)$'):
        df.groupby((7, 8)).mean()

def test_groupby_agg_ohlc_non_first() -> None:
    df: DataFrame = DataFrame([[1], [1]], columns=Index(['foo'], name='mycols'), index=date_range('2018-01-01', periods=2, freq='D', name='dti'))
    expected: DataFrame = DataFrame([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 
                                      columns=MultiIndex.from_tuples((('foo', 'sum', 'foo'), ('foo', 'ohlc', 'open'), ('foo', 'ohlc', 'high'), ('foo', 'ohlc', 'low'), ('foo', 'ohlc', 'close')), names=['mycols', None, None]), 
                                      index=date_range('2018-01-01', periods=2, freq='D', name='dti'))
    result: DataFrame = df.groupby(Grouper(freq='D')).agg(['sum', 'ohlc'])
    tm.assert_frame_equal(result, expected)

def test_groupby_multiindex_nat() -> None:
    values: List[Any] = [(pd.NaT, 'a'), (datetime(2012, 1, 2), 'a'), (datetime(2012, 1, 2), 'b'), (datetime(2012, 1, 3), 'a')]
    mi: MultiIndex = MultiIndex.from_tuples(values, names=['date', None])
    ser: Series = Series([3, 2, 2.5, 4], index=mi)
    result: Series = ser.groupby(level=1).mean()
    expected: Series = Series([3.0, 2.5], index=['a', 'b'])
    tm.assert_series_equal(result, expected)

def test_groupby_empty_list_raises() -> None:
    values = zip(range(10), range(10))
    df: DataFrame = DataFrame(values, columns=['apple', 'b'])
    msg: str = 'Grouper and axis must be same length'
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])

def test_groupby_multiindex_series_keys_len_equal_group_axis() -> None:
    index_array: List[List[Any]] = [['x', 'x'], ['a', 'b'], ['k', 'k']]
    index_names: List[str] = ['first', 'second', 'third']
    ri: MultiIndex = MultiIndex.from_arrays(index_array, names=index_names)
    s: Series = Series(data=[1, 2], index=ri)
    result: Series = s.groupby(['first', 'third']).sum()
    index_array = [['x'], ['k']]
    index_names = ['first', 'third']
    ei: MultiIndex = MultiIndex.from_arrays(index_array, names=index_names)
    expected: Series = Series([3], index=ei)
    tm.assert_series_equal(result, expected)

def test_groupby_groups_in_BaseGrouper() -> None:
    mi: MultiIndex = MultiIndex.from_product([['A', 'B'], ['C', 'D']], names=['alpha', 'beta'])
    df: DataFrame = DataFrame({'foo': [1, 2, 1, 2], 'bar': [1, 2, 3, 4]}, index=mi)
    result: Any = df.groupby([Grouper(level='alpha'), 'beta'])
    expected: Any = df.groupby(['alpha', 'beta'])
    assert result.groups == expected.groups
    result = df.groupby(['beta', Grouper(level='alpha')])
    expected = df.groupby(['beta', 'alpha'])
    assert result.groups == expected.groups

def test_groups_sort_dropna(sort: Any, dropna: Any) -> None:
    df: DataFrame = DataFrame([[2.0, 1.0], [np.nan, 4.0], [0.0, 3.0]])
    keys: List[Any] = [(2.0, 1.0), (np.nan, 4.0), (0.0, 3.0)]
    values: List[RangeIndex] = [RangeIndex(0, 1), RangeIndex(1, 2), RangeIndex(2, 3)]
    if sort:
        taker: List[int] = [2, 0] if dropna else [2, 0, 1]
    else:
        taker = [0, 2] if dropna else [0, 1, 2]
    expected_dict = {keys[idx]: values[idx] for idx in taker}
    gb: Any = df.groupby([0, 1], sort=sort, dropna=dropna)
    result_dict: Any = gb.groups
    for result_key, expected_key in zip(result_dict.keys(), expected_dict.keys()):
        result_key_arr = np.array(result_key)
        expected_key_arr = np.array(expected_key)
        tm.assert_numpy_array_equal(result_key_arr, expected_key_arr)
    for result_value, expected_value in zip(result_dict.values(), expected_dict.values()):
        tm.assert_index_equal(result_value, expected_value)

@pytest.mark.parametrize('op, expected', [
    ('shift', {'time': [None, None, Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), None, None]}),
    ('bfill', {'time': [Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00'), Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00')]}),
    ('ffill', {'time': [Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00')]})
])
def test_shift_bfill_ffill_tz(tz_naive_fixture: Any, op: str, expected: Any) -> None:
    tz: Any = tz_naive_fixture
    data: dict = {
        'id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'time': [Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), None, None, Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00')]
    }
    df: DataFrame = DataFrame(data).assign(time=lambda x: x.time.dt.tz_localize(tz))
    grouped: Any = df.groupby('id')
    result: DataFrame = getattr(grouped, op)()
    expected_df: DataFrame = DataFrame(expected).assign(time=lambda x: x.time.dt.tz_localize(tz))
    tm.assert_frame_equal(result, expected_df)

def test_groupby_only_none_group() -> None:
    df: DataFrame = DataFrame({'g': [None], 'x': 1})
    actual: Any = df.groupby('g')['x'].transform('sum')
    expected: Series = Series([np.nan], name='x')
    tm.assert_series_equal(actual, expected)

def test_groupby_duplicate_index() -> None:
    ser: Series = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    gb: Any = ser.groupby(level=0)
    result: Series = gb.mean()
    expected: Series = Series([2, 5.5, 8], index=[2.0, 4.0, 5.0])
    tm.assert_series_equal(result, expected)

def test_group_on_empty_multiindex(transformation_func: str, request: Any) -> None:
    df: DataFrame = DataFrame(data=[[1, Timestamp('today'), 3, 4]], columns=['col_1', 'col_2', 'col_3', 'col_4'])
    df['col_3'] = df['col_3'].astype(int)
    df['col_4'] = df['col_4'].astype(int)
    df = df.set_index(['col_1', 'col_2'])
    result: Any = df.iloc[:0].groupby(['col_1']).transform(transformation_func)
    expected: Any = df.groupby(['col_1']).transform(transformation_func).iloc[:0]
    if transformation_func in ('diff', 'shift'):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)
    result = df['col_3'].iloc[:0].groupby(['col_1']).transform(transformation_func)
    expected = df['col_3'].groupby(['col_1']).transform(transformation_func).iloc[:0]
    if transformation_func in ('diff', 'shift'):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)

def test_groupby_crash_on_nunique() -> None:
    dti: Any = date_range('2016-01-01', periods=2, name='foo')
    df: DataFrame = DataFrame({('A', 'B'): [1, 2], ('A', 'C'): [1, 3], ('D', 'B'): [0, 0]})
    df.columns.names = ('bar', 'baz')
    df.index = dti
    df = df.T
    gb: Any = df.groupby(level=0)
    result: DataFrame = gb.nunique()
    expected: DataFrame = DataFrame({'A': [1, 2], 'D': [1, 1]}, index=dti)
    expected.columns.name = 'bar'
    expected = expected.T
    tm.assert_frame_equal(result, expected)
    gb2: Any = df[[]].groupby(level=0)
    exp: DataFrame = expected[[]]
    res: DataFrame = gb2.nunique()
    tm.assert_frame_equal(res, exp)

def test_groupby_list_level() -> None:
    expected: DataFrame = DataFrame(np.arange(0, 9).reshape(3, 3), dtype=float)
    result: Any = expected.groupby(level=[0]).mean()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('max_seq_items, expected', [
    (5, '{0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}'),
    (4, '{0: [0], 1: [1], 2: [2], 3: [3], ...}'),
    (1, '{0: [0], ...}')
])
def test_groups_repr_truncates(max_seq_items: int, expected: str) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 1)))
    df['a'] = df.index
    with pd.option_context('display.max_seq_items', max_seq_items):
        result_repr: str = df.groupby('a').groups.__repr__()
        assert result_repr == expected
        result_repr = df.groupby(np.array(df.a)).groups.__repr__()
        assert result_repr == expected

def test_group_on_two_row_multiindex_returns_one_tuple_key() -> None:
    df: DataFrame = DataFrame([{'a': 1, 'b': 2, 'c': 99}, {'a': 1, 'b': 2, 'c': 88}])
    df = df.set_index(['a', 'b'])
    grp: Any = df.groupby(['a', 'b'])
    result: Any = grp.indices
    expected: dict = {(1, 2): np.array([0, 1], dtype=np.int64)}
    assert len(result) == 1
    key: tuple = (1, 2)
    assert (result[key] == expected[key]).all()

@pytest.mark.parametrize('klass, attr, value', [
    (DataFrame, 'level', 'a'),
    (DataFrame, 'as_index', False),
    (DataFrame, 'sort', False),
    (DataFrame, 'group_keys', False),
    (DataFrame, 'observed', True),
    (DataFrame, 'dropna', False),
    (Series, 'level', 'a'),
    (Series, 'as_index', False),
    (Series, 'sort', False),
    (Series, 'group_keys', False),
    (Series, 'observed', True),
    (Series, 'dropna', False)
])
def test_subsetting_columns_keeps_attrs(klass: Any, attr: str, value: Any) -> None:
    df: DataFrame = DataFrame({'a': [1], 'b': [2], 'c': [3]})
    if attr != 'axis':
        df = df.set_index('a')
    expected: Any = df.groupby('a', **{attr: value})
    result: Any = expected[['b']] if klass is DataFrame else expected['b']
    assert getattr(result, attr) == getattr(expected, attr)

@pytest.mark.parametrize('func', ['sum', 'any', 'shift'])
def test_groupby_column_index_name_lost(func: str) -> None:
    expected: Index = Index(['a'], name='idx')
    df: DataFrame = DataFrame([[1]], columns=expected)
    df_grouped: Any = df.groupby([1])
    result: Any = getattr(df_grouped, func)().columns
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
def test_groupby_duplicate_columns(infer_string: bool) -> None:
    if infer_string:
        pytest.importorskip('pyarrow')
    df: DataFrame = DataFrame({'A': ['f', 'e', 'g', 'h'], 'B': ['a', 'b', 'c', 'd'], 'C': [1, 2, 3, 4]}).astype(object)
    df.columns = ['A', 'B', 'B']
    with pd.option_context('future.infer_string', infer_string):
        result: Any = df.groupby([0, 0, 0, 0]).min()
    expected: DataFrame = DataFrame([['e', 'a', 1]], index=np.array([0]), columns=['A', 'B', 'B'], dtype=object)
    tm.assert_frame_equal(result, expected)

def test_groupby_series_with_tuple_name() -> None:
    ser: Series = Series([1, 2, 3, 4], index=[1, 1, 2, 2], name=('a', 'a'))
    ser.index.name = ('b', 'b')
    result: Series = ser.groupby(level=0).last()
    expected: Series = Series([2, 4], index=[1, 2], name=('a', 'a'))
    expected.index.name = ('b', 'b')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func, values', [('sum', [97.0, 98.0]), ('mean', [24.25, 24.5])])
def test_groupby_numerical_stability_sum_mean(func: str, values: List[float]) -> None:
    data: List[float] = [1e+16, 1e+16, 97, 98, -5000000000000000.0, -5000000000000000.0, -5000000000000000.0, -5000000000000000.0]
    df: DataFrame = DataFrame({'group': [1, 2] * 4, 'a': data, 'b': data})
    result: DataFrame = getattr(df.groupby('group'), func)()
    expected: DataFrame = DataFrame({'a': values, 'b': values}, index=Index([1, 2], name='group'))
    tm.assert_frame_equal(result, expected)

def test_groupby_numerical_stability_cumsum() -> None:
    data: List[float] = [1e+16, 1e+16, 97, 98, -5000000000000000.0, -5000000000000000.0, -5000000000000000.0, -5000000000000000.0]
    df: DataFrame = DataFrame({'group': [1, 2] * 4, 'a': data, 'b': data})
    result: DataFrame = df.groupby('group').cumsum()
    exp_data: List[float] = [1e+16] * 2 + [1e+16 + 96, 1e+16 + 98] + [5000000000000000.0 + 97, 5000000000000000.0 + 98] + [97.0, 98.0]
    expected: DataFrame = DataFrame({'a': exp_data, 'b': exp_data})
    tm.assert_frame_equal(result, expected, check_exact=True)

def test_groupby_cumsum_skipna_false() -> None:
    arr: np.ndarray = np.random.default_rng(2).standard_normal((5, 5))
    df: DataFrame = DataFrame(arr)
    for i in range(5):
        df.iloc[i, i] = np.nan
    df['A'] = 1
    gb: Any = df.groupby('A')
    res: DataFrame = gb.cumsum(skipna=False)
    expected: DataFrame = df[[0, 1, 2, 3, 4]].cumsum(skipna=False)
    tm.assert_frame_equal(res, expected)

def test_groupby_cumsum_timedelta64() -> None:
    dti: Any = date_range('2016-01-01', periods=5)
    ser: Series = Series(dti) - dti[0]
    ser[2] = pd.NaT
    df: DataFrame = DataFrame({'A': 1, 'B': ser})
    gb: Any = df.groupby('A')
    res: DataFrame = gb.cumsum(numeric_only=False, skipna=True)
    exp: DataFrame = DataFrame({'B': [ser[0], ser[1], pd.NaT, ser[4], ser[4] * 2]})
    tm.assert_frame_equal(res, exp)
    res = gb.cumsum(numeric_only=False, skipna=False)
    exp = DataFrame({'B': [ser[0], ser[1], pd.NaT, pd.NaT, pd.NaT]})
    tm.assert_frame_equal(res, exp)

def test_groupby_mean_duplicate_index(rand_series_with_duplicate_datetimeindex: Any) -> None:
    dups: Any = rand_series_with_duplicate_datetimeindex
    result: Any = dups.groupby(level=0).mean()
    expected: Any = dups.groupby(dups.index).mean()
    tm.assert_series_equal(result, expected)

def test_groupby_all_nan_groups_drop() -> None:
    s: Series = Series([1, 2, 3], [np.nan, np.nan, np.nan])
    result: Any = s.groupby(s.index).sum()
    expected: Series = Series([], index=Index([], dtype=np.float64), dtype=np.int64)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('numeric_only', [True, False])
def test_groupby_empty_multi_column(as_index: Any, numeric_only: bool) -> None:
    df: DataFrame = DataFrame(data=[], columns=['A', 'B', 'C'])
    gb: Any = df.groupby(['A', 'B'], as_index=as_index)
    result: Any = gb.sum(numeric_only=numeric_only)
    if as_index:
        index: Any = MultiIndex([[], []], [[], []], names=['A', 'B'])
        columns: Any = ['C'] if not numeric_only else Index([], dtype='str')
    else:
        index = RangeIndex(0)
        columns = ['A', 'B', 'C'] if not numeric_only else ['A', 'B']
    expected: DataFrame = DataFrame([], columns=columns, index=index)
    tm.assert_frame_equal(result, expected)

def test_groupby_aggregation_non_numeric_dtype() -> None:
    df: DataFrame = DataFrame([['M', [1]], ['M', [1]], ['W', [10]], ['W', [20]]], columns=['MW', 'v'])
    expected: DataFrame = DataFrame({'v': [[1, 1], [10, 20]]}, index=Index(['M', 'W'], name='MW'))
    gb: Any = df.groupby(by=['MW'])
    result: DataFrame = gb.sum()
    tm.assert_frame_equal(result, expected)

def test_groupby_aggregation_multi_non_numeric_dtype() -> None:
    df: DataFrame = DataFrame({
        'x': [1, 0, 1, 1, 0],
        'y': [Timedelta(i, 'days') for i in range(1, 6)],
        'z': [Timedelta(i * 10, 'days') for i in range(1, 6)]
    })
    expected: DataFrame = DataFrame({
        'y': [Timedelta(i, 'days') for i in range(7, 9)],
        'z': [Timedelta(i * 10, 'days') for i in range(7, 9)]
    }, index=Index([0, 1], dtype='int64', name='x'))
    gb: Any = df.groupby(by=['x'])
    result: DataFrame = gb.sum()
    tm.assert_frame_equal(result, expected)

def test_groupby_aggregation_numeric_with_non_numeric_dtype() -> None:
    df: DataFrame = DataFrame({
        'x': [1, 0, 1, 1, 0],
        'y': [Timedelta(i, 'days') for i in range(1, 6)],
        'z': list(range(1, 6))
    })
    expected: DataFrame = DataFrame({
        'y': [Timedelta(7, 'days'), Timedelta(8, 'days')],
        'z': [7, 8]
    }, index=Index([0, 1], dtype='int64', name='x'))
    gb: Any = df.groupby(by=['x'])
    result: DataFrame = gb.sum()
    tm.assert_frame_equal(result, expected)

def test_groupby_filtered_df_std() -> None:
    dicts: List[dict] = [
        {'filter_col': False, 'groupby_col': True, 'bool_col': True, 'float_col': 10.5},
        {'filter_col': True, 'groupby_col': True, 'bool_col': True, 'float_col': 20.5},
        {'filter_col': True, 'groupby_col': True, 'bool_col': True, 'float_col': 30.5}
    ]
    df: DataFrame = DataFrame(dicts)
    df_filter: DataFrame = df[df['filter_col'] == True]
    dfgb: Any = df_filter.groupby('groupby_col')
    result: DataFrame = dfgb.std()
    expected: DataFrame = DataFrame([[0.0, 0.0, 7.071068]], columns=['filter_col', 'bool_col', 'float_col'], index=Index([True], name='groupby_col'))
    tm.assert_frame_equal(result, expected)

def test_datetime_categorical_multikey_groupby_indices() -> None:
    df: DataFrame = DataFrame({
        'a': Series(list('abc')),
        'b': Series(to_datetime(['2018-01-01', '2018-02-01', '2018-03-01']), dtype='category'),
        'c': Categorical.from_codes([-1, 0, 1], categories=[0, 1])
    })
    result: dict = df.groupby(['a', 'b'], observed=False).indices
    expected: dict = {
        ('a', Timestamp('2018-01-01 00:00:00')): np.array([0]),
        ('b', Timestamp('2018-02-01 00:00:00')): np.array([1]),
        ('c', Timestamp('2018-03-01 00:00:00')): np.array([2])
    }
    assert result == expected

def test_rolling_wrong_param_min_period() -> None:
    name_l: List[str] = ['Alice'] * 5 + ['Bob'] * 5
    val_l: List[Any] = [np.nan, np.nan, 1, 2, 3] + [np.nan, 1, 2, 3, 4]
    test_df: DataFrame = DataFrame([name_l, val_l]).T
    test_df.columns = ['name', 'val']
    result_error_msg: str = "^[a-zA-Z._]*\\(\\) got an unexpected keyword argument 'min_period'"
    with pytest.raises(TypeError, match=result_error_msg):
        test_df.groupby('name')['val'].rolling(window=2, min_period=1).sum()

def test_by_column_values_with_same_starting_value(any_string_dtype: Any) -> None:
    df: DataFrame = DataFrame({
        'Name': ['Thomas', 'Thomas', 'Thomas John'],
        'Credit': [1200, 1300, 900],
        'Mood': Series(['sad', 'happy', 'happy'], dtype=any_string_dtype)
    })
    aggregate_details: dict = {'Mood': Series.mode, 'Credit': 'sum'}
    result: DataFrame = df.groupby(['Name']).agg(aggregate_details)
    expected_result: DataFrame = DataFrame({'Mood': [['happy', 'sad'], 'happy'], 'Credit': [2500, 900], 'Name': ['Thomas', 'Thomas John']}).set_index('Name')
    tm.assert_frame_equal(result, expected_result)

def test_groupby_none_in_first_mi_level() -> None:
    arr: List[List[Any]] = [[None, 1, 0, 1], [2, 3, 2, 3]]
    ser: Series = Series(1, index=MultiIndex.from_arrays(arr, names=['a', 'b']))
    result: Series = ser.groupby(level=[0, 1]).sum()
    expected: Series = Series([1, 2], MultiIndex.from_tuples([(0.0, 2), (1.0, 3)], names=['a', 'b']))
    tm.assert_series_equal(result, expected)

def test_groupby_none_column_name(using_infer_string: Any) -> None:
    df: DataFrame = DataFrame({None: [1, 1, 2, 2], 'b': [1, 1, 2, 3], 'c': [4, 5, 6, 7]})
    by: List[Any] = [np.nan] if using_infer_string else [None]
    gb: Any = df.groupby(by=by)
    result: Any = gb.sum()
    expected: DataFrame = DataFrame({'b': [2, 5], 'c': [9, 13]}, index=Index([1, 2], name=by[0]))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('selection', [None, 'a', ['a']])
def test_single_element_list_grouping(selection: Union[None, str, List[str]]) -> None:
    df: DataFrame = DataFrame({'a': [1, 2], 'b': [np.nan, 5], 'c': [np.nan, 2]}, index=['x', 'y'])
    grouped: Any = df.groupby(['a']) if selection is None else df.groupby(['a'])[selection]
    result: List[Any] = [key for key, _ in grouped]
    expected: List[tuple] = [(1,), (2,)]
    assert result == expected

def test_groupby_string_dtype() -> None:
    df: DataFrame = DataFrame({'str_col': ['a', 'b', 'c', 'a'], 'num_col': [1, 2, 3, 2]})
    df['str_col'] = df['str_col'].astype('string')
    expected: DataFrame = DataFrame({'str_col': ['a', 'b', 'c'], 'num_col': [1.5, 2.0, 3.0]})
    expected['str_col'] = expected['str_col'].astype('string')
    grouped: Any = df.groupby('str_col', as_index=False)
    result: DataFrame = grouped.mean()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('level_arg, multiindex', [([0], False), ((0,), False), ([0], True), ((0,), True)])
def test_single_element_listlike_level_grouping(level_arg: Union[List[int], tuple], multiindex: bool) -> None:
    df: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, index=['x', 'y'])
    if multiindex:
        df = df.set_index(['a', 'b'])
    result: List[Any] = [key for key, _ in df.groupby(level=level_arg)]
    expected: List[tuple] = [(1,), (2,)] if multiindex else [('x',), ('y',)]
    assert result == expected

@pytest.mark.parametrize('func', ['sum', 'cumsum', 'cumprod', 'prod'])
def test_groupby_avoid_casting_to_float(func: str) -> None:
    val: int = 922337203685477580
    df: DataFrame = DataFrame({'a': 1, 'b': [val]})
    result: DataFrame = getattr(df.groupby('a'), func)() - val
    expected: DataFrame = DataFrame({'b': [0]}, index=Index([1], name='a'))
    if func in ['cumsum', 'cumprod']:
        expected = expected.reset_index(drop=True)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func, val', [('sum', 3), ('prod', 2)])
def test_groupby_sum_support_mask(any_numeric_ea_dtype: Any, func: str, val: int) -> None:
    df: DataFrame = DataFrame({'a': 1, 'b': [1, 2, pd.NA]}, dtype=any_numeric_ea_dtype)
    result: DataFrame = getattr(df.groupby('a'), func)()
    expected: DataFrame = DataFrame({'b': [val]}, index=Index([1], name='a', dtype=any_numeric_ea_dtype), dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('skipna, val', [(True, 3), (False, pd.NA)])
def test_groupby_cumsum_mask(any_numeric_ea_dtype: Any, skipna: bool, val: Any) -> None:
    df: DataFrame = DataFrame({'a': 1, 'b': [1, pd.NA, 2]}, dtype=any_numeric_ea_dtype)
    result: DataFrame = df.groupby('a').cumsum(skipna=skipna)
    expected: DataFrame = DataFrame({'b': [1, pd.NA, val]}, dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('val_in, index, val_out', [
    ([1.0, 2.0, 3.0, 4.0, 5.0], ['foo', 'foo', 'bar', 'baz', 'blah'], [3.0, 4.0, 5.0, 3.0]),
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ['foo', 'foo', 'bar', 'baz', 'blah', 'blah'], [3.0, 4.0, 11.0, 3.0])
])
def test_groupby_index_name_in_index_content(val_in: List[float], index: List[str], val_out: List[float]) -> None:
    series: Series = Series(data=val_in, name='values', index=Index(index, name='blah'))
    result: Series = series.groupby('blah').sum()
    expected: Series = Series(data=val_out, name='values', index=Index(['bar', 'baz', 'blah', 'foo'], name='blah'))
    tm.assert_series_equal(result, expected)
    result = series.to_frame().groupby('blah').sum()
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('n', [1, 10, 32, 100, 1000])
def test_sum_of_booleans(n: int) -> None:
    df: DataFrame = DataFrame({'groupby_col': 1, 'bool': [True] * n})
    df['bool'] = df['bool'].eq(True)
    result: DataFrame = df.groupby('groupby_col').sum()
    expected: DataFrame = DataFrame({'bool': [n]}, index=Index([1], name='groupby_col'))
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:invalid value encountered in remainder:RuntimeWarning')
@pytest.mark.parametrize('method', ['head', 'tail', 'nth', 'first', 'last'])
def test_groupby_method_drop_na(method: str) -> None:
    df: DataFrame = DataFrame({'A': ['a', np.nan, 'b', np.nan, 'c'], 'B': range(5)})
    if method == 'nth':
        result: Any = getattr(df.groupby('A'), method)(n=0)
    else:
        result = getattr(df.groupby('A'), method)()
    if method in ['first', 'last']:
        expected: DataFrame = DataFrame({'B': [0, 2, 4]}).set_index(Series(['a', 'b', 'c'], name='A'))
    else:
        expected = DataFrame({'A': ['a', 'b', 'c'], 'B': [0, 2, 4]}, index=range(0, 6, 2))
    tm.assert_frame_equal(result, expected)

def test_groupby_reduce_period() -> None:
    pi: Any = pd.period_range('2016-01-01', periods=100, freq='D')
    grps: List[int] = list(range(10)) * 10
    ser: Series = pi.to_series()
    gb: Any = ser.groupby(grps)
    with pytest.raises(TypeError, match='Period type does not support sum operations'):
        gb.sum()
    with pytest.raises(TypeError, match='Period type does not support cumsum operations'):
        gb.cumsum()
    with pytest.raises(TypeError, match='Period type does not support prod operations'):
        gb.prod()
    with pytest.raises(TypeError, match='Period type does not support cumprod operations'):
        gb.cumprod()
    res: Any = gb.max()
    expected: Series = ser[-10:]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)
    res = gb.min()
    expected = ser[:10]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)

def test_obj_with_exclusions_duplicate_columns() -> None:
    df: DataFrame = DataFrame([[0, 1, 2, 3]])
    df.columns = [0, 1, 2, 0]
    gb: Any = df.groupby(df[1])
    result: DataFrame = gb._obj_with_exclusions
    expected: DataFrame = df.take([0, 2, 3], axis=1)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('numeric_only', [True, False])
def test_groupby_numeric_only_std_no_result(numeric_only: bool) -> None:
    dicts_non_numeric: List[dict] = [{'a': 'foo', 'b': 'bar'}, {'a': 'car', 'b': 'dar'}]
    df: DataFrame = DataFrame(dicts_non_numeric, dtype=object)
    dfgb: Any = df.groupby('a', as_index=False, sort=False)
    if numeric_only:
        result: Any = dfgb.std(numeric_only=True)
        expected_df: DataFrame = DataFrame(['foo', 'car'], columns=['a'])
        tm.assert_frame_equal(result, expected_df)
    else:
        with pytest.raises(ValueError, match="could not convert string to float: 'bar'"):
            dfgb.std(numeric_only=numeric_only)

def test_groupby_agg_namedagg_with_duplicate_columns() -> None:
    df: DataFrame = DataFrame({
        'col1': [2, 1, 1, 0, 2, 0],
        'col2': [4, 5, 36, 7, 4, 5],
        'col3': [3.1, 8.0, 12, 10, 4, 1.1],
        'col4': [17, 3, 16, 15, 5, 6],
        'col5': [-1, 3, -1, 3, -2, -1]
    })
    result: DataFrame = df.groupby(by=['col1', 'col1', 'col2'], as_index=False).agg(
        new_col=pd.NamedAgg(column='col1', aggfunc='min'),
        new_col1=pd.NamedAgg(column='col1', aggfunc='max'),
        new_col2=pd.NamedAgg(column='col2', aggfunc='count')
    )
    expected: DataFrame = DataFrame({
        'col1': [0, 0, 1, 1, 2],
        'col2': [5, 7, 5, 36, 4],
        'new_col': [0, 0, 1, 1, 2],
        'new_col1': [0, 0, 1, 1, 2],
        'new_col2': [1, 1, 1, 1, 2]
    })
    tm.assert_frame_equal(result, expected)

def test_groupby_multi_index_codes() -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 3, 4], 'B': [1, float('nan'), 2, float('nan')], 'C': [2, 4, 6, 8]})
    df_grouped: Any = df.groupby(['A', 'B'], dropna=False).sum()
    index: Any = df_grouped.index
    tm.assert_index_equal(index, MultiIndex.from_frame(index.to_frame()))

# End of annotated code.
