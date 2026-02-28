from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, CategoricalIndex, DataFrame, Index, MultiIndex, Series, qcut
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args

def cartesian_product_for_groupers(result: pd.DataFrame, args: list, names: list, fill_value: np.ndarray = np.nan) -> pd.DataFrame:
    """Reindex to a cartesian production for the groupers,
    preserving the nature (Categorical) of each grouper
    """

    def f(a: Categorical) -> Categorical:
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(np.arange(len(categories)), categories=categories, ordered=a.ordered)
        return a
    index = MultiIndex.from_product(map(f, args), names=names)
    return result.reindex(index, fill_value=fill_value).sort_index()

_results_for_groupbys_with_missing_categories: dict = {'all': True, 'any': False, 'count': 0, 'corrwith': np.nan, 'first': np.nan, 'idxmax': np.nan, 'idxmin': np.nan, 'last': np.nan, 'max': np.nan, 'mean': np.nan, 'median': np.nan, 'min': np.nan, 'nth': np.nan, 'nunique': 0, 'prod': 1, 'quantile': np.nan, 'sem': np.nan, 'size': 0, 'skew': np.nan, 'kurt': np.nan, 'std': np.nan, 'sum': 0, 'var': np.nan}

def test_apply_use_categorical_name(df: pd.DataFrame) -> None:
    cats: pd.Categorical = qcut(df.C, 4)

    def get_stats(group: pd.Series) -> dict:
        return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
    result: pd.DataFrame = df.groupby(cats, observed=False).D.apply(get_stats)
    assert result.index.names[0] == 'C'

def test_basic() -> None:
    cats: pd.Categorical = Categorical(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], categories=['a', 'b', 'c', 'd'], ordered=True)
    data: pd.DataFrame = DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 4, 5], 'b': cats})
    exp_index: pd.CategoricalIndex = CategoricalIndex(list('abcd'), name='b', ordered=True)
    expected: pd.DataFrame = DataFrame({'a': [1, 2, 4, np.nan]}, index=exp_index)
    result: pd.DataFrame = data.groupby('b', observed=False).mean()
    tm.assert_frame_equal(result, expected)

def test_basic_single_grouper() -> None:
    cat1: pd.Categorical = Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'z'], ordered=True)
    cat2: pd.Categorical = Categorical(['c', 'd', 'c', 'd'], categories=['c', 'd', 'y'], ordered=True)
    df: pd.DataFrame = DataFrame({'A': cat1, 'B': cat2, 'values': [1, 2, 3, 4]})
    gb: pd.DataFrameGroupBy = df.groupby('A', observed=False)
    exp_idx: pd.CategoricalIndex = CategoricalIndex(['a', 'b', 'z'], name='A', ordered=True)
    expected: pd.DataFrame = DataFrame({'values': Series([3, 7, 0], index=exp_idx)})
    result: pd.DataFrame = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)

def test_basic_string(using_infer_string: bool, observed: bool) -> None:
    x: pd.DataFrame = DataFrame([[1, 'John P. Doe'], [2, 'Jane Dove'], [1, 'John P. Doe']], columns=['person_id', 'person_name'])
    x['person_name'] = Categorical(x.person_name)
    g: pd.DataFrameGroupBy = x.groupby(['person_id'], observed=observed)
    result: pd.DataFrame = g.transform(lambda x: x)
    tm.assert_frame_equal(result, x[['person_name']])
    result: pd.DataFrame = x.drop_duplicates('person_name')
    expected: pd.DataFrame = x.iloc[[0, 1]]
    tm.assert_frame_equal(result, expected)

    def f(x: pd.Series) -> pd.Series:
        return x.drop_duplicates('person_name').iloc[0]
    result: pd.Series = g.apply(f)
    expected: pd.Series = x[['person_name']].iloc[[0, 1]]
    expected.index = Index([1, 2], name='person_id')
    dtype: str = 'str' if using_infer_string else object
    expected['person_name'] = expected['person_name'].astype(dtype)
    tm.assert_series_equal(result, expected)

def test_basic_monotonic() -> None:
    df: pd.DataFrame = DataFrame({'a': [5, 15, 25]})
    c: pd.Categorical = pd.cut(df.a, bins=[0, 10, 20, 30, 40])
    result: pd.Series = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df['a'])
    tm.assert_series_equal(df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df['a'])
    result: pd.DataFrame = df.groupby(c, observed=False).transform(sum)
    expected: pd.DataFrame = df[['a']]
    tm.assert_frame_equal(result, expected)
    gbc: pd.DataFrameGroupBy = df.groupby(c, observed=False)
    result: pd.DataFrame = gbc.transform(lambda xs: np.max(xs, axis=0))
    tm.assert_frame_equal(result, df[['a']])
    result2: pd.DataFrame = gbc.transform(lambda xs: np.max(xs, axis=0))
    result3: pd.DataFrame = gbc.transform(max)
    result4: pd.DataFrame = gbc.transform(np.maximum.reduce)
    result5: pd.DataFrame = gbc.transform(lambda xs: np.maximum.reduce(xs))
    tm.assert_frame_equal(result2, df[['a']], check_dtype=False)
    tm.assert_frame_equal(result3, df[['a']], check_dtype=False)
    tm.assert_frame_equal(result4, df[['a']])
    tm.assert_frame_equal(result5, df[['a']])
    tm.assert_series_equal(df.a.groupby(c, observed=False).filter(np.all), df['a'])
    tm.assert_frame_equal(df.groupby(c, observed=False).filter(np.all), df)

def test_basic_non_monotonic() -> None:
    df: pd.DataFrame = DataFrame({'a': [5, 15, 25, -5]})
    c: pd.Categorical = pd.cut(df.a, bins=[-10, 0, 10, 20, 30, 40])
    result: pd.Series = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df['a'])
    tm.assert_series_equal(df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df['a'])
    result: pd.DataFrame = df.groupby(c, observed=False).transform(sum)
    expected: pd.DataFrame = df[['a']]
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df[['a']])

def test_basic_cut_grouping() -> None:
    df: pd.DataFrame = DataFrame({'a': [1, 0, 0, 0]})
    c: pd.Categorical = pd.cut(df.a, [0, 1, 2, 3, 4], labels=Categorical(list('abcd')))
    result: pd.Series = df.groupby(c, observed=False).apply(len)
    exp_index: pd.CategoricalIndex = CategoricalIndex(c.values.categories, ordered=c.values.ordered)
    expected: pd.Series = Series([1, 0, 0, 0], index=exp_index)
    expected.index.name = 'a'
    tm.assert_series_equal(result, expected)

def test_more_basic() -> None:
    levels: list = ['foo', 'bar', 'baz', 'qux']
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: pd.Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: pd.DataFrame = data.groupby(cats, observed=False).mean()
    expected: pd.DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    exp_idx: pd.CategoricalIndex = CategoricalIndex(levels, categories=cats.categories, ordered=True)
    expected = expected.reindex(exp_idx)
    tm.assert_frame_equal(result, expected)
    grouped: pd.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: pd.DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: pd.Categorical = cats.take(idx)
    ord_data: pd.DataFrame = data.take(idx)
    exp_cats: pd.Categorical = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp: pd.CategoricalIndex = CategoricalIndex(exp_cats)
    expected = ord_data.groupby(exp_cats, sort=False, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.stack().index.get_level_values(0), exp)
    exp: pd.Index = Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] * 4)
    tm.assert_index_equal(desc_result.stack().index.get_level_values(1), exp)

def test_level_get_group(observed: bool, df: pd.DataFrame) -> None:
    g: pd.DataFrameGroupBy = df.groupby(level=['Index1'], observed=observed)
    expected: pd.DataFrame = DataFrame(data=np.arange(2, 12, 2), index=MultiIndex(levels=[CategoricalIndex(['a', 'b']), range(5)], codes=[[0] * 5, range(5)], names=['Index1', 'Index2']))
    result: pd.DataFrame = g.get_group(('a',))
    tm.assert_frame_equal(result, expected)

def test_sorting_with_different_categoricals() -> None:
    df: pd.DataFrame = DataFrame({'group': ['A'] * 6 + ['B'] * 6, 'dose': ['high', 'med', 'low'] * 4, 'outcomes': np.arange(12.0)})
    df.dose = Categorical(df.dose, categories=['low', 'med', 'high'], ordered=True)
    result: pd.Series = df.groupby('group')['dose'].value_counts()
    result = result.sort_index(level=0, sort_remaining=True)
    index: list = ['low', 'med', 'high', 'low', 'med', 'high']
    index: pd.Categorical = Categorical(index, categories=['low', 'med', 'high'], ordered=True)
    index: pd.MultiIndex = MultiIndex.from_arrays([[['A', 'A', 'A', 'B', 'B', 'B'], index]], names=['group', 'dose'])
    expected: pd.Series = Series([2] * 6, index=index, name='count')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ordered: bool', [True, False])
def test_apply(ordered: bool) -> None:
    dense: pd.Categorical = Categorical(list('abc'), ordered=ordered)
    missing: pd.Categorical = Categorical(list('aaa'), categories=['a', 'b'], ordered=ordered)
    values: np.ndarray = np.arange(len(dense))
    df: pd.DataFrame = DataFrame({'missing': missing, 'dense': dense, 'values': values})
    grouped: pd.DataFrameGroupBy = df.groupby(['missing', 'dense'], observed=True)
    idx: pd.MultiIndex = MultiIndex.from_arrays([missing, dense], names=['missing', 'dense'])
    expected: pd.DataFrame = DataFrame([0, 1, 2.0], index=idx, columns=['values'])
    result: pd.DataFrame = grouped.apply(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)
    result: pd.DataFrame = grouped.mean()
    tm.assert_frame_equal(result, expected)
    result: pd.DataFrame = grouped.agg(np.mean)
    tm.assert_frame_equal(result, expected)
    idx: pd.MultiIndex = MultiIndex.from_arrays([missing, dense], names=['missing', 'dense'])
    expected: pd.Series = Series(1, index=idx)
    result: pd.Series = grouped.apply(lambda x: 1)
    tm.assert_series_equal(result, expected)

def test_observed(request: pytest.Request, using_infer_string: bool, observed: bool) -> None:
    if using_infer_string and (not observed):
        request.applymarker(pytest.mark.xfail(reason='TODO(infer_string)'))
    cat1: pd.Categorical = Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'z'], ordered=True)
    cat2: pd.Categorical = Categorical(['c', 'd', 'c', 'd'], categories=['c', 'd', 'y'], ordered=True)
    df: pd.DataFrame = DataFrame({'A': cat1, 'B': cat2, 'values': [1, 2, 3, 4]})
    df['C'] = ['foo', 'bar'] * 2
    gb: pd.DataFrameGroupBy = df.groupby(['A', 'B', 'C'], observed=observed)
    exp_index: pd.MultiIndex = MultiIndex.from_arrays([cat1, cat2, ['foo', 'bar'] * 2], names=['A', 'B', 'C'])
    expected: pd.DataFrame = DataFrame({'values': Series([1, 2, 3, 4], index=exp_index)}).sort_index()
    result: pd.DataFrame = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(expected, [cat1, cat2, ['foo', 'bar']], list('ABC'), fill_value=0)
    tm.assert_frame_equal(result, expected)
    gb: pd.DataFrameGroupBy = df.groupby(['A', 'B'], observed=observed)
    exp_index: pd.MultiIndex = MultiIndex.from_arrays([cat1, cat2], names=['A', 'B'])
    expected: pd.DataFrame = DataFrame({'values': [1, 2, 3, 4], 'C': ['foo', 'bar', 'foo', 'bar']}, index=exp_index)
    result: pd.DataFrame = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(expected, [cat1, cat2], list('AB'), fill_value=0)
    tm.assert_frame_equal(result, expected)

def test_observed_single_column(observed: bool) -> None:
    d: dict = {'cat': Categorical(['a', 'b', 'a', 'b'], categories=['a', 'b', 'c'], ordered=True), 'ints': [1, 1, 2, 2], 'val': [10, 20, 30, 40]}
    df: pd.DataFrame = DataFrame(d)
    groups_single_key: pd.DataFrameGroupBy = df.groupby('cat', observed=observed)
    result: dict = groups_single_key.groups
    if observed:
        expected: dict = {'a': Index([0, 2], dtype='int64'), 'c': Index([1], dtype='int64')}
    else:
        expected: dict = {'a': Index([0, 2], dtype='int64'), 'b': Index([], dtype='int64'), 'c': Index([], dtype='int64')}
    tm.assert_dict_equal(result, expected)

def test_observed_two_columns(observed: bool) -> None:
    d: dict = {'cat': Categorical(['a', 'b', 'a', 'b'], categories=['a', 'b', 'c'], ordered=True), 'ints': [1, 1, 2, 2], 'val': [10, 20, 30, 40]}
    df: pd.DataFrame = DataFrame(d)
    groups_double_key: pd.DataFrameGroupBy = df.groupby(['cat', 'ints'], observed=observed)
    result: pd.DataFrame = groups_double_key.agg('mean')
    expected: pd.DataFrame = DataFrame({'val': [10.0, 30.0, 20.0, 40.0], 'cat': Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'c'], ordered=True), 'ints': [1, 2, 1, 2]}).set_index(['cat', 'ints'])
    if not observed:
        expected = cartesian_product_for_groupers(expected, [df.cat.values, [1, 2]], ['cat', 'ints'])
    tm.assert_frame_equal(result, expected)
    for key: tuple in [('a', 1), ('b', 2), ('b', 1), ('a', 2)]:
        c, i = key
        result: pd.DataFrame = groups_double_key.get_group(key)
        expected: pd.DataFrame = df[(df.cat == c) & (df.ints == i)]
        tm.assert_frame_equal(result, expected)

def test_observed_with_as_index(observed: bool) -> None:
    d: dict = {'foo': [10, 8, 4, 8, 4, 1, 1], 'bar': [10, 20, 30, 40, 50, 60, 70], 'baz': ['d', 'c', 'e', 'a', 'a', 'd', 'c']}
    df: pd.DataFrame = DataFrame(d)
    cat: pd.Categorical = pd.cut(df['foo'], np.linspace(0, 10, 3))
    df['range'] = cat
    groups: pd.DataFrameGroupBy = df.groupby(['range', 'baz'], as_index=False, observed=observed)
    result: pd.DataFrame = groups.agg('mean')
    groups2: pd.DataFrameGroupBy = df.groupby(['range', 'baz'], as_index=True, observed=observed)
    expected: pd.DataFrame = groups2.agg('mean').reset_index()
    tm.assert_frame_equal(result, expected)

def test_observed_codes_remap(observed: bool) -> None:
    d: dict = {'C1': [3, 3, 4, 5], 'C2': [1, 2, 3, 4], 'C3': [10, 100, 200, 34]}
    df: pd.DataFrame = DataFrame(d)
    values: pd.Categorical = pd.cut(df['C1'], [1, 2, 3, 6])
    values.name = 'cat'
    groups_double_key: pd.DataFrameGroupBy = df.groupby([values, 'C2'], observed=observed)
    idx: pd.MultiIndex = MultiIndex.from_arrays([values, [1, 2, 3, 4]], names=['cat', 'C2'])
    expected: pd.DataFrame = DataFrame({'C1': [3.0, 3.0, 4.0, 5.0], 'C3': [10.0, 100.0, 200.0, 34.0]}, index=idx)
    if not observed:
        expected = cartesian_product_for_groupers(expected, [values.values, [1, 2, 3, 4]], ['cat', 'C2'])
    result: pd.DataFrame = groups_double_key.agg('mean')
    tm.assert_frame_equal(result, expected)

def test_observed_perf() -> None:
    df: pd.DataFrame = DataFrame({'cat': np.random.default_rng(2).integers(0, 255, size=30000), 'int_id': np.random.default_rng(2).integers(0, 255, size=30000), 'other_id': np.random.default_rng(2).integers(0, 10000, size=30000), 'foo': 0})
    df['cat'] = df.cat.astype(str).astype('category')
    grouped: pd.DataFrameGroupBy = df.groupby(['cat', 'int_id', 'other_id'], observed=True)
    result: pd.DataFrame = grouped.count()
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()

def test_observed_groups(observed: bool) -> None:
    cat: pd.Categorical = Categorical(['a', 'c', 'a'], categories=['a', 'b', 'c'])
    df: pd.DataFrame = DataFrame({'cat': cat, 'vals': [1, 2, 3]})
    g: pd.DataFrameGroupBy = df.groupby('cat', observed=observed)
    result: dict = g.groups
    if observed:
        expected: dict = {'a': Index([0, 2], dtype='int64'), 'c': Index([1], dtype='int64')}
    else:
        expected: dict = {'a': Index([0, 2], dtype='int64'), 'b': Index([], dtype='int64'), 'c': Index([], dtype='int64')}
    tm.assert_dict_equal(result, expected)

def test_unobserved_in_index(keys: list, expected_values: list, expected_index_levels: list, test_series: bool) -> None:
    df: pd.DataFrame = DataFrame({'a': Categorical([1, 1, 2], categories=[1, 2, 3]), 'a2': Categorical([1, 1, 2], categories=[1, 2, 3]), 'b': [4, 5, 6], 'c': [7, 8, 9]}).set_index(['a', 'a2'])
    if 'b' not in keys:
        df = df.drop(columns='b')
    gb: pd.DataFrameGroupBy = df.groupby(keys, observed=False)
    if test_series:
        gb = gb['c']
    result: pd.Series = gb.sum()
    if len(keys) == 1:
        index: pd.Index = expected_index_levels
    else:
        codes: np.ndarray = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2], 3 * [0, 1, 2]])
        index: pd.MultiIndex = MultiIndex(expected_index_levels, codes=codes, names=keys)
    expected: pd.DataFrame = DataFrame({'c': expected_values}, index=index)
    if test_series:
        expected = expected['c']
    tm.assert_equal(result, expected)

def test_observed_groups_with_nan(observed: bool) -> None:
    df: pd.DataFrame = DataFrame({'cat': Categorical(['a', np.nan, 'a'], categories=['a', 'b', 'd']), 'vals': [1, 2, 3]})
    g: pd.DataFrameGroupBy = df.groupby('cat', observed=observed)
    result: dict = g.groups
    if observed:
        expected: dict = {'a': Index([0, 2], dtype='int64')}
    else:
        expected: dict = {'a': Index([0, 2], dtype='int64'), 'b': Index([], dtype='int64'), 'd': Index([], dtype='int64')}
    tm.assert_dict_equal(result, expected)

def test_observed_nth() -> None:
    cat: pd.Categorical = Categorical(['a', np.nan, np.nan], categories=['a', 'b', 'c'])
    ser: pd.Series = Series([1, 2, 3])
    df: pd.DataFrame = DataFrame({'cat': cat, 'ser': ser})
    result: pd.Series = df.groupby('cat', observed=False)['ser'].nth(0)
    expected: pd.Series = df['ser'].iloc[[0]]
    tm.assert_series_equal(result, expected)

def test_dataframe_categorical_with_nan(observed: bool) -> None:
    s1: pd.Series = Categorical([np.nan, 'a', np.nan, 'a'], categories=['a', 'b', 'c'])
    s2: pd.Series = Series([1, 2, 3, 4])
    df: pd.DataFrame = DataFrame({'s1': s1, 's2': s2})
    result: pd.DataFrame = df.groupby('s1', observed=observed).first().reset_index()
    if observed:
        expected: pd.DataFrame = DataFrame({'s1': Categorical(['a'], categories=['a', 'b', 'c']), 's2': [2]})
    else:
        expected: pd.DataFrame = DataFrame({'s1': Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c']), 's2': [2, np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func: str, expected_values: list', [('first', ['second', 'first']), ('last', ['fourth', 'third']), ('min', ['fourth', 'first']), ('max', ['second', 'third'])])
def test_preserve_on_ordered_ops(func: str, expected_values: list) -> None:
    c: pd.Categorical = Categorical(['first', 'second', 'third', 'fourth'], ordered=True)
    df: pd.DataFrame = DataFrame({'payload': [-1, -2, -1, -2], 'col': c})
    g: pd.DataFrameGroupBy = df.groupby('payload')
    result: pd.DataFrame = getattr(g, func)()
    expected: pd.DataFrame = DataFrame({'payload': [-2, -1], 'col': Series(expected_values, dtype=c.dtype)}).set_index('payload')
    tm.assert_frame_equal(result, expected)
    sgb: pd.SeriesGroupBy = df.groupby('payload')['col']
    result: pd.Series = getattr(sgb, func)()
    expected: pd.Series = expected['col']
    tm.assert_series_equal(result, expected)

def test_categorical_no_compress() -> None:
    data: pd.Series = Series(np.random.default_rng(2).standard_normal(9))
    codes: np.ndarray = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cats: pd.Categorical = Categorical.from_codes(codes, [0, 1, 2], ordered=True)
    result: pd.Series = data.groupby(cats, observed=False).mean()
    exp: pd.Series = data.groupby(codes, observed=False).mean()
    exp.index = CategoricalIndex(exp.index, categories=cats.categories, ordered=cats.ordered)
    tm.assert_series_equal(result, exp)
    codes: np.ndarray = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3])
    cats: pd.Categorical = Categorical.from_codes(codes, [0, 1, 2, 3], ordered=True)
    result: pd.Series = data.groupby(cats, observed=False).mean()
    exp: pd.Series = data.groupby(codes, observed=False).mean().reindex(cats.categories)
    exp.index = CategoricalIndex(exp.index, categories=cats.categories, ordered=cats.ordered)
    tm.assert_series_equal(result, exp)

def test_categorical_no_compress_string() -> None:
    cats: pd.Categorical = Categorical(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], categories=['a', 'b', 'c', 'd'], ordered=True)
    data: pd.DataFrame = DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 4, 5], 'b': cats})
    result: pd.Series = data.groupby('b', observed=False).mean()
    result: np.ndarray = result['a'].values
    exp: np.ndarray = np.array([1, 2, 4, np.nan])
    tm.assert_numpy_array_equal(result, exp)

def test_groupby_empty_with_category() -> None:
    df: pd.DataFrame = DataFrame({'A': [None] * 3, 'B': Categorical(['train', 'train', 'test'])})
    result: pd.Series = df.groupby('A').first()['B']
    expected: pd.Series = Series(Categorical([], categories=['test', 'train']), index=Series([], dtype='object', name='A'), name='B')
    tm.assert_series_equal(result, expected)

def test_sort() -> None:
    df: pd.DataFrame = DataFrame({'value': np.random.default_rng(2).integers(0, 10000, 10)})
    labels: list = [f'{i} - {i + 499}' for i in range(0, 10000, 500)]
    cat_labels: pd.Categorical = Categorical(labels, labels)
    df = df.sort_values(by=['value'], ascending=True)
    df['value_group'] = pd.cut(df.value, range(0, 10500, 500), right=False, labels=cat_labels)
    res: pd.Series = df.groupby(['value_group'], observed=False)['value_group'].count()
    exp: pd.Series = res[sorted(res.index, key=lambda x: float(x.split()[0]))]
    exp.index = CategoricalIndex(exp.index, name=exp.index.name)
    tm.assert_series_equal(res, exp)

@pytest.mark.parametrize('ordered: bool', [True, False])
def test_sort2(sort: bool, ordered: bool) -> None:
    df: pd.DataFrame = DataFrame([['(7.5, 10]', 10, 10], ['(7.5, 10]', 8, 20], ['(2.5, 5]', 5, 30], ['(5, 7.5]', 6, 40], ['(2.5, 5]', 4, 50], ['(0, 2.5]', 1, 60], ['(5, 7.5]', 7, 70]], columns=['range', 'foo', 'bar'])
    df['range'] = Categorical(df['range'], ordered=ordered)
    result: pd.DataFrame = df.groupby('range', sort=sort, observed=False).first()
    if sort:
        data_values: list = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values: list = ['(0, 2.5]', '(2.5, 5]', '(5, 7.5]', '(7.5, 10]']
    else:
        data_values: list = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values: list = ['(7.5, 10]', '(2.5, 5]', '(5, 7.5]', '(0, 2.5]']
    expected: pd.DataFrame = DataFrame(data_values, columns=['foo', 'bar'], index=CategoricalIndex(index_values, name='range', ordered=ordered))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('ordered: bool', [True, False])
def test_sort_datetimelike(sort: bool, ordered: bool) -> None:
    df: pd.DataFrame = DataFrame({'dt': [datetime(2011, 7, 1), datetime(2011, 7, 1), datetime(2011, 2, 1), datetime(2011, 5, 1), datetime(2011, 2, 1), datetime(2011, 1, 1), datetime(2011, 5, 1)], 'foo': [10, 8, 5, 6, 4, 1, 7], 'bar': [10, 20, 30, 40, 50, 60, 70]}, columns=['dt', 'foo', 'bar'])
    df['dt'] = Categorical(df['dt'], ordered=ordered)
    if sort:
        data_values: list = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values: list = [datetime(2011, 1, 1), datetime(2011, 2, 1), datetime(2011, 5, 1), datetime(2011, 7, 1)]
    else:
        data_values: list = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values: list = [datetime(2011, 7, 1), datetime(2011, 2, 1), datetime(2011, 5, 1), datetime(2011, 1, 1)]
    expected: pd.DataFrame = DataFrame(data_values, columns=['foo', 'bar'], index=CategoricalIndex(index_values, name='dt', ordered=ordered))
    result: pd.DataFrame = df.groupby('dt', sort=sort, observed=False).first()
    tm.assert_frame_equal(result, expected)

def test_empty_sum() -> None:
    df: pd.DataFrame = DataFrame({'A': Categorical(['a', 'a', 'b'], categories=['a', 'b', 'c']), 'B': [1, 2, 1]})
    expected_idx: pd.CategoricalIndex = CategoricalIndex(['a', 'b', 'c'], name='A')
    result: pd.Series = df.groupby('A', observed=False).B.sum()
    expected: pd.Series = Series([3, 1, 0], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result: pd.Series = df.groupby('A', observed=False).B.sum(min_count=0)
    expected: pd.Series = Series([3, 1, 0], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result: pd.Series = df.groupby('A', observed=False).B.sum(min_count=1)
    expected: pd.Series = Series([3, 1, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result: pd.Series = df.groupby('A', observed=False).B.sum(min_count=2)
    expected: pd.Series = Series([3, np.nan, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)

def test_empty_prod() -> None:
    df: pd.DataFrame = DataFrame({'A': Categorical(['a', 'a', 'b'], categories=['a', 'b', 'c']), 'B': [1, 2, 1]})
    expected_idx: pd.CategoricalIndex = CategoricalIndex(['a', 'b', 'c'], name='A')
    result: pd.Series = df.groupby('A', observed=False).B.prod()
    expected: pd.Series = Series([2, 1, 1], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result: pd.Series = df.groupby('A', observed=False).B.prod(min_count=0)
    expected: pd.Series = Series([2, 1, 1], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result: pd.Series = df.groupby('A', observed=False).B.prod(min_count=1)
    expected: pd.Series = Series([2, 1, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)

def test_groupby_multiindex_categorical_datetime() -> None:
    df: pd.DataFrame = DataFrame({'key1': Categorical(list('abcbabcba')), 'key2': Categorical(list(pd.date_range('2018-06-01 00', freq='1min', periods=3)) * 3, ordered=True), 'values': np.arange(9)})
    result: pd.DataFrame = df.groupby(['key1', 'key2'], observed=False).mean()
    idx: pd.MultiIndex = MultiIndex.from_product([Categorical(['a', 'b', 'c']), Categorical(pd.date_range('2018-06-01 00', freq='1min', periods=3))], names=['key1', 'key2'])
    expected: pd.DataFrame = DataFrame({'values': [0, 4, 8, 3, 4, 5, 6, np.nan, 2]}, index=idx)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('as_index: bool, expected: pd.Series', [(True, Series(index=MultiIndex.from_arrays([Series([1, 1, 2], dtype='category'), [1, 2, 2]], names=['a', 'b']), data=[1, 2, 3], name='x')), (False, DataFrame({'a': Series([1, 1, 2], dtype='category'), 'b': [1, 2, 2], 'x': [1, 2, 3]}))])
def test_groupby_agg_observed_true_single_column(as_index: bool, expected: pd.Series) -> None:
    df: pd.DataFrame = DataFrame({'a': Series([1, 1, 2], dtype='category'), 'b': [1, 2, 2], 'x': [1, 2, 3]})
    result: pd.DataFrame = df.groupby(['a', 'b'], as_index=as_index, observed=True)['x'].sum()
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('fill_value', [None, np.nan, pd.NaT])
def test_shift(fill_value: object) -> None:
    ct: pd.Categorical = Categorical(['a', 'b', 'c', 'd'], categories=['a', 'b', 'c', 'd'], ordered=False)
    expected: pd.Categorical = Categorical([None, 'a', 'b', 'c'], categories=['a', 'b', 'c', 'd'], ordered=False)
    res: pd.Categorical = ct.shift(1, fill_value=fill_value)
    tm.assert_equal(res, expected)

@pytest.fixture
def df_cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame with multiple categorical columns and a column of integers.
    Shortened so as not to contain all possible combinations of categories.
    Useful for testing `observed` kwarg functionality on GroupBy objects.

    Parameters
    ----------
    df: DataFrame
        Non-categorical, longer DataFrame from another fixture, used to derive
        this one

    Returns
    -------
    df_cat: DataFrame
    """
    df_cat: pd.DataFrame = df.copy()[:4]
    df_cat['A'] = df_cat['A'].astype('category')
    df_cat['B'] = df_cat['B'].astype('category')
    df_cat['C'] = Series([1, 2, 3, 4])
    df_cat = df_cat.drop(['D'], axis=1)
    return df_cat

@pytest.mark.parametrize('operation: str', ['agg', 'apply'])
def test_seriesgroupby_observed_true(df_cat: pd.DataFrame, operation: str) -> None:
    lev_a: pd.Index = Index(['bar', 'bar', 'foo', 'foo'], dtype=df_cat['A'].dtype, name='A')
    lev_b: pd.Index = Index(['one', 'three', 'one', 'two'], dtype=df_cat['B'].dtype, name='B')
    index: pd.MultiIndex = MultiIndex.from_arrays([lev_a, lev_b])
    expected: pd.Series = Series(data=[2, 4, 1, 3], index=index, name='C').sort_index()
    grouped: pd.SeriesGroupBy = df_cat.groupby(['A', 'B'], observed=True)['C']
    result: pd.Series = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('operation: str', ['agg', 'apply'])
@pytest.mark.parametrize('observed: bool', [False, None])
def test_seriesgroupby_observed_false_or_none(df_cat: pd.DataFrame, observed: bool, operation: str) -> None:
    index, _ = MultiIndex.from_product([CategoricalIndex(['bar', 'foo'], ordered=False), CategoricalIndex(['one', 'three', 'two'], ordered=False)], names=['A', 'B']).sortlevel()
    expected: pd.Series = Series(data=[2, 4, 0, 1, 0, 3], index=index, name='C')
    grouped: pd.SeriesGroupBy = df_cat.groupby(['A', 'B'], observed=observed)['C']
    result: pd.Series = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('observed: bool, index: pd.MultiIndex, data: list', [(True, MultiIndex.from_arrays([Index(['bar'] * 4 + ['foo'] * 4, dtype='category', name='A'), Index(['one', 'one', 'three', 'three', 'one', 'one', 'two', 'two'], dtype='category', name='B'), Index(['min', 'max'] * 4)], names=['A', 'B', None]), [2, 2, 4, 4, 1, 1, 3, 3]), (False, MultiIndex.from_product([CategoricalIndex(['bar', 'foo'], ordered=False), CategoricalIndex(['one', 'three', 'two'], ordered=False), Index(['min', 'max'])], names=['A', 'B', None]), [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3]), (None, MultiIndex.from_product([CategoricalIndex(['bar', 'foo'], ordered=False), CategoricalIndex(['one', 'three', 'two'], ordered=False), Index(['min', 'max'])], names=['A', 'B', None]), [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3])])
def test_seriesgroupby_observed_apply_dict(observed: bool, index: pd.MultiIndex, data: list) -> None:
    expected: pd.Series = Series(data=data, index=index, name='C')
    result: pd.Series = df_cat.groupby(['A', 'B'], observed=observed)['C'].apply(lambda x: {'min': x.min(), 'max': x.max()})
    tm.assert_series_equal(result, expected)

def test_groupby_categorical_series_dataframe_consistent(df_cat: pd.DataFrame) -> None:
    expected: pd.Series = df_cat.groupby(['A', 'B'], observed=False)['C'].mean()
    result: pd.Series = df_cat.groupby(['A', 'B'], observed=False).mean()['C']
    tm.assert_series_equal(result, expected)

def test_groupby_cat_preserves_structure(observed: bool, ordered: bool) -> None:
    df: pd.DataFrame = DataFrame({'Name': Categorical(['Bob', 'Greg'], ordered=ordered), 'Item': [1, 2]}, columns=['Name', 'Item'])
    expected: pd.DataFrame = df.copy()
    result: pd.DataFrame = df.groupby('Name', observed=observed).agg(DataFrame.sum, skipna=True).reset_index()
    tm.assert_frame_equal(result, expected)

def test_get_nonexistent_category() -> None:
    df: pd.DataFrame = DataFrame({'var': ['a', 'a', 'b', 'b'], 'val': range(4)})
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby('var').apply(lambda rows: DataFrame({'val': [rows.iloc[-1]['vau']}))

def test_series_groupby_on_2_categoricals_unobserved(reduction_func: str, observed: bool) -> None:
    if reduction_func == 'ngroup':
        pytest.skip('ngroup is not truly a reduction')
    df: pd.DataFrame = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABCD')), 'cat_2': Categorical(list('AB') * 2, categories=list('ABCD')), 'value': [0.1] * 4})
    args: tuple = get_groupby_method_args(reduction_func, df)
    expected_length: int = 4 if observed else 16
    series_groupby: pd.SeriesGroupBy = df.groupby(['cat_1', 'cat_2'], observed=observed)['value']
    if reduction_func == 'corrwith':
        assert not hasattr(series_groupby, reduction_func)
        return
    agg: callable = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            agg(*args)
        return
    result: pd.Series = agg(*args)
    assert len(result) == expected_length

def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(reduction_func: str, request: pytest.Request) -> None:
    if reduction_func == 'ngroup':
        pytest.skip('ngroup is not truly a reduction')
    if reduction_func == 'corrwith':
        mark: pytest.Mark = pytest.mark.xfail(reason='TODO: implemented SeriesGroupBy.corrwith. See GH 32293')
        request.applymarker(mark)
    df: pd.DataFrame = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABC')), 'cat_2': Categorical(list('AB') * 2, categories=list('ABC')), 'value': [0.1] * 4})
    unobserved: list = [tuple('AC'), tuple('BC'), tuple('CA'), tuple('CB'), tuple('CC')]
    args: tuple = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.SeriesGroupBy = df.groupby(['cat_1', 'cat_2'], observed=False)['value']
    agg: callable = getattr(series_groupby, reduction_func)
    if reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            agg(*args)
        return
    result: pd.Series = agg(*args)
    missing_fillin: object = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved:
        val: object = result.loc[idx]
        assert pd.isna(missing_fillin) and pd.isna(val) or val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ['count', 'nunique', 'size']:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ['sum', 'any']

def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(reduction_func: str) -> None:
    if reduction_func == 'ngroup':
        pytest.skip('ngroup does not return the Categories on the index')
    df: pd.DataFrame = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABC')), 'cat_2': Categorical(list('1111'), categories=list('12')), 'value': [0.1, 0.1, 0.1, 0.1]})
    unobserved_cats: list = [('A', '2'), ('B', '2'), ('C', '1'), ('C', '2')]
    df_grp: pd.DataFrameGroupBy = df.groupby(['cat_1', 'cat_2'], observed=True)
    args: tuple = get_groupby_method_args(reduction_func, df)
    if reduction_func == 'corrwith':
        warn: type = FutureWarning
        warn_msg: str = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn: type = None
        warn_msg: str = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: pd.DataFrame = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index

@pytest.mark.parametrize('observed: bool', [False, None])
def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(reduction_func: str, observed: bool) -> None:
    if reduction_func == 'ngroup':
        pytest.skip('ngroup does not return the Categories on the index')
    df: pd.DataFrame = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABC')), 'cat_2': Categorical(list('1111'), categories=list('12')), 'value': [0.1, 0.1, 0.1, 0.1]})
    unobserved_cats: list = [('A', '2'), ('B', '2'), ('C', '1'), ('C', '2')]
    df_grp: pd.DataFrameGroupBy = df.groupby(['cat_1', 'cat_2'], observed=observed)
    args: tuple = get_groupby_method_args(reduction_func, df)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            getattr(df_grp, reduction_func)(*args)
        return
    if reduction_func == 'corrwith':
        warn: type = FutureWarning
        warn_msg: str = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn: type = None
        warn_msg: str = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: pd.DataFrame = getattr(df_grp, reduction_func)(*args)
    expected: object = _results_for_groupbys_with_missing_categories[reduction_func]
    if expected is np.nan:
        assert res.loc[unobserved_cats].isnull().all().all()
    else:
        assert (res.loc[unobserved_cats] == expected).all().all()

def test_series_groupby_categorical_aggregation_getitem() -> None:
    d: dict = {'foo': [10, 8, 4, 1], 'bar': [10, 20, 30, 40], 'baz': ['d', 'c', 'd', 'c']}
    df: pd.DataFrame = DataFrame(d)
    cat: pd.Categorical = pd.cut(df['foo'], np.linspace(0, 20, 5))
    df['range'] = cat
    groups: pd.DataFrameGroupBy = df.groupby(['range', 'baz'], as_index=True, sort=True, observed=False)
    result: pd.Series = groups['foo'].agg('mean')
    expected: pd.Series = groups.agg('mean')['foo']
    tm.assert_series_equal(result, expected)

def test_groupby_agg_categorical_columns(func: callable, expected_values: list) -> None:
    df: pd.DataFrame = DataFrame({'id': [0, 1, 2, 3, 4], 'groups': [0, 1, 1, 2, 2], 'value': Categorical([0, 0, 0, 0, 1])}).set_index('id')
    result: pd.DataFrame = df.groupby('groups').agg(func)
    expected: pd.DataFrame = DataFrame({'value': expected_values}, index=Index([0, 1, 2], name='groups'))
    tm.assert_frame_equal(result, expected)

def test_groupby_agg_non_numeric() -> None:
    df: pd.DataFrame = DataFrame({'A': Categorical(['a', 'a', 'b'], categories=['a', 'b', 'c'])})
    expected: pd.DataFrame = DataFrame({'A': [2, 1]}, index=np.array([1, 2]))
    result: pd.DataFrame = df.groupby([1, 2, 1]).agg(Series.nunique)
    tm.assert_frame_equal(result, expected)
    result: pd.DataFrame = df.groupby([1, 2, 1]).nunique()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_groupby_first_returned_categorical_instead_of_dataframe(func: str) -> None:
    df: pd.DataFrame = DataFrame({'A': [1997], 'B': Series(['b'], dtype='category').cat.as_ordered()})
    df_grouped: pd.SeriesGroupBy = df.groupby('A')['B']
    result: pd.Series = getattr(df_grouped, func)()
    expected: pd.Series = Series(['b'], index=Index([1997], name='A'), name='B', dtype=df['B'].dtype)
    tm.assert_series_equal(result, expected)

def test_read_only_category_no_sort() -> None:
    cats: np.ndarray = np.array([1, 2])
    cats.flags.writeable = False
    df: pd.DataFrame = DataFrame({'a': [1, 3, 5, 7], 'b': Categorical([1, 1, 2, 2], categories=Index(cats))})
    expected: pd.DataFrame = DataFrame(data={'a': [2.0, 6.0]}, index=CategoricalIndex(cats, name='b'))
    result: pd.DataFrame = df.groupby('b', sort=False, observed=False).mean()
    tm.assert_frame_equal(result, expected)

def test_sorted_missing_category_values() -> None:
    df: pd.DataFrame = DataFrame({'foo': ['small', 'large', 'large', 'large', 'medium', 'large', 'large', 'medium'], 'bar': ['C', 'A', 'A', 'C', 'A', 'C', 'A', 'C']})
    df['foo'] = df['foo'].astype('category').cat.set_categories(['tiny', 'small', 'medium', 'large'], ordered=True)
    expected: pd.DataFrame = DataFrame({'tiny': {'A': 0, 'C': 0}, 'small': {'A': 0, 'C': 1}, 'medium': {'A': 1, 'C': 1}, 'large': {'A': 3, 'C': 2}})
    expected = expected.rename_axis('bar', axis='index')
    expected.columns = CategoricalIndex(['tiny', 'small', 'medium', 'large'], categories=['tiny', 'small', 'medium', 'large'], ordered=True, name='foo', dtype='category')
    result: pd.DataFrame = df.groupby(['bar', 'foo'], observed=False).size().unstack()
    tm.assert_frame_equal(result, expected)

def test_agg_cython_category_not_implemented_fallback() -> None:
    df: pd.DataFrame = DataFrame({'col_num': [1, 1, 2, 3]})
    df['col_cat'] = df['col_num'].astype('category')
    result: pd.Series = df.groupby('col_num').col_cat.first()
    expected: pd.Series = Series([1, 2, 3], index=Index([1, 2, 3], name='col_num'), name='col_cat', dtype=df['col_cat'].dtype)
    tm.assert_series_equal(result, expected)
    result: pd.DataFrame = df.groupby('col_num').agg({'col_cat': 'first'})
    expected: pd.DataFrame = expected.to_frame()
    tm.assert_frame_equal(result, expected)

def test_aggregate_categorical_with_isnan() -> None:
    df: pd.DataFrame = DataFrame({'A': [1, 1, 1, 1], 'B': [1, 2, 1, 2], 'numerical_col': [0.1, 0.2, np.nan, 0.3], 'object_col': ['foo', 'bar', 'foo', 'fee'], 'categorical_col': ['foo', 'bar', 'foo', 'fee']})
    df = df.astype({'categorical_col': 'category'})
    result: pd.DataFrame = df.groupby(['A', 'B']).agg(lambda df: df.isna().sum())
    index: pd.MultiIndex = MultiIndex.from_arrays([[1, 1], [1, 2]], names=('A', 'B'))
    expected: pd.DataFrame = DataFrame(data={'numerical_col': [1, 0], 'object_col': [0, 0], 'categorical_col': [0, 0]}, index=index)
    tm.assert_frame_equal(result, expected)

def test_categorical_transform() -> None:
    df: pd.DataFrame = DataFrame({'package_id': [1, 1, 1, 2, 2, 3], 'status': ['Waiting', 'OnTheWay', 'Delivered', 'Waiting', 'OnTheWay', 'Waiting']})
    delivery_status_type: pd.CategoricalDtype = pd.CategoricalDtype(categories=['Waiting', 'OnTheWay', 'Delivered'], ordered=True)
    df['status'] = df['status'].astype(delivery_status_type)
    df['last_status'] = df.groupby('package_id')['status'].transform(max)
    result: pd.DataFrame = df.copy()
    expected: pd.DataFrame = DataFrame({'package_id': [1, 1, 1, 2, 2, 3], 'status': ['Waiting', 'OnTheWay', 'Delivered', 'Waiting', 'OnTheWay', 'Waiting'], 'last_status': ['Waiting', 'Waiting', 'Waiting', 'Waiting', 'Waiting', 'Waiting']})
    expected['status'] = expected['status'].astype(delivery_status_type)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(func: str, observed: bool) -> None:
    cat: pd.Categorical = Categorical([0, 0, 1, 1])
    val: list = [0, 1, 1, 0]
    df: pd.DataFrame = DataFrame({'a': cat, 'b': cat, 'c': val})
    cat2: pd.Categorical = Categorical([0, 1])
    idx: pd.MultiIndex = MultiIndex.from_product([cat2, cat2], names=['a', 'b'])
    expected_dict: dict = {'first': Series([0, np.nan, np.nan, 1], idx, name='c'), 'last': Series([1, np.nan, np.nan, 0], idx, name='c')}
    expected: pd.Series = expected_dict[func]
    if observed:
        expected = expected.dropna().astype(np.int64)
    srs_grp: pd.SeriesGroupBy = df.groupby(['a', 'b'], observed=observed)['c']
    result: pd.Series = getattr(srs_grp, func)()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_df_groupby_first_on_categorical_col_grouped_on_2_categoricals(func: str, observed: bool) -> None:
    cat: pd.Categorical = Categorical([0, 0, 1, 1])
    val: list = [0, 1, 1, 0]
    df: pd.DataFrame = DataFrame({'a': cat, 'b': cat, 'c': val})
    cat2: pd.Categorical = Categorical([0, 1])
    idx: pd.MultiIndex = MultiIndex.from_product([cat2, cat2], names=['a', 'b'])
    expected_dict: dict = {'first': Series([0, np.nan, np.nan, 1], idx, name='c'), 'last': Series([1, np.nan, np.nan, 0], idx, name='c')}
    expected: pd.DataFrame = expected_dict[func].to_frame()
    if observed:
        expected = expected.dropna().astype(np.int64)
    df_grp: pd.DataFrameGroupBy = df.groupby(['a', 'b'], observed=observed)
    result: pd.DataFrame = getattr(df_grp, func)()
    tm.assert_frame_equal(result, expected)

def test_groupby_categorical_indices_unused_categories() -> None:
    df: pd.DataFrame = DataFrame({'key': Categorical(['b', 'b', 'a'], categories=['a', 'b', 'c']), 'col': range(3)})
    grouped: pd.DataFrameGroupBy = df.groupby('key', sort=False, observed=False)
    result: dict = grouped.indices
    expected: dict = {'b': np.array([0, 1], dtype='intp'), 'a': np.array([2], dtype='intp'), 'c': np.array([], dtype='intp')}
    assert result.keys() == expected.keys()
    for key in result.keys():
        tm.assert_numpy_array_equal(result[key], expected[key])

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_groupby_last_first_preserve_categoricaldtype(func: str) -> None:
    df: pd.DataFrame = DataFrame({'a': [1, 2, 3]})
    df['b'] = df['a'].astype('category')
    result: pd.Series = getattr(df.groupby('a')['b'], func)()
    expected: pd.Series = Series(Categorical([1, 2, 3]), name='b', index=Index([1, 2, 3], name='a'))
    tm.assert_series_equal(expected, result)

def test_groupby_categorical_observed_nunique() -> None:
    df: pd.DataFrame = DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [10, 11]})
    df = df.astype(dtype={'a': 'category', 'b': 'category'})
    result: pd.Series = df.groupby(['a', 'b'], observed=True).nunique()['c']
    expected: pd.Series = Series([1, 1], index=MultiIndex.from_arrays([CategoricalIndex([1, 2], name='a'), CategoricalIndex([1, 2], name='b')]), name='c')
    tm.assert_series_equal(result, expected)

def test_groupby_categorical_aggregate_functions() -> None:
    dtype: pd.CategoricalDtype = pd.CategoricalDtype(categories=['small', 'big'], ordered=True)
    df: pd.DataFrame = DataFrame([[1, 'small'], [1, 'big'], [2, 'small']], columns=['grp', 'description']).astype({'description': dtype})
    result: pd.Series = df.groupby('grp')['description'].max()
    expected: pd.Series = Series(['big', 'small'], index=Index([1, 2], name='grp'), name='description', dtype=dtype)
    tm.assert_series_equal(result, expected)

def test_groupby_categorical_dropna(observed: bool, dropna: bool) -> None:
    cat: pd.Categorical = Categorical([1, 2], categories=[1, 2, 3])
    df: pd.DataFrame = DataFrame({'x': Categorical([1, 2], categories=[1, 2, 3]), 'y': [3, 4]})
    gb: pd.DataFrameGroupBy = df.groupby('x', observed=observed, dropna=dropna)
    result: pd.DataFrame = gb.sum()
    if observed:
        expected: pd.DataFrame = DataFrame({'y': [3, 4]}, index=cat)
    else:
        index: pd.CategoricalIndex = CategoricalIndex([1, 2, 3], [1, 2, 3])
        expected: pd.DataFrame = DataFrame({'y': [3, 4, 0]}, index=index)
    expected.index.name = 'x'
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('index_kind: str', ['range', 'single', 'multi'])
@pytest.mark.parametrize('ordered: bool', [True, False])
def test_category_order_reducer(request: pytest.Request, as_index: bool, sort: bool, observed: bool, reduction_func: str, index_kind: str, ordered: bool) -> None:
    if reduction_func == 'corrwith' and (not as_index) and (index_kind != 'single'):
        msg: str = 'GH#49950 - corrwith with as_index=False may not have grouping column'
        request.applymarker(pytest.mark.xfail(reason=msg))
    elif index_kind != 'range' and (not as_index):
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    df: pd.DataFrame = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered), 'b': range(4)})
    if index_kind == 'range':
        keys: list = ['a']
    elif index_kind == 'single':
        keys: list = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys: list = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    args: tuple = get_groupby_method_args(reduction_func, df)
    gb: pd.DataFrameGroupBy = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            getattr(gb, reduction_func)(*args)
        return
    if reduction_func == 'corrwith':
        warn: type = FutureWarning
        warn_msg: str = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn: type = None
        warn_msg: str = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        op_result: pd.DataFrame = getattr(gb, reduction_func)(*args)
    if as_index:
        result: pd.Index = op_result.index.get_level_values('a').categories
    else:
        result: pd.Index = op_result['a'].cat.categories
    expected: pd.Index = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result: pd.Index = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('index_kind: str', ['single', 'multi'])
@pytest.mark.parametrize('ordered: bool', [True, False])
def test_category_order_transformer(as_index: bool, sort: bool, observed: bool, transformation_func: str, index_kind: str, ordered: bool) -> None:
    df: pd.DataFrame = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered), 'b': range(4)})
    if index_kind == 'single':
        keys: list = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys: list = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    args: tuple = get_groupby_method_args(transformation_func, df)
    gb: pd.DataFrameGroupBy = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result: pd.DataFrame = getattr(gb, transformation_func)(*args)
    result: pd.Index = op_result.index.get_level_values('a').categories
    expected: pd.Index = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result: pd.Index = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('index_kind: str', ['range', 'single', 'multi'])
@pytest.mark.parametrize('method: str', ['head', 'tail'])
@pytest.mark.parametrize('ordered: bool', [True, False])
def test_category_order_head_tail(as_index: bool, sort: bool, observed: bool, method: str, index_kind: str, ordered: bool) -> None:
    df: pd.DataFrame = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered), 'b': range(4)})
    if index_kind == 'range':
        keys: list = ['a']
    elif index_kind == 'single':
        keys: list = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys: list = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    gb: pd.DataFrameGroupBy = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result: pd.DataFrame = getattr(gb, method)()
    if index_kind == 'range':
        result: pd.Index = op_result['a'].cat.categories
    else:
        result: pd.Index = op_result.index.get_level_values('a').categories
    expected: pd.Index = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result: pd.Index = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('index_kind: str', ['range', 'single', 'multi'])
@pytest.mark.parametrize('method: str', ['apply', 'agg', 'transform'])
@pytest.mark.parametrize('ordered: bool', [True, False])
def test_category_order_apply(as_index: bool, sort: bool, observed: bool, method: str, index_kind: str, ordered: bool) -> None:
    if method == 'transform' and index_kind == 'range' or (not as_index and index_kind != 'range'):
        pytest.skip('No categories in result, nothing to test')
    df: pd.DataFrame = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered), 'b': range(4)})
    if index_kind == 'range':
        keys: list = ['a']
    elif index_kind == 'single':
        keys: list = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys: list = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    gb: pd.DataFrameGroupBy = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result: pd.DataFrame = getattr(gb, method)(lambda x: x.sum(numeric_only=True))
    if (method == 'transform' or not as_index) and index_kind == 'range':
        result: pd.Index = op_result['a'].cat.categories
    else:
        result: pd.Index = op_result.index.get_level_values('a').categories
    expected: pd.Index = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result: pd.Index = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('index_kind: str', ['range', 'single', 'multi'])
def test_many_categories(as_index: bool, sort: bool, index_kind: str, ordered: bool) -> None:
    if index_kind != 'range' and (not as_index):
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    categories: np.ndarray = np.arange(9999, -1, -1)
    grouper: pd.Categorical = Categorical([2, 1, 2, 3], categories=categories, ordered=ordered)
    df: pd.DataFrame = DataFrame({'a': grouper, 'b': range(4)})
    if index_kind == 'range':
        keys: list = ['a']
    elif index_kind == 'single':
        keys: list = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys: list = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    gb: pd.DataFrameGroupBy = df.groupby(keys, as_index=as_index, sort=sort, observed=True)
    result: pd.DataFrame = gb.sum()
    data: list = [3, 2, 1] if sort else [2, 1, 3]
    index: pd.CategoricalIndex = CategoricalIndex(data, categories=grouper.categories, ordered=ordered, name='a')
    if as_index:
        expected: pd.DataFrame = DataFrame({'b': data})
        if index_kind == 'multi':
            expected.index = MultiIndex.from_frame(DataFrame({'a': index, 'a2': index}))
        else:
            expected.index = index
    elif index_kind == 'multi':
        expected: pd.DataFrame = DataFrame({'a': Series(index), 'a2': Series(index), 'b': data})
    else:
        expected: pd.DataFrame = DataFrame({'a': Series(index), 'b': data})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('test_series: bool', [True, False])
@pytest.mark.parametrize('keys: list', [['a1'], ['a1', 'a2']])
def test_agg_list(request: pytest.Request, as_index: bool, observed: bool, reduction_func: str, test_series: bool, keys: list) -> None:
    if test_series and reduction_func == 'corrwith':
        assert not hasattr(SeriesGroupBy, 'corrwith')
        pytest.skip('corrwith not implemented for SeriesGroupBy')
    elif reduction_func == 'corrwith':
        msg: str = 'GH#32293: attempts to call SeriesGroupBy.corrwith'
        request.applymarker(pytest.mark.xfail(reason=msg))
    df: pd.DataFrame = DataFrame({'a1': [0, 0, 1], 'a2': [2, 3, 3], 'b': [4, 5, 6]})
    df = df.astype({'a1': 'category', 'a2': 'category'})
    if 'a2' not in keys:
        df = df.drop(columns='a2')
    gb: pd.DataFrameGroupBy = df.groupby(by=keys, as_index=as_index, observed=observed)
    if test_series:
        gb = gb['b']
    args: tuple = get_groupby_method_args(reduction_func, df)
    if not observed and reduction_func in ['idxmin', 'idxmax'] and (keys == ['a1', 'a2']):
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            gb.agg([reduction_func], *args)
        return
    result: pd.DataFrame = gb.agg([reduction_func], *args)
    expected: pd.DataFrame = getattr(gb, reduction_func)(*args)
    if as_index and (test_series or reduction_func == 'size'):
        expected = expected.to_frame(reduction_func)
    if not test_series:
        expected.columns = MultiIndex.from_tuples([(ind, '') for ind in expected.columns[:-1]] + [('b', reduction_func)])
    elif not as_index:
        expected.columns = keys + [reduction_func]
    tm.assert_equal(result, expected)

def test_groupby_categorical_series_dataframe_consistent(df_cat: pd.DataFrame) -> None:
    expected: pd.Series = df_cat.groupby(['A', 'B'], observed=False)['C'].mean()
    result: pd.Series = df_cat.groupby(['A', 'B'], observed=False).mean()['C']
    tm.assert_series_equal(result, expected)

def test_groupby_cat_preserves_structure(observed: bool, ordered: bool) -> None:
    df: pd.DataFrame = DataFrame({'Name': Categorical(['Bob', 'Greg'], ordered=ordered), 'Item': [1, 2]}, columns=['Name', 'Item'])
    expected: pd.DataFrame = df.copy()
    result: pd.DataFrame = df.groupby('Name', observed=observed).agg(DataFrame.sum, skipna=True).reset_index()
    tm.assert_frame_equal(result, expected)

def test_get_nonexistent_category() -> None:
    df: pd.DataFrame = DataFrame({'var': ['a', 'a', 'b', 'b'], 'val': range(4)})
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby('var').apply(lambda rows: DataFrame({'val': [rows.iloc[-1]['vau']}))

def test_series_groupby_on_2_categoricals_unobserved(reduction_func: str) -> None:
    if reduction_func == 'ngroup':
        pytest.skip('ngroup is not truly a reduction')
    df: pd.DataFrame = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABCD')), 'cat_2': Categorical(list('AB') * 2, categories=list('ABCD')), 'value': [0.1] * 4})
    args: tuple = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.SeriesGroupBy = df.groupby(['cat_1', 'cat_2'], observed=False)['value']
    result: pd.Series = series_groupby.agg(reduction_func)(*args)
    expected_length: int = 16
    assert len(result) == expected_length

def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(reduction_func: str) -> None:
    if reduction_func == 'ngroup':
        pytest.skip('ngroup does not return the Categories on the index')
    df: pd.DataFrame = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABC')), 'cat_2': Categorical(list('1111'), categories=list('12')), 'value': [0.1, 0.1, 0.1, 0.1]})
    unobserved_cats: list = [('A', '2'), ('B', '2'), ('C', '1'), ('C', '2')]
    df_grp: pd.DataFrameGroupBy = df.groupby(['cat_1', 'cat_2'], observed=True)
    args: tuple = get_groupby_method_args(reduction_func, df)
    if reduction_func == 'corrwith':
        warn: type = FutureWarning
        warn_msg: str = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn: type = None
        warn_msg: str = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: pd.DataFrame = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index

@pytest.mark.parametrize('observed: bool', [False, None])
def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(reduction_func: str, observed: bool) -> None:
    if reduction_func == 'ngroup':
        pytest.skip('ngroup does not return the Categories on the index')
    df: pd.DataFrame = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABC')), 'cat_2': Categorical(list('1111'), categories=list('12')), 'value': [0.1, 0.1, 0.1, 0.1]})
    unobserved_cats: list = [('A', '2'), ('B', '2'), ('C', '1'), ('C', '2')]
    df_grp: pd.DataFrameGroupBy = df.groupby(['cat_1', 'cat_2'], observed=observed)
    args: tuple = get_groupby_method_args(reduction_func, df)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            getattr(df_grp, reduction_func)(*args)
        return
    if reduction_func == 'corrwith':
        warn: type = FutureWarning
        warn_msg: str = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn: type = None
        warn_msg: str = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: pd.DataFrame = getattr(df_grp, reduction_func)(*args)
    expected: object = _results_for_groupbys_with_missing_categories[reduction_func]
    if expected is np.nan:
        assert res.loc[unobserved_cats].isnull().all().all()
    else:
        assert (res.loc[unobserved_cats] == expected).all().all()

def test_series_groupby_categorical_aggregation_getitem() -> None:
    d: dict = {'foo': [10, 8, 4, 1], 'bar': [10, 20, 30, 40], 'baz': ['d', 'c', 'd', 'c']}
    df: pd.DataFrame = DataFrame(d)
    cat: pd.Categorical = pd.cut(df['foo'], np.linspace(0, 20, 5))
    df['range'] = cat
    groups: pd.DataFrameGroupBy = df.groupby(['range', 'baz'], as_index=True, sort=True, observed=False)
    result: pd.Series = groups['foo'].agg('mean')
    expected: pd.Series = groups.agg('mean')['foo']
    tm.assert_series_equal(result, expected)

def test_groupby_agg_categorical_columns(func: callable, expected_values: list) -> None:
    df: pd.DataFrame = DataFrame({'id': [0, 1, 2, 3, 4], 'groups': [0, 1, 1, 2, 2], 'value': Categorical([0, 0, 0, 0, 1])}).set_index('id')
    result: pd.DataFrame = df.groupby('groups').agg(func)
    expected: pd.DataFrame = DataFrame({'value': expected_values}, index=Index([0, 1, 2], name='groups'))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_groupby_first_returned_categorical_instead_of_dataframe(func: str) -> None:
    df: pd.DataFrame = DataFrame({'A': [1997], 'B': Series(['b'], dtype='category').cat.as_ordered()})
    df_grouped: pd.SeriesGroupBy = df.groupby('A')['B']
    result: pd.Series = getattr(df_grouped, func)()
    expected: pd.Series = Series(['b'], index=Index([1997], name='A'), name='B', dtype=df['B'].dtype)
    tm.assert_series_equal(result, expected)

def test_read_only_category_no_sort() -> None:
    cats: np.ndarray = np.array([1, 2])
    cats.flags.writeable = False
    df: pd.DataFrame = DataFrame({'a': [1, 3, 5, 7], 'b': Categorical([1, 1, 2, 2], categories=Index(cats))})
    expected: pd.DataFrame = DataFrame(data={'a': [2.0, 6.0]}, index=CategoricalIndex(cats, name='b'))
    result: pd.DataFrame = df.groupby('b', sort=False, observed=False).mean()
    tm.assert_frame_equal(result, expected)

def test_sorted_missing_category_values() -> None:
    df: pd.DataFrame = DataFrame({'foo': ['small', 'large', 'large', 'large', 'medium', 'large', 'large', 'medium'], 'bar': ['C', 'A', 'A', 'C', 'A', 'C', 'A', 'C']})
    df['foo'] = df['foo'].astype('category').cat.set_categories(['tiny', 'small', 'medium', 'large'], ordered=True)
    expected: pd.DataFrame = DataFrame({'tiny': {'A': 0, 'C': 0}, 'small': {'A': 0, 'C': 1}, 'medium': {'A': 1, 'C': 1}, 'large': {'A': 3, 'C': 2}})
    expected = expected.rename_axis('bar', axis='index')
    expected.columns = CategoricalIndex(['tiny', 'small', 'medium', 'large'], categories=['tiny', 'small', 'medium', 'large'], ordered=True, name='foo', dtype='category')
    result: pd.DataFrame = df.groupby(['bar', 'foo'], observed=False).size().unstack()
    tm.assert_frame_equal(result, expected)

def test_agg_cython_category_not_implemented_fallback() -> None:
    df: pd.DataFrame = DataFrame({'col_num': [1, 1, 2, 3]})
    df['col_cat'] = df['col_num'].astype('category')
    result: pd.Series = df.groupby('col_num').col_cat.first()
    expected: pd.Series = Series([1, 2, 3], index=Index([1, 2, 3], name='col_num'), name='col_cat', dtype=df['col_cat'].dtype)
    tm.assert_series_equal(result, expected)
    result: pd.DataFrame = df.groupby('col_num').agg({'col_cat': 'first'})
    expected: pd.DataFrame = expected.to_frame()
    tm.assert_frame_equal(result, expected)

def test_aggregate_categorical_with_isnan() -> None:
    df: pd.DataFrame = DataFrame({'A': [1, 1, 1, 1], 'B': [1, 2, 1, 2], 'numerical_col': [0.1, 0.2, np.nan, 0.3], 'object_col': ['foo', 'bar', 'foo', 'fee'], 'categorical_col': ['foo', 'bar', 'foo', 'fee']})
    df = df.astype({'categorical_col': 'category'})
    result: pd.DataFrame = df.groupby(['A', 'B']).agg(lambda df: df.isna().sum())
    index: pd.MultiIndex = MultiIndex.from_arrays([[1, 1], [1, 2]], names=('A', 'B'))
    expected: pd.DataFrame = DataFrame(data={'numerical_col': [1, 0], 'object_col': [0, 0], 'categorical_col': [0, 0]}, index=index)
    tm.assert_frame_equal(result, expected)

def test_categorical_transform() -> None:
    df: pd.DataFrame = DataFrame({'package_id': [1, 1, 1, 2, 2, 3], 'status': ['Waiting', 'OnTheWay', 'Delivered', 'Waiting', 'OnTheWay', 'Waiting']})
    delivery_status_type: pd.CategoricalDtype = pd.CategoricalDtype(categories=['Waiting', 'OnTheWay', 'Delivered'], ordered=True)
    df['status'] = df['status'].astype(delivery_status_type)
    df['last_status'] = df.groupby('package_id')['status'].transform(max)
    result: pd.DataFrame = df.copy()
    expected: pd.DataFrame = DataFrame({'package_id': [1, 1, 1, 2, 2, 3], 'status': ['Waiting', 'OnTheWay', 'Delivered', 'Waiting', 'OnTheWay', 'Waiting'], 'last_status': ['Waiting', 'Waiting', 'Waiting', 'Waiting', 'Waiting', 'Waiting']})
    expected['status'] = expected['status'].astype(delivery_status_type)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(func: str, observed: bool) -> None:
    cat: pd.Categorical = Categorical([0, 0, 1, 1])
    val: list = [0, 1, 1, 0]
    df: pd.DataFrame = DataFrame({'a': cat, 'b': cat, 'c': val})
    cat2: pd.Categorical = Categorical([0, 1])
    idx: pd.MultiIndex = MultiIndex.from_product([cat2, cat2], names=['a', 'b'])
    expected_dict: dict = {'first': Series([0, np.nan, np.nan, 1], idx, name='c'), 'last': Series([1, np.nan, np.nan, 0], idx, name='c')}
    expected: pd.Series = expected_dict[func]
    if observed:
        expected = expected.dropna().astype(np.int64)
    srs_grp: pd.SeriesGroupBy = df.groupby(['a', 'b'], observed=observed)['c']
    result: pd.Series = getattr(srs_grp, func)()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_df_groupby_first_on_categorical_col_grouped_on_2_categoricals(func: str, observed: bool) -> None:
    cat: pd.Categorical = Categorical([0, 0, 1, 1])
    val: list = [0, 1, 1, 0]
    df: pd.DataFrame = DataFrame({'a': cat, 'b': cat, 'c': val})
    cat2: pd.Categorical = Categorical([0, 1])
    idx: pd.MultiIndex = MultiIndex.from_product([cat2, cat2], names=['a', 'b'])
    expected_dict: dict = {'first': Series([0, np.nan, np.nan, 1], idx, name='c'), 'last': Series([1, np.nan, np.nan, 0], idx, name='c')}
    expected: pd.DataFrame = expected_dict[func].to_frame()
    if observed:
        expected = expected.dropna().astype(np.int64)
    df_grp: pd.DataFrameGroupBy = df.groupby(['a', 'b'], observed=observed)
    result: pd.DataFrame = getattr(df_grp, func)()
    tm.assert_frame_equal(result, expected)

def test_groupby_categorical_indices_unused_categories() -> None:
    df: pd.DataFrame = DataFrame({'key': Categorical(['b', 'b', 'a'], categories=['a', 'b', 'c']), 'col': range(3)})
    grouped: pd.DataFrameGroupBy = df.groupby('key', sort=False, observed=False)
    result: dict = grouped.indices
    expected: dict = {'b': np.array([0, 1], dtype='intp'), 'a': np.array([2], dtype='intp'), 'c': np.array([], dtype='intp')}
    assert result.keys() == expected.keys()
    for key in result.keys():
        tm.assert_numpy_array_equal(result[key], expected[key])

@pytest.mark.parametrize('func: str', ['first', 'last'])
def test_groupby_last_first_preserve_categoricaldtype(func: str) -> None:
    df: pd.DataFrame = DataFrame({'a': [1, 2, 3]})
    df['b'] = df['a'].astype('category')
    result: pd.Series = getattr(df.groupby('a')['b'], func)()
    expected: pd.Series = Series(Categorical([1, 2, 3]), name='b', index=Index([1, 2, 3], name='a'))
    tm.assert_series_equal(expected, result)

def test_groupby_categorical_observed_nunique() -> None:
    df: pd.DataFrame = DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [10, 11]})
    df = df.astype(dtype={'a': 'category', 'b': 'category'})
    result: pd.Series = df.groupby(['a', 'b'], observed=True).nunique()['c']
    expected: pd.Series = Series([1, 1], index=MultiIndex.from_arrays([CategoricalIndex([1, 2], name='a'), CategoricalIndex([1, 2], name='b')]), name='c')
    tm.assert_series_equal(result, expected)

def test_groupby_categorical_aggregate_functions() -> None:
    dtype: pd.CategoricalDtype = pd.CategoricalDtype(categories=['small', 'big'], ordered=True)
    df: pd.DataFrame = DataFrame([[1, 'small'], [1, 'big'], [2, 'small']], columns=['grp', 'description']).astype({'description': dtype})
    result: pd.Series = df.groupby('grp')['description'].max()
    expected: pd.Series = Series(['big', 'small'], index=Index([1, 2], name='grp'), name='description', dtype=dtype)
    tm.assert_series_equal(result, expected)

def test_groupby_categorical_dropna(observed: bool, dropna: bool) -> None:
    cat: pd.Categorical = Categorical([1, 2], categories=[1, 2, 3])
    df: pd.DataFrame = DataFrame({'x': Categorical([1, 2], categories=[1, 2, 3]), 'y': [3, 4]})
    gb: pd.DataFrameGroupBy = df.groupby('x', observed=observed, dropna=dropna)
    result: pd.DataFrame = gb.sum()
    if observed:
        expected: pd.DataFrame = DataFrame({'y': [3, 4]}, index=cat)
    else:
        index: pd.CategoricalIndex = CategoricalIndex([1, 2, 3], [1, 2, 3])
        expected: pd.DataFrame = DataFrame({'y': [3, 4, 0]}, index=index)
    expected.index.name = 'x'
    tm.assert_frame_equal(result, expected)
