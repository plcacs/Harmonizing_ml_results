from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, CategoricalIndex, DataFrame, Index, MultiIndex, Series, qcut
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args


def func_ln7ovfd6(result, args, names, fill_value=np.nan):
    """Reindex to a cartesian production for the groupers,
    preserving the nature (Categorical) of each grouper
    """

    def func_pffbycyp(a):
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(np.arange(len(categories)),
                categories=categories, ordered=a.ordered)
        return a
    index = MultiIndex.from_product(map(f, args), names=names)
    return result.reindex(index, fill_value=fill_value).sort_index()


_results_for_groupbys_with_missing_categories = {'all': True, 'any': False,
    'count': 0, 'corrwith': np.nan, 'first': np.nan, 'idxmax': np.nan,
    'idxmin': np.nan, 'last': np.nan, 'max': np.nan, 'mean': np.nan,
    'median': np.nan, 'min': np.nan, 'nth': np.nan, 'nunique': 0, 'prod': 1,
    'quantile': np.nan, 'sem': np.nan, 'size': 0, 'skew': np.nan, 'kurt':
    np.nan, 'std': np.nan, 'sum': 0, 'var': np.nan}


def func_i73gteal(df):
    cats = qcut(df.C, 4)

    def func_51n4zqfi(group):
        return {'min': group.min(), 'max': group.max(), 'count': group.
            count(), 'mean': group.mean()}
    result = df.groupby(cats, observed=False).D.apply(get_stats)
    assert result.index.names[0] == 'C'


def func_ovvpmi8n():
    cats = Categorical(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        categories=['a', 'b', 'c', 'd'], ordered=True)
    data = DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 4, 5], 'b': cats})
    exp_index = CategoricalIndex(list('abcd'), name='b', ordered=True)
    expected = DataFrame({'a': [1, 2, 4, np.nan]}, index=exp_index)
    result = data.groupby('b', observed=False).mean()
    tm.assert_frame_equal(result, expected)


def func_g378453b():
    cat1 = Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'z'],
        ordered=True)
    cat2 = Categorical(['c', 'd', 'c', 'd'], categories=['c', 'd', 'y'],
        ordered=True)
    df = DataFrame({'A': cat1, 'B': cat2, 'values': [1, 2, 3, 4]})
    gb = df.groupby('A', observed=False)
    exp_idx = CategoricalIndex(['a', 'b', 'z'], name='A', ordered=True)
    expected = DataFrame({'values': Series([3, 7, 0], index=exp_idx)})
    result = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)


def func_c09iauss(using_infer_string):
    x = DataFrame([[1, 'John P. Doe'], [2, 'Jane Dove'], [1, 'John P. Doe']
        ], columns=['person_id', 'person_name'])
    x['person_name'] = Categorical(x.person_name)
    g = x.groupby(['person_id'], observed=False)
    result = g.transform(lambda x: x)
    tm.assert_frame_equal(result, x[['person_name']])
    result = x.drop_duplicates('person_name')
    expected = x.iloc[[0, 1]]
    tm.assert_frame_equal(result, expected)

    def func_pffbycyp(x):
        return x.drop_duplicates('person_name').iloc[0]
    result = g.apply(f)
    expected = x[['person_name']].iloc[[0, 1]]
    expected.index = Index([1, 2], name='person_id')
    dtype = 'str' if using_infer_string else object
    expected['person_name'] = expected['person_name'].astype(dtype)
    tm.assert_frame_equal(result, expected)


def func_tux6t0qv():
    df = DataFrame({'a': [5, 15, 25]})
    c = pd.cut(df.a, bins=[0, 10, 20, 30, 40])
    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df['a'])
    tm.assert_series_equal(df.a.groupby(c, observed=False).transform(lambda
        xs: np.sum(xs)), df['a'])
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[['a']]
    tm.assert_frame_equal(result, expected)
    gbc = df.groupby(c, observed=False)
    result = gbc.transform(lambda xs: np.max(xs, axis=0))
    tm.assert_frame_equal(result, df[['a']])
    result2 = gbc.transform(lambda xs: np.max(xs, axis=0))
    result3 = gbc.transform(max)
    result4 = gbc.transform(np.maximum.reduce)
    result5 = gbc.transform(lambda xs: np.maximum.reduce(xs))
    tm.assert_frame_equal(result2, df[['a']], check_dtype=False)
    tm.assert_frame_equal(result3, df[['a']], check_dtype=False)
    tm.assert_frame_equal(result4, df[['a']])
    tm.assert_frame_equal(result5, df[['a']])
    tm.assert_series_equal(df.a.groupby(c, observed=False).filter(np.all),
        df['a'])
    tm.assert_frame_equal(df.groupby(c, observed=False).filter(np.all), df)


def func_idpot71j():
    df = DataFrame({'a': [5, 15, 25, -5]})
    c = pd.cut(df.a, bins=[-10, 0, 10, 20, 30, 40])
    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df['a'])
    tm.assert_series_equal(df.a.groupby(c, observed=False).transform(lambda
        xs: np.sum(xs)), df['a'])
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[['a']]
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df.groupby(c, observed=False).transform(lambda xs:
        np.sum(xs)), df[['a']])


def func_me4cji5z():
    df = DataFrame({'a': [1, 0, 0, 0]})
    c = pd.cut(df.a, [0, 1, 2, 3, 4], labels=Categorical(list('abcd')))
    result = df.groupby(c, observed=False).apply(len)
    exp_index = CategoricalIndex(c.values.categories, ordered=c.values.ordered)
    expected = Series([1, 0, 0, 0], index=exp_index)
    expected.index.name = 'a'
    tm.assert_series_equal(result, expected)


def func_5mab5qsb():
    levels = ['foo', 'bar', 'baz', 'qux']
    codes = np.random.default_rng(2).integers(0, 4, size=10)
    cats = Categorical.from_codes(codes, levels, ordered=True)
    data = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result = data.groupby(cats, observed=False).mean()
    expected = data.groupby(np.asarray(cats), observed=False).mean()
    exp_idx = CategoricalIndex(levels, categories=cats.categories, ordered=True
        )
    expected = expected.reindex(exp_idx)
    tm.assert_frame_equal(result, expected)
    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()
    idx = cats.codes.argsort()
    ord_labels = np.asarray(cats).take(idx)
    ord_data = data.take(idx)
    exp_cats = Categorical(ord_labels, ordered=True, categories=['foo',
        'bar', 'baz', 'qux'])
    expected = ord_data.groupby(exp_cats, sort=False, observed=False).describe(
        )
    tm.assert_frame_equal(desc_result, expected)
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp = CategoricalIndex(expc)
    tm.assert_index_equal(desc_result.stack().index.get_level_values(0), exp)
    exp = Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] * 4
        )
    tm.assert_index_equal(desc_result.stack().index.get_level_values(1), exp)


def func_u7ybe940(observed):
    df = DataFrame(data=np.arange(2, 22, 2), index=MultiIndex(levels=[
        CategoricalIndex(['a', 'b']), range(10)], codes=[[0] * 5 + [1] * 5,
        range(10)], names=['Index1', 'Index2']))
    g = df.groupby(level=['Index1'], observed=observed)
    expected = DataFrame(data=np.arange(2, 12, 2), index=MultiIndex(levels=
        [CategoricalIndex(['a', 'b']), range(5)], codes=[[0] * 5, range(5)],
        names=['Index1', 'Index2']))
    result = g.get_group(('a',))
    tm.assert_frame_equal(result, expected)


def func_slqzr72t():
    df = DataFrame({'group': ['A'] * 6 + ['B'] * 6, 'dose': ['high', 'med',
        'low'] * 4, 'outcomes': np.arange(12.0)})
    df.dose = Categorical(df.dose, categories=['low', 'med', 'high'],
        ordered=True)
    result = df.groupby('group')['dose'].value_counts()
    result = result.sort_index(level=0, sort_remaining=True)
    index = ['low', 'med', 'high', 'low', 'med', 'high']
    index = Categorical(index, categories=['low', 'med', 'high'], ordered=True)
    index = [['A', 'A', 'A', 'B', 'B', 'B'], CategoricalIndex(index)]
    index = MultiIndex.from_arrays(index, names=['group', 'dose'])
    expected = Series([2] * 6, index=index, name='count')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('ordered', [True, False])
def func_o6uu9lw4(ordered):
    dense = Categorical(list('abc'), ordered=ordered)
    missing = Categorical(list('aaa'), categories=['a', 'b'], ordered=ordered)
    values = np.arange(len(dense))
    df = DataFrame({'missing': missing, 'dense': dense, 'values': values})
    grouped = df.groupby(['missing', 'dense'], observed=True)
    idx = MultiIndex.from_arrays([missing, dense], names=['missing', 'dense'])
    expected = DataFrame([0, 1, 2.0], index=idx, columns=['values'])
    result = grouped.apply(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)
    result = grouped.mean()
    tm.assert_frame_equal(result, expected)
    result = grouped.agg(np.mean)
    tm.assert_frame_equal(result, expected)
    idx = MultiIndex.from_arrays([missing, dense], names=['missing', 'dense'])
    expected = Series(1, index=idx)
    result = grouped.apply(lambda x: 1)
    tm.assert_series_equal(result, expected)


def func_4pkh1rse(request, using_infer_string, observed):
    if using_infer_string and not observed:
        request.applymarker(pytest.mark.xfail(reason='TODO(infer_string)'))
    cat1 = Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'z'],
        ordered=True)
    cat2 = Categorical(['c', 'd', 'c', 'd'], categories=['c', 'd', 'y'],
        ordered=True)
    df = DataFrame({'A': cat1, 'B': cat2, 'values': [1, 2, 3, 4]})
    df['C'] = ['foo', 'bar'] * 2
    gb = df.groupby(['A', 'B', 'C'], observed=observed)
    exp_index = MultiIndex.from_arrays([cat1, cat2, ['foo', 'bar'] * 2],
        names=['A', 'B', 'C'])
    expected = DataFrame({'values': Series([1, 2, 3, 4], index=exp_index)}
        ).sort_index()
    result = gb.sum()
    if not observed:
        expected = func_ln7ovfd6(expected, [cat1, cat2, ['foo', 'bar']],
            list('ABC'), fill_value=0)
    tm.assert_frame_equal(result, expected)
    gb = df.groupby(['A', 'B'], observed=observed)
    exp_index = MultiIndex.from_arrays([cat1, cat2], names=['A', 'B'])
    expected = DataFrame({'values': [1, 2, 3, 4], 'C': ['foo', 'bar', 'foo',
        'bar']}, index=exp_index)
    result = gb.sum()
    if not observed:
        expected = func_ln7ovfd6(expected, [cat1, cat2], list('AB'),
            fill_value=0)
    tm.assert_frame_equal(result, expected)


def func_d239nsot(observed):
    d = {'cat': Categorical(['a', 'b', 'a', 'b'], categories=['a', 'b', 'c'
        ], ordered=True), 'ints': [1, 1, 2, 2], 'val': [10, 20, 30, 40]}
    df = DataFrame(d)
    groups_single_key = df.groupby('cat', observed=observed)
    result = groups_single_key.mean()
    exp_index = CategoricalIndex(list('ab'), name='cat', categories=list(
        'abc'), ordered=True)
    expected = DataFrame({'ints': [1.5, 1.5], 'val': [20.0, 30]}, index=
        exp_index)
    if not observed:
        index = CategoricalIndex(list('abc'), name='cat', categories=list(
            'abc'), ordered=True)
        expected = expected.reindex(index)
    tm.assert_frame_equal(result, expected)


def func_yw4r9a6m(observed):
    d = {'cat': Categorical(['a', 'b', 'a', 'b'], categories=['a', 'b', 'c'
        ], ordered=True), 'ints': [1, 1, 2, 2], 'val': [10, 20, 30, 40]}
    df = DataFrame(d)
    groups_double_key = df.groupby(['cat', 'ints'], observed=observed)
    result = groups_double_key.agg('mean')
    expected = DataFrame({'val': [10.0, 30.0, 20.0, 40.0], 'cat':
        Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'c'],
        ordered=True), 'ints': [1, 2, 1, 2]}).set_index(['cat', 'ints'])
    if not observed:
        expected = func_ln7ovfd6(expected, [df.cat.values, [1, 2]], ['cat',
            'ints'])
    tm.assert_frame_equal(result, expected)
    for key in [('a', 1), ('b', 2), ('b', 1), ('a', 2)]:
        c, i = key
        result = groups_double_key.get_group(key)
        expected = df[(df.cat == c) & (df.ints == i)]
        tm.assert_frame_equal(result, expected)


def func_8hn0s1dp(observed):
    d = {'foo': [10, 8, 4, 8, 4, 1, 1], 'bar': [10, 20, 30, 40, 50, 60, 70],
        'baz': ['d', 'c', 'e', 'a', 'a', 'd', 'c']}
    df = DataFrame(d)
    cat = pd.cut(df['foo'], np.linspace(0, 10, 3))
    df['range'] = cat
    groups = df.groupby(['range', 'baz'], as_index=False, observed=observed)
    result = groups.agg('mean')
    groups2 = df.groupby(['range', 'baz'], as_index=True, observed=observed)
    expected = groups2.agg('mean').reset_index()
    tm.assert_frame_equal(result, expected)


def func_9zc9i0xf(observed):
    d = {'C1': [3, 3, 4, 5], 'C2': [1, 2, 3, 4], 'C3': [10, 100, 200, 34]}
    df = DataFrame(d)
    values = pd.cut(df['C1'], [1, 2, 3, 6])
    values.name = 'cat'
    groups_double_key = df.groupby([values, 'C2'], observed=observed)
    idx = MultiIndex.from_arrays([values, [1, 2, 3, 4]], names=['cat', 'C2'])
    expected = DataFrame({'C1': [3.0, 3.0, 4.0, 5.0], 'C3': [10.0, 100.0, 
        200.0, 34.0]}, index=idx)
    if not observed:
        expected = func_ln7ovfd6(expected, [values.values, [1, 2, 3, 4]], [
            'cat', 'C2'])
    result = groups_double_key.agg('mean')
    tm.assert_frame_equal(result, expected)


def func_kfvwqp0y():
    df = DataFrame({'cat': np.random.default_rng(2).integers(0, 255, size=
        30000), 'int_id': np.random.default_rng(2).integers(0, 255, size=
        30000), 'other_id': np.random.default_rng(2).integers(0, 10000,
        size=30000), 'foo': 0})
    df['cat'] = df.cat.astype(str).astype('category')
    grouped = df.groupby(['cat', 'int_id', 'other_id'], observed=True)
    result = grouped.count()
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()


def func_bdpaoxxr(observed):
    cat = Categorical(['a', 'c', 'a'], categories=['a', 'b', 'c'])
    df = DataFrame({'cat': cat, 'vals': [1, 2, 3]})
    g = df.groupby('cat', observed=observed)
    result = g.groups
    if observed:
        expected = {'a': Index([0, 2], dtype='int64'), 'c': Index([1],
            dtype='int64')}
    else:
        expected = {'a': Index([0, 2], dtype='int64'), 'b': Index([], dtype
            ='int64'), 'c': Index([1], dtype='int64')}
    tm.assert_dict_equal(result, expected)


@pytest.mark.parametrize('keys, expected_values, expected_index_levels', [(
    'a', [15, 9, 0], CategoricalIndex([1, 2, 3], name='a')), (['a', 'b'], [
    7, 8, 0, 0, 0, 9, 0, 0, 0], [CategoricalIndex([1, 2, 3], name='a'),
    Index([4, 5, 6])]), (['a', 'a2'], [15, 0, 0, 0, 9, 0, 0, 0, 0], [
    CategoricalIndex([1, 2, 3], name='a'), CategoricalIndex([1, 2, 3], name
    ='a')])])
@pytest.mark.parametrize('test_series', [True, False])
def func_wxxaja70(keys, expected_values, expected_index_levels, test_series):
    df = DataFrame({'a': Categorical([1, 1, 2], categories=[1, 2, 3]), 'a2':
        Categorical([1, 1, 2], categories=[1, 2, 3]), 'b': [4, 5, 6], 'c':
        [7, 8, 9]}).set_index(['a', 'a2'])
    if 'b' not in keys:
        df = df.drop(columns='b')
    gb = df.groupby(keys, observed=False)
    if test_series:
        gb = gb['c']
    result = gb.sum()
    if len(keys) == 1:
        index = expected_index_levels
    else:
        codes = [[0, 0, 0, 1, 1, 1, 2, 2, 2], 3 * [0, 1, 2]]
        index = MultiIndex(expected_index_levels, codes=codes, names=keys)
    expected = DataFrame({'c': expected_values}, index=index)
    if test_series:
        expected = expected['c']
    tm.assert_equal(result, expected)


def func_9c8pli50(observed):
    df = DataFrame({'cat': Categorical(['a', np.nan, 'a'], categories=['a',
        'b', 'd']), 'vals': [1, 2, 3]})
    g = df.groupby('cat', observed=observed)
    result = g.groups
    if observed:
        expected = {'a': Index([0, 2], dtype='int64')}
    else:
        expected = {'a': Index([0, 2], dtype='int64'), 'b': Index([], dtype
            ='int64'), 'd': Index([], dtype='int64')}
    tm.assert_dict_equal(result, expected)


def func_2wkde5dj():
    cat = Categorical(['a', np.nan, np.nan], categories=['a', 'b', 'c'])
    ser = Series([1, 2, 3])
    df = DataFrame({'cat': cat, 'ser': ser})
    result = df.groupby('cat', observed=False)['ser'].nth(0)
    expected = df['ser'].iloc[[0]]
    tm.assert_series_equal(result, expected)


def func_gu7nunl8(observed):
    s1 = Categorical([np.nan, 'a', np.nan, 'a'], categories=['a', 'b', 'c'])
    s2 = Series([1, 2, 3, 4])
    df = DataFrame({'s1': s1, 's2': s2})
    result = df.groupby('s1', observed=observed).first().reset_index()
    if observed:
        expected = DataFrame({'s1': Categorical(['a'], categories=['a', 'b',
            'c']), 's2': [2]})
    else:
        expected = DataFrame({'s1': Categorical(['a', 'b', 'c'], categories
            =['a', 'b', 'c']), 's2': [2, np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('ordered', [True, False])
def func_besw3yuo(ordered, observed, sort):
    label = Categorical(['d', 'a', 'b', 'a', 'd', 'b'], categories=['a',
        'b', 'missing', 'd'], ordered=ordered)
    val = Series(['d', 'a', 'b', 'a', 'd', 'b'])
    df = DataFrame({'label': label, 'val': val})
    result = df.groupby('label', observed=observed, sort=sort)['val'
        ].aggregate('first')
    label = Series(result.index.array, dtype='object')
    aggr = Series(result.array)
    if not observed:
        aggr[aggr.isna()] = 'missing'
    if not all(label == aggr):
        msg = f"""Labels and aggregation results not consistently sorted
for (ordered={ordered}, observed={observed}, sort={sort})
Result:
{result}"""
        pytest.fail(msg)


def func_89mxh063():
    levels = pd.date_range('2014-01-01', periods=4)
    codes = np.random.default_rng(2).integers(0, 4, size=10)
    cats = Categorical.from_codes(codes, levels, ordered=True)
    data = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result = data.groupby(cats, observed=False).mean()
    expected = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(expected.index, categories=expected.
        index, ordered=True)
    tm.assert_frame_equal(result, expected)
    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()
    idx = cats.codes.argsort()
    ord_labels = cats.take(idx)
    ord_data = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(desc_result.index.get_level_values(0), expected.
        index.get_level_values(0))
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp = CategoricalIndex(expc)
    tm.assert_index_equal(desc_result.stack().index.get_level_values(0), exp)
    exp = Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] * 4
        )
    tm.assert_index_equal(desc_result.stack().index.get_level_values(1), exp)


def func_vlbf0tqb():
    s = np.random.default_rng(2)
    levels = ['foo', 'bar', 'baz', 'qux']
    codes = s.integers(0, 4, size=20)
    cats = Categorical.from_codes(codes, levels, ordered=True)
    df = DataFrame(np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list
        ('abcd'))
    df['cats'] = cats
    result = df.set_index('cats').groupby(level=0, observed=False).sum()
    expected = df[list('abcd')].groupby(cats.codes, observed=False).sum()
    expected.index = CategoricalIndex(Categorical.from_codes([0, 1, 2, 3],
        levels, ordered=True), name='cats')
    tm.assert_frame_equal(result, expected)
    result = df.groupby('cats', observed=False).sum()
    expected = df[list('abcd')].groupby(cats.codes, observed=False).sum()
    expected.index = CategoricalIndex(Categorical.from_codes([0, 1, 2, 3],
        levels, ordered=True), name='cats')
    tm.assert_frame_equal(result, expected)


def func_y8h3uu0h():
    cats = CategoricalIndex(['qux', 'foo', 'baz', 'bar'], categories=['foo',
        'bar', 'baz', 'qux'], ordered=True)
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)),
        columns=cats)
    result = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def func_ogi3hfw7():
    df = DataFrame({'a': range(10), 'medium': ['A', 'B'] * 5, 'artist': 
        list('XYXXY') * 2})
    df['medium'] = df['medium'].astype('category')
    gcat = df.groupby(['artist', 'medium'], observed=False)['a'].count(
        ).unstack()
    result = gcat.describe()
    exp_columns = CategoricalIndex(['A', 'B'], ordered=False, name='medium')
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat['A'] + gcat['B']
    expected = Series([6, 4], index=Index(['X', 'Y'], name='artist'))
    tm.assert_series_equal(result, expected)


def func_wbiwnsrf():
    series = Series([np.nan, np.nan, 1, 1, 2, 2, 3, 3, 4, 4])
    bins = pd.cut(series.dropna().values, 4)
    with pytest.raises(ValueError, match='Grouper and axis must be same length'
        ):
        series.groupby(bins).mean()


@pytest.mark.parametrize(['series', 'data'], [(Series(range(4)), {'A': [0, 
    3], 'B': [1, 2]}), (Series(range(4)).rename(lambda idx: idx + 1), {'A':
    [2], 'B': [0, 1]}), (Series(range(7)), {'A': [0, 3], 'B': [1, 2]})])
def func_piecwjd4(series, data):
    groupby = series.groupby(Series(list('ABBA'), dtype='category'),
        observed=False)
    result = groupby.aggregate(list)
    expected = Series(data, index=CategoricalIndex(data.keys()))
    tm.assert_series_equal(result, expected)


def func_3e6eife1():
    df = DataFrame({'cat': Categorical([1, 2, 2], [1, 2, 3]), 'A': [10, 11,
        11], 'B': [101, 102, 103]})
    result = df.groupby(['cat', 'A'], as_index=False, observed=True).sum()
    expected = DataFrame({'cat': Categorical([1, 2], categories=df.cat.cat.
        categories), 'A': [10, 11], 'B': [101, 205]}, columns=['cat', 'A', 'B']
        )
    tm.assert_frame_equal(result, expected)
    f = lambda r: df.loc[r, 'A']
    result = df.groupby(['cat', f], as_index=False, observed=True).sum()
    expected = DataFrame({'cat': Categorical([1, 2], categories=df.cat.cat.
        categories), 'level_1': [10, 11], 'A': [10, 22], 'B': [101, 205]})
    tm.assert_frame_equal(result, expected)
    s = Series(['a', 'b', 'b'], name='cat')
    result = df.groupby(['cat', s], as_index=False, observed=True).sum()
    expected = DataFrame({'cat': ['a', 'b'], 'A': [10, 22], 'B': [101, 205]})
    tm.assert_frame_equal(result, expected)
    group_columns = ['cat', 'A']
    expected = DataFrame({'cat': Categorical([1, 2], categories=df.cat.cat.
        categories), 'A': [10, 11], 'B': [101, 205]}, columns=['cat', 'A', 'B']
        )
    for name in [None, 'X', 'B']:
        df.index = Index(list('abc'), name=name)
        result = df.groupby(group_columns, as_index=False, observed=True).sum()
        tm.assert_frame_equal(result, expected)


def func_qwa9wexb():
    categories = list('abc')
    df = DataFrame({'A': Categorical(list('ba'), categories=categories,
        ordered=True)})
    sort_index = CategoricalIndex(categories, categories, ordered=True,
        name='A')
    nosort_index = CategoricalIndex(list('bac'), categories, ordered=True,
        name='A')
    tm.assert_index_equal(df.groupby('A', sort=True, observed=False).first(
        ).index, sort_index)
    tm.assert_index_equal(df.groupby('A', sort=False, observed=False).first
        ().index, nosort_index)


def func_a3kkk8r9():
    categories = list('abc')
    df = DataFrame({'A': Categorical(list('ba'), categories=categories,
        ordered=False)})
    sort_index = CategoricalIndex(categories, categories, ordered=False,
        name='A')
    nosort_index = CategoricalIndex(list('bac'), list('abc'), ordered=False,
        name='A')
    tm.assert_index_equal(df.groupby('A', sort=True, observed=False).first(
        ).index, sort_index)
    tm.assert_index_equal(df.groupby('A', sort=False, observed=False).first
        ().index, nosort_index)


@pytest.mark.parametrize('col', ['C1', 'C2'])
def func_a1s0oqpv(col):
    df = DataFrame({'A': [1, 2, 1, 1, 2], 'B': [10, 16, 22, 28, 34], 'C1':
        Categorical(list('abaab'), categories=list('bac'), ordered=False),
        'C2': Categorical(list('abaab'), categories=list('bac'), ordered=True)}
        )
    exp_full = DataFrame({'A': [2.0, 1.0, np.nan], 'B': [25.0, 20.0, np.nan
        ], 'C1': Categorical(list('bac'), categories=list('bac'), ordered=
        False), 'C2': Categorical(list('bac'), categories=list('bac'),
        ordered=True)})
    result1 = df.groupby(by=col, as_index=False, observed=False).mean(
        numeric_only=True)
    result2 = df.groupby(by=col, as_index=True, observed=False).mean(
        numeric_only=True).reset_index()
    expected = exp_full.reindex(columns=result1.columns)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)


@pytest.mark.parametrize('func, values', [('first', ['second', 'first']), (
    'last', ['fourth', 'third']), ('min', ['fourth', 'first']), ('max', [
    'second', 'third'])])
def func_00r0hj61(func, values):
    c = Categorical(['first', 'second', 'third', 'fourth'], ordered=True)
    df = DataFrame({'payload': [-1, -2, -1, -2], 'col': c})
    g = df.groupby('payload')
    result = getattr(g, func)()
    expected = DataFrame({'payload': [-2, -1], 'col': Series(values, dtype=
        c.dtype)}).set_index('payload')
    tm.assert_frame_equal(result, expected)
    sgb = df.groupby('payload')['col']
    result = getattr(sgb, func)()
    expected = expected['col']
    tm.assert_series_equal(result, expected)


def func_g68ydj3y():
    data = Series(np.random.default_rng(2).standard_normal(9))
    codes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cats = Categorical.from_codes(codes, [0, 1, 2], ordered=True)
    result = data.groupby(cats, observed=False).mean()
    exp = data.groupby(codes, observed=False).mean()
    exp.index = CategoricalIndex(exp.index, categories=cats.categories,
        ordered=cats.ordered)
    tm.assert_series_equal(result, exp)
    codes = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3])
    cats = Categorical.from_codes(codes, [0, 1, 2, 3], ordered=True)
    result = data.groupby(cats, observed=False).mean()
    exp = data.groupby(codes, observed=False).mean().reindex(cats.categories)
    exp.index = CategoricalIndex(exp.index, categories=cats.categories,
        ordered=cats.ordered)
    tm.assert_series_equal(result, exp)


def func_0e5r7b2o():
    cats = Categorical(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        categories=['a', 'b', 'c', 'd'], ordered=True)
    data = DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 4, 5], 'b': cats})
    result = data.groupby('b', observed=False).mean()
    result = result['a'].values
    exp = np.array([1, 2, 4, np.nan])
    tm.assert_numpy_array_equal(result, exp)


def func_ecsvo68c():
    df = DataFrame({'A': [None] * 3, 'B': Categorical(['train', 'train',
        'test'])})
    result = df.groupby('A').first()['B']
    expected = Series(Categorical([], categories=['test', 'train']), index=
        Series([], dtype='object', name='A'), name='B')
    tm.assert_series_equal(result, expected)


def func_ed7g7mas():
    df = DataFrame({'value': np.random.default_rng(2).integers(0, 10000, 10)})
    labels = [f'{i} - {i + 499}' for i in range(0, 10000, 500)]
    cat_labels = Categorical(labels, labels)
    df = df.sort_values(by=['value'], ascending=True)
    df['value_group'] = pd.cut(df.value, range(0, 10500, 500), right=False,
        labels=cat_labels)
    res = df.groupby(['value_group'], observed=False)['value_group'].count()
    exp = res[sorted(res.index, key=lambda x: float(x.split()[0]))]
    exp.index = CategoricalIndex(exp.index, name=exp.index.name)
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize('ordered', [True, False])
def func_olob8iuw(sort, ordered):
    df = DataFrame([['(7.5, 10]', 10, 10], ['(7.5, 10]', 8, 20], [
        '(2.5, 5]', 5, 30], ['(5, 7.5]', 6, 40], ['(2.5, 5]', 4, 50], [
        '(0, 2.5]', 1, 60], ['(5, 7.5]', 7, 70]], columns=['range', 'foo',
        'bar'])
    df['range'] = Categorical(df['range'], ordered=ordered)
    result = df.groupby('range', sort=sort, observed=False).first()
    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = ['(0, 2.5]', '(2.5, 5]', '(5, 7.5]', '(7.5, 10]']
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = ['(7.5, 10]', '(2.5, 5]', '(5, 7.5]', '(0, 2.5]']
    expected = DataFrame(data_values, columns=['foo', 'bar'], index=
        CategoricalIndex(index_values, name='range', ordered=ordered))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('ordered', [True, False])
def func_rjdxvcmq(sort, ordered):
    df = DataFrame({'dt': [datetime(2011, 7, 1), datetime(2011, 7, 1),
        datetime(2011, 2, 1), datetime(2011, 5, 1), datetime(2011, 2, 1),
        datetime(2011, 1, 1), datetime(2011, 5, 1)], 'foo': [10, 8, 5, 6, 4,
        1, 7], 'bar': [10, 20, 30, 40, 50, 60, 70]}, columns=['dt', 'foo',
        'bar'])
    df['dt'] = Categorical(df['dt'], ordered=ordered)
    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = [datetime(2011, 1, 1), datetime(2011, 2, 1),
            datetime(2011, 5, 1), datetime(2011, 7, 1)]
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = [datetime(2011, 7, 1), datetime(2011, 2, 1),
            datetime(2011, 5, 1), datetime(2011, 1, 1)]
    expected = DataFrame(data_values, columns=['foo', 'bar'], index=
        CategoricalIndex(index_values, name='dt', ordered=ordered))
    result = df.groupby('dt', sort=sort, observed=False).first()
    tm.assert_frame_equal(result, expected)


def func_xujckkhn():
    df = DataFrame({'A': Categorical(['a', 'a', 'b'], categories=['a', 'b',
        'c']), 'B': [1, 2, 1]})
    expected_idx = CategoricalIndex(['a', 'b', 'c'], name='A')
    result = df.groupby('A', observed=False).B.sum()
    expected = Series([3, 1, 0], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.sum(min_count=0)
    expected = Series([3, 1, 0], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.sum(min_count=1)
    expected = Series([3, 1, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.sum(min_count=2)
    expected = Series([3, np.nan, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)


def func_l5i39md2():
    df = DataFrame({'A': Categorical(['a', 'a', 'b'], categories=['a', 'b',
        'c']), 'B': [1, 2, 1]})
    expected_idx = CategoricalIndex(['a', 'b', 'c'], name='A')
    result = df.groupby('A', observed=False).B.prod()
    expected = Series([2, 1, 1], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.prod(min_count=0)
    expected = Series([2, 1, 1], expected_idx, name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', observed=False).B.prod(min_count=1)
    expected = Series([2, 1, np.nan], expected_idx, name='B')
    tm.assert_series_equal(result, expected)


def func_194o3wjx():
    df = DataFrame({'key1': Categorical(list('abcbabcba')), 'key2':
        Categorical(list(pd.date_range('2018-06-01 00', freq='1min',
        periods=3)) * 3), 'values': np.arange(9)})
    result = df.groupby(['key1', 'key2'], observed=False).mean()
    idx = MultiIndex.from_product([Categorical(['a', 'b', 'c']),
        Categorical(pd.date_range('2018-06-01 00', freq='1min', periods=3))
        ], names=['key1', 'key2'])
    expected = DataFrame({'values': [0, 4, 8, 3, 4, 5, 6, np.nan, 2]},
        index=idx)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('as_index, expected', [(True, Series(index=
    MultiIndex.from_arrays([Series([1, 1, 2], dtype='category'), [1, 2, 2]],
    names=['a', 'b']), data=[1, 2, 3], name='x')), (False, DataFrame({'a':
    Series([1, 1, 2], dtype='category'), 'b': [1, 2, 2], 'x': [1, 2, 3]}))])
def func_s7awtlwf(as_index, expected):
    df = DataFrame({'a': Series([1, 1, 2], dtype='category'), 'b': [1, 2, 2
        ], 'x': [1, 2, 3]})
    result = df.groupby(['a', 'b'], as_index=as_index, observed=True)['x'].sum(
        )
    tm.assert_equal(result, expected)


@pytest.mark.parametrize('fill_value', [None, np.nan, pd.NaT])
def func_eiqamu91(fill_value):
    ct = Categorical(['a', 'b', 'c', 'd'], categories=['a', 'b', 'c', 'd'],
        ordered=False)
    expected = Categorical([None, 'a', 'b', 'c'], categories=['a', 'b', 'c',
        'd'], ordered=False)
    res = ct.shift(1, fill_value=fill_value)
    tm.assert_equal(res, expected)


@pytest.fixture
def func_s4zph6jx(df):
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
    df_cat = df.copy()[:4]
    df_cat['A'] = df_cat['A'].astype('category')
    df_cat['B'] = df_cat['B'].astype('category')
    df_cat['C'] = Series([1, 2, 3, 4])
    df_cat = func_s4zph6jx.drop(['D'], axis=1)
    return df_cat


@pytest.mark.parametrize('operation', ['agg', 'apply'])
def func_yavus6jj(df_cat, operation):
    lev_a = Index(['bar', 'bar', 'foo', 'foo'], dtype=df_cat['A'].dtype,
        name='A')
    lev_b = Index(['one', 'three', 'one', 'two'], dtype=df_cat['B'].dtype,
        name='B')
    index = MultiIndex.from_arrays([lev_a, lev_b])
    expected = Series(data=[2, 4, 1, 3], index=index, name='C').sort_index()
    grouped = func_s4zph6jx.groupby(['A', 'B'], observed=True)['C']
    result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('operation', ['agg', 'apply'])
@pytest.mark.parametrize('observed', [False, None])
def func_2cumozeg(df_cat, observed, operation):
    index, _ = MultiIndex.from_product([CategoricalIndex(['bar', 'foo'],
        ordered=False), CategoricalIndex(['one', 'three', 'two'], ordered=
        False)], names=['A', 'B']).sortlevel()
    expected = Series(data=[2, 4, 0, 1, 0, 3], index=index, name='C')
    grouped = func_s4zph6jx.groupby(['A', 'B'], observed=observed)['C']
    result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('observed, index, data', [(True, MultiIndex.
    from_arrays([Index(['bar'] * 4 + ['foo'] * 4, dtype='category', name=
    'A'), Index(['one', 'one', 'three', 'three', 'one', 'one', 'two', 'two'
    ], dtype='category', name='B'), Index(['min', 'max'] * 4)]), [2, 2, 4, 
    4, 1, 1, 3, 3]), (False, MultiIndex.from_product([CategoricalIndex([
    'bar', 'foo'], ordered=False), CategoricalIndex(['one', 'three', 'two'],
    ordered=False), Index(['min', 'max'])], names=['A', 'B', None]), [2, 2,
    4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3]), (None, MultiIndex.
    from_product([CategoricalIndex(['bar', 'foo'], ordered=False),
    CategoricalIndex(['one', 'three', 'two'], ordered=False), Index(['min',
    'max'])], names=['A', 'B', None]), [2, 2, 4, 4, np.nan, np.nan, 1, 1,
    np.nan, np.nan, 3, 3])])
def func_ri13zyew(df_cat, observed, index, data):
    expected = Series(data=data, index=index, name='C')
    result = func_s4zph6jx.groupby(['A', 'B'], observed=observed)['C'].apply(
        lambda x: {'min': x.min(), 'max': x.max()})
    tm.assert_series_equal(result, expected)


def func_v79ef158(df_cat):
    expected = func_s4zph6jx.groupby(['A', 'B'], observed=False)['C'].mean()
    result = func_s4zph6jx.groupby(['A', 'B'], observed=False).mean()['C']
    tm.assert_series_equal(result, expected)


def func_pvx7zq2s(observed, ordered):
    df = DataFrame({'Name': Categorical(['Bob', 'Greg'], ordered=ordered),
        'Item': [1, 2]}, columns=['Name', 'Item'])
    expected = df.copy()
    result = df.groupby('Name', observed=observed).agg(DataFrame.sum,
        skipna=True).reset_index()
    tm.assert_frame_equal(result, expected)


def func_qoikqxfb():
    df = DataFrame({'var': ['a', 'a', 'b', 'b'], 'val': range(4)})
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby('var').apply(lambda rows: DataFrame({'val': [rows.iloc[-
            1]['vau']]}))


def func_gr2akdyq(reduction_func, observed):
    if reduction_func == 'ngroup':
        pytest.skip('ngroup is not truly a reduction')
    df = DataFrame({'cat_1': Categorical(list('AABB'), categories=list(
        'ABCD')), 'cat_2': Categorical(list('AB') * 2, categories=list(
        'ABCD')), 'value': [0.1] * 4})
    args = get_groupby_method_args(reduction_func, df)
    expected_length = 4 if observed else 16
    series_groupby = df.groupby(['cat_1', 'cat_2'], observed=observed)['value']
    if reduction_func == 'corrwith':
        assert not hasattr(series_groupby, reduction_func)
        return
    agg = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match=
            'empty group due to unobserved categories'):
            agg(*args)
        return
    result = agg(*args)
    assert len(result) == expected_length


def func_roeec5p8(reduction_func, request):
    if reduction_func == 'ngroup':
        pytest.skip('ngroup is not truly a reduction')
    if reduction_func == 'corrwith':
        mark = pytest.mark.xfail(reason=
            'TODO: implemented SeriesGroupBy.corrwith. See GH 32293')
        request.applymarker(mark)
    df = DataFrame({'cat_1': Categorical(list('AABB'), categories=list(
        'ABC')), 'cat_2': Categorical(list('AB') * 2, categories=list('ABC'
        )), 'value': [0.1] * 4})
    unobserved = [tuple('AC'), tuple('BC'), tuple('CA'), tuple('CB'), tuple
        ('CC')]
    args = get_groupby_method_args(reduction_func, df)
    series_groupby = df.groupby(['cat_1', 'cat_2'], observed=False)['value']
    agg = getattr(series_groupby, reduction_func)
    if reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match=
            'empty group due to unobserved categories'):
            agg(*args)
        return
    result = agg(*args)
    missing_fillin = _results_for_groupbys_with_missing_categories[
        reduction_func]
    for idx in unobserved:
        val = result.loc[idx]
        assert pd.isna(missing_fillin) and pd.isna(val
            ) or val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ['count', 'nunique', 'size']:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ['sum', 'any']


def func_rla0cfbg(reduction_func):
    if reduction_func == 'ngroup':
        pytest.skip('ngroup does not return the Categories on the index')
    df = DataFrame({'cat_1': Categorical(list('AABB'), categories=list(
        'ABC')), 'cat_2': Categorical(list('1111'), categories=list('12')),
        'value': [0.1, 0.1, 0.1, 0.1]})
    unobserved_cats = [('A', '2'), ('B', '2'), ('C', '1'), ('C', '2')]
    df_grp = df.groupby(['cat_1', 'cat_2'], observed=True)
    args = get_groupby_method_args(reduction_func, df)
    if reduction_func == 'corrwith':
        warn = FutureWarning
        warn_msg = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn = None
        warn_msg = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        res = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize('observed', [False, None])
def func_tsr2vse2(reduction_func, observed):
    if reduction_func == 'ngroup':
        pytest.skip('ngroup does not return the Categories on the index')
    df = DataFrame({'cat_1': Categorical(list('AABB'), categories=list(
        'ABC')), 'cat_2': Categorical(list('1111'), categories=list('12')),
        'value': [0.1, 0.1, 0.1, 0.1]})
    unobserved_cats = [('A', '2'), ('B', '2'), ('C', '1'), ('C', '2')]
    df_grp = df.groupby(['cat_1', 'cat_2'], observed=observed)
    args = get_groupby_method_args(reduction_func, df)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match=
            'empty group due to unobserved categories'):
            getattr(df_grp, reduction_func)(*args)
        return
    if reduction_func == 'corrwith':
        warn = FutureWarning
        warn_msg = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn = None
        warn_msg = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        res = getattr(df_grp, reduction_func)(*args)
    expected = _results_for_groupbys_with_missing_categories[reduction_func]
    if expected is np.nan:
        assert res.loc[unobserved_cats].isnull().all().all()
    else:
        assert (res.loc[unobserved_cats] == expected).all().all()


def func_u970tr07():
    d = {'foo': [10, 8, 4, 1], 'bar': [10, 20, 30, 40], 'baz': ['d', 'c',
        'd', 'c']}
    df = DataFrame(d)
    cat = pd.cut(df['foo'], np.linspace(0, 20, 5))
    df['range'] = cat
    groups = df.groupby(['range', 'baz'], as_index=True, sort=True,
        observed=False)
    result = groups['foo'].agg('mean')
    expected = groups.agg('mean')['foo']
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func, expected_values', [(Series.nunique, [1, 1, 
    2]), (Series.count, [1, 2, 2])])
def func_pqrsfdei(func, expected_values):
    df = DataFrame({'id': [0, 1, 2, 3, 4], 'groups': [0, 1, 1, 2, 2],
        'value': Categorical([0, 0, 0, 0, 1])}).set_index('id')
    result = df.groupby('groups').agg(func)
    expected = DataFrame({'value': expected_values}, index=Index([0, 1, 2],
        name='groups'))
    tm.assert_frame_equal(result, expected)


def func_s5l91ozs():
    df = DataFrame({'A': Categorical(['a', 'a', 'b'], categories=['a', 'b',
        'c'])})
    expected = DataFrame({'A': [2, 1]}, index=np.array([1, 2]))
    result = df.groupby([1, 2, 1]).agg(Series.nunique)
    tm.assert_frame_equal(result, expected)
    result = df.groupby([1, 2, 1]).nunique()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('func', ['first', 'last'])
def func_rtdl3s2s(func):
    df = DataFrame({'A': [1997], 'B': Series(['b'], dtype='category').cat.
        as_ordered()})
    df_grouped = df.groupby('A')['B']
    result = getattr(df_grouped, func)()
    expected = Series(['b'], index=Index([1997], name='A'), name='B', dtype
        =df['B'].dtype)
    tm.assert_series_equal(result, expected)


def func_wr84pvi3():
    cats = np.array([1, 2])
    cats.flags.writeable = False
    df = DataFrame({'a': [1, 3, 5, 7], 'b': Categorical([1, 1, 2, 2],
        categories=Index(cats))})
    expected = DataFrame(data={'a': [2.0, 6.0]}, index=CategoricalIndex(
        cats, name='b'))
    result = df.groupby('b', sort=False, observed=False).mean()
    tm.assert_frame_equal(result, expected)


def func_u0j41vyj():
    df = DataFrame({'foo': ['small', 'large', 'large', 'large', 'medium',
        'large', 'large', 'medium'], 'bar': ['C', 'A', 'A', 'C', 'A', 'C',
        'A', 'C']})
    df['foo'] = df['foo'].astype('category').cat.set_categories(['tiny',
        'small', 'medium', 'large'], ordered=True)
    expected = DataFrame({'tiny': {'A': 0, 'C': 0}, 'small': {'A': 0, 'C': 
        1}, 'medium': {'A': 1, 'C': 1}, 'large': {'A': 3, 'C': 2}})
    expected = expected.rename_axis('bar', axis='index')
    expected.columns = CategoricalIndex(['tiny', 'small', 'medium', 'large'
        ], categories=['tiny', 'small', 'medium', 'large'], ordered=True,
        name='foo', dtype='category')
    result = df.groupby(['bar', 'foo'], observed=False).size().unstack()
    tm.assert_frame_equal(result, expected)


def func_vng1l06i():
    df = DataFrame({'col_num': [1, 1, 2, 3]})
    df['col_cat'] = df['col_num'].astype('category')
    result = df.groupby('col_num').col_cat.first()
    expected = Series([1, 2, 3], index=Index([1, 2, 3], name='col_num'),
        name='col_cat', dtype=df['col_cat'].dtype)
    tm.assert_series_equal(result, expected)
    result = df.groupby('col_num').agg({'col_cat': 'first'})
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)


def func_3gjv8jai():
    df = DataFrame({'A': [1, 1, 1, 1], 'B': [1, 2, 1, 2], 'numerical_col':
        [0.1, 0.2, np.nan, 0.3], 'object_col': ['foo', 'bar', 'foo', 'fee'],
        'categorical_col': ['foo', 'bar', 'foo', 'fee']})
    df = df.astype({'categorical_col': 'category'})
    result = df.groupby(['A', 'B']).agg(lambda df: df.isna().sum())
    index = MultiIndex.from_arrays([[1, 1], [1, 2]], names=('A', 'B'))
    expected = DataFrame(data={'numerical_col': [1, 0], 'object_col': [0, 0
        ], 'categorical_col': [0, 0]}, index=index)
    tm.assert_frame_equal(result, expected)


def func_zdwfil38():
    df = DataFrame({'package_id': [1, 1, 1, 2, 2, 3], 'status': ['Waiting',
        'OnTheWay', 'Delivered', 'Waiting', 'OnTheWay', 'Waiting']})
    delivery_status_type = pd.CategoricalDtype(categories=['Waiting',
        'OnTheWay', 'Delivered'], ordered=True)
    df['status'] = df['status'].astype(delivery_status_type)
    df['last_status'] = df.groupby('package_id')['status'].transform(max)
    result = df.copy()
    expected = DataFrame({'package_id': [1, 1, 1, 2, 2, 3], 'status': [
        'Waiting', 'OnTheWay', 'Delivered', 'Waiting', 'OnTheWay',
        'Waiting'], 'last_status': ['Waiting', 'Waiting', 'Waiting',
        'Waiting', 'Waiting', 'Waiting']})
    expected['status'] = expected['status'].astype(delivery_status_type)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('func', ['first', 'last'])
def func_ghq00pr2(func, observed):
    cat = Categorical([0, 0, 1, 1])
    val = [0, 1, 1, 0]
    df = DataFrame({'a': cat, 'b': cat, 'c': val})
    cat2 = Categorical([0, 1])
    idx = MultiIndex.from_product([cat2, cat2], names=['a', 'b'])
    expected_dict = {'first': Series([0, np.nan, np.nan, 1], idx, name='c'),
        'last': Series([1, np.nan, np.nan, 0], idx, name='c')}
    expected = expected_dict[func]
    if observed:
        expected = expected.dropna().astype(np.int64)
    srs_grp = df.groupby(['a', 'b'], observed=observed)['c']
    result = getattr(srs_grp, func)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func', ['first', 'last'])
def func_8k94f0z8(func, observed):
    cat = Categorical([0, 0, 1, 1])
    val = [0, 1, 1, 0]
    df = DataFrame({'a': cat, 'b': cat, 'c': val})
    cat2 = Categorical([0, 1])
    idx = MultiIndex.from_product([cat2, cat2], names=['a', 'b'])
    expected_dict = {'first': Series([0, np.nan, np.nan, 1], idx, name='c'),
        'last': Series([1, np.nan, np.nan, 0], idx, name='c')}
    expected = expected_dict[func].to_frame()
    if observed:
        expected = expected.dropna().astype(np.int64)
    df_grp = df.groupby(['a', 'b'], observed=observed)
    result = getattr(df_grp, func)()
    tm.assert_frame_equal(result, expected)


def func_zxosunw7():
    df = DataFrame({'key': Categorical(['b', 'b', 'a'], categories=['a',
        'b', 'c']), 'col': range(3)})
    grouped = df.groupby('key', sort=False, observed=False)
    result = grouped.indices
    expected = {'b': np.array([0, 1], dtype='intp'), 'a': np.array([2],
        dtype='intp'), 'c': np.array([], dtype='intp')}
    assert result.keys() == expected.keys()
    for key in result.keys():
        tm.assert_numpy_array_equal(result[key], expected[key])


@pytest.mark.parametrize('func', ['first', 'last'])
def func_olzg9fhq(func):
    df = DataFrame({'a': [1, 2, 3]})
    df['b'] = df['a'].astype('category')
    result = getattr(df.groupby('a')['b'], func)()
    expected = Series(Categorical([1, 2, 3]), name='b', index=Index([1, 2, 
        3], name='a'))
    tm.assert_series_equal(expected, result)


def func_hxh1ttku():
    df = DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [10, 11]})
    df = df.astype(dtype={'a': 'category', 'b': 'category'})
    result = df.groupby(['a', 'b'], observed=True).nunique()['c']
    expected = Series([1, 1], index=MultiIndex.from_arrays([
        CategoricalIndex([1, 2], name='a'), CategoricalIndex([1, 2], name=
        'b')]), name='c')
    tm.assert_series_equal(result, expected)


def func_q199cy1n():
    dtype = pd.CategoricalDtype(categories=['small', 'big'], ordered=True)
    df = DataFrame([[1, 'small'], [1, 'big'], [2, 'small']], columns=['grp',
        'description']).astype({'description': dtype})
    result = df.groupby('grp')['description'].max()
    expected = Series(['big', 'small'], index=Index([1, 2], name='grp'),
        name='description', dtype=pd.CategoricalDtype(categories=['small',
        'big'], ordered=True))
    tm.assert_series_equal(result, expected)


def func_0ocv90rr(observed, dropna):
    cat = Categorical([1, 2], categories=[1, 2, 3])
    df = DataFrame({'x': Categorical([1, 2], categories=[1, 2, 3]), 'y': [3,
        4]})
    gb = df.groupby('x', observed=observed, dropna=dropna)
    result = gb.sum()
    if observed:
        expected = DataFrame({'y': [3, 4]}, index=cat)
    else:
        index = CategoricalIndex([1, 2, 3], [1, 2, 3])
        expected = DataFrame({'y': [3, 4, 0]}, index=index)
    expected.index.name = 'x'
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'])
@pytest.mark.parametrize('ordered', [True, False])
def func_xf85ycxs(request, as_index, sort, observed, reduction_func,
    index_kind, ordered):
    if (reduction_func == 'corrwith' and not as_index and index_kind !=
        'single'):
        msg = (
            'GH#49950 - corrwith with as_index=False may not have grouping column'
            )
        request.applymarker(pytest.mark.xfail(reason=msg))
    elif index_kind != 'range' and not as_index:
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    df = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2],
        ordered=ordered), 'b': range(4)})
    if index_kind == 'range':
        keys = ['a']
    elif index_kind == 'single':
        keys = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    args = get_groupby_method_args(reduction_func, df)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match=
            'empty group due to unobserved categories'):
            getattr(gb, reduction_func)(*args)
        return
    if reduction_func == 'corrwith':
        warn = FutureWarning
        warn_msg = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn = None
        warn_msg = ''
    with tm.assert_produces_warning(warn, match=warn_msg):
        op_result = getattr(gb, reduction_func)(*args)
    if as_index:
        result = op_result.index.get_level_values('a').categories
    else:
        result = op_result['a'].cat.categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize('index_kind', ['single', 'multi'])
@pytest.mark.parametrize('ordered', [True, False])
def func_472nvnzn(as_index, sort, observed, transformation_func, index_kind,
    ordered):
    df = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2],
        ordered=ordered), 'b': range(4)})
    if index_kind == 'single':
        keys = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    args = get_groupby_method_args(transformation_func, df)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, transformation_func)(*args)
    result = op_result.index.get_level_values('a').categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'])
@pytest.mark.parametrize('method', ['head', 'tail'])
@pytest.mark.parametrize('ordered', [True, False])
def func_eqwsiptw(as_index, sort, observed, method, index_kind, ordered):
    df = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2],
        ordered=ordered), 'b': range(4)})
    if index_kind == 'range':
        keys = ['a']
    elif index_kind == 'single':
        keys = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, method)()
    if index_kind == 'range':
        result = op_result['a'].cat.categories
    else:
        result = op_result.index.get_level_values('a').categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'])
@pytest.mark.parametrize('method', ['apply', 'agg', 'transform'])
@pytest.mark.parametrize('ordered', [True, False])
def func_kc6dgxee(as_index, sort, observed, method, index_kind, ordered):
    if (method == 'transform' and index_kind == 'range' or not as_index and
        index_kind != 'range'):
        pytest.skip('No categories in result, nothing to test')
    df = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2],
        ordered=ordered), 'b': range(4)})
    if index_kind == 'range':
        keys = ['a']
    elif index_kind == 'single':
        keys = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, method)(lambda x: x.sum(numeric_only=True))
    if (method == 'transform' or not as_index) and index_kind == 'range':
        result = op_result['a'].cat.categories
    else:
        result = op_result.index.get_level_values('a').categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'])
def func_z1ndy7s5(as_index, sort, index_kind, ordered):
    if index_kind != 'range' and not as_index:
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    categories = np.arange(9999, -1, -1)
    grouper = Categorical([2, 1, 2, 3], categories=categories, ordered=ordered)
    df = DataFrame({'a': grouper, 'b': range(4)})
    if index_kind == 'range':
        keys = ['a']
    elif index_kind == 'single':
        keys = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=True)
    result = gb.sum()
    data = [3, 2, 1] if sort else [2, 1, 3]
    index = CategoricalIndex(data, categories=grouper.categories, ordered=
        ordered, name='a')
    if as_index:
        expected = DataFrame({'b': data})
        if index_kind == 'multi':
            expected.index = MultiIndex.from_frame(DataFrame({'a': index,
                'a2': index}))
        else:
            expected.index = index
    elif index_kind == 'multi':
        expected = DataFrame({'a': Series(index), 'a2': Series(index), 'b':
            data})
    else:
        expected = DataFrame({'a': Series(index), 'b': data})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('test_series', [True, False])
@pytest.mark.parametrize('keys', [['a1'], ['a1', 'a2']])
def func_xulz23f6(request, as_index, observed, reduction_func, test_series,
    keys):
    if test_series and reduction_func == 'corrwith':
        assert not hasattr(SeriesGroupBy, 'corrwith')
        pytest.skip('corrwith not implemented for SeriesGroupBy')
    elif reduction_func == 'corrwith':
        msg = 'GH#32293: attempts to call SeriesGroupBy.corrwith'
        request.applymarker(pytest.mark.xfail(reason=msg))
    df = DataFrame({'a1': [0, 0, 1], 'a2': [2, 3, 3], 'b': [4, 5, 6]})
    df = df.astype({'a1': 'category', 'a2': 'category'})
    if 'a2' not in keys:
        df = df.drop(columns='a2')
    gb = df.groupby(by=keys, as_index=as_index, observed=observed)
    if test_series:
        gb = gb['b']
    args = get_groupby_method_args(reduction_func, df)
    if not observed and reduction_func in ['idxmin', 'idxmax'] and keys == [
        'a1', 'a2']:
        with pytest.raises(ValueError, match=
            'empty group due to unobserved categories'):
            gb.agg([reduction_func], *args)
        return
    result = gb.agg([reduction_func], *args)
    expected = getattr(gb, reduction_func)(*args)
    if as_index and (test_series or reduction_func == 'size'):
        expected = expected.to_frame(reduction_func)
    if not test_series:
        expected.columns = MultiIndex.from_tuples([(ind, '') for ind in
            expected.columns[:-1]] + [('b', reduction_func)])
    elif not as_index:
        expected.columns = keys + [reduction_func]
    tm.assert_equal(result, expected)
