import re
from typing import Any, Callable, Iterable, List, Optional, Union
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame, Index, MultiIndex, Series, _testing as tm, concat, option_context


@pytest.fixture
def index_or_series2(index_or_series):
    return index_or_series


@pytest.mark.parametrize('other', [None, Series, Index])
def test_str_cat_name(index_or_series, other):
    box: Union[Index, Series] = index_or_series
    values: List[str] = ['a', 'b']
    if other:
        other = other(values)
    else:
        other = values
    result: Union[Index, Series] = box(values, name='name').str.cat(other,
        sep=',')
    assert result.name == 'name'


@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=
    td.skip_if_no('pyarrow'))])
def test_str_cat(index_or_series, infer_string):
    with option_context('future.infer_string', infer_string):
        box: Union[Index, Series] = index_or_series
        s: Union[Index, Series] = box(['a', 'a', 'b', 'b', 'c', np.nan])
        result: Union[str, Series, Index] = s.str.cat()
        expected: str = 'aabbc'
        assert result == expected
        result = s.str.cat(na_rep='-')
        expected = 'aabbc-'
        assert result == expected
        result = s.str.cat(sep='_', na_rep='NA')
        expected = 'a_a_b_b_c_NA'
        assert result == expected
        t: np.ndarray = np.array(['a', np.nan, 'b', 'd', 'foo', np.nan],
            dtype=object)
        expected_index: Union[Index, Series] = box(['aa', 'a-', 'bb', 'bd',
            'cfoo', '--'])
        result = s.str.cat(t, na_rep='-')
        tm.assert_equal(result, expected_index)
        result = s.str.cat(list(t), na_rep='-')
        tm.assert_equal(result, expected_index)
        rgx: str = (
            'If `others` contains arrays or lists \\(or other list-likes.*')
        z: Series = Series(['1', '2', '3'])
        with pytest.raises(ValueError, match=rgx):
            s.str.cat(z.values)
        with pytest.raises(ValueError, match=rgx):
            s.str.cat(list(z))


def test_str_cat_raises_intuitive_error(index_or_series):
    box: Union[Index, Series] = index_or_series
    s: Union[Index, Series] = box(['a', 'b', 'c', 'd'])
    message: str = 'Did you mean to supply a `sep` keyword?'
    with pytest.raises(ValueError, match=message):
        s.str.cat('|')
    with pytest.raises(ValueError, match=message):
        s.str.cat('    ')


@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=
    td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('sep', [None, ''])
@pytest.mark.parametrize('dtype_target', ['object', 'category'])
@pytest.mark.parametrize('dtype_caller', ['object', 'category'])
def test_str_cat_categorical(index_or_series, dtype_caller, dtype_target,
    sep, infer_string):
    box: Union[Index, Series] = index_or_series
    with option_context('future.infer_string', infer_string):
        s: Union[Index, Series] = Index(['a', 'a', 'b', 'a'], dtype=
            dtype_caller)
        s = s if box == Index else Series(s, index=s, dtype=s.dtype)
        t: Index = Index(['b', 'a', 'b', 'c'], dtype=dtype_target)
        expected: Union[Index, Series] = Index(['ab', 'aa', 'bb', 'ac'],
            dtype=object if dtype_caller == 'object' else None)
        expected = expected if box == Index else Series(expected, index=
            Index(s, dtype=dtype_caller), dtype=expected.dtype)
        result: Union[Index, Series] = s.str.cat(t.values, sep=sep)
        tm.assert_equal(result, expected)
        t_series: Series = Series(t.values, index=Index(s, dtype=dtype_caller))
        result = s.str.cat(t_series, sep=sep)
        tm.assert_equal(result, expected)
        result = s.str.cat(t_series.values, sep=sep)
        tm.assert_equal(result, expected)
        t_series_diff: Series = Series(t.values, index=t.values)
        expected_diff: Union[Index, Series] = Index(['aa', 'aa', 'bb', 'bb',
            'aa'], dtype=object if dtype_caller == 'object' else None)
        dtype = (object if dtype_caller == 'object' else s.dtype.categories
            .dtype)
        expected_diff = expected_diff if box == Index else Series(expected_diff
            , index=Index(expected_diff.str[:1], dtype=dtype), dtype=
            expected_diff.dtype)
        result = s.str.cat(t_series_diff, sep=sep)
        tm.assert_equal(result, expected_diff)


@pytest.mark.parametrize('data', [[1, 2, 3], [0.1, 0.2, 0.3], [1, 2, 'b']],
    ids=['integers', 'floats', 'mixed'])
@pytest.mark.parametrize('box', [Series, Index, list, lambda x: np.array(x,
    dtype=object)], ids=['Series', 'Index', 'list', 'np.array'])
def test_str_cat_wrong_dtype_raises(box, data):
    s: Series = Series(['a', 'b', 'c'])
    t: Any = box(data)
    msg: str = 'Concatenation requires list-likes containing only strings.*'
    with pytest.raises(TypeError, match=msg):
        s.str.cat(t, join='outer', na_rep='-')


def test_str_cat_mixed_inputs(index_or_series):
    box: Union[Index, Series] = index_or_series
    s: Union[Index, Series] = Index(['a', 'b', 'c', 'd'])
    s = s if box == Index else Series(s, index=s)
    t: Series = Series(['A', 'B', 'C', 'D'], index=s.values)
    d: DataFrame = concat([t, Series(s, index=s)], axis=1)
    expected: Union[Index, Series] = Index(['aAa', 'bBb', 'cCc', 'dDd'])
    expected = expected if box == Index else Series(expected.values, index=
        s.values)
    result: Union[Index, Series] = s.str.cat(d)
    tm.assert_equal(result, expected)
    result = s.str.cat(d.values)
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    t.index = ['b', 'c', 'd', 'a']
    expected = box(['aDa', 'bAb', 'cBc', 'dCd'])
    expected = expected if box == Index else Series(expected.values, index=
        s.values)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    d.index = ['b', 'c', 'd', 'a']
    expected = box(['aDd', 'bAa', 'cBb', 'dCc'])
    expected = expected if box == Index else Series(expected.values, index=
        s.values)
    result = s.str.cat(d)
    tm.assert_equal(result, expected)
    rgx: str = 'If `others` contains arrays or lists \\(or other list-likes.*'
    z: Series = Series(['1', '2', '3'])
    e: DataFrame = concat([z, z], axis=1)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(e.values)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s.values])
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s])
    rgx_error: str = 'others must be Series, Index, DataFrame,.*'
    u: Series = Series(['a', np.nan, 'c', None])
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat([u, 'u'])
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat([u, d])
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat([u, d.values])
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat([u, [u, d]])
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat(set(u))
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat([u, set(u)])
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat(1)
    with pytest.raises(TypeError, match=rgx_error):
        s.str.cat(iter([t.values, list(s)]))


def test_str_cat_align_indexed(index_or_series, join_type):
    box: Union[Index, Series] = index_or_series
    s: Series = Series(['a', 'b', 'c', 'd'], index=['a', 'b', 'c', 'd'])
    t: Series = Series(['D', 'A', 'E', 'B'], index=['d', 'a', 'e', 'b'])
    sa: Union[Series, Index]
    ta: Union[Series, Index]
    sa, ta = s.align(t, join=join_type)
    expected: Union[Series, Index] = sa.str.cat(ta, na_rep='-')
    if box == Index:
        s = Index(s)
        sa = Index(sa)
        expected = Index(expected)
    result: Union[Series, Index] = s.str.cat(t, join=join_type, na_rep='-')
    tm.assert_equal(result, expected)


def test_str_cat_align_mixed_inputs(join_type):
    s: Series = Series(['a', 'b', 'c', 'd'])
    t: Series = Series(['d', 'a', 'e', 'b'], index=[3, 0, 4, 1])
    d: DataFrame = concat([t, t], axis=1)
    expected_outer: Series = Series(['aaa', 'bbb', 'c--', 'ddd', '-ee'])
    expected: Series = expected_outer.loc[s.index.join(t.index, how=join_type)]
    result: Series = s.str.cat([t, t], join=join_type, na_rep='-')
    tm.assert_series_equal(result, expected)
    result = s.str.cat(d, join=join_type, na_rep='-')
    tm.assert_series_equal(result, expected)
    u: np.ndarray = np.array(['A', 'B', 'C', 'D'])
    expected_outer = Series(['aaA', 'bbB', 'c-C', 'ddD', '-e-'])
    rhs_idx: Index
    if join_type == 'inner':
        rhs_idx = t.index.intersection(s.index)
    elif join_type == 'outer':
        rhs_idx = t.index.union(s.index)
    else:
        rhs_idx = t.index.append(s.index.difference(t.index))
    expected = expected_outer.loc[s.index.join(rhs_idx, how=join_type)]
    result = s.str.cat([t, u], join=join_type, na_rep='-')
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match='others must be Series,.*'):
        s.str.cat([t, list(u)], join=join_type)
    rgx: str = 'If `others` contains arrays or lists \\(or other list-likes.*'
    z: np.ndarray = Series(['1', '2', '3']).values
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(z, join=join_type)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([t, z], join=join_type)


def test_str_cat_all_na(index_or_series, index_or_series2):
    box: Union[Index, Series] = index_or_series
    other: Union[Index, Series] = index_or_series2
    s: Union[Index, Series] = Index(['a', 'b', 'c', 'd'])
    s = s if box == Index else Series(s, index=s)
    t: Union[Index, Series] = index_or_series2([np.nan] * 4, dtype=object)
    t = t if index_or_series2 == Index else Series(t, index=s)
    if box == Series:
        expected: Series = Series([np.nan] * 4, index=s.index, dtype=s.dtype)
    else:
        expected = Index([np.nan] * 4, dtype=object)
    result: Union[Index, Series] = s.str.cat(t, join='left')
    tm.assert_equal(result, expected)
    if index_or_series2 == Series:
        expected = Series([np.nan] * 4, dtype=object, index=t.index)
        result = t.str.cat(s, join='left')
        tm.assert_series_equal(result, expected)


def test_str_cat_special_cases():
    s: Series = Series(['a', 'b', 'c', 'd'])
    t: Series = Series(['d', 'a', 'e', 'b'], index=[3, 0, 4, 1])
    expected: Series = Series(['aaa', 'bbb', 'c-c', 'ddd', '-e-'])
    result: Series = s.str.cat(iter([t, s.values]), join='outer', na_rep='-')
    tm.assert_series_equal(result, expected)
    expected_right: Series = Series(['aa-', 'd-d'], index=[0, 3])
    result = s.str.cat([t.loc[[0]], t.loc[[3]]], join='right', na_rep='-')
    tm.assert_series_equal(result, expected_right)


def test_cat_on_filtered_index():
    df: DataFrame = DataFrame(index=MultiIndex.from_product([[2011, 2012],
        [1, 2, 3]], names=['year', 'month']))
    df = df.reset_index()
    df = df[df.month > 1]
    str_year: Series = df.year.astype('str')
    str_month: Series = df.month.astype('str')
    str_both: Series = str_year.str.cat(str_month, sep=' ')
    assert str_both.loc[1] == '2011 2'
    str_multiple: Series = str_year.str.cat([str_month, str_month], sep=' ')
    assert str_multiple.loc[1] == '2011 2 2'


@pytest.mark.parametrize('klass', [tuple, list, np.array, Series, Index])
def test_cat_different_classes(klass):
    s: Series = Series(['a', 'b', 'c'])
    result: Series = s.str.cat(klass(['x', 'y', 'z']))
    expected: Series = Series(['ax', 'by', 'cz'])
    tm.assert_series_equal(result, expected)


def test_cat_on_series_dot_str():
    ps: Series = Series(['AbC', 'de', 'FGHI', 'j', 'kLLLm'])
    message: str = re.escape(
        'others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])'
        )
    with pytest.raises(TypeError, match=message):
        ps.str.cat(others=ps.str)
