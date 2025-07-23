import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    concat,
    option_context,
)
from typing import Optional, Type, Union, List, Any, Iterable, Callable

@pytest.fixture
def index_or_series2(index_or_series: Union[Index, Series]) -> Union[Index, Series]:
    return index_or_series

@pytest.mark.parametrize('other', [None, Series, Index])
def test_str_cat_name(
    index_or_series: Union[Index, Series],
    other: Optional[Type[Union[Series, Index]]]
) -> None:
    box = index_or_series
    values: List[str] = ['a', 'b']
    if other:
        other_instance: Union[Series, Index] = other(values)
    else:
        other_instance = values
    result: Union[Series, Index] = box(values, name='name').str.cat(other_instance, sep=',')
    assert result.name == 'name'

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
def test_str_cat(
    index_or_series: Union[Index, Series],
    infer_string: bool
) -> None:
    with option_context('future.infer_string', infer_string):
        box = index_or_series
        s: Union[Series, Index] = box(['a', 'a', 'b', 'b', 'c', np.nan])
        result = s.str.cat()
        expected: str = 'aabbc'
        assert result == expected
        result = s.str.cat(na_rep='-')
        expected = 'aabbc-'
        assert result == expected
        result = s.str.cat(sep='_', na_rep='NA')
        expected = 'a_a_b_b_c_NA'
        assert result == expected
        t: np.ndarray = np.array(['a', np.nan, 'b', 'd', 'foo', np.nan], dtype=object)
        expected: Union[Series, Index] = box(['aa', 'a-', 'bb', 'bd', 'cfoo', '--'])
        result = s.str.cat(t, na_rep='-')
        tm.assert_equal(result, expected)
        result = s.str.cat(list(t), na_rep='-')
        tm.assert_equal(result, expected)
        rgx: str = r'If `others` contains arrays or lists \(or other list-likes.*'
        z: Series = Series(['1', '2', '3'])
        with pytest.raises(ValueError, match=rgx):
            s.str.cat(z.values)
        with pytest.raises(ValueError, match=rgx):
            s.str.cat(list(z))

def test_str_cat_raises_intuitive_error(index_or_series: Union[Index, Series]) -> None:
    box = index_or_series
    s: Union[Series, Index] = box(['a', 'b', 'c', 'd'])
    message: str = 'Did you mean to supply a `sep` keyword?'
    with pytest.raises(ValueError, match=message):
        s.str.cat('|')  # type: ignore
    with pytest.raises(ValueError, match=message):
        s.str.cat('    ')  # type: ignore

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('sep', ['', None])
@pytest.mark.parametrize('dtype_target', ['object', 'category'])
@pytest.mark.parametrize('dtype_caller', ['object', 'category'])
def test_str_cat_categorical(
    index_or_series: Union[Index, Series],
    dtype_caller: str,
    dtype_target: str,
    sep: Optional[str],
    infer_string: bool
) -> None:
    box = index_or_series
    with option_context('future.infer_string', infer_string):
        s_index: Index = Index(['a', 'a', 'b', 'a'], dtype=dtype_caller)
        s: Union[Index, Series] = s_index if box == Index else Series(s_index, index=s_index, dtype=s_index.dtype)
        t_index: Index = Index(['b', 'a', 'b', 'c'], dtype=dtype_target)
        expected_index: Index = Index(['ab', 'aa', 'bb', 'ac'], dtype=object if dtype_caller == 'object' else None)
        expected: Union[Series, Index] = expected_index if box == Index else Series(expected_index, index=Index(s, dtype=dtype_caller), dtype=expected_index.dtype)
        result: Union[Series, Index] = s.str.cat(t_index.values, sep=sep)
        tm.assert_equal(result, expected)
        t_series: Series = Series(t_index.values, index=Index(s, dtype=dtype_caller))
        result = s.str.cat(t_series, sep=sep)
        tm.assert_equal(result, expected)
        result = s.str.cat(t_index.values, sep=sep)
        tm.assert_equal(result, expected)
        t_series = Series(t_index.values, index=t_index.values)
        expected_index = Index(['aa', 'aa', 'bb', 'bb', 'aa'])
        dtype = object if dtype_caller == 'object' else s.dtype.categories.dtype
        expected_index = Index(['aa', 'aa', 'bb', 'bb', 'aa'], dtype=object if dtype_caller == 'object' else None)
        expected = expected_index if box == Index else Series(expected_index, index=Index(expected_index.str[:1], dtype=dtype), dtype=object)
        result = s.str.cat(t_series, sep=sep)
        tm.assert_equal(result, expected)

@pytest.mark.parametrize('data', [[1, 2, 3], [0.1, 0.2, 0.3], [1, 2, 'b']], ids=['integers', 'floats', 'mixed'])
@pytest.mark.parametrize('box', [Series, Index, list, lambda x: np.array(x, dtype=object)], ids=['Series', 'Index', 'list', 'np.array'])
def test_str_cat_wrong_dtype_raises(
    box: Callable[[List[Any]], Any],
    data: List[Any]
) -> None:
    s: Series = Series(['a', 'b', 'c'])
    t: Any = box(data)
    msg: str = 'Concatenation requires list-likes containing only strings.*'
    with pytest.raises(TypeError, match=msg):
        s.str.cat(t, join='outer', na_rep='-')

def test_str_cat_mixed_inputs(index_or_series: Union[Index, Series]) -> None:
    box = index_or_series
    s_index: Index = Index(['a', 'b', 'c', 'd'])
    s: Union[Index, Series] = s_index if box == Index else Series(s_index, index=s_index)
    t: Series = Series(['A', 'B', 'C', 'D'], index=s_index.values)
    d: DataFrame = concat([t, Series(s, index=s)], axis=1)
    expected_index: Index = Index(['aAa', 'bBb', 'cCc', 'dDd'])
    expected: Union[Series, Index] = expected_index if box == Index else Series(expected_index.values, index=s_index.values)
    result: Union[Series, Index] = s.str.cat(d)
    tm.assert_equal(result, expected)
    result = s.str.cat(d.values)
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    t.index = ['b', 'c', 'd', 'a']
    expected_index = Index(['aDa', 'bAb', 'cBc', 'dCd'])
    expected = expected_index if box == Index else Series(expected_index.values, index=s_index.values)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    d.index = ['b', 'c', 'd', 'a']
    expected_index = Index(['aDd', 'bAa', 'cBb', 'dCc'])
    expected = expected_index if box == Index else Series(expected_index.values, index=s_index.values)
    result = s.str.cat(d)
    tm.assert_equal(result, expected)
    rgx: str = r'If `others` contains arrays or lists \(or other list-likes.*'
    z: Series = Series(['1', '2', '3'])
    e: DataFrame = concat([z, z], axis=1)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(e.values)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s.values])
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s])
    rgx = r'others must be Series, Index, DataFrame,.*'
    u: Series = Series(['a', np.nan, 'c', None])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, 'u'])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d.values])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, [u, d]])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(set(u))
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, set(u)])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(1)
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(iter([t.values, list(s)]))

def test_str_cat_align_indexed(
    index_or_series: Union[Index, Series],
    join_type: str
) -> None:
    box = index_or_series
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

def test_str_cat_align_mixed_inputs(join_type: str) -> None:
    s: Series = Series(['a', 'b', 'c', 'd'])
    t: Series = Series(['d', 'a', 'e', 'b'], index=[3, 0, 4, 1])
    d: DataFrame = concat([t, t], axis=1)
    expected_outer: Series = Series(['aaa', 'bbb', 'c--', 'ddd', '-ee'])
    if join_type == 'inner':
        rhs_idx = t.index.intersection(s.index)
    elif join_type == 'outer':
        rhs_idx = t.index.union(s.index)
    else:
        rhs_idx = t.index.append(s.index.difference(t.index))
    expected: Series = expected_outer.loc[s.index.join(rhs_idx, how=join_type)]
    result: Series = s.str.cat([t, t], join=join_type, na_rep='-')
    tm.assert_series_equal(result, expected)
    result = s.str.cat(d, join=join_type, na_rep='-')
    tm.assert_series_equal(result, expected)
    u: np.ndarray = np.array(['A', 'B', 'C', 'D'])
    expected_outer = Series(['aaA', 'bbB', 'c-C', 'ddD', '-e-'])
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
    rgx = r'If `others` contains arrays or lists \(or other list-likes.*'
    z: np.ndarray = Series(['1', '2', '3']).values
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(z, join=join_type)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([t, z], join=join_type)

def test_str_cat_all_na(
    index_or_series: Union[Index, Series],
    index_or_series2: Union[Index, Series]
) -> None:
    box: Union[Index, Series] = index_or_series
    other: Union[Index, Series] = index_or_series2
    s: Union[Index, Series] = Index(['a', 'b', 'c', 'd'])
    s = s if box == Index else Series(s, index=s)
    t: Union[Index, Series] = other([np.nan] * 4, dtype=object)
    t = t if other == Index else Series(t, index=s)
    if box == Series:
        expected: Series = Series([np.nan] * 4, index=s.index, dtype=s.dtype)
    else:
        expected = Index([np.nan] * 4, dtype=object)
    result: Union[Series, Index] = s.str.cat(t, join='left')
    tm.assert_equal(result, expected)
    if isinstance(other, Series):
        expected = Series([np.nan] * 4, dtype=object, index=t.index)
        result = t.str.cat(s, join='left')
        tm.assert_series_equal(result, expected)

def test_str_cat_special_cases() -> None:
    s: Series = Series(['a', 'b', 'c', 'd'])
    t: Series = Series(['d', 'a', 'e', 'b'], index=[3, 0, 4, 1])
    expected: Series = Series(['aaa', 'bbb', 'c-c', 'ddd', '-e-'])
    result: Series = s.str.cat(iter([t, s.values]), join='outer', na_rep='-')
    tm.assert_series_equal(result, expected)
    expected = Series(['aa-', 'd-d'], index=[0, 3])
    result = s.str.cat([t.loc[[0]], t.loc[[3]]], join='right', na_rep='-')
    tm.assert_series_equal(result, expected)

def test_cat_on_filtered_index() -> None:
    df: DataFrame = DataFrame(index=MultiIndex.from_product([[2011, 2012], [1, 2, 3]], names=['year', 'month']))
    df = df.reset_index()
    df = df[df.month > 1]
    str_year: Series = df.year.astype('str')
    str_month: Series = df.month.astype('str')
    str_both: Series = str_year.str.cat(str_month, sep=' ')
    assert str_both.loc[1] == '2011 2'
    str_multiple: Series = str_year.str.cat([str_month, str_month], sep=' ')
    assert str_multiple.loc[1] == '2011 2 2'

@pytest.mark.parametrize('klass', [tuple, list, np.array, Series, Index])
def test_cat_different_classes(klass: Type) -> None:
    s: Series = Series(['a', 'b', 'c'])
    result: Series = s.str.cat(klass(['x', 'y', 'z']))
    expected: Series = Series(['ax', 'by', 'cz'])
    tm.assert_series_equal(result, expected)

def test_cat_on_series_dot_str() -> None:
    ps: Series = Series(['AbC', 'de', 'FGHI', 'j', 'kLLLm'])
    message: str = re.escape(
        'others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])'
    )
    with pytest.raises(TypeError, match=message):
        ps.str.cat(others=ps.str)
