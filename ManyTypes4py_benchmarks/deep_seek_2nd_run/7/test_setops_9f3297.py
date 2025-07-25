import numpy as np
import pytest
import pandas as pd
from pandas import CategoricalIndex, DataFrame, Index, IntervalIndex, MultiIndex, Series
import pandas._testing as tm
from pandas.api.types import is_float_dtype, is_unsigned_integer_dtype
from typing import Any, List, Optional, Tuple, Union

@pytest.mark.parametrize('case', [0.5, 'xxx'])
@pytest.mark.parametrize('method', ['intersection', 'union', 'difference', 'symmetric_difference'])
def test_set_ops_error_cases(idx: MultiIndex, case: Union[float, str], sort: Optional[bool], method: str) -> None:
    msg = 'Input must be Index or array-like'
    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)(case, sort=sort)

@pytest.mark.parametrize('klass', [MultiIndex, np.array, Series, list])
def test_intersection_base(idx: MultiIndex, sort: Optional[bool], klass: type) -> None:
    first = idx[2::-1]
    second = idx[:5]
    if klass is not MultiIndex:
        second = klass(second.values)
    intersect = first.intersection(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(intersect, expected)
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        first.intersection([1, 2, 3], sort=sort)

@pytest.mark.arm_slow
@pytest.mark.parametrize('klass', [MultiIndex, np.array, Series, list])
def test_union_base(idx: MultiIndex, sort: Optional[bool], klass: type) -> None:
    first = idx[::-1]
    second = idx[:5]
    if klass is not MultiIndex:
        second = klass(second.values)
    union = first.union(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(union, expected)
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        first.union([1, 2, 3], sort=sort)

def test_difference_base(idx: MultiIndex, sort: Optional[bool]) -> None:
    second = idx[4:]
    answer = idx[:4]
    result = idx.difference(second, sort=sort)
    if sort is None:
        answer = answer.sort_values()
    assert result.equals(answer)
    tm.assert_index_equal(result, answer)
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = idx.difference(case, sort=sort)
        tm.assert_index_equal(result, answer)
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        idx.difference([1, 2, 3], sort=sort)

def test_symmetric_difference(idx: MultiIndex, sort: Optional[bool]) -> None:
    first = idx[1:]
    second = idx[:-1]
    answer = idx[[-1, 0]]
    result = first.symmetric_difference(second, sort=sort)
    if sort is None:
        answer = answer.sort_values()
    tm.assert_index_equal(result, answer)
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = first.symmetric_difference(case, sort=sort)
        tm.assert_index_equal(result, answer)
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        first.symmetric_difference([1, 2, 3], sort=sort)

def test_multiindex_symmetric_difference() -> None:
    idx = MultiIndex.from_product([['a', 'b'], ['A', 'B']], names=['a', 'b'])
    result = idx.symmetric_difference(idx)
    assert result.names == idx.names
    idx2 = idx.copy().rename(['A', 'B'])
    result = idx.symmetric_difference(idx2)
    assert result.names == [None, None]

def test_empty(idx: MultiIndex) -> None:
    assert not idx.empty
    assert idx[:0].empty

def test_difference(idx: MultiIndex, sort: Optional[bool]) -> None:
    first = idx
    result = first.difference(idx[-3:], sort=sort)
    vals = idx[:-3].values
    if sort is None:
        vals = sorted(vals)
    expected = MultiIndex.from_tuples(vals, sortorder=0, names=idx.names)
    assert isinstance(result, MultiIndex)
    assert result.equals(expected)
    assert result.names == idx.names
    tm.assert_index_equal(result, expected)
    result = idx.difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names
    result = idx[-3:].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names
    result = idx[:0].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names
    chunklet = idx[-3:]
    chunklet.names = ['foo', 'baz']
    result = first.difference(chunklet, sort=sort)
    assert result.names == (None, None)
    result = idx.difference(idx.sortlevel(1)[0], sort=sort)
    assert len(result) == 0
    result = first.difference(first.values, sort=sort)
    assert result.equals(first[:0])
    result = first.difference([], sort=sort)
    assert first.equals(result)
    assert first.names == result.names
    result = first.difference([('foo', 'one')], sort=sort)
    expected = MultiIndex.from_tuples([('bar', 'one'), ('baz', 'two'), ('foo', 'two'), ('qux', 'one'), ('qux', 'two')])
    expected.names = first.names
    assert first.names == result.names
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        first.difference([1, 2, 3, 4, 5], sort=sort)

def test_difference_sort_special() -> None:
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    result = idx.difference([])
    tm.assert_index_equal(result, idx)

def test_difference_sort_special_true() -> None:
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    result = idx.difference([], sort=True)
    expected = MultiIndex.from_product([[0, 1], ['a', 'b']])
    tm.assert_index_equal(result, expected)

def test_difference_sort_incomparable() -> None:
    idx = MultiIndex.from_product([[1, pd.Timestamp('2000'), 2], ['a', 'b']])
    other = MultiIndex.from_product([[3, pd.Timestamp('2000'), 4], ['c', 'd']])
    msg = 'sort order is undefined for incomparable objects'
    with tm.assert_produces_warning(RuntimeWarning, match=msg):
        result = idx.difference(other)
    tm.assert_index_equal(result, idx)
    result = idx.difference(other, sort=False)
    tm.assert_index_equal(result, idx)

def test_difference_sort_incomparable_true() -> None:
    idx = MultiIndex.from_product([[1, pd.Timestamp('2000'), 2], ['a', 'b']])
    other = MultiIndex.from_product([[3, pd.Timestamp('2000'), 4], ['c', 'd']])
    msg = "'values' is not ordered, please explicitly specify the categories order "
    with pytest.raises(TypeError, match=msg):
        idx.difference(other, sort=True)

def test_union(idx: MultiIndex, sort: Optional[bool]) -> None:
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]
    the_union = piece1.union(piece2, sort=sort)
    if sort in (None, False):
        tm.assert_index_equal(the_union.sort_values(), idx.sort_values())
    else:
        tm.assert_index_equal(the_union, idx)
    the_union = idx.union(idx, sort=sort)
    tm.assert_index_equal(the_union, idx)
    the_union = idx.union(idx[:0], sort=sort)
    tm.assert_index_equal(the_union, idx)
    tuples = idx.values
    result = idx[:4].union(tuples[4:], sort=sort)
    if sort is None:
        tm.assert_index_equal(result.sort_values(), idx.sort_values())
    else:
        assert result.equals(idx)

def test_union_with_regular_index(idx: MultiIndex, using_infer_string: bool) -> None:
    other = Index(['A', 'B', 'C'])
    result = other.union(idx)
    assert ('foo', 'one') in result
    assert 'B' in result
    if using_infer_string:
        with pytest.raises(NotImplementedError, match='Can only union'):
            idx.union(other)
    else:
        msg = 'The values in the array are unorderable'
        with tm.assert_produces_warning(RuntimeWarning, match=msg):
            result2 = idx.union(other)
        assert not result.equals(result2)

def test_intersection(idx: MultiIndex, sort: Optional[bool]) -> None:
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]
    the_int = piece1.intersection(piece2, sort=sort)
    if sort in (None, True):
        tm.assert_index_equal(the_int, idx[3:5])
    else:
        tm.assert_index_equal(the_int.sort_values(), idx[3:5])
    the_int = idx.intersection(idx, sort=sort)
    tm.assert_index_equal(the_int, idx)
    empty = idx[:2].intersection(idx[2:], sort=sort)
    expected = idx[:0]
    assert empty.equals(expected)
    tuples = idx.values
    result = idx.intersection(tuples)
    assert result.equals(idx)

@pytest.mark.parametrize('method', ['intersection', 'union', 'difference', 'symmetric_difference'])
def test_setop_with_categorical(idx: MultiIndex, sort: Optional[bool], method: str) -> None:
    other = idx.to_flat_index().astype('category')
    res_names = [None] * idx.nlevels
    result = getattr(idx, method)(other, sort=sort)
    expected = getattr(idx, method)(idx, sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)
    result = getattr(idx, method)(other[:5], sort=sort)
    expected = getattr(idx, method)(idx[:5], sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)

def test_intersection_non_object(idx: MultiIndex, sort: Optional[bool]) -> None:
    other = Index(range(3), name='foo')
    result = idx.intersection(other, sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
    tm.assert_index_equal(result, expected, exact=True)
    result = idx.intersection(np.asarray(other)[:0], sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
    tm.assert_index_equal(result, expected, exact=True)
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        idx.intersection(np.asarray(other), sort=sort)

def test_intersect_equal_sort() -> None:
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
    tm.assert_index_equal(idx.intersection(idx, sort=None), idx)

def test_intersect_equal_sort_true() -> None:
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    expected = MultiIndex.from_product([[0, 1], ['a', 'b']])
    result = idx.intersection(idx, sort=True)
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('slice_', [slice(None), slice(0)])
def test_union_sort_other_empty(slice_: slice) -> None:
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    other = idx[slice_]
    tm.assert_index_equal(idx.union(other), idx)
    tm.assert_index_equal(other.union(idx), idx)
    tm.assert_index_equal(idx.union(other, sort=False), idx)

def test_union_sort_other_empty_sort() -> None:
    idx = MultiIndex.from_product([[1, 0], ['a', 'b']])
    other = idx[:0]
    result = idx.union(other, sort=True)
    expected = MultiIndex.from_product([[0, 1], ['a', 'b']])
    tm.assert_index_equal(result, expected)

def test_union_sort_other_incomparable() -> None:
    idx = MultiIndex.from_product([[1, pd.Timestamp('2000')], ['a', 'b']])
    with tm.assert_produces_warning(RuntimeWarning, match='are unorderable'):
        result = idx.union(idx[:1])
    tm.assert_index_equal(result, idx)
    result = idx.union(idx[:1], sort=False)
    tm.assert_index_equal(result, idx)

def test_union_sort_other_incomparable_sort() -> None:
    idx = MultiIndex.from_product([[1, pd.Timestamp('2000')], ['a', 'b']])
    msg = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=msg):
        idx.union(idx[:1], sort=True)

def test_union_non_object_dtype_raises() -> None:
    mi = MultiIndex.from_product([['a', 'b'], [1, 2]])
    idx = mi.levels[1]
    msg = 'Can only union MultiIndex with MultiIndex or Index of tuples'
    with pytest.raises(NotImplementedError, match=msg):
        mi.union(idx)

def test_union_empty_self_different_names() -> None:
    mi = MultiIndex.from_arrays([[]])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['a', 'b'])
    result = mi.union(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]])
    tm.assert_index_equal(result, expected)

def test_union_multiindex_empty_rangeindex() -> None:
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['a', 'b'])
    ri = pd.RangeIndex(0)
    result_left = mi.union(ri)
    tm.assert_index_equal(mi, result_left, check_names=False)
    result_right = ri.union(mi)
    tm.assert_index_equal(mi, result_right, check_names=False)

@pytest.mark.parametrize('method', ['union', 'intersection', 'difference', 'symmetric_difference'])
def test_setops_sort_validation(method: str) -> None:
    idx1 = MultiIndex.from_product([['a', 'b'], [1, 2]])
    idx2 = MultiIndex.from_product([['b', 'c'], [1, 2]])
    with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
        getattr(idx1, method)(idx2, sort=2)
    getattr(idx1, method)(idx2, sort=True)

@pytest.mark.parametrize('val', [pd.NA, 100])
def test_difference_keep_ea_dtypes(any_numeric_ea_dtype: str, val: Any) -> None:
    midx = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=['a', None])
    midx2 = MultiIndex.from_arrays([Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]])
    result = midx.difference(midx2)
    expected = MultiIndex.from_arrays([Series([1], dtype=any_numeric_ea_dtype), [2]])
    tm.assert_index_equal(result, expected)
    result = midx.difference(midx.sort_values(ascending=False))
    expected = MultiIndex.from_arrays([Series([], dtype=any_numeric_ea_dtype), Series([], dtype=np.int64)], names=['a', None])
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('val', [pd.NA, 5])
def test_symmetric_difference_keeping_ea_dtype(any_numeric_ea_dtype: str, val: Any) -> None:
    midx = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=['a', None])
    midx2 = MultiIndex.from_arrays([Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]])
    result = midx.symmetric_difference(midx2)
    expected = MultiIndex.from_arrays([Series([1, 1, val], dtype=any_numeric_ea_dtype), [1, 2, 3]])
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize(('tuples', 'exp_tuples'), [([('val1', 'test1')], [('val1', 'test1')]), ([('val1', 'test1'), ('val1', 'test1')], [('