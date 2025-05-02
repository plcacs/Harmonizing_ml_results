"""
The tests in this package are to ensure the proper resultant dtypes of
set operations.
"""
from datetime import datetime
from typing import Any, List, Optional, Set, Tuple, Union, cast
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import CategoricalDtype, CategoricalIndex, DatetimeTZDtype, Index, MultiIndex, PeriodDtype, RangeIndex, Series, Timestamp
import pandas._testing as tm
from pandas.api.types import is_signed_integer_dtype, pandas_dtype
from pandas.core.indexes.base import Index
from numpy.typing import NDArray

def equal_contents(arr1: NDArray[Any], arr2: NDArray[Any]) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)

@pytest.fixture(params=tm.ALL_REAL_NUMPY_DTYPES + ['object', 'category', 'datetime64[ns]', 'timedelta64[ns]'])
def any_dtype_for_small_pos_integer_indexes(request: pytest.FixtureRequest) -> Any:
    """
    Dtypes that can be given to an Index with small positive integers.

    This means that for any dtype `x` in the params list, `Index([1, 2, 3], dtype=x)` is
    valid and gives the correct Index (sub-)class.
    """
    return request.param

@pytest.fixture
def index_flat2(index_flat: Index) -> Index:
    return index_flat

def test_union_same_types(index: Index) -> None:
    idx1 = index.sort_values()
    idx2 = index.sort_values()
    assert idx1.union(idx2).dtype == idx1.dtype

def test_union_different_types(index_flat: Index, index_flat2: Index, request: pytest.FixtureRequest) -> None:
    idx1 = index_flat
    idx2 = index_flat2
    if not idx1.is_unique and (not idx2.is_unique) and (idx1.dtype.kind == 'i') and (idx2.dtype.kind == 'b') or (not idx2.is_unique and (not idx1.is_unique) and (idx2.dtype.kind == 'i') and (idx1.dtype.kind == 'b')):
        mark = pytest.mark.xfail(reason='GH#44000 True==1', raises=ValueError, strict=False)
        request.applymarker(mark)
    common_dtype = find_common_type([idx1.dtype, idx2.dtype])
    warn = None
    msg = "'<' not supported between"
    if not len(idx1) or not len(idx2):
        pass
    elif idx1.dtype.kind == 'c' and (not lib.is_np_dtype(idx2.dtype, 'iufc')) or (idx2.dtype.kind == 'c' and (not lib.is_np_dtype(idx1.dtype, 'iufc'))):
        warn = RuntimeWarning
    elif isinstance(idx1.dtype, PeriodDtype) and isinstance(idx2.dtype, CategoricalDtype) or (isinstance(idx2.dtype, PeriodDtype) and isinstance(idx1.dtype, CategoricalDtype)):
        warn = FutureWarning
        msg = 'PeriodDtype\\[B\\] is deprecated'
        mark = pytest.mark.xfail(reason='Warning not produced on all builds', raises=AssertionError, strict=False)
        request.applymarker(mark)
    any_uint64 = np.uint64 in (idx1.dtype, idx2.dtype)
    idx1_signed = is_signed_integer_dtype(idx1.dtype)
    idx2_signed = is_signed_integer_dtype(idx2.dtype)
    idx1 = idx1.sort_values()
    idx2 = idx2.sort_values()
    with tm.assert_produces_warning(warn, match=msg):
        res1 = idx1.union(idx2)
        res2 = idx2.union(idx1)
    if any_uint64 and (idx1_signed or idx2_signed):
        assert res1.dtype == np.dtype('O')
        assert res2.dtype == np.dtype('O')
    else:
        assert res1.dtype == common_dtype
        assert res2.dtype == common_dtype

@pytest.mark.parametrize('idx1,idx2', [(Index(np.arange(5), RangeIndex(5)), (Index(np.arange(5), Index(np.arange(5))), (Index(np.arange(5)), RangeIndex(5)), (Index(np.arange(5)), Index(np.arange(5)))])
def test_compatible_inconsistent_pairs(idx1: Index, idx2: Index) -> None:
    res1 = idx1.union(idx2)
    res2 = idx2.union(idx1)
    assert res1.dtype in (idx1.dtype, idx2.dtype)
    assert res2.dtype in (idx1.dtype, idx2.dtype)

@pytest.mark.parametrize('left, right, expected', [('int64', 'int64', 'int64'), ('int64', 'uint64', 'object'), ('int64', 'float64', 'float64'), ('uint64', 'float64', 'float64'), ('uint64', 'uint64', 'uint64'), ('float64', 'float64', 'float64'), ('datetime64[ns]', 'int64', 'object'), ('datetime64[ns]', 'uint64', 'object'), ('datetime64[ns]', 'float64', 'object'), ('datetime64[ns, CET]', 'int64', 'object'), ('datetime64[ns, CET]', 'uint64', 'object'), ('datetime64[ns, CET]', 'float64', 'object'), ('Period[D]', 'int64', 'object'), ('Period[D]', 'uint64', 'object'), ('Period[D]', 'float64', 'object')])
@pytest.mark.parametrize('names', [('foo', 'foo', 'foo'), ('foo', 'bar', None)])
def test_union_dtypes(left: str, right: str, expected: str, names: Tuple[str, str, Optional[str]]) -> None:
    left_dtype = pandas_dtype(left)
    right_dtype = pandas_dtype(right)
    a = Index([], dtype=left_dtype, name=names[0])
    b = Index([], dtype=right_dtype, name=names[1])
    result = a.union(b)
    assert result.dtype == expected
    assert result.name == names[2]
    result = a.intersection(b)
    assert result.name == names[2]

@pytest.mark.parametrize('values', [[1, 2, 2, 3], [3, 3]])
def test_intersection_duplicates(values: List[int]) -> None:
    a = Index(values)
    b = Index([3, 3])
    result = a.intersection(b)
    expected = Index([3])
    tm.assert_index_equal(result, expected)

class TestSetOps:

    @pytest.mark.parametrize('case', [0.5, 'xxx'])
    @pytest.mark.parametrize('method', ['intersection', 'union', 'difference', 'symmetric_difference'])
    def test_set_ops_error_cases(self, case: Any, method: str, index: Index) -> None:
        msg = 'Input must be Index or array-like'
        with pytest.raises(TypeError, match=msg):
            getattr(index, method)(case)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_intersection_base(self, index: Index) -> None:
        if isinstance(index, CategoricalIndex):
            pytest.skip(f'Not relevant for {type(index).__name__}')
        first = index[:5].unique()
        second = index[:3].unique()
        intersect = first.intersection(second)
        tm.assert_index_equal(intersect, second)
        if isinstance(index.dtype, DatetimeTZDtype):
            return
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.intersection(case)
            assert equal_contents(result, second)
        if isinstance(index, MultiIndex):
            msg = 'other must be a MultiIndex or a list of tuples'
            with pytest.raises(TypeError, match=msg):
                first.intersection([1, 2, 3])

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_union_base(self, index: Index) -> None:
        index = index.unique()
        first = index[3:]
        second = index[:5]
        everything = index
        union = first.union(second)
        tm.assert_index_equal(union.sort_values(), everything.sort_values())
        if isinstance(index.dtype, DatetimeTZDtype):
            return
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.union(case)
            assert equal_contents(result, everything)
        if isinstance(index, MultiIndex):
            msg = 'other must be a MultiIndex or a list of tuples'
            with pytest.raises(TypeError, match=msg):
                first.union([1, 2, 3])

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_difference_base(self, sort: bool, index: Index) -> None:
        first = index[2:]
        second = index[:4]
        if index.inferred_type == 'boolean':
            answer = set(first).difference(set(second))
        elif isinstance(index, CategoricalIndex):
            answer = []
        else:
            answer = index[4:]
        result = first.difference(second, sort)
        assert equal_contents(result, answer)
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.difference(case, sort)
            assert equal_contents(result, answer)
        if isinstance(index, MultiIndex):
            msg = 'other must be a MultiIndex or a list of tuples'
            with pytest.raises(TypeError, match=msg):
                first.difference([1, 2, 3], sort)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_symmetric_difference(self, index: Index, using_infer_string: bool, request: pytest.FixtureRequest) -> None:
        if using_infer_string and index.dtype == 'object' and (index.inferred_type == 'string'):
            request.applymarker(pytest.mark.xfail(reason='TODO: infer_string'))
        if isinstance(index, CategoricalIndex):
            pytest.skip(f'Not relevant for {type(index).__name__}')
        if len(index) < 2:
            pytest.skip('Too few values for test')
        if index[0] in index[1:] or index[-1] in index[:-1]:
            pytest.skip('Index values no not satisfy test condition.')
        first = index[1:]
        second = index[:-1]
        answer = index[[0, -1]]
        result = first.symmetric_difference(second)
        tm.assert_index_equal(result.sort_values(), answer.sort_values())
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.symmetric_difference(case)
            assert equal_contents(result, answer)
        if isinstance(index, MultiIndex):
            msg = 'other must be a MultiIndex or a list of tuples'
            with pytest.raises(TypeError, match=msg):
                first.symmetric_difference([1, 2, 3])

    @pytest.mark.parametrize('fname, sname, expected_name', [('A', 'A', 'A'), ('A', 'B', None), ('A', None, None), (None, 'B', None), (None, None, None)])
    def test_corner_union(self, index_flat: Index, fname: Optional[str], sname: Optional[str], expected_name: Optional[str]) -> None:
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize('fname, sname, expected_name', [('A', 'A', 'A'), ('A', 'B', None), ('A', None, None), (None, 'B', None), (None, None, None)])
    def test_union_unequal(self, index_flat: Index, fname: Optional[str], sname: Optional[str], expected_name: Optional[str]) -> None:
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        union = first.union(second).sort_values()
        expected = index.set_names(expected_name).sort_values()
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize('fname, sname, expected_name', [('A', 'A', 'A'), ('A', 'B', None), ('A', None, None), (None, 'B', None), (None, None, None)])
    def test_corner_intersect(self, index_flat: Index, fname: Optional[str], sname: Optional[str], expected_name: Optional[str]) -> None:
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(intersect, expected)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.parametrize('fname, sname, expected_name', [('A', 'A', 'A'), ('A', 'B', None), ('A', None, None), (None, 'B', None), (None, None, None)])
    def test_intersect_unequal(self, index_flat: Index, fname: Optional[str], sname: Optional[str], expected_name: Optional[str]) -> None:
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        intersect = first.intersection(second).sort_values()
        expected = index[1:].set_names(expected_name).sort_values()
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_intersection_name_retention_with_nameless(self, index: Index) -> None:
        if isinstance(index, MultiIndex):
            index = index.rename(list(range(index.nlevels)))
        else:
            index = index.rename('foo')
        other = np.asarray(index)
        result = index.intersection(other)
        assert result.name == index.name
        result = index.intersection(other[:0])
        assert result.name == index.name
        result = index[:0].intersection(other)
        assert result.name == index.name

    def test_difference_preserves_type_empty(self, index: Index, sort: bool) -> None:
        if not index.is_unique:
            pytest.skip('Not relevant since index is not unique')
        result = index.difference(index, sort=sort)
        expected = index[:0]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_name_retention_equals(self, index: Index, names: Tuple[str, str, Optional[str]]) -> None:
        if isinstance(index, MultiIndex):
            names_list = [[x] * index.nlevels for x in names]
            index = index.rename(names_list[0])
            other = index.rename(names_list[1])
        else:
            index = index.rename(names[0])
            other = index.rename(names[1])
        assert index.equals(other)
        result = index.difference(other)
        expected = index[:0].rename(names[2])
        tm.assert_index_equal(result, expected)

    def test_intersection_difference_match_empty(self, index: Index, sort: bool) -> None:
        if not index.is_unique:
            pytest.skip('Not relevant because index is not unique')
        inter = index.inters