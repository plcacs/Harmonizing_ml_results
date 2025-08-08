from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import Index, MultiIndex, Series, Timestamp
from pandas.core.dtypes.cast import find_common_type
from pandas.api.types import is_signed_integer_dtype, pandas_dtype

def equal_contents(arr1: list, arr2: list) -> bool:
    return frozenset(arr1) == frozenset(arr2)

@pytest.fixture(params=tm.ALL_REAL_NUMPY_DTYPES + ['object', 'category', 'datetime64[ns]', 'timedelta64[ns]'])
def any_dtype_for_small_pos_integer_indexes(request: pytest.FixtureRequest) -> str:
    return request.param

@pytest.fixture
def index_flat2(index_flat: Index) -> Index:
    return index_flat

def test_union_same_types(index: Index):
    idx1 = index.sort_values()
    idx2 = index.sort_values()
    assert idx1.union(idx2).dtype == idx1.dtype

def test_union_different_types(index_flat: Index, index_flat2: Index, request: pytest.FixtureRequest):
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
    elif idx1.dtype.kind == 'c' and (not lib.is_np_dtype(idx2.dtype, 'iufc')) or (idx2.dtype.kind == 'c' and (not lib.is_np_dtype(idx1.dtype, 'iufc')):
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

@pytest.mark.parametrize('idx1,idx2', [(Index(np.arange(5), dtype=np.int64), RangeIndex(5)), (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.int64)), (Index(np.arange(5), dtype=np.float64), RangeIndex(5)), (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.uint64))])
def test_compatible_inconsistent_pairs(idx1: Index, idx2: Index):
    res1 = idx1.union(idx2)
    res2 = idx2.union(idx1)
    assert res1.dtype in (idx1.dtype, idx2.dtype)
    assert res2.dtype in (idx1.dtype, idx2.dtype)

@pytest.mark.parametrize('left, right, expected', [('int64', 'int64', 'int64'), ('int64', 'uint64', 'object'), ('int64', 'float64', 'float64'), ('uint64', 'float64', 'float64'), ('uint64', 'uint64', 'uint64'), ('float64', 'float64', 'float64'), ('datetime64[ns]', 'int64', 'object'), ('datetime64[ns]', 'uint64', 'object'), ('datetime64[ns]', 'float64', 'object'), ('datetime64[ns, CET]', 'int64', 'object'), ('datetime64[ns, CET]', 'uint64', 'object'), ('datetime64[ns, CET]', 'float64', 'object'), ('Period[D]', 'int64', 'object'), ('Period[D]', 'uint64', 'object'), ('Period[D]', 'float64', 'object')])
@pytest.mark.parametrize('names', [('foo', 'foo', 'foo'), ('foo', 'bar', None)])
def test_union_dtypes(left: str, right: str, expected: str, names: tuple):
    left = pandas_dtype(left)
    right = pandas_dtype(right)
    a = Index([], dtype=left, name=names[0])
    b = Index([], dtype=right, name=names[1])
    result = a.union(b)
    assert result.dtype == expected
    assert result.name == names[2]
    result = a.intersection(b)
    assert result.name == names[2]

@pytest.mark.parametrize('values', [[1, 2, 2, 3], [3, 3]])
def test_intersection_duplicates(values: list):
    a = Index(values)
    b = Index([3, 3])
    result = a.intersection(b)
    expected = Index([3])
    tm.assert_index_equal(result, expected)

class TestSetOps:

    @pytest.mark.parametrize('case', [0.5, 'xxx'])
    @pytest.mark.parametrize('method', ['intersection', 'union', 'difference', 'symmetric_difference'])
    def test_set_ops_error_cases(self, case: str, method: str, index: Index):
        msg = 'Input must be Index or array-like'
        with pytest.raises(TypeError, match=msg):
            getattr(index, method)(case)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_intersection_base(self, index: Index):
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
    def test_union_base(self, index: Index):
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
    def test_difference_base(self, sort: bool, index: Index):
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
    def test_symmetric_difference(self, index: Index, using_infer_string: bool, request: pytest.FixtureRequest):
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
    def test_corner_union(self, index_flat: Index, fname: str, sname: str, expected_name: str):
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
    def test_union_unequal(self, index_flat: Index, fname: str, sname: str, expected_name: str):
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
    def test_corner_intersect(self, index_flat: Index, fname: str, sname: str, expected_name: str):
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

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_intersection_name_retention_with_nameless(self, index: Index):
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

    def test_difference_preserves_type_empty(self, index: Index, sort: bool):
        if not index.is_unique:
            pytest.skip('Not relevant since index is not unique')
        result = index.difference(index, sort=sort)
        expected = index[:0]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_name_retention_equals(self, index: Index, names: list):
        if isinstance(index, MultiIndex):
            names = [[x] * index.nlevels for x in names]
        index = index.rename(names[0])
        other = index.rename(names[1])
        assert index.equals(other)
        result = index.difference(other)
        expected = index[:0].rename(names[2])
        tm.assert_index_equal(result, expected)

    def test_intersection_difference_match_empty(self, index: Index, sort: bool):
        if not index.is_unique:
            pytest.skip('Not relevant because index is not unique')
        inter = index.intersection(index[:0])
        diff = index.difference(index, sort=sort)
        tm.assert_index_equal(inter, diff, exact=True)

@pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.parametrize('method', ['intersection', 'union', 'difference', 'symmetric_difference'])
def test_setop_with_categorical(index_flat: Index, sort: bool, method: str):
    index = index_flat
    other = index.astype('category')
    exact = 'equiv' if isinstance(index, RangeIndex) else True
    result = getattr(index, method)(other, sort=sort)
    expected = getattr(index, method)(index, sort=sort)
    tm.assert_index_equal(result, expected, exact=exact)
    result = getattr(index, method)(other[:5], sort=sort)
    expected = getattr(index, method)(index[:5], sort=sort)
    tm.assert_index_equal(result, expected, exact=exact)

def test_intersection_duplicates_all_indexes(index: Index):
    if index.empty:
        pytest.skip('Not relevant for empty Index')
    idx = index
    idx_non_unique = idx[[0, 0, 1, 2]]
    assert idx.intersection(idx_non_unique).equals(idx_non_unique.intersection(idx))
    assert idx.intersection(idx_non_unique).is_unique

def test_union_duplicate_index_subsets_of_each_other(any_dtype_for_small_pos_integer_indexes: str):
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 2, 2, 3], dtype=dtype)
    b = Index([3, 3, 4], dtype=dtype)
    expected = Index([1, 2, 2, 3, 3, 4], dtype=dtype)
    if isinstance(a, CategoricalIndex):
        expected = Index([1, 2, 2, 3, 3, 4])
    result = a.union(b)
    tm.assert_index_equal(result, expected)
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)

def test_union_with_duplicate_index_and_non_monotonic(any_dtype_for_small_pos_integer_indexes: str):
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 0, 0], dtype=dtype)
    b = Index([0, 1], dtype=dtype)
    expected = Index([0, 0, 1], dtype=dtype)
    result = a.union(b)
    tm.assert_index_equal(result, expected)
    result = b.union(a)
    tm.assert_index_equal(result, expected)

def test_union_duplicate_index_different_dtypes():
    a = Index([1, 2, 2, 3])
    b = Index(['1', '0', '0'])
    expected = Index([1, 2, 2, 3, '1', '0', '0'])
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)

def test_union_same_value_duplicated_in_both():
    a = Index([0, 0, 1])
    b = Index([0, 0, 1, 2])
    result = a.union(b)
    expected = Index([0, 0, 1, 2])
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('dup', [1, np.nan])
def test_union_nan_in_both(dup: int):
    a = Index([np.nan, 1, 2, 2])
    b = Index([np.nan, dup, 1, 2])
    result = a.union(b, sort=False)
    expected = Index([np.nan, dup, 1.0