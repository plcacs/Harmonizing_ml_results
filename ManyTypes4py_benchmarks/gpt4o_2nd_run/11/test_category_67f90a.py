import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import Categorical, CategoricalDtype
import pandas._testing as tm
from pandas.core.indexes.api import CategoricalIndex, Index
from typing import List, Any, Dict

class TestCategoricalIndex:

    @pytest.fixture
    def simple_index(self) -> CategoricalIndex:
        """
        Fixture that provides a CategoricalIndex.
        """
        return CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=False)

    def test_can_hold_identifiers(self) -> None:
        idx: CategoricalIndex = CategoricalIndex(list('aabbca'), categories=None, ordered=False)
        key: Any = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is True

    def test_insert(self, simple_index: CategoricalIndex) -> None:
        ci: CategoricalIndex = simple_index
        categories: List[str] = ci.categories
        result: CategoricalIndex = ci.insert(0, 'a')
        expected: CategoricalIndex = CategoricalIndex(list('aaabbca'), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)
        result = ci.insert(-1, 'a')
        expected = CategoricalIndex(list('aabbcaa'), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)
        result = CategoricalIndex([], categories=categories).insert(0, 'a')
        expected = CategoricalIndex(['a'], categories=categories)
        tm.assert_index_equal(result, expected, exact=True)
        expected = ci.astype(object).insert(0, 'd')
        result = ci.insert(0, 'd').astype(object)
        tm.assert_index_equal(result, expected, exact=True)
        expected = CategoricalIndex(['a', np.nan, 'a', 'b', 'c', 'b'])
        for na in (np.nan, pd.NaT, None):
            result = CategoricalIndex(list('aabcb')).insert(1, na)
            tm.assert_index_equal(result, expected)

    def test_insert_na_mismatched_dtype(self) -> None:
        ci: CategoricalIndex = CategoricalIndex([0, 1, 1])
        result: Index = ci.insert(0, pd.NaT)
        expected: Index = Index([pd.NaT, 0, 1, 1], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_delete(self, simple_index: CategoricalIndex) -> None:
        ci: CategoricalIndex = simple_index
        categories: List[str] = ci.categories
        result: CategoricalIndex = ci.delete(0)
        expected: CategoricalIndex = CategoricalIndex(list('abbca'), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)
        result = ci.delete(-1)
        expected = CategoricalIndex(list('aabbc'), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)
        with tm.external_error_raised((IndexError, ValueError)):
            ci.delete(10)

    @pytest.mark.parametrize('data, non_lexsorted_data', [[[1, 2, 3], [9, 0, 1, 2, 3]], [list('abc'), list('fabcd')]])
    def test_is_monotonic(self, data: List[Any], non_lexsorted_data: List[Any]) -> None:
        c: CategoricalIndex = CategoricalIndex(data)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False
        c = CategoricalIndex(data, ordered=True)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False
        c = CategoricalIndex(data, categories=reversed(data))
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is True
        c = CategoricalIndex(data, categories=reversed(data), ordered=True)
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is True
        reordered_data: List[Any] = [data[0], data[2], data[1]]
        c = CategoricalIndex(reordered_data, categories=reversed(data))
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is False
        categories: List[Any] = non_lexsorted_data
        c = CategoricalIndex(categories[:2], categories=categories)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False
        c = CategoricalIndex(categories[1:3], categories=categories)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

    def test_has_duplicates(self) -> None:
        idx: CategoricalIndex = CategoricalIndex([0, 0, 0], name='foo')
        assert idx.is_unique is False
        assert idx.has_duplicates is True
        idx = CategoricalIndex([0, 1], categories=[2, 3], name='foo')
        assert idx.is_unique is False
        assert idx.has_duplicates is True
        idx = CategoricalIndex([0, 1, 2, 3], categories=[1, 2, 3], name='foo')
        assert idx.is_unique is True
        assert idx.has_duplicates is False

    @pytest.mark.parametrize('data, categories, expected', [([1, 1, 1], [1, 2, 3], {'first': np.array([False, True, True]), 'last': np.array([True, True, False]), False: np.array([True, True, True])}), ([1, 1, 1], list('abc'), {'first': np.array([False, True, True]), 'last': np.array([True, True, False]), False: np.array([True, True, True])}), ([2, 'a', 'b'], list('abc'), {'first': np.zeros(shape=3, dtype=np.bool_), 'last': np.zeros(shape=3, dtype=np.bool_), False: np.zeros(shape=3, dtype=np.bool_)}), (list('abb'), list('abc'), {'first': np.array([False, False, True]), 'last': np.array([False, True, False]), False: np.array([False, True, True])})])
    def test_drop_duplicates(self, data: List[Any], categories: List[Any], expected: Dict[Any, np.ndarray]) -> None:
        idx: CategoricalIndex = CategoricalIndex(data, categories=categories, name='foo')
        for keep, e in expected.items():
            tm.assert_numpy_array_equal(idx.duplicated(keep=keep), e)
            e = idx[~e]
            result: CategoricalIndex = idx.drop_duplicates(keep=keep)
            tm.assert_index_equal(result, e)

    @pytest.mark.parametrize('data, categories, expected_data', [([1, 1, 1], [1, 2, 3], [1]), ([1, 1, 1], list('abc'), [np.nan]), ([1, 2, 'a'], [1, 2, 3], [1, 2, np.nan]), ([2, 'a', 'b'], list('abc'), [np.nan, 'a', 'b'])])
    def test_unique(self, data: List[Any], categories: List[Any], expected_data: List[Any], ordered: bool) -> None:
        dtype: CategoricalDtype = CategoricalDtype(categories, ordered=ordered)
        idx: CategoricalIndex = CategoricalIndex(data, dtype=dtype)
        expected: CategoricalIndex = CategoricalIndex(expected_data, dtype=dtype)
        tm.assert_index_equal(idx.unique(), expected)

    @pytest.mark.xfail(using_string_dtype(), reason="repr doesn't roundtrip")
    def test_repr_roundtrip(self) -> None:
        ci: CategoricalIndex = CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=True)
        str(ci)
        tm.assert_index_equal(eval(repr(ci)), ci, exact=True)
        str(ci)
        ci = CategoricalIndex(np.random.default_rng(2).integers(0, 5, size=100))
        str(ci)

    def test_isin(self) -> None:
        ci: CategoricalIndex = CategoricalIndex(list('aabca') + [np.nan], categories=['c', 'a', 'b'])
        tm.assert_numpy_array_equal(ci.isin(['c']), np.array([False, False, False, True, False, False]))
        tm.assert_numpy_array_equal(ci.isin(['c', 'a', 'b']), np.array([True] * 5 + [False]))
        tm.assert_numpy_array_equal(ci.isin(['c', 'a', 'b', np.nan]), np.array([True] * 6))
        result: np.ndarray = ci.isin(ci.set_categories(list('abcdefghi')))
        expected: np.ndarray = np.array([True] * 6)
        tm.assert_numpy_array_equal(result, expected)
        result = ci.isin(ci.set_categories(list('defghi')))
        expected = np.array([False] * 5 + [True])
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_overlapping_intervals(self) -> None:
        idx: pd.IntervalIndex = pd.IntervalIndex([pd.Interval(0, 2), pd.Interval(0, 1)])
        result: np.ndarray = CategoricalIndex(idx).isin(idx)
        expected: np.ndarray = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_identical(self) -> None:
        ci1: CategoricalIndex = CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=True)
        ci2: CategoricalIndex = CategoricalIndex(['a', 'b'], categories=['a', 'b', 'c'], ordered=True)
        assert ci1.identical(ci1)
        assert ci1.identical(ci1.copy())
        assert not ci1.identical(ci2)

    def test_ensure_copied_data(self) -> None:
        index: CategoricalIndex = CategoricalIndex(list('ab') * 5)
        result: CategoricalIndex = CategoricalIndex(index.values, copy=True)
        tm.assert_index_equal(index, result)
        assert not np.shares_memory(result._data._codes, index._data._codes)
        result = CategoricalIndex(index.values, copy=False)
        assert result._data._codes is index._data._codes

class TestCategoricalIndex2:

    def test_view_i8(self) -> None:
        ci: CategoricalIndex = CategoricalIndex(list('ab') * 50)
        msg: str = 'When changing to a larger dtype, its size must be a divisor'
        with pytest.raises(ValueError, match=msg):
            ci.view('i8')
        with pytest.raises(ValueError, match=msg):
            ci._data.view('i8')
        ci = ci[:-4]
        res: np.ndarray = ci.view('i8')
        expected: np.ndarray = ci._data.codes.view('i8')
        tm.assert_numpy_array_equal(res, expected)
        cat = ci._data
        tm.assert_numpy_array_equal(cat.view('i8'), expected)

    @pytest.mark.parametrize('dtype, engine_type', [(np.int8, libindex.Int8Engine), (np.int16, libindex.Int16Engine), (np.int32, libindex.Int32Engine), (np.int64, libindex.Int64Engine)])
    def test_engine_type(self, dtype: np.dtype, engine_type: type) -> None:
        if dtype != np.int64:
            num_uniques: Dict[np.dtype, int] = {np.int8: 1, np.int16: 128, np.int32: 32768}
            ci: CategoricalIndex = CategoricalIndex(range(num_uniques[dtype]))
        else:
            ci = CategoricalIndex(range(32768))
            arr: np.ndarray = ci.values._ndarray.astype('int64')
            NDArrayBacked.__init__(ci._data, arr, ci.dtype)
        assert np.issubdtype(ci.codes.dtype, dtype)
        assert isinstance(ci._engine, engine_type)

    @pytest.mark.parametrize('func,op_name', [(lambda idx: idx - idx, '__sub__'), (lambda idx: idx + idx, '__add__'), (lambda idx: idx - ['a', 'b'], '__sub__'), (lambda idx: idx + ['a', 'b'], '__add__'), (lambda idx: ['a', 'b'] - idx, '__rsub__'), (lambda idx: ['a', 'b'] + idx, '__radd__')])
    def test_disallow_addsub_ops(self, func: Any, op_name: str) -> None:
        idx: Index = Index(Categorical(['a', 'b']))
        cat_or_list: str = "'(Categorical|list)' and '(Categorical|list)'"
        msg: str = '|'.join([f'cannot perform {op_name} with this index type: CategoricalIndex', 'can only concatenate list', f'unsupported operand type\\(s\\) for [\\+-]: {cat_or_list}'])
        with pytest.raises(TypeError, match=msg):
            func(idx)

    def test_method_delegation(self) -> None:
        ci: CategoricalIndex = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
        result: CategoricalIndex = ci.set_categories(list('cab'))
        tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cab')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
        result = ci.rename_categories(list('efg'))
        tm.assert_index_equal(result, CategoricalIndex(list('ffggef'), categories=list('efg')))
        result = ci.rename_categories(lambda x: x.upper())
        tm.assert_index_equal(result, CategoricalIndex(list('AABBCA'), categories=list('CAB')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
        result = ci.add_categories(['d'])
        tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cabd')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
        result = ci.remove_categories(['c'])
        tm.assert_index_equal(result, CategoricalIndex(list('aabb') + [np.nan] + ['a'], categories=list('ab')))
        ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
        result = ci.as_unordered()
        tm.assert_index_equal(result, ci)
        ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
        result = ci.as_ordered()
        tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cabdef'), ordered=True))
        msg: str = 'cannot use inplace with CategoricalIndex'
        with pytest.raises(ValueError, match=msg):
            ci.set_categories(list('cab'), inplace=True)

    def test_remove_maintains_order(self) -> None:
        ci: CategoricalIndex = CategoricalIndex(list('abcdda'), categories=list('abcd'))
        result: CategoricalIndex = ci.reorder_categories(['d', 'c', 'b', 'a'], ordered=True)
        tm.assert_index_equal(result, CategoricalIndex(list('abcdda'), categories=list('dcba'), ordered=True))
        result = result.remove_categories(['c'])
        tm.assert_index_equal(result, CategoricalIndex(['a', 'b', np.nan, 'd', 'd', 'a'], categories=list('dba'), ordered=True))

def test_contains_rangeindex_categories_no_engine() -> None:
    ci: CategoricalIndex = CategoricalIndex(range(3))
    assert 2 in ci
    assert 5 not in ci
    assert '_engine' not in ci._cache
