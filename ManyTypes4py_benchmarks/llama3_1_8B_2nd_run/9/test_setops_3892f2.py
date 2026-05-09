from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import Index, Series
import pandas._testing as tm
from pandas.core.algorithms import safe_sort

def equal_contents(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)

class TestIndexSetOps:
    @pytest.mark.parametrize('method: str', ['union', 'intersection', 'difference', 'symmetric_difference'])
    def test_setops_sort_validation(self, method: str) -> None:
        idx1: Index = Index(['a', 'b'])
        idx2: Index = Index(['b', 'c'])
        with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
            getattr(idx1, method)(idx2, sort=2)
        getattr(idx1, method)(idx2, sort=True)

    def test_setops_preserve_object_dtype(self) -> None:
        idx: Index = Index([1, 2, 3], dtype=object)
        result: Index = idx.intersection(idx[1:])
        expected: Index = idx[1:]
        tm.assert_index_equal(result, expected)
        result = idx.intersection(idx[1:][::-1])
        tm.assert_index_equal(result, expected)
        result = idx._union(idx[1:], sort=None)
        expected = idx
        tm.assert_numpy_array_equal(result, expected.values)
        result = idx.union(idx[1:], sort=None)
        tm.assert_index_equal(result, expected)
        result = idx._union(idx[1:][::-1], sort=None)
        tm.assert_numpy_array_equal(result, expected.values)
        result = idx.union(idx[1:][::-1], sort=None)
        tm.assert_index_equal(result, expected)

    def test_union_base(self) -> None:
        index: Index = Index([0, 'a', 1, 'b', 2, 'c'])
        first: Index = index[3:]
        second: Index = index[:5]
        result: Index = first.union(second)
        expected: Index = Index([0, 1, 2, 'a', 'b', 'c'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('klass: type', [np.array, Series, list])
    def test_union_different_type_base(self, klass: type) -> None:
        index: Index = Index([0, 'a', 1, 'b', 2, 'c'])
        first: Index = index[3:]
        second: Index = index[:5]
        result: Index = first.union(klass(second.values))
        assert equal_contents(result, index)

    def test_union_sort_other_incomparable(self) -> None:
        idx: Index = Index([1, pd.Timestamp('2000')])
        with tm.assert_produces_warning(RuntimeWarning, match='not supported between'):
            result: Index = idx.union(idx[:1])
        tm.assert_index_equal(result, idx)
        with tm.assert_produces_warning(RuntimeWarning, match='not supported between'):
            result = idx.union(idx[:1], sort=None)
        tm.assert_index_equal(result, idx)
        result: Index = idx.union(idx[:1], sort=False)
        tm.assert_index_equal(result, idx)

    def test_union_sort_other_incomparable_true(self) -> None:
        idx: Index = Index([1, pd.Timestamp('2000')])
        with pytest.raises(TypeError, match='.*'):
            idx.union(idx[:1], sort=True)

    def test_intersection_equal_sort_true(self) -> None:
        idx: Index = Index(['c', 'a', 'b'])
        sorted_: Index = Index(['a', 'b', 'c'])
        tm.assert_index_equal(idx.intersection(idx, sort=True), sorted_)

    def test_intersection_base(self, sort: bool) -> None:
        index: Index = Index([0, 'a', 1, 'b', 2, 'c'])
        first: Index = index[:5]
        second: Index = index[:3]
        expected: Index = Index([0, 1, 'a']) if sort is None else Index([0, 'a', 1])
        result: Index = first.intersection(second, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('klass: type', [np.array, Series, list])
    def test_intersection_different_type_base(self, klass: type, sort: bool) -> None:
        index: Index = Index([0, 'a', 1, 'b', 2, 'c'])
        first: Index = index[:5]
        second: Index = index[:3]
        result: Index = first.intersection(klass(second.values), sort=sort)
        assert equal_contents(result, second)

    def test_intersection_nosort(self) -> None:
        result: Index = Index(['c', 'b', 'a']).intersection(['b', 'a'])
        expected: Index = Index(['b', 'a'])
        tm.assert_index_equal(result, expected)

    def test_intersection_equal_sort(self) -> None:
        idx: Index = Index(['c', 'a', 'b'])
        tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
        tm.assert_index_equal(idx.intersection(idx, sort=None), idx)

    def test_intersection_str_dates(self, sort: bool) -> None:
        dt_dates: list[datetime] = [datetime(2012, 2, 9), datetime(2012, 2, 22)]
        i1: Index = Index(dt_dates, dtype=object)
        i2: Index = Index(['aa'], dtype=object)
        result: Index = i2.intersection(i1, sort=sort)
        assert len(result) == 0

    @pytest.mark.parametrize('index2: list[str]', [['B', 'D'], ['B', 'D', 'A']])
    @pytest.mark.parametrize('expected_arr: list[str]', [['B'], ['A', 'B']])
    def test_intersection_non_monotonic_non_unique(self, index2: list[str], expected_arr: list[str], sort: bool) -> None:
        index1: Index = Index(['A', 'B', 'A', 'C'])
        expected: Index = Index(expected_arr)
        result: Index = index1.intersection(Index(index2), sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_difference_base(self, sort: bool) -> None:
        index: Index = Index([0, 'a', 1, 'b', 2, 'c'])
        first: Index = index[:4]
        second: Index = index[3:]
        result: Index = first.difference(second, sort)
        expected: Index = Index([0, 'a', 1])
        if sort is None:
            expected = Index(safe_sort(expected))
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self) -> None:
        index: Index = Index([0, 'a', 1, 'b', 2, 'c'])
        first: Index = index[:4]
        second: Index = index[3:]
        result: Index = first.symmetric_difference(second)
        expected: Index = Index([0, 1, 2, 'a', 'c'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('method: str', ['intersection', 'union'])
    @pytest.mark.parametrize('expected: np.ndarray', [np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B')], dtype=[('num', int), ('let', 'S1')]), np.array([(1, 'A'), (1, 'B'), (1, 'C'), (2, 'A'), (2, 'B'), (2, 'C')], dtype=[('num', int), ('let', 'S1')])])
    @pytest.mark.parametrize('sort: bool', [False, None])
    def test_tuple_union_bug(self, method: str, expected: np.ndarray, sort: bool) -> None:
        index1: Index = Index(np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B')], dtype=[('num', int), ('let', 'S1')]))
        index2: Index = Index(np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B'), (1, 'C'), (2, 'C')], dtype=[('num', int), ('let', 'S1')]))
        result: Index = getattr(index1, method)(index2, sort=sort)
        assert result.ndim == 1
        expected = Index(expected)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('first_list: list[str]', [['b', 'a'], []])
    @pytest.mark.parametrize('second_list: list[str]', [['a', 'b'], []])
    @pytest.mark.parametrize('first_name: str', ['A', None])
    @pytest.mark.parametrize('second_name: str', ['B', None])
    @pytest.mark.parametrize('expected_name: str | None', [None, None])
    def test_union_name_preservation(self, first_list: list[str], second_list: list[str], first_name: str, second_name: str, expected_name: str | None, sort: bool) -> None:
        expected_dtype: str = object if not first_list or not second_list else 'str'
        first: Index = Index(first_list, name=first_name)
        second: Index = Index(second_list, name=second_name)
        union: Index = first.union(second, sort=sort)
        vals: set[str] = set(first_list).union(second_list)
        if sort is None and len(first_list) > 0 and (len(second_list) > 0):
            expected: Index = Index(sorted(vals), name=expected_name)
            tm.assert_index_equal(union, expected)
        else:
            expected: Index = Index(vals, name=expected_name, dtype=expected_dtype)
            tm.assert_index_equal(union.sort_values(), expected.sort_values())

    @pytest.mark.parametrize('diff_type: str', ['difference', 'symmetric_difference'])
    @pytest.mark.parametrize('expected: list', [['1', 'B'], ['1', '2', 'B', 'C']])
    def test_difference_object_type(self, diff_type: str, expected: list) -> None:
        idx1: Index = Index([0, 1, 'A', 'B'])
        idx2: Index = Index([0, 2, 'A', 'C'])
        result: Index = getattr(idx1, diff_type)(idx2)
        expected = Index(expected)
        tm.assert_index_equal(result, expected)
