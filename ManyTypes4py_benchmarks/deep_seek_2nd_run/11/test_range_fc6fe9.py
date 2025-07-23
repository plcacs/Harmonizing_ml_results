import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import Index, RangeIndex
import pandas._testing as tm
from pandas.core.indexes.range import min_fitting_element
from typing import Any, List, Tuple, Union, cast

class TestRangeIndex:

    @pytest.fixture
    def simple_index(self) -> RangeIndex:
        return RangeIndex(start=0, stop=20, step=2)

    def test_constructor_unwraps_index(self) -> None:
        result: RangeIndex = RangeIndex(1, 3)
        expected: np.ndarray = np.array([1, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_can_hold_identifiers(self, simple_index: RangeIndex) -> None:
        idx: RangeIndex = simple_index
        key: int = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_too_many_names(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        with pytest.raises(ValueError, match='^Length'):
            index.names = ['roger', 'harold']

    @pytest.mark.parametrize('index, start, stop, step', [(RangeIndex(5), 0, 5, 1), (RangeIndex(0, 5), 0, 5, 1), (RangeIndex(5, step=2), 0, 5, 2), (RangeIndex(1, 5, 2), 1, 5, 2)])
    def test_start_stop_step_attrs(self, index: RangeIndex, start: int, stop: int, step: int) -> None:
        assert index.start == start
        assert index.stop == stop
        assert index.step == step

    def test_copy(self) -> None:
        i: RangeIndex = RangeIndex(5, name='Foo')
        i_copy: RangeIndex = i.copy()
        assert i_copy is not i
        assert i_copy.identical(i)
        assert i_copy._range == range(0, 5, 1)
        assert i_copy.name == 'Foo'

    def test_repr(self) -> None:
        i: RangeIndex = RangeIndex(5, name='Foo')
        result: str = repr(i)
        expected: str = "RangeIndex(start=0, stop=5, step=1, name='Foo')"
        assert result == expected
        result: RangeIndex = eval(result)
        tm.assert_index_equal(result, i, exact=True)
        i = RangeIndex(5, 0, -1)
        result = repr(i)
        expected = 'RangeIndex(start=5, stop=0, step=-1)'
        assert result == expected
        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

    def test_insert(self) -> None:
        idx: RangeIndex = RangeIndex(5, name='Foo')
        result: RangeIndex = idx[1:4]
        tm.assert_index_equal(idx[0:4], result.insert(0, idx[0]), exact='equiv')
        expected: Index = Index([0, np.nan, 1, 2, 3, 4], dtype=np.float64)
        for na in [np.nan, None, pd.NA]:
            result = RangeIndex(5).insert(1, na)
            tm.assert_index_equal(result, expected)
        result = RangeIndex(5).insert(1, pd.NaT)
        expected = Index([0, pd.NaT, 1, 2, 3, 4], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_insert_edges_preserves_rangeindex(self) -> None:
        idx: Index = Index(range(4, 9, 2))
        result: Index = idx.insert(0, 2)
        expected: Index = Index(range(2, 9, 2))
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.insert(3, 10)
        expected = Index(range(4, 11, 2))
        tm.assert_index_equal(result, expected, exact=True)

    def test_insert_middle_preserves_rangeindex(self) -> None:
        idx: Index = Index(range(0, 3, 2))
        result: Index = idx.insert(1, 1)
        expected: Index = Index(range(3))
        tm.assert_index_equal(result, expected, exact=True)
        idx = idx * 2
        result = idx.insert(1, 2)
        expected = expected * 2
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete(self) -> None:
        idx: RangeIndex = RangeIndex(5, name='Foo')
        expected: RangeIndex = idx[1:]
        result: RangeIndex = idx.delete(0)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name
        expected = idx[:-1]
        result = idx.delete(-1)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name
        msg: str = 'index 5 is out of bounds for axis 0 with size 5'
        with pytest.raises((IndexError, ValueError), match=msg):
            result = idx.delete(len(idx))

    def test_delete_preserves_rangeindex(self) -> None:
        idx: Index = Index(range(2), name='foo')
        result: Index = idx.delete([1])
        expected: Index = Index(range(1), name='foo')
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(1)
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_middle(self) -> None:
        idx: Index = Index(range(3), name='foo')
        result: Index = idx.delete(1)
        expected: Index = idx[::2]
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(-2)
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_list_at_end(self) -> None:
        idx: RangeIndex = RangeIndex(0, 6, 1)
        loc: List[int] = [2, 3, 4, 5]
        result: RangeIndex = idx.delete(loc)
        expected: RangeIndex = idx[:2]
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_list_middle(self) -> None:
        idx: RangeIndex = RangeIndex(0, 6, 1)
        loc: List[int] = [1, 2, 3, 4]
        result: RangeIndex = idx.delete(loc)
        expected: RangeIndex = RangeIndex(0, 6, 5)
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_all_preserves_rangeindex(self) -> None:
        idx: RangeIndex = RangeIndex(0, 6, 1)
        loc: List[int] = [0, 1, 2, 3, 4, 5]
        result: RangeIndex = idx.delete(loc)
        expected: RangeIndex = idx[:0]
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_not_preserving_rangeindex(self) -> None:
        idx: RangeIndex = RangeIndex(0, 6, 1)
        loc: List[int] = [0, 3, 5]
        result: Index = idx.delete(loc)
        expected: Index = Index([1, 2, 4])
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_view(self) -> None:
        i: RangeIndex = RangeIndex(0, name='Foo')
        i_view: RangeIndex = i.view()
        assert i_view.name == 'Foo'
        i_view: np.ndarray = i.view('i8')
        tm.assert_numpy_array_equal(i.values, i_view)

    def test_dtype(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        assert index.dtype == np.int64

    def test_cache(self) -> None:
        idx: RangeIndex = RangeIndex(0, 100, 10)
        assert idx._cache == {}
        repr(idx)
        assert idx._cache == {}
        str(idx)
        assert idx._cache == {}
        idx.get_loc(20)
        assert idx._cache == {}
        90 in idx
        assert idx._cache == {}
        91 in idx
        assert idx._cache == {}
        idx.all()
        assert idx._cache == {}
        idx.any()
        assert idx._cache == {}
        for _ in idx:
            pass
        assert idx._cache == {}
        df: pd.DataFrame = pd.DataFrame({'a': range(10)}, index=idx)
        str(df)
        assert idx._cache == {}
        df.loc[50]
        assert idx._cache == {}
        with pytest.raises(KeyError, match='51'):
            df.loc[51]
        assert idx._cache == {}
        df.loc[10:50]
        assert idx._cache == {}
        df.iloc[5:10]
        assert idx._cache == {}
        idx.take([3, 0, 1])
        assert '_data' not in idx._cache
        df.loc[[50]]
        assert '_data' not in idx._cache
        df.iloc[[5, 6, 7, 8, 9]]
        assert '_data' not in idx._cache
        idx._data
        assert isinstance(idx._data, np.ndarray)
        assert idx._data is idx._data
        assert '_data' in idx._cache
        expected: np.ndarray = np.arange(0, 100, 10, dtype='int64')
        tm.assert_numpy_array_equal(idx._cache['_data'], expected)

    def test_is_monotonic(self) -> None:
        index: RangeIndex = RangeIndex(0, 20, 2)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is False
        index = RangeIndex(4, 0, -1)
        assert index.is_monotonic_increasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is True
        index = RangeIndex(1, 2)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True
        index = RangeIndex(2, 1)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True
        index = RangeIndex(1, 1)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    @pytest.mark.parametrize('left,right', [(RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)), (RangeIndex(0), RangeIndex(1, -1, 3)), (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)), (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2))])
    def test_equals_range(self, left: RangeIndex, right: RangeIndex) -> None:
        assert left.equals(right)
        assert right.equals(left)

    def test_logical_compat(self, simple_index: RangeIndex) -> None:
        idx: RangeIndex = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        i: Index = Index(index.copy())
        assert i.identical(index)
        if isinstance(index, RangeIndex):
            return
        same_values_different_type: Index = Index(i, dtype=object)
        assert not i.identical(same_values_different_type)
        i = index.copy(dtype=object)
        i = i.rename('foo')
        same_values: Index = Index(i, dtype=object)
        assert same_values.identical(index.copy(dtype=object))
        assert not i.identical(index)
        assert Index(same_values, name='foo', dtype=object).identical(i)
        assert not index.copy(dtype=object).identical(index.copy(dtype='int64'))

    def test_nbytes(self) -> None:
        idx: RangeIndex = RangeIndex(0, 1000)
        assert idx.nbytes < Index(idx._values).nbytes / 10
        i2: RangeIndex = RangeIndex(0, 10)
        assert idx.nbytes == i2.nbytes

    @pytest.mark.parametrize('start,stop,step', [('foo', 'bar', 'baz'), ('0', '1', '2')])
    def test_cant_or_shouldnt_cast(self, start: str, stop: str, step: str) -> None:
        msg: str = f'Wrong type {type(start)} for value {start}'
        with pytest.raises(TypeError, match=msg):
            RangeIndex(start, stop, step)

    def test_view_index(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        msg: str = 'Cannot change data-type for array of references.|Cannot change data-type for object array.|'
        with pytest.raises(TypeError, match=msg):
            index.view(Index)

    def test_prevent_casting(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        result: Index = index.astype('O')
        assert result.dtype == np.object_

    def test_repr_roundtrip(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        tm.assert_index_equal(eval(repr(index)), index)

    def test_slice_keep_name(self) -> None:
        idx: RangeIndex = RangeIndex(1, 2, name='asdf')
        assert idx.name == idx[1:].name

    @pytest.mark.parametrize('index', [RangeIndex(start=0, stop=20, step=2, name='foo'), RangeIndex(start=18, stop=-1, step=-2, name='bar')], ids=['index_inc', 'index_dec'])
    def test_has_duplicates(self, index: RangeIndex) -> None:
        assert index.is_unique
        assert not index.has_duplicates

    def test_extended_gcd(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        result: Tuple[int, int, int] = index._extended_gcd(6, 10)
        assert result[0] == result[1] * 6 + result[2] * 10
        assert 2 == result[0]
        result = index._extended_gcd(10, 6)
        assert 2 == result[1] * 10 + result[2] * 6
        assert 2 == result[0]

    def test_min_fitting_element(self) -> None:
        result: int = min_fitting_element(0, 2, 1)
        assert 2 == result
        result = min_fitting_element(1, 1, 1)
        assert 1 == result
        result = min_fitting_element(18, -2, 1)
        assert 2 == result
        result = min_fitting_element(5, -1, 1)
        assert 1 == result
        big_num: int = 500000000000000000000000
        result = min_fitting_element(5, 1, big_num)
        assert big_num == result

    def test_slice_specialised(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        index.name = 'foo'
        res: int = index[1]
        expected: int = 2
        assert res == expected
        res = index[-1]
        expected = 18
        assert res == expected
        index_slice: RangeIndex = index[:]
        expected = index
        tm.assert_index_equal(index_slice, expected)
        index_slice = index[7:10:2]
        expected = Index([14, 18], name='foo')
        tm.assert_index_equal(index_slice, expected, exact='equiv')
        index_slice = index[-1:-5:-2]
        expected = Index([18, 14], name='foo')
        tm.assert_index_equal(index_slice, expected, exact='equiv')
        index_slice = index[2:100:4]
        expected = Index([4, 12], name='foo')
        tm.assert_index_equal(index_slice, expected, exact='equiv')
        index_slice = index[::-1]
        expected = Index(index.values[::-1], name='foo')
        tm.assert_index_equal(index_slice, expected, exact='equiv')
        index_slice = index[-8::-1]
        expected = Index([4, 2, 0], name='foo')
        tm.assert_index_equal(index_slice, expected, exact='equiv')
       