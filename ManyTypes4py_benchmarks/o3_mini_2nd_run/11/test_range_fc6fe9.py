import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import Index, RangeIndex
import pandas._testing as tm
from pandas.core.indexes.range import min_fitting_element
from typing import Any, List, Sequence, Union


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
        key: Any = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_too_many_names(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        with pytest.raises(ValueError, match='^Length'):
            index.names = ['roger', 'harold']

    @pytest.mark.parametrize('index, start, stop, step', [
        (RangeIndex(5), 0, 5, 1),
        (RangeIndex(0, 5), 0, 5, 1),
        (RangeIndex(5, step=2), 0, 5, 2),
        (RangeIndex(1, 5, 2), 1, 5, 2)
    ])
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
        result = repr(i)
        result_eval: RangeIndex = eval(result)
        tm.assert_index_equal(result_eval, i, exact=True)
        i = RangeIndex(5, 0, -1)
        result = repr(i)
        expected = 'RangeIndex(start=5, stop=0, step=-1)'
        assert result == expected
        result_eval = eval(result)
        tm.assert_index_equal(result_eval, i, exact=True)

    def test_insert(self) -> None:
        idx: RangeIndex = RangeIndex(5, name='Foo')
        result: Index = idx[1:4]
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
        result: Index = idx.delete(0)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name
        expected = idx[:-1]
        result = idx.delete(-1)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name
        msg: str = 'index 5 is out of bounds for axis 0 with size 5'
        with pytest.raises((IndexError, ValueError), match=msg):
            _ = idx.delete(len(idx))

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
        result: Index = idx.delete(loc)
        expected: RangeIndex = idx[:2]
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_list_middle(self) -> None:
        idx: RangeIndex = RangeIndex(0, 6, 1)
        loc: List[int] = [1, 2, 3, 4]
        result: Index = idx.delete(loc)
        expected: RangeIndex = RangeIndex(0, 6, 5)
        tm.assert_index_equal(result, expected, exact=True)
        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_all_preserves_rangeindex(self) -> None:
        idx: RangeIndex = RangeIndex(0, 6, 1)
        loc: List[int] = [0, 1, 2, 3, 4, 5]
        result: Index = idx.delete(loc)
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
        i_view: Any = i.view()
        assert i_view.name == 'Foo'
        i_view = i.view('i8')
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
            _ = df.loc[51]
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
        _ = idx._data
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

    @pytest.mark.parametrize('left,right', [
        (RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)),
        (RangeIndex(0), RangeIndex(1, -1, 3)),
        (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)),
        (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2))
    ])
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

    @pytest.mark.parametrize('start,stop,step', [
        ('foo', 'bar', 'baz'),
        ('0', '1', '2')
    ])
    def test_cant_or_shouldnt_cast(self, start: Any, stop: Any, step: Any) -> None:
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

    @pytest.mark.parametrize('index', [
        RangeIndex(start=0, stop=20, step=2, name='foo'),
        RangeIndex(start=18, stop=-1, step=-2, name='bar')
    ], ids=['index_inc', 'index_dec'])
    def test_has_duplicates(self, index: RangeIndex) -> None:
        assert index.is_unique
        assert not index.has_duplicates

    def test_extended_gcd(self, simple_index: RangeIndex) -> None:
        index: RangeIndex = simple_index
        result: Sequence[int] = index._extended_gcd(6, 10)
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
        res: Any = index[1]
        expected: int = 2
        assert res == expected
        res = index[-1]
        expected = 18
        assert res == expected
        index_slice: Index = index[:]
        expected_index: RangeIndex = index
        tm.assert_index_equal(index_slice, expected_index)
        index_slice = index[7:10:2]
        expected_index = Index([14, 18], name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')
        index_slice = index[-1:-5:-2]
        expected_index = Index([18, 14], name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')
        index_slice = index[2:100:4]
        expected_index = Index([4, 12], name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')
        index_slice = index[::-1]
        expected_index = Index(index.values[::-1], name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')
        index_slice = index[-8::-1]
        expected_index = Index([4, 2, 0], name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')
        index_slice = index[-40::-1]
        expected_index = Index(np.array([], dtype=np.int64), name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')
        index_slice = index[40::-1]
        expected_index = Index(index.values[40::-1], name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')
        index_slice = index[10::-1]
        expected_index = Index(index.values[::-1], name='foo')
        tm.assert_index_equal(index_slice, expected_index, exact='equiv')

    @pytest.mark.parametrize('step', set(range(-5, 6)) - {0})
    def test_len_specialised(self, step: int) -> None:
        if step > 0:
            start, stop = (0, 5)
        else:
            start, stop = (5, 0)
        arr: np.ndarray = np.array(range(start, stop, step))
        index: RangeIndex = RangeIndex(start, stop, step)
        assert len(index) == len(arr)
        index = RangeIndex(stop, start, step)
        assert len(index) == 0

    @pytest.mark.parametrize(
        'indices, expected',
        [
            ([RangeIndex(1, 12, 5)], RangeIndex(1, 12, 5)),
            ([RangeIndex(0, 6, 4)], RangeIndex(0, 6, 4)),
            ([RangeIndex(1, 3), RangeIndex(3, 7)], RangeIndex(1, 7)),
            ([RangeIndex(1, 5, 2), RangeIndex(5, 6)], RangeIndex(1, 6, 2)),
            ([RangeIndex(1, 3, 2), RangeIndex(4, 7, 3)], RangeIndex(1, 7, 3)),
            ([RangeIndex(-4, 3, 2), RangeIndex(4, 7, 2)], RangeIndex(-4, 7, 2)),
            ([RangeIndex(-4, -8), RangeIndex(-8, -12)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, -4)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(-4, -2), RangeIndex(3, 5)], Index([-4, -3, 3, 4])),
            ([RangeIndex(-2), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(2), RangeIndex(2)], Index([0, 1, 0, 1])),
            ([RangeIndex(2), RangeIndex(2, 5), RangeIndex(5, 8, 4)], RangeIndex(0, 6)),
            ([RangeIndex(2), RangeIndex(3, 5), RangeIndex(5, 8, 4)], Index([0, 1, 3, 4, 5])),
            ([RangeIndex(-2, 2), RangeIndex(2, 5), RangeIndex(5, 8, 4)], RangeIndex(-2, 6)),
            ([RangeIndex(3), Index([-1, 3, 15])], Index([0, 1, 2, -1, 3, 15])),
            ([RangeIndex(3), Index([-1, 3.1, 15.0])], Index([0, 1, 2, -1, 3.1, 15.0])),
            ([RangeIndex(3), Index(['a', None, 14])], Index([0, 1, 2, 'a', None, 14])),
            ([RangeIndex(3, 1), Index(['a', None, 14])], Index(['a', None, 14]))
        ]
    )
    def test_append(self, indices: List[Union[RangeIndex, Index]], expected: Union[RangeIndex, Index]) -> None:
        result: Union[RangeIndex, Index] = indices[0].append(indices[1:])
        tm.assert_index_equal(result, expected, exact=True)
        if len(indices) == 2:
            result2: Union[RangeIndex, Index] = indices[0].append(indices[1])
            tm.assert_index_equal(result2, expected, exact=True)

    def test_engineless_lookup(self) -> None:
        idx: RangeIndex = RangeIndex(2, 10, 3)
        assert idx.get_loc(5) == 1
        tm.assert_numpy_array_equal(idx.get_indexer([2, 8]), ensure_platform_int(np.array([0, 2])))
        with pytest.raises(KeyError, match='3'):
            _ = idx.get_loc(3)
        assert '_engine' not in idx._cache
        with pytest.raises(KeyError, match="'a'"):
            _ = idx.get_loc('a')
        assert '_engine' not in idx._cache

    @pytest.mark.parametrize('ri', [
        RangeIndex(0, -1, -1),
        RangeIndex(0, 1, 1),
        RangeIndex(1, 3, 2),
        RangeIndex(0, -1, -2),
        RangeIndex(-3, -5, -2)
    ])
    def test_append_len_one(self, ri: RangeIndex) -> None:
        result: Union[RangeIndex, Index] = ri.append([])
        tm.assert_index_equal(result, ri, exact=True)

    @pytest.mark.parametrize('base', [RangeIndex(0, 2), Index([0, 1])])
    def test_isin_range(self, base: Union[RangeIndex, Index]) -> None:
        values: RangeIndex = RangeIndex(0, 1)
        result: np.ndarray = base.isin(values)
        expected: np.ndarray = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_sort_values_key(self) -> None:
        sort_order: dict = {8: 2, 6: 0, 4: 8, 2: 10, 0: 12}
        values: RangeIndex = RangeIndex(0, 10, 2)
        result: Index = values.sort_values(key=lambda x: x.map(sort_order))
        expected: Index = Index([6, 8, 4, 2, 0], dtype='int64')
        tm.assert_index_equal(result, expected, check_exact=True)
        ser: pd.Series = values.to_series()
        result2: pd.Series = ser.sort_values(key=lambda x: x.map(sort_order))
        tm.assert_series_equal(result2, expected.to_series(), check_exact=True)

    def test_range_index_rsub_by_const(self) -> None:
        result: RangeIndex = 3 - RangeIndex(0, 4, 1)
        expected: RangeIndex = RangeIndex(3, -1, -1)
        tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize('rng, decimals', [
    (range(5), 0),
    (range(5), 2),
    (range(10, 30, 10), -1),
    (range(30, 10, -10), -1)
])
def test_range_round_returns_rangeindex(rng: range, decimals: int) -> None:
    ri: RangeIndex = RangeIndex(rng)
    expected: RangeIndex = ri.copy()
    result: Union[RangeIndex, Index] = ri.round(decimals=decimals)
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize('rng, decimals', [
    (range(10, 30, 1), -1),
    (range(30, 10, -1), -1),
    (range(11, 14), -10)
])
def test_range_round_returns_index(rng: range, decimals: int) -> None:
    ri: RangeIndex = RangeIndex(rng)
    expected: Index = Index(list(rng)).round(decimals=decimals)
    result: Union[RangeIndex, Index] = ri.round(decimals=decimals)
    tm.assert_index_equal(result, expected, exact=True)


def test_reindex_1_value_returns_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(0, 10, 2, name='foo')
    result, result_indexer = ri.reindex([2])
    expected: RangeIndex = RangeIndex(2, 4, 2, name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    expected_indexer: np.ndarray = np.array([1], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_empty_returns_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(0, 10, 2, name='foo')
    result, result_indexer = ri.reindex([])
    expected: RangeIndex = RangeIndex(0, 0, 2, name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    expected_indexer: np.ndarray = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_insert_empty_0_loc() -> None:
    ri: RangeIndex = RangeIndex(0, step=10, name='foo')
    result: RangeIndex = ri.insert(0, 5)
    expected: RangeIndex = RangeIndex(5, 15, 10, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


def test_append_non_rangeindex_return_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(1)
    result: RangeIndex = ri.append(Index([1]))
    expected: RangeIndex = RangeIndex(2)
    tm.assert_index_equal(result, expected, exact=True)


def test_append_non_rangeindex_return_index() -> None:
    ri: RangeIndex = RangeIndex(1)
    result: Index = ri.append(Index([1, 3, 4]))
    expected: Index = Index([0, 1, 3, 4])
    tm.assert_index_equal(result, expected, exact=True)


def test_reindex_returns_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(2, name='foo')
    result, result_indexer = ri.reindex([1, 2, 3])
    expected: RangeIndex = RangeIndex(1, 4, name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    expected_indexer: np.ndarray = np.array([1, -1, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_returns_index() -> None:
    ri: RangeIndex = RangeIndex(4, name='foo')
    result, result_indexer = ri.reindex([0, 1, 3])
    expected: Index = Index([0, 1, 3], name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    expected_indexer: np.ndarray = np.array([0, 1, 3], dtype=np.intp)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_take_return_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(5, name='foo')
    result: RangeIndex = ri.take([])
    expected: RangeIndex = RangeIndex(0, name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    result = ri.take([3, 4])
    expected = RangeIndex(3, 5, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize('rng, exp_rng', [
    (range(5), range(3, 4)),
    (range(0, -10, -2), range(-6, -8, -2)),
    (range(0, 10, 2), range(6, 8, 2))
])
def test_take_1_value_returns_rangeindex(rng: range, exp_rng: range) -> None:
    ri: RangeIndex = RangeIndex(rng, name='foo')
    result: RangeIndex = ri.take([3])
    expected: RangeIndex = RangeIndex(exp_rng, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


def test_append_one_nonempty_preserve_step() -> None:
    expected: RangeIndex = RangeIndex(0, -1, -1)
    result: RangeIndex = RangeIndex(0).append([expected])
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_all_true() -> None:
    ri: RangeIndex = RangeIndex(3, name='foo')
    expected: RangeIndex = ri.copy()
    result: RangeIndex = ri[[True] * 3]
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_all_false() -> None:
    ri: RangeIndex = RangeIndex(3, name='foo')
    result: RangeIndex = ri[[False] * 3]
    expected: RangeIndex = RangeIndex(0, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_returns_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(3, name='foo')
    result: RangeIndex = ri[[False, True, True]]
    expected: RangeIndex = RangeIndex(1, 3, name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    result = ri[[True, False, True]]
    expected = RangeIndex(0, 3, 2, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_boolmask_returns_index() -> None:
    ri: RangeIndex = RangeIndex(4, name='foo')
    result: Index = ri[[True, True, False, True]]
    expected: Index = Index([0, 1, 3], name='foo')
    tm.assert_index_equal(result, expected)


def test_getitem_boolmask_wrong_length() -> None:
    ri: RangeIndex = RangeIndex(4, name='foo')
    with pytest.raises(IndexError, match='Boolean index has wrong length'):
        _ = ri[[True]]


def test_pos_returns_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(2, name='foo')
    expected: RangeIndex = ri.copy()
    result: RangeIndex = +ri
    tm.assert_index_equal(result, expected, exact=True)


def test_neg_returns_rangeindex() -> None:
    ri: RangeIndex = RangeIndex(2, name='foo')
    result: RangeIndex = -ri
    expected: RangeIndex = RangeIndex(0, -2, -1, name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    ri = RangeIndex(-2, 2, name='foo')
    result = -ri
    expected = RangeIndex(2, -2, -1, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize('rng, exp_rng', [
    (range(0), range(0)),
    (range(10), range(10)),
    (range(-2, 1, 1), range(2, -1, -1)),
    (range(0, -10, -1), range(0, 10, 1))
])
def test_abs_returns_rangeindex(rng: range, exp_rng: range) -> None:
    ri: RangeIndex = RangeIndex(rng, name='foo')
    expected: RangeIndex = RangeIndex(exp_rng, name='foo')
    result: Union[RangeIndex, Index] = abs(ri)
    tm.assert_index_equal(result, expected, exact=True)


def test_abs_returns_index() -> None:
    ri: RangeIndex = RangeIndex(-2, 2, name='foo')
    result: Index = abs(ri)
    expected: Index = Index([2, 1, 0, 1], name='foo')
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize('rng', [
    range(0), range(5), range(0, -5, -1), range(-2, 2, 1),
    range(2, -2, -2), range(0, 5, 2)
])
def test_invert_returns_rangeindex(rng: range) -> None:
    ri: RangeIndex = RangeIndex(rng, name='foo')
    result: RangeIndex = ~ri
    assert isinstance(result, RangeIndex)
    expected: Index = ~Index(list(rng), name='foo')
    tm.assert_index_equal(result, expected, exact=False)


@pytest.mark.parametrize('rng', [
    range(0, 5, 1), range(0, 5, 2), range(10, 15, 1),
    range(10, 5, -1), range(10, 5, -2), range(5, 0, -1)
])
@pytest.mark.parametrize('meth', ['argmax', 'argmin'])
def test_arg_min_max(rng: range, meth: str) -> None:
    ri: RangeIndex = RangeIndex(rng)
    idx: Index = Index(list(rng))
    assert getattr(ri, meth)() == getattr(idx, meth)()


@pytest.mark.parametrize('meth', ['argmin', 'argmax'])
def test_empty_argmin_argmax_raises(meth: str) -> None:
    with pytest.raises(ValueError, match=f'attempt to get {meth} of an empty sequence'):
        getattr(RangeIndex(0), meth)()


def test_getitem_integers_return_rangeindex() -> None:
    result: RangeIndex = RangeIndex(0, 10, 2, name='foo')[[0, -1]]
    expected: RangeIndex = RangeIndex(start=0, stop=16, step=8, name='foo')
    tm.assert_index_equal(result, expected, exact=True)
    result = RangeIndex(0, 10, 2, name='foo')[[3]]
    expected = RangeIndex(start=6, stop=8, step=2, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_empty_return_rangeindex() -> None:
    result: RangeIndex = RangeIndex(0, 10, 2, name='foo')[[]]
    expected: RangeIndex = RangeIndex(start=0, stop=0, step=1, name='foo')
    tm.assert_index_equal(result, expected, exact=True)


def test_getitem_integers_return_index() -> None:
    result: Index = RangeIndex(0, 10, 2, name='foo')[[0, 1, -1]]
    expected: Index = Index([0, 2, 8], dtype='int64', name='foo')
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize('sort, dropna, ascending, normalize, rng',
                         [(True, True, True, True, range(3)),
                          (False, False, False, False, range(0)),
                          (True, False, False, True, range(0, 3, 2)),
                          (False, True, True, False, range(3, -3, -2))])
def test_value_counts(sort: bool, dropna: bool, ascending: bool, normalize: bool, rng: range) -> None:
    ri: RangeIndex = RangeIndex(rng, name='A')
    result: pd.Series = ri.value_counts(normalize=normalize, sort=sort, ascending=ascending, dropna=dropna)
    expected: pd.Series = Index(list(rng), name='A').value_counts(normalize=normalize, sort=sort, ascending=ascending, dropna=dropna)
    tm.assert_series_equal(result, expected, check_index_type=False)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('value', [0, -5, 5, -3, np.array([-5, -3, 0, 5])])
def test_searchsorted(side: str, value: Union[int, np.ndarray]) -> None:
    ri: RangeIndex = RangeIndex(-3, 3, 2)
    result = ri.searchsorted(value=value, side=side)
    expected = Index(list(ri)).searchsorted(value=value, side=side)
    if isinstance(value, int):
        assert result == expected
    else:
        tm.assert_numpy_array_equal(result, expected)


# The tests below use 'sort' parameter in test_value_counts; already defined above.
def test_dummy() -> None:
    pass  # placeholder if needed

# End of file.
