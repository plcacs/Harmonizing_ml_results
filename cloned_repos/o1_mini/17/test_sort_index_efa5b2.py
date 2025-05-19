import numpy as np
import pytest
from pandas import DatetimeIndex, IntervalIndex, MultiIndex, Series
import pandas._testing as tm
from typing import Callable, List, Optional, Union, Tuple


@pytest.fixture(params=['quicksort', 'mergesort', 'heapsort', 'stable'])
def sort_kind(request) -> str:
    return request.param


class TestSeriesSortIndex:

    def test_sort_index_name(self, datetime_series: Series) -> None:
        result: Series = datetime_series.sort_index(ascending=False)
        assert result.name == datetime_series.name

    def test_sort_index(self, datetime_series: Series) -> None:
        datetime_series.index = datetime_series.index._with_freq(None)
        rindex: List = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)
        random_order: Series = datetime_series.reindex(rindex)
        sorted_series: Series = random_order.sort_index()
        tm.assert_series_equal(sorted_series, datetime_series)
        sorted_series = random_order.sort_index(ascending=False)
        tm.assert_series_equal(sorted_series, datetime_series.reindex(datetime_series.index[::-1]))
        sorted_series = random_order.sort_index(level=0)
        tm.assert_series_equal(sorted_series, datetime_series)
        sorted_series = random_order.sort_index(axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)
        msg: str = 'No axis named 1 for object type Series'
        with pytest.raises(ValueError, match=msg):
            random_order.sort_values(axis=1)
        sorted_series = random_order.sort_index(level=0, axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)
        with pytest.raises(ValueError, match=msg):
            random_order.sort_index(level=0, axis=1)

    def test_sort_index_inplace(self, datetime_series: Series) -> None:
        datetime_series.index = datetime_series.index._with_freq(None)
        rindex: List = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)
        random_order: Series = datetime_series.reindex(rindex)
        result: Optional[None] = random_order.sort_index(ascending=False, inplace=True)
        assert result is None
        expected: Series = datetime_series.reindex(datetime_series.index[::-1])
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=True, inplace=True)
        assert result is None
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)

    def test_sort_index_level(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s: Series = Series([1, 2], mi)
        backwards: Series = s.iloc[[1, 0]]
        res: Series = s.sort_index(level='A')
        tm.assert_series_equal(backwards, res)
        res = s.sort_index(level=['A', 'B'])
        tm.assert_series_equal(backwards, res)
        res = s.sort_index(level='A', sort_remaining=False)
        tm.assert_series_equal(s, res)
        res = s.sort_index(level=['A', 'B'], sort_remaining=False)
        tm.assert_series_equal(s, res)

    @pytest.mark.parametrize('level', ['A', 0])
    def test_sort_index_multiindex(self, level: Union[str, int]) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s: Series = Series([1, 2], mi)
        backwards: Series = s.iloc[[1, 0]]
        res: Series = s.sort_index(level=level)
        tm.assert_series_equal(backwards, res)
        res = s.sort_index(level=level, sort_remaining=False)
        tm.assert_series_equal(s, res)

    def test_sort_index_kind(self, sort_kind: str) -> None:
        series: Series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series: Series = Series(index=[1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series = series.sort_index(kind=sort_kind)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position(self) -> None:
        series: Series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first: Series = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series = series.sort_index(na_position='first')
        tm.assert_series_equal(expected_series_first, index_sorted_series)
        expected_series_last: Series = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)
        index_sorted_series = series.sort_index(na_position='last')
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_sort_index_intervals(self) -> None:
        s: Series = Series([np.nan, 1, 2, 3], IntervalIndex.from_arrays([0, 1, 2, 3], [1, 2, 3, 4]))
        result: Series = s.sort_index()
        expected: Series = s
        tm.assert_series_equal(result, expected)
        result = s.sort_index(ascending=False)
        expected = Series([3, 2, 1, np.nan], IntervalIndex.from_arrays([3, 2, 1, 0], [4, 3, 2, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'inplace, original_list, sorted_list, ascending, ignore_index, output_index',
        [
            ([2, 3, 6, 1], [2, 3, 6, 1], True, True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [2, 3, 6, 1], True, False, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [1, 6, 3, 2], False, True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [1, 6, 3, 2], False, False, [3, 2, 1, 0]),
        ]
    )
    @pytest.mark.parametrize('inplace', [True, False])
    def test_sort_index_ignore_index(
        self,
        inplace: bool,
        original_list: List[int],
        sorted_list: List[int],
        ascending: bool,
        ignore_index: bool,
        output_index: List[int]
    ) -> None:
        ser: Series = Series(original_list)
        expected: Series = Series(sorted_list, index=output_index)
        kwargs: dict = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_ser: Series = ser.copy()
            result_ser.sort_index(**kwargs)
        else:
            result_ser = ser.sort_index(**kwargs)
        tm.assert_series_equal(result_ser, expected)
        tm.assert_series_equal(ser, Series(original_list))

    def test_sort_index_ascending_list(self) -> None:
        arrays: List[List[Union[str, int]]] = [
            ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'],
            [4, 3, 2, 1, 4, 3, 2, 1]
        ]
        tuples: List[Tuple[Union[str, int], ...]] = list(zip(*arrays))
        mi: MultiIndex = MultiIndex.from_tuples(tuples, names=['first', 'second', 'third'])
        ser: Series = Series(range(8), index=mi)
        result: Series = ser.sort_index(level=['third', 'first'], ascending=False)
        expected: Series = ser.iloc[[4, 0, 5, 1, 6, 2, 7, 3]]
        tm.assert_series_equal(result, expected)
        result = ser.sort_index(level=['third', 'first'], ascending=[False, True])
        expected = ser.iloc[[0, 4, 1, 5, 2, 6, 3, 7]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ascending', [None, (True, None), (False, 'True')])
    def test_sort_index_ascending_bad_value_raises(self, ascending: Union[None, Tuple[bool, Optional[bool]], Tuple[bool, str]]) -> None:
        ser: Series = Series(range(10), index=[0, 3, 2, 1, 4, 5, 7, 6, 8, 9])
        match: str = 'For argument "ascending" expected type bool'
        with pytest.raises(ValueError, match=match):
            ser.sort_index(ascending=ascending)


class TestSeriesSortIndexKey:

    def test_sort_index_multiindex_key(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s: Series = Series([1, 2], mi)
        backwards: Series = s.iloc[[1, 0]]
        result: Series = s.sort_index(level='C', key=lambda x: -x)
        tm.assert_series_equal(s, result)
        result = s.sort_index(level='C', key=lambda x: x)
        tm.assert_series_equal(backwards, result)

    def test_sort_index_multiindex_key_multi_level(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s: Series = Series([1, 2], mi)
        backwards: Series = s.iloc[[1, 0]]
        result: Series = s.sort_index(level=['A', 'C'], key=lambda x: -x)
        tm.assert_series_equal(s, result)
        result = s.sort_index(level=['A', 'C'], key=lambda x: x)
        tm.assert_series_equal(backwards, result)

    def test_sort_index_key(self) -> None:
        series: Series = Series(np.arange(6, dtype='int64'), index=list('aaBBca'))
        result: Series = series.sort_index()
        expected: Series = series.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: x.str.lower())
        expected = series.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = series.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_key_int(self) -> None:
        series: Series = Series(np.arange(6, dtype='int64'), index=np.arange(6, dtype='int64'))
        result: Series = series.sort_index()
        tm.assert_series_equal(result, series)
        result = series.sort_index(key=lambda x: -x)
        expected: Series = series.sort_index(ascending=False)
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: 2 * x)
        tm.assert_series_equal(result, series)

    def test_sort_index_kind_key(self, sort_kind: str, sort_by_key: Callable[[Series], Series]) -> None:
        series: Series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series: Series = Series(index=[1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series = series.sort_index(kind=sort_kind, key=sort_by_key)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_kind_neg_key(self, sort_kind: str) -> None:
        series: Series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series: Series = Series(index=[4, 3, 3, 2, 1], dtype=object)
        index_sorted_series: Series = series.sort_index(kind=sort_kind, key=lambda x: -x)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position_key(self, sort_by_key: Callable[[Series], Series]) -> None:
        series: Series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first: Series = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series = series.sort_index(na_position='first', key=sort_by_key)
        tm.assert_series_equal(expected_series_first, index_sorted_series)
        expected_series_last: Series = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)
        index_sorted_series = series.sort_index(na_position='last', key=sort_by_key)
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_changes_length_raises(self) -> None:
        s: Series = Series([1, 2, 3])
        with pytest.raises(ValueError, match='change the shape'):
            s.sort_index(key=lambda x: x[:1])  # type: ignore

    def test_sort_values_key_type(self) -> None:
        s: Series = Series(
            [1, 2, 3],
            DatetimeIndex(['2008-10-24', '2008-11-23', '2007-12-22'])
        )
        result: Series = s.sort_index(key=lambda x: x.month)
        expected: Series = s.iloc[[0, 1, 2]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(key=lambda x: x.day)
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(key=lambda x: x.year)
        expected = s.iloc[[2, 0, 1]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(key=lambda x: x.month_name())
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ascending', [[True, False], [False, True]])
    def test_sort_index_multi_already_monotonic(self, ascending: List[bool]) -> None:
        mi: MultiIndex = MultiIndex.from_product([[1, 2], [3, 4]])
        ser: Series = Series(range(len(mi)), index=mi)
        result: Series = ser.sort_index(ascending=ascending)
        if ascending == [True, False]:
            expected: Series = ser.take([1, 0, 3, 2])
        elif ascending == [False, True]:
            expected = ser.take([2, 3, 0, 1])
        tm.assert_series_equal(result, expected)
