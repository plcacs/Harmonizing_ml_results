import numpy as np
import pytest
from pandas import DatetimeIndex, IntervalIndex, MultiIndex, Series
import pandas._testing as tm
from typing import Any, Callable, List, Tuple, Union

@pytest.fixture(params=['quicksort', 'mergesort', 'heapsort', 'stable'])
def sort_kind(request: pytest.FixtureRequest) -> str:
    return request.param

class TestSeriesSortIndex:

    def test_sort_index_name(self, datetime_series: Series[Any]) -> None:
        result: Series[Any] = datetime_series.sort_index(ascending=False)
        assert result.name == datetime_series.name

    def test_sort_index(self, datetime_series: Series[Any]) -> None:
        datetime_series.index = datetime_series.index._with_freq(None)  # type: ignore
        rindex: List[Any] = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)
        random_order: Series[Any] = datetime_series.reindex(rindex)
        sorted_series: Series[Any] = random_order.sort_index()
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

    def test_sort_index_inplace(self, datetime_series: Series[Any]) -> None:
        datetime_series.index = datetime_series.index._with_freq(None)  # type: ignore
        rindex: List[Any] = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)
        random_order: Series[Any] = datetime_series.reindex(rindex)
        result: None = random_order.sort_index(ascending=False, inplace=True)
        assert result is None
        expected: Series[Any] = datetime_series.reindex(datetime_series.index[::-1])
        expected.index = expected.index._with_freq(None)  # type: ignore
        tm.assert_series_equal(random_order, expected)
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=True, inplace=True)
        assert result is None
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)  # type: ignore
        tm.assert_series_equal(random_order, expected)

    def test_sort_index_level(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s: Series[int] = Series([1, 2], mi)
        backwards: Series[int] = s.iloc[[1, 0]]
        res: Series[int] = s.sort_index(level='A')
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
        s: Series[int] = Series([1, 2], mi)
        backwards: Series[int] = s.iloc[[1, 0]]
        res: Series[int] = s.sort_index(level=level)
        tm.assert_series_equal(backwards, res)
        res = s.sort_index(level=level, sort_remaining=False)
        tm.assert_series_equal(s, res)

    def test_sort_index_kind(self, sort_kind: str) -> None:
        series: Series[Any] = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series: Series[Any] = Series(index=[1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series[Any] = series.sort_index(kind=sort_kind)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position(self) -> None:
        series: Series[Any] = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first: Series[Any] = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series[Any] = series.sort_index(na_position='first')
        tm.assert_series_equal(expected_series_first, index_sorted_series)
        expected_series_last: Series[Any] = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)
        index_sorted_series = series.sort_index(na_position='last')
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_sort_index_intervals(self) -> None:
        s: Series[Union[float, int]] = Series(
            [np.nan, 1, 2, 3],
            IntervalIndex.from_arrays([0, 1, 2, 3], [1, 2, 3, 4])
        )
        result: Series[Union[float, int]] = s.sort_index()
        expected: Series[Union[float, int]] = s
        tm.assert_series_equal(result, expected)
        result = s.sort_index(ascending=False)
        expected = Series(
            [3, 2, 1, np.nan],
            IntervalIndex.from_arrays([3, 2, 1, 0], [4, 3, 2, 1])
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'inplace, original_list, sorted_list, ascending, ignore_index, output_index',
        [
            ([True, False][0], [2, 3, 6, 1], [2, 3, 6, 1], True, True, [0, 1, 2, 3]),
            ([True, False][1], [2, 3, 6, 1], [2, 3, 6, 1], True, False, [0, 1, 2, 3]),
            ([True, False][0], [2, 3, 6, 1], [1, 6, 3, 2], False, True, [0, 1, 2, 3]),
            ([True, False][1], [2, 3, 6, 1], [1, 6, 3, 2], False, False, [3, 2, 1, 0])
        ]
    )
    def test_sort_index_ignore_index(
        self,
        inplace: bool,
        original_list: List[int],
        sorted_list: List[int],
        ascending: bool,
        ignore_index: bool,
        output_index: List[int]
    ) -> None:
        ser: Series[int] = Series(original_list)
        expected: Series[int] = Series(sorted_list, index=output_index)
        kwargs: dict = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_ser: Series[int] = ser.copy()
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
        tuples: List[Tuple[Union[str, int], Union[str, int], int]] = list(zip(*arrays))
        mi: MultiIndex = MultiIndex.from_tuples(tuples, names=['first', 'second', 'third'])
        ser: Series[int] = Series(range(8), index=mi)
        result: Series[int] = ser.sort_index(level=['third', 'first'], ascending=False)
        expected: Series[int] = ser.iloc[[4, 0, 5, 1, 6, 2, 7, 3]]
        tm.assert_series_equal(result, expected)
        result = ser.sort_index(level=['third', 'first'], ascending=[False, True])
        expected = ser.iloc[[0, 4, 1, 5, 2, 6, 3, 7]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ascending', [None, (True, None), (False, 'True')])
    def test_sort_index_ascending_bad_value_raises(self, ascending: Any) -> None:
        ser: Series[int] = Series(range(10), index=[0, 3, 2, 1, 4, 5, 7, 6, 8, 9])
        match: str = 'For argument "ascending" expected type bool'
        with pytest.raises(ValueError, match=match):
            ser.sort_index(ascending=ascending)

class TestSeriesSortIndexKey:

    def test_sort_index_multiindex_key(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s: Series[int] = Series([1, 2], mi)
        backwards: Series[int] = s.iloc[[1, 0]]
        result: Series[int] = s.sort_index(level='C', key=lambda x: -x)
        tm.assert_series_equal(s, result)
        result = s.sort_index(level='C', key=lambda x: x)
        tm.assert_series_equal(backwards, result)

    def test_sort_index_multiindex_key_multi_level(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s: Series[int] = Series([1, 2], mi)
        backwards: Series[int] = s.iloc[[1, 0]]
        result: Series[int] = s.sort_index(level=['A', 'C'], key=lambda x: -x)
        tm.assert_series_equal(s, result)
        result = s.sort_index(level=['A', 'C'], key=lambda x: x)
        tm.assert_series_equal(backwards, result)

    def test_sort_index_key(self) -> None:
        series: Series[int] = Series(np.arange(6, dtype='int64'), index=list('aaBBca'))
        result: Series[int] = series.sort_index()
        expected: Series[int] = series.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: x.str.lower())
        expected = series.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = series.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_key_int(self) -> None:
        series: Series[int] = Series(np.arange(6, dtype='int64'), index=np.arange(6, dtype='int64'))
        result: Series[int] = series.sort_index()
        tm.assert_series_equal(result, series)
        result = series.sort_index(key=lambda x: -x)
        expected: Series[int] = series.sort_index(ascending=False)
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: 2 * x)
        tm.assert_series_equal(result, series)

    def test_sort_index_kind_key(self, sort_kind: str, sort_by_key: Callable[[Any], Any]) -> None:
        series: Series[Any] = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series: Series[Any] = Series(index=[1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series[Any] = series.sort_index(kind=sort_kind, key=sort_by_key)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_kind_neg_key(self, sort_kind: str) -> None:
        series: Series[Any] = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series: Series[Any] = Series(index=[4, 3, 3, 2, 1], dtype=object)
        index_sorted_series: Series[Any] = series.sort_index(kind=sort_kind, key=lambda x: -x)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position_key(self, sort_by_key: Callable[[Any], Any]) -> None:
        series: Series[Any] = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first: Series[Any] = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)
        index_sorted_series: Series[Any] = series.sort_index(na_position='first', key=sort_by_key)
        tm.assert_series_equal(expected_series_first, index_sorted_series)
        expected_series_last: Series[Any] = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)
        index_sorted_series = series.sort_index(na_position='last', key=sort_by_key)
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_changes_length_raises(self) -> None:
        s: Series[int] = Series([1, 2, 3])
        with pytest.raises(ValueError, match='change the shape'):
            s.sort_index(key=lambda x: x[:1])

    def test_sort_values_key_type(self) -> None:
        s: Series[int] = Series(
            [1, 2, 3],
            DatetimeIndex(['2008-10-24', '2008-11-23', '2007-12-22'])
        )
        result: Series[int] = s.sort_index(key=lambda x: x.month)
        expected: Series[int] = s.iloc[[0, 1, 2]]
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
        ser: Series[int] = Series(range(len(mi)), index=mi)
        result: Series[int] = ser.sort_index(ascending=ascending)
        if ascending == [True, False]:
            expected: Series[int] = ser.take([1, 0, 3, 2])
        elif ascending == [False, True]:
            expected = ser.take([2, 3, 0, 1])
        tm.assert_series_equal(result, expected)
