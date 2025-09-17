import numpy as np
import pytest
from typing import Union
from pandas import PeriodIndex, Series, date_range, period_range, timedelta_range, Index, DatetimeIndex, TimedeltaIndex
import pandas._testing as tm
from _pytest.fixtures import SubRequest


class DropDuplicates:
    def test_drop_duplicates_metadata(self, idx: Index) -> None:
        result = idx.drop_duplicates()
        tm.assert_index_equal(idx, result)
        assert idx.freq == result.freq
        idx_dup = idx.append(idx)
        result = idx_dup.drop_duplicates()
        expected = idx
        if not isinstance(idx, PeriodIndex):
            assert idx_dup.freq is None
            assert result.freq is None
            expected = idx._with_freq(None)
        else:
            assert result.freq == expected.freq
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'keep, expected, index', [
            ('first', np.concatenate(([False] * 10, [True] * 5)),
             np.arange(0, 10, dtype=np.int64)),
            ('last', np.concatenate(([True] * 5, [False] * 10)),
             np.arange(5, 15, dtype=np.int64)),
            (False, np.concatenate(([True] * 5, [False] * 5, [True] * 5)),
             np.arange(5, 10, dtype=np.int64))
        ]
    )
    def test_drop_duplicates(self, keep: Union[str, bool],
                             expected: np.ndarray, index: np.ndarray,
                             idx: Index) -> None:
        idx = idx.append(idx[:5])
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected)
        expected_idx = idx[~expected]
        result = idx.drop_duplicates(keep=keep)
        tm.assert_index_equal(result, expected_idx)
        result_series = Series(idx).drop_duplicates(keep=keep)
        expected_series = Series(expected_idx, index=index)
        tm.assert_series_equal(result_series, expected_series)


class TestDropDuplicatesPeriodIndex(DropDuplicates):

    @pytest.fixture(params=['D', '3D', 'h', '2h', 'min', '2min', 's', '3s'])
    def freq(self, request: SubRequest) -> str:
        """
        Fixture to test for different frequencies for PeriodIndex.
        """
        return request.param

    @pytest.fixture
    def idx(self, freq: str) -> PeriodIndex:
        """
        Fixture to get PeriodIndex for 10 periods for different frequencies.
        """
        return period_range('2011-01-01', periods=10, freq=freq, name='idx')


class TestDropDuplicatesDatetimeIndex(DropDuplicates):

    @pytest.fixture
    def idx(self, freq_sample: str) -> DatetimeIndex:
        """
        Fixture to get DatetimeIndex for 10 periods for different frequencies.
        """
        return date_range('2011-01-01', freq=freq_sample, periods=10, name='idx')


class TestDropDuplicatesTimedeltaIndex(DropDuplicates):

    @pytest.fixture
    def idx(self, freq_sample: str) -> TimedeltaIndex:
        """
        Fixture to get TimedeltaIndex for 10 periods for different frequencies.
        """
        return timedelta_range('1 day', periods=10, freq=freq_sample, name='idx')