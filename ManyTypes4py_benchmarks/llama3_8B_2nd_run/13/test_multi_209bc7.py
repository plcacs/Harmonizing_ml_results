import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, RangeIndex, Series, Timestamp, option_context
from pandas._testing import tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge

@pytest.fixture
def left():
    """left dataframe (not multi-indexed) for multi-index join tests"""
    key1 = ['bar', 'bar', 'bar', 'foo', 'foo', 'baz', 'baz', 'qux', 'qux', 'snap']
    key2 = ['two', 'one', 'three', 'one', 'two', 'one', 'two', 'two', 'three', 'one']
    data = np.random.default_rng(2).standard_normal(len(key1))
    return DataFrame({'key1': key1, 'key2': key2, 'data': data})

@pytest.fixture
def right(multiindex_dataframe_random_data):
    """right dataframe (multi-indexed) for multi-index join tests"""
    df = multiindex_dataframe_random_data
    df.index.names = ['key1', 'key2']
    df.columns = ['j_one', 'j_two', 'j_three']
    return df

@pytest.fixture
def left_multi():
    return DataFrame({'Origin': ['A', 'A', 'B', 'B', 'C'], 'Destination': ['A', 'B', 'A', 'C', 'A'], 'Period': ['AM', 'AM', 'IP', 'AM', 'OP'], 'TripPurp': ['hbw', 'nhb', 'hbo', 'nhb', 'hbw'], 'Trips': [1987, 3647, 2470, 4296, 4444]}, columns=['Origin', 'Destination', 'Period', 'TripPurp', 'Trips']).set_index(['Origin', 'Destination', 'Period', 'TripPurp'])

@pytest.fixture
def right_multi():
    return DataFrame({'Origin': ['A', 'A', 'B', 'B', 'C', 'C', 'E'], 'Destination': ['A', 'B', 'A', 'B', 'A', 'B', 'F'], 'Period': ['AM', 'AM', 'IP', 'AM', 'OP', 'IP', 'AM'], 'LinkType': ['a', 'b', 'c', 'b', 'a', 'b', 'a'], 'Distance': [100, 80, 90, 80, 75, 35, 55]}, columns=['Origin', 'Destination', 'Period', 'LinkType', 'Distance']).set_index(['Origin', 'Destination', 'Period', 'LinkType'])

@pytest.fixture
def on_cols_multi():
    return ['Origin', 'Destination', 'Period']

class TestMergeMulti:
    def test_merge_on_multikey(self, left: DataFrame, right: DataFrame, join_type: str) -> None:
        on_cols = ['key1', 'key2']
        result = left.join(right, on=on_cols, how=join_type)
        expected = merge(left, right.reset_index(), on=on_cols, how=join_type)
        tm.assert_frame_equal(result, expected)

    def test_compress_group_combinations(self) -> None:
        key1 = [str(i) for i in range(10000)]
        key1 = np.tile(key1, 2)
        key2 = key1[::-1]
        df = DataFrame({'key1': key1, 'key2': key2, 'value1': np.random.default_rng(2).standard_normal(20000)})
        df2 = DataFrame({'key1': key1[::2], 'key2': key2[::2], 'value2': np.random.default_rng(2).standard_normal(10000)})
        merge(df, df2, how='outer')

    def test_left_join_index_preserve_order(self) -> None:
        on_cols = ['k1', 'k2']
        left = DataFrame({'k1': [0, 1, 2] * 8, 'k2': ['foo', 'bar'] * 12, 'v': np.array(np.arange(24), dtype=np.int64)})
        index = MultiIndex.from_tuples([(2, 'bar'), (1, 'foo')])
        right = DataFrame({'v2': [5, 7]}, index=index)
        result = left.join(right, on=on_cols)
        expected = left.copy()
        expected['v2'] = np.nan
        expected.loc[(expected.k1 == 2) & (expected.k2 == 'bar'), 'v2'] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == 'foo'), 'v2'] = 7
        tm.assert_frame_equal(result, expected)

    # ... and so on
