import string
from typing import List, Optional, Tuple
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, array, concat, date_range, merge, merge_asof
try:
    from pandas import merge_ordered
except ImportError:
    from pandas import ordered_merge as merge_ordered

class Concat:
    params = [0, 1]
    param_names = ['axis']

    def setup(self, axis: int) -> None:
        N: int = 1000
        s: Series = Series(N, index=Index([f'i-{i}' for i in range(N)], dtype=object))
        self.series: List[Series] = [s[i:-i] for i in range(1, 10)] * 50
        self.small_frames: List[DataFrame] = [DataFrame(np.random.randn(5, 4))] * 1000
        df: DataFrame = DataFrame({'A': range(N)}, index=date_range('20130101', periods=N, freq='s'))
        self.empty_left: List[DataFrame] = [DataFrame(), df]
        self.empty_right: List[DataFrame] = [df, DataFrame()]
        self.mixed_ndims: List[DataFrame] = [df, df.head(N // 2)]

    def time_concat_series(self, axis: int) -> None:
        concat(self.series, axis=axis, sort=False)

    def time_concat_small_frames(self, axis: int) -> None:
        concat(self.small_frames, axis=axis)

    def time_concat_empty_right(self, axis: int) -> None:
        concat(self.empty_right, axis=axis)

    def time_concat_empty_left(self, axis: int) -> None:
        concat(self.empty_left, axis=axis)

    def time_concat_mixed_ndims(self, axis: int) -> None:
        concat(self.mixed_ndims, axis=axis)

class ConcatDataFrames:
    params = ([0, 1], [True, False])
    param_names = ['axis', 'ignore_index']

    def setup(self, axis: int, ignore_index: bool) -> None:
        frame_c: DataFrame = DataFrame(np.zeros((10000, 200), dtype=np.float32, order='C'))
        self.frame_c: List[DataFrame] = [frame_c] * 20
        frame_f: DataFrame = DataFrame(np.zeros((10000, 200), dtype=np.float32, order='F'))
        self.frame_f: List[DataFrame] = [frame_f] * 20

    def time_c_ordered(self, axis: int, ignore_index: bool) -> None:
        concat(self.frame_c, axis=axis, ignore_index=ignore_index)

    def time_f_ordered(self, axis: int, ignore_index: bool) -> None:
        concat(self.frame_f, axis=axis, ignore_index=ignore_index)

class ConcatIndexDtype:
    params = (
        ['datetime64[ns]', 'int64', 'Int64', 'int64[pyarrow]', 'string[python]', 'string[pyarrow]'],
        ['monotonic', 'non_monotonic', 'has_na'],
        [0, 1],
        [True, False]
    )
    param_names = ['dtype', 'structure', 'axis', 'sort']

    def setup(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        N: int = 10000
        if dtype == 'datetime64[ns]':
            vals = date_range('1970-01-01', periods=N)
        elif dtype in ('int64', 'Int64', 'int64[pyarrow]'):
            vals = np.arange(N, dtype=np.int64)
        elif dtype in ('string[python]', 'string[pyarrow]'):
            vals = Index([f'i-{i}' for i in range(N)], dtype=object)
        else:
            raise NotImplementedError
        idx: Index = Index(vals, dtype=dtype)
        if structure == 'monotonic':
            idx = idx.sort_values()
        elif structure == 'non_monotonic':
            idx = idx[::-1]
        elif structure == 'has_na':
            if not idx._can_hold_na:
                raise NotImplementedError
            idx = Index([None], dtype=dtype).append(idx)
        else:
            raise NotImplementedError
        self.series: List[Series] = [Series(i, idx[:-i]) for i in range(1, 6)]

    def time_concat_series(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        concat(self.series, axis=axis, sort=sort)

class Join:
    params = [True, False]
    param_names = ['sort']

    def setup(self, sort: bool) -> None:
        level1: np.ndarray = Index([f'i-{i}' for i in range(10)], dtype=object).values
        level2: np.ndarray = Index([f'i-{i}' for i in range(1000)], dtype=object).values
        codes1: np.ndarray = np.arange(10).repeat(1000)
        codes2: np.ndarray = np.tile(np.arange(1000), 10)
        index2: MultiIndex = MultiIndex(levels=[level1, level2], codes=[codes1, codes2])
        self.df_multi: DataFrame = DataFrame(np.random.randn(len(index2), 4), index=index2, columns=['A', 'B', 'C', 'D'])
        self.key1: np.ndarray = np.tile(level1.take(codes1), 10)
        self.key2: np.ndarray = np.tile(level2.take(codes2), 10)
        self.df: DataFrame = DataFrame({
            'data1': np.random.randn(100000),
            'data2': np.random.randn(100000),
            'key1': self.key1,
            'key2': self.key2
        })
        self.df_key1: DataFrame = DataFrame(np.random.randn(len(level1), 4), index=level1, columns=['A', 'B', 'C', 'D'])
        self.df_key2: DataFrame = DataFrame(np.random.randn(len(level2), 4), index=level2, columns=['A', 'B', 'C', 'D'])
        shuf: np.ndarray = np.arange(100000)
        np.random.shuffle(shuf)
        self.df_shuf: DataFrame = self.df.reindex(self.df.index[shuf])

    def time_join_dataframe_index_multi(self, sort: bool) -> None:
        self.df.join(self.df_multi, on=['key1', 'key2'], sort=sort)

    def time_join_dataframe_index_single_key_bigger(self, sort: bool) -> None:
        self.df.join(self.df_key2, on='key2', sort=sort)

    def time_join_dataframe_index_single_key_small(self, sort: bool) -> None:
        self.df.join(self.df_key1, on='key1', sort=sort)

    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort: bool) -> None:
        self.df_shuf.join(self.df_key2, on='key2', sort=sort)

    def time_join_dataframes_cross(self, sort: bool) -> None:
        self.df.loc[:2000].join(self.df_key1, how='cross', sort=sort)

class JoinIndex:
    def setup(self) -> None:
        N: int = 5000
        self.left: DataFrame = DataFrame(np.random.randint(1, N / 50, (N, 2)), columns=['jim', 'joe'])
        self.right: DataFrame = DataFrame(np.random.randint(1, N / 50, (N, 2)), columns=['jolie', 'jolia']).set_index('jolie')

    def time_left_outer_join_index(self) -> None:
        self.left.join(self.right, on='jim')

class JoinMultiindexSubset:
    def setup(self) -> None:
        N: int = 100000
        mi1: MultiIndex = MultiIndex.from_arrays([np.arange(N)] * 4, names=['a', 'b', 'c', 'd'])
        mi2: MultiIndex = MultiIndex.from_arrays([np.arange(N)] * 2, names=['a', 'b'])
        self.left: DataFrame = DataFrame({'col1': 1}, index=mi1)
        self.right: DataFrame = DataFrame({'col2': 2}, index=mi2)

    def time_join_multiindex_subset(self) -> None:
        self.left.join(self.right)

class JoinEmpty:
    def setup(self) -> None:
        N: int = 100000
        self.df: DataFrame = DataFrame({'A': np.arange(N)})
        self.df_empty: DataFrame = DataFrame(columns=['B', 'C'], dtype='int64')

    def time_inner_join_left_empty(self) -> None:
        self.df_empty.join(self.df, how='inner')

    def time_inner_join_right_empty(self) -> None:
        self.df.join(self.df_empty, how='inner')

class JoinNonUnique:
    def setup(self) -> None:
        date_index = date_range('01-Jan-2013', '23-Jan-2013', freq='min')
        daily_dates = date_index.to_period('D').to_timestamp('s', 's')
        fracofday = date_index.values - daily_dates.values
        fracofday = fracofday.astype('timedelta64[ns]')
        fracofday = fracofday.astype(np.float64) / 86400000000000
        self.fracofday: Series = Series(fracofday, daily_dates)
        index = date_range(date_index.min(), date_index.max(), freq='D')
        self.temp: Series = Series(1.0, index)[self.fracofday.index]

    def time_join_non_unique_equal(self) -> None:
        self.fracofday * self.temp

class Merge:
    params = [True, False]
    param_names = ['sort']

    def setup(self, sort: bool) -> None:
        N: int = 10000
        indices: np.ndarray = Index([f'i-{i}' for i in range(N)], dtype=object).values
        indices2: np.ndarray = Index([f'i-{i}' for i in range(N)], dtype=object).values
        key = np.tile(indices[:8000], 10)
        key2 = np.tile(indices2[:8000], 10)
        self.left: DataFrame = DataFrame({'key': key, 'key2': key2, 'value': np.random.randn(80000)})
        self.right: DataFrame = DataFrame({'key': indices[2000:], 'key2': indices2[2000:], 'value2': np.random.randn(8000)})
        self.df: DataFrame = DataFrame({
            'key1': np.tile(np.arange(500).repeat(10), 2),
            'key2': np.tile(np.arange(250).repeat(10), 4),
            'value': np.random.randn(10000)
        })
        self.df2: DataFrame = DataFrame({'key1': np.arange(500), 'value2': np.random.randn(500)})
        self.df3: DataFrame = self.df[:5000]

    def time_merge_2intkey(self, sort: bool) -> None:
        merge(self.left, self.right, sort=sort)

    def time_merge_dataframe_integer_2key(self, sort: bool) -> None:
        merge(self.df, self.df3, sort=sort)

    def time_merge_dataframe_integer_key(self, sort: bool) -> None:
        merge(self.df, self.df2, on='key1', sort=sort)

    def time_merge_dataframe_empty_right(self, sort: bool) -> None:
        merge(self.left, self.right.iloc[:0], sort=sort)

    def time_merge_dataframe_empty_left(self, sort: bool) -> None:
        merge(self.left.iloc[:0], self.right, sort=sort)

    def time_merge_dataframes_cross(self, sort: bool) -> None:
        merge(self.left.loc[:2000], self.right.loc[:2000], how='cross', sort=sort)

class MergeEA:
    params = [['Int64', 'Int32', 'Int16', 'UInt64', 'UInt32', 'UInt16', 'Float64', 'Float32'], [True, False]]
    param_names = ['dtype', 'monotonic']

    def setup(self, dtype: str, monotonic: bool) -> None:
        N: int = 10000
        indices: np.ndarray = np.arange(1, N)
        key = np.tile(indices[:8000], 10)
        self.left: DataFrame = DataFrame({
            'key': Series(key, dtype=dtype),
            'value': np.random.randn(80000)
        })
        self.right: DataFrame = DataFrame({
            'key': Series(indices[2000:], dtype=dtype),
            'value2': np.random.randn(7999)
        })
        if monotonic:
            self.left = self.left.sort_values('key')
            self.right = self.right.sort_values('key')

    def time_merge(self, dtype: str, monotonic: bool) -> None:
        merge(self.left, self.right)

class I8Merge:
    params = ['inner', 'outer', 'left', 'right']
    param_names = ['how']

    def setup(self, how: str) -> None:
        low: int = -1000
        high: int = 1000
        n: int = 10 ** 6
        self.left: DataFrame = DataFrame(np.random.randint(low, high, (n, 7)), columns=list('ABCDEFG'))
        self.left['left'] = self.left.sum(axis=1)
        self.right: DataFrame = self.left.sample(frac=1).rename({'left': 'right'}, axis=1)
        self.right = self.right.reset_index(drop=True)
        self.right['right'] *= -1

    def time_i8merge(self, how: str) -> None:
        merge(self.left, self.right, how=how)

class UniqueMerge:
    params = [4000000, 1000000]
    param_names = ['unique_elements']

    def setup(self, unique_elements: int) -> None:
        N: int = 1000000
        self.left: DataFrame = DataFrame({'a': np.random.randint(1, unique_elements, (N,))})
        self.right: DataFrame = DataFrame({'a': np.random.randint(1, unique_elements, (N,))})
        uniques: Series = self.right.a.drop_duplicates()
        self.right['a'] = concat([uniques, Series(np.arange(0, -(N - len(uniques)), -1))], ignore_index=True)

    def time_unique_merge(self, unique_elements: int) -> None:
        merge(self.left, self.right, how='inner')

class MergeDatetime:
    params = [[('ns', 'ns'), ('ms', 'ms'), ('ns', 'ms')], [None, 'Europe/Brussels'], [True, False]]
    param_names = ['units', 'tz', 'monotonic']

    def setup(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> None:
        unit_left, unit_right = units
        N: int = 10000
        keys: Series = Series(date_range('2012-01-01', freq='min', periods=N, tz=tz))
        self.left: DataFrame = DataFrame({
            'key': keys.sample(N * 10, replace=True).dt.as_unit(unit_left),
            'value1': np.random.randn(N * 10)
        })
        self.right: DataFrame = DataFrame({
            'key': keys[:8000].dt.as_unit(unit_right),
            'value2': np.random.randn(8000)
        })
        if monotonic:
            self.left = self.left.sort_values('key')
            self.right = self.right.sort_values('key')

    def time_merge(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> None:
        merge(self.left, self.right)

class MergeCategoricals:
    def setup(self) -> None:
        self.left_object: DataFrame = DataFrame({
            'X': np.random.choice(range(10), size=(10000,)),
            'Y': np.random.choice(['one', 'two', 'three'], size=(10000,))
        })
        self.right_object: DataFrame = DataFrame({
            'X': np.random.choice(range(10), size=(10000,)),
            'Z': np.random.choice(['jjj', 'kkk', 'sss'], size=(10000,))
        })
        self.left_cat: DataFrame = self.left_object.assign(Y=self.left_object['Y'].astype('category'))
        self.right_cat: DataFrame = self.right_object.assign(Z=self.right_object['Z'].astype('category'))
        self.left_cat_col: DataFrame = self.left_object.astype({'X': 'category'})
        self.right_cat_col: DataFrame = self.right_object.astype({'X': 'category'})
        self.left_cat_idx: DataFrame = self.left_cat_col.set_index('X')
        self.right_cat_idx: DataFrame = self.right_cat_col.set_index('X')

    def time_merge_object(self) -> None:
        merge(self.left_object, self.right_object, on='X')

    def time_merge_cat(self) -> None:
        merge(self.left_cat, self.right_cat, on='X')

    def time_merge_on_cat_col(self) -> None:
        merge(self.left_cat_col, self.right_cat_col, on='X')

    def time_merge_on_cat_idx(self) -> None:
        merge(self.left_cat_idx, self.right_cat_idx, on='X')

class MergeOrdered:
    def setup(self) -> None:
        groups: np.ndarray = Index([f'i-{i}' for i in range(10)], dtype=object).values
        self.left: DataFrame = DataFrame({
            'group': groups.repeat(5000),
            'key': np.tile(np.arange(0, 10000, 2), 10),
            'lvalue': np.random.randn(50000)
        })
        self.right: DataFrame = DataFrame({
            'key': np.arange(10000),
            'rvalue': np.random.randn(10000)
        })

    def time_merge_ordered(self) -> None:
        merge_ordered(self.left, self.right, on='key', left_by='group')

class MergeAsof:
    params = [['backward', 'forward', 'nearest'], [None, 5]]
    param_names = ['direction', 'tolerance']

    def setup(self, direction: str, tolerance: Optional[int]) -> None:
        one_count: int = 200000
        two_count: int = 1000000
        df1: DataFrame = DataFrame({
            'time': np.random.randint(0, one_count / 20, one_count),
            'key': np.random.choice(list(string.ascii_uppercase), one_count),
            'key2': np.random.randint(0, 25, one_count),
            'value1': np.random.randn(one_count)
        })
        df2: DataFrame = DataFrame({
            'time': np.random.randint(0, two_count / 20, two_count),
            'key': np.random.choice(list(string.ascii_uppercase), two_count),
            'key2': np.random.randint(0, 25, two_count),
            'value2': np.random.randn(two_count)
        })
        df1 = df1.sort_values('time')
        df2 = df2.sort_values('time')
        df1['time32'] = np.int32(df1.time)
        df2['time32'] = np.int32(df2.time)
        df1['timeu64'] = np.uint64(df1.time)
        df2['timeu64'] = np.uint64(df2.time)
        self.df1a: DataFrame = df1[['time', 'value1']]
        self.df2a: DataFrame = df2[['time', 'value2']]
        self.df1b: DataFrame = df1[['time', 'key', 'value1']]
        self.df2b: DataFrame = df2[['time', 'key', 'value2']]
        self.df1c: DataFrame = df1[['time', 'key2', 'value1']]
        self.df2c: DataFrame = df2[['time', 'key2', 'value2']]
        self.df1d: DataFrame = df1[['time32', 'value1']]
        self.df2d: DataFrame = df2[['time32', 'value2']]
        self.df1e: DataFrame = df1[['time', 'key', 'key2', 'value1']]
        self.df2e: DataFrame = df2[['time', 'key', 'key2', 'value2']]
        self.df1f: DataFrame = df1[['timeu64', 'value1']]
        self.df2f: DataFrame = df2[['timeu64', 'value2']]

    def time_on_int(self, direction: str, tolerance: Optional[int]) -> None:
        merge_asof(self.df1a, self.df2a, on='time', direction=direction, tolerance=tolerance)

    def time_on_int32(self, direction: str, tolerance: Optional[int]) -> None:
        merge_asof(self.df1d, self.df2d, on='time32', direction=direction, tolerance=tolerance)

    def time_on_uint64(self, direction: str, tolerance: Optional[int]) -> None:
        merge_asof(self.df1f, self.df2f, on='timeu64', direction=direction, tolerance=tolerance)

    def time_by_object(self, direction: str, tolerance: Optional[int]) -> None:
        merge_asof(self.df1b, self.df2b, on='time', by='key', direction=direction, tolerance=tolerance)

    def time_by_int(self, direction: str, tolerance: Optional[int]) -> None:
        merge_asof(self.df1c, self.df2c, on='time', by='key2', direction=direction, tolerance=tolerance)

    def time_multiby(self, direction: str, tolerance: Optional[int]) -> None:
        merge_asof(self.df1e, self.df2e, on='time', by=['key', 'key2'], direction=direction, tolerance=tolerance)

class MergeMultiIndex:
    params = [[('int64', 'int64'), ('datetime64[ns]', 'int64'), ('Int64', 'Int64')], ['left', 'right', 'inner', 'outer']]
    param_names = ['dtypes', 'how']

    def setup(self, dtypes: Tuple[str, str], how: str) -> None:
        n: int = 100000
        offset: int = 50000
        mi1: MultiIndex = MultiIndex.from_arrays([array(np.arange(n), dtype=dtypes[0]), array(np.arange(n), dtype=dtypes[1])])
        mi2: MultiIndex = MultiIndex.from_arrays([array(np.arange(offset, n + offset), dtype=dtypes[0]), array(np.arange(offset, n + offset), dtype=dtypes[1])])
        self.df1: DataFrame = DataFrame({'col1': 1}, index=mi1)
        self.df2: DataFrame = DataFrame({'col2': 2}, index=mi2)

    def time_merge_sorted_multiindex(self, dtypes: Tuple[str, str], how: str) -> None:
        df1: DataFrame = self.df1.copy()
        df2: DataFrame = self.df2.copy()
        merge(df1, df2, how=how, left_index=True, right_index=True)

class Align:
    def setup(self) -> None:
        size: int = 5 * 10 ** 5
        rng: np.ndarray = np.arange(0, 10 ** 13, 10 ** 7)
        stamps: np.int64 = np.datetime64('now').view('i8') + rng
        idx1: np.ndarray = np.sort(np.random.choice(stamps, size, replace=False))
        idx2: np.ndarray = np.sort(np.random.choice(stamps, size, replace=False))
        self.ts1: Series = Series(np.random.randn(size), idx1)
        self.ts2: Series = Series(np.random.randn(size), idx2)

    def time_series_align_int64_index(self) -> None:
        self.ts1 + self.ts2

    def time_series_align_left_monotonic(self) -> None:
        self.ts1.align(self.ts2, join='left')

from .pandas_vb_common import setup
