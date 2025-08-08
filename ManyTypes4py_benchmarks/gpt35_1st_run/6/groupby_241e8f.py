from pandas import DataFrame, MultiIndex, Series, Timestamp, date_range, period_range, to_timedelta
import numpy as np

class ApplyDictReturn:

    def setup(self):
        self.labels: np.ndarray = np.arange(1000).repeat(10)
        self.data: Series = Series(np.random.randn(len(self.labels)))

    def time_groupby_apply_dict_return(self):
        self.data.groupby(self.labels).apply(lambda x: {'first': x.values[0], 'last': x.values[-1]})

class Apply:
    param_names: list[str] = ['factor']
    params: list[int] = [4, 5]

    def setup(self, factor: int):
        N: int = 10 ** factor
        labels: np.ndarray = np.random.randint(0, 2000 if factor == 4 else 20, size=N)
        labels2: np.ndarray = np.random.randint(0, 3, size=N)
        df: DataFrame = DataFrame({'key': labels, 'key2': labels2, 'value1': np.random.randn(N), 'value2': ['foo', 'bar', 'baz', 'qux'] * (N // 4)})
        self.df: DataFrame = df

    def time_scalar_function_multi_col(self, factor: int):
        self.df.groupby(['key', 'key2']).apply(lambda x: 1)

    def time_scalar_function_single_col(self, factor: int):
        self.df.groupby('key').apply(lambda x: 1)

    @staticmethod
    def df_copy_function(g: Series) -> Series:
        g.name
        return g.copy()

    def time_copy_function_multi_col(self, factor: int):
        self.df.groupby(['key', 'key2']).apply(self.df_copy_function)

    def time_copy_overhead_single_col(self, factor: int):
        self.df.groupby('key').apply(self.df_copy_function)

class ApplyNonUniqueUnsortedIndex:

    def setup(self):
        idx: np.ndarray = np.arange(100)[::-1]
        idx: Index = Index(np.repeat(idx, 200), name='key')
        self.df: DataFrame = DataFrame(np.random.randn(len(idx), 10), index=idx)

    def time_groupby_apply_non_unique_unsorted_index(self):
        self.df.groupby('key', group_keys=False).apply(lambda x: x)

class Groups:
    param_names: list[str] = ['key']
    params: list[str] = ['int64_small', 'int64_large', 'object_small', 'object_large']

    def setup_cache(self) -> dict[str, Series]:
        size: int = 10 ** 6
        data: dict[str, Series] = {'int64_small': Series(np.random.randint(0, 100, size=size)), 'int64_large': Series(np.random.randint(0, 10000, size=size)), 'object_small': Series(Index([f'i-{i}' for i in range(100)], dtype=object).take(np.random.randint(0, 100, size=size)), 'object_large': Series(Index([f'i-{i}' for i in range(10000)], dtype=object).take(np.random.randint(0, 10000, size=size)))}
        return data

    def setup(self, data: dict[str, Series], key: str):
        self.ser: Series = data[key]

    def time_series_groups(self, data: dict[str, Series], key: str):
        self.ser.groupby(self.ser).groups

    def time_series_indices(self, data: dict[str, Series], key: str):
        self.ser.groupby(self.ser).indices

class GroupManyLabels:
    params: list[int] = [1, 1000]
    param_names: list[str] = ['ncols']

    def setup(self, ncols: int):
        N: int = 1000
        data: np.ndarray = np.random.randn(N, ncols)
        labels: np.ndarray = np.random.randint(0, 100, size=N)
        df: DataFrame = DataFrame(data)
        self.labels: np.ndarray = labels
        self.df: DataFrame = df

    def time_sum(self, ncols: int):
        self.df.groupby(self.labels).sum()

class Nth:
    param_names: list[str] = ['dtype']
    params: list[str] = ['float32', 'float64', 'datetime', 'object']

    def setup(self, dtype: str):
        N: int = 10 ** 5
        if dtype == 'datetime':
            values: np.ndarray = date_range('1/1/2011', periods=N, freq='s')
        elif dtype == 'object':
            values: list[str] = ['foo'] * N
        else:
            values: np.ndarray = np.arange(N).astype(dtype)
        key: np.ndarray = np.arange(N)
        self.df: DataFrame = DataFrame({'key': key, 'values': values})
        self.df.iloc[1, 1] = np.nan

    def time_frame_nth_any(self, dtype: str):
        self.df.groupby('key').nth(0, dropna='any')

    def time_groupby_nth_all(self, dtype: str):
        self.df.groupby('key').nth(0, dropna='all')

    def time_frame_nth(self, dtype: str):
        self.df.groupby('key').nth(0)

    def time_series_nth_any(self, dtype: str):
        self.df['values'].groupby(self.df['key']).nth(0, dropna='any')

    def time_series_nth_all(self, dtype: str):
        self.df['values'].groupby(self.df['key']).nth(0, dropna='all')

    def time_series_nth(self, dtype: str):
        self.df['values'].groupby(self.df['key']).nth(0)

class DateAttributes:

    def setup(self):
        rng: date_range = date_range('1/1/2000', '12/31/2005', freq='h')
        self.year, self.month, self.day = (rng.year, rng.month, rng.day)
        self.ts: Series = Series(np.random.randn(len(rng)), index=rng)

    def time_len_groupby_object(self):
        len(self.ts.groupby([self.year, self.month, self.day]))

class Int64:

    def setup(self):
        arr: np.ndarray = np.random.randint(-1 << 12, 1 << 12, (1 << 17, 5))
        i: np.ndarray = np.random.choice(len(arr), len(arr) * 5)
        arr: np.ndarray = np.vstack((arr, arr[i]))
        i: np.ndarray = np.random.permutation(len(arr))
        arr: np.ndarray = arr[i]
        cols: list[str] = list('abcde')
        df: DataFrame = DataFrame(arr, columns=cols)
        df['jim'], df['joe'] = np.random.randn(2, len(df)) * 10
        self.cols: list[str] = cols
        self.df: DataFrame = df

    def time_overflow(self):
        self.df.groupby(self.cols).max()

class CountMultiDtype:

    def setup_cache(self) -> DataFrame:
        n: int = 10000
        offsets: np.ndarray = np.random.randint(n, size=n).astype('timedelta64[ns]')
        dates: np.ndarray = np.datetime64('now') + offsets
        dates[np.random.rand(n) > 0.5] = np.datetime64('nat')
        offsets[np.random.rand(n) > 0.5] = np.timedelta64('nat')
        value2: np.ndarray = np.random.randn(n)
        value2[np.random.rand(n) > 0.5] = np.nan
        obj: np.ndarray = np.random.choice(list('ab'), size=n).astype(object)
        obj[np.random.randn(n) > 0.5] = np.nan
        df: DataFrame = DataFrame({'key1': np.random.randint(0, 500, size=n), 'key2': np.random.randint(0, 100, size=n), 'dates': dates, 'value2': value2, 'value3': np.random.randn(n), 'ints': np.random.randint(0, 1000, size=n), 'obj': obj, 'offsets': offsets})
        return df

    def time_multi_count(self, df: DataFrame):
        df.groupby(['key1', 'key2']).count()

class CountMultiInt:

    def setup_cache(self) -> DataFrame:
        n: int = 10000
        df: DataFrame = DataFrame({'key1': np.random.randint(0, 500, size=n), 'key2': np.random.randint(0, 100, size=n), 'ints': np.random.randint(0, 1000, size=n), 'ints2': np.random.randint(0, 1000, size=n)})
        return df

    def time_multi_int_count(self, df: DataFrame):
        df.groupby(['key1', 'key2']).count()

    def time_multi_int_nunique(self, df: DataFrame):
        df.groupby(['key1', 'key2']).nunique()

class AggFunctions:

    def setup_cache(self) -> DataFrame:
        N: int = 10 ** 5
        fac1: np.ndarray = np.array(['A', 'B', 'C'], dtype='O')
        fac2: np.ndarray = np.array(['one', 'two'], dtype='O')
        df: DataFrame = DataFrame({'key1': fac1.take(np.random.randint(0, 3, size=N)), 'key2': fac2.take(np.random.randint(0, 2, size=N)), 'value1': np.random.randn(N), 'value2': np.random.randn(N), 'value3': np.random.randn(N)})
        return df

    def time_different_str_functions(self, df: DataFrame):
        df.groupby(['key1', 'key2']).agg({'value1': 'mean', 'value2': 'var', 'value3': 'sum'})

    def time_different_str_functions_multicol(self, df: DataFrame):
        df.groupby(['key1', 'key2']).agg(['sum', 'min', 'max'])

    def time_different_str_functions_singlecol(self, df: DataFrame):
        df.groupby('key1').agg({'value1': 'mean', 'value2': 'var', 'value3': 'sum'})

class GroupStrings:

    def setup(self):
        n: int = 2 * 10 ** 5
        alpha: list[str] = list(map(''.join, product(ascii_letters, repeat=4)))
        data: np.ndarray = np.random.choice(alpha, (n // 5, 4), replace=False)
        data: np.ndarray = np.repeat(data, 5, axis=0)
        df: DataFrame = DataFrame(data, columns=list('abcd'))
        df['joe'] = (np.random.randn(len(df)) * 10).round(3)
        df: DataFrame = df.sample(frac=1).reset_index(drop=True)
        self.df: DataFrame = df

    def time_multi_columns(self):
        self.df.groupby(list('abcd')).max()

class MultiColumn:

    def setup_cache(self) -> DataFrame:
        N: int = 10 ** 5
        key1: np.ndarray = np.tile(np.arange(100, dtype=object), 1000)
        key2: np.ndarray = key1.copy()
        np.random.shuffle(key1)
        np.random.shuffle(key2)
        df: DataFrame = DataFrame({'key1': key1, 'key2': key2, 'data1': np.random.randn(N), 'data2': np.random.randn(N)})
        return df

    def time_lambda_sum(self, df: DataFrame):
        df.groupby(['key1', 'key2']).agg(lambda x: x.values.sum())

    def time_cython_sum(self, df: DataFrame):
        df.groupby(['key1', 'key2']).sum()

    def time_col_select_lambda_sum(self, df: DataFrame):
        df.groupby(['key1', 'key2'])['data1'].agg(lambda x: x.values.sum())

    def time_col_select_str_sum(self, df: DataFrame):
        df.groupby(['key1', 'key2'])['data1'].agg('sum')

class Size:

    def setup(self):
        n: int = 10 ** 5
        offsets: np.ndarray = np.random.randint(n, size=n).astype('timedelta64[ns]')
        dates: np.ndarray = np.datetime64('now') + offsets
        self.df: DataFrame = DataFrame({'key1': np.random.randint(0, 500, size=n), 'key2': np.random.randint(0, 100, size=n), 'value1': np.random.randn(n), 'value2': np.random.randn(n), 'value3': np.random.randn(n), 'dates': dates})
        self.draws: Series = Series(np.random.randn(n))
        labels: Series = Series(['foo', 'bar', 'baz', 'qux'] * (n // 4))
        self.cats: Series = labels.astype('category')

    def time_multi_size(self):
        self.df.groupby(['key1', 'key2']).size()

    def time_category_size(self):
        self.draws.groupby(self.cats, observed=True).size()

class Shift:

    def setup(self):
        N: int = 18
        self.df: DataFrame = DataFrame({'g': ['a', 'b'] * 9, 'v': list(range(N))})

    def time_defaults(self):
        self.df.groupby('g').shift()

    def time_fill_value(self):
        self.df.groupby('g').shift(fill_value=99)

class Fillna:

    def setup(self):
        N: int = 100
        self.df: DataFrame = DataFrame({'group': [1] * N + [2] * N, 'value': [np.nan, 1.0] * N}).set_index('group')

    def time_df_ffill(self):
        self.df.groupby('group').ffill()

    def time_df_bfill(self):
        self.df.groupby('group').bfill()

    def time_srs_ffill(self):
        self.df.groupby('group')['value'].ffill()

    def time_srs_bfill(self):
        self.df.groupby('group')['value'].bfill()

class GroupByMethods:
    param_names: list[str] = ['dtype', 'method', 'application', 'ncols']
    params: list[tuple[str, str, str, int]] = [('int', 'int16', 'float', 'object', 'datetime', 'uint'), ('all', 'any', 'bfill', 'count', 'cumcount', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'ffill', 'first', 'head', 'last', 'max', 'min', 'median', 'mean', 'nunique', 'pct_change', 'prod', 'quantile', 'rank', 'sem', 'shift', 'size', 'skew', 'std', 'sum', 'tail', 'unique', 'value_counts', 'var'), ('direct', 'transformation'), (1, 5), ('cython', 'numba')]

    def setup(self, dtype: str, method: str, application: str, ncols: int, engine: str):
        if method in method_blocklist.get(dtype, {}):
            raise NotImplementedError
        if ncols != 1 and method in ['value_counts', 'unique']:
            raise NotImplementedError
        if application == 'transformation' and method in ['describe', 'head', 'tail', 'unique', 'value_counts', 'size']:
            raise NotImplementedError
        if engine == 'numba' and method in _numba_unsupported_methods or ncols > 1 or application == 'transformation' or (dtype == 'datetime'):
            raise NotImplementedError
        if method == 'describe':
            ngroups: int = 20
        elif method == 'skew':
            ngroups: int = 100
        else:
            ngroups: int = 1000
        size: int = ngroups * 2
        rng: np.ndarray = np.arange(ngroups).reshape(-1, 1)
        rng: np.ndarray = np.broadcast_to(rng, (len(rng), ncols))
        taker: np.ndarray = np.random.randint(0, ngroups, size=size)
        values: np.ndarray = rng.take(taker, axis=0)
        if dtype == 'int':
            key: np.ndarray = np.random.randint(0, size, size=size)
        elif dtype in ('int16', 'uint'):
            key: np.ndarray = np.random.randint(0, size, size=size, dtype=dtype)
        elif dtype == 'float':
            key: np.ndarray = np.concatenate([np.random.random(ngroups) * 0.1, np.random.random(ngroups) * 10.0])
        elif dtype == 'object':
            key: list[str] = ['foo'] * size
        elif dtype == 'datetime':
            key: np.ndarray = date_range('1/1/2011', periods=size, freq='s')
        cols: list[str] = [f'values{n}' for n in range(ncols)]
        df: DataFrame = DataFrame(values, columns=cols)
        df['key'] = key
        if len(cols) == 1:
            cols: str = cols[0]
        kwargs: dict = {}
        if engine == 'numba':
            kwargs['engine'] = engine
        if application == 'transformation':
            self.as_group_method = lambda: df.groupby('key')[cols].transform(method, **kwargs)
            self.as_field_method = lambda: df.groupby(cols)['key'].transform(method, **kwargs)
        else:
            self.as_group_method = partial(getattr(df.groupby('key')[cols], method), **kwargs)
            self.as_field_method = partial(getattr(df.groupby(cols)['key'], method), **kwargs)

    def time_dtype_as_group(self, dtype: str, method: str, application: str, ncols: int, engine: str):
        self.as_group_method()

    def time_dtype_as_field(self, dtype: str, method: str, application: str, ncols: int, engine: str):
        self.as_field_method()

class GroupByCythonAgg:
    """
    Benchmarks specifically targeting our cython aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """
    param_names: list[str] = ['dtype', 'method']
    params: list[tuple[str, str]] = [('float64'], ['sum', 'prod', 'min', 'max', 'idxmin', 'idxmax', 'mean', 'median', 'var', 'first', 'last', 'any', 'all']]

    def setup(self, dtype: str, method: str):
        N: int = 1000000
        df: DataFrame = DataFrame(np.random.randn(N, 10), columns=list('abcdefghij'))
        df['key'] = np.random.randint(0, 100, size=N)
        self.df: DataFrame = df

    def time_frame_agg(self, dtype: str, method: str):
        self.df.groupby('key').agg(method)

class GroupByNumbaAgg