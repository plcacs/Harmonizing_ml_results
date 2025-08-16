from functools import partial
from itertools import product
from string import ascii_letters
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pandas import (
    NA,
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
    to_timedelta,
)
from pandas.core.base import SelectionMixin
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

method_blocklist: Dict[str, Set[str]] = {
    "object": {
        "diff",
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "cummin",
        "mean",
        "max",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "min",
        "var",
        "describe",
        "std",
        "quantile",
    },
    "datetime": {
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "mean",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "var",
        "describe",
        "std",
    },
}

_numba_unsupported_methods: List[str] = [
    "all",
    "any",
    "bfill",
    "count",
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "diff",
    "ffill",
    "first",
    "head",
    "idxmax",
    "idxmin",
    "last",
    "median",
    "nunique",
    "pct_change",
    "prod",
    "quantile",
    "rank",
    "sem",
    "shift",
    "size",
    "skew",
    "tail",
    "unique",
    "value_counts",
]


class ApplyDictReturn:
    def setup(self) -> None:
        self.labels: np.ndarray = np.arange(1000).repeat(10)
        self.data: Series = Series(np.random.randn(len(self.labels)))

    def time_groupby_apply_dict_return(self) -> None:
        self.data.groupby(self.labels).apply(
            lambda x: {"first": x.values[0], "last": x.values[-1]}
        )


class Apply:
    param_names: List[str] = ["factor"]
    params: List[int] = [4, 5]

    def setup(self, factor: int) -> None:
        N = 10**factor
        labels = np.random.randint(0, 2000 if factor == 4 else 20, size=N)
        labels2 = np.random.randint(0, 3, size=N)
        df = DataFrame(
            {
                "key": labels,
                "key2": labels2,
                "value1": np.random.randn(N),
                "value2": ["foo", "bar", "baz", "qux"] * (N // 4),
            }
        )
        self.df = df

    def time_scalar_function_multi_col(self, factor: int) -> None:
        self.df.groupby(["key", "key2"]).apply(lambda x: 1)

    def time_scalar_function_single_col(self, factor: int) -> None:
        self.df.groupby("key").apply(lambda x: 1)

    @staticmethod
    def df_copy_function(g: DataFrame) -> DataFrame:
        g.name
        return g.copy()

    def time_copy_function_multi_col(self, factor: int) -> None:
        self.df.groupby(["key", "key2"]).apply(self.df_copy_function)

    def time_copy_overhead_single_col(self, factor: int) -> None:
        self.df.groupby("key").apply(self.df_copy_function)


class ApplyNonUniqueUnsortedIndex:
    def setup(self) -> None:
        idx = np.arange(100)[::-1]
        idx = Index(np.repeat(idx, 200), name="key")
        self.df = DataFrame(np.random.randn(len(idx), 10), index=idx)

    def time_groupby_apply_non_unique_unsorted_index(self) -> None:
        self.df.groupby("key", group_keys=False).apply(lambda x: x)


class Groups:
    param_names: List[str] = ["key"]
    params: List[str] = ["int64_small", "int64_large", "object_small", "object_large"]

    def setup_cache(self) -> Dict[str, Series]:
        size = 10**6
        data = {
            "int64_small": Series(np.random.randint(0, 100, size=size)),
            "int64_large": Series(np.random.randint(0, 10000, size=size)),
            "object_small": Series(
                Index([f"i-{i}" for i in range(100)], dtype=object).take(
                    np.random.randint(0, 100, size=size)
                )
            ),
            "object_large": Series(
                Index([f"i-{i}" for i in range(10000)], dtype=object).take(
                    np.random.randint(0, 10000, size=size)
                )
            ),
        }
        return data

    def setup(self, data: Dict[str, Series], key: str) -> None:
        self.ser = data[key]

    def time_series_groups(self, data: Dict[str, Series], key: str) -> None:
        self.ser.groupby(self.ser).groups

    def time_series_indices(self, data: Dict[str, Series], key: str) -> None:
        self.ser.groupby(self.ser).indices


class GroupManyLabels:
    params: List[int] = [1, 1000]
    param_names: List[str] = ["ncols"]

    def setup(self, ncols: int) -> None:
        N = 1000
        data = np.random.randn(N, ncols)
        self.labels = np.random.randint(0, 100, size=N)
        self.df = DataFrame(data)

    def time_sum(self, ncols: int) -> None:
        self.df.groupby(self.labels).sum()


class Nth:
    param_names: List[str] = ["dtype"]
    params: List[str] = ["float32", "float64", "datetime", "object"]

    def setup(self, dtype: str) -> None:
        N = 10**5
        if dtype == "datetime":
            values = date_range("1/1/2011", periods=N, freq="s")
        elif dtype == "object":
            values = ["foo"] * N
        else:
            values = np.arange(N).astype(dtype)

        key = np.arange(N)
        self.df = DataFrame({"key": key, "values": values})
        self.df.iloc[1, 1] = np.nan

    def time_frame_nth_any(self, dtype: str) -> None:
        self.df.groupby("key").nth(0, dropna="any")

    def time_groupby_nth_all(self, dtype: str) -> None:
        self.df.groupby("key").nth(0, dropna="all")

    def time_frame_nth(self, dtype: str) -> None:
        self.df.groupby("key").nth(0)

    def time_series_nth_any(self, dtype: str) -> None:
        self.df["values"].groupby(self.df["key"]).nth(0, dropna="any")

    def time_series_nth_all(self, dtype: str) -> None:
        self.df["values"].groupby(self.df["key"]).nth(0, dropna="all")

    def time_series_nth(self, dtype: str) -> None:
        self.df["values"].groupby(self.df["key"]).nth(0)


class DateAttributes:
    def setup(self) -> None:
        rng = date_range("1/1/2000", "12/31/2005", freq="h")
        self.year, self.month, self.day = rng.year, rng.month, rng.day
        self.ts = Series(np.random.randn(len(rng)), index=rng)

    def time_len_groupby_object(self) -> None:
        len(self.ts.groupby([self.year, self.month, self.day]))


class Int64:
    def setup(self) -> None:
        arr = np.random.randint(-1 << 12, 1 << 12, (1 << 17, 5))
        i = np.random.choice(len(arr), len(arr) * 5)
        arr = np.vstack((arr, arr[i]))
        i = np.random.permutation(len(arr))
        arr = arr[i]
        self.cols = list("abcde")
        self.df = DataFrame(arr, columns=self.cols)
        self.df["jim"], self.df["joe"] = np.random.randn(2, len(self.df)) * 10

    def time_overflow(self) -> None:
        self.df.groupby(self.cols).max()


class CountMultiDtype:
    def setup_cache(self) -> DataFrame:
        n = 10000
        offsets = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates = np.datetime64("now") + offsets
        dates[np.random.rand(n) > 0.5] = np.datetime64("nat")
        offsets[np.random.rand(n) > 0.5] = np.timedelta64("nat")
        value2 = np.random.randn(n)
        value2[np.random.rand(n) > 0.5] = np.nan
        obj = np.random.choice(list("ab"), size=n).astype(object)
        obj[np.random.randn(n) > 0.5] = np.nan
        df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "dates": dates,
                "value2": value2,
                "value3": np.random.randn(n),
                "ints": np.random.randint(0, 1000, size=n),
                "obj": obj,
                "offsets": offsets,
            }
        )
        return df

    def time_multi_count(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"]).count()


class CountMultiInt:
    def setup_cache(self) -> DataFrame:
        n = 10000
        df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "ints": np.random.randint(0, 1000, size=n),
                "ints2": np.random.randint(0, 1000, size=n),
            }
        )
        return df

    def time_multi_int_count(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"]).count()

    def time_multi_int_nunique(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"]).nunique()


class AggFunctions:
    def setup_cache(self) -> DataFrame:
        N = 10**5
        fac1 = np.array(["A", "B", "C"], dtype="O")
        fac2 = np.array(["one", "two"], dtype="O")
        df = DataFrame(
            {
                "key1": fac1.take(np.random.randint(0, 3, size=N)),
                "key2": fac2.take(np.random.randint(0, 2, size=N)),
                "value1": np.random.randn(N),
                "value2": np.random.randn(N),
                "value3": np.random.randn(N),
            }
        )
        return df

    def time_different_str_functions(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"]).agg(
            {"value1": "mean", "value2": "var", "value3": "sum"}
        )

    def time_different_str_functions_multicol(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"]).agg(["sum", "min", "max"])

    def time_different_str_functions_singlecol(self, df: DataFrame) -> None:
        df.groupby("key1").agg({"value1": "mean", "value2": "var", "value3": "sum"})


class GroupStrings:
    def setup(self) -> None:
        n = 2 * 10**5
        alpha = list(map("".join, product(ascii_letters, repeat=4)))
        data = np.random.choice(alpha, (n // 5, 4), replace=False)
        data = np.repeat(data, 5, axis=0)
        self.df = DataFrame(data, columns=list("abcd"))
        self.df["joe"] = (np.random.randn(len(self.df)) * 10).round(3)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def time_multi_columns(self) -> None:
        self.df.groupby(list("abcd")).max()


class MultiColumn:
    def setup_cache(self) -> DataFrame:
        N = 10**5
        key1 = np.tile(np.arange(100, dtype=object), 1000)
        key2 = key1.copy()
        np.random.shuffle(key1)
        np.random.shuffle(key2)
        df = DataFrame(
            {
                "key1": key1,
                "key2": key2,
                "data1": np.random.randn(N),
                "data2": np.random.randn(N),
            }
        )
        return df

    def time_lambda_sum(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"]).agg(lambda x: x.values.sum())

    def time_cython_sum(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"]).sum()

    def time_col_select_lambda_sum(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"])["data1"].agg(lambda x: x.values.sum())

    def time_col_select_str_sum(self, df: DataFrame) -> None:
        df.groupby(["key1", "key2"])["data1"].agg("sum")


class Size:
    def setup(self) -> None:
        n = 10**5
        offsets = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates = np.datetime64("now") + offsets
        self.df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "value1": np.random.randn(n),
                "value2": np.random.randn(n),
                "value3": np.random.randn(n),
                "dates": dates,
            }
        )
        self.draws = Series(np.random.randn(n))
        labels = Series(["foo", "bar", "baz", "qux"] * (n // 4))
        self.cats = labels.astype("category")

    def time_multi_size(self) -> None:
        self.df.groupby(["key1", "key2"]).size()

    def time_category_size(self) -> None:
        self.draws.groupby(self.cats, observed=True).size()


class Shift:
    def setup(self) -> None:
        N = 18
        self.df = DataFrame({"g": ["a", "b"] * 9, "v": list(range(N))})

    def time_defaults(self) -> None:
        self.df.groupby("g").shift()

    def time_fill_value(self) -> None:
        self.df.groupby("g").shift(fill_value=99)


class Fillna:
    def setup(self) -> None:
        N = 100
        self.df = DataFrame(
            {"group": [1] * N + [2] * N, "value": [np.nan, 1.0] * N}
        ).set_index("group")

    def time_df_ffill(self) -> None:
        self.df.groupby("group").ffill()

    def time_df_bfill(self) -> None:
        self.df.groupby("group").bfill()

    def time_srs_ffill(self) -> None:
        self.df.groupby("group")["value"].ffill()

    def time_srs_bfill(self) -> None:
        self.df.groupby("group")["value"].bfill()


class GroupByMethods:
    param_names: List[str] = ["dtype", "method", "application", "ncols", "engine"]
    params = [
        ["int", "int16", "float", "object", "datetime", "uint"],
        [
            "all",
            "any",
            "bfill",
            "count",
            "cumcount",
            "cummax",
            "cummin",
            "cumprod",
            "cumsum",
            "describe",
            "diff",
            "ffill",
            "first",
            "head",
            "last",
            "max",
            "min",
            "median",
            "mean",
            "nunique",
            "pct_change",
            "prod",
            "quantile",
            "rank",
            "sem",
            "shift",
            "size",
            "skew",
            "std",
            "sum",
            "tail",
            "unique",
            "value_counts",
            "var",
        ],
        ["direct", "transformation"],
        [1, 5],
        ["cython", "numba"],
    ]

    def setup(
        self,
        dtype: str,
        method: str,
        application: str,
        ncols: int,
        engine: str,
    ) -> None:
        if method in method_blocklist.get(dtype, {}):
            raise NotImplementedError

        if ncols != 1 and method in ["value_counts", "unique"]:
            raise NotImplementedError

        if application == "transformation" and method in [
            "describe",
            "head",
            "tail",
            "unique",
            "value_counts",
            "size",
        ]:
            raise NotImplementedError

        if (
            (