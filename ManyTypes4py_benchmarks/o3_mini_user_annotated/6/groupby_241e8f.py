from functools import partial
from itertools import product
from string import ascii_letters
from typing import Any, Callable, Dict, List, Union

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

method_blocklist: Dict[str, set] = {
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
        N: int = 10 ** factor
        labels: np.ndarray = np.random.randint(0, 2000 if factor == 4 else 20, size=N)
        labels2: np.ndarray = np.random.randint(0, 3, size=N)
        df: DataFrame = DataFrame(
            {
                "key": labels,
                "key2": labels2,
                "value1": np.random.randn(N),
                "value2": ["foo", "bar", "baz", "qux"] * (N // 4),
            }
        )
        self.df: DataFrame = df

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
        idx: Index = np.arange(100)[::-1]
        idx = Index(np.repeat(idx, 200), name="key")
        self.df: DataFrame = DataFrame(np.random.randn(len(idx), 10), index=idx)

    def time_groupby_apply_non_unique_unsorted_index(self) -> None:
        self.df.groupby("key", group_keys=False).apply(lambda x: x)


class Groups:
    param_names: List[str] = ["key"]
    params: List[str] = ["int64_small", "int64_large", "object_small", "object_large"]

    def setup_cache(self) -> Dict[str, Series]:
        size: int = 10 ** 6
        data: Dict[str, Series] = {
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
        self.ser: Series = data[key]

    def time_series_groups(self, data: Dict[str, Series], key: str) -> None:
        self.ser.groupby(self.ser).groups

    def time_series_indices(self, data: Dict[str, Series], key: str) -> None:
        self.ser.groupby(self.ser).indices


class GroupManyLabels:
    params: List[int] = [1, 1000]
    param_names: List[str] = ["ncols"]

    def setup(self, ncols: int) -> None:
        N: int = 1000
        data: np.ndarray = np.random.randn(N, ncols)
        self.labels: np.ndarray = np.random.randint(0, 100, size=N)
        self.df: DataFrame = DataFrame(data)

    def time_sum(self, ncols: int) -> None:
        self.df.groupby(self.labels).sum()


class Nth:
    param_names: List[str] = ["dtype"]
    params: List[str] = ["float32", "float64", "datetime", "object"]

    def setup(self, dtype: str) -> None:
        N: int = 10 ** 5
        if dtype == "datetime":
            values = date_range("1/1/2011", periods=N, freq="s")
        elif dtype == "object":
            values = ["foo"] * N
        else:
            values = np.arange(N).astype(dtype)
        key: np.ndarray = np.arange(N)
        self.df: DataFrame = DataFrame({"key": key, "values": values})
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
        self.year: Any = rng.year
        self.month: Any = rng.month
        self.day: Any = rng.day
        self.ts: Series = Series(np.random.randn(len(rng)), index=rng)

    def time_len_groupby_object(self) -> None:
        len(self.ts.groupby([self.year, self.month, self.day]))


class Int64:
    def setup(self) -> None:
        arr: np.ndarray = np.random.randint(-1 << 12, 1 << 12, (1 << 17, 5))
        i: np.ndarray = np.random.choice(len(arr), len(arr) * 5)
        arr = np.vstack((arr, arr[i]))
        i = np.random.permutation(len(arr))
        arr = arr[i]
        self.cols: List[str] = list("abcde")
        self.df: DataFrame = DataFrame(arr, columns=self.cols)
        self.df["jim"] = np.random.randn(len(self.df)) * 10
        self.df["joe"] = np.random.randn(len(self.df)) * 10

    def time_overflow(self) -> None:
        self.df.groupby(self.cols).max()


class CountMultiDtype:
    def setup_cache(self) -> DataFrame:
        n: int = 10000
        offsets: np.ndarray = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates: np.ndarray = np.datetime64("now") + offsets
        dates[np.random.rand(n) > 0.5] = np.datetime64("nat")
        offsets[np.random.rand(n) > 0.5] = np.timedelta64("nat")
        value2: np.ndarray = np.random.randn(n)
        value2[np.random.rand(n) > 0.5] = np.nan
        obj: np.ndarray = np.random.choice(list("ab"), size=n).astype(object)
        obj[np.random.randn(n) > 0.5] = np.nan
        df: DataFrame = DataFrame(
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
        n: int = 10000
        df: DataFrame = DataFrame(
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
        N: int = 10 ** 5
        fac1: np.ndarray = np.array(["A", "B", "C"], dtype="O")
        fac2: np.ndarray = np.array(["one", "two"], dtype="O")
        df: DataFrame = DataFrame(
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
        n: int = 2 * 10 ** 5
        alpha: List[str] = list(map("".join, product(ascii_letters, repeat=4)))
        data: np.ndarray = np.random.choice(alpha, (n // 5, 4), replace=False)
        data = np.repeat(data, 5, axis=0)
        self.df: DataFrame = DataFrame(data, columns=list("abcd"))
        self.df["joe"] = (np.random.randn(len(self.df)) * 10).round(3)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def time_multi_columns(self) -> None:
        self.df.groupby(list("abcd")).max()


class MultiColumn:
    def setup_cache(self) -> DataFrame:
        N: int = 10 ** 5
        key1: np.ndarray = np.tile(np.arange(100, dtype=object), 1000)
        key2: np.ndarray = key1.copy()
        np.random.shuffle(key1)
        np.random.shuffle(key2)
        df: DataFrame = DataFrame(
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
        n: int = 10 ** 5
        offsets: np.ndarray = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates: np.ndarray = np.datetime64("now") + offsets
        self.df: DataFrame = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "value1": np.random.randn(n),
                "value2": np.random.randn(n),
                "value3": np.random.randn(n),
                "dates": dates,
            }
        )
        self.draws: Series = Series(np.random.randn(n))
        labels: Series = Series(["foo", "bar", "baz", "qux"] * (n // 4))
        self.cats: Categorical = labels.astype("category")

    def time_multi_size(self) -> None:
        self.df.groupby(["key1", "key2"]).size()

    def time_category_size(self) -> None:
        self.draws.groupby(self.cats, observed=True).size()


class Shift:
    def setup(self) -> None:
        N: int = 18
        self.df: DataFrame = DataFrame({"g": ["a", "b"] * 9, "v": list(range(N))})

    def time_defaults(self) -> None:
        self.df.groupby("g").shift()

    def time_fill_value(self) -> None:
        self.df.groupby("g").shift(fill_value=99)


class Fillna:
    def setup(self) -> None:
        N: int = 100
        self.df: DataFrame = DataFrame(
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
    param_names: List[str] = ["dtype", "method", "application", "ncols"]
    params: List[List[Any]] = [
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

    def setup(self, dtype: str, method: str, application: str, ncols: int, engine: str) -> None:
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
            (engine == "numba" and method in _numba_unsupported_methods)
            or ncols > 1
            or application == "transformation"
            or dtype == "datetime"
        ):
            raise NotImplementedError
        if method == "describe":
            ngroups: int = 20
        elif method == "skew":
            ngroups = 100
        else:
            ngroups = 1000
        size: int = ngroups * 2
        rng: np.ndarray = np.arange(ngroups).reshape(-1, 1)
        rng = np.broadcast_to(rng, (len(rng), ncols))
        taker: np.ndarray = np.random.randint(0, ngroups, size=size)
        values: np.ndarray = rng.take(taker, axis=0)
        if dtype == "int":
            key: np.ndarray = np.random.randint(0, size, size=size)
        elif dtype in ("int16", "uint"):
            key = np.random.randint(0, size, size=size, dtype=dtype)
        elif dtype == "float":
            key = np.concatenate(
                [np.random.random(ngroups) * 0.1, np.random.random(ngroups) * 10.0]
            )
        elif dtype == "object":
            key = ["foo"] * size
        elif dtype == "datetime":
            key = date_range("1/1/2011", periods=size, freq="s")
        cols: Union[List[str], str] = [f"values{n}" for n in range(ncols)]
        df: DataFrame = DataFrame(values, columns=[f"values{n}" for n in range(ncols)])
        df["key"] = key
        if isinstance(cols, list) and len(cols) == 1:
            cols = cols[0]
        kwargs: Dict[str, Any] = {}
        if engine == "numba":
            kwargs["engine"] = engine
        if application == "transformation":
            self.as_group_method: Callable[[], Any] = lambda: df.groupby("key")[cols].transform(
                method, **kwargs
            )
            self.as_field_method: Callable[[], Any] = lambda: df.groupby(cols)["key"].transform(
                method, **kwargs
            )
        else:
            self.as_group_method = partial(getattr(df.groupby("key")[cols], method), **kwargs)
            self.as_field_method = partial(getattr(df.groupby(cols)["key"], method), **kwargs)

    def time_dtype_as_group(self, dtype: str, method: str, application: str, ncols: int, engine: str) -> None:
        self.as_group_method()

    def time_dtype_as_field(self, dtype: str, method: str, application: str, ncols: int, engine: str) -> None:
        self.as_field_method()


class GroupByCythonAgg:
    param_names: List[str] = ["dtype", "method"]
    params: List[List[str]] = [
        ["float64"],
        [
            "sum",
            "prod",
            "min",
            "max",
            "idxmin",
            "idxmax",
            "mean",
            "median",
            "var",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype: str, method: str) -> None:
        N: int = 1_000_000
        df: DataFrame = DataFrame(np.random.randn(N, 10), columns=list("abcdefghij"))
        df["key"] = np.random.randint(0, 100, size=N)
        self.df: DataFrame = df

    def time_frame_agg(self, dtype: str, method: str) -> None:
        self.df.groupby("key").agg(method)


class GroupByNumbaAgg(GroupByCythonAgg):
    def setup(self, dtype: str, method: str) -> None:
        if method in _numba_unsupported_methods:
            raise NotImplementedError
        super().setup(dtype, method)

    def time_frame_agg(self, dtype: str, method: str) -> None:
        self.df.groupby("key").agg(method, engine="numba")


class GroupByCythonAggEaDtypes:
    param_names: List[str] = ["dtype", "method"]
    params: List[List[str]] = [
        ["Float64", "Int64", "Int32"],
        [
            "sum",
            "prod",
            "min",
            "max",
            "mean",
            "median",
            "var",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype: str, method: str) -> None:
        N: int = 1_000_000
        df: DataFrame = DataFrame(
            np.random.randint(0, high=100, size=(N, 10)),
            columns=list("abcdefghij"),
            dtype=dtype,
        )
        df.loc[list(range(1, N, 5)), list("abcdefghij")] = NA
        df["key"] = np.random.randint(0, 100, size=N)
        self.df: DataFrame = df

    def time_frame_agg(self, dtype: str, method: str) -> None:
        self.df.groupby("key").agg(method)


class Cumulative:
    param_names: List[str] = ["dtype", "method", "with_nans"]
    params: List[List[Union[str, bool]]] = [
        ["float64", "int64", "Float64", "Int64"],
        ["cummin", "cummax", "cumsum"],
        [True, False],
    ]

    def setup(self, dtype: str, method: str, with_nans: bool) -> None:
        if with_nans and dtype == "int64":
            raise NotImplementedError("Construction of df would raise")
        N: int = 500_000
        keys: np.ndarray = np.random.randint(0, 100, size=N)
        vals: np.ndarray = np.random.randint(-10, 10, (N, 5))
        if with_nans:
            null_vals: np.ndarray = vals.astype(float, copy=True)
            null_vals[::2, :] = np.nan
            null_vals[::3, :] = np.nan
            df: DataFrame = DataFrame(null_vals, columns=list("abcde"), dtype=dtype)
            df["key"] = keys
            self.df: DataFrame = df
        else:
            df = DataFrame(vals, columns=list("abcde")).astype(dtype, copy=False)
            df["key"] = keys
            self.df = df

    def time_frame_transform(self, dtype: str, method: str, with_nans: bool) -> None:
        self.df.groupby("key").transform(method)


class RankWithTies:
    param_names: List[str] = ["dtype", "tie_method"]
    params: List[List[str]] = [
        ["float64", "float32", "int64", "datetime64"],
        ["first", "average", "dense", "min", "max"],
    ]

    def setup(self, dtype: str, tie_method: str) -> None:
        N: int = 10 ** 4
        if dtype == "datetime64":
            data: np.ndarray = np.array([Timestamp("2011/01/01")] * N, dtype=dtype)
        else:
            data = np.ones(N, dtype=dtype)
        self.df: DataFrame = DataFrame({"values": data, "key": ["foo"] * N})

    def time_rank_ties(self, dtype: str, tie_method: str) -> None:
        self.df.groupby("key").rank(method=tie_method)


class Float32:
    def setup(self) -> None:
        tmp1: np.ndarray = (np.random.random(10000) * 0.1).astype(np.float32)
        tmp2: np.ndarray = (np.random.random(10000) * 10.0).astype(np.float32)
        tmp: np.ndarray = np.concatenate((tmp1, tmp2))
        arr: np.ndarray = np.repeat(tmp, 10)
        self.df: DataFrame = DataFrame({"a": arr, "b": arr})

    def time_sum(self) -> None:
        self.df.groupby(["a"])["b"].sum()


class String:
    param_names: List[str] = ["dtype", "method"]
    params: List[List[str]] = [
        ["str", "string[python]"],
        [
            "sum",
            "min",
            "max",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype: str, method: str) -> None:
        cols: List[str] = list("abcdefghjkl")
        self.df: DataFrame = DataFrame(
            np.random.randint(0, 100, size=(10_000, len(cols))),
            columns=cols,
            dtype=dtype,
        )

    def time_str_func(self, dtype: str, method: str) -> None:
        self.df.groupby("a")[self.df.columns[1:]].agg(method)


class Categories:
    params: List[bool] = [True, False]
    param_names: List[str] = ["observed"]

    def setup(self, observed: bool) -> None:
        N: int = 10 ** 5
        arr: np.ndarray = np.random.random(N)
        data: Dict[str, Any] = {"a": Categorical(np.random.randint(10000, size=N)), "b": arr}
        self.df: DataFrame = DataFrame(data)
        data = {
            "a": Categorical(np.random.randint(10000, size=N), ordered=True),
            "b": arr,
        }
        self.df_ordered: DataFrame = DataFrame(data)
        data = {
            "a": Categorical(
                np.random.randint(100, size=N), categories=np.arange(10000)
            ),
            "b": arr,
        }
        self.df_extra_cat: DataFrame = DataFrame(data)

    def time_groupby_sort(self, observed: bool) -> None:
        self.df.groupby("a", observed=observed)["b"].count()

    def time_groupby_nosort(self, observed: bool) -> None:
        self.df.groupby("a", observed=observed, sort=False)["b"].count()

    def time_groupby_ordered_sort(self, observed: bool) -> None:
        self.df_ordered.groupby("a", observed=observed)["b"].count()

    def time_groupby_ordered_nosort(self, observed: bool) -> None:
        self.df_ordered.groupby("a", observed=observed, sort=False)["b"].count()

    def time_groupby_extra_cat_sort(self, observed: bool) -> None:
        self.df_extra_cat.groupby("a", observed=observed)["b"].count()

    def time_groupby_extra_cat_nosort(self, observed: bool) -> None:
        self.df_extra_cat.groupby("a", observed=observed, sort=False)["b"].count()


class MultipleCategories:
    def setup(self) -> None:
        N: int = 10 ** 3
        arr: np.ndarray = np.random.random(N)
        data: Dict[str, Any] = {
            "a1": Categorical(np.random.randint(10000, size=N)),
            "a2": Categorical(np.random.randint(10000, size=N)),
            "b": arr,
        }
        self.df: DataFrame = DataFrame(data)
        data = {
            "a1": Categorical(np.random.randint(10000, size=N), ordered=True),
            "a2": Categorical(np.random.randint(10000, size=N), ordered=True),
            "b": arr,
        }
        self.df_ordered: DataFrame = DataFrame(data)
        data = {
            "a1": Categorical(np.random.randint(100, size=N), categories=np.arange(N)),
            "a2": Categorical(np.random.randint(100, size=N), categories=np.arange(N)),
            "b": arr,
        }
        self.df_extra_cat: DataFrame = DataFrame(data)

    def time_groupby_sort(self) -> None:
        self.df.groupby(["a1", "a2"], observed=False)["b"].count()

    def time_groupby_nosort(self) -> None:
        self.df.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    def time_groupby_ordered_sort(self) -> None:
        self.df_ordered.groupby(["a1", "a2"], observed=False)["b"].count()

    def time_groupby_ordered_nosort(self) -> None:
        self.df_ordered.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    def time_groupby_extra_cat_sort(self) -> None:
        self.df_extra_cat.groupby(["a1", "a2"], observed=False)["b"].count()

    def time_groupby_extra_cat_nosort(self) -> None:
        self.df_extra_cat.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    def time_groupby_transform(self) -> None:
        self.df_extra_cat.groupby(["a1", "a2"], observed=False)["b"].cumsum()


class Datelike:
    params: List[str] = ["period_range", "date_range", "date_range_tz"]
    param_names: List[str] = ["grouper"]

    def setup(self, grouper: str) -> None:
        N: int = 10 ** 4
        rng_map: Dict[str, Callable[..., Any]] = {
            "period_range": period_range,
            "date_range": date_range,
            "date_range_tz": partial(date_range, tz="US/Central"),
        }
        self.grouper: Any = rng_map[grouper]("1900-01-01", freq="D", periods=N)
        self.df: DataFrame = DataFrame(np.random.randn(10 ** 4, 2))

    def time_sum(self, grouper: str) -> None:
        self.df.groupby(self.grouper).sum()


class SumBools:
    def setup(self) -> None:
        N: int = 500
        self.df: DataFrame = DataFrame({"ii": list(range(N)), "bb": [True] * N})

    def time_groupby_sum_booleans(self) -> None:
        self.df.groupby("ii").sum()


class SumMultiLevel:
    timeout: float = 120.0

    def setup(self) -> None:
        N: int = 50
        self.df: DataFrame = DataFrame(
            {"A": list(range(N)) * 2, "B": list(range(N * 2)), "C": 1}
        ).set_index(["A", "B"])

    def time_groupby_sum_multiindex(self) -> None:
        self.df.groupby(level=[0, 1]).sum()


class SumTimeDelta:
    def setup(self) -> None:
        N: int = 10 ** 4
        self.df: DataFrame = DataFrame(
            np.random.randint(1000, 100000, (N, 100)),
            index=np.random.randint(200, size=(N,)),
        ).astype("timedelta64[ns]")
        self.df_int: DataFrame = self.df.copy().astype("int64")

    def time_groupby_sum_timedelta(self) -> None:
        self.df.groupby(lambda x: x).sum()

    def time_groupby_sum_int(self) -> None:
        self.df_int.groupby(lambda x: x).sum()


class Transform:
    def setup(self) -> None:
        n1: int = 400
        n2: int = 250
        index: MultiIndex = MultiIndex(
            levels=[np.arange(n1), Index([f"i-{i}" for i in range(n2)], dtype=object)],
            codes=[np.repeat(list(range(n1)), n2).tolist(), list(range(n2)) * n1],
            names=["lev1", "lev2"],
        )
        arr: np.ndarray = np.random.randn(n1 * n2, 3)
        arr[::10000, 0] = np.nan
        arr[1::10000, 1] = np.nan
        arr[2::10000, 2] = np.nan
        data: DataFrame = DataFrame(arr, index=index, columns=["col1", "col20", "col3"])
        self.df: DataFrame = data

        n: int = 1000
        self.df_wide: DataFrame = DataFrame(
            np.random.randn(n, n),
            index=np.random.choice(range(10), n),
        )

        n = 1_000_000
        self.df_tall: DataFrame = DataFrame(
            np.random.randn(n, 3),
            index=np.random.randint(0, 5, n),
        )

        n = 20000
        self.df1: DataFrame = DataFrame(
            np.random.randint(1, n, (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df2: DataFrame = self.df1.copy()
        self.df2["jim"] = self.df2["joe"]

        self.df3: DataFrame = DataFrame(
            np.random.randint(1, (n / 10), (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df4: DataFrame = self.df3.copy()
        self.df4["jim"] = self.df4["joe"]

    def time_transform_lambda_max(self) -> None:
        self.df.groupby(level="lev1").transform(lambda x: max(x))

    def time_transform_str_max(self) -> None:
        self.df.groupby(level="lev1").transform("max")

    def time_transform_lambda_max_tall(self) -> None:
        self.df_tall.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    def time_transform_lambda_max_wide(self) -> None:
        self.df_wide.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    def time_transform_multi_key1(self) -> None:
        self.df1.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key2(self) -> None:
        self.df2.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key3(self) -> None:
        self.df3.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key4(self) -> None:
        self.df4.groupby(["jim", "joe"])["jolie"].transform("max")


class TransformBools:
    def setup(self) -> None:
        N: int = 120000
        transition_points: np.ndarray = np.sort(np.random.choice(np.arange(N), 1400))
        transitions: np.ndarray = np.zeros(N, dtype=np.bool_)
        transitions[transition_points] = True
        self.g: np.ndarray = transitions.cumsum()
        self.df: DataFrame = DataFrame({"signal": np.random.rand(N)})

    def time_transform_mean(self) -> None:
        self.df["signal"].groupby(self.g).transform("mean")


class TransformNaN:
    def setup(self) -> None:
        self.df_nans: DataFrame = DataFrame(
            {"key": np.repeat(np.arange(1000), 10), "B": np.nan, "C": np.nan}
        )
        self.df_nans.loc[4::10, "B":"C"] = 5

    def time_first(self) -> None:
        self.df_nans.groupby("key").transform("first")


class TransformEngine:
    param_names: List[str] = ["parallel"]
    params: List[List[bool]] = [[True, False]]

    def setup(self, parallel: bool) -> None:
        N: int = 10 ** 3
        data: DataFrame = DataFrame(
            {0: [str(i) for i in range(100)] * N, 1: list(range(100)) * N},
            columns=[0, 1],
        )
        self.parallel: bool = parallel
        self.grouper: Any = data.groupby(0)

    def time_series_numba(self, parallel: bool) -> None:
        def function(values: Series, index: Any) -> Series:
            return values * 5

        self.grouper[1].transform(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_series_cython(self, parallel: bool) -> None:
        def function(values: Series) -> Series:
            return values * 5

        self.grouper[1].transform(function, engine="cython")

    def time_dataframe_numba(self, parallel: bool) -> None:
        def function(values: DataFrame, index: Any) -> DataFrame:
            return values * 5

        self.grouper.transform(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_dataframe_cython(self, parallel: bool) -> None:
        def function(values: DataFrame) -> DataFrame:
            return values * 5

        self.grouper.transform(function, engine="cython")


class AggEngine:
    param_names: List[str] = ["parallel"]
    params: List[List[bool]] = [[True, False]]

    def setup(self, parallel: bool) -> None:
        N: int = 10 ** 3
        data: DataFrame = DataFrame(
            {0: [str(i) for i in range(100)] * N, 1: list(range(100)) * N},
            columns=[0, 1],
        )
        self.parallel: bool = parallel
        self.grouper: Any = data.groupby(0)

    def time_series_numba(self, parallel: bool) -> None:
        def function(values: Series, index: Any) -> Any:
            total: int = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper[1].agg(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_series_cython(self, parallel: bool) -> None:
        def function(values: Series) -> Any:
            total: int = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper[1].agg(function, engine="cython")

    def time_dataframe_numba(self, parallel: bool) -> None:
        def function(values: DataFrame, index: Any) -> Any:
            total: int = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper.agg(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_dataframe_cython(self, parallel: bool) -> None:
        def function(values: DataFrame) -> Any:
            total: int = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper.agg(function, engine="cython")


class Sample:
    def setup(self) -> None:
        N: int = 10 ** 3
        self.df: DataFrame = DataFrame({"a": np.zeros(N)})
        self.groups: np.ndarray = np.arange(0, N)
        self.weights: np.ndarray = np.ones(N)

    def time_sample(self) -> None:
        self.df.groupby(self.groups).sample(n=1)

    def time_sample_weights(self) -> None:
        self.df.groupby(self.groups).sample(n=1, weights=self.weights)


class Resample:
    def setup(self) -> None:
        num_timedeltas: int = 20_000
        num_groups: int = 3
        index: MultiIndex = MultiIndex.from_product(
            [
                np.arange(num_groups),
                to_timedelta(np.arange(num_timedeltas), unit="s"),
            ],
            names=["groups", "timedeltas"],
        )
        data: np.ndarray = np.random.randint(0, 1000, size=(len(index)))
        self.df: DataFrame = DataFrame(data, index=index).reset_index("timedeltas")
        self.df_multiindex: DataFrame = DataFrame(data, index=index)

    def time_resample(self) -> None:
        self.df.groupby(level="groups").resample("10s", on="timedeltas").mean()

    def time_resample_multiindex(self) -> None:
        self.df_multiindex.groupby(level="groups").resample(
            "10s", level="timedeltas"
        ).mean()


from .pandas_vb_common import setup  # noqa: F401 isort:skip