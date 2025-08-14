import string
from typing import List, Tuple, Union, Optional, Any

import numpy as np
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    array,
    concat,
    date_range,
    merge,
    merge_asof,
)

try:
    from pandas import merge_ordered
except ImportError:
    from pandas import ordered_merge as merge_ordered


class Concat:
    params: List[int] = [0, 1]
    param_names: List[str] = ["axis"]

    def setup(self, axis: int) -> None:
        N = 1000
        s = Series(N, index=Index([f"i-{i}" for i in range(N)], dtype=object))
        self.series: List[Series] = [s[i:-i] for i in range(1, 10)] * 50
        self.small_frames: List[DataFrame] = [DataFrame(np.random.randn(5, 4))] * 1000
        df = DataFrame(
            {"A": range(N)}, index=date_range("20130101", periods=N, freq="s")
        )
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
    params: List[List[Union[int, bool]] = [[0, 1], [True, False]]
    param_names: List[str] = ["axis", "ignore_index"]

    def setup(self, axis: int, ignore_index: bool) -> None:
        frame_c = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="C"))
        self.frame_c: List[DataFrame] = [frame_c] * 20
        frame_f = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="F"))
        self.frame_f: List[DataFrame] = [frame_f] * 20

    def time_c_ordered(self, axis: int, ignore_index: bool) -> None:
        concat(self.frame_c, axis=axis, ignore_index=ignore_index)

    def time_f_ordered(self, axis: int, ignore_index: bool) -> None:
        concat(self.frame_f, axis=axis, ignore_index=ignore_index)


class ConcatIndexDtype:
    params: Tuple[List[str], List[str], List[int], List[bool]] = (
        [
            "datetime64[ns]",
            "int64",
            "Int64",
            "int64[pyarrow]",
            "string[python]",
            "string[pyarrow]",
        ],
        ["monotonic", "non_monotonic", "has_na"],
        [0, 1],
        [True, False],
    )
    param_names: List[str] = ["dtype", "structure", "axis", "sort"]

    def setup(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        N = 10_000
        if dtype == "datetime64[ns]":
            vals = date_range("1970-01-01", periods=N)
        elif dtype in ("int64", "Int64", "int64[pyarrow]"):
            vals = np.arange(N, dtype=np.int64)
        elif dtype in ("string[python]", "string[pyarrow]"):
            vals = Index([f"i-{i}" for i in range(N)], dtype=object)
        else:
            raise NotImplementedError

        idx = Index(vals, dtype=dtype)

        if structure == "monotonic":
            idx = idx.sort_values()
        elif structure == "non_monotonic":
            idx = idx[::-1]
        elif structure == "has_na":
            if not idx._can_hold_na:
                raise NotImplementedError
            idx = Index([None], dtype=dtype).append(idx)
        else:
            raise NotImplementedError

        self.series: List[Series] = [Series(i, idx[:-i]) for i in range(1, 6)]

    def time_concat_series(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        concat(self.series, axis=axis, sort=sort)


class Join:
    params: List[bool] = [True, False]
    param_names: List[str] = ["sort"]

    def setup(self, sort: bool) -> None:
        level1 = Index([f"i-{i}" for i in range(10)], dtype=object).values
        level2 = Index([f"i-{i}" for i in range(1000)], dtype=object).values
        codes1 = np.arange(10).repeat(1000)
        codes2 = np.tile(np.arange(1000), 10)
        index2 = MultiIndex(levels=[level1, level2], codes=[codes1, codes2])
        self.df_multi = DataFrame(
            np.random.randn(len(index2), 4), index=index2, columns=["A", "B", "C", "D"]
        )

        self.key1 = np.tile(level1.take(codes1), 10)
        self.key2 = np.tile(level2.take(codes2), 10)
        self.df = DataFrame(
            {
                "data1": np.random.randn(100000),
                "data2": np.random.randn(100000),
                "key1": self.key1,
                "key2": self.key2,
            }
        )

        self.df_key1 = DataFrame(
            np.random.randn(len(level1), 4), index=level1, columns=["A", "B", "C", "D"]
        )
        self.df_key2 = DataFrame(
            np.random.randn(len(level2), 4), index=level2, columns=["A", "B", "C", "D"]
        )

        shuf = np.arange(100000)
        np.random.shuffle(shuf)
        self.df_shuf = self.df.reindex(self.df.index[shuf])

    def time_join_dataframe_index_multi(self, sort: bool) -> None:
        self.df.join(self.df_multi, on=["key1", "key2"], sort=sort)

    def time_join_dataframe_index_single_key_bigger(self, sort: bool) -> None:
        self.df.join(self.df_key2, on="key2", sort=sort)

    def time_join_dataframe_index_single_key_small(self, sort: bool) -> None:
        self.df.join(self.df_key1, on="key1", sort=sort)

    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort: bool) -> None:
        self.df_shuf.join(self.df_key2, on="key2", sort=sort)

    def time_join_dataframes_cross(self, sort: bool) -> None:
        self.df.loc[:2000].join(self.df_key1, how="cross", sort=sort)


class JoinIndex:
    def setup(self) -> None:
        N = 5000
        self.left = DataFrame(
            np.random.randint(1, N / 50, (N, 2)), columns=["jim", "joe"]
        )
        self.right = DataFrame(
            np.random.randint(1, N / 50, (N, 2)), columns=["jolie", "jolia"]
        ).set_index("jolie")

    def time_left_outer_join_index(self) -> None:
        self.left.join(self.right, on="jim")


class JoinMultiindexSubset:
    def setup(self) -> None:
        N = 100_000
        mi1 = MultiIndex.from_arrays([np.arange(N)] * 4, names=["a", "b", "c", "d"])
        mi2 = MultiIndex.from_arrays([np.arange(N)] * 2, names=["a", "b"])
        self.left = DataFrame({"col1": 1}, index=mi1)
        self.right = DataFrame({"col2": 2}, index=mi2)

    def time_join_multiindex_subset(self) -> None:
        self.left.join(self.right)


class JoinEmpty:
    def setup(self) -> None:
        N = 100_000
        self.df = DataFrame({"A": np.arange(N)})
        self.df_empty = DataFrame(columns=["B", "C"], dtype="int64")

    def time_inner_join_left_empty(self) -> None:
        self.df_empty.join(self.df, how="inner")

    def time_inner_join_right_empty(self) -> None:
        self.df.join(self.df_empty, how="inner")


class JoinNonUnique:
    def setup(self) -> None:
        date_index = date_range("01-Jan-2013", "23-Jan-2013", freq="min")
        daily_dates = date_index.to_period("D").to_timestamp("s", "s")
        self.fracofday = date_index.values - daily_dates.values
        self.fracofday = self.fracofday.astype("timedelta64[ns]")
        self.fracofday = self.fracofday.astype(np.float64) / 86_400_000_000_000
        self.fracofday = Series(self.fracofday, daily_dates)
        index = date_range(date_index.min(), date_index.max(), freq="D")
        self.temp = Series(1.0, index)[self.fracofday.index]

    def time_join_non_unique_equal(self) -> None:
        self.fracofday * self.temp


class Merge:
    params: List[bool] = [True, False]
    param_names: List[str] = ["sort"]

    def setup(self, sort: bool) -> None:
        N = 10000
        indices = Index([f"i-{i}" for i in range(N)], dtype=object).values
        indices2 = Index([f"i-{i}" for i in range(N)], dtype=object).values
        key = np.tile(indices[:8000], 10)
        key2 = np.tile(indices2[:8000], 10)
        self.left = DataFrame(
            {"key": key, "key2": key2, "value": np.random.randn(80000)}
        )
        self.right = DataFrame(
            {
                "key": indices[2000:],
                "key2": indices2[2000:],
                "value2": np.random.randn(8000),
            }
        )

        self.df = DataFrame(
            {
                "key1": np.tile(np.arange(500).repeat(10), 2),
                "key2": np.tile(np.arange(250).repeat(10), 4),
                "value": np.random.randn(10000),
            }
        )
        self.df2 = DataFrame({"key1": np.arange(500), "value2": np.random.randn(500)})
        self.df3 = self.df[:5000]

    def time_merge_2intkey(self, sort: bool) -> None:
        merge(self.left, self.right, sort=sort)

    def time_merge_dataframe_integer_2key(self, sort: bool) -> None:
        merge(self.df, self.df3, sort=sort)

    def time_merge_dataframe_integer_key(self, sort: bool) -> None:
        merge(self.df, self.df2, on="key1", sort=sort)

    def time_merge_dataframe_empty_right(self, sort: bool) -> None:
        merge(self.left, self.right.iloc[:0], sort=sort)

    def time_merge_dataframe_empty_left(self, sort: bool) -> None:
        merge(self.left.iloc[:0], self.right, sort=sort)

    def time_merge_dataframes_cross(self, sort: bool) -> None:
        merge(self.left.loc[:2000], self.right.loc[:2000], how="cross", sort=sort)


class MergeEA:
    params: List[List[str], List[bool]] = [
        [
            "Int64",
            "Int32",
            "Int16",
            "UInt64",
            "UInt32",
            "UInt16",
            "Float64",
            "Float32",
        ],
        [True, False],
    ]
    param_names: List[str] = ["dtype", "monotonic"]

    def setup(self, dtype: str, monotonic: bool) -> None:
        N = 10_000
        indices = np.arange(1, N)
        key = np.tile(indices[:8000], 10)
        self.left = DataFrame(
            {"key": Series(key, dtype=dtype), "value": np.random.randn(80000)}
        )
        self.right = DataFrame(
            {
                "key": Series(indices[2000:], dtype=dtype),
                "value2": np.random.randn(7999),
            }
        )
        if monotonic:
            self.left = self.left.sort_values("key")
            self.right = self.right.sort_values("key")

    def time_merge(self, dtype: str, monotonic: bool) -> None:
        merge(self.left, self.right)


class I8Merge:
    params: List[str] = ["inner", "outer", "left", "right"]
    param_names: List[str] = ["how"]

    def setup(self, how: str) -> None:
        low, high, n = -1000, 1000, 10**6
        self.left = DataFrame(
            np.random.randint(low, high, (n, 7)), columns=list("ABCDEFG")
        )
        self.left["left"] = self.left.sum(axis=1)
        self.right = self.left.sample(frac=1).rename({"left": "right"}, axis=1)
        self.right = self.right.reset_index(drop=True)
        self.right["right"] *= -1

    def time_i8merge(self, how: str) -> None:
        merge(self.left, self.right, how=how)


class UniqueMerge:
    params: List[int] = [4_000_000, 1_000_000]
    param_names: List[str] = ["unique_elements"]

    def setup(self, unique_elements: int) -> None:
        N = 1_000_000
        self.left = DataFrame({"a": np.random.randint(1, unique_elements, (N,))})
        self.right = DataFrame({"a": np.random.randint(1, unique_elements, (N,))})
        uniques = self.right.a.drop_duplicates()
        self.right["a"] = concat(
            [uniques, Series(np.arange(0, -(N - len(uniques)), -1))], ignore_index=True
        )

    def time_unique_merge(self, unique_elements: int) -> None:
        merge(self.left, self.right, how="inner")


class MergeDatetime:
    params: List[List[Tuple[str, str]], List[Optional[str]], List[bool]] = [
        [
            ("ns", "ns"),
            ("ms", "ms"),
            ("ns", "ms"),
        ],
        [None, "Europe/Brussels"],
        [True, False],
    ]
    param_names: List[str] = ["units", "tz", "monotonic"]

    def setup(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> None:
        unit_left, unit_right = units
        N = 10_000
        keys = Series(date_range("2012-01-01", freq="min", periods=N, tz=tz))
        self.left = DataFrame(
            {
                "key": keys.sample(N * 10, replace=True).dt.as_unit(unit_left),
                "value1": np.random.randn(N * 10),
            }
        )
        self.right = DataFrame(
            {
                "key": keys[:8000].dt.as_unit(unit_right),
                "value2": np.random.randn(8000),
            }
        )
        if monotonic:
            self.left = self.left.sort_values("key")
            self.right = self.right.sort_values("key")

    def time_merge(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> None:
        merge(self.left, self.right)


class MergeCategoricals:
    def setup(self) -> None:
        self.left_object = DataFrame(
            {
                "X": np.random.choice(range(10), size=(10000,)),
                "Y": np.random.choice(["one", "two", "three"], size=(10000,)),
            }
        )

        self.right_object = DataFrame(
            {
                "X": np.random.choice(range(10), size=(10000,)),
                "Z": np.random.choice(["jjj", "kkk", "sss"], size=(10000,)),
            }
        )

        self.left_cat = self.left_object.assign(
            Y=self.left_object["Y"].astype("category")
        )
        self.right_cat = self.right_object.assign(
            Z=self.right_object["Z"].astype("category")
        )

        self.left_cat_col = self.left_object.astype({"X": "category"})
        self.right_cat_col = self.right_object.astype({"X": "category"})

        self.left_cat_idx = self.left_cat_col.set_index("X")
        self.right_cat_idx = self.right_cat_col.set_index("X")

    def time_merge_object(self) -> None:
        merge(self.left_object, self.right_object, on="X")

    def time_merge_cat(self) -> None:
        merge(self.left_cat, self.right_cat, on="X")

    def time_