from collections import defaultdict
from datetime import datetime
from itertools import product
from typing import Any, List, Tuple, Union, Optional, Sequence, Type
import numpy as np
import pytest
from pandas import NA, DataFrame, MultiIndex, Series, array, concat, merge
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
    _decons_group_index,
    get_group_index,
    is_int64_overflow_possible,
    lexsort_indexer,
    nargsort,
)


@pytest.fixture
def left_right() -> Tuple[DataFrame, DataFrame]:
    low: int = -1 << 10
    high: int = 1 << 10
    n: int = 1 << 20
    left: DataFrame = DataFrame(
        np.random.default_rng(2).integers(low, high, (n, 7)), columns=list("ABCDEFG")
    )
    left["left"] = left.sum(axis=1)
    right: DataFrame = left.sample(
        frac=1, random_state=np.random.default_rng(2), ignore_index=True
    )
    right.columns = right.columns[:-1].tolist() + ["right"]
    right["right"] *= -1
    return left, right


class TestSorting:
    @pytest.mark.slow
    def test_int64_overflow(self) -> None:
        B: np.ndarray = np.concatenate((np.arange(1000), np.arange(1000), np.arange(500)))
        A: np.ndarray = np.arange(2500)
        df: DataFrame = DataFrame(
            {
                "A": A,
                "B": B,
                "C": A,
                "D": B,
                "E": A,
                "F": B,
                "G": A,
                "H": B,
                "values": np.random.default_rng(2).standard_normal(2500),
            }
        )
        lg = df.groupby(["A", "B", "C", "D", "E", "F", "G", "H"])
        rg = df.groupby(["H", "G", "F", "E", "D", "C", "B", "A"])
        left: Series = lg.sum()["values"]
        right: Series = rg.sum()["values"]
        exp_index, _ = left.index.sortlevel()
        tm.assert_index_equal(left.index, exp_index)
        exp_index, _ = right.index.sortlevel(0)
        tm.assert_index_equal(right.index, exp_index)
        tups: List[Tuple[Any, ...]] = list(
            map(tuple, df[["A", "B", "C", "D", "E", "F", "G", "H"]].values)
        )
        tups = com.asarray_tuplesafe(tups)
        expected: Series = df.groupby(tups).sum()["values"]
        for k, v in expected.items():
            assert left[k] == right[k[::-1]]
            assert left[k] == v
        assert len(left) == len(right)

    def test_int64_overflow_groupby_large_range(self) -> None:
        values = range(55109)
        data: DataFrame = DataFrame.from_dict({"a": values, "b": values, "c": values, "d": values})
        grouped = data.groupby(["a", "b", "c", "d"])
        assert len(grouped) == len(values)

    @pytest.mark.parametrize("agg", ["mean", "median"])
    def test_int64_overflow_groupby_large_df_shuffled(self, agg: str) -> None:
        rs = np.random.default_rng(2)
        arr: np.ndarray = rs.integers(-1 << 12, 1 << 12, (1 << 15, 5))
        i: np.ndarray = rs.choice(len(arr), len(arr) * 4)
        arr = np.vstack((arr, arr[i]))
        i = rs.permutation(len(arr))
        arr = arr[i]
        df: DataFrame = DataFrame(arr, columns=list("abcde"))
        df["jim"], df["joe"] = np.zeros((2, len(df)))
        gr = df.groupby(list("abcde"))
        assert is_int64_overflow_possible(
            tuple((ping.ngroups for ping in gr._grouper.groupings))
        )
        mi: MultiIndex = MultiIndex.from_arrays(
            [ar.ravel() for ar in np.array_split(np.unique(arr, axis=0), 5, axis=1)],
            names=list("abcde"),
        )
        res: DataFrame = DataFrame(np.zeros((len(mi), 2)), columns=["jim", "joe"], index=mi).sort_index()
        tm.assert_frame_equal(getattr(gr, agg)(), res)

    @pytest.mark.parametrize(
        "order, na_position, exp",
        [
            [True, "last", list(range(5, 105)) + list(range(5)) + list(range(105, 110))],
            [True, "first", list(range(5)) + list(range(105, 110)) + list(range(5, 105))],
            [False, "last", list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110))],
            [False, "first", list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1))],
        ],
    )
    def test_lexsort_indexer(self, order: bool, na_position: str, exp: List[int]) -> None:
        keys: List[Union[float, int]] = [([np.nan] * 5) + list(range(100)) + ([np.nan] * 5)]
        result: np.ndarray = lexsort_indexer(keys, orders=order, na_position=na_position)
        tm.assert_numpy_array_equal(result, np.array(exp, dtype=np.intp))

    @pytest.mark.parametrize(
        "ascending, na_position, exp",
        [
            [True, "last", list(range(5, 105)) + list(range(5)) + list(range(105, 110))],
            [True, "first", list(range(5)) + list(range(105, 110)) + list(range(5, 105))],
            [False, "last", list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110))],
            [False, "first", list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1))],
        ],
    )
    def test_nargsort(self, ascending: bool, na_position: str, exp: List[int]) -> None:
        items: np.ndarray = np.array([np.nan] * 5 + list(range(100)) + [np.nan] * 5, dtype="O")
        result: np.ndarray = nargsort(items, kind="mergesort", ascending=ascending, na_position=na_position)
        tm.assert_numpy_array_equal(result, np.array(exp), check_dtype=False)


class TestMerge:
    def test_int64_overflow_outer_merge(self) -> None:
        df1: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 7)),
            columns=list("ABCDEF") + ["G1"],
        )
        df2: DataFrame = DataFrame(
            np.random.default_rng(3).standard_normal((1000, 7)),
            columns=list("ABCDEF") + ["G2"],
        )
        result: DataFrame = merge(df1, df2, how="outer")
        assert len(result) == 2000

    @pytest.mark.slow
    def test_int64_overflow_check_sum_col(self, left_right: Tuple[DataFrame, DataFrame]) -> None:
        left, right = left_right
        out: DataFrame = merge(left, right, how="outer")
        assert len(out) == len(left)
        tm.assert_series_equal(out["left"], -out["right"], check_names=False)
        result: Series = out.iloc[:, :-2].sum(axis=1)
        tm.assert_series_equal(out["left"], result, check_names=False)
        assert result.name is None

    @pytest.mark.slow
    def test_int64_overflow_how_merge(
        self, left_right: Tuple[DataFrame, DataFrame], join_type: str
    ) -> None:
        left, right = left_right
        out: DataFrame = merge(left, right, how="outer")
        out.sort_values(out.columns.tolist(), inplace=True)
        tm.assert_frame_equal(out, merge(left, right, how=join_type, sort=True))

    @pytest.mark.slow
    def test_int64_overflow_sort_false_order(self, left_right: Tuple[DataFrame, DataFrame]) -> None:
        left, right = left_right
        out: DataFrame = merge(left, right, how="left", sort=False)
        tm.assert_frame_equal(left, out[left.columns.tolist()])
        out = merge(right, left, how="left", sort=False)
        tm.assert_frame_equal(right, out[right.columns.tolist()])

    @pytest.mark.slow
    def test_int64_overflow_one_to_many_none_match(
        self, join_type: str, sort: bool
    ) -> None:
        how: str = join_type
        low: int = -1 << 10
        high: int = 1 << 10
        n: int = 1 << 11
        left: DataFrame = DataFrame(
            np.random.default_rng(2).integers(low, high, (n, 7)).astype("int64"),
            columns=list("ABCDEFG"),
        )
        shape: np.ndarray = left.apply(Series.nunique).values
        assert is_int64_overflow_possible(shape)
        left = concat([left, left], ignore_index=True)
        right: DataFrame = DataFrame(
            np.random.default_rng(3).integers(low, high, (n // 2, 7)).astype("int64"),
            columns=list("ABCDEFG"),
        )
        i: np.ndarray = np.random.default_rng(4).choice(len(left), n)
        right = concat([right, right, left.iloc[i]], ignore_index=True)
        left["left"] = np.random.default_rng(2).standard_normal(len(left))
        right["right"] = np.random.default_rng(2).standard_normal(len(right))
        left = left.sample(frac=1, ignore_index=True, random_state=np.random.default_rng(5))
        right = right.sample(frac=1, ignore_index=True, random_state=np.random.default_rng(6))
        ldict: defaultdict = defaultdict(list)
        rdict: defaultdict = defaultdict(list)
        for idx, row in left.set_index(list("ABCDEFG")).iterrows():
            ldict[idx].append(row["left"])
        for idx, row in right.set_index(list("ABCDEFG")).iterrows():
            rdict[idx].append(row["right"])
        vals: List[Tuple[Any, ...]] = []
        for k, lval in ldict.items():
            rval: List[Any] = rdict.get(k, [np.nan])
            for lv, rv in product(lval, rval):
                vals.append(k + (lv, rv))
        for k, rval in rdict.items():
            if k not in ldict:
                vals.extend((k + (np.nan, rv) for rv in rval))
        out: DataFrame = DataFrame(vals, columns=list("ABCDEFG") + ["left", "right"])
        out = out.sort_values(out.columns.to_list(), ignore_index=True)
        jmask: dict = {
            "left": out["left"].notna(),
            "right": out["right"].notna(),
            "inner": out["left"].notna() & out["right"].notna(),
            "outer": np.ones(len(out), dtype="bool"),
        }
        mask = jmask[how]
        frame: DataFrame = out[mask].sort_values(out.columns.to_list(), ignore_index=True)
        assert mask.all() ^ mask.any() or how == "outer"
        res: DataFrame = merge(left, right, how=how, sort=sort)
        if sort:
            kcols: List[str] = list("ABCDEFG")
            tm.assert_frame_equal(res[kcols], res[kcols].sort_values(kcols, kind="mergesort"))
        tm.assert_frame_equal(frame, res.sort_values(res.columns.to_list(), ignore_index=True))


@pytest.mark.parametrize(
    "codes_list, shape",
    [
        [
            [
                np.tile([0, 1, 2, 3, 0, 1, 2, 3], 100).astype(np.int64),
                np.tile([0, 2, 4, 3, 0, 1, 2, 3], 100).astype(np.int64),
                np.tile([5, 1, 0, 2, 3, 0, 5, 4], 100).astype(np.int64),
            ],
            (4, 5, 6),
        ],
        [
            [np.tile(np.arange(10000, dtype=np.int64), 5), np.tile(np.arange(10000, dtype=np.int64), 5)],
            (10000, 10000),
        ],
    ],
)
def test_decons(codes_list: List[np.ndarray], shape: Tuple[int, ...]) -> None:
    group_index: np.ndarray = get_group_index(codes_list, shape, sort=True, xnull=True)
    codes_list2: List[np.ndarray] = _decons_group_index(group_index, shape)
    for a, b in zip(codes_list, codes_list2):
        tm.assert_numpy_array_equal(a, b)


class TestSafeSort:
    @pytest.mark.parametrize(
        "arg, exp",
        [
            [[3, 1, 2, 0, 4], [0, 1, 2, 3, 4]],
            [np.array(list("baaacb"), dtype=object), np.array(list("aaabbc"), dtype=object)],
            [[], []],
        ],
    )
    def test_basic_sort(self, arg: Union[List[Any], np.ndarray], exp: Union[List[Any], np.ndarray]) -> None:
        result: np.ndarray = safe_sort(np.array(arg))
        expected: np.ndarray = np.array(exp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("verify", [True, False])
    @pytest.mark.parametrize(
        "codes, exp_codes",
        [
            [[0, 1, 1, 2, 3, 0, -1, 4], [3, 1, 1, 2, 0, 3, -1, 4]],
            [[], []],
        ],
    )
    def test_codes(
        self, verify: bool, codes: List[int], exp_codes: List[int]
    ) -> None:
        values: np.ndarray = np.array([3, 1, 2, 0, 4])
        expected: np.ndarray = np.array([0, 1, 2, 3, 4])
        result, result_codes = safe_sort(values, codes, use_na_sentinel=True, verify=verify)
        expected_codes: np.ndarray = np.array(exp_codes, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_codes_out_of_bound(self) -> None:
        values: np.ndarray = np.array([3, 1, 2, 0, 4])
        expected: np.ndarray = np.array([0, 1, 2, 3, 4])
        codes: List[int] = [0, 101, 102, 2, 3, 0, 99, 4]
        result, result_codes = safe_sort(values, codes, use_na_sentinel=True)
        expected_codes: np.ndarray = np.array([3, -1, -1, 2, 0, 3, -1, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    @pytest.mark.parametrize("codes", [[-1, -1], [2, -1], [2, 2]])
    def test_codes_empty_array_out_of_bound(self, codes: List[int]) -> None:
        empty_values: np.ndarray = np.array([])
        expected_codes: np.ndarray = -np.ones_like(codes, dtype=np.intp)
        _, result_codes = safe_sort(empty_values, codes)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_mixed_integer(self) -> None:
        values: np.ndarray = np.array(["b", 1, 0, "a", 0, "b"], dtype=object)
        result: np.ndarray = safe_sort(values)
        expected: np.ndarray = np.array([0, 0, 1, "a", "b", "b"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_integer_with_codes(self) -> None:
        values: np.ndarray = np.array(["b", 1, 0, "a"], dtype=object)
        codes: List[int] = [0, 1, -1, 2, 0, -1, 1]
        result, result_codes = safe_sort(values, codes)
        expected: np.ndarray = np.array([0, 1, "a", "b"], dtype=object)
        expected_codes: np.ndarray = np.array([3, 1, 0, 2, 3, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_unsortable(self) -> None:
        arr: np.ndarray = np.array([1, 2, datetime.now(), 0, 3], dtype=object)
        msg: str = "'[<>]' not supported between instances of .*"
        with pytest.raises(TypeError, match=msg):
            safe_sort(arr)

    @pytest.mark.parametrize(
        "arg, codes, err, msg",
        [
            [1, None, TypeError, "Only np.ndarray, ExtensionArray, and Index"],
            [np.array([0, 1, 2]), 1, TypeError, "Only list-like objects or None"],
            [np.array([0, 1, 2, 1]), [0, 1], ValueError, "values should be unique"],
        ],
    )
    def test_exceptions(
        self, arg: Any, codes: Any, err: Type[Exception], msg: str
    ) -> None:
        with pytest.raises(err, match=msg):
            safe_sort(values=arg, codes=codes)

    @pytest.mark.parametrize(
        "arg, exp",
        [
            [[1, 3, 2], [1, 2, 3]],
            [[1, 3, np.nan, 2], [1, 2, 3, np.nan]],
        ],
    )
    def test_extension_array(
        self, arg: List[Union[int, float, np.nan]], exp: List[Union[int, float, np.nan]]
    ) -> None:
        a = array(arg, dtype="Int64")
        result = safe_sort(a)
        expected = array(exp, dtype="Int64")
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("verify", [True, False])
    def test_extension_array_codes(self, verify: bool) -> None:
        a = array([1, 3, 2], dtype="Int64")
        result, codes = safe_sort(a, [0, 1, -1, 2], use_na_sentinel=True, verify=verify)
        expected_values = array([1, 2, 3], dtype="Int64")
        expected_codes = np.array([0, 2, -1, 1], dtype=np.intp)
        tm.assert_extension_array_equal(result, expected_values)
        tm.assert_numpy_array_equal(codes, expected_codes)


def test_mixed_str_null(nulls_fixture: Any) -> None:
    values: np.ndarray = np.array(["b", nulls_fixture, "a", "b"], dtype=object)
    result: np.ndarray = safe_sort(values)
    expected: np.ndarray = np.array(["a", "b", "b", nulls_fixture], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_safe_sort_multiindex() -> None:
    arr1: Series = Series([2, 1, NA, NA], dtype="Int64")
    arr2: List[int] = [2, 1, 3, 3]
    midx: MultiIndex = MultiIndex.from_arrays([arr1, arr2])
    result: MultiIndex = safe_sort(midx)
    expected: MultiIndex = MultiIndex.from_arrays([Series([1, 2, NA, NA], dtype="Int64"), [1, 2, 3, 3]])
    tm.assert_index_equal(result, expected)