from typing import Any, Callable, Optional, Tuple

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    isna,
    notna,
)
import pandas._testing as tm


def assert_equal(a: Any, b: Any) -> None:
    assert a == b


class TestMultiIndexSetItem:
    def check(
        self,
        target: DataFrame,
        indexers: Any,
        value: Any,
        compare_fn: Callable[[Any, Any], None] = assert_equal,
        expected: Optional[Any] = None,
    ) -> None:
        target.loc[indexers] = value
        result = target.loc[indexers]
        if expected is None:
            expected = value
        compare_fn(result, expected)

    def test_setitem_multiindex(self) -> None:
        # GH#7190
        cols = ["A", "w", "l", "a", "x", "X", "d", "profit"]
        index = MultiIndex.from_product(
            [np.arange(0, 100), np.arange(0, 80)], names=["time", "firm"]
        )
        t: int = 0
        n: int = 2

        df: DataFrame = DataFrame(
            np.nan,
            columns=cols,
            index=index,
        )
        self.check(target=df, indexers=((t, n), "X"), value=0)

        df = DataFrame(-999, columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=1)

        df = DataFrame(columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=2)

        # gh-7218: assigning with 0-dim arrays
        df = DataFrame(-999, columns=cols, index=index)
        self.check(
            target=df,
            indexers=((t, n), "X"),
            value=np.array(3),
            expected=3,
        )

    def test_setitem_multiindex2(self) -> None:
        # GH#5206
        df: DataFrame = DataFrame(
            np.arange(25).reshape(5, 5), columns="A,B,C,D,E".split(","), dtype=float
        )
        df["F"] = 99
        row_selection = df["A"] % 2 == 0
        col_selection = ["B", "C"]
        df.loc[row_selection, col_selection] = df["F"]
        output: DataFrame = DataFrame(99.0, index=[0, 2, 4], columns=["B", "C"])
        tm.assert_frame_equal(df.loc[row_selection, col_selection], output)
        self.check(
            target=df,
            indexers=(row_selection, col_selection),
            value=df["F"],
            compare_fn=tm.assert_frame_equal,
            expected=output,
        )

    def test_setitem_multiindex3(self) -> None:
        # GH#11372
        idx: MultiIndex = MultiIndex.from_product(
            [["A", "B", "C"], date_range("2015-01-01", "2015-04-01", freq="MS")]
        )
        cols: MultiIndex = MultiIndex.from_product(
            [["foo", "bar"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((12, 4)), index=idx, columns=cols
        )

        subidx: MultiIndex = MultiIndex.from_arrays(
            [["A", "A"], date_range("2015-01-01", "2015-02-01", freq="MS")]
        )
        subcols: MultiIndex = MultiIndex.from_arrays(
            [["foo", "foo"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        vals: DataFrame = DataFrame(
            np.random.default_rng(2).random((2, 2)), index=subidx, columns=subcols
        )
        self.check(
            target=df,
            indexers=(subidx, subcols),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )
        # set all columns
        vals = DataFrame(
            np.random.default_rng(2).random((2, 4)), index=subidx, columns=cols
        )
        self.check(
            target=df,
            indexers=(subidx, slice(None, None, None)),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )
        # identity
        copy: DataFrame = df.copy()
        self.check(
            target=df,
            indexers=(df.index, df.columns),
            value=df,
            compare_fn=tm.assert_frame_equal,
            expected=copy,
        )

    def test_multiindex_setitem(self) -> None:
        # GH 3738
        # setting with a multi-index right hand side
        arrays = [
            np.array(["bar", "bar", "baz", "qux", "qux", "bar"]),
            np.array(["one", "two", "one", "one", "two", "one"]),
            np.arange(0, 6, 1),
        ]

        df_orig: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((6, 3)),
            index=arrays,
            columns=["A", "B", "C"],
        ).sort_index()

        expected: DataFrame = df_orig.loc[["bar"]] * 2
        df = df_orig.copy()
        df.loc[["bar"]] *= 2
        tm.assert_frame_equal(df.loc[["bar"]], expected)

        # raise because these have differing levels
        msg: str = "cannot align on a multi-index with out specifying the join levels"
        with pytest.raises(TypeError, match=msg):
            df.loc["bar"] *= 2

    def test_multiindex_setitem2(self) -> None:
        # from SO
        # https://stackoverflow.com/questions/24572040/pandas-access-the-level-of-multiindex-for-inplace-operation
        df_orig: DataFrame = DataFrame.from_dict(
            {
                "price": {
                    ("DE", "Coal", "Stock"): 2,
                    ("DE", "Gas", "Stock"): 4,
                    ("DE", "Elec", "Demand"): 1,
                    ("FR", "Gas", "Stock"): 5,
                    ("FR", "Solar", "SupIm"): 0,
                    ("FR", "Wind", "SupIm"): 0,
                }
            }
        )
        df_orig.index = MultiIndex.from_tuples(
            df_orig.index, names=["Sit", "Com", "Type"]
        )

        expected: DataFrame = df_orig.copy()
        expected.iloc[[0, 1, 3]] *= 2

        idx = pd.IndexSlice
        df: DataFrame = df_orig.copy()
        df.loc[idx[:, :, "Stock"], :] *= 2
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[idx[:, :, "Stock"], "price"] *= 2
        tm.assert_frame_equal(df, expected)

    def test_multiindex_assignment(self) -> None:
        # GH3777 part 2

        # mixed dtype
        df: DataFrame = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
        )
        df["d"] = np.nan
        arr: np.ndarray = np.array([0.0, 1.0])

        df.loc[4, "d"] = arr
        tm.assert_series_equal(df.loc[4, "d"], Series(arr, index=[8, 10], name="d"))

    def test_multiindex_assignment_single_dtype(self) -> None:
        # GH3777 part 2b
        # single dtype
        arr: np.ndarray = np.array([0.0, 1.0])

        df: DataFrame = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
            dtype=np.int64,
        )

        # arr can be losslessly cast to int, so this setitem is inplace
        df.loc[4, "c"] = arr
        exp: Series = Series(arr, index=[8, 10], name="c", dtype="int64")
        result: Series = df.loc[4, "c"]
        tm.assert_series_equal(result, exp)

        # arr + 0.5 cannot be cast losslessly to int, so we upcast
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[4, "c"] = arr + 0.5
        # Upcast so that we can add .5
        df = df.astype({"c": "float64"})
        df.loc[4, "c"] = arr + 0.5

        # scalar ok
        df.loc[4, "c"] = 10
        exp = Series(10, index=[8, 10], name="c", dtype="float64")
        tm.assert_series_equal(df.loc[4, "c"], exp)

        # invalid assignments
        msg: str = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0, 1, 2, 3]

        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0]

        # But with a length-1 listlike column indexer this behaves like
        #  `df.loc[4, "c"] = 0
        df.loc[4, ["c"]] = [0]
        assert (df.loc[4, "c"] == 0).all()

    def test_groupby_example(self) -> None:
        # groupby example
        NUM_ROWS: int = 100
        NUM_COLS: int = 10
        col_names = ["A" + num for num in map(str, np.arange(NUM_COLS).tolist())]
        index_cols = col_names[:5]

        df: DataFrame = DataFrame(
            np.random.default_rng(2).integers(5, size=(NUM_ROWS, NUM_COLS)),
            dtype=np.int64,
            columns=col_names,
        )
        df = df.set_index(index_cols).sort_index()
        grp = df.groupby(level=index_cols[:4])
        df["new_col"] = np.nan

        # we are actually operating on a copy here
        # but in this case, that's ok
        for name, df2 in grp:
            new_vals = np.arange(df2.shape[0])
            df.loc[name, "new_col"] = new_vals

    def test_series_setitem(self, multiindex_year_month_day_dataframe_random_data: DataFrame) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        s: Series = ymd["A"]

        s[2000, 3] = np.nan
        assert isna(s.values[42:65]).all()
        assert notna(s.values[:42]).all()
        assert notna(s.values[65:]).all()

        s[2000, 3, 10] = np.nan
        assert isna(s.iloc[49])

        with pytest.raises(KeyError, match="49"):
            # GH#33355 dont fall-back to positional when leading level is int
            s[49]

    def test_frame_getitem_setitem_boolean(self, multiindex_dataframe_random_data: DataFrame) -> None:
        frame: DataFrame = multiindex_dataframe_random_data
        df: DataFrame = frame.T.copy()
        values: Any = df.values.copy()

        result: DataFrame = df[df > 0]
        expected: DataFrame = df.where(df > 0)
        tm.assert_frame_equal(result, expected)

        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)

        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)

        # a df that needs alignment first
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)

        with pytest.raises(TypeError, match="boolean values only"):
            df[df * 0] = 2

    def test_frame_getitem_setitem_multislice(self) -> None:
        levels = [["t1", "t2"], ["a", "b", "c"]]
        codes = [[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]]
        midx: MultiIndex = MultiIndex(codes=codes, levels=levels, names=[None, "id"])
        df: DataFrame = DataFrame({"value": [1, 2, 3, 7, 8]}, index=midx)

        result: Series = df.loc[:, "value"]
        tm.assert_series_equal(df["value"], result)

        result = df.loc[df.index[1:3], "value"]
        tm.assert_series_equal(df["value"][1:3], result)

        result = df.loc[:, :]
        tm.assert_frame_equal(df, result)

        result = df
        df.loc[:, "value"] = 10
        result["value"] = 10
        tm.assert_frame_equal(df, result)

        df.loc[:, :] = 10
        tm.assert_frame_equal(df, result)

    def test_frame_setitem_multi_column(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=[["a", "a", "b", "b"], [0, 1, 0, 1]],
        )

        cp: DataFrame = df.copy()
        cp["a"] = cp["b"]
        tm.assert_frame_equal(cp["a"], cp["b"])

        # set with ndarray
        cp = df.copy()
        cp["a"] = cp["b"].values
        tm.assert_frame_equal(cp["a"], cp["b"])

    def test_frame_setitem_multi_column2(self) -> None:
        # ---------------------------------------
        # GH#1803
        columns: MultiIndex = MultiIndex.from_tuples([("A", "1"), ("A", "2"), ("B", "1")])
        df: DataFrame = DataFrame(index=[1, 3, 5], columns=columns)

        # Works, but adds a column instead of updating the two existing ones
        df["A"] = 0.0  # Doesn't work
        assert (df["A"].values == 0).all()

        # it broadcasts
        df["B", "1"] = [1, 2, 3]
        df["A"] = df["B", "1"]

        sliced_a1: Series = df["A", "1"]
        sliced_a2: Series = df["A", "2"]
        sliced_b1: Series = df["B", "1"]
        tm.assert_series_equal(sliced_a1, sliced_b1, check_names=False)
        tm.assert_series_equal(sliced_a2, sliced_b1, check_names=False)
        assert sliced_a1.name == ("A", "1")
        assert sliced_a2.name == ("A", "2")
        assert sliced_b1.name == ("B", "1")

    def test_loc_getitem_tuple_plus_columns(
        self, multiindex_year_month_day_dataframe_random_data: DataFrame
    ) -> None:
        # GH #1013
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        df: DataFrame = ymd[:5]

        result: Series = df.loc[(2000, 1, 6), ["A", "B", "C"]]
        expected: Series = df.loc[2000, 1, 6][["A", "B", "C"]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_loc_getitem_setitem_slice_integers(self, frame_or_series: Any) -> None:
        index: MultiIndex = MultiIndex(
            levels=[[0, 1, 2], [0, 2]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]
        )

        obj: Any = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 4)),
            index=index,
            columns=["a", "b", "c", "d"],
        )
        obj = tm.get_obj(obj, frame_or_series)

        res = obj.loc[1:2]
        exp = obj.reindex(obj.index[2:])
        tm.assert_equal(res, exp)

        obj.loc[1:2] = 7
        assert (obj.loc[1:2] == 7).values.all()

    def test_setitem_change_dtype(self, multiindex_dataframe_random_data: DataFrame) -> None:
        frame: DataFrame = multiindex_dataframe_random_data
        dft: DataFrame = frame.T
        s: Series = dft["foo", "two"]
        dft["foo", "two"] = s > s.median()
        tm.assert_series_equal(dft["foo", "two"], s > s.median())
        # assert isinstance(dft._data.blocks[1].items, MultiIndex)

        reindexed: DataFrame = dft.reindex(columns=[("foo", "two")])
        tm.assert_series_equal(reindexed["foo", "two"], s > s.median())

    def test_set_column_scalar_with_loc(self, multiindex_dataframe_random_data: DataFrame) -> None:
        frame: DataFrame = multiindex_dataframe_random_data
        subset = frame.index[[1, 4, 5]]

        frame.loc[subset] = 99
        assert (frame.loc[subset].values == 99).all()

        frame_original: DataFrame = frame.copy()
        col: DataFrame = frame["B"]
        col[subset] = 97
        # chained setitem doesn't work with CoW
        tm.assert_frame_equal(frame, frame_original)

    def test_nonunique_assignment_1750(self) -> None:
        df: DataFrame = DataFrame(
            [[1, 1, "x", "X"], [1, 1, "y", "Y"], [1, 2, "z", "Z"]], columns=list("ABCD")
        )

        df = df.set_index(["A", "B"])
        mi: MultiIndex = MultiIndex.from_tuples([(1, 1)])

        df.loc[mi, "C"] = "_"

        assert (df.xs((1, 1))["C"] == "_").all()

    def test_astype_assignment_with_dups(self) -> None:
        # GH 4686
        # assignment with dups that has a dtype change
        cols: MultiIndex = MultiIndex.from_tuples([("A", "1"), ("B", "1"), ("A", "2")])
        df: DataFrame = DataFrame(np.arange(3).reshape((1, 3)), columns=cols, dtype=object)
        index = df.index.copy()

        df["A"] = df["A"].astype(np.float64)
        tm.assert_index_equal(df.index, index)

    def test_setitem_nonmonotonic(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/31449
        index: MultiIndex = MultiIndex.from_tuples(
            [("a", "c"), ("b", "x"), ("a", "d")], names=["l1", "l2"]
        )
        df: DataFrame = DataFrame(data=[0, 1, 2], index=index, columns=["e"])
        df.loc["a", "e"] = np.arange(99, 101, dtype="int64")
        expected: DataFrame = DataFrame({"e": [99, 1, 100]}, index=index)
        tm.assert_frame_equal(df, expected)


class TestSetitemWithExpansionMultiIndex:
    def test_setitem_new_column_mixed_depth(self) -> None:
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        tuples = sorted(zip(*arrays))
        index: MultiIndex = MultiIndex.from_tuples(tuples)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        result: DataFrame = df.copy()
        expected: DataFrame = df.copy()
        result["b"] = [1, 2, 3, 4]
        expected["b", "", ""] = [1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    def test_setitem_new_column_all_na(self) -> None:
        # GH#1534
        mix: MultiIndex = MultiIndex.from_tuples([("1a", "2a"), ("1a", "2b"), ("1a", "2c")])
        df: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
        s: Series = Series({(1, 1): 1, (1, 2): 2})
        df["new"] = s
        assert df["new"].isna().all()

    def test_setitem_enlargement_keep_index_names(self) -> None:
        # GH#53053
        mi: MultiIndex = MultiIndex.from_tuples([(1, 2, 3)], names=["i1", "i2", "i3"])
        df: DataFrame = DataFrame(data=[[10, 20, 30]], index=mi, columns=["A", "B", "C"])
        df.loc[(0, 0, 0)] = df.loc[(1, 2, 3)]
        mi_expected: MultiIndex = MultiIndex.from_tuples(
            [(1, 2, 3), (0, 0, 0)], names=["i1", "i2", "i3"]
        )
        expected: DataFrame = DataFrame(
            data=[[10, 20, 30], [10, 20, 30]],
            index=mi_expected,
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(df, expected)


def test_frame_setitem_view_direct(multiindex_dataframe_random_data: DataFrame) -> None:
    # this works because we are modifying the underlying array
    # really a no-no
    df: DataFrame = multiindex_dataframe_random_data.T
    with pytest.raises(ValueError, match="read-only"):
        df["foo"].values[:] = 0
    assert (df["foo"].values != 0).all()


def test_frame_setitem_copy_raises(multiindex_dataframe_random_data: DataFrame) -> None:
    # will raise/warn as its chained assignment
    df: DataFrame = multiindex_dataframe_random_data.T
    with tm.raises_chained_assignment_error():
        df["foo"]["one"] = 2


def test_frame_setitem_copy_no_write(multiindex_dataframe_random_data: DataFrame) -> None:
    frame: DataFrame = multiindex_dataframe_random_data.T
    expected: DataFrame = frame
    df: DataFrame = frame.copy()
    with tm.raises_chained_assignment_error():
        df["foo"]["one"] = 2

    tm.assert_frame_equal(df, expected)


def test_frame_setitem_partial_multiindex() -> None:
    # GH 54875
    df: DataFrame = DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 4, 5],
            "c": 6,
            "d": 7,
        }
    ).set_index(["a", "b", "c"])
    ser: Series = Series(8, index=df.index.droplevel("c"))
    result: DataFrame = df.copy()
    result["d"] = ser
    expected: DataFrame = df.copy()
    expected["d"] = 8
    tm.assert_frame_equal(result, expected)