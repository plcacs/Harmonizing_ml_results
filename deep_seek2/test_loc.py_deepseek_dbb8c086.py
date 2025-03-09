from collections import namedtuple
import contextlib
from datetime import date, datetime, time, timedelta
import re
from typing import Any, Callable, List, Optional, Tuple, Union

from dateutil.tz import gettz
import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.errors import IndexingError

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IndexSlice,
    MultiIndex,
    Period,
    PeriodIndex,
    Series,
    SparseDtype,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises


@pytest.mark.parametrize(
    "series, new_series, expected_ser",
    [
        [[np.nan, np.nan, "b"], ["a", np.nan, np.nan], [False, True, True]],
        [[np.nan, "b"], ["a", np.nan], [False, True]],
    ],
)
def test_not_change_nan_loc(series: List[Union[float, str]], new_series: List[Union[float, str]], expected_ser: List[bool]]) -> None:
    # GH 28403
    df: DataFrame = DataFrame({"A": series})
    df.loc[:, "A"] = new_series
    expected: DataFrame = DataFrame({"A": expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)


class TestLoc:
    def test_none_values_on_string_columns(self, using_infer_string: bool) -> None:
        # Issue #32218
        df: DataFrame = DataFrame(["1", "2", None], columns=["a"], dtype=object)
        assert df.loc[2, "a"] is None

        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        if using_infer_string:
            assert np.isnan(df.loc[2, "a"])
        else:
            assert df.loc[2, "a"] is None

    def test_loc_getitem_int(self, frame_or_series: Callable) -> None:
        # int label
        obj: Union[DataFrame, Series] = frame_or_series(range(3), index=Index(list("abc"), dtype=object))
        check_indexing_smoketest_or_raises(obj, "loc", 2, fails=KeyError)

    def test_loc_getitem_label(self, frame_or_series: Callable) -> None:
        # label
        obj: Union[DataFrame, Series] = frame_or_series()
        check_indexing_smoketest_or_raises(obj, "loc", "c", fails=KeyError)

    @pytest.mark.parametrize("key", ["f", 20])
    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcd"), dtype=object),
            Index([2, 4, "null", 8], dtype=object),
            date_range("20130101", periods=4),
            Index(range(0, 8, 2), dtype=np.float64),
            Index([]),
        ],
    )
    def test_loc_getitem_label_out_of_range(self, key: Union[str, int], index: Index, frame_or_series: Callable) -> None:
        obj: Union[DataFrame, Series] = frame_or_series(range(len(index)), index=index)
        # out of range label
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize("key", [[0, 1, 2], [1, 3.0, "A"]])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_loc_getitem_label_list(self, key: List[Union[int, float, str]], dtype: np.dtype, frame_or_series: Callable) -> None:
        obj: Union[DataFrame, Series] = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
        # list of labels
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize(
        "index",
        [
            None,
            Index([0, 1, 2], dtype=np.int64),
            Index([0, 1, 2], dtype=np.uint64),
            Index([0, 1, 2], dtype=np.float64),
            MultiIndex.from_arrays([range(3), range(3)]),
        ],
    )
    @pytest.mark.parametrize(
        "key", [[0, 1, 2], [0, 2, 10], [3, 6, 7], [(1, 3), (1, 4), (2, 5)]]
    )
    def test_loc_getitem_label_list_with_missing(self, key: List[Union[int, Tuple[int, int]]], index: Optional[Index], frame_or_series: Callable) -> None:
        if index is None:
            obj: Union[DataFrame, Series] = frame_or_series()
        else:
            obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_loc_getitem_label_list_fails(self, dtype: np.dtype, frame_or_series: Callable) -> None:
        # fails
        obj: Union[DataFrame, Series] = frame_or_series(range(3), Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(
            obj, "loc", [20, 30, 40], axes=1, fails=KeyError
        )

    def test_loc_getitem_bool(self, frame_or_series: Callable) -> None:
        obj: Union[DataFrame, Series] = frame_or_series()
        # boolean indexers
        b: List[bool] = [True, False, True, False]

        check_indexing_smoketest_or_raises(obj, "loc", b, fails=IndexError)

    @pytest.mark.parametrize(
        "slc, indexes, axes, fails",
        [
            [
                slice(1, 3),
                [
                    Index(list("abcd"), dtype=object),
                    Index([2, 4, "null", 8], dtype=object),
                    None,
                    date_range("20130101", periods=4),
                    Index(range(0, 12, 3), dtype=np.float64),
                ],
                None,
                TypeError,
            ],
            [
                slice("20130102", "20130104"),
                [date_range("20130101", periods=4)],
                1,
                TypeError,
            ],
            [slice(2, 8), [Index([2, 4, "null", 8], dtype=object)], 0, TypeError],
            [slice(2, 8), [Index([2, 4, "null", 8], dtype=object)], 1, KeyError],
            [slice(2, 4, 2), [Index([2, 4, "null", 8], dtype=object)], 0, TypeError],
        ],
    )
    def test_loc_getitem_label_slice(self, slc: slice, indexes: List[Index], axes: Optional[int], fails: type, frame_or_series: Callable) -> None:
        # label slices (with ints)

        # real label slices

        # GH 14316
        for index in indexes:
            if index is None:
                obj: Union[DataFrame, Series] = frame_or_series()
            else:
                obj = frame_or_series(range(len(index)), index=index)
            check_indexing_smoketest_or_raises(
                obj,
                "loc",
                slc,
                axes=axes,
                fails=fails,
            )

    def test_setitem_from_duplicate_axis(self) -> None:
        # GH#34034
        df: DataFrame = DataFrame(
            [[20, "a"], [200, "a"], [200, "a"]],
            columns=["col1", "col2"],
            index=[10, 1, 1],
        )
        df.loc[1, "col1"] = np.arange(2)
        expected: DataFrame = DataFrame(
            [[20, "a"], [0, "a"], [1, "a"]], columns=["col1", "col2"], index=[10, 1, 1]
        )
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self) -> None:
        # GH 26779
        df: DataFrame = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", np.nan, "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    pd.NaT,
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        df2: DataFrame = DataFrame(
            data={"A": ["String 3"], "B": [Timestamp("2019-06-11 12:00:00")]}
        )
        # Change Columns A and B to df2.values wherever Column A is NaN
        df.loc[df["A"].isna(), ["A", "B"]] = df2.values
        expected: DataFrame = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", "String 3", "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "obj, key, exp",
        [
            (
                DataFrame([[1]], columns=Index([False])),
                IndexSlice[:, False],
                Series([1], name=False),
            ),
            (Series([1], index=Index([False])), False, [1]),
            (DataFrame([[1]], index=Index([False])), False, Series([1], name=False)),
        ],
    )
    def test_loc_getitem_single_boolean_arg(self, obj: Union[DataFrame, Series], key: Union[IndexSlice, bool], exp: Union[Series, List[int]]) -> None:
        # GH 44322
        res: Union[DataFrame, Series] = obj.loc[key]
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp


class TestLocBaseIndependent:
    # Tests for loc that do not depend on subclassing Base
    def test_loc_npstr(self) -> None:
        # GH#45580
        df: DataFrame = DataFrame(index=date_range("2021", "2022"))
        result: DataFrame = df.loc[np.array(["2021/6/1"])[0] :]
        expected: DataFrame = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "msg, key",
        [
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'bar'", (Period(2019), "foo", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'y1', 'bar'", (Period(2019), "y1", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'z1'", (Period(2019), "foo", "z1")),
            (
                r"Period\('2018', 'Y-DEC'\), Period\('2016', 'Y-DEC'\), 'bar'",
                (Period(2018), Period(2016), "bar"),
            ),
            (r"Period\('2018', 'Y-DEC'\), 'foo', 'y1'", (Period(2018), "foo", "y1")),
            (
                r"Period\('2017', 'Y-DEC'\), 'foo', Period\('2015', 'Y-DEC'\)",
                (Period(2017), "foo", Period(2015)),
            ),
            (r"Period\('2017', 'Y-DEC'\), 'z1', 'bar'", (Period(2017), "z1", "bar")),
        ],
    )
    def test_contains_raise_error_if_period_index_is_in_multi_index(self, msg: str, key: Tuple[Period, str, str]) -> None:
        # GH#20684
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        df: DataFrame = DataFrame(
            {
                "A": [Period(2019), "x1", "x2"],
                "B": [Period(2018), Period(2016), "y1"],
                "C": [Period(2017), "z1", Period(2015)],
                "V1": [1, 2, 3],
                "V2": [10, 20, 30],
            }
        ).set_index(["A", "B", "C"])
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    def test_loc_getitem_missing_unicode_key(self) -> None:
        df: DataFrame = DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="\u05d0"):
            df.loc[:, "\u05d0"]  # should not raise UnicodeEncodeError

    def test_loc_getitem_dups(self) -> None:
        # GH 5678
        # repeated getitems on a dup index returning a ndarray
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((20, 5)),
            index=["ABCDE"[x % 5] for x in range(20)],
        )
        expected: Series = df.loc["A", 0]
        result: Series = df.loc[:, 0].loc["A"]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self) -> None:
        # GH4726
        # dup indexing with iloc/loc
        df: DataFrame = DataFrame(
            [[1, 2, "foo", "bar", Timestamp("20130101")]],
            columns=["a", "a", "a", "a", "a"],
            index=[1],
        )
        expected: Series = Series(
            [1, 2, "foo", "bar", Timestamp("20130101")],
            index=["a", "a", "a", "a", "a"],
            name=1,
        )

        result: Series = df.iloc[0]
        tm.assert_series_equal(result, expected)

        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self) -> None:
        # GH 6541
        df_orig: DataFrame = DataFrame(
            {
                "me": list("rttti"),
                "foo": list("aaade"),
                "bar": np.arange(5, dtype="float64") * 1.34 + 2,
                "bar2": np.arange(5, dtype="float64") * -0.34 + 2,
            }
        ).set_index("me")

        indexer: Tuple[str, List[str]] = (
            "r",
            ["bar", "bar2"],
        )
        df: DataFrame = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

        indexer = (
            "r",
            "bar",
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]

        indexer = (
            "t",
            ["bar", "bar2"],
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self) -> None:
        # GH10503

        # assigning the same type should not change the type
        df1: DataFrame = DataFrame({"a": [0, 1, 1], "b": Series([100, 200, 300], dtype="uint32")})
        ix: Series = df1["a"] == 1
        newb1: Series = df1.loc[ix, "b"] + 1
        df1.loc[ix, "b"] = newb1
        expected: DataFrame = DataFrame(
            {"a": [0, 1, 1], "b": Series([100, 201, 301], dtype="uint32")}
        )
        tm.assert_frame_equal(df1, expected)

        # assigning a new type should get the inferred type
        df2: DataFrame = DataFrame({"a": [0, 1, 1], "b": [100, 200, 300]}, dtype="uint64")
        ix = df1["a"] == 1
        newb2: Series = df2.loc[ix, "b"]
        with pytest.raises(TypeError, match="Invalid value"):
            df1.loc[ix, "b"] = newb2

    def test_loc_setitem_dtype(self) -> None:
        # GH31340
        df: DataFrame = DataFrame({"id": ["A"], "a": [1.2], "b": [0.0], "c": [-2.5]})
        cols: List[str] = ["a", "b", "c"]
        df.loc[:, cols] = df.loc[:, cols].astype("float32")

        # pre-2.0 this setting would swap in new arrays, in 2.0 it is correctly
        #  in-place, consistent with non-split-path
        expected: DataFrame = DataFrame(
            {
                "id": ["A"],
                "a": np.array([1.2], dtype="float64"),
                "b": np.array([0.0], dtype="float64"),
                "c": np.array([-2.5], dtype="float64"),
            }
        )  # id is inferred as object

        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self) -> None:
        s: Series = Series(range(3), index=["a", "b", "c"])

        # consistency
        with pytest.raises(KeyError, match="not in index"):
            s[["a", "d"]]

        s = Series(range(3))
        with pytest.raises(KeyError, match="not in index"):
            s[[0, 3]]

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index: List[bool]) -> None:
        # GH26658
        s: Series = Series([1, 2, 3])
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self) -> None:
        # TODO: test something here?
        pass

    def test_loc_to_fail(self) -> None:
        # GH3449
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            index=["a", "b", "c"],
            columns=["e", "f", "g"],
        )

        msg = (
            rf"\"None of \[Index\(\[1, 2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self) -> None:
        # GH  7496
        # loc should not fallback

        s: Series = Series(dtype=object)
        s.loc[1] = 1
        s.loc["a"] = 2

        with pytest.raises(KeyError, match=r"^-1$"):
            s.loc[-1]

        msg = (
            rf"\"None of \[Index\(\[-1, -2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]

        msg = r"\"None of \[Index\(\['4'\], dtype='object'\)\] are in the \[index\]\""
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(["4"], dtype=object)]

        s.loc[-1] = 3
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[-1, -2]]

        s["a"] = 2
        msg = (
            rf"\"None of \[Index\(\[-2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]

        del s["a"]

        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0

    def test_loc_to_fail3(self) -> None:
        # inconsistency between .loc[values] and .loc[values,:]
        # GH 7999
        df: DataFrame = DataFrame([["a"], ["b"]], index=[1, 2], columns=["value"])

        msg = (
            rf"\"None of \[Index\(\[3\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]

        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self) -> None:
        # 15747
        # should KeyError if *any* missing labels

        s: Series = Series([1, 2, 3])

        s.loc[[2]]

        msg = "None of [RangeIndex(start=3, stop=4, step=1)] are in the [index]"
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]

        # a non-match and a match
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[2, 3]]

    def test_loc_index(self) -> None:
        # gh-17131
        # a boolean index should index like a boolean numpy array

        df: DataFrame = DataFrame(
            np.random.default_rng(2).random(size=(5, 10)),
            index=["alpha_0", "alpha_1", "alpha_2", "beta_0", "beta_1"],
        )

        mask: Series = df.index.map(lambda x: "alpha" in x)
        expected: DataFrame = df.loc[np.array(mask)]

        result: DataFrame = df.loc[mask]
        tm.assert_frame_equal(result, expected)

        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)

        result = df.loc[pd.array(mask, dtype="boolean")]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random((4, 4)),
            columns=["A", "B", "C", "D"],
            index=["A", "B", "C", "D"],
        )

        # want this to work
        result: DataFrame = df.loc[:, "A":"B"].iloc[0:2, :]
        assert (result.columns == ["A", "B"]).all()
        assert (result.index == ["A", "B"]).all()

        # mixed type
        result: Series = DataFrame({"a": [Timestamp("20130101")], "b": [1]}).iloc[0]
        expected: Series = Series([Timestamp("20130101"), 1], index=["a", "b"], name=0)
        tm.assert_series_equal(result, expected)
        assert result.dtype == object

    @pytest.fixture
    def frame_for_consistency(self) -> DataFrame:
        return DataFrame(
            {
                "date": date_range("2000-01-01", "2000-01-5"),
                "val": Series(range(5), dtype=np.int64),
            }
        )

    @pytest.mark.parametrize(
        "val",
        [0, np.array(0, dtype=np.int64), np.array([0, 0, 0, 0, 0], dtype=np.int64)],
    )
    def test_loc_setitem_consistency(self, frame_for_consistency: DataFrame, val: Union[int, np.ndarray]]) -> None:
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice
        df: DataFrame = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = val

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency: DataFrame) -> None:
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice

        df: DataFrame = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = "foo"

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency: DataFrame) -> None:
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice
        df: DataFrame = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = 1.0

    def test_loc_setitem_consistency_single_row(self) -> None:
        # GH 15494
        # setting on frame with single row
        df: DataFrame = DataFrame({"date": Series([Timestamp("20180101")])})
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = "string"

    def test_loc_setitem_consistency_empty(self) -> None:
        # empty (essentially noops)
        # before the enforcement of #45333 in 2.0, the loc.setitem here would
        #  change the dtype of df.x to int64
        expected: DataFrame = DataFrame(columns=["x", "y"])
        df: DataFrame = DataFrame(columns=["x", "y"])
        with tm.assert_produces_warning(None):
            df.loc[:, "x"] = 1
        tm.assert_frame_equal(df, expected)

        # setting with setitem swaps in a new array, so changes the dtype
        df = DataFrame(columns=["x", "y"])
        df["x"] = 1
        expected["x"] = expected["x"].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_slice_column_len(self, using_infer_string: bool) -> None:
        # .loc[:,column] setting with slice == len of the column
        # GH10408
        levels: List[List[str]] = [
            ["Region_1"] * 4,
            ["Site_1", "Site_1", "Site_2", "Site_2"],
            [3987227376, 3980680971, 3977723249, 3977723089],
        ]
        mi: MultiIndex = MultiIndex.from_arrays(levels, names=["Region", "Site", "RespondentID"])

        clevels: List[List[str]] = [
            ["Respondent", "Respondent", "Respondent", "OtherCat", "OtherCat"],
            ["Something", "StartDate", "EndDate", "Yes/No", "SomethingElse"],
        ]
        cols: MultiIndex = MultiIndex.from_arrays(clevels, names=["Level_0", "Level_1"])

        values: List[List[Union[str, float]]] = [
            ["A", "5/25/2015 10:59", "5/25/2015 11:22", "Yes", np.nan],
            ["A", "5/21/2015 9:40", "5/21/2015 9:52", "Yes", "Yes"],
            ["A", "5/20/2015 8:27", "5/20/2015 8:41", "Yes", np.nan],
            ["A", "5/20/2015 8:33", "5/20/2015 9:09", "Yes", "No"],
        ]
        df: DataFrame = DataFrame(values, index=mi, columns=cols)

        ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
        if using_infer_string:
            ctx = pytest.raises(TypeError, match="Invalid value")

        with ctx:
            df.loc[:, ("Respondent", "StartDate")] = to_datetime(
                df.loc[:, ("Respondent", "StartDate")]
            )
        with ctx:
            df.loc[:, ("Respondent", "EndDate")] = to_datetime(
                df.loc[:, ("Respondent", "EndDate")]
            )

        if using_infer_string:
            # infer-objects won't infer stuff anymore
            return

        df = df.infer_objects()

        # Adding a new key
        df.loc[:, ("Respondent", "Duration")] = (
            df.loc[:, ("Respondent", "EndDate")]
            - df.loc[:, ("Respondent", "StartDate")]
        )

        # timedelta64[m] -> float, so this cannot be done inplace, so
        #  no warning
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, ("Respondent", "Duration")] = df.loc[
                :, ("Respondent", "Duration")
            ] / Timedelta(60_000_000_000)

    @pytest.mark.parametrize("unit", ["Y", "M", "D", "h", "m", "s", "ms", "us"])
    def test_loc_assign_non_ns_datetime(self, unit: str) -> None:
        # GH 27395, non-ns dtype assignment via .loc should work
        # and return the same result when using simple assignment
        df: DataFrame = DataFrame(
            {
                "timestamp": [
                    np.datetime64("2017-02-11 12:41:29"),
                    np.datetime64("1991-11-07 04:22:37"),
                ]
            }
        )

        df.loc[:, unit] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        df["expected"] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        expected: Series = Series(df.loc[:, "expected"], name=unit)
        tm.assert_series_equal(df.loc[:, unit], expected)

    def test_loc_modify_datetime(self) -> None:
        # see gh-28837
        df: DataFrame = DataFrame.from_dict(
            {"date": [1485264372711, 1485265925110, 1540215845888, 1540282121025]}
        )

        df["date_dt"] = to_datetime(df["date"], unit="ms", cache=True).dt.as_unit("ms")

        df.loc[:, "date_dt_cp"] = df.loc[:, "date_dt"]
        df.loc[[2, 3], "date_dt_cp"] = df.loc[[2, 3], "date_dt"]

        expected: DataFrame = DataFrame(
            [
                [1485264372711, "2017-01-24 13:26:12.711", "2017-01-24 13:26:12.711"],
                [1485265925110, "2017-01-24 13:52:05.110", "2017-01-24 13:52:05.110"],
                [1540215845888, "2018-10-22 13:44:05.888", "2018-10-22 13:44:05.888"],
                [1540282121025, "2018-10-23 08:08:41.025", "2018-10-23 08:08:41.025"],
            ],
            columns=["date", "date_dt", "date_dt_cp"],
        )

        columns: List[str] = ["date_dt", "date_dt_cp"]
        expected[columns] = expected[columns].apply(to_datetime)

        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex(self) -> None:
        # GH#6254 setting issue
        df: DataFrame = DataFrame(index=[3, 5, 4], columns=["A"], dtype=float)
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")

        # setting integer values into a float dataframe with loc is inplace,
        #  so we retain float dtype
        ser: Series = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected: DataFrame = DataFrame({"A": ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self) -> None:
        # GH#40480
        df: DataFrame = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser: Series = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        # pre-2.0 this setting swapped in a new array, now it is inplace
        #  consistent with non-split-path
        expected: DataFrame = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self) -> None:
        # GH#40480
        df: DataFrame = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        df["B"] = "string