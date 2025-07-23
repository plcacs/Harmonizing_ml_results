"""test label based indexing with loc"""
from collections import namedtuple
import contextlib
from datetime import date, datetime, time, timedelta
import re
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
from typing import Any, List, Tuple, Union, Callable
from pandas._typing import Scalar, Label

@pytest.mark.parametrize(
    "series, new_series, expected_ser",
    [
        [[np.nan, np.nan, "b"], ["a", np.nan, np.nan], [False, True, True]],
        [[np.nan, "b"], ["a", np.nan], [False, True]],
    ],
)
def test_not_change_nan_loc(
    series: List[Union[float, str]], new_series: List[Union[float, str]], expected_ser: List[bool]
) -> None:
    df = DataFrame({"A": series})
    df.loc[:, "A"] = new_series
    expected = DataFrame({"A": expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)


class TestLoc:
    def test_none_values_on_string_columns(self, using_infer_string: bool) -> None:
        df = DataFrame(["1", "2", None], columns=["a"], dtype=object)
        assert df.loc[2, "a"] is None
        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        if using_infer_string:
            assert np.isnan(df.loc[2, "a"])
        else:
            assert df.loc[2, "a"] is None

    def test_loc_getitem_int(self, frame_or_series: Union[DataFrame, Series]) -> None:
        obj = frame_or_series(range(3), index=Index(list("abc"), dtype=object))
        check_indexing_smoketest_or_raises(obj, "loc", 2, fails=KeyError)

    def test_loc_getitem_label(self, frame_or_series: Union[DataFrame, Series]) -> None:
        obj = frame_or_series()
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
    def test_loc_getitem_label_out_of_range(
        self, key: Any, index: Index, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize("key", [[0, 1, 2], [1, 3.0, "A"]])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_loc_getitem_label_list(
        self, key: List[Any], dtype: np.dtype, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        obj = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
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
        "key",
        [
            [0, 1, 2],
            [0, 2, 10],
            [3, 6, 7],
            [(1, 3), (1, 4), (2, 5)],
        ],
    )
    def test_loc_getitem_label_list_with_missing(
        self,
        key: List[Any],
        index: Union[Index, MultiIndex, None],
        frame_or_series: Union[DataFrame, Series],
    ) -> None:
        if index is None:
            obj = frame_or_series()
        else:
            obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_loc_getitem_label_list_fails(
        self, dtype: np.dtype, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        obj = frame_or_series(range(3), Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(
            obj, "loc", [20, 30, 40], axes=1, fails=KeyError
        )

    def test_loc_getitem_bool(self, frame_or_series: Union[DataFrame, Series]) -> None:
        obj = frame_or_series()
        b: List[bool] = [True, False, True, False]
        check_indexing_smoketest_or_raises(obj, "loc", b, fails=IndexError)

    @pytest.mark.parametrize(
        "slc,indexes,axes,fails",
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
            [
                slice(2, 8),
                [Index([2, 4, "null", 8], dtype=object)],
                0,
                TypeError,
            ],
            [
                slice(2, 8),
                [Index([2, 4, "null", 8], dtype=object)],
                1,
                KeyError,
            ],
            [
                slice(2, 4, 2),
                [Index([2, 4, "null", 8], dtype=object)],
                0,
                TypeError,
            ],
        ],
    )
    def test_loc_getitem_label_slice(
        self,
        slc: slice,
        indexes: List[Union[Index, MultiIndex, None]],
        axes: Union[int, None],
        fails: Any,
        frame_or_series: Union[DataFrame, Series],
    ) -> None:
        for index in indexes:
            if index is None:
                obj = frame_or_series()
            else:
                obj = frame_or_series(range(len(index)), index=index)
            check_indexing_smoketest_or_raises(
                obj, "loc", slc, axes=axes, fails=fails
            )

    def test_setitem_from_duplicate_axis(self) -> None:
        df = DataFrame(
            [[20, "a"], [200, "a"], [200, "a"]],
            columns=["col1", "col2"],
            index=[10, 1, 1],
        )
        df.loc[1, "col1"] = np.arange(2)
        expected = DataFrame(
            [[20, "a"], [0, "a"], [1, "a"]],
            columns=["col1", "col2"],
            index=[10, 1, 1],
        )
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self) -> None:
        df = DataFrame(
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
        df2 = DataFrame(
            data={"A": ["String 3"], "B": [Timestamp("2019-06-11 12:00:00")]}
        )
        df.loc[df["A"].isna(), ["A", "B"]] = df2.values
        expected = DataFrame(
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
    def test_loc_getitem_single_boolean_arg(
        self, obj: Union[DataFrame, Series], key: Any, exp: Any
    ) -> None:
        res = obj.loc[key]
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp


class TestLocBaseIndependent:
    def test_loc_npstr(self) -> None:
        df = DataFrame(index=date_range("2021", "2022"))
        result = df.loc[np.array(["2021/6/1"])[0] :]
        expected = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "msg, key",
        [
            (
                "Period\\('2019', 'Y-DEC'\\), 'foo', 'bar'",
                (Period(2019), "foo", "bar"),
            ),
            (
                "Period\\('2019', 'Y-DEC'\\), 'y1', 'bar'",
                (Period(2019), "y1", "bar"),
            ),
            (
                "Period\\('2019', 'Y-DEC'\\), 'foo', 'z1'",
                (Period(2019), "foo", "z1"),
            ),
            (
                "Period\\('2018', 'Y-DEC'\\), Period\\('2016', 'Y-DEC'\\), 'bar'",
                (Period(2018), Period(2016), "bar"),
            ),
            (
                "Period\\('2018', 'Y-DEC'\\), 'foo', 'y1'",
                (Period(2018), "foo", "y1"),
            ),
            (
                "Period\\('2017', 'Y-DEC'\\), 'foo', Period\\('2015', 'Y-DEC'\\)",
                (Period(2017), "foo", Period(2015)),
            ),
            (
                "Period\\('2017', 'Y-DEC'\\), 'z1', 'bar'",
                (Period(2017), "z1", "bar"),
            ),
        ],
    )
    def test_contains_raise_error_if_period_index_is_in_multi_index(
        self, msg: str, key: Tuple[Any, ...]
    ) -> None:
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        df = DataFrame(
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
        df = DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="א"):
            df.loc[:, "א"]

    def test_loc_getitem_dups(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).random((20, 5)),
            index=["ABCDE"[x % 5] for x in range(20)],
        )
        expected = df.loc["A", 0]
        result = df.loc[:, 0].loc["A"]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self) -> None:
        df = DataFrame(
            [[1, 2, "foo", "bar", Timestamp("20130101")]],
            columns=["a", "a", "a", "a", "a"],
            index=[1],
        )
        expected = Series(
            [1, 2, "foo", "bar", Timestamp("20130101")],
            index=["a", "a", "a", "a", "a"],
            name=1,
        )
        result = df.iloc[0]
        tm.assert_series_equal(result, expected)
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self) -> None:
        df_orig = DataFrame(
            {
                "me": list("rttti"),
                "foo": list("aaade"),
                "bar": np.arange(5, dtype="float64") * 1.34 + 2,
                "bar2": np.arange(5, dtype="float64") * -0.34 + 2,
            }
        ).set_index("me")
        indexer: Tuple[str, List[str]]
        indexer = ("r", ["bar", "bar2"])
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])
        indexer = ("r", "bar")
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]
        indexer = ("t", ["bar", "bar2"])
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self) -> None:
        df1 = DataFrame({"a": [0, 1, 1], "b": Series([100, 200, 300], dtype="uint32")})
        ix = df1["a"] == 1
        newb1 = df1.loc[ix, "b"] + 1
        df1.loc[ix, "b"] = newb1
        expected = DataFrame({"a": [0, 1, 1], "b": Series([100, 201, 301], dtype="uint32")})
        tm.assert_frame_equal(df1, expected)
        df2 = DataFrame({"a": [0, 1, 1], "b": [100, 200, 300]}, dtype="uint64")
        ix = df1["a"] == 1
        newb2 = df2.loc[ix, "b"]
        with pytest.raises(TypeError, match="Invalid value"):
            df1.loc[ix, "b"] = newb2

    def test_loc_setitem_dtype(self) -> None:
        df = DataFrame(
            {"id": ["A"], "a": [1.2], "b": [0.0], "c": [-2.5]}
        )
        cols = ["a", "b", "c"]
        df.loc[:, cols] = df.loc[:, cols].astype("float32")
        expected = DataFrame(
            {
                "id": ["A"],
                "a": np.array([1.2], dtype="float64"),
                "b": np.array([0.0], dtype="float64"),
                "c": np.array([-2.5], dtype="float64"),
            }
        )
        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self) -> None:
        s = Series(range(3), index=["a", "b", "c"])
        with pytest.raises(KeyError, match="not in index"):
            s[["a", "d"]]
        s = Series(range(3))
        with pytest.raises(KeyError, match="not in index"):
            s[[0, 3]]

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index: List[bool]) -> None:
        s = Series([1, 2, 3])
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self) -> None:
        pass

    def test_loc_to_fail(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            index=["a", "b", "c"],
            columns=["e", "f", "g"],
        )
        msg = f"\"None of \\[Index\\(\\[1, 2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\""
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self) -> None:
        s = Series(dtype=object)
        s.loc[1] = 1
        s.loc["a"] = 2
        with pytest.raises(KeyError, match="^-1$"):
            s.loc[-1]
        msg = f"\"None of \\[Index\\(\\[-1, -2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\""
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]
        msg = "\\\"None of \\[Index\\(\\['4'\\], dtype='object'\\)\\] are in the \\[index\\]\\\""
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(["4"], dtype=object)]
        s.loc[-1] = 3
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[-1, -2]]
        s["a"] = 2
        msg = f"\"None of \\[Index\\(\\[-2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\""
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]
        del s["a"]
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0

    def test_loc_to_fail3(self) -> None:
        df = DataFrame([["a"], ["b"]], index=[1, 2], columns=["value"])
        msg = f"\"None of \\[Index\\(\\[3\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\""
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]
        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self) -> None:
        s = Series([1, 2, 3])
        s.loc[[2]]
        msg = "None of [RangeIndex(start=3, stop=4, step=1)] are in the [index]"
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[2, 3]]

    def test_loc_index(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).random(size=(5, 10)),
            index=["alpha_0", "alpha_1", "alpha_2", "beta_0", "beta_1"],
        )
        mask = df.index.map(lambda x: "alpha" in x)
        expected = df.loc[np.array(mask)]
        result = df.loc[mask]
        tm.assert_frame_equal(result, expected)
        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)
        result = df.loc[pd.array(mask, dtype="boolean")]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).random((4, 4)),
            columns=["A", "B", "C", "D"],
            index=["A", "B", "C", "D"],
        )
        result = df.loc[:, "A":"B"].iloc[0:2, :]
        assert (result.columns == ["A", "B"]).all()
        assert (result.index == ["A", "B"]).all()
        result = DataFrame({"a": [Timestamp("20130101")], "b": [1]}).iloc[0]
        expected = Series(
            [Timestamp("20130101"), 1],
            index=["a", "b"],
            name=0,
        )
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
        "val,expected",
        [
            (0, 0),
            (np.array(0, dtype=np.int64), 0),
            (np.array([0, 0, 0, 0, 0], dtype=np.int64), 0),
        ],
    )
    def test_loc_setitem_consistency(
        self, frame_for_consistency: DataFrame, val: Union[int, np.ndarray], expected: Any
    ) -> None:
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = val

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency: DataFrame) -> None:
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = "foo"

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency: DataFrame) -> None:
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = 1.0

    def test_loc_setitem_consistency_single_row(self) -> None:
        df = DataFrame({"date": Series([Timestamp("20180101")])})
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "date"] = "string"

    def test_loc_setitem_consistency_empty(self) -> None:
        expected = DataFrame(columns=["x", "y"])
        df = DataFrame(columns=["x", "y"])
        with tm.assert_produces_warning(None):
            df.loc[:, "x"] = 1
        tm.assert_frame_equal(df, expected)
        df = DataFrame(columns=["x", "y"])
        df["x"] = 1
        expected["x"] = expected["x"].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("using_infer_string", [True, False])
    def test_loc_setitem_consistency_slice_column_len(
        self, using_infer_string: bool
    ) -> None:
        levels = [["Region_1"] * 4, ["Site_1", "Site_1", "Site_2", "Site_2"], [3987227376, 3980680971, 3977723249, 3977723089]]
        mi = MultiIndex.from_arrays(levels, names=["Region", "Site", "RespondentID"])
        clevels = [
            [
                "Respondent",
                "Respondent",
                "Respondent",
                "OtherCat",
                "OtherCat",
            ],
            ["Something", "StartDate", "EndDate", "Yes/No", "SomethingElse"],
        ]
        cols = MultiIndex.from_arrays(clevels, names=["Level_0", "Level_1"])
        values = [
            ["A", "5/25/2015 10:59", "5/25/2015 11:22", "Yes", np.nan],
            ["A", "5/21/2015 9:40", "5/21/2015 9:52", "Yes", "Yes"],
            ["A", "5/20/2015 8:27", "5/20/2015 8:41", "Yes", np.nan],
            ["A", "5/20/2015 8:33", "5/20/2015 9:09", "Yes", "No"],
        ]
        df = DataFrame(values, index=mi, columns=cols)
        ctx: contextlib._GeneratorContextManager[Any] = contextlib.nullcontext()
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
            return
        df = df.infer_objects()
        df.loc[:, ("Respondent", "Duration")] = (
            df.loc[:, ("Respondent", "EndDate")] - df.loc[:, ("Respondent", "StartDate")]
        )
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, ("Respondent", "Duration")] = df.loc[:, ("Respondent", "Duration")] / Timedelta(
                60000000000
            )

    @pytest.mark.parametrize("unit", ["Y", "M", "D", "h", "m", "s", "ms", "us"])
    def test_loc_assign_non_ns_datetime(
        self, unit: str
    ) -> None:
        df = DataFrame(
            {
                "timestamp": [
                    np.datetime64("2017-02-11 12:41:29"),
                    np.datetime64("1991-11-07 04:22:37"),
                ]
            }
        )
        df.loc[:, unit] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        df["expected"] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        expected = Series(df.loc[:, "expected"], name=unit)
        tm.assert_series_equal(df.loc[:, unit], expected)

    def test_loc_modify_datetime(self) -> None:
        df = DataFrame.from_dict(
            {"date": [1485264372711, 1485265925110, 1540215845888, 1540282121025]}
        )
        df["date_dt"] = (
            to_datetime(df["date"], unit="ms", cache=True)
            .dt.as_unit("ms")
        )
        df.loc[:, "date_dt_cp"] = df.loc[:, "date_dt"]
        df.loc[[2, 3], "date_dt_cp"] = df.loc[[2, 3], "date_dt"]
        expected = DataFrame(
            [
                [1485264372711, "2017-01-24 13:26:12.711", "2017-01-24 13:26:12.711"],
                [1485265925110, "2017-01-24 13:52:05.110", "2017-01-24 13:52:05.110"],
                [1540215845888, "2018-10-22 13:44:05.888", "2018-10-22 13:44:05.888"],
                [1540282121025, "2018-10-23 08:08:41.025", "2018-10-23 08:08:41.025"],
            ],
            columns=["date", "date_dt", "date_dt_cp"],
        )
        columns = ["date_dt", "date_dt_cp"]
        expected[columns] = expected[columns].apply(to_datetime)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex(self) -> None:
        df = DataFrame(index=[3, 5, 4], columns=["A"], dtype=float)
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({"A": ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self) -> None:
        df = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        expected = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self) -> None:
        df = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[slice(3, 0, -1), "A"] = np.array([1, 2, 3], dtype="int64")
        expected = DataFrame(
            {"A": [3.0, 2.0, 1.0], "B": "string"}, index=[1, 2, 3]
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_frame(self) -> None:
        keys1 = ["@" + str(i) for i in range(5)]
        val1 = np.arange(5, dtype="int64")
        keys2 = ["@" + str(i) for i in range(4)]
        val2 = np.arange(4, dtype="int64")
        index = list(set(keys1).union(keys2))
        df = DataFrame(index=index)
        df["A"] = np.nan
        df.loc[keys1, "A"] = val1
        df["B"] = np.nan
        df.loc[keys2, "B"] = val2
        sera = Series(val1, index=keys1, dtype=np.float64)
        serb = Series(val2, index=keys2)
        expected = DataFrame(
            {"A": sera, "B": serb}, columns=Index(["A", "B"], dtype=object)
        ).reindex(index=index)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=list("abcd"),
            columns=list("ABCD"),
        )
        result = df.iloc[0, 0]
        df.loc["a", "A"] = 1
        result = df.loc["a", "A"]
        assert result == 1
        result = df.iloc[0, 0]
        assert result == 1
        df.loc[:, "B":"D"] = 0
        expected = df.loc[:, "B":"D"]
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self) -> None:
        df = DataFrame({"A": [1, 2, 3], "B": np.nan})
        df.loc[df.B > df.A, "B"] = df.A
        expected = DataFrame({"A": [1, 2, 3], "B": np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self) -> None:
        df = DataFrame({1: [1, 2], 2: [3, 4], "a": ["a", "b"]})
        result = df.loc[0, [1, 2]]
        expected = Series(
            [1, 3],
            index=Index([1, 2], dtype=object),
            dtype=object,
            name=0,
        )
        tm.assert_series_equal(result, expected)
        expected = DataFrame({"1": [5, 2], "2": [6, 4], "a": ["a", "b"]})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(self) -> None:
        df = DataFrame(
            {"A": ["foo", "bar", "baz"], "B": Series(range(3), dtype=np.int64)}
        )
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame({"A": ["bar", "baz", "baz"], "B": Series([1, 2, 2], dtype=np.int64)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame(
            {"date": date_range("2000-01-01", "2000-01-5"), "val": Series(range(5), dtype=np.int64)}
        )
        expected = DataFrame(
            {
                "date": [
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000103"),
                ],
                "val": Series([0, 1, 0, 1, 2], dtype=np.int64),
            }
        )
        expected["date"] = expected["date"].astype("M8[ns]")
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "indexer, expected",
        [
            (
                ["A"],
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                slice(None, "A", None),
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                np.array(["A"]),
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                "var",
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                ["var"],
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                ("var", slice(None)),
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                ["A"],
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                ["A"],
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "value,expected",
        [
            (
                ["Z"],
                [0, 0, 3, 4, 5, 6, 7, 8, 9, 10],
            ),
            (
                ["A"],
                [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                [0, 1, 2],
                [0, 0, 0, 0, 4, 5, 6, 7, 8, 9],
            ),
            (
                [1, 2],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                [1],
                [0, 1, 0, 3, 4, 5, 6, 7, 8, 9],
            ),
        ],
    )
    def test_loc_setitem_missing_columns(
        self,
        indexer: Union[List[Any], Tuple[Any, ...]],
        value: List[Any],
        expected: List[Any],
    ) -> None:
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "B"])
        df.loc[indexer] = value
        tm.assert_frame_equal(
            df, DataFrame({"A": expected, "B": [2, 4, 6]}), check_index_type=True
        )

    def test_loc_setitem_coercion(self, using_infer_string: bool) -> None:
        df = DataFrame({"date": [Timestamp("2010-10-01")] * 3})
        df.loc[0:1, "date"] = np.datetime64("2008-08-08")
        assert Timestamp("2008-08-08") == df.loc[0, "date"]
        assert Timestamp("2008-08-08") == df.loc[1, "date"]
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[2, "date"] = date(2005, 5, 5)

    @pytest.mark.parametrize("idxer", ["var", ["var"]])
    def test_loc_setitem_new_key_tz(
        self, idxer: Union[str, List[str]], tz_naive_fixture: str
    ) -> None:
        vals = [
            to_datetime(42).tz_localize("UTC"),
            to_datetime(666).tz_localize("UTC"),
        ]
        expected = Series(vals, index=Index(["foo", "bar"], dtype=object))
        ser = Series(dtype=object)
        indexer_sl: Callable[[Series[Any]], Series[Any]] = idxer
        indexer_sl(ser)["foo"] = vals[0]
        indexer_sl(ser)["bar"] = vals[1]
        tm.assert_series_equal(ser, expected)

    def test_loc_non_unique(self) -> None:
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]},
            index=[0, 1, 0, 1, 2, 3],
        )
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:]
        msg = "'Cannot get left slice bound for non-unique label: 0'"
        with pytest.raises(KeyError, match=msg):
            df.loc[0:]
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:2]
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]},
            index=[0, 1, 0, 1, 2, 3],
        ).sort_index(axis=0)
        result = df.loc[1:]
        expected = DataFrame(
            {"A": [2, 4, 5, 6], "B": [4, 6, 7, 8]}, index=[1, 1, 2, 3]
        )
        tm.assert_frame_equal(result, expected)
        result = df.loc[0:]
        tm.assert_frame_equal(result, df)
        result = df.loc[1:2]
        expected = DataFrame(
            {"A": [2, 4, 5], "B": [4, 6, 7]}, index=[1, 1, 2]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("length, l2", [[900, 100], [900000, 100000]])
    def test_loc_non_unique_memory_error(
        self, length: int, l2: int
    ) -> None:
        columns = list("ABCDEFG")
        df = pd.concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((length, len(columns))),
                    index=np.arange(length),
                    columns=columns,
                ),
                DataFrame(
                    np.ones((l2, len(columns))),
                    index=[0] * l2,
                    columns=columns,
                ),
            ]
        )
        assert not df.index.is_unique
        mask = np.arange(l2)
        result = df.loc[mask]
        expected = pd.concat(
            [
                df.take([0]),
                DataFrame(
                    np.ones((len(mask), len(columns))),
                    index=[0] * len(mask),
                    columns=columns,
                ),
                df.take(mask[1:]),
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_name(self) -> None:
        df = DataFrame([[1, 1], [1, 1]])
        df.index.name = "index_name"
        result = df.iloc[[0, 1]].index.name
        assert result == "index_name"
        result = df.loc[[0, 1]].index.name
        assert result == "index_name"

    def test_loc_empty_list_indexer_is_ok(self) -> None:
        df = DataFrame(
            np.ones((5, 2)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(2)], name="a"),
        )
        tm.assert_frame_equal(
            df.loc[:, []],
            df.iloc[:, :0],
            check_index_type=True,
            check_column_type=True,
        )
        tm.assert_frame_equal(
            df.loc[[], :],
            df.iloc[:0, :],
            check_index_type=True,
            check_column_type=True,
        )
        tm.assert_frame_equal(
            df.loc[[]],
            df.iloc[:0, :],
            check_index_type=True,
            check_column_type=True,
        )

    def test_identity_slice_returns_new_object(self) -> None:
        original_df = DataFrame({"a": [1, 2, 3]})
        sliced_df = original_df.loc[:]
        assert sliced_df is not original_df
        assert original_df[:] is not original_df
        assert original_df.loc[:, :] is not original_df
        assert np.shares_memory(original_df["a"]._values, sliced_df["a"]._values)
        original_df.loc[:, "a"] = [4, 4, 4]
        assert (sliced_df["a"] == [1, 2, 3]).all()
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        assert df[0] is not df.loc[:, 0]
        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.loc[:]
        assert sliced_series is not original_series
        assert original_series[:] is not original_series
        original_series[:3] = [7, 8, 9]
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_loc_copy_vs_view(self, request: Any) -> None:
        x = DataFrame(zip(range(3), range(3)), columns=["a", "b"])
        y = x.copy()
        q = y.loc[:, "a"]
        q += 2
        tm.assert_frame_equal(x, y)
        z = x.copy()
        q = z.loc[x.index, "a"]
        q += 2
        tm.assert_frame_equal(x, z)

    def test_loc_uint64(self) -> None:
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])
        result = ser.loc[umax - 1]
        expected = ser.iloc[0]
        assert result == expected
        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        tm.assert_series_equal(result, expected)
        result = ser.loc[[umax - 1, umax]]
        tm.assert_series_equal(result, ser)

    def test_loc_uint64_disallow_negative(self) -> None:
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])
        with pytest.raises(KeyError, match="-1"):
            ser.loc[-1]
        with pytest.raises(KeyError, match="-1"):
            ser.loc[[-1]]

    def test_loc_setitem_empty_append_expands_rows(self) -> None:
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )
        df = DataFrame(columns=["x", "y"])
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self) -> None:
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )
        df = DataFrame(columns=["x", "y"])
        df["x"] = df["x"].astype(np.int64)
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self) -> None:
        expected = DataFrame({"x": [1.0], "y": [np.nan]})
        df = DataFrame(columns=["x", "y"], dtype=float)
        df.loc[0, "x"] = expected.loc[0, "x"]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self) -> None:
        data = [1, 2]
        df = DataFrame(columns=["x", "y"])
        df.index = df.index.astype(np.int64)
        msg = "None of .*Index.* are in the \\[index\\]"
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], "x"] = data
        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, "x"] = data

    def test_indexing_zerodim_np_array(self) -> None:
        df = DataFrame([[1, 2], [3, 4]])
        result = df.loc[np.array(0)]
        s = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self) -> None:
        s = Series([1, 2])
        result = s.loc[np.array(0)]
        assert result == 1

    def test_loc_reverse_assignment(self) -> None:
        data = [1, 2, 3, 4, 5, 6] + [None] * 4
        expected = Series(data, index=range(2010, 2020))
        result = Series(index=range(2010, 2020), dtype=np.float64)
        result.loc[2015:2010:-1] = [6, 5, 4, 3, 2, 1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_str_to_small_float_conversion_type(self, using_infer_string: bool) -> None:
        col_data = [
            str(np.random.default_rng(2).random() * 1e-12) for _ in range(5)
        ]
        result = DataFrame(col_data, columns=["A"])
        expected = DataFrame(col_data, columns=["A"])
        tm.assert_frame_equal(result, expected)
        if using_infer_string:
            with pytest.raises(TypeError, match="Invalid value"):
                result.loc[result.index, "A"] = [float(x) for x in col_data]
        else:
            result.loc[result.index, "A"] = [float(x) for x in col_data]
            expected = DataFrame(
                col_data, columns=["A"], dtype=float
            ).astype(object)
            tm.assert_frame_equal(result, expected)
        result["A"] = [float(x) for x in col_data]
        expected = DataFrame({"A": [float(x) for x in col_data]}, dtype=float)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_time_object(
        self, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        mask = (rng.hour == 9) & (rng.minute == 30)
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)),
            index=rng,
        )
        obj = tm.get_obj(obj, frame_or_series)
        result = obj.loc[time(9, 30)]
        exp = obj.loc[mask]
        tm.assert_equal(result, exp)
        chunk = obj.loc["1/4/2000":]
        result = chunk.loc[time(9, 30)]
        expected = result[-1:]
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "spmatrix_t",
        ["coo_matrix", "csc_matrix", "csr_matrix"],
    )
    @pytest.mark.parametrize(
        "dtype", [np.complex128, np.float64, np.int64, bool],
    )
    def test_loc_getitem_range_from_spmatrix(
        self, spmatrix_t: str, dtype: np.dtype
    ) -> None:
        sp_sparse = pytest.importorskip("scipy.sparse")
        spmatrix_t = getattr(sp_sparse, spmatrix_t)
        rows, cols = (5, 7)
        spmatrix = spmatrix_t(
            np.eye(rows, cols, dtype=dtype), dtype=dtype
        )
        df = DataFrame.sparse.from_spmatrix(spmatrix)
        itr_idx = range(2, rows)
        result = np.nan_to_num(df.loc[itr_idx].values)
        expected = spmatrix.toarray()[itr_idx]
        tm.assert_numpy_array_equal(result, expected)
        result = df.loc[itr_idx].dtypes.values
        expected = np.full(cols, SparseDtype(dtype))
        tm.assert_numpy_array_equal(result, expected)

    def test_loc_getitem_listlike_all_retains_sparse(self) -> None:
        df = DataFrame({"A": pd.array([0, 0], dtype=SparseDtype("int64"))})
        result = df.loc[[0, 1]]
        tm.assert_frame_equal(result, df)

    def test_loc_getitem_sparse_frame(self) -> None:
        sp_sparse = pytest.importorskip("scipy.sparse")
        df = DataFrame.sparse.from_spmatrix(sp_sparse.eye(5, dtype=np.int64))
        result = df.loc[range(2)]
        expected = DataFrame(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=SparseDtype(np.int64),
        )
        tm.assert_frame_equal(result, expected)
        result = df.loc[range(2)].loc[range(1)]
        expected = DataFrame([[1, 0, 0, 0, 0]], dtype=SparseDtype(np.int64))
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sparse_series(self) -> None:
        s = Series(
            [1.0, 0.0, 0.0, 0.0, 0.0],
            dtype=SparseDtype("float64", 0.0),
        )
        result = s.loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)
        result = s.loc[range(3)].loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer",
        ["loc", "iloc"],
    )
    def test_getitem_single_row_sparse_df(
        self, indexer: str
    ) -> None:
        df = DataFrame(
            [[1.0, 0.0, 1.5], [0.0, 2.0, 0.0]],
            dtype=SparseDtype(float),
        )
        result = getattr(df, indexer)[0]
        expected = Series(
            [1.0, 0.0, 1.5],
            dtype=SparseDtype(float),
            name=0,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "key_type",
        [iter, np.array, Series, Index],
    )
    def test_loc_getitem_iterable(
        self,
        float_frame: DataFrame,
        key_type: Callable[[List[Any]], Any],
    ) -> None:
        idx = key_type(["A", "B", "C"])
        result = float_frame.loc[:, idx]
        expected = float_frame.loc[:, ["A", "B", "C"]]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_timedelta_0seconds(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).normal(size=(10, 4)),
            index=timedelta_range(start="0s", periods=10, freq="s"),
        )
        result = df.loc[np.timedelta64(0, "s"):, :]
        expected = df.iloc[0:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "val,expected",
        [
            (2**63 - 1, 1),
            (2**63, 2),
        ],
    )
    def test_loc_getitem_uint64_scalar(
        self, val: int, expected: int
    ) -> None:
        df = DataFrame([1, 2], index=[2**63 - 1, 2**63])
        result = df.loc[val]
        expected = Series([expected], name=val)
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_int_label_with_float_index(
        self, float_numpy_dtype: np.dtype
    ) -> None:
        dtype = float_numpy_dtype
        ser = Series(["a", "b", "c"], index=Index([0, 0.5, 1], dtype=dtype))
        expected = ser.copy()
        ser.loc[1] = "zoo"
        expected.iloc[2] = "zoo"
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "indexer, expected",
        [
            (
                0,
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                slice(4, 8),
                [0, 1, 2, 3, 20, 20, 20, 20, 8, 9],
            ),
            (
                [3, 5],
                [0, 1, 2, 20, 4, 20, 6, 7, 8, 9],
            ),
        ],
    )
    def test_loc_setitem_listlike_with_timedelta64index(
        self,
        indexer: Union[int, slice, List[int]],
        expected: List[Any],
    ) -> None:
        tdi = to_timedelta(range(10), unit="s")
        df = DataFrame({"x": range(10)}, dtype="int64", index=tdi)
        df.loc[df.index[indexer], "x"] = 20
        expected = DataFrame(
            {
                "x": [0, 1, 2, 20, 4, 20, 6, 7, 8, 9],
            },
            index=tdi,
            columns=["x"],
            dtype="int64",
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_values_partial_column_slice(
        self,
        using_infer_string: bool,
    ) -> None:
        df = DataFrame(
            {"a": [1, 1, 1, 1, 1], "b": list("aaaaa")}, dtype="object"
        )
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[1:2, "a"] = Categorical(["b", "b"], categories=["a", "b"])
            df.loc[2:3, "b"] = Categorical(["b", "b"], categories=["a", "b"])

    def test_loc_setitem_single_row_categorical(
        self,
        using_infer_string: bool,
    ) -> None:
        df = DataFrame(
            {"Alpha": ["a"], "Numeric": [0]}
        )
        categories = Categorical(df["Alpha"], categories=["a", "b", "c"])
        df.loc[:, "Alpha"] = categories
        result = df["Alpha"]
        expected = Series(
            categories,
            index=df.index,
            name="Alpha",
        ).astype(object if not using_infer_string else "str")
        tm.assert_series_equal(result, expected)
        df["Alpha"] = categories
        tm.assert_series_equal(df["Alpha"], Series(categories, name="Alpha"))

    def test_loc_setitem_datetime_coercion(self) -> None:
        df = DataFrame({"c": [1485264372711, 1485265925110, 1540215845888, 1540282121025]})
        df["date_dt"] = (
            to_datetime(df["c"], unit="ms", cache=True).dt.as_unit("ms")
        )
        df.loc[:, "date_dt_cp"] = df.loc[:, "date_dt"]
        df.loc[[2, 3], "date_dt_cp"] = df.loc[[2, 3], "date_dt"]
        expected = DataFrame(
            [
                [1485264372711, "2017-01-24 13:26:12.711", "2017-01-24 13:26:12.711"],
                [
                    1485265925110,
                    "2017-01-24 13:52:05.110",
                    "2017-01-24 13:52:05.110",
                ],
                [1540215845888, "2018-10-22 13:44:05.888", "2018-10-22 13:44:05.888"],
                [1540282121025, "2018-10-23 08:08:41.025", "2018-10-23 08:08:41.025"],
            ],
            columns=["date", "date_dt", "date_dt_cp"],
        )
        columns = ["date_dt", "date_dt_cp"]
        expected[columns] = expected[columns].apply(to_datetime)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self) -> None:
        df = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        expected = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self) -> None:
        df = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[slice(3, 0, -1), "A"] = np.array([1, 2, 3], dtype="int64")
        expected = DataFrame(
            {"A": [3.0, 2.0, 1.0], "B": "string"}, index=[1, 2, 3]
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows(self) -> None:
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )
        df = DataFrame(columns=["x", "y"])
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self) -> None:
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )
        df = DataFrame(columns=["x", "y"])
        df["x"] = df["x"].astype(np.int64)
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self) -> None:
        expected = DataFrame({"x": [1.0], "y": [np.nan]})
        df = DataFrame(columns=["x", "y"], dtype=float)
        df.loc[0, "x"] = expected.loc[0, "x"]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self) -> None:
        data = [1, 2]
        df = DataFrame(columns=["x", "y"])
        df.index = df.index.astype(np.int64)
        msg = "None of .*Index.* are in the \\[index\\]"
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], "x"] = data
        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, "x"] = data

    def test_loc_setitem_frame_with_reindex(self) -> None:
        df = DataFrame(index=[3, 5, 4], columns=["A"], dtype=float)
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({"A": ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self) -> None:
        df = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        expected = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self) -> None:
        df = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[slice(3, 0, -1), "A"] = np.array([1, 2, 3], dtype="int64")
        expected = DataFrame(
            {"A": [3.0, 2.0, 1.0], "B": "string"}, index=[1, 2, 3]
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=list("abcd"),
            columns=list("ABCD"),
        )
        result = df.iloc[0, 0]
        df.loc["a", "A"] = 1
        result = df.loc["a", "A"]
        assert result == 1
        result = df.iloc[0, 0]
        assert result == 1
        df.loc[:, "B":"D"] = 0
        expected = df.loc[:, "B":"D"]
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self) -> None:
        df = DataFrame({"A": [1, 2, 3], "B": np.nan})
        df.loc[df.B > df.A, "B"] = df.A
        expected = DataFrame({"A": [1, 2, 3], "B": np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self) -> None:
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": ["a", "b"]})
        result = df.loc[0, [1, 2]]
        expected = Series(
            [1, 3],
            index=Index([1, 2], dtype=object),
            dtype=object,
            name=0,
        )
        tm.assert_series_equal(result, expected)
        expected = DataFrame({"1": [5, 2], "2": [6, 4], "a": ["a", "b"]})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(self) -> None:
        df = DataFrame(
            {"A": ["foo", "bar", "baz"], "B": Series(range(3), dtype=np.int64)}
        )
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame(
            {"A": ["bar", "baz", "baz"], "B": Series([1, 2, 2], dtype=np.int64)}
        )
        tm.assert_frame_equal(df, expected)
        df = DataFrame(
            {"date": date_range("2000-01-01", "2000-01-5"), "val": Series(range(5), dtype=np.int64)}
        )
        expected = DataFrame(
            {
                "date": [
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000103"),
                ],
                "val": Series([0, 1, 0, 1, 2], dtype=np.int64),
            }
        )
        expected["date"] = expected["date"].astype("M8[ns]")
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "indexer, exp",
        [
            (
                ["A"],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                slice(1, 3),
                [0, 1, 2, 3, 20, 20, 20, 20, 8, 9],
            ),
            (
                [3, 5],
                [0, 1, 2, 20, 4, 20, 6, 7, 8, 9],
            ),
        ],
    )
    def test_loc_setitem_listlike_with_timedelta64index(
        self,
        indexer: Union[List[int], slice],
        exp: List[Any],
    ) -> None:
        tdi = to_timedelta(range(10), unit="s")
        df = DataFrame({"x": range(10)}, dtype="int64", index=tdi)
        df.loc[df.index[indexer], "x"] = 20
        expected = DataFrame(
            {"x": exp},
            index=tdi,
            columns=["x"],
            dtype="int64",
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_column_retains_dtype(self, ordered: bool) -> None:
        df = DataFrame({"A": [1, 1, 1, 1, 1], "B": list("aaaaa")})
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[df.index[::2], "B"] = np.nan

    def test_loc_setitem_cast2(self) -> None:
        df = DataFrame({"A": np.arange(6, dtype=np.int8)})
        df.loc[1, "A"] = 6
        assert df.dtypes.A == np.dtype(np.int8)
        df.A = np.int8(7)
        assert df.dtypes.A == np.dtype(np.int8)

    def test_loc_setitem_range_key(
        self, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])
        values = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]
        obj.loc[range(3)] = values
        expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        tm.assert_equal(obj, expected)

    def test_loc_setitem_numpy_frame_categorical_value(self) -> None:
        df = DataFrame({"a": [1, 1, 1, 1, 1], "b": list("aaaaa")})
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[1:2, "a"] = Categorical([2, 2], categories=[1, 2])
            df.loc[2:3, "b"] = Categorical([2, 2], categories=[1, 2])

    def test_loc_series_indexing_zerodim_np_array(self) -> None:
        s = Series([1, 2])
        result = s.loc[np.array(0)]
        assert result == 1

    def test_loc_getitem_index_namedtuple(self) -> None:
        IndexType = namedtuple("IndexType", ["a", "b"])
        idx1 = IndexType("foo", "bar")
        idx2 = IndexType("baz", "bof")
        index = Index([idx1, idx2], name="composite_index", tupleize_cols=False)
        df = DataFrame([(1, 2), (3, 4)], index=index, columns=["A", "B"])
        result = df.loc[IndexType("foo", "bar")]["A"]
        assert result == 1

    def test_loc_setitem_single_column_mixed(
        self, using_infer_string: bool
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )
        df["str"] = "qux"
        df.loc[df.index[::2], "str"] = np.nan
        expected = Series(
            [np.nan, "qux", np.nan, "qux", np.nan],
            dtype=object if not using_infer_string else "str",
        ).values
        tm.assert_almost_equal(df["str"].values, expected)

    def test_loc_setitem_cast3(self) -> None:
        df = DataFrame({"one": np.arange(6, dtype=np.int8)})
        df.loc[1, "one"] = 6
        assert df.dtypes.one == np.dtype(np.int8)
        df.one = np.int8(7)
        assert df.dtypes.one == np.dtype(np.int8)

    def test_loc_setitem_range_key_multiindex(
        self, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])
        values = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]
        obj.loc[range(3)] = values
        expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        tm.assert_equal(obj, expected)

    def test_loc_setitem_frequency_preservation(self) -> None:
        idx = date_range("2022-01-01", periods=5, freq='D')
        df = DataFrame({"A": range(5)}, index=idx)
        df.loc['2022-01-06'] = 5
        expected = DataFrame({"A": range(6)}, index=date_range("2022-01-01", periods=6, freq='D'))
        tm.assert_frame_equal(df, expected)


class TestLocWithEllipsis:
    @pytest.fixture
    def indexer(self, indexer_li: Callable[..., Any]) -> Callable[..., Any]:
        return indexer_li

    @pytest.fixture
    def obj(
        self, series_with_simple_index: Series, frame_or_series: Union[DataFrame, Series]
    ) -> Union[DataFrame, Series]:
        obj = series_with_simple_index
        if frame_or_series is not Series:
            obj = obj.to_frame()
        return obj

    def test_loc_iloc_getitem_ellipsis(
        self,
        obj: Union[DataFrame, Series],
        indexer: Callable[..., Any],
    ) -> None:
        result = indexer(obj)[...]
        tm.assert_equal(result, obj)

    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_loc_iloc_getitem_leading_ellipses(
        self,
        series_with_simple_index: Series,
        indexer_li: Callable[..., Any],
    ) -> None:
        obj: Series = series_with_simple_index
        key: Any = 0 if indexer_li is tm.iloc or len(obj) == 0 else obj.index[0]
        if indexer_li is tm.loc and obj.index.inferred_type == "boolean":
            return
        if indexer_li is tm.loc and isinstance(obj.index, MultiIndex):
            msg = "MultiIndex does not support indexing with Ellipsis"
            with pytest.raises(NotImplementedError, match=msg):
                result = indexer_li(obj)[..., [key]]
        elif len(obj) != 0:
            result = indexer_li(obj)[..., [key]]
            expected = indexer_li(obj)[[key]]
            tm.assert_series_equal(result, expected)
        key2: Any = 0 if indexer_li is tm.iloc else obj.name
        df = obj.to_frame()
        result = indexer_li(df)[..., [key2]]
        expected = indexer_li(df)[:, [key2]]
        tm.assert_frame_equal(result, expected)

    def test_loc_iloc_getitem_ellipses_only_one_ellipsis(
        self,
        obj: Union[DataFrame, Series],
        indexer: Callable[..., Any],
    ) -> None:
        key: Any = 0 if indexer is tm.iloc or len(obj) == 0 else obj.index[0]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ...]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., [key], ...]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ..., key]
        with pytest.raises(IndexingError, match="Too many indexers"):
            indexer(obj)[key, ..., ...]

    class TestLocWithMultiIndex:
        @pytest.mark.parametrize(
            "keys, expected",
            [
                (
                    ["b", "a"],
                    [["b", "b", "a", "a"], [1, 2, 1, 2]],
                ),
                (
                    ["a", "b"],
                    [["a", "a", "b", "b"], [1, 2, 1, 2]],
                ),
                (
                    (["a", "b"], [1, 2]),
                    [["a", "a", "b", "b"], [1, 2, 1, 2]],
                ),
                (
                    (["a", "b"], [2, 1]),
                    [["a", "a", "b", "b"], [2, 1, 2, 1]],
                ),
                (
                    (["b", "a"], [2, 1]),
                    [["b", "b", "a", "a"], [2, 1, 2, 1]],
                ),
                (
                    (["b", "a"], [1, 2]),
                    [["b", "b", "a", "a"], [1, 2, 1, 2]],
                ),
                (
                    (["c", "a"], [2, 1]),
                    [["c", "a", "a"], [1, 2, 1]],
                ),
            ],
        )
        @pytest.mark.parametrize(
            "dim",
            ["index", "columns"],
        )
        def test_loc_getitem_multilevel_index_order(
            self,
            dim: str,
            keys: Union[List[Any], Tuple[Any, ...]],
            expected: List[List[Any]],
        ) -> None:
            kwargs = {
                dim: [["c", "a", "a", "b", "b"], [1, 1, 2, 1, 2]]
            }
            df = DataFrame(
                np.arange(25).reshape(5, 5),
                **kwargs,
            )
            exp_index = MultiIndex.from_arrays(expected)
            if dim == "index":
                res = df.loc[keys, :]
                tm.assert_index_equal(res.index, exp_index)
            elif dim == "columns":
                res = df.loc[:, keys]
                tm.assert_index_equal(res.columns, exp_index)

        def test_loc_preserve_names(
            self,
            multiindex_year_month_day_dataframe_random_data: DataFrame,
        ) -> None:
            ymd = multiindex_year_month_day_dataframe_random_data
            result = ymd.loc[2000].index.name
            assert result == ymd.index.names[1:]
            result = ymd["A"].loc[2000].index.name
            assert result == ymd.index.names[1:]
            result = ymd.loc[2000, 2].index.name
            assert result == ymd.index.names[2]
            result = ymd["A"].loc[2000, 2].index.name
            assert result == ymd.index.names[2]

        def test_loc_getitem_slice_multiindex_nonunique_len_zero(self) -> None:
            mi = MultiIndex.from_tuples([("foo", "bar"), ("foo", "bar")])
            ser = Series([0, 1], index=mi)
            res = ser.loc[[]]
            expected = ser[:0]
            tm.assert_series_equal(res, expected)
            res2 = ser.loc[ser.iloc[0:0]]
            tm.assert_series_equal(res2, expected)

        def test_loc_getitem_access_none_value_in_multiindex(self) -> None:
            ser = Series([None], MultiIndex.from_arrays([["Level1"], ["Level2"]]))
            result = ser.loc["Level1", "Level2"]
            assert result is None
            midx = MultiIndex.from_product([["Level1"], ["Level2_a", "Level2_b"]])
            ser = Series([None] * len(midx), dtype=object, index=midx)
            result = ser.loc["Level1", "Level2_a"]
            assert result is None
            ser = Series([1] * len(midx), dtype=object, index=midx)
            result = ser.loc["Level1", "Level2_a"]
            assert result == 1

        def test_loc_setitem_multiindex_slice(self) -> None:
            mi = MultiIndex.from_product(
                [list("ab"), list("xy"), [1, 2]],
                names=["x", "y", "z"],
            )
            ser = Series(range(8), index=mi)
            loc_result = ser.loc["a", :, :]
            expected = ser.index.droplevel(0)[:4]
            tm.assert_index_equal(loc_result.index, expected)

        def test_loc_getitem_slice_datetime_objs_with_datetimeindex(self) -> None:
            times = date_range("2000-01-01", freq="1h", periods=5)
            ser = Series(range(5), index=times)
            result = ser.loc["2000-01-01 01:00": "2000-01-01 04:00"]
            expected = ser.iloc[1:5]
            tm.assert_series_equal(result, expected)

        def test_loc_getitem_slice_datetime_objs_with_non_datetimeindex(self) -> None:
            idx = list("ABCDE")
            ser = Series(range(5), index=idx)
            with pytest.raises(KeyError, match="not in index"):
                ser.loc[slice("A", "C")]

        def test_loc_setitem_preserves_nullable_int(self, any_numeric_ea_dtype: np.dtype) -> None:
            ser = Series([0, 1, 2, 3], dtype=CategoricalDtype(Index([0, 1, 2, 3], dtype=any_numeric_ea_dtype)))
            df = DataFrame({"data": ser})
            result = DataFrame(index=df.index)
            result.loc[df.index, "data"] = ser
            tm.assert_frame_equal(result, df, check_column_type=False)
            result = DataFrame(index=df.index)
            result.loc[df.index, "data"] = ser._values
            tm.assert_frame_equal(result, df, check_column_type=False)

        def test_loc_setitem_ea_not_full_column(self) -> None:
            df = DataFrame(
                {"a": [1, 1, 1, 1, 1], "b": ["a", "a", "a", "a", "a"]}
            )
            df.loc[1:2, "a"] = Categorical([2, 2], categories=[1, 2])
            expected = DataFrame({"a": [1, 2, 2, 1, 1], "b": ["a", "a", "a", "a", "a"]})
            tm.assert_frame_equal(df, expected)

    class TestLocSetitemWithExpansion:
        def test_loc_setitem_with_expansion_large_dataframe(
            self, monkeypatch: pytest.MonkeyPatch
        ) -> None:
            size_cutoff = 50
            with monkeypatch.context():
                monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
                result = DataFrame({"x": range(size_cutoff)}, dtype="int64")
                result.loc[size_cutoff] = size_cutoff
            expected = DataFrame(
                {"x": range(size_cutoff + 1)}, dtype="int64"
            )
            tm.assert_frame_equal(result, expected)

        def test_loc_setitem_empty_series(self) -> None:
            ser = Series(dtype=object)
            ser.loc[1] = 1
            tm.assert_series_equal(ser, Series([1], index=range(1, 2)))
            ser.loc[3] = 3
            tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

        def test_loc_setitem_empty_series_float(self) -> None:
            ser = Series(dtype=object)
            ser.loc[1] = 1.0
            tm.assert_series_equal(ser, Series([1.0], index=range(1, 2)))
            ser.loc[3] = 3.0
            tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

        def test_loc_setitem_empty_series_str_idx(self) -> None:
            ser = Series(dtype=object)
            ser.loc["foo"] = 1
            tm.assert_series_equal(ser, Series([1], index=Index(["foo"], dtype=object)))
            ser.loc["bar"] = 3
            tm.assert_series_equal(
                ser, Series([1, 3], index=Index(["foo", "bar"], dtype=object))
            )
            ser.loc[3] = 4
            tm.assert_series_equal(
                ser,
                Series([1, 3, 4], index=Index(["foo", "bar", 3], dtype=object)),
            )

        def test_loc_setitem_incremental_with_dst(self) -> None:
            base = datetime(2015, 11, 1, tzinfo=gettz("US/Pacific"))
            idxs = [base + timedelta(seconds=i * 900) for i in range(16)]
            result = Series([0], index=[idxs[0]])
            for ts in idxs:
                result.loc[ts] = 1
            expected = Series(1, index=idxs)
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "conv", [lambda x: x, lambda x: x.to_datetime64(), lambda x: x.to_pydatetime(), lambda x: np.datetime64(x)]
        )
        def test_loc_setitem_datetime_keys_cast(
            self, conv: Callable[[Timestamp], Union[Timestamp, np.datetime64, datetime]]
        ) -> None:
            dt1 = Timestamp("20130101 09:00:00")
            dt2 = Timestamp("20130101 10:00:00")
            df = DataFrame(columns=["a"], dtype=np.float64)
            df.loc[conv(dt1), "a"] = 100
            df.loc[conv(dt2), "a"] = 200
            expected = DataFrame(
                {"a": [100.0, 200.0]},
                index=pd.Index([conv(dt1), conv(dt2)], dtype=object),
                columns=pd.Index(["a"], dtype=object),
            )
            tm.assert_frame_equal(df, expected)

        def test_loc_setitem_categorical_column_retains_dtype(
            self, using_infer_string: bool
        ) -> None:
            df = DataFrame(
                {"A": ["a", "b", "c"], "B": ["d", "e", "f"]},
                dtype="category",
            )
            df.loc[:, "A"] = "a"
            expected = DataFrame(
                {"A": ["a", "a", "a"], "B": ["d", "e", "f"]},
                dtype="category",
            )
            tm.assert_frame_equal(df, expected)

        def test_loc_setitem_with_expansion_and_existing_dst(self) -> None:
            start = Timestamp("2017-10-29 00:00:00", tz="Europe/Madrid")
            end = Timestamp("2017-10-29 03:00:00", tz="Europe/Madrid")
            ts = Timestamp("2016-10-10 03:00:00", tz="Europe/Madrid")
            idx = date_range(start, end, inclusive="left", freq="h")
            assert ts not in idx
            result = DataFrame(index=idx, columns=["value"])
            result.loc[ts, "value"] = 12
            expected = DataFrame(
                {"value": [np.nan] * len(idx) + [12]},
                index=idx.append(DatetimeIndex([ts])),
                columns=["value"],
                dtype=object,
            )
            tm.assert_frame_equal(result, expected)

        def test_setitem_with_expansion(self) -> None:
            df = DataFrame(
                {"date": to_datetime(["2015-03-30 20:12:32", "2015-03-12 00:11:11"]), "new_col": ["new", "old"]},
                dtype="object",
            )
            df.date = df.set_index("date").index.tz_localize("UTC")
            v = df[df.new_col == "new"].set_index("date").index.tz_convert("US/Pacific")
            df2 = df.copy()
            df2.loc[df.index, "date"] = v
            expected = DataFrame(
                {"date": v, "new_col": ["new", "old"]},
                index=df.index,
                columns=["date", "new_col"],
            )
            tm.assert_frame_equal(df2, expected)

        def test_loc_setitem_with_expansion_inf_upcast_empty(self) -> None:
            df = DataFrame()
            df.loc[0, 0] = 1
            df.loc[1, 1] = 2
            df.loc[0, np.inf] = 3
            expected = DataFrame(
                {"x": [1, 2, 3], "y": [np.nan, np.nan, 3]},
                dtype="object",
            )
            tm.assert_frame_equal(df, expected)

        def test_loc_setitem_with_expansion_nonunique_index(self, index: Index) -> None:
            if not len(index):
                pytest.skip("Not relevant for empty Index")
            index = index.repeat(2)
            N = len(index)
            arr = np.arange(N).astype(np.int64)
            orig = DataFrame(arr, index=index)
            key = "kapow"
            assert key not in index
            exp_index = index.insert(len(index), key)
            if isinstance(index, MultiIndex):
                assert exp_index[-1][0] == key
            else:
                assert exp_index[-1] == key
            exp_data = np.arange(N + 1).astype(np.float64)
            expected = DataFrame(
                {"x": exp_data},
                index=exp_index,
                columns=["x"],
            )
            df = orig.copy()
            df.loc[key, "x"] = N
            tm.assert_frame_equal(df, expected)
            ser = orig.copy()["x"]
            ser.loc[key] = N
            expected = expected["x"].astype(np.int64)
            tm.assert_series_equal(ser, expected)
            df = orig.copy()
            df.loc[key, "x"] = N
            expected = DataFrame(
                {"x": list(arr) + [np.nan]},
                index=exp_index,
                columns=["x"],
            )
            tm.assert_frame_equal(df, expected)
            df = orig.copy()
            df.loc[key, "x"] = N
            tm.assert_frame_equal(df, expected)

        def test_loc_setitem_nullable_int_drop(self, any_int_numpy_dtype: np.dtype) -> None:
            ser = Series([1, 2, 3], dtype=any_int_numpy_dtype)
            ser.loc[0] = 4
            expected = Series([4, 2, 3], dtype=any_int_numpy_dtype)
            tm.assert_series_equal(ser, expected)

        def test_loc_setitem_td64_non_nano(self) -> None:
            ser = Series(np.array([np.timedelta64(10, "m")] * 10))
            ser.loc[[1, 2, 3]] = np.timedelta64(20, "m")
            expected = Series(np.array([np.timedelta64(10, "m")] * 10))
            expected.loc[[1, 2, 3]] = Timedelta(np.timedelta64(20, "m"))
            tm.assert_series_equal(ser, expected)

        def test_loc_setitem_2d_to_1d_raises(self) -> None:
            data = np.random.default_rng(2).standard_normal((2, 2))
            ser = Series(range(2), dtype="float64")
            msg = "setting an array element with a sequence."
            with pytest.raises(ValueError, match=msg):
                ser.loc[range(2)] = data
            with pytest.raises(ValueError, match=msg):
                ser.loc[:] = data

        def test_loc_getitem_interval_index(self) -> None:
            index = pd.interval_range(start=0, periods=3)
            df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"])
            expected = 1
            result = df.loc[0.5, "A"]
            tm.assert_almost_equal(result, expected)

        def test_loc_getitem_interval_index2(self) -> None:
            index = pd.interval_range(start=0, periods=3, closed="both")
            df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"])
            index_exp = pd.interval_range(start=0, periods=2, freq=1, closed="both")
            expected = Series([1, 4], index=index_exp, name="A")
            result = df.loc[1, "A"]
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "tpl", [(1,), (1, 2)],
        )
        def test_loc_getitem_index_single_double_tuples(
            self,
            tpl: Tuple[Any, ...],
        ) -> None:
            idx = Index([(1,), (1, 2)], name="A", tupleize_cols=False)
            ser = Series([1, 2], index=idx)
            result = ser.loc[[tpl]]
            idx = Index([tpl], name="A", tupleize_cols=False)
            expected = Series([1], index=idx)
            tm.assert_series_equal(result, expected)

        def test_loc_getitem_index_namedtuple(self) -> None:
            IndexType = namedtuple("IndexType", ["a", "b"])
            idx1 = IndexType("foo", "bar")
            idx2 = IndexType("baz", "bof")
            index = Index([idx1, idx2], name="composite_index", tupleize_cols=False)
            df = DataFrame(
                [(1, 2), (3, 4)],
                index=index,
                columns=["A", "B"],
            )
            result = df.loc[IndexType("foo", "bar")]["A"]
            assert result == 1

        def test_loc_setitem_single_column_mixed(
            self,
            using_infer_string: bool,
        ) -> None:
            df = DataFrame(
                np.random.default_rng(2).standard_normal((5, 3)),
                index=["a", "b", "c", "d", "e"],
                columns=["foo", "bar", "baz"],
            )
            df["str"] = "qux"
            df.loc[df.index[::2], "str"] = np.nan
            expected = Series(
                [np.nan, "qux", np.nan, "qux", np.nan],
                dtype=object if not using_infer_string else "str",
            ).values
            tm.assert_almost_equal(df["str"].values, expected)

        def test_loc_setitem_cast2(self) -> None:
            df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            df.loc[1, "A"] = 4
            expected = DataFrame({"A": [1, 4, 3], "B": [4, 5, 6]})
            tm.assert_frame_equal(df, expected)

    class TestLocCallable:
        def test_frame_loc_getitem_callable(
            self, float_frame: DataFrame
        ) -> None:
            df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})
            res = df.loc[lambda x: x.A > 2]
            tm.assert_frame_equal(res, df.loc[df.A > 2])
            res = df.loc[lambda x: x.B == "b", :]
            tm.assert_frame_equal(res, df.loc[df.B == "b", :])
            res = df.loc[lambda x: x.A > 2, lambda x: "C"]
            tm.assert_series_equal(res, df.loc[df.A > 2, "C"])
            res = df.loc[lambda x: x.A > 2, lambda x: ["A", "B"]]
            tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])
            res = df.loc[lambda x: x.A == 2, lambda x: ["A", "B"]]
            tm.assert_frame_equal(res, df.loc[df.A == 2, ["A", "B"]])
            res = df.loc[lambda x: 1, lambda x: "A"]
            assert res == df.loc[1, "A"]

        def test_frame_loc_getitem_callable_mixture(self, float_frame: DataFrame) -> None:
            df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})
            res = df.loc[lambda x: x.A > 2, ["A", "B"]]
            tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])
            res = df.loc[[2, 3], lambda x: ["A", "B"]]
            tm.assert_frame_equal(res, df.loc[[2, 3], ["A", "B"]])
            res = df.loc[3, lambda x: ["A", "B"]]
            expected = df.loc[3, ["A", "B"]]
            tm.assert_series_equal(res, expected)

        def test_frame_loc_getitem_callable_labels(
            self,
            float_frame: DataFrame,
        ) -> None:
            df = DataFrame(
                {"X": [1, 2, 3, 4], "Y": list("aabb")},
                index=list("ABCD"),
            )
            res = df.loc[lambda x: ["A", "C"]]
            tm.assert_frame_equal(res, df.loc[["A", "C"]])
            res = df.loc[lambda x: ["A", "C"], :]
            tm.assert_frame_equal(res, df.loc[["A", "C"], :])
            res = df.loc[lambda x: ["A", "C"], lambda x: "X"]
            tm.assert_series_equal(res, df.loc[["A", "C"], "X"])
            res = df.loc[lambda x: ["A", "C"], lambda x: ["X"]]
            tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])
            res = df.loc[["A", "C"], lambda x: "X"]
            tm.assert_series_equal(res, df.loc[["A", "C"], "X"])
            res = df.loc[["A", "C"], lambda x: ["X"]]
            tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])
            res = df.loc[lambda x: ["A", "C"], "X"]
            tm.assert_series_equal(res, df.loc[["A", "C"], "X"])
            res = df.loc[lambda x: ["A", "C"], ["X"]]
            tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

        def test_frame_loc_setitem_callable(
            self,
            using_infer_string: bool,
        ) -> None:
            df = DataFrame(
                {"X": [1, 2, 3, 4], "Y": list("aabb")},
                index=list("ABCD"),
            )
            res = df.copy()
            res.loc[lambda x: ["A", "C"]] = -20
            expected = df.copy()
            expected.loc[["A", "C"]] = -20
            tm.assert_frame_equal(res, expected)
            res = df.copy()
            res.loc[lambda x: ["A", "C"], :] = 20
            expected = df.copy()
            expected.loc[["A", "C"], :] = 20
            tm.assert_frame_equal(res, expected)
            res = df.copy()
            res.loc[lambda x: ["A", "C"], lambda x: "X"] = -1
            expected = df.copy()
            expected.loc[["A", "C"], ["X"]] = -1
            tm.assert_frame_equal(res, expected)
            res = df.copy()
            res.loc[lambda x: ["A", "C"], lambda x: ["X"]] = [5, 10]
            expected = df.copy()
            expected.loc[["A", "C"], ["X"]] = [5, 10]
            tm.assert_frame_equal(res, expected)
            ser = Series([0, 1, 2, 3], index=["A", "B", "C", "D"])
            ser.loc["A"] = 5
            ser.loc["C"] = 10
            expected_ser = Series([5, 1, 10, 3], index=["A", "B", "C", "D"])
            tm.assert_series_equal(ser, expected_ser)

    class TestPartialStringSlicing:
        def test_loc_getitem_partial_string_slicing_datetimeindex(self) -> None:
            df = DataFrame(
                {"col1": ["a", "b", "c"], "col2": [1, 2, 3]},
                index=to_datetime(["2020-08-01", "2020-08-02", "2020-08-05"]),
            )
            expected = DataFrame(
                {"col1": ["a", "c"], "col2": [1, 3]},
                index=to_datetime(["2020-08-01", "2020-08-05"]),
            )
            result = df.loc["2020-08"]
            tm.assert_frame_equal(result, expected)

        def test_loc_getitem_partial_string_slicing_with_periodindex(
            self,
        ) -> None:
            pi = pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")
            ser = pi.to_series()
            result = ser.loc[: "2017-12"]
            expected = ser.iloc[:-1]
            tm.assert_series_equal(result, expected)

        def test_loc_getitem_partial_string_slicing_with_timedeltaindex(
            self,
        ) -> None:
            ix = timedelta_range(start="1 day", end="2 days", freq="1h")
            ser = ix.to_series()
            result = ser.loc[: "1 day"]
            expected = ser.iloc[:-1]
            tm.assert_series_equal(result, expected)

        def test_loc_getitem_str_timedeltaindex(self) -> None:
            df = DataFrame({"x": range(3)}, index=to_timedelta(range(3), unit="days"))
            expected = df.iloc[0]
            sliced = df.loc["0 days"]
            tm.assert_series_equal(sliced, expected)

        @pytest.mark.parametrize(
            "start,stop,expected_slice",
            [
                (np.timedelta64(0, "ns"), None, slice(0, 11)),
                (np.timedelta64(1, "D"), np.timedelta64(6, "D"), slice(1, 7)),
                (None, np.timedelta64(4, "D"), slice(0, 5)),
            ],
        )
        def test_loc_getitem_slice_label_td64obj(
            self,
            start: np.timedelta64,
            stop: np.timedelta64,
            expected_slice: slice,
        ) -> None:
            ser = Series(range(11), timedelta_range("0 days", "10 days"))
            result = ser.loc[slice(start, stop)]
            expected = ser.iloc[expected_slice]
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "indexer_end",
            [None, "2020-01-02 23:59:59.999999999"],
        )
        def test_loc_getitem_partial_slice_non_monotonicity(
            self,
            tz_aware_fixture: str,
            indexer_end: Union[None, str],
        ) -> None:
            obj = Series(
                [1] * 5,
                index=DatetimeIndex(
                    [
                        Timestamp("2019-12-30"),
                        Timestamp("2020-01-01"),
                        Timestamp("2019-12-25"),
                        Timestamp("2020-01-02 23:59:59.999999999"),
                        Timestamp("2019-12-19"),
                    ],
                    tz=tz_aware_fixture,
                ),
            )
            expected = Series(
                [1, 1],
                index=DatetimeIndex(
                    [Timestamp("2020-01-01"), Timestamp("2020-01-02 23:59:59.999999999")],
                    tz=tz_aware_fixture,
                ),
            )
            indexer = slice("2020-01-01", indexer_end)
            result = obj.loc[indexer]
            tm.assert_series_equal(result, expected)
            result = obj.loc[["2020-01-01", "2020-01-02 23:59:59.999999999"]]
            tm.assert_series_equal(result, expected)

        def test_loc_slice_disallows_positional(
            self,
        ) -> None:
            dti = date_range("2016-01-01", periods=3)
            df = DataFrame(
                np.random.default_rng(2).random((3, 2)),
                index=dti,
            )
            ser = df[0]
            msg = (
                "cannot do slice indexing on DatetimeIndex with these indexers \\[1\\] of type int"
            )
            for obj in [df, ser]:
                with pytest.raises(TypeError, match=msg):
                    obj.loc[1:3]
                with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
                    obj.loc[1:3] = 1
            with pytest.raises(TypeError, match=msg):
                df.loc[1:3, 1]
            with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
                df.loc[1:3, 1] = 2

        def test_loc_datetimelike_mismatched_dtypes(self) -> None:
            df = DataFrame(
                np.random.default_rng(2).standard_normal((5, 3)),
                columns=["a", "b", "c"],
                index=date_range("2016-01-01", periods=5),
            )
            df = df.iloc[[0, 2, 2, 3, 4]].copy()
            dti = df.index
            tdi = pd.TimedeltaIndex(dti.asi8)
            msg = "None of \\[TimedeltaIndex.* are in the \\[index\\]"
            with pytest.raises(KeyError, match=msg):
                df.loc[tdi]
            with pytest.raises(KeyError, match=msg):
                df["a"].loc[tdi]

        def test_loc_with_period_index_indexer(self) -> None:
            idx = pd.period_range("2002-01", "2003-12", freq="M")
            df = DataFrame(
                np.random.default_rng(2).standard_normal((24, 10)),
                index=idx,
            )
            tm.assert_frame_equal(df, df.loc[idx])
            tm.assert_frame_equal(df, df.loc[list(idx)])
            tm.assert_frame_equal(df, df.loc[list(idx)])
            tm.assert_frame_equal(df.iloc[0:5], df.loc[idx[0:5]])
            tm.assert_frame_equal(df, df.loc[list(idx)])

        def test_loc_setitem_multiindex_timestamp(self) -> None:
            vals = np.random.default_rng(2).standard_normal((8, 6))
            idx = date_range("1/1/2000", periods=8)
            cols = ["A", "B", "C", "D", "E", "F"]
            exp = DataFrame(vals, index=idx, columns=cols)
            exp.loc[exp.index[1], ("A", "B")] = np.nan
            vals[1][0:2] = np.nan
            res = DataFrame(vals, index=idx, columns=cols)
            tm.assert_frame_equal(res, exp)

    class TestLocWithMultiIndex:
        @pytest.mark.parametrize(
            "keys, expected",
            [
                (
                    ["b", "a"],
                    [["b", "b", "a", "a"], [1, 2, 1, 2]],
                ),
                (
                    ["a", "b"],
                    [["a", "a", "b", "b"], [1, 2, 1, 2]],
                ),
                (
                    (["a", "b"], [1, 2]),
                    [["a", "a", "b", "b"], [1, 2, 1, 2]],
                ),
                (
                    (["a", "b"], [2, 1]),
                    [["a", "a", "b", "b"], [2, 1, 2, 1]],
                ),
                (
                    (["b", "a"], [2, 1]),
                    [["b", "b", "a", "a"], [2, 1, 2, 1]],
                ),
                (
                    (["b", "a"], [1, 2]),
                    [["b", "b", "a", "a"], [1, 2, 1, 2]],
                ),
                (
                    (["c", "a"], [2, 1]),
                    [["c", "a", "a"], [1, 2, 1]],
                ),
            ],
        )
        @pytest.mark.parametrize("dim", ["index", "columns"])
        def test_loc_getitem_multilevel_index_order(
            self,
            dim: str,
            keys: Union[List[Any], Tuple[Any, ...]],
            expected: List[List[Any]],
        ) -> None:
            kwargs = {
                dim: [["c", "a", "a", "b", "b"], [1, 1, 2, 1, 2]]
            }
            df = DataFrame(
                np.arange(25).reshape(5, 5),
                **kwargs,
            )
            exp_index = MultiIndex.from_arrays(expected)
            if dim == "index":
                res = df.loc[keys, :]
                tm.assert_index_equal(res.index, exp_index)
            elif dim == "columns":
                res = df.loc[:, keys]
                tm.assert_index_equal(res.columns, exp_index)

        def test_loc_preserve_names(
            self,
            multiindex_year_month_day_dataframe_random_data: DataFrame,
        ) -> None:
            ymd = multiindex_year_month_day_dataframe_random_data
            result = ymd.loc[2000].index.name
            assert result == ymd.index.names[1:]
            result = ymd["A"].loc[2000].index.name
            assert result == ymd.index.names[1:]
            result = ymd.loc[2000, 2].index.name
            assert result == ymd.index.names[2]
            result = ymd["A"].loc[2000, 2].index.name
            assert result == ymd.index.names[2]

        def test_loc_getitem_slice_multiindex_nonunique_len_zero(self) -> None:
            mi = MultiIndex.from_tuples([("foo", "bar"), ("foo", "bar")])
            ser = Series([0, 1], index=mi)
            res = ser.loc[[]]
            expected = ser[:0]
            tm.assert_series_equal(res, expected)
            res2 = ser.loc[ser.iloc[0:0]]
            tm.assert_series_equal(res2, expected)

        def test_loc_getitem_access_none_value_in_multiindex(self) -> None:
            ser = Series([None], MultiIndex.from_arrays([["Level1"], ["Level2"]]))
            result = ser.loc["Level1", "Level2"]
            assert result is None
            midx = MultiIndex.from_product([["Level1"], ["Level2_a", "Level2_b"]])
            ser = Series([None] * len(midx), dtype=object, index=midx)
            result = ser.loc["Level1", "Level2_a"]
            assert result is None
            ser = Series([1] * len(midx), dtype=object, index=midx)
            result = ser.loc["Level1", "Level2_a"]
            assert result == 1

        def test_loc_setitem_multiindex_slice(self) -> None:
            mi = MultiIndex.from_product(
                [list("ab"), list("xy"), [1, 2]],
                names=["x", "y", "z"],
            )
            ser = Series(range(8), index=mi)
            loc_result = ser.loc["a", :, :]
            expected = ser.index.droplevel(0)[:4]
            tm.assert_index_equal(loc_result.index, expected)

        def test_loc_getitem_slice_datetime_objs_with_datetimeindex(
            self,
        ) -> None:
            times = date_range("2000-01-01", freq="1h", periods=5)
            ser = Series(range(5), index=times)
            result = ser.loc["2000-01-01 01:00":"2000-01-01 04:00"]
            expected = ser.iloc[1:5]
            tm.assert_series_equal(result, expected)

        def test_loc_getitem_slice_datetime_objs_with_non_datetimeindex(
            self,
        ) -> None:
            idx = list("ABCDE")
            ser = Series(range(5), index=idx)
            with pytest.raises(KeyError, match="not in index"):
                ser.loc[slice("A", "C")]

        def test_loc_setitem_preserves_nullable_int(self, any_numeric_ea_dtype: np.dtype) -> None:
            ser = Series(
                [0, 1, 2, 3],
                dtype=CategoricalDtype(Index([0, 1, 2, 3], dtype=any_numeric_ea_dtype)),
            )
            df = DataFrame({"data": ser})
            result = DataFrame(index=df.index)
            result.loc[df.index, "data"] = ser
            tm.assert_frame_equal(result, df, check_column_type=False)
            result = DataFrame(index=df.index)
            result.loc[df.index, "data"] = ser._values
            tm.assert_frame_equal(result, df, check_column_type=False)

        def test_loc_setitem_ea_not_full_column(self) -> None:
            df = DataFrame(
                {"a": [1, 1, 1, 1, 1], "b": ["a", "a", "a", "a", "a"]},
                dtype="CategoricalDtype",
            )
            df.loc[1:2, "a"] = Categorical([2, 2], categories=[1, 2])
            expected = DataFrame(
                {"a": [1, 2, 2, 1, 1], "b": ["a", "a", "a", "a", "a"]},
                dtype="object",
            )
            tm.assert_frame_equal(df, expected)

    class TestLocSetitemWithExpansion:
        def test_loc_setitem_with_expansion_large_dataframe(
            self, monkeypatch: pytest.MonkeyPatch
        ) -> None:
            size_cutoff = 50
            with monkeypatch.context():
                monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
                result = DataFrame({"x": range(size_cutoff)}, dtype="int64")
                result.loc[size_cutoff] = size_cutoff
            expected = DataFrame(
                {"x": range(size_cutoff + 1)}, dtype="int64"
            )
            tm.assert_frame_equal(result, expected)

        def test_loc_setitem_empty_series(
            self,
        ) -> None:
            ser = Series(dtype=object)
            ser.loc[1] = 1
            tm.assert_series_equal(ser, Series([1], index=range(1, 2)))
            ser.loc[3] = 3
            tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

        def test_loc_setitem_empty_series_float(
            self,
        ) -> None:
            ser = Series(dtype=object)
            ser.loc[1] = 1.0
            tm.assert_series_equal(ser, Series([1.0], index=range(1, 2)))
            ser.loc[3] = 3.0
            tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

        def test_loc_setitem_empty_series_str_idx(
            self,
        ) -> None:
            ser = Series(dtype=object)
            ser.loc["foo"] = 1
            tm.assert_series_equal(
                ser,
                Series([1], index=Index(["foo"], dtype=object)),
            )
            ser.loc["bar"] = 3
            tm.assert_series_equal(
                ser,
                Series([1, 3], index=Index(["foo", "bar"], dtype=object)),
            )
            ser.loc[3] = 4
            tm.assert_series_equal(
                ser,
                Series([1, 3, 4], index=Index(["foo", "bar", 3], dtype=object)),
            )

        def test_loc_setitem_incremental_with_dst(self) -> None:
            base = datetime(2015, 11, 1, tzinfo=gettz("US/Pacific"))
            idxs = [base + timedelta(seconds=i * 900) for i in range(16)]
            result = Series([0], index=[idxs[0]])
            for ts in idxs:
                result.loc[ts] = 1
            expected = Series([1] * len(idxs), index=idxs)
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "conv",
            [
                lambda x: x,
                lambda x: x.to_datetime64(),
                lambda x: x.to_pydatetime(),
                lambda x: np.datetime64(x),
            ],
            ids=["self", "to_datetime64", "to_pydatetime", "np.datetime64"],
        )
        def test_loc_setitem_datetime_keys_cast(
            self,
            conv: Callable[[Timestamp], Union[Timestamp, np.datetime64, datetime]],
        ) -> None:
            dt1 = Timestamp("20130101 09:00:00")
            dt2 = Timestamp("20130101 10:00:00")
            df = DataFrame(columns=["a"], dtype=np.float64)
            df.loc[conv(dt1), "a"] = 100
            df.loc[conv(dt2), "a"] = 200
            expected = DataFrame(
                {"a": [100.0, 200.0]},
                index=pd.Index([conv(dt1), conv(dt2)], dtype=object),
                columns=pd.Index(["a"], dtype=object),
            )
            tm.assert_frame_equal(df, expected)

        def test_loc_setitem_categorical_column_retains_dtype(
            self,
            using_infer_string: bool,
        ) -> None:
            df = DataFrame(
                {"A": ["a", "b", "c"], "B": ["d", "e", "f"]},
                dtype="category",
            )
            df.loc[:, "A"] = "a"
            expected = DataFrame(
                {"A": ["a", "a", "a"], "B": ["d", "e", "f"]},
                dtype="category",
            )
            tm.assert_frame_equal(df, expected)

        def test_loc_setitem_with_expansion_and_existing_dst(self) -> None:
            start = Timestamp("2017-10-29 00:00:00", tz="Europe/Madrid")
            end = Timestamp("2017-10-29 03:00:00", tz="Europe/Madrid")
            ts = Timestamp("2016-10-10 03:00:00", tz="Europe/Madrid")
            idx = date_range(start, end, inclusive="left", freq="h")
            assert ts not in idx
            result = DataFrame(index=idx, columns=["value"])
            result.loc[ts, "value"] = 12
            expected = DataFrame(
                {"value": [np.nan] * len(idx) + [12]},
                index=idx.append(DatetimeIndex([ts])),
                columns=["value"],
                dtype=object,
            )
            tm.assert_frame_equal(result, expected)

        def test_loc_setitem_with_expansion_preserves_nullable_int(
            self,
            any_numeric_ea_dtype: np.dtype,
        ) -> None:
            ser = Series(
                [0, 1, 2, 3],
                dtype=CategoricalDtype(Index([0, 1, 2, 3], dtype=any_numeric_ea_dtype)),
            )
            df = DataFrame({"data": ser})
            result = DataFrame(index=df.index)
            result.loc[df.index, "data"] = ser
            tm.assert_frame_equal(result, df, check_column_type=False)
            result = DataFrame(index=df.index)
            result.loc[df.index, "data"] = ser._values
            tm.assert_frame_equal(result, df, check_column_type=False)

        def test_loc_setitem_ea_not_full_column(self) -> None:
            df = DataFrame(
                {"a": [1, 1, 1, 1, 1], "b": ["a", "a", "a", "a", "a"]},
                dtype="CategoricalDtype",
            )
            df.loc[1:2, "a"] = Categorical([2, 2], categories=[1, 2])
            expected = DataFrame(
                {"a": [1, 2, 2, 1, 1], "b": ["a", "a", "a", "a", "a"]},
                dtype="object",
            )
            tm.assert_frame_equal(df, expected)


class TestLocCallable:
    def test_frame_loc_getitem_callable(
        self, frame_or_series: Union[DataFrame, Series]
    ) -> None:
        df = DataFrame(
            {"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]}
        )
        res = df.loc[lambda x: x.A > 2]
        tm.assert_frame_equal(res, df.loc[df.A > 2])
        res = df.loc[lambda x: x.B == "b", :]
        tm.assert_frame_equal(res, df.loc[df.B == "b", :])
        res = df.loc[lambda x: x.A > 2, "C"]
        tm.assert_series_equal(res, df.loc[df.A > 2, "C"])
        res = df.loc[lambda x: x.A > 2, ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])
        res = df.loc[lambda x: x.A == 2, ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A == 2, ["A", "B"]])
        res = df.loc[lambda x: 1, "A"]
        assert res == df.loc[1, "A"]

    def test_frame_loc_getitem_callable_mixture(self, frame_or_series: Union[DataFrame, Series]) -> None:
        df = DataFrame(
            {"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]}
        )
        res = df.loc[lambda x: x.A > 2, ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])
        res = df.loc[[2, 3], ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[[2, 3], ["A", "B"]])
        res = df.loc[3, ["A", "B"]]
        expected = df.loc[3, ["A", "B"]]
        tm.assert_series_equal(res, expected)

    def test_frame_loc_getitem_callable_labels(
        self,
        frame_or_series: Union[DataFrame, Series],
    ) -> None:
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": list("aabb")},
            index=list("ABCD"),
        )
        res = df.loc[lambda x: ["A", "C"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"]])
        res = df.loc[lambda x: ["A", "C"], :]
        tm.assert_frame_equal(res, df.loc[["A", "C"], :])
        res = df.loc[lambda x: ["A", "C"], "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])
        res = df.loc[lambda x: ["A", "C"], ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])
        res = df.loc[["A", "C"], lambda x: "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])
        res = df.loc[["A", "C"], lambda x: ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])
        res = df.loc[lambda x: ["A", "C"], "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])
        res = df.loc[lambda x: ["A", "C"], ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

    def test_frame_loc_setitem_callable(
        self,
        using_infer_string: bool,
    ) -> None:
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": list("aabb")},
            index=list("ABCD"),
        )
        res = df.copy()
        res.loc[lambda x: ["A", "C"]] = -20
        expected = df.copy()
        expected.loc[["A", "C"]] = -20
        tm.assert_frame_equal(res, expected)
        res = df.copy()
        res.loc[lambda x: ["A", "C"], :] = 20
        expected = df.copy()
        expected.loc[["A", "C"], :] = 20
        tm.assert_frame_equal(res, expected)
        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: "X"] = -1
        expected = df.copy()
        expected.loc[["A", "C"], ["X"]] = -1
        tm.assert_frame_equal(res, expected)
        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: ["X"]] = [5, 10]
        expected = df.copy()
        expected.loc[["A", "C"], ["X"]] = [5, 10]
        tm.assert_frame_equal(res, expected)
        ser = Series([0, 1, 2, 3], index=["A", "B", "C", "D"])
        ser.loc["A"] = 5
        ser.loc["C"] = 10
        expected_ser = Series([5, 1, 10, 3], index=["A", "B", "C", "D"])
        tm.assert_series_equal(ser, expected_ser)

    def test_frame_loc_setitem_callable_labels(
        self,
        frame_or_series: Union[DataFrame, Series],
    ) -> None:
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": list("aabb")},
            index=list("ABCD"),
        )
        res = df.loc[lambda x: ["A", "C"], "X"] = 10
        tm.assert_series_equal(df.loc[["A", "C"], "X"], Series([10, 10], index=["A", "C"]))
        res = df.copy()
        res.loc[lambda x: ["A", "C"], ["X"]] = [20, 30]
        expected = df.copy()
        expected.loc[["A", "C"], ["X"]] = [20, 30]
        tm.assert_frame_equal(res, expected)


class TestPartialStringSlicing:
    def test_loc_getitem_partial_string_slicing_datetimeindex(self) -> None:
        df = DataFrame(
            {"col1": ["a", "b", "c"], "col2": [1, 2, 3]},
            index=to_datetime(["2020-08-01", "2020-08-02", "2020-08-05"]),
        )
        expected = DataFrame(
            {"col1": ["a", "c"], "col2": [1, 3]},
            index=to_datetime(["2020-08-01", "2020-08-05"]),
        )
        result = df.loc["2020-08"]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_periodindex(
        self,
    ) -> None:
        pi = pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")
        ser = pi.to_series()
        result = ser.loc[: "2017-12"]
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_timedeltaindex(
        self,
    ) -> None:
        ix = timedelta_range(start="1 day", end="2 days", freq="1h")
        ser = ix.to_series()
        result = ser.loc[: "1 day"]
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_str_timedeltaindex(self) -> None:
        df = DataFrame({"x": range(3)}, index=to_timedelta(range(3), unit="days"))
        expected = df.iloc[0]
        sliced = df.loc["0 days"]
        tm.assert_series_equal(sliced, expected)

    @pytest.mark.parametrize(
        "start,stop,expected_slice",
        [
            (np.timedelta64(0, "ns"), None, slice(0, 11)),
            (np.timedelta64(1, "D"), np.timedelta64(6, "D"), slice(1, 7)),
            (None, np.timedelta64(4, "D"), slice(0, 5)),
        ],
    )
    def test_loc_getitem_slice_label_td64obj(
        self,
        start: np.timedelta64,
        stop: np.timedelta64,
        expected_slice: slice,
    ) -> None:
        ser = Series(range(11), timedelta_range("0 days", "10 days"))
        result = ser.loc[slice(start, stop)]
        expected = ser.iloc[expected_slice]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_slice_non_monotonicity(
        self,
        tz_aware_fixture: str,
    ) -> None:
        obj = Series(
            [1] * 5,
            index=DatetimeIndex(
                [
                    Timestamp("2019-12-30"),
                    Timestamp("2020-01-01"),
                    Timestamp("2019-12-25"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                    Timestamp("2019-12-19"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        expected = Series(
            [1, 1],
            index=DatetimeIndex(
                [
                    Timestamp("2020-01-01"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        indexer = slice("2020-01-01", "2020-01-02 23:59:59.999999999")
        result = obj.loc[indexer]
        tm.assert_series_equal(result, expected)
        result = obj.loc[["2020-01-01", "2020-01-02 23:59:59.999999999"]]
        tm.assert_series_equal(result, expected)

    def test_loc_slice_disallows_positional(
        self,
    ) -> None:
        dti = date_range("2016-01-01", periods=3)
        df = DataFrame(
            np.random.default_rng(2).random((3, 2)),
            index=dti,
        )
        ser = df[0]
        msg = (
            "cannot do slice indexing on DatetimeIndex with these indexers \\[1\\] of type int"
        )
        for obj in [df, ser]:
            with pytest.raises(TypeError, match=msg):
                obj.loc[1:3]
            with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
                obj.loc[1:3] = 1
        with pytest.raises(TypeError, match=msg):
            df.loc[1:3, 1]
        with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
            df.loc[1:3, 1] = 2

    def test_loc_datetimelike_mismatched_dtypes(
        self,
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            columns=["a", "b", "c"],
            index=date_range("2016-01-01", periods=5),
        )
        df = df.iloc[[0, 2, 2, 3, 4]].copy()
        dti = df.index
        tdi = pd.TimedeltaIndex(dti.asi8)
        msg = "None of \\[TimedeltaIndex.* are in the \\[index\\]"
        with pytest.raises(KeyError, match=msg):
            df.loc[tdi]
        with pytest.raises(KeyError, match=msg):
            df["a"].loc[tdi]

    def test_loc_with_period_index_indexer(self) -> None:
        idx = pd.period_range("2002-01", "2003-12", freq="M")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((24, 10)),
            index=idx,
        )
        tm.assert_frame_equal(df, df.loc[idx])
        tm.assert_frame_equal(df, df.loc[list(idx)])
        tm.assert_frame_equal(df, df.loc[list(idx)])
        tm.assert_frame_equal(df.iloc[0:5], df.loc[idx[0:5]])
        tm.assert_frame_equal(df, df.loc[list(idx)])

    def test_loc_setitem_multiindex_timestamp(self) -> None:
        vals = np.random.default_rng(2).standard_normal((8, 6))
        idx = date_range("1/1/2000", periods=8)
        cols = ["A", "B", "C", "D", "E", "F"]
        exp = DataFrame(vals, index=idx, columns=cols)
        exp.loc[exp.index[1], ("A", "B")] = np.nan
        vals[1][0:2] = np.nan
        res = DataFrame(vals, index=idx, columns=cols)
        tm.assert_frame_equal(res, exp)

    def test_loc_getitem_multiindex_tuple_level(self) -> None:
        lev1 = ["a", "b", "c"]
        lev2 = [(0, 1), (1, 0)]
        lev3 = [0, 1]
        cols = MultiIndex.from_product([lev1, lev2, lev3], names=["x", "y", "z"])
        df = DataFrame(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            index=[1, 2],
            columns=cols,
        )
        result = df.loc["a", ("a", "y", "z")]
        expected = df.loc["a", ("a", "y", "z")]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_identity_slice(self) -> None:
        original_df = DataFrame({"a": [1, 2, 3]})
        sliced_df = original_df.loc[:]
        assert sliced_df is not original_df
        assert original_df[:] is not original_df
        assert original_df.loc[:, :] is not original_df
        assert np.shares_memory(original_df["a"]._values, sliced_df["a"]._values)
        original_df.loc[:, "a"] = [4, 4, 4]
        assert (sliced_df["a"] == [1, 2, 3]).all()
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        assert df[0] is not df.loc[:, 0]
        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.loc[:]
        assert sliced_series is not original_series
        assert original_series[:] is not original_series
        original_series[:3] = [7, 8, 9]
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_loc_copy_vs_view(
        self,
        request: Any,
    ) -> None:
        x = DataFrame(zip(range(3), range(3)), columns=["a", "b"])
        y = x.copy()
        q = y.loc[:, "a"]
        q += 2
        tm.assert_frame_equal(x, y)
        z = x.copy()
        q = z.loc[x.index, "a"]
        q += 2
        tm.assert_frame_equal(x, z)

    def test_loc_uint64(self) -> None:
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])
        result = ser.loc[umax - 1]
        expected = ser.iloc[0]
        assert result == expected
        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        tm.assert_series_equal(result, expected)
        result = ser.loc[[umax - 1, umax]]
        tm.assert_series_equal(result, ser)

    def test_loc_uint64_disallow_negative(self) -> None:
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])
        with pytest.raises(KeyError, match="-1"):
            ser.loc[-1]
        with pytest.raises(KeyError, match="-1"):
            ser.loc[[-1]]

    def test_loc_setitem_empty_append_expands_rows(self) -> None:
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )
        df = DataFrame(columns=["x", "y"])
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(
        self,
    ) -> None:
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )
        df = DataFrame(columns=["x", "y"])
        df["x"] = df["x"].astype(np.int64)
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(
        self,
    ) -> None:
        expected = DataFrame({"x": [1.0], "y": [np.nan]})
        df = DataFrame(columns=["x", "y"], dtype=float)
        df.loc[0, "x"] = expected.loc[0, "x"]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(
        self,
    ) -> None:
        data = [1, 2]
        df = DataFrame(columns=["x", "y"])
        df.index = df.index.astype(np.int64)
        msg = "None of .*Index.* are in the \\[index\\]"
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], "x"] = data
        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, "x"] = data

    def test_loc_setitem_frame_with_reindex(self) -> None:
        df = DataFrame(index=[3, 5, 4], columns=["A"], dtype=float)
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({"A": ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self) -> None:
        df = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        expected = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(
        self,
    ) -> None:
        df = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[slice(3, 0, -1), "A"] = np.array([1, 2, 3], dtype="int64")
        expected = DataFrame(
            {"A": [3.0, 2.0, 1.0], "B": "string"},
            index=[1, 2, 3],
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=list("abcd"),
            columns=list("ABCD"),
        )
        result = df.iloc[0, 0]
        df.loc["a", "A"] = 1
        result = df.loc["a", "A"]
        assert result == 1
        result = df.iloc[0, 0]
        assert result == 1
        df.loc[:, "B":"D"] = 0
        expected = df.loc[:, "B":"D"]
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(
        self,
    ) -> None:
        df = DataFrame({"A": [1, 2, 3], "B": np.nan})
        df.loc[df.B > df.A, "B"] = df.A
        expected = DataFrame({"A": [1, 2, 3], "B": np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(
        self,
    ) -> None:
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": ["a", "b"]})
        result = df.loc[0, [1, 2]]
        expected = Series(
            [1, 3],
            index=Index([1, 2], dtype=object),
            dtype=object,
            name=0,
        )
        tm.assert_series_equal(result, expected)
        expected = DataFrame({"1": [5, 2], "2": [6, 4], "a": ["a", "b"]})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(
        self,
    ) -> None:
        df = DataFrame(
            {"A": ["foo", "bar", "baz"], "B": Series(range(3), dtype=np.int64)}
        )
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame(
            {"A": ["bar", "baz", "baz"], "B": Series([1, 2, 2], dtype=np.int64)}
        )
        tm.assert_frame_equal(df, expected)
        df = DataFrame(
            {"date": date_range("2000-01-01", "2000-01-5"), "val": Series(range(5), dtype=np.int64)}
        )
        expected = DataFrame(
            {
                "date": [
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000103"),
                ],
                "val": Series([0, 1, 0, 1, 2], dtype=np.int64),
            }
        )
        expected["date"] = expected["date"].astype("M8[ns]")
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "indexer, expected",
        [
            (
                0,
                [20, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            (
                slice(4, 8),
                [0, 1, 2, 3, 20, 20, 20, 20, 8, 9],
            ),
            (
                [3, 5],
                [0, 1, 2, 20, 4, 20, 6, 7, 8, 9],
            ),
        ],
    )
    def test_loc_setitem_listlike_with_timedelta64index(
        self,
        indexer: Union[int, slice, List[int]],
        expected: List[Any],
    ) -> None:
        tdi = to_timedelta(range(10), unit="s")
        df = DataFrame({"x": range(10)}, dtype="int64", index=tdi)
        df.loc[df.index[indexer], "x"] = 20
        expected = DataFrame(
            {"x": expected},
            index=tdi,
            columns=["x"],
            dtype="int64",
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_column_retains_dtype(
        self,
        using_infer_string: bool,
    ) -> None:
        df = DataFrame({"A": ["a", "b", "c"], "B": ["d", "e", "f"]}, dtype="object")
        df.loc[:, "A"] = Categorical(["x", "y", "z"], categories=["a", "b", "c", "x", "y", "z"])
        expected = DataFrame(
            {"A": Categorical(["x", "y", "z"], categories=["a", "b", "c", "x", "y", "z"]), "B": ["d", "e", "f"]},
            dtype="object",
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_cast3(
        self,
    ) -> None:
        df = DataFrame({"one": np.arange(6, dtype=np.int8)})
        df.loc[1, "one"] = 6
        assert df.dtypes.one == np.dtype(np.int8)
        df.one = np.int8(7)
        assert df.dtypes.one == np.dtype(np.int8)

    def test_loc_setitem_range_key(
        self,
        frame_or_series: Union[DataFrame, Series],
    ) -> None:
        obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])
        values = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]
        obj.loc[range(3)] = values
        expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        tm.assert_equal(obj, expected)

    def test_loc_setitem_numpy_frame_categorical_value(
        self,
    ) -> None:
        df = DataFrame(
            {"a": [1, 1, 1, 1, 1], "b": list("aaaaa")}, dtype="object"
        )
        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[:, "a"] = Categorical(["b", "b", "c", "c", "c"], categories=["a", "b", "c"])


class TestLocWithMultiIndex:
    @pytest.mark.parametrize(
        "keys, expected",
        [
            (
                ["b", "a"],
                [["b", "b", "a", "a"], [1, 2, 1, 2]],
            ),
            (
                ["a", "b"],
                [["a", "a", "b", "b"], [1, 2, 1, 2]],
            ),
            (
                (["a", "b"], [1, 2]),
                [["a", "a", "b", "b"], [1, 2, 1, 2]],
            ),
            (
                (["a", "b"], [2, 1]),
                [["a", "a", "b", "b"], [2, 1, 2, 1]],
            ),
            (
                (["b", "a"], [2, 1]),
                [["b", "b", "a", "a"], [2, 1, 2, 1]],
            ),
            (
                (["b", "a"], [1, 2]),
                [["b", "b", "a", "a"], [1, 2, 1, 2]],
            ),
            (
                (["c", "a"], [2, 1]),
                [["c", "a", "a"], [1, 2, 1]],
            ),
        ],
    )
    @pytest.mark.parametrize("dim", ["index", "columns"])
    def test_loc_getitem_multilevel_index_order(
        self,
        dim: str,
        keys: Union[List[Any], Tuple[Any, ...]],
        expected: List[List[Any]],
    ) -> None:
        kwargs = {
            dim: [["c", "a", "a", "b", "b"], [1, 1, 2, 1, 2]]
        }
        df = DataFrame(
            np.arange(25).reshape(5, 5),
            **kwargs,
        )
        exp_index = MultiIndex.from_arrays(expected)
        if dim == "index":
            res = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == "columns":
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

    def test_loc_preserve_names(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
    ) -> None:
        ymd = multiindex_year_month_day_dataframe_random_data
        result = ymd.loc[2000].index.name
        assert result == ymd.index.names[1:]
        result = ymd["A"].loc[2000].index.name
        assert result == ymd.index.names[1:]
        result = ymd.loc[2000, 2].index.name
        assert result == ymd.index.names[2]
        result = ymd["A"].loc[2000, 2].index.name
        assert result == ymd.index.names[2]

    def test_loc_getitem_slice_multiindex_nonunique_len_zero(
        self,
    ) -> None:
        mi = MultiIndex.from_tuples([("foo", "bar"), ("foo", "bar")])
        ser = Series([0, 1], index=mi)
        res = ser.loc[[]]
        expected = ser[:0]
        tm.assert_series_equal(res, expected)
        res2 = ser.loc[ser.iloc[0:0]]
        tm.assert_series_equal(res2, expected)

    def test_loc_getitem_access_none_value_in_multiindex(
        self,
    ) -> None:
        ser = Series([None], MultiIndex.from_arrays([["Level1"], ["Level2"]]))
        result = ser.loc["Level1", "Level2"]
        assert result is None
        midx = MultiIndex.from_product([["Level1"], ["Level2_a", "Level2_b"]])
        ser = Series([None] * len(midx), dtype=object, index=midx)
        result = ser.loc["Level1", "Level2_a"]
        assert result is None
        ser = Series([1] * len(midx), dtype=object, index=midx)
        result = ser.loc["Level1", "Level2_a"]
        assert result == 1

    def test_loc_setitem_multiindex_slice(
        self,
    ) -> None:
        mi = MultiIndex.from_product(
            [list("ab"), list("xy"), [1, 2]],
            names=["x", "y", "z"],
        )
        ser = Series(range(8), index=mi)
        loc_result = ser.loc["a", :, :]
        expected = ser.index.droplevel(0)[:4]
        tm.assert_index_equal(loc_result.index, expected)

    def test_loc_getitem_slice_datetime_objs_with_datetimeindex(
        self,
    ) -> None:
        times = date_range("2000-01-01", freq="1h", periods=5)
        ser = Series(range(5), index=times)
        result = ser.loc["2000-01-01 01:00":"2000-01-01 04:00"]
        expected = ser.iloc[1:5]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_datetime_objs_with_non_datetimeindex(
        self,
    ) -> None:
        idx = list("ABCDE")
        ser = Series(range(5), index=idx)
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[slice("A", "C")]


class TestLocBooleanLabelsAndSlices:
    @pytest.mark.parametrize(
        "bool_value",
        [True, False],
    )
    def test_loc_bool_incompatible_index_raises(
        self,
        index: Index,
        frame_or_series: Union[DataFrame, Series],
        bool_value: bool,
    ) -> None:
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        if index.inferred_type != "boolean":
            obj = frame_or_series(index=index, dtype="object")
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

    @pytest.mark.parametrize(
        "bool_value",
        [True, False],
    )
    def test_loc_bool_should_not_raise(
        self,
        frame_or_series: Union[DataFrame, Series],
        bool_value: bool,
    ) -> None:
        obj = frame_or_series(index=Index([True, False], dtype="boolean"), dtype="object")
        obj.loc[bool_value]

    def test_loc_bool_slice_raises(
        self,
        index: Index,
        frame_or_series: Union[DataFrame, Series],
    ) -> None:
        message = "slice\\(True, False, None\\): boolean values can not be used in a slice"
        obj = frame_or_series(index=index, dtype="object")
        with pytest.raises(TypeError, match=message):
            obj.loc[True:False]


class TestLocBooleanMask:
    def test_loc_setitem_bool_mask_timedeltaindex(
        self,
    ) -> None:
        df = DataFrame({"x": range(10)})
        df.index = to_timedelta(range(10), unit="s")
        conditions = [df["x"] > 3, df["x"] == 3, df["x"] < 3]
        expected_data = [
            [0, 1, 2, 3, 10, 10, 10, 10, 10, 10],
            [0, 1, 2, 10, 4, 5, 6, 7, 8, 9],
            [10, 10, 10, 3, 4, 5, 6, 7, 8, 9],
        ]
        for cond, data in zip(conditions, expected_data):
            result = df.copy()
            result.loc[cond, "x"] = 10
            expected = DataFrame(
                data,
                index=to_timedelta(range(10), unit="s"),
                columns=["x"],
                dtype="int64",
            )
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "tz",
        [None, "UTC"],
    )
    def test_loc_setitem_mask_with_datetimeindex_tz(
        self,
        tz: Union[None, str],
    ) -> None:
        mask = np.array([True, False, True, False])
        idx = date_range("20010101", periods=4, tz=tz)
        df = DataFrame({"a": np.arange(4)}, index=idx).astype("object")
        result = df.copy()
        result.loc[mask, :] = df.loc[mask, :]
        tm.assert_frame_equal(result, df)
        result = df.copy()
        result.loc[mask] = df.loc[mask]
        tm.assert_frame_equal(result, df)

    def test_loc_setitem_mask_and_label_with_datetimeindex(self) -> None:
        df = DataFrame(
            {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6]},
            index=date_range("2000-01-01", freq="H", periods=4),
        )
        expected = df.copy()
        result = df.copy()
        mask = df["a"] > 3
        result.loc[mask, "B"] = 0
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_mask_td64_series_value(self) -> None:
        df = DataFrame(
            {"col": [1, 2, 3]},
            dtype="object",
        )
        df.loc[[True, False, True], "col"] = Series(
            [np.timedelta64(10, "m")],
            index=[None],
            dtype="object",
        )
        expected = DataFrame({"col": [np.timedelta64(10, "m"), 2, 3]}, dtype="object")
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_boolean_and_column(self) -> None:
        df = DataFrame({"a": [1, 2, 3, 4], "b": [3, 4, 5, 6]})
        mask = df["a"] > df["a"].median()
        result = df.copy()
        result.loc[mask, "b"] = 0
        expected = DataFrame({"a": [1, 2, 3, 4], "b": [3, 4, 0, 0]})
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_ndframe_values_alignment(self) -> None:
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], "a"] = DataFrame({"a": [10, 11, 12]}, index=[2, 1, 0])
        expected = DataFrame({"a": [1, np.nan, 10], "b": [4, 5, 6]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], "a"] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], "a"] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)

    def test_loc_indexing_zerodim_np_array(
        self,
    ) -> None:
        df = DataFrame([[1, 2], [3, 4]])
        result = df.loc[np.array(0)]
        s = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_loc_setitem_ctor_series_astype_float(self) -> None:
        df = DataFrame({"A": [1, None, None, 1], "B": [1, 1, 2, 2]}, dtype="object")
        df.loc[df["A"].isna(), "A"] = [3, 4]
        expected = DataFrame(
            {"A": [1, 3, 4, 1], "B": [1, 1, 2, 2]}, dtype="object"
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_index_accessor_preserves_properties(self) -> None:
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.index.name = "idx"
        result = df.iloc[[0, 1]].loc[:, "A"]
        assert result.name == "A"
        assert result.index.name == "idx"

    def test_loc_exhaustive_with_non_unique_index(self) -> None:
        idx = Index([1, 1, 2, 2, 3, 3], dtype=np.int64)
        df = DataFrame({"A": np.arange(len(idx))}, index=idx)
        res = df.loc[[1, 2], "A"]
        expected = Series([0, 1, 2, 3], index=[1, 1, 2, 2], name="A")
        tm.assert_series_equal(res, expected)


class TestLocListlike:
    @pytest.mark.parametrize(
        "columns, column_key, expected_columns",
        [
            ([2011, 2012, 2013], [2011, 2012], [0, 1]),
            ([2011, 2012, "All"], [2011, 2012], [0, 1]),
            ([2011, 2012, "All"], [2011, "All"], [0, 2]),
        ],
    )
    def test_loc_getitem_label_list_integer_labels(
        self,
        columns: List[Any],
        column_key: List[Any],
        expected_columns: List[int],
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=columns,
            index=list("ABCDEF"),
        )
        expected = df.iloc[:, expected_columns]
        result = df.loc[["A", "B", "C"], column_key]
        tm.assert_frame_equal(result, expected, check_column_type=True)

    def test_loc_setitem_float_intindex(
        self,
    ) -> None:
        rand_data = np.random.default_rng(2).standard_normal((8, 4))
        result = DataFrame(rand_data)
        result.loc[:, 0.5] = np.nan
        expected_data = np.hstack(
            (rand_data, np.array([np.nan] * 8).reshape(8, 1))
        )
        expected = DataFrame(
            expected_data,
            columns=[0.0, 1.0, 2.0, 3.0, 0.5],
        )
        tm.assert_frame_equal(result, expected)
        result = DataFrame(rand_data)
        result.loc[:, 0.5] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_loc_axis_1_slice(
        self,
    ) -> None:
        cols = [(yr, m) for yr in [2014, 2015] for m in [7, 8, 9, 10]]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 8)),
            index=list("ABCDE"),
            columns=MultiIndex.from_tuples(cols),
        )
        result = df.loc(axis=1)[(2014, 9):(2015, 8)]
        expected = DataFrame(
            np.ones((5, 4)),
            index=list("ABCDE"),
            columns=MultiIndex.from_tuples([(2014, 9), (2014, 10), (2015, 7), (2015, 8)]),
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_missing_columns(
        self,
    ) -> None:
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "B"])
        df.loc[["A", "B", "C"], ["A", "B", "C"]] = [7, 8, 9, 10, 11, 12, 13]
        expected = DataFrame(
            {"A": [7, 3, 5], "B": [8, 4, 6], "C": [9, 10, 11]},
            index=["A", "B", "C"],
        )
        tm.assert_frame_equal(df, expected)


def test_loc_setitem_ctor_series_astype_float() -> None:
    df = DataFrame({"A": [1, None, None, 1], "B": [1, 1, 2, 2]}, dtype="object")
    df.loc[df["A"].isna(), "A"] = [3, 4]
    expected = DataFrame(
        {"A": [1, 3, 4, 1], "B": [1, 1, 2, 2]},
        dtype="object",
    )
    tm.assert_frame_equal(df, expected)


def test_loc_setitem_ctor_series_astype_integer_dtype() -> None:
    df = DataFrame({"A": [1, None, None, 1], "B": [1, 1, 2, 2]}, dtype="object")
    df.loc[df["A"].isna(), "A"] = [3, 4]
    expected = DataFrame(
        {"A": [1, 3, 4, 1], "B": [1, 1, 2, 2]},
        dtype="object",
    )
    tm.assert_frame_equal(df, expected)


def test_loc_setitem_constructor_series_astype_Float64() -> None:
    df = DataFrame({"A": [np.nan, 1, 2], "B": [3, 4, 5]}, dtype="object")
    df.loc[0, "A"] = 0.0
    expected = DataFrame({"A": [0.0, 1.0, 2.0], "B": [3, 4, 5]}, dtype="float64")
    tm.assert_frame_equal(df, expected)


def test_loc_setitem_with_tz_localized_index():
    tz = "Europe/Madrid"
    idx = date_range("2020-01-01", periods=3, freq="D", tz=tz)
    df = DataFrame({"val": [1, 2, 3]}, index=idx)
    df.loc["2020-01-02", "val"] = 20
    expected = DataFrame({"val": [1, 20, 3]}, index=idx)
    tm.assert_frame_equal(df, expected)


def test_loc_setitem_with_period_index():
    pi = pd.period_range("2020-01", periods=3, freq="M")
    ser = Series([1, 2, 3], index=pi)
    ser.loc[pi[1]] = 20
    expected = Series([1, 20, 3], index=pi)
    tm.assert_series_equal(ser, expected)


def test_loc_boolean_indexing_after_setitem_binary_op():
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.loc[:, "A"] = df["A"] + 1
    mask = df["A"] > 2
    df.loc[mask, "B"] = 0
    expected = DataFrame({"A": [2, 3, 4], "B": [4, 0, 0]})
    tm.assert_frame_equal(df, expected)


def test_loc_setitem_with_boolean_series() -> None:
    ser = Series([1, 2, 3, 4, 5])
    mask = ser > 3
    ser.loc[mask] = 10
    expected = Series([1, 2, 3, 10, 10])
    tm.assert_series_equal(ser, expected)


def test_loc_setitem_with_all_true_boolean_series() -> None:
    ser = Series([1, 2, 3])
    ser.loc[[True, True, True]] = 10
    expected = Series([10, 10, 10])
    tm.assert_series_equal(ser, expected)


def test_loc_setitem_with_all_false_boolean_series() -> None:
    ser = Series([1, 2, 3])
    ser.loc[[False, False, False]] = 10
    expected = Series([1, 2, 3])
    tm.assert_series_equal(ser, expected)


def test_loc_setitem_with_partial_true_boolean_series() -> None:
    ser = Series([1, 2, 3, 4])
    ser.loc[[True, False, True, False]] = 10
    expected = Series([10, 2, 10, 4])
    tm.assert_series_equal(ser, expected)


def test_loc_setitem_with_boolean_series_and_new_values():
    ser = Series([1, 2, 3])
    mask = ser > 1
    ser.loc[mask] = [20, 30]
    expected = Series([1, 20, 30])
    tm.assert_series_equal(ser, expected)


def test_loc_setitem_with_nan_in_boolean_mask():
    ser = Series([1, 2, 3, 4])
    mask = [True, False, np.nan, True]
    ser.loc[mask] = [10, 20]
    expected = Series([10, 2, 3, 10])
    tm.assert_series_equal(ser, expected)
