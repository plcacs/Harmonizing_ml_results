import datetime
from datetime import timedelta
from io import StringIO
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import IS64
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    NA,
    DataFrame,
    DatetimeIndex,
    Index,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    read_json,
)
import pandas._testing as tm

from pandas.io.json import ujson_dumps


def test_literal_json_raises() -> None:
    # PR 53409
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""

    msg = r".* does not exist"

    with pytest.raises(FileNotFoundError, match=msg):
        read_json(jsonl, lines=False)

    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=True)

    with pytest.raises(FileNotFoundError, match=msg):
        read_json(
            '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n',
            lines=False,
        )

    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=False)


def assert_json_roundtrip_equal(result: DataFrame, expected: DataFrame, orient: str) -> None:
    if orient in ("records", "values"):
        expected = expected.reset_index(drop=True)
    if orient == "values":
        expected.columns = range(len(expected.columns))
    tm.assert_frame_equal(result, expected)


class TestPandasContainer:
    @pytest.fixture
    def datetime_series(self) -> Series:
        # Same as usual datetime_series, but with index freq set to None,
        #  since that doesn't round-trip, see GH#33711
        ser = Series(
            1.1 * np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        ser.index = ser.index._with_freq(None)
        return ser

    @pytest.fixture
    def datetime_frame(self) -> DataFrame:
        # Same as usual datetime_frame, but with index freq set to None,
        #  since that doesn't round-trip, see GH#33711
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=30, freq="B"),
        )
        df.index = df.index._with_freq(None)
        return df

    def test_frame_double_encoded_labels(self, orient: str) -> None:
        df = DataFrame(
            [["a", "b"], ["c", "d"]],
            index=['index " 1', "index / 2"],
            columns=["a \\ b", "y / z"],
        )

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("orient", ["split", "records", "values"])
    def test_frame_non_unique_index(self, orient: str) -> None:
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("orient", ["index", "columns"])
    def test_frame_non_unique_index_raises(self, orient: str) -> None:
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        msg = f"DataFrame index must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    @pytest.mark.parametrize("orient", ["split", "values"])
    @pytest.mark.parametrize(
        "data",
        [
            [["a", "b"], ["c", "d"]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[1, 2.5], [3, 4.5]],
            [[Timestamp("20130101"), 3.5], [Timestamp("20130102"), 4.5]],
        ],
    )
    def test_frame_non_unique_columns(self, orient: str, data: List[List[Any]], request: Any) -> None:
        if isinstance(data[0][0], Timestamp) and orient == "split":
            mark = pytest.mark.xfail(
                reason="GH#55827 non-nanosecond dt64 fails to round-trip"
            )
            request.applymarker(mark)

        df = DataFrame(data, index=[1, 2], columns=["x", "x"])

        expected_warning = None
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        if df.iloc[:, 0].dtype == "datetime64[s]":
            expected_warning = FutureWarning

        with tm.assert_produces_warning(expected_warning, match=msg):
            result = read_json(
                StringIO(df.to_json(orient=orient)), orient=orient, convert_dates=["x"]
            )
        if orient == "values":
            expected = DataFrame(data)
            if expected.iloc[:, 0].dtype == "datetime64[s]":
                # orient == "values" by default will write Timestamp objects out
                # in milliseconds; these are internally stored in nanosecond,
                # so divide to get where we need
                # TODO: a to_epoch method would also solve; see GH 14772
                expected.isetitem(0, expected.iloc[:, 0].astype(np.int64) // 1000000)
        elif orient == "split":
            expected = df
            expected.columns = ["x", "x.1"]

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("orient", ["index", "columns", "records"])
    def test_frame_non_unique_columns_raises(self, orient: str) -> None:
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 2], columns=["x", "x"])

        msg = f"DataFrame columns must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    def test_frame_default_orient(self, float_frame: DataFrame) -> None:
        assert float_frame.to_json() == float_frame.to_json(orient="columns")

    @pytest.mark.parametrize("dtype", [False, float])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_simple(self, orient: str, convert_axes: bool, dtype: Any, float_frame: DataFrame) -> None:
        data = StringIO(float_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = float_frame

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("dtype", [False, np.int64])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_intframe(self, orient: str, convert_axes: bool, dtype: Any, int_frame: DataFrame) -> None:
        data = StringIO(int_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
        expected = int_frame
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("dtype", [None, np.float64, int, "U3"])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_str_axes(self, orient: str, convert_axes: bool, dtype: Any) -> None:
        df = DataFrame(
            np.zeros((200, 4)),
            columns=[str(i) for i in range(4)],
            index=[str(i) for i in range(200)],
            dtype=dtype,
        )

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = df.copy()
        if not dtype:
            expected = expected.astype(np.int64)

        # index columns, and records orients cannot fully preserve the string
        # dtype for axes as the index and column labels are used as keys in
        # JSON objects. JSON keys are by definition strings, so there's no way
        # to disambiguate whether those keys actually were strings or numeric
        # beforehand and numeric wins out.
        if convert_axes and (orient in ("index", "columns")):
            expected.columns = expected.columns.astype(np.int64)
            expected.index = expected.index.astype(np.int64)
        elif orient == "records" and convert_axes:
            expected.columns = expected.columns.astype(np.int64)
        elif convert_axes and orient == "split":
            expected.columns = expected.columns.astype(np.int64)

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_categorical(
        self, request: Any, orient: str, convert_axes: bool, using_infer_string: bool
    ) -> None:
        # TODO: create a better frame to test with and improve coverage
        if orient in ("index", "columns"):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"Can't have duplicate index values for orient '{orient}')"
                )
            )

        data = {
            c: np.random.default_rng(i).standard_normal(30)
            for i, c in enumerate(list("ABCD"))
        }
        cat = ["bah"] * 5 + ["bar"] * 5 + ["baz"] * 5 + ["foo"] * 15
        data["E"] = list(reversed(cat))
        data["sort"] = np.arange(30, dtype="int64")
        categorical_frame = DataFrame(data, index=pd.CategoricalIndex(cat, name="E"))
        data = StringIO(categorical_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        expected = categorical_frame.copy()
        expected.index = expected.index.astype(
            str if not using_infer_string else "str"
        )  # Categorical not preserved
        expected.index.name = None  # index names aren't preserved in JSON
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_empty(self, orient: str, convert_axes: bool) -> None:
        empty_frame = DataFrame()
        data = StringIO(empty_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        if orient == "split":
            idx = Index([], dtype=(float if convert_axes else object))
            expected = DataFrame(index=idx, columns=idx)
        elif orient in ["index", "columns"]:
            expected = DataFrame()
        else:
            expected = empty_frame.copy()

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_timestamp(self, orient: str, convert_axes: bool, datetime_frame: DataFrame) -> None:
        # TODO: improve coverage with date_format parameter
        data = StringIO(datetime_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        expected = datetime_frame.copy()

        if not convert_axes:  # one off for ts handling
            # DTI gets converted to epoch values
            idx = expected.index.view(np.int64) // 1000000
            if orient != "split":  # TODO: handle consistently across orients
                idx = idx.astype(str)

            expected.index = idx

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_mixed(self, orient: str, convert_axes: bool) -> None:
        index = Index(["a", "b", "c", "d", "e"])
        values = {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
            "D": [True, False, True, False, True],
        }

        df = DataFrame(data=values, index=index)

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        expected = df.copy()
        expected = expected.assign(**expected.select_dtypes("number").astype(np.int64))

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.xfail(
        reason="#50456 Column multiindex is stored and loaded differently",
        raises=AssertionError,
    )
    @pytest.mark.parametrize(
        "columns",
        [
            [["2022", "2022"], ["JAN", "FEB"]],
            [["2022", "2023"], ["JAN", "JAN"]],
            [["2022", "2022"], ["JAN", "JAN"]],
        ],
    )
    def test_roundtrip_multiindex(self, columns: List[List[str]]) -> None:
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_arrays(columns),
        )
        data = StringIO(df.to_json(orient="split"))
        result = read_json(data, orient="split")
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "data,msg,orient",
        [
            ('{"key":b:a:d}', "Expected object or value", "columns"),
            # too few indices
            (
                '{"columns":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "|".join(
                    [
                        r"Length of values \(3\) does not match length of index \(2\)",
                    ]
                ),
                "split",
            ),
            # too many columns
            (
                '{"columns":["A","B","C"],'
                '"index":["1","2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "3 columns passed, passed data had 2 columns",
                "split",
            ),
            # bad key
            (
                '{"badkey":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                r"unexpected key\(s\): badkey",
                "split",
            ),
        ],
    )
    def test_frame_from_json_bad_data_raises(self, data: str, msg: str, orient: str) -> None:
        with pytest.raises(ValueError, match=msg):
            read_json(StringIO(data), orient=orient)

    @pytest.mark.parametrize("dtype", [True, False])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_frame_from_json_missing_data(self, orient: str, convert_axes: bool, dtype: Any) -> None:
        num_df = DataFrame([[1, 2], [4, 5, 6]])

        result = read_json(
            StringIO(num_df.to_json(orient=orient)),
            orient=orient,
            convert_axes=convert_axes,
            dtype=dtype,
        )
        assert np.isnan(result.iloc[0, 2])

        obj_df = DataFrame([["1", "2"], ["4", "5", "6"]])
        result = read_json(
            StringIO(obj_df.to_json(orient=orient)),
            orient=orient,
            convert_axes=convert_axes,
            dtype=dtype,
        )
        assert np.isnan(result.iloc[0, 2])

    @pytest.mark.parametrize("dtype", [True, False])
    def test_frame_read_json_dtype_missing_value(self, dtype: Any) -> None:
        # GH28501 Parse missing values using read_json with dtype=False
        # to NaN instead of None
        result = read_json(StringIO("[null]"), dtype=dtype)
        expected = DataFrame([np.nan], dtype=object if not dtype else None)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("inf", [np.inf, -np.inf])
    @pytest.mark.parametrize("dtype", [