import io
import os
import sys
from zipfile import ZipFile
from typing import Any, Dict, List, Optional, Union, IO

from _csv import Error
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    compat,
)
import pandas._testing as tm


class TestToCSV:
    def test_to_csv_with_single_column(self) -> None:
        df1 = DataFrame([None, 1])
        expected1 = """\
""
1.0
"""
        with tm.ensure_clean("test.csv") as path:
            df1.to_csv(path, header=None, index=None)
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected1

        df2 = DataFrame([1, None])
        expected2 = """\
1.0
""
"""
        with tm.ensure_clean("test.csv") as path:
            df2.to_csv(path, header=None, index=None)
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected2

    def test_to_csv_default_encoding(self) -> None:
        df = DataFrame({"col": ["AAAAA", "ÄÄÄÄÄ", "ßßßßß", "聞聞聞聞聞"]})

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path)
            tm.assert_frame_equal(pd.read_csv(path, index_col=0), df)

    def test_to_csv_quotechar(self) -> None:
        df = DataFrame({"col": [1, 2]})
        expected = """\
"","col"
"0","1"
"1","2"
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=1)
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        expected = """\
$$,$col$
$0$,$1$
$1$,$2$
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=1, quotechar="$")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        with tm.ensure_clean("test.csv") as path:
            with pytest.raises(TypeError, match="quotechar"):
                df.to_csv(path, quoting=1, quotechar=None)

    def test_to_csv_doublequote(self) -> None:
        df = DataFrame({"col": ['a"a', '"bb"']})
        expected = '''\
"","col"
"0","a""a"
"1","""bb"""
'''

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=1, doublequote=True)
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        with tm.ensure_clean("test.csv") as path:
            with pytest.raises(Error, match="escapechar"):
                df.to_csv(path, doublequote=False)

    def test_to_csv_escapechar(self) -> None:
        df = DataFrame({"col": ['a"a', '"bb"']})
        expected = """\
"","col"
"0","a\\"a"
"1","\\"bb\\""
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=1, doublequote=False, escapechar="\\")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        df = DataFrame({"col": ["a,a", ",bb,"]})
        expected = """\
,col
0,a\\,a
1,\\,bb\\,
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=3, escapechar="\\")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

    def test_csv_to_string(self) -> None:
        df = DataFrame({"col": [1, 2]})
        expected_rows = [",col", "0,1", "1,2"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected

    def test_to_csv_decimal(self) -> None:
        df = DataFrame({"col1": [1], "col2": ["a"], "col3": [10.1]})

        expected_rows = [",col1,col2,col3", "0,1,a,10.1"]
        expected_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,1"]
        expected_european_excel = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(decimal=",", sep=";") == expected_european_excel

        expected_rows = [",col1,col2,col3", "0,1,a,10.10"]
        expected_float_format_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(float_format="%.2f") == expected_float_format_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,10"]
        expected_float_format = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (
            df.to_csv(decimal=",", sep=";", float_format="%.2f")
            == expected_float_format
        )

        df = DataFrame({"a": [0, 1.1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0^0,2^2,1", "1^1,3^3,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(index=False, decimal="^") == expected

        assert df.set_index("a").to_csv(decimal="^") == expected
        assert df.set_index(["a", "b"]).to_csv(decimal="^") == expected

    def test_to_csv_float_format(self) -> None:
        df = DataFrame({"a": [0, 1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0,2.20,1", "1,3.30,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.set_index("a").to_csv(float_format="%.2f") == expected
        assert df.set_index(["a", "b"]).to_csv(float_format="%.2f") == expected

    def test_to_csv_na_rep(self) -> None:
        df = DataFrame({"a": [0, np.nan], "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "0.0,0,2", "_,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        df = DataFrame({"a": np.nan, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "_,0,2", "_,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        df = DataFrame({"a": 0, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "0,0,2", "0,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        csv = pd.Series(["a", pd.NA, "c"]).to_csv(na_rep="ZZZZZ")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        assert expected == csv

    def test_to_csv_na_rep_nullable_string(self, nullable_string_dtype: str) -> None:
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        csv = pd.Series(["a", pd.NA, "c"], dtype=nullable_string_dtype).to_csv(
            na_rep="ZZZZZ"
        )
        assert expected == csv

    def test_to_csv_date_format(self) -> None:
        df_sec = DataFrame({"A": pd.date_range("20130101", periods=5, freq="s")})
        df_day = DataFrame({"A": pd.date_range("20130101", periods=5, freq="D")})

        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-01 00:00:01",
            "2,2013-01-01 00:00:02",
            "3,2013-01-01 00:00:03",
            "4,2013-01-01 00:00:04",
        ]
        expected_default_sec = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_sec.to_csv() == expected_default_sec

        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-02 00:00:00",
            "2,2013-01-03 00:00:00",
            "3,2013-01-04 00:00:00",
            "4,2013-01-05 00:00:00",
        ]
        expected_ymdhms_day = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_day.to_csv(date_format="%Y-%m-%d %H:%M:%S") == expected_ymdhms_day

        expected_rows = [
            ",A",
            "0,2013-01-01",
            "1,2013-01-01",
            "2,2013-01-01",
            "3,2013-01-01",
            "4,2013-01-01",
        ]
        expected_ymd_sec = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_sec.to_csv(date_format="%Y-%m-%d") == expected_ymd_sec

        expected_rows = [
            ",A",
            "0,2013-01-01",
            "1,2013-01-02",
            "2,2013-01-03",
            "3,2013-01-04",
            "4,2013-01-05",
        ]
        expected_default_day = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_day.to_csv() == expected_default_day
        assert df_day.to_csv(date_format="%Y-%m-%d") == expected_default_day

        df_sec["B"] = 0
        df_sec["C"] = 1

        expected_rows = ["A,B,C", "2013-01-01,0,1.0"]
        expected_ymd_sec = tm.convert_rows_list_to_csv_str(expected_rows)

        df_sec_grouped = df_sec.groupby([pd.Grouper(key="A", freq="1h"), "B"])
        assert df_sec_grouped.mean().to_csv(date_format="%Y-%m-%d") == expected_ymd_sec

    def test_to_csv_different_datetime_formats(self) -> None:
        df = DataFrame(
            {
                "date": pd.to_datetime("1970-01-01"),
                "datetime": pd.date_range("1970-01-01", periods=2, freq="h"),
            }
        )
        expected_rows = [
            "date,datetime",
            "1970-01-01,1970-01-01 00:00:00",
            "1970-01-01,1970-01-01 01:00:00",
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(index=False) == expected

    def test_to_csv_date_format_in_categorical(self) -> None:
        ser = pd.Series(pd.to_datetime(["2021-03-27", pd.NaT], format="%Y-%m-%d"))
        ser = ser.astype("category")
        expected = tm.convert_rows_list_to_csv_str(["0", "2021-03-27", '""'])
        assert ser.to_csv(index=False) == expected

        ser = pd.Series(
            pd.date_range(
                start="2021-03-27", freq="D", periods=1, tz="Europe/Berlin"
            ).append(pd.DatetimeIndex([pd.NaT]))
        )
        ser = ser.astype("category")
        assert ser.to_csv(index=False, date_format="%Y-%m-%d") == expected

    def test_to_csv_float_ea_float_format(self) -> None:
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        df["a"] = df["a"].astype("Float64")
        result = df.to_csv(index=False, float_format="%.5f")
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.10000,c", "2.02000,c", ",c", "6.00001,c"]
        )
        assert result == expected

    def test_to_csv_float_ea_no_float_format(self) -> None:
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        df["a"] = df["a"].astype("Float64")
        result = df.to_csv(index=False)
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.1,c", "2.02,c", ",c", "6.000006,c"]
        )
        assert result == expected

    def test_to_csv_multi_index(self) -> None:
        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([[1], [2]]))

        exp_rows = [",1", ",2", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

        df = DataFrame(
            [1],
            columns=pd.MultiIndex.from_arrays([[1], [2]]),
            index=pd.MultiIndex.from_arrays([[1], [2]]),
        )

        exp_rows = [",,1", ",,2", "1,2,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([["foo"], ["bar"]]))

        exp_rows = [",foo", ",bar", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["foo", "bar", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

    @pytest.mark.parametrize(
        "ind,expected",
        [
            (
                pd.MultiIndex(levels=[[1.0]], codes=[[0]], names=["x"]),
                "x,data\n1.0,1\n",
            ),
            (
                pd.MultiIndex(
                    levels=[[1.0], [2.0]], codes=[[0], [0]], names=["x", "y"]
                ),
                "x,y,data\n1.0,2.0,1\n",
            ),
        ],
    )
    def test_to_csv_single_level_multi_index(
        self, ind: pd.MultiIndex, expected: str, frame_or_series: Any
    ) -> None:
        obj = frame_or_series(pd.Series([1], ind, name="data"))

        result = obj.to_csv(lineterminator="\n", header=True)
        assert result == expected

    def test_to_csv_string_array_ascii(self) -> None:
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        df = DataFrame(str_array)
        expected_ascii = """\
,names
0,"['foo', 'bar']"
1,"['baz', 'qux']"
"""
        with tm.ensure_clean("str_test.csv") as path:
            df.to_csv(path, encoding="ascii")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected_ascii

    def test_to_csv_string_array_utf8(self) -> None:
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        df = DataFrame(str_array)
        expected_utf8 = """\
,names
0,"['foo', 'bar']"
1,"['baz', 'qux']"
"""
        with tm.ensure_clean("unicode_test.csv") as path:
            df.to_csv(path, encoding="utf-8")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected_utf8

    def test_to_csv_string_with_lf(self) -> None:
        data = {"int": [1,