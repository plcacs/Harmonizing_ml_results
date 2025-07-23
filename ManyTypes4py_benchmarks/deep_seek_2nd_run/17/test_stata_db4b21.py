import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import itertools
import os
import string
import struct
import tarfile
import zipfile
from typing import Any, Dict, List, Optional, Union, cast, IO, BinaryIO, Iterator, Tuple

import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import DataFrame, Series
from pandas.io.parsers import read_csv
from pandas.io.stata import (
    CategoricalConversionWarning,
    InvalidColumnName,
    PossiblePrecisionLoss,
    StataMissingValue,
    StataReader,
    StataWriter,
    StataWriterUTF8,
    ValueLabelTypeMismatch,
    read_stata,
)

@pytest.fixture
def mixed_frame() -> DataFrame:
    return DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1.0, 3.0, 27.0, 81.0],
            "c": ["Atlanta", "Birmingham", "Cincinnati", "Detroit"],
        }
    )

@pytest.fixture
def parsed_114(datapath: Any) -> DataFrame:
    dta14_114 = datapath("io", "data", "stata", "stata5_114.dta")
    parsed_114 = read_stata(dta14_114, convert_dates=True)
    parsed_114.index.name = "index"
    return parsed_114

class TestStata:
    def read_dta(self, file: Union[str, BinaryIO]) -> DataFrame:
        return read_stata(file, convert_dates=True)

    def read_csv(self, file: str) -> DataFrame:
        return read_csv(file, parse_dates=True)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_read_empty_dta(self, version: Optional[int], temp_file: str) -> None:
        empty_ds = DataFrame(columns=["unit"])
        path = temp_file
        empty_ds.to_stata(path, write_index=False, version=version)
        empty_ds2 = read_stata(path)
        tm.assert_frame_equal(empty_ds, empty_ds2)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_read_empty_dta_with_dtypes(
        self, version: Optional[int], temp_file: str
    ) -> None:
        empty_df_typed = DataFrame(
            {
                "i8": np.array([0], dtype=np.int8),
                "i16": np.array([0], dtype=np.int16),
                "i32": np.array([0], dtype=np.int32),
                "i64": np.array([0], dtype=np.int64),
                "u8": np.array([0], dtype=np.uint8),
                "u16": np.array([0], dtype=np.uint16),
                "u32": np.array([0], dtype=np.uint32),
                "u64": np.array([0], dtype=np.uint64),
                "f32": np.array([0], dtype=np.float32),
                "f64": np.array([0], dtype=np.float64),
            }
        )
        path = temp_file
        empty_df_typed.to_stata(path, write_index=False, version=version)
        empty_reread = read_stata(path)
        expected = empty_df_typed
        expected["u8"] = expected["u8"].astype(np.int8)
        expected["u16"] = expected["u16"].astype(np.int16)
        expected["u32"] = expected["u32"].astype(np.int32)
        expected["u64"] = expected["u64"].astype(np.int32)
        expected["i64"] = expected["i64"].astype(np.int32)
        tm.assert_frame_equal(expected, empty_reread)
        tm.assert_series_equal(expected.dtypes, empty_reread.dtypes)

    @pytest.mark.parametrize("version", [114, 117, 118, 119, None])
    def test_read_index_col_none(self, version: Optional[int], temp_file: str) -> None:
        df = DataFrame({"a": range(5), "b": ["b1", "b2", "b3", "b4", "b5"]})
        path = temp_file
        df.to_stata(path, write_index=False, version=version)
        read_df = read_stata(path)
        assert isinstance(read_df.index, pd.RangeIndex)
        expected = df
        expected["a"] = expected["a"].astype(np.int32)
        tm.assert_frame_equal(read_df, expected, check_index_type=True)

    @pytest.mark.parametrize(
        "version", [102, 103, 104, 105, 108, 110, 111, 113, 114, 115, 117, 118, 119]
    )
    def test_read_dta1(self, version: int, datapath: Any) -> None:
        file = datapath("io", "data", "stata", f"stata1_{version}.dta")
        parsed = self.read_dta(file)
        expected = DataFrame(
            [(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=[
                "float_miss",
                "double_miss",
                "byte_miss",
                "int_miss",
                "long_miss",
            ],
        )
        expected["float_miss"] = expected["float_miss"].astype(np.float32)
        if version <= 108:
            expected = expected.rename(
                columns={
                    "float_miss": "f_miss",
                    "double_miss": "d_miss",
                    "byte_miss": "b_miss",
                    "int_miss": "i_miss",
                    "long_miss": "l_miss",
                }
            )
        tm.assert_frame_equal(parsed, expected)

    def test_read_dta2(self, datapath: Any) -> None:
        expected = DataFrame.from_records(
            [
                (
                    datetime(2006, 11, 19, 23, 13, 20),
                    1479596223000,
                    datetime(2010, 1, 20),
                    datetime(2010, 1, 8),
                    datetime(2010, 1, 1),
                    datetime(1974, 7, 1),
                    datetime(2010, 1, 1),
                    datetime(2010, 1, 1),
                ),
                (
                    datetime(1959, 12, 31, 20, 3, 20),
                    -1479590,
                    datetime(1953, 10, 2),
                    datetime(1948, 6, 10),
                    datetime(1955, 1, 1),
                    datetime(1955, 7, 1),
                    datetime(1955, 1, 1),
                    datetime(2, 1, 1),
                ),
                (pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT),
            ],
            columns=[
                "datetime_c",
                "datetime_big_c",
                "date",
                "weekly_date",
                "monthly_date",
                "quarterly_date",
                "half_yearly_date",
                "yearly_date",
            ],
        )
        expected["datetime_c"] = expected["datetime_c"].astype("M8[ms]")
        expected["date"] = expected["date"].astype("M8[s]")
        expected["weekly_date"] = expected["weekly_date"].astype("M8[s]")
        expected["monthly_date"] = expected["monthly_date"].astype("M8[s]")
        expected["quarterly_date"] = expected["quarterly_date"].astype("M8[s]")
        expected["half_yearly_date"] = expected["half_yearly_date"].astype("M8[s]")
        expected["yearly_date"] = expected["yearly_date"].astype("M8[s]")
        path1 = datapath("io", "data", "stata", "stata2_114.dta")
        path2 = datapath("io", "data", "stata", "stata2_115.dta")
        path3 = datapath("io", "data", "stata", "stata2_117.dta")
        msg = "Leaving in Stata Internal Format"
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_114 = self.read_dta(path1)
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_115 = self.read_dta(path2)
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_117 = self.read_dta(path3)
        tm.assert_frame_equal(parsed_114, expected)
        tm.assert_frame_equal(parsed_115, expected)
        tm.assert_frame_equal(parsed_117, expected)

    @pytest.mark.parametrize(
        "file", ["stata3_113", "stata3_114", "stata3_115", "stata3_117"]
    )
    def test_read_dta3(self, file: str, datapath: Any) -> None:
        file = datapath("io", "data", "stata", f"{file}.dta")
        parsed = self.read_dta(file)
        expected = self.read_csv(datapath("io", "data", "stata", "stata3.csv"))
        expected = expected.astype(np.float32)
        expected["year"] = expected["year"].astype(np.int16)
        expected["quarter"] = expected["quarter"].astype(np.int8)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize("version", [110, 111, 113, 114, 115, 117])
    def test_read_dta4(self, version: int, datapath: Any) -> None:
        file = datapath("io", "data", "stata", f"stata4_{version}.dta")
        parsed = self.read_dta(file)
        expected = DataFrame.from_records(
            [
                ["one", "ten", "one", "one", "one"],
                ["two", "nine", "two", "two", "two"],
                ["three", "eight", "three", "three", "three"],
                ["four", "seven", 4, "four", "four"],
                ["five", "six", 5, np.nan, "five"],
                ["six", "five", 6, np.nan, "six"],
                ["seven", "four", 7, np.nan, "seven"],
                ["eight", "three", 8, np.nan, "eight"],
                ["nine", "two", 9, np.nan, "nine"],
                ["ten", "one", "ten", np.nan, "ten"],
            ],
            columns=[
                "fully_labeled",
                "fully_labeled2",
                "incompletely_labeled",
                "labeled_with_missings",
                "float_labelled",
            ],
        )
        for col in expected:
            orig = expected[col].copy()
            categories = np.asarray(expected["fully_labeled"][orig.notna()])
            if col == "incompletely_labeled":
                categories = orig
            cat = orig.astype("category")._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize("version", [102, 103, 104, 105, 108])
    def test_readold_dta4(self, version: int, datapath: Any) -> None:
        file = datapath("io", "data", "stata", f"stata4_{version}.dta")
        parsed = self.read_dta(file)
        expected = DataFrame.from_records(
            [
                ["one", "ten", "one", "one", "one"],
                ["two", "nine", "two", "two", "two"],
                ["three", "eight", "three", "three", "three"],
                ["four", "seven", 4, "four", "four"],
                ["five", "six", 5, np.nan, "five"],
                ["six", "five", 6, np.nan, "six"],
                ["seven", "four", 7, np.nan, "seven"],
                ["eight", "three", 8, np.nan, "eight"],
                ["nine", "two", 9, np.nan, "nine"],
                ["ten", "one", "ten", np.nan, "ten"],
            ],
            columns=["fulllab", "fulllab2", "incmplab", "misslab", "floatlab"],
        )
        for col in expected:
            orig = expected[col].copy()
            categories = np.asarray(expected["fulllab"][orig.notna()])
            if col == "incmplab":
                categories = orig
            cat = orig.astype("category")._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize(
        "file",
        [
            "stata12_117",
            "stata12_be_117",
            "stata12_118",
            "stata12_be_118",
            "stata12_119",
            "stata12_be_119",
        ],
    )
    def test_read_dta_strl(self, file: str, datapath: Any) -> None:
        parsed = self.read_dta(datapath("io", "data", "stata", f"{file}.dta"))
        expected = DataFrame.from_records(
            [[1, "abc", "abcdefghi"], [3, "cba", "qwertywertyqwerty"], [93, "", "strl"]],
            columns=["x", "y", "z"],
        )
        tm.assert_frame_equal(parsed, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "file", ["stata14_118", "stata14_be_118", "stata14_119", "stata14_be_119"]
    )
    def test_read_dta118_119(self, file: str, datapath: Any) -> None:
        parsed_118 = self.read_dta(datapath("io", "data", "stata", f"{file}.dta"))
        parsed_118["Bytes"] = parsed_118["Bytes"].astype("O")
        expected = DataFrame.from_records(
            [
                ["Cat", "Bogota", "Bogotá", 1, 1.0, "option b Ünicode", 1.0],
                ["Dog", "Boston", "Uzunköprü", np.nan, np.nan, np.nan, np.nan],
                ["Plane", "Rome", "Tromsø", 0, 0.0, "option a", 0.0],
                ["Potato", "Tokyo", "Elâzığ", -4, 4.0, 4, 4],
                ["", "", "", 0, 0.3332999, "option a", 1 / 3.0],
            ],
            columns=[
                "Things",
                "Cities",
                "Unicode_Cities_Strl",
                "Ints",
                "Floats",
                "Bytes",
                "Longs",
            ],
        )
        expected["Floats"] = expected["Floats"].astype(np.float32)
        for col in parsed_118.columns:
            tm.assert_almost_equal(parsed_118[col], expected[col])
        with StataReader(datapath("io", "data", "stata", f"{file}.dta")) as rdr:
            vl = rdr.variable_labels()
            vl_expected = {
                "Unicode_Cities_Strl": "Here are some strls with Ünicode chars",
                "Longs": "long data",
                "Things": "Here are some things",
                "Bytes": "byte data",
                "Ints": "int data",
                "Cities": "Here are some cities",
                "Floats": "float data",
            }
            tm.assert_dict_equal(vl, vl_expected)
            assert rdr.data_label == "This is a  Ünicode data label"

    def test_read_write_dta5(self, temp_file: str) -> None:
        original = DataFrame(
            [(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=["float_miss", "double_miss", "byte_miss", "int_miss", "long_miss"],
        )
        original.index.name = "index"
        path = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        expected = original
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index("index"), expected)

    def test_write_dta6(self, datapath: Any, temp_file: str) -> None:
        original = self.read_csv(datapath("io", "data", "stata", "stata3.csv"))
        original.index.name = "index"
        original.index = original.index.astype(np.int32)
        original["year"] = original["year"].astype(np.int32)
        original["quarter"] = original["quarter"].astype(np.int32)
        path = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(
            written_and_read_again.set_index("index"), original, check_index_type=False
        )

    @pytest.mark.parametrize("version", [114, 117, 118, 119,