from datetime import (
    date,
    datetime,
    timedelta,
)
from decimal import Decimal
from functools import partial
from io import BytesIO
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pytest

from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    date_range,
    option_context,
    period_range,
)
import pandas._testing as tm
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series
from pandas.io.excel import (
    ExcelFile,
    ExcelWriter,
    _OpenpyxlWriter,
    _XlsxWriter,
    register_writer,
)
from pandas.io.excel._util import _writers


def get_exp_unit(path: str) -> str:
    if path.endswith(".ods"):
        return "s"
    return "us"


@pytest.fixture
def frame(float_frame: DataFrame) -> DataFrame:
    """
    Returns the first ten items in fixture "float_frame".
    """
    return float_frame[:10]


@pytest.fixture(params=[True, False, "columns"])
def merge_cells(request: pytest.FixtureRequest) -> Union[bool, str]:
    return request.param


@pytest.fixture
def tmp_excel(ext: str, tmp_path: Any) -> str:
    """
    Fixture to open file for use in each test case.
    """
    tmp = tmp_path / f"{uuid.uuid4()}{ext}"
    tmp.touch()
    return str(tmp)


@pytest.fixture
def set_engine(engine: str, ext: str) -> None:
    """
    Fixture to set engine for use in each test case.

    Rather than requiring `engine=...` to be provided explicitly as an
    argument in each test, this fixture sets a global option to dictate
    which engine should be used to write Excel files. After executing
    the test it rolls back said change to the global option.
    """
    option_name = f"io.excel.{ext.strip('.')}.writer"
    with option_context(option_name, engine):
        yield


@pytest.mark.parametrize(
    "ext",
    [
        pytest.param(".xlsx", marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")]),
        pytest.param(".xlsm", marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")]),
        pytest.param(
            ".xlsx", marks=[td.skip_if_no("xlsxwriter"), td.skip_if_no("xlrd")]
        ),
        pytest.param(".ods", marks=td.skip_if_no("odf")),
    ],
)
class TestRoundTrip:
    @pytest.mark.parametrize(
        "header,expected",
        [(None, [np.nan] * 4), (0, {"Unnamed: 0": [np.nan] * 3})],
    )
    def test_read_one_empty_col_no_header(
        self, tmp_excel: str, header: Optional[int], expected: Union[List[float], Dict[str, List[float]]]
    ) -> None:
        # xref gh-12292
        filename = "no_header"
        df = DataFrame([["", 1, 100], ["", 2, 200], ["", 3, 300], ["", 4, 400]])

        df.to_excel(tmp_excel, sheet_name=filename, index=False, header=False)
        result = pd.read_excel(
            tmp_excel, sheet_name=filename, usecols=[0], header=header
        )
        expected_df = DataFrame(expected)
        tm.assert_frame_equal(result, expected_df)

    @pytest.mark.parametrize(
        "header,expected_extra",
        [(None, [0]), (0, [])],
    )
    def test_read_one_empty_col_with_header(
        self, tmp_excel: str, header: Optional[int], expected_extra: List[int]
    ) -> None:
        filename = "with_header"
        df = DataFrame([["", 1, 100], ["", 2, 200], ["", 3, 300], ["", 4, 400]])

        df.to_excel(tmp_excel, sheet_name="with_header", index=False, header=True)
        result = pd.read_excel(
            tmp_excel, sheet_name=filename, usecols=[0], header=header
        )
        expected = DataFrame(expected_extra + [np.nan] * 4)
        tm.assert_frame_equal(result, expected)

    def test_set_column_names_in_parameter(self, tmp_excel: str) -> None:
        # GH 12870 : pass down column names associated with
        # keyword argument names
        refdf = DataFrame([[1, "foo"], [2, "bar"], [3, "baz"]], columns=["a", "b"])

        with ExcelWriter(tmp_excel) as writer:
            refdf.to_excel(writer, sheet_name="Data_no_head", header=False, index=False)
            refdf.to_excel(writer, sheet_name="Data_with_head", index=False)

        refdf.columns = ["A", "B"]

        with ExcelFile(tmp_excel) as reader:
            xlsdf_no_head = pd.read_excel(
                reader, sheet_name="Data_no_head", header=None, names=["A", "B"]
            )
            xlsdf_with_head = pd.read_excel(
                reader,
                sheet_name="Data_with_head",
                index_col=None,
                names=["A", "B"],
            )

        tm.assert_frame_equal(xlsdf_no_head, refdf)
        tm.assert_frame_equal(xlsdf_with_head, refdf)

    def test_creating_and_reading_multiple_sheets(self, tmp_excel: str) -> None:
        # see gh-9450
        #
        # Test reading multiple sheets, from a runtime
        # created Excel file with multiple sheets.
        def tdf(col_sheet_name: str) -> DataFrame:
            d, i = [11, 22, 33], [1, 2, 3]
            return DataFrame(d, i, columns=[col_sheet_name])

        sheets = ["AAA", "BBB", "CCC"]

        dfs = [tdf(s) for s in sheets]
        dfs_dict = dict(zip(sheets, dfs))

        with ExcelWriter(tmp_excel) as ew:
            for sheetname, df in dfs_dict.items():
                df.to_excel(ew, sheet_name=sheetname)

        dfs_returned = pd.read_excel(tmp_excel, sheet_name=sheets, index_col=0)

        for s in sheets:
            tm.assert_frame_equal(dfs_dict[s], dfs_returned[s])

    def test_read_excel_multiindex_empty_level(self, tmp_excel: str) -> None:
        # see gh-12453
        df = DataFrame(
            {
                ("One", "x"): {0: 1},
                ("Two", "X"): {0: 3},
                ("Two", "Y"): {0: 7},
                ("Zero", ""): {0: 0},
            }
        )

        expected = DataFrame(
            {
                ("One", "x"): {0: 1},
                ("Two", "X"): {0: 3},
                ("Two", "Y"): {0: 7},
                ("Zero", "Unnamed: 4_level_1"): {0: 0},
            }
        )

        df.to_excel(tmp_excel)
        actual = pd.read_excel(tmp_excel, header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)

        df = DataFrame(
            {
                ("Beg", ""): {0: 0},
                ("Middle", "x"): {0: 1},
                ("Tail", "X"): {0: 3},
                ("Tail", "Y"): {0: 7},
            }
        )

        expected = DataFrame(
            {
                ("Beg", "Unnamed: 1_level_1"): {0: 0},
                ("Middle", "x"): {0: 1},
                ("Tail", "X"): {0: 3},
                ("Tail", "Y"): {0: 7},
            }
        )

        df.to_excel(tmp_excel)
        actual = pd.read_excel(tmp_excel, header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("c_idx_names", ["a", None])
    @pytest.mark.parametrize("r_idx_names", ["b", None])
    @pytest.mark.parametrize("c_idx_levels", [1, 3])
    @pytest.mark.parametrize("r_idx_levels", [1, 3])
    def test_excel_multindex_roundtrip(
        self,
        tmp_excel: str,
        c_idx_names: Optional[str],
        r_idx_names: Optional[str],
        c_idx_levels: int,
        r_idx_levels: int,
    ) -> None:
        # see gh-4679
        # Empty name case current read in as
        # unnamed levels, not Nones.
        check_names = bool(r_idx_names) or r_idx_levels <= 1

        if c_idx_levels == 1:
            columns = Index(list("abcde"))
        else:
            columns = MultiIndex.from_arrays(
                [range(5) for _ in range(c_idx_levels)],
                names=[f"{c_idx_names}-{i}" for i in range(c_idx_levels)],
            )
        if r_idx_levels == 1:
            index = Index(list("ghijk"))
        else:
            index = MultiIndex.from_arrays(
                [range(5) for _ in range(r_idx_levels)],
                names=[f"{r_idx_names}-{i}" for i in range(r_idx_levels)],
            )
        df = DataFrame(
            1.1 * np.ones((5, 5)),
            columns=columns,
            index=index,
        )
        df.to_excel(tmp_excel)

        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        tm.assert_frame_equal(df, act, check_names=check_names)

        df.iloc[0, :] = np.nan
        df.to_excel(tmp_excel)

        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        tm.assert_frame_equal(df, act, check_names=check_names)

        df.iloc[-1, :] = np.nan
        df.to_excel(tmp_excel)
        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        tm.assert_frame_equal(df, act, check_names=check_names)

    def test_read_excel_parse_dates(self, tmp_excel: str) -> None:
        # see gh-11544, gh-12051
        df = DataFrame(
            {"col": [1, 2, 3], "date_strings": date_range("2012-01-01", periods=3)}
        )
        df2 = df.copy()
        df2["date_strings"] = df2["date_strings"].dt.strftime("%m/%d/%Y")

        df2.to_excel(tmp_excel)

        res = pd.read_excel(tmp_excel, index_col=0)
        tm.assert_frame_equal(df2, res)

        res = pd.read_excel(tmp_excel, parse_dates=["date_strings"], index_col=0)
        expected = df[:]
        expected["date_strings"] = expected["date_strings"].astype("M8[s]")
        tm.assert_frame_equal(res, expected)

        res = pd.read_excel(
            tmp_excel, parse_dates=["date_strings"], date_format="%m/%d/%Y", index_col=0
        )
        expected["date_strings"] = expected["date_strings"].astype("M8[s]")
        tm.assert_frame_equal(expected, res)

    def test_multiindex_interval_datetimes(self, tmp_excel: str) -> None:
        # GH 30986
        midx = MultiIndex.from_arrays(
            [
                range(4),
                pd.interval_range(
                    start=pd.Timestamp("2020-01-01"), periods=4, freq="6ME"
                ),
            ]
        )
        df = DataFrame(range(4), index=midx)
        df.to_excel(tmp_excel)
        result = pd.read_excel(tmp_excel, index_col=[0, 1])
        expected = DataFrame(
            range(4),
            MultiIndex.from_arrays(
                [
                    range(4),
                    [
                        "(2020-01-31 00:00:00, 2020-07-31 00:00:00]",
                        "(2020-07-31 00:00:00, 2021-01-31 00:00:00]",
                        "(2021-01-31 00:00:00, 2021-07-31 00:00:00]",
                        "(2021-07-31 00:00:00, 2022-01-31 00:00:00]",
                    ],
                ]
            ),
            columns=Index([0]),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("merge_cells", [True, False, "columns"])
    def test_excel_round_trip_with_periodindex(
        self, tmp_excel: str, merge_cells: Union[bool, str]
    ) -> None:
        # GH#60099
        df = DataFrame(
            {"A": [1, 2]},
            index=MultiIndex.from_arrays(
                [
                    period_range(start="2006-10-06", end="2006-10-07", freq="D"),
                    ["X", "Y"],
                ],
                names=["date", "category"],
            ),
        )
        df.to_excel(tmp_excel, merge_cells=merge_cells)
        result = pd.read_excel(tmp_excel, index_col=[0, 1])
        expected = DataFrame(
            {"A": [1, 2]},
            MultiIndex.from_arrays(
                [
                    [
                        pd.to_datetime("2006-10-06 00:00:00"),
                        pd.to_datetime("2006-10-07 00:00:00"),
                    ],
                    ["X", "Y"],
                ],
                names=["date", "category"],
            ),
        )
        time_format = (
            "datetime64[s]" if tmp_excel.endswith(".ods") else "datetime64[us]"
        )
        expected.index = expected.index.set_levels(
            expected.index.levels[0].astype(time_format), level=0
        )

        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "engine,ext",
    [
        pytest.param(
            "openpyxl",
            ".xlsx",
            marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")],
        ),
        pytest.param(
            "openpyxl",
            ".xlsm",
            marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")],
        ),
        pytest.param(
            "xlsxwriter",
            ".xlsx",
            marks=[td.skip_if_no("xlsxwriter"), td.skip_if_no("xlrd")],
        ),
        pytest.param("odf", ".ods", marks=td.skip_if_no("odf")),
    ],
)
@pytest.mark.usefixtures("set_engine")
class TestExcelWriter:
    def test_excel_sheet_size(self, tmp_excel: str) -> None:
        # GH 26080
        breaking_row_count = 2**20 + 1
        breaking_col_count = 2**14 + 1
        # purposely using two arrays to prevent memory issues while testing
        row_arr = np.zeros(shape=(breaking_row_count, 1))
        col_arr = np.zeros(shape=(1, breaking_col_count))
        row_df = DataFrame(row_arr)
        col_df = DataFrame(col_arr)

        msg = "sheet is too large"
        with pytest.raises(ValueError, match=msg):
            row_df.to_excel(tmp_excel)

        with pytest.raises(ValueError, match=msg):
            col_df.to_excel(tmp_excel)

    def test_excel_sheet_by_name_raise(self, tmp_excel: str) -> None:
        gt = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=Index(list(range(10))),
        )
        gt.to_excel(tmp_excel)

        with ExcelFile(tmp_excel) as xl:
            df = pd.read_excel(xl, sheet_name=0, index_col=0)

        tm.assert_frame_equal(gt, df)

        msg = "Worksheet named '0' not found"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(xl, "0")

    def test_excel_writer_context_manager(
        self, frame: DataFrame, tmp_excel: str
    ) -> None:
        with ExcelWriter(tmp_excel) as writer:
            frame.to_excel(writer, sheet_name="Data1")
            frame2 = frame.copy()
            frame2.columns = frame.columns[::-1]
            frame2.to_excel(writer, sheet_name="Data2")

        with ExcelFile(tmp_excel) as reader:
            found_df = pd.read_excel(reader, sheet_name="Data1", index_col=0)
            found_df2 = pd.read_excel(reader, sheet_name="Data2", index_col=0)

            tm.assert_frame_equal(found_df, frame)
            tm.assert_frame_equal(found_df2, frame2)

    def test_roundtrip(self, frame: DataFrame, tmp_excel: str) -> None:
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan

        frame.to_excel(tmp_excel, sheet_name="test1")
        frame.to_excel(tmp_ex