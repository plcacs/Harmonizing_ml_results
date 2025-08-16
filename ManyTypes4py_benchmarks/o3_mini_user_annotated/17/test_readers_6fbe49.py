from __future__ import annotations

from datetime import (
    datetime,
    time,
)
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
import uuid
from zipfile import BadZipFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    read_csv,
)
import pandas._testing as tm


read_ext_params: List[str] = [".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"]
engine_params: List[pytest.param] = [
    pytest.param(
        "xlrd",
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param(
        "openpyxl",
        marks=[
            td.skip_if_no("openpyxl"),
        ],
    ),
    pytest.param(
        None,
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param("pyxlsb", marks=td.skip_if_no("pyxlsb")),
    pytest.param("odf", marks=td.skip_if_no("odf")),
    pytest.param("calamine", marks=td.skip_if_no("python_calamine")),
]


def _is_valid_engine_ext_pair(engine: Any, read_ext: str) -> bool:
    """
    Filter out invalid (engine, ext) pairs instead of skipping, as that
    produces 500+ pytest.skips.
    """
    engine_value: Any = engine.values[0]
    if engine_value == "openpyxl" and read_ext == ".xls":
        return False
    if engine_value == "odf" and read_ext != ".ods":
        return False
    if read_ext == ".ods" and engine_value not in {"odf", "calamine"}:
        return False
    if engine_value == "pyxlsb" and read_ext != ".xlsb":
        return False
    if read_ext == ".xlsb" and engine_value not in {"pyxlsb", "calamine"}:
        return False
    if engine_value == "xlrd" and read_ext != ".xls":
        return False
    return True


def _transfer_marks(engine: Any, read_ext: str) -> Any:
    """
    engine gives us a pytest.param object with some marks, read_ext is just
    a string.  We need to generate a new pytest.param inheriting the marks.
    """
    values = engine.values + (read_ext,)
    new_param = pytest.param(values, marks=engine.marks)
    return new_param


@pytest.fixture(
    params=[
        _transfer_marks(eng, ext)
        for eng in engine_params
        for ext in read_ext_params
        if _is_valid_engine_ext_pair(eng, ext)
    ],
    ids=str,
)
def engine_and_read_ext(request: pytest.FixtureRequest) -> Tuple[Optional[str], str]:
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param


@pytest.fixture
def engine(engine_and_read_ext: Tuple[Optional[str], str]) -> Optional[str]:
    engine_value, _ = engine_and_read_ext
    return engine_value


@pytest.fixture
def read_ext(engine_and_read_ext: Tuple[Optional[str], str]) -> str:
    _, read_ext_value = engine_and_read_ext
    return read_ext_value


@pytest.fixture
def tmp_excel(read_ext: str, tmp_path: Path) -> str:
    tmp: Path = tmp_path / f"{uuid.uuid4()}{read_ext}"
    tmp.touch()
    return str(tmp)


@pytest.fixture
def df_ref(datapath: Any) -> DataFrame:
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath: str = datapath("io", "data", "csv", "test1.csv")
    df_ref_val: DataFrame = read_csv(filepath, index_col=0, parse_dates=True, engine="python")
    return df_ref_val


def get_exp_unit(read_ext: str, engine: Optional[str]) -> str:
    unit: str = "us"
    if (read_ext == ".ods") ^ (engine == "calamine"):
        unit = "s"
    return unit


def adjust_expected(expected: DataFrame, read_ext: str, engine: Optional[str]) -> None:
    expected.index.name = None
    unit: str = get_exp_unit(read_ext, engine)
    # error: "Index" has no attribute "as_unit"
    expected.index = expected.index.as_unit(unit)  # type: ignore[attr-defined]


def xfail_datetimes_with_pyxlsb(engine: Optional[str], request: pytest.FixtureRequest) -> None:
    if engine == "pyxlsb":
        request.applymarker(
            pytest.mark.xfail(
                reason="Sheets containing datetimes not supported by pyxlsb"
            )
        )


class TestReaders:
    @pytest.mark.parametrize("col", [[True, None, False], [True], [True, False]])
    def test_read_excel_type_check(self, col: List[Optional[bool]], tmp_excel: str, read_ext: str) -> None:
        # GH 58159
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({"bool_column": col}, dtype="boolean")
        df.to_excel(tmp_excel, index=False)
        df2 = pd.read_excel(tmp_excel, dtype={"bool_column": "boolean"})
        tm.assert_frame_equal(df, df2)

    def test_pass_none_type(self, datapath: Any) -> None:
        # GH 58159
        f_path: str = datapath("io", "data", "excel", "test_none_type.xlsx")
        with pd.ExcelFile(f_path) as excel:
            parsed: DataFrame = pd.read_excel(
                excel,
                sheet_name="Sheet1",
                keep_default_na=True,
                na_values=["nan", "None", "abcd"],
                dtype="boolean",
                engine="openpyxl",
            )
        expected: DataFrame = DataFrame(
            {"Test": [True, None, False, None, False, None, True]},
            dtype="boolean",
        )
        tm.assert_frame_equal(parsed, expected)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine: Optional[str], datapath: Any, monkeypatch: Any) -> None:
        """
        Change directory and set engine for read_excel calls.
        """
        func = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath("io", "data", "excel"))
        monkeypatch.setattr(pd, "read_excel", func)

    def test_engine_used(self, read_ext: str, engine: Optional[str], monkeypatch: Any) -> None:
        # GH 38884
        def parser(self: Any, *args: Any, **kwargs: Any) -> Any:
            return self.engine

        monkeypatch.setattr(pd.ExcelFile, "parse", parser)

        expected_defaults: Dict[str, str] = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }

        with open("test1" + read_ext, "rb") as f:
            result: Any = pd.read_excel(f)

        if engine is not None:
            expected: str = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_engine_kwargs(self, read_ext: str, engine: Optional[str]) -> None:
        expected_defaults: Dict[str, Any] = {
            "xlsx": {"foo": "abcd"},
            "xlsm": {"foo": 123},
            "xlsb": {"foo": "True"},
            "xls": {"foo": True},
            "ods": {"foo": "abcd"},
        }

        if engine in {"xlrd", "pyxlsb"}:
            msg = re.escape(r"open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == "odf":
            msg = re.escape(r"load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape(r"load_workbook() got an unexpected keyword argument 'foo'")

        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel(
                    "test1" + read_ext,
                    sheet_name="Sheet1",
                    index_col=0,
                    engine_kwargs=expected_defaults[read_ext[1:]],
                )

    def test_usecols_int(self, read_ext: str) -> None:
        msg: str = "Passing an integer for `usecols`"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=3
            )
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet2",
                skiprows=[1],
                index_col=0,
                usecols=3,
            )

    def test_usecols_list(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)
        df1: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=[0, 2, 3]
        )
        df2: DataFrame = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols=[0, 2, 3],
        )
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_usecols_str(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref[["A", "B", "C"]]
        adjust_expected(expected, read_ext, engine)
        df2: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A:D"
        )
        df3: DataFrame = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A:D",
        )
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        expected = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C,D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C,D",
        )
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C:D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C:D",
        )
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize(
        "usecols", [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]]
    )
    def test_usecols_diff_positional_int_columns_order(
        self,
        request: pytest.FixtureRequest,
        engine: Optional[str],
        read_ext: str,
        usecols: List[int],
        df_ref: DataFrame,
    ) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref[["A", "C"]]
        adjust_expected(expected, read_ext, engine)
        result: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=usecols
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("usecols", [["B", "D"], ["D", "B"]])
    def test_usecols_diff_positional_str_columns_order(self, read_ext: str, usecols: List[str], df_ref: DataFrame) -> None:
        expected: DataFrame = df_ref[["B", "D"]]
        expected.index = range(len(expected))
        result: DataFrame = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols=usecols)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_without_slicing(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref
        adjust_expected(expected, read_ext, engine)
        result: DataFrame = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref[["C", "D"]]
        adjust_expected(expected, read_ext, engine)
        result: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,D:E"
        )
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str_invalid(self, read_ext: str) -> None:
        msg: str = "Invalid column name: E1"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols="D:E1")

    def test_index_col_label_error(self, read_ext: str) -> None:
        msg: str = "list indices must be integers.*, not str"
        with pytest.raises(TypeError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet1",
                index_col=["A"],
                usecols=["A", "C"],
            )

    def test_index_col_str(self, read_ext: str) -> None:
        result: DataFrame = pd.read_excel("test1" + read_ext, sheet_name="Sheet3", index_col="A")
        expected: DataFrame = DataFrame(
            columns=["B", "C", "D", "E", "F"], index=Index([], name="A")
        )
        tm.assert_frame_equal(result, expected)

    def test_index_col_empty(self, read_ext: str) -> None:
        result: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet3", index_col=["A", "B", "C"]
        )
        expected: DataFrame = DataFrame(
            columns=["D", "E", "F"],
            index=MultiIndex(levels=[[]] * 3, codes=[[]] * 3, names=["A", "B", "C"]),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("index_col", [None, 2])
    def test_index_col_with_unnamed(self, read_ext: str, index_col: Optional[int]) -> None:
        result: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet4", index_col=index_col
        )
        expected: DataFrame = DataFrame(
            [["i1", "a", "x"], ["i2", "b", "y"]], columns=["Unnamed: 0", "col1", "col2"]
        )
        if index_col:
            expected = expected.set_index(expected.columns[index_col])
        tm.assert_frame_equal(result, expected)

    def test_usecols_pass_non_existent_column(self, read_ext: str) -> None:
        msg: str = (
            "Usecols do not match columns, "
            "columns expected but not found: "
            r"\['E'\]"
        )
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E"])

    def test_usecols_wrong_type(self, read_ext: str) -> None:
        msg: str = (
            "'usecols' must either be list-like of "
            "all strings, all unicode, all integers or a callable."
        )
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E1", 0])

    def test_excel_stop_iterator(self, read_ext: str) -> None:
        parsed: DataFrame = pd.read_excel("test2" + read_ext, sheet_name="Sheet1")
        expected: DataFrame = DataFrame([["aaaa", "bbbbb"]], columns=["Test", "Test1"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_cell_error_na(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(reason="Calamine can't extract error from ods files")
            )
        parsed: DataFrame = pd.read_excel("test3" + read_ext, sheet_name="Sheet1")
        expected: DataFrame = DataFrame([[np.nan]], columns=["Test"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref
        adjust_expected(expected, read_ext, engine)
        df1: DataFrame = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        df2: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet2", skiprows=[1], index_col=0
        )
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        df3: DataFrame = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, skipfooter=1
        )
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_reader_special_dtypes(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        unit: str = get_exp_unit(read_ext, engine)
        expected: DataFrame = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, 4, 0],
                "FloatCol": [1.25, 2.25, 1.83, 1.92, 0.0000000005],
                "BoolCol": [True, False, True, True, False],
                "StrCol": [1, 2, 3, 4, 5],
                "Str2Col": ["a", 3, "c", "d", "e"],
                "DateCol": Index(
                    [
                        datetime(2013, 10, 30),
                        datetime(2013, 10, 31),
                        datetime(1905, 1, 1),
                        datetime(2013, 12, 14),
                        datetime(2015, 3, 14),
                    ],
                    dtype=f"M8[{unit}]",
                ),
            },
        )
        basename: str = "test_types"
        actual: DataFrame = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)
        float_expected: DataFrame = expected.copy()
        float_expected.loc[float_expected.index[1], "Str2Col"] = 3.0
        actual = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, float_expected)
        for icol, name in enumerate(expected.columns):
            actual = pd.read_excel(
                basename + read_ext, sheet_name="Sheet1", index_col=icol
            )
            exp: DataFrame = expected.set_index(name)
            tm.assert_frame_equal(actual, exp)
        expected["StrCol"] = expected["StrCol"].apply(str)
        actual = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters={"StrCol": str}
        )
        tm.assert_frame_equal(actual, expected)

    def test_reader_converters(self, read_ext: str) -> None:
        basename: str = "test_converters"
        expected: DataFrame = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, -1000, 0],
                "FloatCol": [12.5, np.nan, 18.3, 19.2, 0.000000005],
                "BoolCol": ["Found", "Found", "Found", "Not found", "Found"],
                "StrCol": ["1", np.nan, "3", "4", "5"],
            }
        )
        converters: Dict[Union[str, int], Callable[[Any], Any]] = {
            "IntCol": lambda x: int(x) if x != "" else -1000,
            "FloatCol": lambda x: 10 * x if x else np.nan,
            2: lambda x: "Found" if x != "" else "Not found",
            3: lambda x: str(x) if x else "",
        }
        actual: DataFrame = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters=converters
        )
        tm.assert_frame_equal(actual, expected)

    def test_reader_dtype(self, read_ext: str) -> None:
        basename: str = "testdtype"
        actual: DataFrame = pd.read_excel(basename + read_ext)
        expected: DataFrame = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [2.5, 3.5, 4.5, 5.5],
                "c": [1, 2, 3, 4],
                "d": [1.0, 2.0, np.nan, 4.0],
            }
        )
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(
            basename + read_ext, dtype={"a": "float64", "b": "float32", "c": str}
        )
        expected["a"] = expected["a"].astype("float64")
        expected["b"] = expected["b"].astype("float32")
        expected["c"] = Series(["001", "002", "003", "004"], dtype="str")
        tm.assert_frame_equal(actual, expected)
        msg: str = "Unable to convert column d to type int64"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(basename + read_ext, dtype={"d": "int64"})

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (
                None,
                {
                    "a": [1, 2, 3, 4],
                    "b": [2.5, 3.5, 4.5, 5.5],
                    "c": [1, 2, 3, 4],
                    "d": [1.0, 2.0, np.nan, 4.0],
                },
            ),
            (
                {"a": "float64", "b": "float32", "c": str, "d": str},
                {
                    "a": Series([1, 2, 3, 4], dtype="float64"),
                    "b": Series([2.5, 3.5, 4.5, 5.5], dtype="float32"),
                    "c": Series(["001", "002", "003", "004"], dtype="str"),
                    "d": Series(["1", "2", np.nan, "4"], dtype="str"),
                },
            ),
        ],
    )
    def test_reader_dtype_str(self, read_ext: str, dtype: Optional[Dict[str, Any]], expected: Any) -> None:
        basename: str = "testdtype"
        actual: DataFrame = pd.read_excel(basename + read_ext, dtype=dtype)
        expected_df: DataFrame = DataFrame(expected)
        tm.assert_frame_equal(actual, expected_df)

    def test_dtype_backend(self, read_ext: str, dtype_backend: str, engine: Optional[str], tmp_excel: str) -> None:
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df: DataFrame = DataFrame(
            {
                "a": Series([1, 3], dtype="Int64"),
                "b": Series([2.5, 4.5], dtype="Float64"),
                "c": Series([True, False], dtype="boolean"),
                "d": Series(["a", "b"], dtype="string"),
                "e": Series([pd.NA, 6], dtype="Int64"),
                "f": Series([pd.NA, 7.5], dtype="Float64"),
                "g": Series([pd.NA, True], dtype="boolean"),
                "h": Series([pd.NA, "a"], dtype="string"),
                "i": Series([pd.Timestamp("2019-12-31")] * 2),
                "j": Series([pd.NA, pd.NA], dtype="Int64"),
            }
        )
        df.to_excel(tmp_excel, sheet_name="test", index=False)
        result: DataFrame = pd.read_excel(
            tmp_excel, sheet_name="test", dtype_backend=dtype_backend
        )
        if dtype_backend == "pyarrow":
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            expected: DataFrame = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(df[col], from_pandas=True))
                    for col in df.columns
                }
            )
            expected["i"] = ArrowExtensionArray(
                expected["i"].array._pa_array.cast(pa.timestamp(unit="us"))
            )
            expected["j"] = ArrowExtensionArray(pa.array([None, None]))
        else:
            expected = df
            unit: str = get_exp_unit(read_ext, engine)
            expected["i"] = expected["i"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(result, expected)

    def test_dtype_backend_and_dtype(self, read_ext: str, tmp_excel: str) -> None:
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df: DataFrame = DataFrame({"a": [np.nan, 1.0], "b": [2.5, np.nan]})
        df.to_excel(tmp_excel, sheet_name="test", index=False)
        result: DataFrame = pd.read_excel(
            tmp_excel,
            sheet_name="test",
            dtype_backend="numpy_nullable",
            dtype="float64",
        )
        tm.assert_frame_equal(result, df)

    def test_dtype_backend_string(self, read_ext: str, string_storage: str, tmp_excel: str) -> None:
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df: DataFrame = DataFrame(
            {
                "a": np.array(["a", "b"], dtype=np.object_),
                "b": np.array(["x", pd.NA], dtype=np.object_),
            }
        )
        df.to_excel(tmp_excel, sheet_name="test", index=False)
        with pd.option_context("mode.string_storage", string_storage):
            result: DataFrame = pd.read_excel(
                tmp_excel, sheet_name="test", dtype_backend="numpy_nullable"
            )
        expected: DataFrame = DataFrame(
            {
                "a": Series(["a", "b"], dtype=pd.StringDtype(string_storage)),
                "b": Series(["x", None], dtype=pd.StringDtype(string_storage)),
            }
        )
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.parametrize("dtypes, exp_value", [({}, 1), ({"a.1": "int64"}, 1)])
    def test_dtype_mangle_dup_cols(self, read_ext: str, dtypes: Dict[str, Any], exp_value: int) -> None:
        basename: str = "df_mangle_dup_col_dtypes"
        dtype_dict: Dict[str, Any] = {"a": object, **dtypes}
        dtype_dict_copy: Dict[str, Any] = dtype_dict.copy()
        result: DataFrame = pd.read_excel(basename + read_ext, dtype=dtype_dict)
        expected: DataFrame = DataFrame(
            {
                "a": Series([1], dtype=object),
                "a.1": Series([exp_value], dtype=object if not dtypes else None),
            }
        )
        assert dtype_dict == dtype_dict_copy, "dtype dict changed"
        tm.assert_frame_equal(result, expected)

    def test_reader_spaces(self, read_ext: str) -> None:
        basename: str = "test_spaces"
        actual: DataFrame = pd.read_excel(basename + read_ext)
        expected: DataFrame = DataFrame(
            {
                "testcol": [
                    "this is great",
                    "4    spaces",
                    "1 trailing ",
                    " 1 leading",
                    "2  spaces  multiple  times",
                ]
            }
        )
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "basename,expected",
        [
            ("gh-35802", DataFrame({"COLUMN": ["Test (1)"]})),
            ("gh-36122", DataFrame(columns=["got 2nd sa"])),
        ],
    )
    def test_read_excel_ods_nested_xml(self, engine: Optional[str], read_ext: str, basename: str, expected: DataFrame) -> None:
        if engine != "odf":
            pytest.skip(f"Skipped for engine: {engine}")
        actual: DataFrame = pd.read_excel(basename + read_ext)
        tm.assert_frame_equal(actual, expected)

    def test_reading_all_sheets(self, read_ext: str) -> None:
        basename: str = "test_multisheet"
        dfs: Dict[Any, DataFrame] = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys: List[str] = ["Charlie", "Alpha", "Beta"]
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert expected_keys == list(dfs.keys())

    def test_reading_multiple_specific_sheets(self, read_ext: str) -> None:
        basename: str = "test_multisheet"
        expected_keys: List[Union[int, str]] = [2, "Charlie", "Charlie"]
        dfs: Dict[Any, DataFrame] = pd.read_excel(basename + read_ext, sheet_name=expected_keys)
        expected_keys = list(set(expected_keys))
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert len(expected_keys) == len(dfs.keys())

    def test_reading_all_sheets_with_blank(self, read_ext: str) -> None:
        basename: str = "blank_with_header"
        dfs: Dict[Any, DataFrame] = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys: List[str] = ["Sheet1", "Sheet2", "Sheet3"]
        tm.assert_contains_all(expected_keys, dfs.keys())

    def test_read_excel_blank(self, read_ext: str) -> None:
        actual: DataFrame = pd.read_excel("blank" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, DataFrame())

    def test_read_excel_blank_with_header(self, read_ext: str) -> None:
        expected: DataFrame = DataFrame(columns=["col_1", "col_2"])
        actual: DataFrame = pd.read_excel("blank_with_header" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

    def test_exception_message_includes_sheet_name(self, read_ext: str) -> None:
        with pytest.raises(ValueError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("blank_with_header" + read_ext, header=[1], sheet_name=None)
        with pytest.raises(ZeroDivisionError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("test1" + read_ext, usecols=lambda x: 1 / 0, sheet_name=None)

    @pytest.mark.filterwarnings("ignore:Cell A4 is marked:UserWarning:openpyxl")
    def test_date_conversion_overflow(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = DataFrame(
            [
                [pd.Timestamp("2016-03-12"), "Marc Johnson"],
                [pd.Timestamp("2016-03-16"), "Jack Black"],
                [1e20, "Timothy Brown"],
            ],
            columns=["DateColWithBigInt", "StringCol"],
        )
        if engine == "openpyxl":
            request.applymarker(
                pytest.mark.xfail(reason="Maybe not supported by openpyxl")
            )
        if engine is None and read_ext in (".xlsx", ".xlsm"):
            request.applymarker(
                pytest.mark.xfail(reason="Defaults to openpyxl, maybe not supported")
            )
        result: DataFrame = pd.read_excel("testdateoverflow" + read_ext)
        tm.assert_frame_equal(result, expected)

    def test_sheet_name(self, request: pytest.FixtureRequest, read_ext: str, engine: Optional[str], df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        filename: str = "test1"
        sheet_name: str = "Sheet1"
        expected: DataFrame = df_ref
        adjust_expected(expected, read_ext, engine)
        df1: DataFrame = pd.read_excel(
            filename + read_ext, sheet_name=sheet_name, index_col=0
        )
        df2: DataFrame = pd.read_excel(filename + read_ext, index_col=0, sheet_name=sheet_name)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_excel_read_buffer(self, read_ext: str) -> None:
        pth: str = "test1" + read_ext
        expected: DataFrame = pd.read_excel(pth, sheet_name="Sheet1", index_col=0)
        with open(pth, "rb") as f:
            actual: DataFrame = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            tm.assert_frame_equal(expected, actual)

    def test_close_from_py_localpath(self, read_ext: str) -> None:
        str_path: str = os.path.join("test1" + read_ext)
        with open(str_path, "rb") as f:
            x = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            del x
            f.read()

    def test_reader_seconds(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(
                    reason="ODS file contains bad datetime (seconds as text)"
                )
            )
        expected: DataFrame = DataFrame.from_dict(
            {
                "Time": [
                    time(1, 2, 3),
                    time(2, 45, 56, 100000),
                    time(4, 29, 49, 200000),
                    time(6, 13, 42, 300000),
                    time(7, 57, 35, 400000),
                    time(9, 41, 28, 500000),
                    time(11, 25, 21, 600000),
                    time(13, 9, 14, 700000),
                    time(14, 53, 7, 800000),
                    time(16, 37, 0, 900000),
                    time(18, 20, 54),
                ]
            }
        )
        actual: DataFrame = pd.read_excel("times_1900" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel("times_1904" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_multiindex(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        unit: str = get_exp_unit(read_ext, engine)
        mi: MultiIndex = MultiIndex.from_product([["foo", "bar"], ["a", "b"]])
        mi_file: str = "testmultiindex" + read_ext
        expected: DataFrame = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
        )
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")
        actual: DataFrame = pd.read_excel(
            mi_file, sheet_name="mi_column", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        expected.columns = ["a", "b", "c", "d"]
        actual = pd.read_excel(mi_file, sheet_name="mi_index", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi
        actual = pd.read_excel(
            mi_file, sheet_name="both", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)
        expected.columns = ["a", "b", "c", "d"]
        expected.index = mi.set_names(["ilvl1", "ilvl2"])
        actual = pd.read_excel(mi_file, sheet_name="mi_index_name", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.index = range(4)
        expected.columns = mi.set_names(["c1", "c2"])
        actual = pd.read_excel(
            mi_file, sheet_name="mi_column_name", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi.set_levels([1, 2], level=1).set_names(["c1", "c2"])
        actual = pd.read_excel(
            mi_file, sheet_name="name_with_int", index_col=0, header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi.set_names(["c1", "c2"])
        expected.index = mi.set_names(["ilvl1", "ilvl2"])
        actual = pd.read_excel(
            mi_file, sheet_name="both_name", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(
            mi_file,
            sheet_name="both_name_skiprows",
            index_col=[0, 1],
            header=[0, 1],
            skiprows=2,
        )
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "sheet_name,idx_lvl2",
        [
            ("both_name_blank_after_mi_name", [np.nan, "b", "a", "b"]),
            ("both_name_multiple_blanks", [np.nan] * 4),
        ],
    )
    def test_read_excel_multiindex_blank_after_name(
        self,
        request: pytest.FixtureRequest,
        engine: Optional[str],
        read_ext: str,
        sheet_name: str,
        idx_lvl2: List[Any],
    ) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        mi_file: str = "testmultiindex" + read_ext
        mi: MultiIndex = MultiIndex.from_product([["foo", "bar"], ["a", "b"]], names=["c1", "c2"])
        unit: str = get_exp_unit(read_ext, engine)
        expected: DataFrame = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
            index=MultiIndex.from_arrays(
                (["foo", "foo", "bar", "bar"], idx_lvl2),
                names=["ilvl1", "ilvl2"],
            ),
        )
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")
        result: DataFrame = pd.read_excel(
            mi_file,
            sheet_name=sheet_name,
            index_col=[0, 1],
            header=[0, 1],
        )
        tm.assert_frame_equal(result, expected)

    def test_read_excel_multiindex_header_only(self, read_ext: str) -> None:
        mi_file: str = "testmultiindex" + read_ext
        result: DataFrame = pd.read_excel(mi_file, sheet_name="index_col_none", header=[0, 1])
        exp_columns: MultiIndex = MultiIndex.from_product([("A", "B"), ("key", "val")])
        expected: DataFrame = DataFrame([[1, 2, 3, 4]] * 2, columns=exp_columns)
        tm.assert_frame_equal(result, expected)

    def test_excel_old_index_format(self, read_ext: str) -> None:
        filename: str = "test_index_name_pre17" + read_ext
        data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ],
            dtype=object,
        )
        columns: List[str] = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        mi: MultiIndex = MultiIndex(
            levels=[
                ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R1", "R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            names=[None, None],
        )
        si: Index = Index(
            ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None
        )
        expected: DataFrame = DataFrame(data, index=si, columns=columns)
        actual: DataFrame = pd.read_excel(filename, sheet_name="single_names", index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        actual = pd.read_excel(filename, sheet_name="multi_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        data = np.array(
            [
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ]
        )
        columns = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        mi = MultiIndex(
            levels=[
                ["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
            names=[None, None],
        )
        si = Index(["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None)
        expected = DataFrame(data, index=si, columns=columns)
        actual = pd.read_excel(filename, sheet_name="single_no_names", index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        actual = pd.read_excel(filename, sheet_name="multi_no_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_bool_header_arg(self, read_ext: str) -> None:
        msg: str = "Passing a bool to header is invalid"
        for arg in [True, False]:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel("test1" + read_ext, header=arg)

    def test_read_excel_skiprows(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        unit: str = get_exp_unit(read_ext, engine)
        actual: DataFrame = pd.read_excel(
            "testskiprows" + read_ext, sheet_name="skiprows_list", skiprows=[0, 2]
        )
        expected: DataFrame = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=np.array([0, 2]),
        )
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x in [0, 2],
        )
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=3,
            names=["a", "b", "c", "d"],
        )
        expected = DataFrame(
            [
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_skiprows_callable_not_in(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        unit: str = get_exp_unit(read_ext, engine)
        actual: DataFrame = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x not in [1, 3, 5],
        )
        expected: DataFrame = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows(self, read_ext: str) -> None:
        num_rows_to_pull: int = 5
        actual: DataFrame = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        expected: DataFrame = pd.read_excel("test1" + read_ext)
        expected = expected[:num_rows_to_pull]
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_greater_than_nrows_in_file(self, read_ext: str) -> None:
        expected: DataFrame = pd.read_excel("test1" + read_ext)
        num_records_in_file: int = len(expected)
        num_rows_to_pull: int = num_records_in_file + 10
        actual: DataFrame = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_non_integer_parameter(self, read_ext: str) -> None:
        msg: str = "'nrows' must be an integer >=0"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, nrows="5")

    @pytest.mark.parametrize(
        "filename,sheet_name,header,index_col,skiprows",
        [
            ("testmultiindex", "mi_column", [0, 1], 0, None),
            ("testmultiindex", "mi_index", None, [0, 1], None),
            ("testmultiindex", "both", [0, 1], [0, 1], None),
            ("testmultiindex", "mi_column_name", [0, 1], 0, None),
            ("testskiprows", "skiprows_list", None, None, [0, 2]),
            ("testskiprows", "skiprows_list", None, None, lambda x: x in (0, 2)),
        ],
    )
    def test_read_excel_nrows_params(
        self,
        read_ext: str,
        filename: str,
        sheet_name: Union[str, int, List[Union[int, str]]],
        header: Optional[Union[int, List[int]]],
        index_col: Optional[Union[int, List[int]]],
        skiprows: Optional[Union[List[int], Callable[[int], bool]]],
    ) -> None:
        expected: DataFrame = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
        ).iloc[:3]
        actual: DataFrame = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
            nrows=3,
        )
        tm.assert_frame_equal(actual, expected)

    def test_deprecated_kwargs(self, read_ext: str) -> None:
        with pytest.raises(TypeError, match="but 3 positional arguments"):
            pd.read_excel("test1" + read_ext, "Sheet1", 0)

    def test_no_header_with_list_index_col(self, read_ext: str) -> None:
        file_name: str = "testmultiindex" + read_ext
        data = [("B", "B"), ("key", "val"), (3, 4), (3, 4)]
        idx: MultiIndex = MultiIndex.from_tuples(
            [("A", "A"), ("key", "val"), (1, 2), (1, 2)], names=(0, 1)
        )
        expected: DataFrame = DataFrame(data, index=idx, columns=(2, 3))
        result: DataFrame = pd.read_excel(
            file_name, sheet_name="index_col_none", index_col=[0, 1], header=None
        )
        tm.assert_frame_equal(expected, result)

    def test_one_col_noskip_blank_line(self, read_ext: str) -> None:
        file_name: str = "one_col_blank_line" + read_ext
        data: List[Union[float, None]] = [0.5, np.nan, 1, 2]
        expected: DataFrame = DataFrame(data, columns=["numbers"])
        result: DataFrame = pd.read_excel(file_name)
        tm.assert_frame_equal(result, expected)

    def test_multiheader_two_blank_lines(self, read_ext: str) -> None:
        file_name: str = "testmultiindex" + read_ext
        columns: MultiIndex = MultiIndex.from_tuples([("a", "A"), ("b", "B")])
        data = [[np.nan, np.nan], [np.nan, np.nan], [1, 3], [2, 4]]
        expected: DataFrame = DataFrame(data, columns=columns)
        result: DataFrame = pd.read_excel(
            file_name, sheet_name="mi_column_empty_rows", header=[0, 1]
        )
        tm.assert_frame_equal(result, expected)

    def test_trailing_blanks(self, read_ext: str) -> None:
        file_name: str = "trailing_blanks" + read_ext
        result: DataFrame = pd.read_excel(file_name)
        assert result.shape == (3, 3)

    def test_ignore_chartsheets_by_str(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pytest.raises(ValueError, match="Worksheet named 'Chart1' not found"):
            pd.read_excel("chartsheet" + read_ext, sheet_name="Chart1")

    def test_ignore_chartsheets_by_int(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pytest.raises(
            ValueError, match="Worksheet index 1 is invalid, 1 worksheets found"
        ):
            pd.read_excel("chartsheet" + read_ext, sheet_name=1)

    def test_euro_decimal_format(self, read_ext: str) -> None:
        result: DataFrame = pd.read_excel("test_decimal" + read_ext, decimal=",", skiprows=1)
        expected: DataFrame = DataFrame(
            [
                [1, 1521.1541, 187101.9543, "ABC", "poi", 4.738797819],
                [2, 121.12, 14897.76, "DEF", "uyt", 0.377320872],
                [3, 878.158, 108013.434, "GHI", "rez", 2.735694704],
            ],
            columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
        )
        tm.assert_frame_equal(result, expected)


class TestExcelFileRead:
    def test_raises_bytes_input(self, engine: Optional[str], read_ext: str) -> None:
        msg: str = "Expected file path name or file-like object"
        with pytest.raises(TypeError, match=msg):
            with open("test1" + read_ext, "rb") as f:
                pd.read_excel(f.read(), engine=engine)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine: Optional[str], datapath: Any, monkeypatch: Any) -> None:
        func = partial(pd.ExcelFile, engine=engine)
        monkeypatch.chdir(datapath("io", "data", "excel"))
        monkeypatch.setattr(pd, "ExcelFile", func)

    def test_engine_used(self, read_ext: str, engine: Optional[str]) -> None:
        expected_defaults: Dict[str, str] = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }
        with pd.ExcelFile("test1" + read_ext) as excel:
            result: Any = excel.engine
        if engine is not None:
            expected: str = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_excel_passes_na(self, read_ext: str) -> None:
        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed: DataFrame = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        expected: DataFrame = DataFrame(
            [["NA"], [1], ["NA"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        expected = DataFrame(
            [["1.#QNAN"], [1], ["nan"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize("na_filter", [None, True, False])
    def test_excel_passes_na_filter(self, read_ext: str, na_filter: Optional[bool]) -> None:
        kwargs: Dict[str, Any] = {}
        if na_filter is not None:
            kwargs["na_filter"] = na_filter
        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed: DataFrame = pd.read_excel(
                excel,
                sheet_name="Sheet1",
                keep_default_na=True,
                na_values=["apple"],
                **kwargs,
            )
        if na_filter is False:
            expected = [["1.#QNAN"], [1], ["nan"], ["apple"], ["rabbit"]]
        else:
            expected = [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]]
        expected = DataFrame(expected, columns=["Test"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table_sheet_by_index(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref
        adjust_expected(expected, read_ext, engine)
        with pd.ExcelFile("test1" + read_ext) as excel:
            df1: DataFrame = pd.read_excel(excel, sheet_name=0, index_col=0)
            df2: DataFrame = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile("test1" + read_ext) as excel:
            df1 = excel.parse(0, index_col=0)
            df2 = excel.parse(1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile("test1" + read_ext) as excel:
            df3: DataFrame = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])
        with pd.ExcelFile("test1" + read_ext) as excel:
            df3 = excel.parse(0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_sheet_name(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str, df_ref: DataFrame) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        expected: DataFrame = df_ref
        adjust_expected(expected, read_ext, engine)
        filename: str = "test1"
        sheet_name: str = "Sheet1"
        with pd.ExcelFile(filename + read_ext) as excel:
            df1_parse: DataFrame = excel.parse(sheet_name=sheet_name, index_col=0)
        with pd.ExcelFile(filename + read_ext) as excel:
            df2_parse: DataFrame = excel.parse(index_col=0, sheet_name=sheet_name)
        tm.assert_frame_equal(df1_parse, expected)
        tm.assert_frame_equal(df2_parse, expected)

    @pytest.mark.parametrize(
        "sheet_name",
        [3, [0, 3], [3, 0], "Sheet4", ["Sheet1", "Sheet4"], ["Sheet4", "Sheet1"]],
    )
    def test_bad_sheetname_raises(self, read_ext: str, sheet_name: Union[int, List[Union[int, str]], str]) -> None:
        msg: str = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            with pd.ExcelFile("blank" + read_ext) as excel:
                excel.parse(sheet_name=sheet_name)

    def test_excel_read_buffer(self, engine: Optional[str], read_ext: str) -> None:
        pth: str = "test1" + read_ext
        expected: DataFrame = pd.read_excel(pth, sheet_name="Sheet1", index_col=0, engine=engine)
        with open(pth, "rb") as f:
            with pd.ExcelFile(f) as xls:
                actual: DataFrame = pd.read_excel(xls, sheet_name="Sheet1", index_col=0)
        tm.assert_frame_equal(expected, actual)

    def test_reader_closes_file(self, engine: Optional[str], read_ext: str) -> None:
        with open("test1" + read_ext, "rb") as f:
            with pd.ExcelFile(f) as xlsx:
                pd.read_excel(xlsx, sheet_name="Sheet1", index_col=0, engine=engine)
        assert f.closed

    def test_conflicting_excel_engines(self, read_ext: str) -> None:
        msg: str = "Engine should not be specified when passing an ExcelFile"
        with pd.ExcelFile("test1" + read_ext) as xl:
            with pytest.raises(ValueError, match=msg):
                pd.read_excel(xl, engine="foo")

    def test_excel_read_binary(self, engine: Optional[str], read_ext: str) -> None:
        expected: DataFrame = pd.read_excel("test1" + read_ext, engine=engine)
        with open("test1" + read_ext, "rb") as f:
            data: bytes = f.read()
        actual: DataFrame = pd.read_excel(BytesIO(data), engine=engine)
        tm.assert_frame_equal(expected, actual)

    def test_excel_read_binary_via_read_excel(self, read_ext: str, engine: Optional[str]) -> None:
        with open("test1" + read_ext, "rb") as f:
            result: DataFrame = pd.read_excel(f, engine=engine)
        expected: DataFrame = pd.read_excel("test1" + read_ext, engine=engine)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_header_index_out_of_range(self, engine: Optional[str]) -> None:
        with open("df_header_oob.xlsx", "rb") as f:
            with pytest.raises(ValueError, match="exceeds maximum"):
                pd.read_excel(f, header=[0, 1])

    @pytest.mark.parametrize("filename", ["df_empty.xlsx", "df_equals.xlsx"])
    def test_header_with_index_col(self, filename: str) -> None:
        idx: Index = Index(["Z"], name="I2")
        cols: MultiIndex = MultiIndex.from_tuples([("A", "B"), ("A", "B.1")], names=["I11", "I12"])
        expected: DataFrame = DataFrame([[1, 3]], index=idx, columns=cols, dtype="int64")
        result: DataFrame = pd.read_excel(
            filename, sheet_name="Sheet1", index_col=0, header=[0, 1]
        )
        tm.assert_frame_equal(expected, result)

    def test_read_datetime_multiindex(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        xfail_datetimes_with_pyxlsb(engine, request)
        f: str = "test_datetime_mi" + read_ext
        with pd.ExcelFile(f) as excel:
            actual: DataFrame = pd.read_excel(excel, header=[0, 1], index_col=0, engine=engine)
        unit: str = get_exp_unit(read_ext, engine)
        dti: pd.DatetimeIndex = pd.DatetimeIndex(["2020-02-29", "2020-03-01"], dtype=f"M8[{unit}]")
        expected_column_index: MultiIndex = MultiIndex.from_arrays(
            [dti[:1], dti[1:]],
            names=[
                dti[0].to_pydatetime(),
                dti[1].to_pydatetime(),
            ],
        )
        expected: DataFrame = DataFrame([], index=[], columns=expected_column_index)
        tm.assert_frame_equal(expected, actual)

    def test_engine_invalid_option(self, read_ext: str) -> None:
        with pytest.raises(ValueError, match="Value must be one of *"):
            with pd.option_context(f"io.excel{read_ext}.reader", "abc"):
                pass

    def test_ignore_chartsheets(self, request: pytest.FixtureRequest, engine: Optional[str], read_ext: str) -> None:
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pd.ExcelFile("chartsheet" + read_ext) as excel:
            assert excel.sheet_names == ["Sheet1"]

    def test_corrupt_files_closed(self, engine: Optional[str], tmp_excel: str) -> None:
        errors: Tuple[Any, ...] = (BadZipFile,)
        if engine is None:
            pytest.skip(f"Invalid test for engine={engine}")
        elif engine == "xlrd":
            import xlrd
            errors = (BadZipFile, xlrd.biffh.XLRDError)
        elif engine == "calamine":
            from python_calamine import CalamineError
            errors = (CalamineError,)
        Path(tmp_excel).write_text("corrupt", encoding="utf-8")
        with tm.assert_produces_warning(False):
            try:
                pd.ExcelFile(tmp_excel, engine=engine)
            except errors:
                pass