#!/usr/bin/env python3
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
from typing import Any, Union, Optional, Dict, List
import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat import HAS_PYARROW
from pandas.errors import EmptyDataError, ParserError, ParserWarning
from pandas import DataFrame, Index, compat
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings("ignore:Passing a BlockManager to DataFrame:DeprecationWarning")
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_read_csv_local(all_parsers: Any, csv1: str) -> None:
    prefix: str = "file:///" if compat.is_platform_windows() else "file://"
    parser: Any = all_parsers
    fname: str = prefix + str(os.path.abspath(csv1))
    result: DataFrame = parser.read_csv(fname, index_col=0, parse_dates=True)
    expected: DataFrame = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.34627],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            dtype="M8[s]",
            name="index",
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_1000_sep(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "A|B|C\n1|2,334|5\n10|13|10.\n"
    expected: DataFrame = DataFrame({"A": [1, 10], "B": [2334, 13], "C": [5, 10.0]})
    if parser.engine == "pyarrow":
        msg: str = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep="|", thousands=",")
        return
    result: DataFrame = parser.read_csv(StringIO(data), sep="|", thousands=",")
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_unnamed_columns(all_parsers: Any) -> None:
    data: str = "A,B,C,,\n1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15\n"
    parser: Any = all_parsers
    expected: DataFrame = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=np.int64,
        columns=["A", "B", "C", "Unnamed: 3", "Unnamed: 4"],
    )
    result: DataFrame = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_csv_mixed_type(all_parsers: Any) -> None:
    data: str = "A,B,C\na,1,2\nb,3,4\nc,4,5\n"
    parser: Any = all_parsers
    expected: DataFrame = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]})
    result: DataFrame = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_read_csv_low_memory_no_rows_with_index(all_parsers: Any) -> None:
    parser: Any = all_parsers
    if not parser.low_memory:
        pytest.skip("This is a low-memory specific test")
    data: str = "A,B,C\n1,1,1,2\n2,2,3,4\n3,3,4,5\n"
    if parser.engine == "pyarrow":
        msg: str = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
        return
    result: DataFrame = parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
    expected: DataFrame = DataFrame(columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


def test_read_csv_dataframe(all_parsers: Any, csv1: str) -> None:
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(csv1, index_col=0, parse_dates=True)
    expected: DataFrame = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.34627],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            dtype="M8[s]",
            name="index",
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [3, 3.0])
def test_read_nrows(all_parsers: Any, nrows: Union[int, float]) -> None:
    data: str = (
        "index,A,B,C,D\n"
        "foo,2,3,4,5\n"
        "bar,7,8,9,10\n"
        "baz,12,13,14,15\n"
        "qux,12,13,14,15\n"
        "foo2,12,13,14,15\n"
        "bar2,12,13,14,15\n"
    )
    expected: DataFrame = DataFrame(
        [["foo", 2, 3, 4, 5], ["bar", 7, 8, 9, 10], ["baz", 12, 13, 14, 15]],
        columns=["index", "A", "B", "C", "D"],
    )
    parser: Any = all_parsers
    if parser.engine == "pyarrow":
        msg: str = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), nrows=nrows)
        return
    result: DataFrame = parser.read_csv(StringIO(data), nrows=nrows)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1.2, "foo", -1])
def test_read_nrows_bad(all_parsers: Any, nrows: Union[int, float, str]) -> None:
    data: str = (
        "index,A,B,C,D\n"
        "foo,2,3,4,5\n"
        "bar,7,8,9,10\n"
        "baz,12,13,14,15\n"
        "qux,12,13,14,15\n"
        "foo2,12,13,14,15\n"
        "bar2,12,13,14,15\n"
    )
    msg: str = "'nrows' must be an integer >=0"
    parser: Any = all_parsers
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), nrows=nrows)


def test_nrows_skipfooter_errors(all_parsers: Any) -> None:
    msg: str = "'skipfooter' not supported with 'nrows'"
    data: str = "a\n1\n2\n3\n4\n5\n6"
    parser: Any = all_parsers
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=1, nrows=5)


@skip_pyarrow
def test_missing_trailing_delimiters(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "A,B,C,D\n1,2,3,4\n1,3,3,\n1,4,5"
    result: DataFrame = parser.read_csv(StringIO(data))
    expected: DataFrame = DataFrame(
        [[1, 2, 3, 4], [1, 3, 3, np.nan], [1, 4, 5, np.nan]], columns=["A", "B", "C", "D"]
    )
    tm.assert_frame_equal(result, expected)


def test_skip_initial_space(all_parsers: Any) -> None:
    data: str = (
        '"09-Apr-2012", "01:10:18.300", 2456026.548822908, 12849, 1.00361,  '
        "1.12551, 330.65659, 0355626618.16711,  73.48821, 314.11625,  1917.09447,   "
        "179.71425,  80.000, 240.000, -350,  70.06056, 344.98370, 1,   1, -0.689265, "
        "-0.692787,  0.212036,    14.7674,   41.605,   -9999.0,   -9999.0,   -9999.0,   "
        "-9999.0,   -9999.0,  -9999.0, 000, 012, 128"
    )
    parser: Any = all_parsers
    if parser.engine == "pyarrow":
        msg: str = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=list(range(33)),
                header=None,
                na_values=["-9999.0"],
                skipinitialspace=True,
            )
        return
    result: DataFrame = parser.read_csv(
        StringIO(data),
        names=list(range(33)),
        header=None,
        na_values=["-9999.0"],
        skipinitialspace=True,
    )
    expected: DataFrame = DataFrame(
        [
            [
                "09-Apr-2012",
                "01:10:18.300",
                2456026.548822908,
                12849,
                1.00361,
                1.12551,
                330.65659,
                355626618.16711,
                73.48821,
                314.11625,
                1917.09447,
                179.71425,
                80.0,
                240.0,
                -350,
                70.06056,
                344.9837,
                1,
                1,
                -0.689265,
                -0.692787,
                0.212036,
                14.7674,
                41.605,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                12,
                128,
            ]
        ]
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_trailing_delimiters(all_parsers: Any) -> None:
    data: str = "A,B,C\n1,2,3,\n4,5,6,\n7,8,9,"
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), index_col=False)
    expected: DataFrame = DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


def test_escapechar(all_parsers: Any) -> None:
    data: str = (
        'SEARCH_TERM,ACTUAL_URL\n'
        '"bra tv board","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"\n'
        '"tv pÃ¥ hjul","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"\n'
        '"SLAGBORD, \\"Bergslagen\\", IKEA:s 1700-tals series","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"'
    )
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), escapechar="\\", quotechar='"', encoding="utf-8")
    assert result["SEARCH_TERM"][2] == 'SLAGBORD, "Bergslagen", IKEA:s 1700-tals series'
    tm.assert_index_equal(result.columns, Index(["SEARCH_TERM", "ACTUAL_URL"]))


def test_ignore_leading_whitespace(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = " a b c\n 1 2 3\n 4 5 6\n 7 8 9"
    if parser.engine == "pyarrow":
        msg: str = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep="\\s+")
        return
    result: DataFrame = parser.read_csv(StringIO(data), sep="\\s+")
    expected: DataFrame = DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("usecols", [None, [0, 1], ["a", "b"]])
def test_uneven_lines_with_usecols(all_parsers: Any, usecols: Optional[Union[List[int], List[str]]]) -> None:
    parser: Any = all_parsers
    data: str = "a,b,c\n0,1,2\n3,4,5,6,7\n8,9,10"
    if usecols is None:
        msg: str = "Expected \\d+ fields in line \\d+, saw \\d+"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
    else:
        expected: DataFrame = DataFrame({"a": [0, 3, 8], "b": [1, 4, 9]})
        result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols)
        tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        ("", {}, None),
        ("", {"usecols": ["X"]}, None),
        (
            ",,",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"], index=[0], dtype=np.float64),
        ),
        (
            "",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"]),
        ),
    ],
)
def test_read_empty_with_usecols(
    all_parsers: Any, data: str, kwargs: Dict[str, Any], expected: Optional[DataFrame]
) -> None:
    parser: Any = all_parsers
    if expected is None:
        msg: str = "No columns to parse from file"
        with pytest.raises(EmptyDataError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result: DataFrame = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,expected_data",
    [
        (
            {"header": None, "sep": "\\s+", "skiprows": [0, 1, 2, 3, 5, 6], "skip_blank_lines": True},
            [[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]],
        ),
        (
            {"sep": "\\s+", "skiprows": [1, 2, 3, 5, 6], "skip_blank_lines": True},
            {"A": [1.0, 5.1], "B": [2.0, np.nan], "C": [4.0, 10]},
        ),
    ],
)
def test_trailing_spaces(all_parsers: Any, kwargs: Dict[str, Any], expected_data: Any) -> None:
    data: str = (
        "A B C  \n"
        "random line with trailing spaces    \n"
        "skip\n"
        "1,2,3\n"
        "1,2.,4.\n"
        "random line with trailing tabs\t\t\t\n"
        "   \n"
        "5.1,NaN,10.0\n"
    )
    parser: Any = all_parsers
    if parser.engine == "pyarrow":
        with pytest.raises(ValueError, match="the 'pyarrow' engine does not support"):
            parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
        return
    expected: DataFrame = DataFrame(expected_data)
    result: DataFrame = parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
    tm.assert_frame_equal(result, expected)


def test_read_filepath_or_buffer(all_parsers: Any) -> None:
    parser: Any = all_parsers
    with pytest.raises(TypeError, match="Expected file path name or file-like"):
        parser.read_csv(filepath_or_buffer=b"input")


def test_single_char_leading_whitespace(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "MyColumn\na\nb\na\nb\n"
    if parser.engine == "pyarrow":
        msg: str = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), skipinitialspace=True)
        return
    expected: DataFrame = DataFrame({"MyColumn": list("abab")})
    result: DataFrame = parser.read_csv(StringIO(data), skipinitialspace=True, sep="\\s+")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sep,skip_blank_lines,exp_data",
    [
        (",", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        ("\\s+", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        (
            ",",
            False,
            [
                [1.0, 2.0, 4.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [5.0, np.nan, 10.0],
                [np.nan, np.nan, np.nan],
                [-70.0, 0.4, 1.0],
            ],
        ),
    ],
)
def test_empty_lines(
    all_parsers: Any, sep: str, skip_blank_lines: bool, exp_data: List[List[Union[float, int]]], request: Any
) -> None:
    parser: Any = all_parsers
    data: str = "A,B,C\n1,2.,4.\n\n\n5.,NaN,10.0\n\n-70,.4,1\n"
    if sep == "\\s+":
        data = data.replace(",", "  ")
        if parser.engine == "pyarrow":
            msg: str = "the 'pyarrow' engine does not support regex separators"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines)
            return
    result: DataFrame = parser.read_csv(StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines)
    expected: DataFrame = DataFrame(exp_data, columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_whitespace_lines(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "\n\n\t  \t\t\n\t\nA,B,C\n\t    1,2.,4.\n5.,NaN,10.0\n"
    expected: DataFrame = DataFrame([[1, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"])
    result: DataFrame = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            "   A   B   C   D\na   1   2   3   4\nb   1   2   3   4\nc   1   2   3   4\n",
            DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], columns=["A", "B", "C", "D"], index=["a", "b", "c"]),
        ),
        (
            "    a b c\n1 2 3 \n4 5  6\n 7 8 9",
            DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]),
        ),
    ],
)
def test_whitespace_regex_separator(all_parsers: Any, data: str, expected: DataFrame) -> None:
    parser: Any = all_parsers
    if parser.engine == "pyarrow":
        msg: str = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep="\\s+")
        return
    result: DataFrame = parser.read_csv(StringIO(data), sep="\\s+")
    tm.assert_frame_equal(result, expected)


def test_sub_character(all_parsers: Any, csv_dir_path: str) -> None:
    filename: str = os.path.join(csv_dir_path, "sub_char.csv")
    expected: DataFrame = DataFrame([[1, 2, 3]], columns=["a", "\x1ab", "c"])
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(filename)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("filename", ["sé-es-vé.csv", "ru-sй.csv", "中文文件名.csv"])
def test_filename_with_special_chars(all_parsers: Any, filename: str) -> None:
    parser: Any = all_parsers
    df: DataFrame = DataFrame({"a": [1, 2, 3]})
    with tm.ensure_clean(filename) as path:
        df.to_csv(path, index=False)
        result: DataFrame = parser.read_csv(path)
        tm.assert_frame_equal(result, df)


def test_read_table_same_signature_as_read_csv(all_parsers: Any) -> None:
    parser: Any = all_parsers
    table_sign = signature(parser.read_table)
    csv_sign = signature(parser.read_csv)
    assert table_sign.parameters.keys() == csv_sign.parameters.keys()
    assert table_sign.return_annotation == csv_sign.return_annotation
    for key, csv_param in csv_sign.parameters.items():
        table_param = table_sign.parameters[key]
        if key == "sep":
            assert csv_param.default == ","
            assert table_param.default == "\t"
            assert table_param.annotation == csv_param.annotation
            assert table_param.kind == csv_param.kind
            continue
        assert table_param == csv_param


def test_read_table_equivalency_to_read_csv(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "a\tb\n1\t2\n3\t4"
    expected: DataFrame = parser.read_csv(StringIO(data), sep="\t")
    result: DataFrame = parser.read_table(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("read_func", ["read_csv", "read_table"])
def test_read_csv_and_table_sys_setprofile(all_parsers: Any, read_func: str) -> None:
    parser: Any = all_parsers
    data: str = "a b\n0 1"
    sys.setprofile(lambda *a, **k: None)
    result: DataFrame = getattr(parser, read_func)(StringIO(data))
    sys.setprofile(None)
    expected: DataFrame = DataFrame({"a b": ["0 1"]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_first_row_bom(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "\ufeff\"Head1\"\t\"Head2\"\t\"Head3\""
    result: DataFrame = parser.read_csv(StringIO(data), delimiter="\t")
    expected: DataFrame = DataFrame(columns=["Head1", "Head2", "Head3"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_first_row_bom_unquoted(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "\ufeffHead1\tHead2\tHead3"
    result: DataFrame = parser.read_csv(StringIO(data), delimiter="\t")
    expected: DataFrame = DataFrame(columns=["Head1", "Head2", "Head3"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers: Any, nrows: int) -> None:
    ref: DataFrame = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [1, 2], [np.nan, np.nan], [3, 4]], columns=list("ab")
    )
    csv: str = "\nheader\n\na,b\n\n\n1,2\n\n3,4"
    parser: Any = all_parsers
    if parser.engine == "pyarrow":
        msg: str = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
        return
    df: DataFrame = parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
    tm.assert_frame_equal(df, ref[:nrows])


@skip_pyarrow
def test_no_header_two_extra_columns(all_parsers: Any) -> None:
    column_names: List[str] = ["one", "two", "three"]
    ref: DataFrame = DataFrame([["foo", "bar", "baz"]], columns=column_names)
    stream: StringIO = StringIO("foo,bar,baz,bam,blah")
    parser: Any = all_parsers
    df: DataFrame = parser.read_csv_check_warnings(
        ParserWarning,
        "Length of header or names does not match length of data. This leads to a loss of data with index_col=False.",
        stream,
        header=None,
        names=column_names,
        index_col=False,
    )
    tm.assert_frame_equal(df, ref)


def test_read_csv_names_not_accepting_sets(all_parsers: Any) -> None:
    data: str = "    1,2,3\n    4,5,6\n"
    parser: Any = all_parsers
    with pytest.raises(ValueError, match="Names should be an ordered collection."):
        parser.read_csv(StringIO(data), names=set("QAZ"))


def test_read_csv_delimiter_and_sep_no_default(all_parsers: Any) -> None:
    f: StringIO = StringIO("a,b\n1,2")
    parser: Any = all_parsers
    msg: str = "Specified a sep and a delimiter; you can only specify one."
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(f, sep=" ", delimiter=".")


@pytest.mark.parametrize("kwargs", [{"delimiter": "\n"}, {"sep": "\n"}])
def test_read_csv_line_break_as_separator(kwargs: Dict[str, Any], all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "a,b,c\n1,2,3\n    "
    msg: str = (
        "Specified \\\\n as separator or delimiter. This forces the python engine which does not accept a line terminator. "
        "Hence it is not allowed to use the line terminator as separator."
    )
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


@skip_pyarrow
def test_dict_keys_as_names(all_parsers: Any) -> None:
    data: str = "1,2"
    keys = {"a": int, "b": int}.keys()
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), names=keys)
    expected: DataFrame = DataFrame({"a": [1], "b": [2]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(using_string_dtype() and HAS_PYARROW, reason="TODO(infer_string)")
@xfail_pyarrow
def test_encoding_surrogatepass(all_parsers: Any) -> None:
    parser: Any = all_parsers
    content: bytes = b"\xed\xbd\xbf"
    decoded: str = content.decode("utf-8", errors="surrogatepass")
    expected: DataFrame = DataFrame({decoded: [decoded]}, index=[decoded * 2])
    expected.index.name = decoded * 2
    with tm.ensure_clean() as path:
        Path(path).write_bytes(content * 2 + b"," + content + b"\n" + content * 2 + b"," + content)
        df: DataFrame = parser.read_csv(path, encoding_errors="surrogatepass", index_col=0)
        tm.assert_frame_equal(df, expected)
        with pytest.raises(UnicodeDecodeError, match="'utf-8' codec can't decode byte"):
            parser.read_csv(path)


def test_malformed_second_line(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = "\na\nb\n"
    result: DataFrame = parser.read_csv(StringIO(data), skip_blank_lines=False, header=1)
    expected: DataFrame = DataFrame({"a": ["b"]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_short_single_line(all_parsers: Any) -> None:
    parser: Any = all_parsers
    columns: List[str] = ["a", "b", "c"]
    data: str = "1,2"
    result: DataFrame = parser.read_csv(StringIO(data), header=None, names=columns)
    expected: DataFrame = DataFrame({"a": [1], "b": [2], "c": [np.nan]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_short_multi_line(all_parsers: Any) -> None:
    parser: Any = all_parsers
    columns: List[str] = ["a", "b", "c"]
    data: str = "1,2\n1,2"
    result: DataFrame = parser.read_csv(StringIO(data), header=None, names=columns)
    expected: DataFrame = DataFrame({"a": [1, 1], "b": [2, 2], "c": [np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_read_seek(all_parsers: Any) -> None:
    parser: Any = all_parsers
    prefix: str = "### DATA\n"
    content: str = "nkey,value\ntables,rectangular\n"
    with tm.ensure_clean() as path:
        Path(path).write_text(prefix + content, encoding="utf-8")
        with open(path, encoding="utf-8") as file:
            file.readline()
            actual: DataFrame = parser.read_csv(file)
        expected: DataFrame = parser.read_csv(StringIO(content))
    tm.assert_frame_equal(actual, expected)