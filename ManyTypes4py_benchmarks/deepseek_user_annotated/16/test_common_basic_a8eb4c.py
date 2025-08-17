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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import HAS_PYARROW
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)

from pandas import (
    DataFrame,
    Index,
    compat,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_read_csv_local(all_parsers: Any, csv1: str) -> None:
    prefix = "file:///" if compat.is_platform_windows() else "file://"
    parser = all_parsers

    fname = prefix + str(os.path.abspath(csv1))
    result = parser.read_csv(fname, index_col=0, parse_dates=True)

    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
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
    parser = all_parsers
    data = """A|B|C
1|2,334|5
10|13|10.
"""
    expected = DataFrame({"A": [1, 10], "B": [2334, 13], "C": [5, 10.0]})

    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep="|", thousands=",")
        return

    result = parser.read_csv(StringIO(data), sep="|", thousands=",")
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_unnamed_columns(all_parsers: Any) -> None:
    data = """A,B,C,,
1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=np.int64,
        columns=["A", "B", "C", "Unnamed: 3", "Unnamed: 4"],
    )
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_csv_mixed_type(all_parsers: Any) -> None:
    data = """A,B,C
a,1,2
b,3,4
c,4,5
"""
    parser = all_parsers
    expected = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]})
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_read_csv_low_memory_no_rows_with_index(all_parsers: Any) -> None:
    # see gh-21141
    parser = all_parsers

    if not parser.low_memory:
        pytest.skip("This is a low-memory specific test")

    data = """A,B,C
1,1,1,2
2,2,3,4
3,3,4,5
"""

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
        return

    result = parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
    expected = DataFrame(columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


def test_read_csv_dataframe(all_parsers: Any, csv1: str) -> None:
    parser = all_parsers
    result = parser.read_csv(csv1, index_col=0, parse_dates=True)
    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
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
    # see gh-10476
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    expected = DataFrame(
        [["foo", 2, 3, 4, 5], ["bar", 7, 8, 9, 10], ["baz", 12, 13, 14, 15]],
        columns=["index", "A", "B", "C", "D"],
    )
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), nrows=nrows)
        return

    result = parser.read_csv(StringIO(data), nrows=nrows)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1.2, "foo", -1])
def test_read_nrows_bad(all_parsers: Any, nrows: Any) -> None:
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    msg = r"'nrows' must be an integer >=0"
    parser = all_parsers
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), nrows=nrows)


def test_nrows_skipfooter_errors(all_parsers: Any) -> None:
    msg = "'skipfooter' not supported with 'nrows'"
    data = "a\n1\n2\n3\n4\n5\n6"
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=1, nrows=5)


@skip_pyarrow
def test_missing_trailing_delimiters(all_parsers: Any) -> None:
    parser = all_parsers
    data = """A,B,C,D
1,2,3,4
1,3,3,
1,4,5"""

    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [[1, 2, 3, 4], [1, 3, 3, np.nan], [1, 4, 5, np.nan]],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)


def test_skip_initial_space(all_parsers: Any) -> None:
    data = (
        '"09-Apr-2012", "01:10:18.300", 2456026.548822908, 12849, '
        "1.00361,  1.12551, 330.65659, 0355626618.16711,  73.48821, "
        "314.11625,  1917.09447,   179.71425,  80.000, 240.000, -350,  "
        "70.06056, 344.98370, 1,   1, -0.689265, -0.692787,  "
        "0.212036,    14.7674,   41.605,   -9999.0,   -9999.0,   "
        "-9999.0,   -9999.0,   -9999.0,  -9999.0, 000, 012, 128"
    )
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=list(range(33)),
                header=None,
                na_values=["-9999.0"],
                skipinitialspace=True,
            )
        return

    result = parser.read_csv(
        StringIO(data),
        names=list(range(33)),
        header=None,
        na_values=["-9999.0"],
        skipinitialspace=True,
    )
    expected = DataFrame(
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
    # see gh-2442
    data = """A,B,C
1,2,3,
4,5,6,
7,8,9,"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)

    expected = DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


def test_escapechar(all_parsers: Any) -> None:
    # https://stackoverflow.com/questions/13824840/feature-request-for-
    # pandas-read-csv
    data = '''SEARCH_TERM,ACTUAL_URL
"bra tv board","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"tv p\xc3\xa5 hjul","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"SLAGBORD, \\"Bergslagen\\", IKEA:s 1700-tals series","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"'''

    parser = all_parsers
    result = parser.read_csv(
        StringIO(data), escapechar="\\", quotechar='"', encoding="utf-8"
    )

    assert result["SEARCH_TERM"][2] == 'SLAGBORD, "Bergslagen", IKEA:s 1700-tals series'

    tm.assert_index_equal(result.columns, Index(["SEARCH_TERM", "ACTUAL_URL"]))


def test_ignore_leading_whitespace(all_parsers: Any) -> None:
    # see gh-3374, gh-6607
    parser = all_parsers
    data = " a b c\n 1 2 3\n 4 5 6\n 7 8 9"

    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+")
        return
    result = parser.read_csv(StringIO(data), sep=r"\s+")

    expected = DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("usecols", [None, [0, 1], ["a", "b"]])
def test_uneven_lines_with_usecols(all_parsers: Any, usecols: Any) -> None:
    # see gh-12203
    parser = all_parsers
    data = r"""a,b,c
0,1,2
3,4,5,6,7
8,9,10"""

    if usecols is None:
        # Make sure that an error is still raised
        # when the "usecols" parameter is not provided.
        msg = r"Expected \d+ fields in line \d+, saw \d+"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
    else:
        expected = DataFrame({"a": [0, 3, 8], "b": [1, 4, 9]})

        result = parser.read_csv(StringIO(data), usecols=usecols)
        tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # First, check to see that the response of parser when faced with no
        # provided columns raises the correct error, with or without usecols.
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
def test_read_empty_with_usecols(all_parsers: Any, data: str, kwargs: Dict[str