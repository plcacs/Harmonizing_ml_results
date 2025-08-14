"""
Tests that NA values are properly handled during
parsing for all of the parsers defined in parsers.py
"""

from io import StringIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest

from pandas._libs.parsers import STR_NA_VALUES

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_string_nas(all_parsers: Any) -> None:
    parser = all_parsers
    data = """A,B,C
a,b,c
d,,f
,g,h
"""
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [["a", "b", "c"], ["d", np.nan, "f"], [np.nan, "g", "h"]],
        columns=["A", "B", "C"],
    )
    if parser.engine == "pyarrow":
        expected.loc[2, "A"] = None
        expected.loc[1, "B"] = None
    tm.assert_frame_equal(result, expected)


def test_detect_string_na(all_parsers: Any) -> None:
    parser = all_parsers
    data = """A,B
foo,bar
NA,baz
NaN,nan
"""
    expected = DataFrame(
        [["foo", "bar"], [np.nan, "baz"], [np.nan, np.nan]], columns=["A", "B"]
    )
    if parser.engine == "pyarrow":
        expected.loc[[1, 2], "A"] = None
        expected.loc[2, "B"] = None
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_values",
    [
        ["-999.0", "-999"],
        [-999, -999.0],
        [-999.0, -999],
        ["-999.0"],
        ["-999"],
        [-999.0],
        [-999],
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        """A,B
-999,1.2
2,-999
3,4.5
""",
        """A,B
-999,1.200
2,-999.000
3,4.500
""",
    ],
)
def test_non_string_na_values(
    all_parsers: Any, data: str, na_values: Any, request: Any
) -> None:
    # see gh-3611: with an odd float format, we can't match
    # the string "999.0" exactly but still need float matching
    parser = all_parsers
    expected = DataFrame([[np.nan, 1.2], [2.0, np.nan], [3.0, 4.5]], columns=["A", "B"])

    if parser.engine == "pyarrow" and not all(isinstance(x, str) for x in na_values):
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    elif parser.engine == "pyarrow" and "-999.000" in data:
        # bc the pyarrow engine does not include the float-ified version
        #  of "-999" -> -999, it does not match the entry with the trailing
        #  zeros, so "-999.000" is not treated as null.
        mark = pytest.mark.xfail(
            reason="pyarrow engined does not recognize equivalent floats"
        )
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), na_values=na_values)
    tm.assert_frame_equal(result, expected)


def test_default_na_values(all_parsers: Any) -> None:
    _NA_VALUES = {
        "-1.#IND",
        "1.#QNAN",
        "1.#IND",
        "-1.#QNAN",
        "#N/A",
        "N/A",
        "n/a",
        "NA",
        "<NA>",
        "#NA",
        "NULL",
        "null",
        "NaN",
        "nan",
        "-NaN",
        "-nan",
        "#N/A N/A",
        "",
        "None",
    }
    assert _NA_VALUES == STR_NA_VALUES

    parser = all_parsers
    nv = len(_NA_VALUES)

    def f(i: int, v: str) -> str:
        if i == 0:
            buf = ""
        elif i > 0:
            buf = "".join([","] * i)

        buf = f"{buf}{v}"

        if i < nv - 1:
            joined = "".join([","] * (nv - i - 1))
            buf = f"{buf}{joined}"

        return buf

    data = StringIO("\n".join([f(i, v) for i, v in enumerate(_NA_VALUES)]))
    expected = DataFrame(np.nan, columns=range(nv), index=range(nv))

    result = parser.read_csv(data, header=None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("na_values", ["baz", ["baz"]])
def test_custom_na_values(all_parsers: Any, na_values: Any) -> None:
    parser = all_parsers
    data = """A,B,C
ignore,this,row
1,NA,3
-1.#IND,5,baz
7,8,NaN
"""
    expected = DataFrame(
        [[1.0, np.nan, 3], [np.nan, 5, np.nan], [7, 8, np.nan]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "skiprows argument must be an integer when using engine='pyarrow'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
        return

    result = parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
    tm.assert_frame_equal(result, expected)


def test_bool_na_values(all_parsers: Any) -> None:
    data = """A,B,C
True,False,True
NA,True,False
False,NA,True"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        {
            "A": np.array([True, np.nan, False], dtype=object),
            "B": np.array([False, True, np.nan], dtype=object),
            "C": [True, False, True],
        }
    )
    if parser.engine == "pyarrow":
        expected.loc[1, "A"] = None
        expected.loc[2, "B"] = None
    tm.assert_frame_equal(result, expected)


def test_na_value_dict(all_parsers: Any) -> None:
    data = """A,B,C
foo,bar,NA
bar,foo,foo
foo,bar,NA
bar,foo,foo"""
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
        return

    df = parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
    expected = DataFrame(
        {
            "A": [np.nan, "bar", np.nan, "bar"],
            "B": [np.nan, "foo", np.nan, "foo"],
            "C": [np.nan, "foo", np.nan, "foo"],
        }
    )
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "index_col,expected",
    [
        (
            [0],
            DataFrame({"b": [np.nan], "c": [1], "d": [5]}, index=Index([0], name="a")),
        ),
        (
            [0, 2],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
        (
            ["a", "c"],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
    ],
)
def test_na_value_dict_multi_index(
    all_parsers: Any, index_col: Any, expected: DataFrame
) -> None:
    data = """\
a,b,c,d
0,NA,1,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), na_values=set(), index_col=index_col)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {},
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
            },
        ),
        (
            {"na_values": {"A": [], "C": []}, "keep_default_na": False},
            {
                "A": ["a", "b", "", "d", "e", "nan", "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", "nan", "five", "", "seven"],
            },
        ),
        (
            {"na_values": ["a"], "keep_default_na": False},
            {
                "A": [np.nan, "b", "", "d", "e", "nan", "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", "nan", "five", "", "seven"],
            },
        ),
        (
            {"na_values": {"A": [], "C": []}},
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
            },
        ),
    ],
)
def test_na_values_keep_default(
    all_parsers: Any,
    kwargs: Dict[str, Any],
    expected: Dict[str, List[Any]],
    request: Any,
    using_infer_string: bool,
) -> None:
    data = """\
A,B,C
a,1,one
b,2,two
,3,three
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    parser = all_parsers
    if parser.engine == "pyarrow":
        if "na_values" in kwargs and isinstance(kwargs["na_values"], dict):
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), **kwargs)
            return
        if not using_infer_string or "na_values" in kwargs:
            mark = pytest.mark.xfail()
            request.applymarker(mark)

    result = parser.read_csv(StringIO(data), **kwargs)
    expected_df = DataFrame(expected)
    tm.assert_frame_equal(result, expected_df)


def test_no_na_values_no_keep_default(all_parsers: Any) -> None:
    # see gh-4318: passing na_values=None and
    # keep_default_na=False yields 'None" as a na_value
    data = """\
A,B,C
a,1,None
b,2,two
,3,None
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), keep_default_na=False)

    expected = DataFrame(
        {
            "A": ["a", "b", "", "d", "e", "nan", "g"],
            "B": [1, 2, 3, 4, 5, 6, 7],
            "C": ["None", "two", "None", "nan", "five", "", "seven"],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_values(all_parsers: Any) -> None:
    # see gh-19227
    data = "a,b\n,2"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
            )
        return

    result = parser.read_csv(
        StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
    )
    expected = DataFrame({"a": [""], "b": [np.nan]})
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_scalar_values(all_parsers: Any) -> None:
    # see gh-19227
    #
    # Scalar values shouldn't cause the parsing to crash or fail.
    data = "a,b\n1,2"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
        return

    df = parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
    expected = DataFrame({"a": [1], "b": [np.nan]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("col_zero_na_values", [113125, "113125"])
def test_no_keep_default_na_dict_na_values_diff_reprs(
    all_parsers: Any, col_zero_na_values: Any
) -> None:
    # see gh-19227
    data = """\
113125,"blah","/blaha",kjsdkj,412.166,225.874,214.008
729639,"qwer","",asdfkj,466.681,,252.373
"""
    parser = all_parsers
    expected = DataFrame(
        {
            0: [np.nan, 729639.0],
            1: [np.nan, "qwer"],
            2: ["/blaha", np.nan],
            3: ["kjsdkj", "asdfkj"],
            4: [412.166, 466.681],
            5: ["225.874", ""],
            6: [np.nan, 252.373],
        }
    )

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=None,
                keep_default_na=False,
                na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
            )
        return

    result = parser.read_csv(
        StringIO(data),
        header=None,
        keep_default_na=False,
        na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_filter,row_data",
    [
        (True, [[1, "A"], [np.nan, np.nan], [3, "C"]]),
        (False, [["1", "A"], ["nan", "B"], ["3", "C"]]),
    ],
)
def test_na_values_na_filter_override(
    request: Any,
    all_parsers: Any,
    na_filter: bool,
    row_data: List[List[Any]],
    using_infer_string: bool,
) -> None:
    parser = all_parsers
    if parser.engine == "pyarrow":
        # mismatched dtypes in both cases, FutureWarning in the True case
        if not (using_infer_string and na_filter):
            mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
            request.applymarker(mark)
    data = """\
A,B
1,A
nan,B
3,C
"""
    result = parser.read_csv(StringIO(data), na_values=["B"], na_filter=na_filter)

    expected = DataFrame(row_data, columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 8 columns, got 5:
def test_na_trailing_columns(all_parsers: Any) -> None:
    parser = all_parsers
    data = """Date,Currency,Symbol,Type,Units,UnitPrice,Cost,Tax
2012-03-14,USD,AAPL,BUY,1000
2012-05-12,USD,SBUX,SELL,500"""

    # Trailing columns should be all NaN.
    result = parser.read_csv(StringIO(data))
    expected