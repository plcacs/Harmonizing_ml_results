#!/usr/bin/env python3
"""
Tests encoding functionality during parsing
for all of the parsers defined in parsers.py
"""

from io import (
    BytesIO,
    TextIOWrapper,
)
import os
import tempfile
import uuid
from typing import Any, Callable, Iterator, Tuple

import numpy as np
import pytest

from pandas import (
    DataFrame,
    read_csv,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_bytes_io_input(all_parsers: Any) -> None:
    encoding: str = "cp1255"
    parser: Any = all_parsers

    data: BytesIO = BytesIO("שלום:1234\n562:123".encode(encoding))
    result: DataFrame = parser.read_csv(data, sep=":", encoding=encoding)

    expected: DataFrame = DataFrame([[562, 123]], columns=["שלום", "1234"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_read_csv_unicode(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: BytesIO = BytesIO("\u0141aski, Jan;1".encode())

    result: DataFrame = parser.read_csv(data, sep=";", encoding="utf-8", header=None)
    expected: DataFrame = DataFrame([["\u0141aski, Jan", 1]])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("sep", [",", "\t"])
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16le", "utf-16be"])
def test_utf16_bom_skiprows(all_parsers: Any, sep: str, encoding: str) -> None:
    # see gh-2298
    parser: Any = all_parsers
    data: str = """skip this
skip this too
A,B,C
1,2,3
4,5,6""".replace(",", sep)
    path: str = f"__{uuid.uuid4()}__.csv"
    kwargs: dict = {"sep": sep, "skiprows": 2}
    utf8: str = "utf-8"

    with tm.ensure_clean(path) as path_clean:
        bytes_data: bytes = data.encode(encoding)

        with open(path_clean, "wb") as f:
            f.write(bytes_data)

        with TextIOWrapper(BytesIO(data.encode(utf8)), encoding=utf8) as bytes_buffer:
            result: DataFrame = parser.read_csv(path_clean, encoding=encoding, **kwargs)
            expected: DataFrame = parser.read_csv(bytes_buffer, encoding=utf8, **kwargs)
        tm.assert_frame_equal(result, expected)


def test_utf16_example(all_parsers: Any, csv_dir_path: str) -> None:
    path: str = os.path.join(csv_dir_path, "utf16_ex.txt")
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(path, encoding="utf-16", sep="\t")
    assert len(result) == 50


def test_unicode_encoding(all_parsers: Any, csv_dir_path: str) -> None:
    path: str = os.path.join(csv_dir_path, "unicode_series.csv")
    parser: Any = all_parsers

    result: DataFrame = parser.read_csv(path, header=None, encoding="latin-1")
    result = result.set_index(0)
    got: Any = result[1][1632]

    expected: str = "\xc1 k\xf6ldum klaka (Cold Fever) (1994)"
    assert got == expected


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # Basic test
        ("a\n1", {}, [1]),
        # "Regular" quoting
        ('"a"\n1', {"quotechar": '"'}, [1]),
        # Test in a data row instead of header
        ("b\n1", {"names": ["a"]}, ["b", "1"]),
        # Test in empty data row with skipping
        ("\n1", {"names": ["a"], "skip_blank_lines": True}, [1]),
        # Test in empty data row without skipping
        (
            "\n1",
            {"names": ["a"], "skip_blank_lines": False},
            [np.nan, 1],
        ),
    ],
)
def test_utf8_bom(all_parsers: Any, data: str, kwargs: dict, expected: list) -> None:
    # see gh-4793
    parser: Any = all_parsers
    bom: str = "\ufeff"
    utf8: str = "utf-8"

    def _encode_data_with_bom(_data: str) -> BytesIO:
        bom_data: bytes = (bom + _data).encode(utf8)
        return BytesIO(bom_data)

    if (
        parser.engine == "pyarrow"
        and data == "\n1"
        and kwargs.get("skip_blank_lines", True)
    ):
        # CSV parse error: Empty CSV file or block: cannot infer number of columns
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    result: DataFrame = parser.read_csv(_encode_data_with_bom(data), encoding=utf8, **kwargs)
    expected_df: DataFrame = DataFrame({"a": expected})
    tm.assert_frame_equal(result, expected_df)


def test_readcsv_utf_aliases(all_parsers: Any, utf_value: int, encoding_fmt: str) -> None:
    # see gh-13549
    expected: DataFrame = DataFrame({"mb_num": [4.8], "multibyte": ["test"]})
    parser: Any = all_parsers

    encoding: str = encoding_fmt.format(utf_value)
    data: bytes = "mb_num,multibyte\n4.8,test".encode(encoding)

    result: DataFrame = parser.read_csv(BytesIO(data), encoding=encoding)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "file_path,encoding",
    [
        (("io", "data", "csv", "test1.csv"), "utf-8"),
        (("io", "parser", "data", "unicode_series.csv"), "latin-1"),
        (("io", "parser", "data", "sauron.SHIFT_JIS.csv"), "shiftjis"),
    ],
)
def test_binary_mode_file_buffers(
    all_parsers: Any, file_path: Tuple[str, ...], encoding: str, datapath: Callable[..., str]
) -> None:
    # gh-23779: Python csv engine shouldn't error on files opened in binary.
    # gh-31575: Python csv engine shouldn't error on files opened in raw binary.
    parser: Any = all_parsers

    fpath: str = datapath(*file_path)
    expected: DataFrame = parser.read_csv(fpath, encoding=encoding)

    with open(fpath, encoding=encoding) as fa:
        result: DataFrame = parser.read_csv(fa)
        assert not fa.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb") as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb", buffering=0) as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("pass_encoding", [True, False])
def test_encoding_temp_file(
    all_parsers: Any, utf_value: int, encoding_fmt: str, pass_encoding: bool, temp_file: Any
) -> None:
    # see gh-24130
    parser: Any = all_parsers
    encoding: str = encoding_fmt.format(utf_value)

    if parser.engine == "pyarrow" and pass_encoding is True and utf_value in [16, 32]:
        # FIXME: this is bad!
        pytest.skip("These cases freeze")

    expected: DataFrame = DataFrame({"foo": ["bar"]})

    with temp_file.open(mode="w+", encoding=encoding) as f:
        f.write("foo\nbar")
        f.seek(0)

        result: DataFrame = parser.read_csv(f, encoding=encoding if pass_encoding else None)
        tm.assert_frame_equal(result, expected)


def test_encoding_named_temp_file(all_parsers: Any) -> None:
    # see gh-31819
    parser: Any = all_parsers
    encoding: str = "shift-jis"

    title: str = "てすと"
    data: str = "こむ"

    expected: DataFrame = DataFrame({title: [data]})

    with tempfile.NamedTemporaryFile() as f:
        f.write(f"{title}\n{data}".encode(encoding))
        f.seek(0)
        result: DataFrame = parser.read_csv(f, encoding=encoding)
        tm.assert_frame_equal(result, expected)
        assert not f.closed


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-16", "utf-16-be", "utf-16-le", "utf-32"]
)
def test_parse_encoded_special_characters(encoding: str) -> None:
    # GH16218 Verify parsing of data with encoded special characters
    # Data contains a Unicode 'FULLWIDTH COLON' (U+FF1A) at position (0,"a")
    data: str = "a\tb\n：foo\t0\nbar\t1\nbaz\t2"  # noqa: RUF001
    encoded_data: BytesIO = BytesIO(data.encode(encoding))
    result: DataFrame = read_csv(encoded_data, delimiter="\t", encoding=encoding)

    expected: DataFrame = DataFrame(
        data=[["：foo", 0], ["bar", 1], ["baz", 2]],  # noqa: RUF001
        columns=["a", "b"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", ["utf-8", None, "utf-16", "cp1255", "latin-1"])
def test_encoding_memory_map(all_parsers: Any, encoding: Any) -> None:
    # GH40986
    parser: Any = all_parsers
    expected: DataFrame = DataFrame(
        {
            "name": ["Raphael", "Donatello", "Miguel Angel", "Leonardo"],
            "mask": ["red", "purple", "orange", "blue"],
            "weapon": ["sai", "bo staff", "nunchunk", "katana"],
        }
    )
    with tm.ensure_clean() as file:
        expected.to_csv(file, index=False, encoding=encoding)

        if parser.engine == "pyarrow":
            msg: str = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(file, encoding=encoding, memory_map=True)
            return

        df: DataFrame = parser.read_csv(file, encoding=encoding, memory_map=True)
    tm.assert_frame_equal(df, expected)


def test_chunk_splits_multibyte_char(all_parsers: Any) -> None:
    """
    Chunk splits a multibyte character with memory_map=True

    GH 43540
    """
    parser: Any = all_parsers
    # DEFAULT_CHUNKSIZE = 262144, defined in parsers.pyx
    df: DataFrame = DataFrame(data=["a" * 127] * 2048)

    # Put two-bytes utf-8 encoded character "ą" at the end of chunk
    # utf-8 encoding of "ą" is b'\xc4\x85'
    df.iloc[2047] = "a" * 127 + "ą"
    with tm.ensure_clean("bug-gh43540.csv") as fname:
        df.to_csv(fname, index=False, header=False, encoding="utf-8")

        if parser.engine == "pyarrow":
            msg: str = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(fname, header=None, memory_map=True)
            return

        dfr: DataFrame = parser.read_csv(fname, header=None, memory_map=True)
    tm.assert_frame_equal(dfr, df)


def test_readcsv_memmap_utf8(all_parsers: Any) -> None:
    """
    GH 43787

    Test correct handling of UTF-8 chars when memory_map=True and encoding is UTF-8
    """
    lines: list[str] = []
    line_length: int = 128
    start_char: str = " "
    end_char: str = "\U00010080"
    # This for loop creates a list of 128-char strings
    # consisting of consecutive Unicode chars
    for lnum in range(ord(start_char), ord(end_char), line_length):
        line: str = "".join([chr(c) for c in range(lnum, lnum + 0x80)]) + "\n"
        try:
            line.encode("utf-8")
        except UnicodeEncodeError:
            continue
        lines.append(line)
    parser: Any = all_parsers
    df: DataFrame = DataFrame(lines)
    with tm.ensure_clean("utf8test.csv") as fname:
        df.to_csv(fname, index=False, header=False, encoding="utf-8")

        if parser.engine == "pyarrow":
            msg: str = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(fname, header=None, memory_map=True, encoding="utf-8")
            return

        dfr: DataFrame = parser.read_csv(fname, header=None, memory_map=True, encoding="utf-8")
    tm.assert_frame_equal(df, dfr)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("mode", ["w+b", "w+t"])
def test_not_readable(all_parsers: Any, mode: str) -> None:
    # GH43439
    parser: Any = all_parsers
    content: Any = b"abcd"
    if "t" in mode:
        content = "abcd"
    with tempfile.SpooledTemporaryFile(mode=mode, encoding="utf-8") as handle:
        handle.write(content)
        handle.seek(0)
        df: DataFrame = parser.read_csv(handle)
    expected: DataFrame = DataFrame([], columns=["abcd"])
    tm.assert_frame_equal(df, expected)