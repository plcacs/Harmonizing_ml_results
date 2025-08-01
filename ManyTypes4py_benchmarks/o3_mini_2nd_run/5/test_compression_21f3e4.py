import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
from typing import Any, Tuple, Optional, Dict, Union

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

@pytest.fixture(params=[True, False])
def buffer(request: pytest.FixtureRequest) -> bool:
    return request.param

@pytest.fixture
def parser_and_data(all_parsers: Any, csv1: str) -> Tuple[Any, bytes, DataFrame]:
    parser = all_parsers
    with open(csv1, "rb") as f:
        data: bytes = f.read()
    expected: DataFrame = parser.read_csv(csv1)
    return (parser, data, expected)

@pytest.mark.parametrize("compression", ["zip", "infer", "zip2"])
def test_zip(parser_and_data: Tuple[Any, bytes, DataFrame], compression: str) -> None:
    parser, data, expected = parser_and_data
    with tm.ensure_clean("test_file.zip") as path:
        with zipfile.ZipFile(path, mode="w") as tmp:
            tmp.writestr("test_file", data)
        if compression == "zip2":
            with open(path, "rb") as f:
                result: DataFrame = parser.read_csv(f, compression="zip")
        else:
            result = parser.read_csv(path, compression=compression)
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize("compression", ["zip", "infer"])
def test_zip_error_multiple_files(parser_and_data: Tuple[Any, bytes, DataFrame], compression: str) -> None:
    parser, data, _ = parser_and_data
    with tm.ensure_clean("combined_zip.zip") as path:
        inner_file_names = ["test_file", "second_file"]
        with zipfile.ZipFile(path, mode="w") as tmp:
            for file_name in inner_file_names:
                tmp.writestr(file_name, data)
        with pytest.raises(ValueError, match="Multiple files"):
            parser.read_csv(path, compression=compression)

def test_zip_error_no_files(parser_and_data: Tuple[Any, bytes, DataFrame]) -> None:
    parser, _, _ = parser_and_data
    with tm.ensure_clean() as path:
        with zipfile.ZipFile(path, mode="w"):
            pass
        with pytest.raises(ValueError, match="Zero files"):
            parser.read_csv(path, compression="zip")

def test_zip_error_invalid_zip(parser_and_data: Tuple[Any, bytes, DataFrame]) -> None:
    parser, _, _ = parser_and_data
    with tm.ensure_clean() as path:
        with open(path, "rb") as f:
            with pytest.raises(zipfile.BadZipFile, match="File is not a zip file"):
                parser.read_csv(f, compression="zip")

@pytest.mark.parametrize("filename", [None, "test.{ext}"])
def test_compression(
    request: pytest.FixtureRequest,
    parser_and_data: Tuple[Any, bytes, DataFrame],
    compression_only: str,
    buffer: bool,
    filename: Optional[str],
    compression_to_extension: Dict[Any, str],
) -> None:
    parser, data, expected = parser_and_data
    compress_type: str = compression_only
    ext: str = compression_to_extension[compress_type]
    filename = filename if filename is None else filename.format(ext=ext)
    if filename and buffer:
        request.applymarker(
            pytest.mark.xfail(
                reason="Cannot deduce compression from buffer of compressed data."
            )
        )
    with tm.ensure_clean(filename=filename) as path:
        tm.write_to_compressed(compress_type, path, data)
        compression: str = "infer" if filename else compress_type
        if buffer:
            with open(path, "rb") as f:
                result: DataFrame = parser.read_csv(f, compression=compression)
        else:
            result = parser.read_csv(path, compression=compression)
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize("ext", [None, "gz", "bz2"])
def test_infer_compression(
    all_parsers: Any, csv1: str, buffer: bool, ext: Optional[str]
) -> None:
    parser = all_parsers
    kwargs: Dict[str, Any] = {"index_col": 0, "parse_dates": True}
    expected: DataFrame = parser.read_csv(csv1, **kwargs)
    kwargs["compression"] = "infer"
    if buffer:
        with open(csv1, encoding="utf-8") as f:
            result: DataFrame = parser.read_csv(f, **kwargs)
    else:
        ext_str: str = "." + ext if ext else ""
        result = parser.read_csv(csv1 + ext_str, **kwargs)
    tm.assert_frame_equal(result, expected)

def test_compression_utf_encoding(
    all_parsers: Any, csv_dir_path: str, utf_value: Union[str, int], encoding_fmt: str
) -> None:
    parser = all_parsers
    encoding: str = encoding_fmt.format(utf_value)
    path: str = os.path.join(csv_dir_path, f"utf{utf_value}_ex_small.zip")
    result: DataFrame = parser.read_csv(path, encoding=encoding, compression="zip", sep="\t")
    expected: DataFrame = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize("invalid_compression", ["sfark", "bz3", "zipper"])
def test_invalid_compression(all_parsers: Any, invalid_compression: str) -> None:
    parser = all_parsers
    compress_kwargs: Dict[str, Any] = {"compression": invalid_compression}
    msg: str = f"Unrecognized compression type: {invalid_compression}"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv("test_file.zip", **compress_kwargs)

def test_compression_tar_archive(all_parsers: Any, csv_dir_path: str) -> None:
    parser = all_parsers
    path: str = os.path.join(csv_dir_path, "tar_csv.tar.gz")
    df: DataFrame = parser.read_csv(path)
    assert list(df.columns) == ["a"]

def test_ignore_compression_extension(all_parsers: Any) -> None:
    parser = all_parsers
    df: DataFrame = DataFrame({"a": [0, 1]})
    with tm.ensure_clean("test.csv") as path_csv:
        with tm.ensure_clean("test.csv.zip") as path_zip:
            df.to_csv(path_csv, index=False)
            Path(path_zip).write_text(
                Path(path_csv).read_text(encoding="utf-8"), encoding="utf-8"
            )
            tm.assert_frame_equal(parser.read_csv(path_zip, compression=None), df)

def test_writes_tar_gz(all_parsers: Any) -> None:
    parser = all_parsers
    data: DataFrame = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )
    with tm.ensure_clean("test.tar.gz") as tar_path:
        data.to_csv(tar_path, index=False)
        tm.assert_frame_equal(parser.read_csv(tar_path), data)
        with tarfile.open(tar_path, "r:gz") as tar:
            result: DataFrame = parser.read_csv(tar.extractfile(tar.getnames()[0]), compression="infer")
            tm.assert_frame_equal(result, data)