#!/usr/bin/env python3
import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
from typing import Any, Callable, Dict, Union, Type

import numpy as np
import pytest

from pandas.compat import is_platform_windows

import pandas as pd
import pandas._testing as tm

import pandas.io.common as icom


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_compression_size(
    obj: Union[pd.DataFrame, pd.Series],
    method: str,
    compression_only: Union[str, Dict[str, Any]],
) -> None:
    if compression_only == "tar":
        compression_only = {"method": "tar", "mode": "w:gz"}

    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression=compression_only)
        compressed_size: int = os.path.getsize(path)
        getattr(obj, method)(path, compression=None)
        uncompressed_size: int = os.path.getsize(path)
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_csv", "to_json"])
def test_compression_size_fh(
    obj: Union[pd.DataFrame, pd.Series],
    method: str,
    compression_only: Union[str, Dict[str, Any]],
) -> None:
    with tm.ensure_clean() as path:
        with icom.get_handle(
            path,
            "w:gz" if compression_only == "tar" else "w",
            compression=compression_only,
        ) as handles:
            getattr(obj, method)(handles.handle)
            assert not handles.handle.closed
        compressed_size: int = os.path.getsize(path)
    with tm.ensure_clean() as path:
        with icom.get_handle(path, "w", compression=None) as handles:
            getattr(obj, method)(handles.handle)
            assert not handles.handle.closed
        uncompressed_size: int = os.path.getsize(path)
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "write_method, write_kwargs, read_method",
    [
        ("to_csv", {"index": False}, pd.read_csv),
        ("to_json", {}, pd.read_json),
        ("to_pickle", {}, pd.read_pickle),
    ],
)
def test_dataframe_compression_defaults_to_infer(
    write_method: str,
    write_kwargs: Dict[str, Any],
    read_method: Callable[..., Any],
    compression_only: Union[str, Dict[str, Any]],
    compression_to_extension: Dict[str, str],
) -> None:
    # GH22004
    input_df: pd.DataFrame = pd.DataFrame([[1.0, 0, -4], [3.4, 5, 2]], columns=["X", "Y", "Z"])
    extension: str = compression_to_extension[compression_only]  # type: ignore
    with tm.ensure_clean("compressed" + extension) as path:
        getattr(input_df, write_method)(path, **write_kwargs)
        output_df: pd.DataFrame = read_method(path, compression=compression_only)
    tm.assert_frame_equal(output_df, input_df)


@pytest.mark.parametrize(
    "write_method,write_kwargs,read_method,read_kwargs",
    [
        ("to_csv", {"index": False, "header": True}, pd.read_csv, {"squeeze": True}),
        ("to_json", {}, pd.read_json, {"typ": "series"}),
        ("to_pickle", {}, pd.read_pickle, {}),
    ],
)
def test_series_compression_defaults_to_infer(
    write_method: str,
    write_kwargs: Dict[str, Any],
    read_method: Callable[..., Any],
    read_kwargs: Dict[str, Any],
    compression_only: Union[str, Dict[str, Any]],
    compression_to_extension: Dict[str, str],
) -> None:
    # GH22004
    input_series: pd.Series = pd.Series([0, 5, -2, 10], name="X")
    extension: str = compression_to_extension[compression_only]  # type: ignore
    with tm.ensure_clean("compressed" + extension) as path:
        getattr(input_series, write_method)(path, **write_kwargs)
        if "squeeze" in read_kwargs:
            kwargs: Dict[str, Any] = read_kwargs.copy()
            del kwargs["squeeze"]
            output_series: pd.Series = read_method(path, compression=compression_only, **kwargs).squeeze("columns")
        else:
            output_series = read_method(path, compression=compression_only, **read_kwargs)
    tm.assert_series_equal(output_series, input_series, check_names=False)


def test_compression_warning(
    compression_only: Union[str, Dict[str, Any]]
) -> None:
    # Assert that passing a file object to to_csv while explicitly specifying a
    # compression protocol triggers a RuntimeWarning, as per GH21227.
    df: pd.DataFrame = pd.DataFrame(
        100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        columns=["X", "Y", "Z"],
    )
    with tm.ensure_clean() as path:
        with icom.get_handle(path, "w", compression=compression_only) as handles:
            with tm.assert_produces_warning(RuntimeWarning, match="has no effect"):
                df.to_csv(handles.handle, compression=compression_only)


def test_compression_binary(
    compression_only: Union[str, Dict[str, Any]]
) -> None:
    """
    Binary file handles support compression.

    GH22555
    """
    df: pd.DataFrame = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )

    # with a file
    with tm.ensure_clean() as path:
        with open(path, mode="wb") as file:
            df.to_csv(file, mode="wb", compression=compression_only)
            file.seek(0)  # file shouldn't be closed
        tm.assert_frame_equal(
            df, pd.read_csv(path, index_col=0, compression=compression_only)
        )

    # with BytesIO
    file_obj: io.BytesIO = io.BytesIO()
    df.to_csv(file_obj, mode="wb", compression=compression_only)
    file_obj.seek(0)  # file shouldn't be closed
    tm.assert_frame_equal(
        df, pd.read_csv(file_obj, index_col=0, compression=compression_only)
    )


def test_gzip_reproducibility_file_name() -> None:
    """
    Gzip should create reproducible archives with mtime.

    Note: Archives created with different filenames will still be different!

    GH 28103
    """
    df: pd.DataFrame = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    compression_options: Dict[str, Any] = {"method": "gzip", "mtime": 1}

    # test for filename
    with tm.ensure_clean() as path:
        path_obj: Path = Path(path)
        df.to_csv(path_obj, compression=compression_options)
        time.sleep(0.1)
        output: bytes = path_obj.read_bytes()
        df.to_csv(path_obj, compression=compression_options)
        assert output == path_obj.read_bytes()


def test_gzip_reproducibility_file_object() -> None:
    """
    Gzip should create reproducible archives with mtime.

    GH 28103
    """
    df: pd.DataFrame = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    compression_options: Dict[str, Any] = {"method": "gzip", "mtime": 1}

    # test for file object
    buffer: io.BytesIO = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode="wb")
    output: bytes = buffer.getvalue()
    time.sleep(0.1)
    buffer = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode="wb")
    assert output == buffer.getvalue()


@pytest.mark.single_cpu
def test_with_missing_lzma() -> None:
    """Tests if import pandas works when lzma is not present."""
    code: str = textwrap.dedent(
        """\
        import sys
        sys.modules['lzma'] = None
        import pandas
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


@pytest.mark.single_cpu
def test_with_missing_lzma_runtime() -> None:
    """Tests if ModuleNotFoundError is hit when calling lzma without
    having the module available.
    """
    code: str = textwrap.dedent(
        """
        import sys
        import pytest
        sys.modules['lzma'] = None
        import pandas as pd
        df = pd.DataFrame()
        with pytest.raises(ModuleNotFoundError, match='import of lzma'):
            df.to_csv('foo.csv', compression='xz')
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_gzip_compression_level(
    obj: Union[pd.DataFrame, pd.Series],
    method: str,
) -> None:
    # GH33196
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression="gzip")
        compressed_size_default: int = os.path.getsize(path)
        getattr(obj, method)(path, compression={"method": "gzip", "compresslevel": 1})
        compressed_size_fast: int = os.path.getsize(path)
        assert compressed_size_default < compressed_size_fast


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_xz_compression_level_read(
    obj: Union[pd.DataFrame, pd.Series],
    method: str,
) -> None:
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression="xz")
        compressed_size_default: int = os.path.getsize(path)
        getattr(obj, method)(path, compression={"method": "xz", "preset": 1})
        compressed_size_fast: int = os.path.getsize(path)
        assert compressed_size_default < compressed_size_fast
        if method == "to_csv":
            pd.read_csv(path, compression="xz")


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_bzip_compression_level(
    obj: Union[pd.DataFrame, pd.Series],
    method: str,
) -> None:
    """GH33196 bzip needs file size > 100k to show a size difference between
    compression levels, so here we just check if the call works when
    compression is passed as a dict.
    """
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression={"method": "bz2", "compresslevel": 1})


@pytest.mark.parametrize(
    "suffix,archive",
    [
        (".zip", zipfile.ZipFile),
        (".tar", tarfile.TarFile),
    ],
)
def test_empty_archive_zip(
    suffix: str,
    archive: Type[Any],
) -> None:
    with tm.ensure_clean(filename=suffix) as path:
        with archive(path, "w"):
            pass
        with pytest.raises(ValueError, match="Zero files found"):
            pd.read_csv(path)


def test_ambiguous_archive_zip() -> None:
    with tm.ensure_clean(filename=".zip") as path:
        with zipfile.ZipFile(path, "w") as file:
            file.writestr("a.csv", "foo,bar")
            file.writestr("b.csv", "foo,bar")
        with pytest.raises(ValueError, match="Multiple files found in ZIP file"):
            pd.read_csv(path)


def test_ambiguous_archive_tar(tmp_path: Path) -> None:
    csvAPath: Path = tmp_path / "a.csv"
    with open(csvAPath, "w", encoding="utf-8") as a:
        a.write("foo,bar\n")
    csvBPath: Path = tmp_path / "b.csv"
    with open(csvBPath, "w", encoding="utf-8") as b:
        b.write("foo,bar\n")

    tarpath: Path = tmp_path / "archive.tar"
    with tarfile.TarFile(tarpath, "w") as tar:
        tar.add(csvAPath, arcname="a.csv")
        tar.add(csvBPath, arcname="b.csv")

    with pytest.raises(ValueError, match="Multiple files found in TAR archive"):
        pd.read_csv(tarpath)


def test_tar_gz_to_different_filename() -> None:
    with tm.ensure_clean(filename=".foo") as file:
        pd.DataFrame(
            [["1", "2"]],
            columns=["foo", "bar"],
        ).to_csv(file, compression={"method": "tar", "mode": "w:gz"}, index=False)
        with gzip.open(file) as uncompressed:
            with tarfile.TarFile(fileobj=uncompressed) as archive:
                members = archive.getmembers()
                assert len(members) == 1
                content: str = archive.extractfile(members[0]).read().decode("utf8")

                if is_platform_windows():
                    expected: str = "foo,bar\r\n1,2\r\n"
                else:
                    expected = "foo,bar\n1,2\n"

                assert content == expected


def test_tar_no_error_on_close() -> None:
    with io.BytesIO() as buffer:
        with icom._BytesTarFile(fileobj=buffer, mode="w"):
            pass
