"""
Tests for the pandas.io.common functionalities
"""

import codecs
import errno
from functools import partial
from io import (
    BytesIO,
    StringIO,
    UnsupportedOperation,
)
import mmap
import os
from pathlib import Path
import pickle
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    IO,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import pytest

from pandas.compat import (
    WASM,
    is_platform_windows,
)
from pandas.compat.pyarrow import pa_version_under19p0
import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

import pandas.io.common as icom

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


class CustomFSPath:
    """For testing fspath on unknown objects"""

    def __init__(self, path: str) -> None:
        self.path = path

    def __fspath__(self) -> str:
        return self.path


HERE: str = os.path.abspath(os.path.dirname(__file__))


# https://github.com/cython/cython/issues/1720
class TestCommonIOCapabilities:
    data1: str = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    def test_expand_user(self) -> None:
        filename: str = "~/sometest"
        expanded_name: str = icom._expand_user(filename)

        assert expanded_name != filename
        assert os.path.isabs(expanded_name)
        assert os.path.expanduser(filename) == expanded_name

    def test_expand_user_normal_path(self) -> None:
        filename: str = "/somefolder/sometest"
        expanded_name: str = icom._expand_user(filename)

        assert expanded_name == filename
        assert os.path.expanduser(filename) == expanded_name

    def test_stringify_path_pathlib(self) -> None:
        rel_path: str = icom.stringify_path(Path("."))
        assert rel_path == "."
        redundant_path: str = icom.stringify_path(Path("foo//bar"))
        assert redundant_path == os.path.join("foo", "bar")

    def test_stringify_path_fspath(self) -> None:
        p: CustomFSPath = CustomFSPath("foo/bar.csv")
        result: str = icom.stringify_path(p)
        assert result == "foo/bar.csv"

    def test_stringify_file_and_path_like(self) -> None:
        # GH 38125: do not stringify file objects that are also path-like
        fsspec = pytest.importorskip("fsspec")
        with tm.ensure_clean() as path:
            with fsspec.open(f"file://{path}", mode="wb") as fsspec_obj:
                assert fsspec_obj == icom.stringify_path(fsspec_obj)

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    def test_infer_compression_from_path(self, compression_format: Tuple[str, str], path_type: Type[Union[str, CustomFSPath, Path]]) -> None:
        extension, expected = compression_format
        path = path_type("foo/bar.csv" + extension)
        compression: Optional[str] = icom.infer_compression(path, compression="infer")
        assert compression == expected

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type: Type[Union[str, CustomFSPath, Path]]) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            filename = path_type("~/" + Path(tmp).name + "/sometest")
            with icom.get_handle(filename, "w") as handles:
                assert Path(handles.handle.name).is_absolute()
                assert os.path.expanduser(filename) == handles.handle.name

    def test_get_handle_with_buffer(self) -> None:
        with StringIO() as input_buffer:
            with icom.get_handle(input_buffer, "r") as handles:
                assert handles.handle == input_buffer
            assert not input_buffer.closed
        assert input_buffer.closed

    # Test that BytesIOWrapper(get_handle) returns correct amount of bytes every time
    def test_bytesiowrapper_returns_correct_bytes(self) -> None:
        # Test latin1, ucs-2, and ucs-4 chars
        data: str = """a,b,c
1,2,3
©,®,®
Look,a snake,🐍"""
        with icom.get_handle(StringIO(data), "rb", is_text=False) as handles:
            result: bytes = b""
            chunksize: int = 5
            while True:
                chunk: bytes = handles.handle.read(chunksize)
                # Make sure each chunk is correct amount of bytes
                assert len(chunk) <= chunksize
                if len(chunk) < chunksize:
                    # Can be less amount of bytes, but only at EOF
                    # which happens when read returns empty
                    assert len(handles.handle.read()) == 0
                    result += chunk
                    break
                result += chunk
            assert result == data.encode("utf-8")

    # Test that pyarrow can handle a file opened with get_handle
    def test_get_handle_pyarrow_compat(self) -> None:
        pa_csv = pytest.importorskip("pyarrow.csv")

        # Test latin1, ucs-2, and ucs-4 chars
        data: str = """a,b,c
1,2,3
©,®,®
Look,a snake,🐍"""
        expected: pd.DataFrame = pd.DataFrame(
            {"a": ["1", "©", "Look"], "b": ["2", "®", "a snake"], "c": ["3", "®", "🐍"]}
        )
        s: StringIO = StringIO(data)
        with icom.get_handle(s, "rb", is_text=False) as handles:
            df: pd.DataFrame = pa_csv.read_csv(handles.handle).to_pandas()
            if pa_version_under19p0:
                expected = expected.astype("object")
            tm.assert_frame_equal(df, expected)
            assert not s.closed

    def test_iterator(self) -> None:
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result: pd.DataFrame = pd.concat(reader, ignore_index=True)
        expected: pd.DataFrame = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)

        # GH12153
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first: pd.DataFrame = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", OSError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", FileNotFoundError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_non_existent(self, reader: Callable, module: str, error_class: Type[Exception], fn_ext: str) -> None:
        pytest.importorskip(module)

        path: str = os.path.join(HERE, "data", "does_not_exist." + fn_ext)
        msg1: str = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2: str = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3: str = "Expected object or value"
        msg4: str = "path_or_buf needs to be a string file path or file-like"
        msg5: str = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6: str = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7: str = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8: str = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "method, module, error_class, fn_ext",
        [
            (pd.DataFrame.to_csv, "os", OSError, "csv"),
            (pd.DataFrame.to_html, "os", OSError, "html"),
            (pd.DataFrame.to_excel, "xlrd", OSError, "xlsx"),
            (pd.DataFrame.to_feather, "pyarrow", OSError, "feather"),
            (pd.DataFrame.to_parquet, "pyarrow", OSError, "parquet"),
            (pd.DataFrame.to_stata, "os", OSError, "dta"),
            (pd.DataFrame.to_json, "os", OSError, "json"),
            (pd.DataFrame.to_pickle, "os", OSError, "pickle"),
        ],
    )
    # NOTE: Missing parent directory for pd.DataFrame.to_hdf is handled by PyTables
    def test_write_missing_parent_directory(self, method: Callable, module: str, error_class: Type[Exception], fn_ext: str) -> None:
        pytest.importorskip(module)

        dummy_frame: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5]})

        path: str = os.path.join(HERE, "data", "missing_folder", "does_not_exist." + fn_ext)

        with pytest.raises(
            error_class,
            match=r"Cannot save file into a non-existent directory: .*missing_folder",
        ):
            method(dummy_frame, path)

    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_table, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", OSError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", FileNotFoundError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_expands_user_home_dir(
        self, reader: Callable, module: str, error_class: Type[Exception], fn_ext: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip(module)

        path: str = os.path.join("~", "does_not_exist." + fn_ext)
        monkeypatch.setattr(icom, "_expand_user", lambda x: os.path.join("foo", x))

        msg1: str = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2: str = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3: str = "Unexpected character found when decoding 'false'"
        msg4: str = "path_or_buf needs to be a string file path or file-like"
        msg5: str = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6: str = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7: str = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8: str = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "reader, module, path",
        [
            (pd.read_csv, "os", ("io", "data", "csv", "iris.csv")),
            (pd.read_table, "os", ("io", "data", "csv", "iris.csv")),
            (
                pd.read_fwf,
                "os",
                ("io", "data", "fixed_width", "fixed_width_format.txt"),
            ),
            (pd.read_excel, "xlrd", ("io", "data", "excel", "test1.xlsx")),
            (
                pd.read_feather,
                "pyarrow",
                ("io", "data", "feather", "feather-0_3_1.feather"),
            ),
            (
                pd.read_hdf,
                "tables",
                ("io", "data", "legacy_hdf", "pytables_native2.h5"),
            ),
            (pd.read_stata, "os", ("io", "data", "stata", "stata10_115.dta")),
            (pd.read_sas, "os", ("io", "sas", "data", "test1.sas7bdat")),
            (pd.read_json, "os", ("io", "json", "data", "tsframe_v012.json")),
            (
                pd.read_pickle,
                "os",
                ("io", "data", "pickle", "categorical.0.25.0.pickle"),
            ),
        ],
    )
    def test_read_fspath_all(self, reader: Callable, module: str, path: Tuple[str, ...], datapath: Callable) -> None:
        pytest.importorskip(module)
        path_str: str = datapath(*path)

        mypath: CustomFSPath = CustomFSPath(path_str)
        result: pd.DataFrame = reader(mypath)
        expected: pd.DataFrame = reader(path_str)

        if path_str.endswith(".pickle"):
            # categorical
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "writer_name, writer_kwargs, module",
        [
            ("to_csv", {}, "os"),
            ("to_excel", {"engine": "openpyxl"}, "openpyxl"),
            ("to_feather", {}, "pyarrow"),
            ("to_html", {}, "os"),
            ("to_json", {}, "os"),
            ("to_latex", {}, "os"),
            ("to_pickle", {}, "os"),
            ("to_stata", {"time_stamp": pd.to_datetime("2019-01-01 00:00")}, "os"),
        ],
    )
    def test_write_fspath_all(self, writer_name: str, writer_kwargs: Dict[str, Any], module: str) -> None:
        if writer_name in ["to_latex"]:  # uses Styler implementation
            pytest.importorskip("jinja2")
        p1: str = tm.ensure_clean("string")
        p2: str = tm.ensure_clean("fspath")
        df: pd.DataFrame = pd.DataFrame({"A": [1, 2]})

        with p1 as string, p2 as fspath:
            pytest.importorskip(module)
            mypath: CustomFSPath = CustomFSPath(fspath)
            writer: Callable = getattr(df, writer_name)

            writer(string, **writer_kwargs)
            writer(mypath, **writer_kwargs)
            with open(string, "rb") as f_str, open(fspath, "rb") as f_path:
                if writer_name == "to_excel":
                    # binary representation of excel contains time creation
                    # data that causes flaky CI failures
                    result: pd.DataFrame = pd.read_excel(f_str, **writer_kwargs)
                    expected: pd.DataFrame = pd.read_excel(f_path, **writer_kwargs)
                    tm.assert_frame_equal(result, expected)
                else:
                    result: bytes = f_str.read()
                    expected: bytes = f_path.read()
                    assert result == expected

    def test_write_fspath_hdf5(self) -> None:
        # Same test as write_fspath_all, except HDF5 files aren't
