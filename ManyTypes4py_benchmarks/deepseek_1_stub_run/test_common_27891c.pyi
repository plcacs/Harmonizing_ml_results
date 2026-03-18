```python
import _io
import codecs
import errno
import mmap
import os
from pathlib import Path
import pickle
from typing import (
    Any,
    IO,
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    overload,
    Sequence,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import Buffer, Protocol

_T = TypeVar("_T")
_AnyFSPath = Union[str, bytes, Path, "CustomFSPath"]

class CustomFSPath:
    def __init__(self, path: str) -> None: ...
    def __fspath__(self) -> str: ...

def _expand_user(filepath_or_buffer: _AnyFSPath) -> str: ...
def stringify_path(filepath_or_buffer: _AnyFSPath) -> str: ...
def infer_compression(
    filepath_or_buffer: _AnyFSPath,
    compression: Optional[str] = ...,
) -> Optional[str]: ...
def get_handle(
    path_or_buf: Union[_AnyFSPath, IO[Any]],
    mode: str,
    encoding: Optional[str] = ...,
    compression: Optional[str] = ...,
    memory_map: bool = ...,
    is_text: bool = ...,
    errors: Optional[str] = ...,
) -> Any: ...
def _maybe_memory_map(
    file: IO[Any],
    memory_map: bool,
) -> IO[Any]: ...
def is_fsspec_url(url: str) -> bool: ...

class TestCommonIOCapabilities:
    data1: str
    def test_expand_user(self) -> None: ...
    def test_expand_user_normal_path(self) -> None: ...
    def test_stringify_path_pathlib(self) -> None: ...
    def test_stringify_path_fspath(self) -> None: ...
    def test_stringify_file_and_path_like(self) -> None: ...
    @overload
    def test_infer_compression_from_path(
        self,
        compression_format: Tuple[str, Optional[str]],
        path_type: Type[str],
    ) -> None: ...
    @overload
    def test_infer_compression_from_path(
        self,
        compression_format: Tuple[str, Optional[str]],
        path_type: Type[CustomFSPath],
    ) -> None: ...
    @overload
    def test_infer_compression_from_path(
        self,
        compression_format: Tuple[str, Optional[str]],
        path_type: Type[Path],
    ) -> None: ...
    def test_infer_compression_from_path(
        self,
        compression_format: Tuple[str, Optional[str]],
        path_type: Union[Type[str], Type[CustomFSPath], Type[Path]],
    ) -> None: ...
    @overload
    def test_get_handle_with_path(
        self,
        path_type: Type[str],
    ) -> None: ...
    @overload
    def test_get_handle_with_path(
        self,
        path_type: Type[CustomFSPath],
    ) -> None: ...
    @overload
    def test_get_handle_with_path(
        self,
        path_type: Type[Path],
    ) -> None: ...
    def test_get_handle_with_path(
        self,
        path_type: Union[Type[str], Type[CustomFSPath], Type[Path]],
    ) -> None: ...
    def test_get_handle_with_buffer(self) -> None: ...
    def test_bytesiowrapper_returns_correct_bytes(self) -> None: ...
    def test_get_handle_pyarrow_compat(self) -> None: ...
    def test_iterator(self) -> None: ...
    @overload
    def test_read_non_existent(
        self,
        reader: Callable[..., Any],
        module: str,
        error_class: Type[FileNotFoundError],
        fn_ext: str,
    ) -> None: ...
    @overload
    def test_read_non_existent(
        self,
        reader: Callable[..., Any],
        module: str,
        error_class: Type[OSError],
        fn_ext: str,
    ) -> None: ...
    def test_read_non_existent(
        self,
        reader: Callable[..., Any],
        module: str,
        error_class: Union[Type[FileNotFoundError], Type[OSError]],
        fn_ext: str,
    ) -> None: ...
    @overload
    def test_write_missing_parent_directory(
        self,
        method: Callable[..., Any],
        module: str,
        error_class: Type[OSError],
        fn_ext: str,
    ) -> None: ...
    def test_write_missing_parent_directory(
        self,
        method: Callable[..., Any],
        module: str,
        error_class: Type[OSError],
        fn_ext: str,
    ) -> None: ...
    @overload
    def test_read_expands_user_home_dir(
        self,
        reader: Callable[..., Any],
        module: str,
        error_class: Type[FileNotFoundError],
        fn_ext: str,
        monkeypatch: Any,
    ) -> None: ...
    @overload
    def test_read_expands_user_home_dir(
        self,
        reader: Callable[..., Any],
        module: str,
        error_class: Type[OSError],
        fn_ext: str,
        monkeypatch: Any,
    ) -> None: ...
    def test_read_expands_user_home_dir(
        self,
        reader: Callable[..., Any],
        module: str,
        error_class: Union[Type[FileNotFoundError], Type[OSError]],
        fn_ext: str,
        monkeypatch: Any,
    ) -> None: ...
    def test_read_fspath_all(
        self,
        reader: Callable[..., Any],
        module: str,
        path: Tuple[str, ...],
        datapath: Callable[..., str],
    ) -> None: ...
    def test_write_fspath_all(
        self,
        writer_name: str,
        writer_kwargs: dict[str, Any],
        module: str,
    ) -> None: ...
    def test_write_fspath_hdf5(self) -> None: ...

class TestMMapWrapper:
    def test_constructor_bad_file(self, mmap_file: str) -> None: ...
    def test_next(self, mmap_file: str) -> None: ...
    def test_unknown_engine(self) -> None: ...
    def test_binary_mode(self) -> None: ...
    def test_warning_missing_utf_bom(
        self,
        encoding: str,
        compression_: str,
    ) -> None: ...

def test_is_fsspec_url() -> None: ...
def test_codecs_encoding(
    encoding: Optional[str],
    format: Literal["csv", "json"],
) -> None: ...
def test_codecs_get_writer_reader() -> None: ...
def test_explicit_encoding(
    io_class: Union[Type[BytesIO], Type[StringIO]],
    mode: str,
    msg: str,
) -> None: ...
def test_encoding_errors(
    encoding_errors: str,
    format: Literal["csv", "json"],
) -> None: ...
def test_encoding_errors_badtype(encoding_errors: Any) -> None: ...
def test_bad_encdoing_errors() -> None: ...
def test_errno_attribute() -> None: ...
def test_fail_mmap() -> None: ...
def test_close_on_error() -> None: ...
def test_read_csv_chained_url_no_error(compression: Optional[str]) -> None: ...
def test_pickle_reader(reader: Callable[..., Any]) -> None: ...
def test_pyarrow_read_csv_datetime_dtype() -> None: ...
```