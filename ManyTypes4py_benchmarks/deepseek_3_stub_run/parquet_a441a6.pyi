from __future__ import annotations

import io
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
    Union,
    Optional,
    IO,
    BinaryIO,
    TypeVar,
)
from warnings import catch_warnings

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeBackend,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )
    from pandas import DataFrame
    from pyarrow.fs import FileSystem as PyArrowFileSystem
    from fsspec.spec import AbstractFileSystem as FsspecFileSystem
    import pyarrow
    import fastparquet

_T = TypeVar("_T")

def get_engine(
    engine: Literal["auto", "pyarrow", "fastparquet"]
) -> Union["PyArrowImpl", "FastParquetImpl"]: ...

def _get_path_or_handle(
    path: Union[str, bytes, os.PathLike, BinaryIO],
    fs: Optional[Union["PyArrowFileSystem", "FsspecFileSystem"]] = None,
    storage_options: Optional["StorageOptions"] = None,
    mode: str = "rb",
    is_dir: bool = False,
) -> tuple[Union[str, bytes, BinaryIO], Optional[Any], Optional[Any]]: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: "DataFrame") -> None: ...
    
    def write(
        self,
        df: "DataFrame",
        path: Union[str, bytes, os.PathLike, BinaryIO],
        compression: Optional[str] = "snappy",
        **kwargs: Any,
    ) -> None: ...
    
    def read(
        self,
        path: Union[str, bytes, os.PathLike, BinaryIO],
        columns: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "DataFrame": ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    
    def write(
        self,
        df: "DataFrame",
        path: Union[str, bytes, os.PathLike, BinaryIO],
        compression: Optional[str] = "snappy",
        index: Optional[bool] = None,
        storage_options: Optional["StorageOptions"] = None,
        partition_cols: Optional[Union[str, list[str]]] = None,
        filesystem: Optional[Union["PyArrowFileSystem", "FsspecFileSystem"]] = None,
        **kwargs: Any,
    ) -> None: ...
    
    def read(
        self,
        path: Union[str, bytes, os.PathLike, BinaryIO],
        columns: Optional[list[str]] = None,
        filters: Optional[Union[list[tuple], list[list[tuple]]]] = None,
        dtype_backend: Union["DtypeBackend", Literal[lib.no_default]] = lib.no_default,
        storage_options: Optional["StorageOptions"] = None,
        filesystem: Optional[Union["PyArrowFileSystem", "FsspecFileSystem"]] = None,
        to_pandas_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "DataFrame": ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    
    def write(
        self,
        df: "DataFrame",
        path: Union[str, bytes, os.PathLike, BinaryIO],
        compression: Optional[str] = "snappy",
        index: Optional[bool] = None,
        partition_cols: Optional[Union[str, list[str]]] = None,
        storage_options: Optional["StorageOptions"] = None,
        filesystem: Optional[Any] = None,
        **kwargs: Any,
    ) -> None: ...
    
    def read(
        self,
        path: Union[str, bytes, os.PathLike, BinaryIO],
        columns: Optional[list[str]] = None,
        filters: Optional[Union[list[tuple], list[list[tuple]]]] = None,
        storage_options: Optional["StorageOptions"] = None,
        filesystem: Optional[Any] = None,
        to_pandas_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "DataFrame": ...

@overload
def to_parquet(
    df: "DataFrame",
    path: None = None,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    compression: Optional[str] = "snappy",
    index: Optional[bool] = None,
    storage_options: Optional["StorageOptions"] = None,
    partition_cols: Optional[Union[str, list[str]]] = None,
    filesystem: Optional[Union["PyArrowFileSystem", "FsspecFileSystem"]] = None,
    **kwargs: Any,
) -> bytes: ...

@overload
def to_parquet(
    df: "DataFrame",
    path: Union[str, bytes, os.PathLike, BinaryIO],
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    compression: Optional[str] = "snappy",
    index: Optional[bool] = None,
    storage_options: Optional["StorageOptions"] = None,
    partition_cols: Optional[Union[str, list[str]]] = None,
    filesystem: Optional[Union["PyArrowFileSystem", "FsspecFileSystem"]] = None,
    **kwargs: Any,
) -> None: ...

def to_parquet(
    df: "DataFrame",
    path: Optional[Union[str, bytes, os.PathLike, BinaryIO]] = None,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    compression: Optional[str] = "snappy",
    index: Optional[bool] = None,
    storage_options: Optional["StorageOptions"] = None,
    partition_cols: Optional[Union[str, list[str]]] = None,
    filesystem: Optional[Union["PyArrowFileSystem", "FsspecFileSystem"]] = None,
    **kwargs: Any,
) -> Optional[bytes]: ...

def read_parquet(
    path: Union[str, bytes, os.PathLike, BinaryIO],
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    columns: Optional[list[str]] = None,
    storage_options: Optional["StorageOptions"] = None,
    dtype_backend: Union["DtypeBackend", Literal[lib.no_default]] = lib.no_default,
    filesystem: Optional[Union["PyArrowFileSystem", "FsspecFileSystem"]] = None,
    filters: Optional[Union[list[tuple], list[list[tuple]]]] = None,
    to_pandas_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> "DataFrame": ...