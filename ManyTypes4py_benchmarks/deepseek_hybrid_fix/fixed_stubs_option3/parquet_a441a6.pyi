from __future__ import annotations
import io
import json
import os
from typing import TYPE_CHECKING, Any, Literal, Union, Optional, BinaryIO

from warnings import catch_warnings, filterwarnings

from pandas._libs import lib
from pandas import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import get_handle, is_fsspec_url, is_url, stringify_path

if TYPE_CHECKING:
    from pandas._typing import DtypeBackend, FilePath, ReadBuffer, StorageOptions, WriteBuffer

PyArrowFileSystem = Any
FsspecFileSystem = Any

def get_engine(engine: Literal['auto', 'pyarrow', 'fastparquet']) -> BaseImpl: ...

def _get_path_or_handle(
    path: str | bytes | os.PathLike | BinaryIO,
    fs: Any | None,
    storage_options: StorageOptions | None = None,
    mode: str = 'rb',
    is_dir: bool = False
) -> tuple[str | bytes | BinaryIO, Any | None, Any | None]: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path: str | bytes | os.PathLike | BinaryIO, compression: str | None, **kwargs: Any) -> None: ...
    def read(self, path: str | bytes | os.PathLike | BinaryIO, columns: list[str] | None = None, **kwargs: Any) -> DataFrame: ...

class PyArrowImpl(BaseImpl):
    api: Any
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: str | bytes | os.PathLike | BinaryIO,
        compression: str | None = 'snappy',
        index: bool | None = None,
        storage_options: StorageOptions | None = None,
        partition_cols: str | list[str] | None = None,
        filesystem: Any | None = None,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: str | bytes | os.PathLike | BinaryIO,
        columns: list[str] | None = None,
        filters: list[tuple] | list[list[tuple]] | None = None,
        dtype_backend: DtypeBackend | Any = ...,
        storage_options: StorageOptions | None = None,
        filesystem: Any | None = None,
        to_pandas_kwargs: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    api: Any
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: str | bytes | os.PathLike | BinaryIO,
        compression: str | None = 'snappy',
        index: bool | None = None,
        partition_cols: str | list[str] | None = None,
        storage_options: StorageOptions | None = None,
        filesystem: Any = None,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: str | bytes | os.PathLike | BinaryIO,
        columns: list[str] | None = None,
        filters: list[tuple] | list[list[tuple]] | None = None,
        storage_options: StorageOptions | None = None,
        filesystem: Any = None,
        to_pandas_kwargs: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> DataFrame: ...

def to_parquet(
    df: DataFrame,
    path: str | bytes | os.PathLike | BinaryIO | None = None,
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: str | None = 'snappy',
    index: bool | None = None,
    storage_options: StorageOptions | None = None,
    partition_cols: str | list[str] | None = None,
    filesystem: Any | None = None,
    **kwargs: Any
) -> bytes | None: ...

def read_parquet(
    path: str | bytes | os.PathLike | BinaryIO,
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    columns: list[str] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | Any = ...,
    filesystem: Any | None = None,
    filters: list[tuple] | list[list[tuple]] | None = None,
    to_pandas_kwargs: dict[str, Any] | None = None,
    **kwargs: Any
) -> DataFrame: ...