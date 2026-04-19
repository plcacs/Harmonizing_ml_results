from __future__ import annotations

from typing import Any, Literal, Optional

from pandas import DataFrame
from pandas._libs import lib
from pandas._typing import DtypeBackend, FilePath, ReadBuffer, StorageOptions, WriteBuffer
from pandas.io.common import IOHandles


def get_engine(engine: Literal["auto", "pyarrow", "fastparquet"] | str) -> BaseImpl: ...


def _get_path_or_handle(
    path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
    fs: object | None,
    storage_options: StorageOptions | None = ...,
    mode: str = ...,
    is_dir: bool = ...,
) -> tuple[object, IOHandles | None, object | None]: ...


class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path: object, compression: Optional[str], **kwargs: Any) -> None: ...
    def read(self, path: object, columns: list[str] | None = ..., **kwargs: Any) -> DataFrame: ...


class PyArrowImpl(BaseImpl):
    api: Any
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: Optional[str] = ...,
        index: bool | None = ...,
        storage_options: StorageOptions | None = ...,
        partition_cols: list[str] | None = ...,
        filesystem: object | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        filters: object | None = ...,
        dtype_backend: DtypeBackend | lib.NoDefault = ...,
        storage_options: StorageOptions | None = ...,
        filesystem: object | None = ...,
        to_pandas_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> DataFrame: ...


class FastParquetImpl(BaseImpl):
    api: Any
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: Optional[str] = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions | None = ...,
        filesystem: object | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        filters: object | None = ...,
        storage_options: StorageOptions | None = ...,
        filesystem: object | None = ...,
        to_pandas_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> DataFrame: ...


def to_parquet(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = ...,
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    compression: Optional[str] = ...,
    index: bool | None = ...,
    storage_options: StorageOptions | None = ...,
    partition_cols: str | list[str] | None = ...,
    filesystem: object | None = ...,
    **kwargs: Any,
) -> bytes | None: ...


def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    columns: list[str] | None = ...,
    storage_options: StorageOptions | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    filesystem: object | None = ...,
    filters: object | None = ...,
    to_pandas_kwargs: dict[str, Any] | None = ...,
    **kwargs: Any,
) -> DataFrame: ...