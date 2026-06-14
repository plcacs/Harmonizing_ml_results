from __future__ import annotations

from typing import Any, Literal

from pandas import DataFrame
from pandas._libs import lib
from pandas.io.common import IOHandles
from pandas._typing import FilePath, ReadBuffer, WriteBuffer


def get_engine(engine: Literal["auto", "pyarrow", "fastparquet"]) -> BaseImpl: ...
def _get_path_or_handle(
    path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
    fs: object | None,
    storage_options: dict[str, Any] | None = ...,
    mode: str = ...,
    is_dir: bool = ...,
) -> tuple[object, IOHandles[bytes] | None, object | None]: ...


class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path: object, compression: str | None, **kwargs: Any) -> None: ...
    def read(self, path: FilePath | ReadBuffer[bytes], columns: list[str] | None = ..., **kwargs: Any) -> DataFrame: ...


class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = ...,
        index: bool | None = ...,
        storage_options: dict[str, Any] | None = ...,
        partition_cols: list[str] | None = ...,
        filesystem: object | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        filters: object | None = ...,
        dtype_backend: object = lib.no_default,
        storage_options: dict[str, Any] | None = ...,
        filesystem: object | None = ...,
        to_pandas_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> DataFrame: ...


class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: dict[str, Any] | None = ...,
        filesystem: object | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        filters: object | None = ...,
        storage_options: dict[str, Any] | None = ...,
        filesystem: object | None = ...,
        to_pandas_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> DataFrame: ...


def to_parquet(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = ...,
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = ...,
    index: bool | None = ...,
    storage_options: dict[str, Any] | None = ...,
    partition_cols: str | list[str] | None = ...,
    filesystem: object | None = ...,
    **kwargs: Any,
) -> bytes | None: ...
def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    columns: list[str] | None = ...,
    storage_options: dict[str, Any] | None = ...,
    dtype_backend: object = lib.no_default,
    filesystem: object | None = ...,
    filters: object | None = ...,
    to_pandas_kwargs: dict[str, Any] | None = ...,
    **kwargs: Any,
) -> DataFrame: ...