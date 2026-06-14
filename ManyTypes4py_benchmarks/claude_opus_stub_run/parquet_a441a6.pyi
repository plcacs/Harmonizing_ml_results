from __future__ import annotations

import io
from typing import Any, TYPE_CHECKING

from pandas._libs import lib
from pandas import DataFrame

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeBackend,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )
    from pandas.io.common import IOHandles


def get_engine(engine: str) -> BaseImpl: ...


def _get_path_or_handle(
    path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
    fs: Any,
    storage_options: StorageOptions = ...,
    mode: str = ...,
    is_dir: bool = ...,
) -> tuple[FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], IOHandles[bytes] | None, Any]: ...


class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path: Any, compression: str, **kwargs: Any) -> None: ...
    def read(self, path: Any, columns: list[str] | None = ..., **kwargs: Any) -> DataFrame: ...


class PyArrowImpl(BaseImpl):
    api: Any

    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: str | None = ...,
        index: bool | None = ...,
        storage_options: StorageOptions = ...,
        partition_cols: list[str] | None = ...,
        filesystem: Any = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        filters: Any = ...,
        dtype_backend: DtypeBackend | lib.NoDefault = ...,
        storage_options: StorageOptions = ...,
        filesystem: Any = ...,
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
        compression: str | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        filesystem: Any = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        filters: Any = ...,
        storage_options: StorageOptions = ...,
        filesystem: Any = ...,
        to_pandas_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> DataFrame: ...


def to_parquet(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = ...,
    engine: str = ...,
    compression: str | None = ...,
    index: bool | None = ...,
    storage_options: StorageOptions = ...,
    partition_cols: list[str] | str | None = ...,
    filesystem: Any = ...,
    **kwargs: Any,
) -> bytes | None: ...


def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: str = ...,
    columns: list[str] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    filesystem: Any = ...,
    filters: list[tuple[Any, ...]] | list[list[tuple[Any, ...]]] | None = ...,
    to_pandas_kwargs: dict[str, Any] | None = ...,
    **kwargs: Any,
) -> DataFrame: ...