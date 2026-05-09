"""parquet compat"""
from __future__ import annotations
import io
import json
import os
from typing import (
    Any,
    Dict,
    IO,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)
from pandas._typing import (
    DtypeBackend,
    FilePath,
    ReadBuffer,
    StorageOptions,
    WriteBuffer,
)
from pandas import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import IOHandles

def get_engine(engine: str) -> Union[PyArrowImpl, FastParquetImpl]: ...

def _get_path_or_handle(
    path: Union[str, os.PathLike, IO],
    fs: Optional[Any],
    storage_options: Optional[StorageOptions] = None,
    mode: Literal['rb', 'wb'] = 'rb',
    is_dir: bool = False,
) -> Tuple[Union[str, IO], Optional[IOHandles], Optional[Any]]: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path: Union[str, os.PathLike, IO], **kwargs: Any) -> None: ...
    def read(self, path: Union[str, os.PathLike, IO], **kwargs: Any) -> DataFrame: ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Union[str, os.PathLike, IO],
        compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd', None] = 'snappy',
        index: Optional[bool] = None,
        storage_options: Optional[StorageOptions] = None,
        partition_cols: Optional[List[str]] = None,
        filesystem: Optional[Any] = None,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Union[str, os.PathLike, IO],
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        dtype_backend: DtypeBackend = lib.no_default,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Optional[Any] = None,
        to_pandas_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Union[str, os.PathLike, IO],
        compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd', None] = 'snappy',
        index: Optional[bool] = None,
        partition_cols: Optional[List[str]] = None,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Optional[Any] = None,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Union[str, os.PathLike, IO],
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        dtype_backend: DtypeBackend = lib.no_default,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Optional[Any] = None,
        to_pandas_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> DataFrame: ...

@doc(storage_options=_shared_docs['storage_options'])
def to_parquet(
    df: DataFrame,
    path: Optional[Union[str, os.PathLike, IO]] = None,
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd', None] = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[List[str]] = None,
    filesystem: Optional[Any] = None,
    **kwargs: Any
) -> Optional[bytes]: ...

@doc(storage_options=_shared_docs['storage_options'])
def read_parquet(
    path: Union[str, os.PathLike, IO],
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    columns: Optional[List[str]] = None,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: DtypeBackend = lib.no_default,
    filesystem: Optional[Any] = None,
    filters: Optional[List[Tuple]] = None,
    to_pandas_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> DataFrame: ...