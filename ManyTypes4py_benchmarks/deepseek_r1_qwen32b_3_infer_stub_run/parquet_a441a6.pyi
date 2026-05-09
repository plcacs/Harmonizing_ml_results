"""parquet compat"""

from __future__ import annotations
import io
import json
import os
from typing import (
    Any,
    IO,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)
from pyarrow import parquet as pa_parquet
from pandas import DataFrame
from pandas._typing import DtypeBackend, FilePath, ReadBuffer, StorageOptions, WriteBuffer
from pandas.io.common import IOHandles

def get_engine(engine: str) -> Union[PyArrowImpl, FastParquetImpl]: ...

def _get_path_or_handle(
    path: Union[str, os.PathLike, IO[bytes]],
    fs: Any,
    storage_options: Optional[StorageOptions] = None,
    mode: str = 'rb',
    is_dir: bool = False,
) -> Tuple[Any, Optional[IOHandles], Any]: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    
    def write(
        self,
        df: DataFrame,
        path: Union[str, os.PathLike, IO[bytes]],
        compression: str = 'snappy',
        **kwargs: Any
    ) -> None: ...
    
    def read(
        self,
        path: Union[str, os.PathLike, IO[bytes]],
        columns: Optional[List[str]] = None,
        **kwargs: Any
    ) -> DataFrame: ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    
    def write(
        self,
        df: DataFrame,
        path: Union[str, os.PathLike, IO[bytes]],
        compression: str = 'snappy',
        index: Optional[bool] = None,
        storage_options: Optional[StorageOptions] = None,
        partition_cols: Optional[List[str]] = None,
        filesystem: Any = None,
        **kwargs: Any
    ) -> None: ...
    
    def read(
        self,
        path: Union[str, os.PathLike, IO[bytes]],
        columns: Optional[List[str]] = None,
        filters: Optional[List[List[Tuple[str, str, Any]]]] = None,
        dtype_backend: DtypeBackend = lib.no_default,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        to_pandas_kwargs: Optional[dict] = None,
        **kwargs: Any
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    
    def write(
        self,
        df: DataFrame,
        path: Union[str, os.PathLike, IO[bytes]],
        compression: str = 'snappy',
        index: Optional[bool] = None,
        partition_cols: Optional[List[str]] = None,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        **kwargs: Any
    ) -> None: ...
    
    def read(
        self,
        path: Union[str, os.PathLike, IO[bytes]],
        columns: Optional[List[str]] = None,
        filters: Optional[List[List[Tuple[str, str, Any]]]] = None,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        to_pandas_kwargs: Optional[dict] = None,
        **kwargs: Any
    ) -> DataFrame: ...

@overload
def to_parquet(
    df: DataFrame,
    path: None = None,
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd', None] = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[List[str]] = None,
    filesystem: Any = None,
    **kwargs: Any
) -> bytes: ...

@overload
def to_parquet(
    df: DataFrame,
    path: Union[str, os.PathLike, IO[bytes]],
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd', None] = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[List[str]] = None,
    filesystem: Any = None,
    **kwargs: Any
) -> None: ...

def to_parquet(
    df: DataFrame,
    path: Union[str, os.PathLike, IO[bytes], None] = None,
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd', None] = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[List[str]] = None,
    filesystem: Any = None,
    **kwargs: Any
) -> Optional[bytes]: ...

def read_parquet(
    path: Union[str, os.PathLike, IO[bytes]],
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    columns: Optional[List[str]] = None,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: DtypeBackend = lib.no_default,
    filesystem: Any = None,
    filters: Optional[List[List[Tuple[str, str, Any]]]] = None,
    to_pandas_kwargs: Optional[dict] = None,
    **kwargs: Any
) -> DataFrame: ...