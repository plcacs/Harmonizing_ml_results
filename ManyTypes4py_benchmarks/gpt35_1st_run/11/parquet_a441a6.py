from __future__ import annotations
import io
import json
import os
from typing import TYPE_CHECKING, Any, Literal
from warnings import catch_warnings, filterwarnings
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend
from pandas import DataFrame, get_option
from pandas.core.shared_docs import _shared_docs
from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import IOHandles, get_handle, is_fsspec_url, is_url, stringify_path
if TYPE_CHECKING:
    from pandas._typing import DtypeBackend, FilePath, ReadBuffer, StorageOptions, WriteBuffer

def get_engine(engine: Literal['auto', 'pyarrow', 'fastparquet']) -> BaseImpl:
    ...

def _get_path_or_handle(path: FilePath, fs: Any, storage_options: StorageOptions = None, mode: str = 'rb', is_dir: bool = False) -> tuple:
    ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None:
        ...

    def write(self, df: DataFrame, path: FilePath, compression: str, **kwargs: Any) -> None:
        ...

    def read(self, path: FilePath, columns: list[str] = None, **kwargs: Any) -> None:
        ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None:
        ...

    def write(self, df: DataFrame, path: FilePath, compression: str = 'snappy', index: bool = None, storage_options: StorageOptions = None, partition_cols: list[str] = None, filesystem: Any = None, **kwargs: Any) -> None:
        ...

    def read(self, path: FilePath, columns: list[str] = None, filters: Any = None, dtype_backend: DtypeBackend = lib.no_default, storage_options: StorageOptions = None, filesystem: Any = None, to_pandas_kwargs: Any = None, **kwargs: Any) -> DataFrame:
        ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None:
        ...

    def write(self, df: DataFrame, path: FilePath, compression: str = 'snappy', index: bool = None, partition_cols: list[str] = None, storage_options: StorageOptions = None, filesystem: Any = None, **kwargs: Any) -> None:
        ...

    def read(self, path: FilePath, columns: list[str] = None, filters: Any = None, storage_options: StorageOptions = None, filesystem: Any = None, to_pandas_kwargs: Any = None, **kwargs: Any) -> DataFrame:
        ...

@doc(storage_options=_shared_docs['storage_options'])
def to_parquet(df: DataFrame, path: FilePath = None, engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto', compression: str = 'snappy', index: bool = None, storage_options: StorageOptions = None, partition_cols: list[str] = None, filesystem: Any = None, **kwargs: Any) -> bytes | None:
    ...

@doc(storage_options=_shared_docs['storage_options'])
def read_parquet(path: FilePath, engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto', columns: list[str] = None, storage_options: StorageOptions = None, dtype_backend: DtypeBackend = lib.no_default, filesystem: Any = None, filters: Any = None, to_pandas_kwargs: Any = None, **kwargs: Any) -> DataFrame:
    ...
