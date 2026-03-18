```python
from __future__ import annotations
import io
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
    Union,
    Optional,
    List,
    Tuple,
)
from pandas import DataFrame
from pandas._libs import lib

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeBackend,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )
    from fsspec.spec import AbstractFileSystem
    import pyarrow.fs

def get_engine(
    engine: Literal["auto", "pyarrow", "fastparquet"]
) -> Union[PyArrowImpl, FastParquetImpl]: ...

def _get_path_or_handle(
    path: Any,
    fs: Any,
    storage_options: Optional[StorageOptions] = ...,
    mode: str = ...,
    is_dir: bool = ...
) -> tuple[Any, Any, Any]: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: Any) -> None: ...
    def write(self, df: Any, path: Any, compression: Any, **kwargs: Any) -> Any: ...
    def read(self, path: Any, columns: Any = ..., **kwargs: Any) -> Any: ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: str = ...,
        index: Optional[bool] = ...,
        storage_options: Optional[StorageOptions] = ...,
        partition_cols: Optional[Union[str, List[str]]] = ...,
        filesystem: Optional[Union[AbstractFileSystem, pyarrow.fs.FileSystem]] = ...,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Optional[List[str]] = ...,
        filters: Any = ...,
        dtype_backend: Union[DtypeBackend, lib._NoDefault] = ...,
        storage_options: Optional[StorageOptions] = ...,
        filesystem: Optional[Union[AbstractFileSystem, pyarrow.fs.FileSystem]] = ...,
        to_pandas_kwargs: Optional[dict[str, Any]] = ...,
        **kwargs: Any
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: str = ...,
        index: Optional[bool] = ...,
        partition_cols: Optional[Union[str, List[str]]] = ...,
        storage_options: Optional[StorageOptions] = ...,
        filesystem: Any = ...,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Optional[List[str]] = ...,
        filters: Any = ...,
        storage_options: Optional[StorageOptions] = ...,
        filesystem: Any = ...,
        to_pandas_kwargs: Any = ...,
        **kwargs: Any
    ) -> DataFrame: ...

@overload
def to_parquet(
    df: DataFrame,
    path: None = ...,
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    compression: str = ...,
    index: Optional[bool] = ...,
    storage_options: Optional[StorageOptions] = ...,
    partition_cols: Optional[Union[str, List[str]]] = ...,
    filesystem: Optional[Union[AbstractFileSystem, pyarrow.fs.FileSystem]] = ...,
    **kwargs: Any
) -> bytes: ...

@overload
def to_parquet(
    df: DataFrame,
    path: Union[FilePath, WriteBuffer[bytes]],
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    compression: str = ...,
    index: Optional[bool] = ...,
    storage_options: Optional[StorageOptions] = ...,
    partition_cols: Optional[Union[str, List[str]]] = ...,
    filesystem: Optional[Union[AbstractFileSystem, pyarrow.fs.FileSystem]] = ...,
    **kwargs: Any
) -> None: ...

def to_parquet(
    df: DataFrame,
    path: Union[FilePath, WriteBuffer[bytes], None] = ...,
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    compression: str = ...,
    index: Optional[bool] = ...,
    storage_options: Optional[StorageOptions] = ...,
    partition_cols: Optional[Union[str, List[str]]] = ...,
    filesystem: Optional[Union[AbstractFileSystem, pyarrow.fs.FileSystem]] = ...,
    **kwargs: Any
) -> Union[bytes, None]: ...

def read_parquet(
    path: Union[FilePath, ReadBuffer[bytes]],
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    columns: Optional[List[str]] = ...,
    storage_options: Optional[StorageOptions] = ...,
    dtype_backend: Union[DtypeBackend, lib._NoDefault] = ...,
    filesystem: Optional[Union[AbstractFileSystem, pyarrow.fs.FileSystem]] = ...,
    filters: Any = ...,
    to_pandas_kwargs: Optional[dict[str, Any]] = ...,
    **kwargs: Any
) -> DataFrame: ...
```