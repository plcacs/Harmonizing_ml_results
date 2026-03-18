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

def get_engine(engine: str) -> Any: ...

def _get_path_or_handle(
    path: Any,
    fs: Any,
    storage_options: Any = ...,
    mode: str = ...,
    is_dir: bool = ...,
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
        storage_options: Any = ...,
        partition_cols: Any = ...,
        filesystem: Any = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Any = ...,
        filters: Any = ...,
        dtype_backend: Any = ...,
        storage_options: Any = ...,
        filesystem: Any = ...,
        to_pandas_kwargs: Any = ...,
        **kwargs: Any,
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: str = ...,
        index: Optional[bool] = ...,
        partition_cols: Any = ...,
        storage_options: Any = ...,
        filesystem: Any = ...,
        **kwargs: Any,
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Any = ...,
        filters: Any = ...,
        storage_options: Any = ...,
        filesystem: Any = ...,
        to_pandas_kwargs: Any = ...,
        **kwargs: Any,
    ) -> DataFrame: ...

@overload
def to_parquet(
    df: DataFrame,
    path: None = ...,
    engine: str = ...,
    compression: str = ...,
    index: Optional[bool] = ...,
    storage_options: Any = ...,
    partition_cols: Any = ...,
    filesystem: Any = ...,
    **kwargs: Any,
) -> bytes: ...

@overload
def to_parquet(
    df: DataFrame,
    path: Any,
    engine: str = ...,
    compression: str = ...,
    index: Optional[bool] = ...,
    storage_options: Any = ...,
    partition_cols: Any = ...,
    filesystem: Any = ...,
    **kwargs: Any,
) -> None: ...

def to_parquet(
    df: DataFrame,
    path: Any = ...,
    engine: str = ...,
    compression: str = ...,
    index: Optional[bool] = ...,
    storage_options: Any = ...,
    partition_cols: Any = ...,
    filesystem: Any = ...,
    **kwargs: Any,
) -> Union[bytes, None]: ...

def read_parquet(
    path: Any,
    engine: str = ...,
    columns: Any = ...,
    storage_options: Any = ...,
    dtype_backend: Any = ...,
    filesystem: Any = ...,
    filters: Any = ...,
    to_pandas_kwargs: Any = ...,
    **kwargs: Any,
) -> DataFrame: ...
```