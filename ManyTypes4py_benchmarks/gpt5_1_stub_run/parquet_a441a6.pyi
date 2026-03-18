from typing import Any, Mapping, Optional, Sequence, Union, Literal
from pandas import DataFrame

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path: Any, compression: Any, **kwargs: Any) -> None: ...
    def read(self, path: Any, columns: Optional[Sequence[str]] = None, **kwargs: Any) -> DataFrame: ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: Optional[str] = ...,
        index: Optional[bool] = ...,
        storage_options: Optional[Mapping[str, Any]] = ...,
        partition_cols: Optional[Union[str, Sequence[str]]] = ...,
        filesystem: Any = ...,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Optional[Sequence[str]] = ...,
        filters: Any = ...,
        dtype_backend: Any = ...,
        storage_options: Optional[Mapping[str, Any]] = ...,
        filesystem: Any = ...,
        to_pandas_kwargs: Optional[Mapping[str, Any]] = ...,
        **kwargs: Any
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: Optional[str] = ...,
        index: Optional[bool] = ...,
        partition_cols: Optional[Union[str, Sequence[str]]] = ...,
        storage_options: Optional[Mapping[str, Any]] = ...,
        filesystem: Any = ...,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Optional[Sequence[str]] = ...,
        filters: Any = ...,
        storage_options: Optional[Mapping[str, Any]] = ...,
        filesystem: Any = ...,
        to_pandas_kwargs: Optional[Mapping[str, Any]] = ...,
        **kwargs: Any
    ) -> DataFrame: ...

def get_engine(engine: Literal["auto", "pyarrow", "fastparquet"]) -> BaseImpl: ...

def to_parquet(
    df: DataFrame,
    path: Optional[Any] = ...,
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    compression: Optional[Literal["snappy", "gzip", "brotli", "lz4", "zstd"]] = ...,
    index: Optional[bool] = ...,
    storage_options: Optional[Mapping[str, Any]] = ...,
    partition_cols: Optional[Union[str, Sequence[str]]] = ...,
    filesystem: Any = ...,
    **kwargs: Any
) -> Optional[bytes]: ...

def read_parquet(
    path: Any,
    engine: Literal["auto", "pyarrow", "fastparquet"] = ...,
    columns: Optional[Sequence[str]] = ...,
    storage_options: Optional[Mapping[str, Any]] = ...,
    dtype_backend: Any = ...,
    filesystem: Any = ...,
    filters: Any = ...,
    to_pandas_kwargs: Optional[Mapping[str, Any]] = ...,
    **kwargs: Any
) -> DataFrame: ...