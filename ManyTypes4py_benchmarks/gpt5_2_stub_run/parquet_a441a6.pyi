from typing import Any, Optional, Literal
from pandas import DataFrame
from pandas._typing import DtypeBackend

def get_engine(engine: Literal["auto", "pyarrow", "fastparquet"]) -> "BaseImpl": ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path: Any, compression: Any, **kwargs: Any) -> None: ...
    def read(self, path: Any, columns: Any = ..., **kwargs: Any) -> DataFrame: ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: Optional[str] = "snappy",
        index: Optional[bool] = None,
        storage_options: Any = None,
        partition_cols: Any = None,
        filesystem: Any = None,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Any = None,
        filters: Any = None,
        dtype_backend: DtypeBackend | Any = ...,
        storage_options: Any = None,
        filesystem: Any = None,
        to_pandas_kwargs: Any = None,
        **kwargs: Any
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: Optional[str] = "snappy",
        index: Optional[bool] = None,
        partition_cols: Any = None,
        storage_options: Any = None,
        filesystem: Any = None,
        **kwargs: Any
    ) -> None: ...
    def read(
        self,
        path: Any,
        columns: Any = None,
        filters: Any = None,
        storage_options: Any = None,
        filesystem: Any = None,
        to_pandas_kwargs: Any = None,
        **kwargs: Any
    ) -> DataFrame: ...

def to_parquet(
    df: DataFrame,
    path: Any | None = None,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    compression: Optional[str] = "snappy",
    index: Optional[bool] = None,
    storage_options: Any | None = None,
    partition_cols: Any | None = None,
    filesystem: Any | None = None,
    **kwargs: Any
) -> Optional[bytes]: ...

def read_parquet(
    path: Any,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    columns: Any | None = None,
    storage_options: Any | None = None,
    dtype_backend: DtypeBackend | Any = ...,
    filesystem: Any | None = None,
    filters: Any | None = None,
    to_pandas_kwargs: Any | None = None,
    **kwargs: Any
) -> DataFrame: ...