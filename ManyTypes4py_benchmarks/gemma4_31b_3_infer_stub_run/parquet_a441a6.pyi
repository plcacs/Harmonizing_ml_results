from __future__ import annotations
import io
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, Sequence, List, Tuple, overload
from pandas import DataFrame
from pandas._libs import lib

if TYPE_CHECKING:
    from pandas._typing import DtypeBackend, FilePath, ReadBuffer, StorageOptions, WriteBuffer

def get_engine(engine: Literal['auto', 'pyarrow', 'fastparquet']) -> Union['PyArrowImpl', 'FastParquetImpl']: ...

def _get_path_or_handle(
    path: Any,
    fs: Any,
    storage_options: Optional[StorageOptions] = ...,
    mode: str = ...,
    is_dir: bool = ...,
) -> Tuple[Any, Any, Any]: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...

    def write(self, df: DataFrame, path: Any, compression: str, **kwargs: Any) -> None: ...

    def read(self, path: Any, columns: Optional[Sequence[str]] = ..., **kwargs: Any) -> DataFrame: ...

class PyArrowImpl(BaseImpl):
    api: Any

    def __init__(self) -> None: ...

    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: str = 'snappy',
        index: Optional[bool] = None,
        storage_options: Optional[StorageOptions] = None,
        partition_cols: Optional[Union[str, Sequence[str]]] = None,
        filesystem: Any = None,
        **kwargs: Any,
    ) -> None: ...

    def read(
        self,
        path: Any,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[Union[List[Tuple[Any, ...]], List[List[Tuple[Any, ...]]]]] = None,
        dtype_backend: DtypeBackend = lib.no_default,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        to_pandas_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    api: Any

    def __init__(self) -> None: ...

    def write(
        self,
        df: DataFrame,
        path: Any,
        compression: str = 'snappy',
        index: Optional[bool] = None,
        partition_cols: Optional[Union[str, Sequence[str]]] = None,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        **kwargs: Any,
    ) -> None: ...

    def read(
        self,
        path: Any,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[Union[List[Tuple[Any, ...]], List[List[Tuple[Any, ...]]]]] = None,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        to_pandas_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DataFrame: ...

@overload
def to_parquet(
    df: DataFrame,
    path: None,
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: Optional[str] = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[Union[str, Sequence[str]]] = None,
    filesystem: Any = None,
    **kwargs: Any,
) -> bytes: ...

@overload
def to_parquet(
    df: DataFrame,
    path: Union[str, FilePath, WriteBuffer],
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: Optional[str] = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[Union[str, Sequence[str]]] = None,
    filesystem: Any = None,
    **kwargs: Any,
) -> None: ...

def to_parquet(
    df: DataFrame,
    path: Optional[Union[str, FilePath, WriteBuffer]] = None,
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    compression: Optional[str] = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[Union[str, Sequence[str]]] = None,
    filesystem: Any = None,
    **kwargs: Any,
) -> Optional[bytes]: ...

def read_parquet(
    path: Union[str, FilePath, ReadBuffer],
    engine: Literal['auto', 'pyarrow', 'fastparquet'] = 'auto',
    columns: Optional[Sequence[str]] = None,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: DtypeBackend = lib.no_default,
    filesystem: Any = None,
    filters: Optional[Union[List[Tuple[Any, ...]], List[List[Tuple[Any, ...]]]]] = None,
    to_pandas_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> DataFrame: ...