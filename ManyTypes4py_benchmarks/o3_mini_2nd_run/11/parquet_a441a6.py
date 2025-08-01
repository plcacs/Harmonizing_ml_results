from __future__ import annotations
import io
import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
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

EngineType = Literal['auto', 'pyarrow', 'fastparquet']
PathOrBuffer = Union[str, bytes, os.PathLike, io.BytesIO, io.BufferedIOBase]


def get_engine(engine: EngineType) -> BaseImpl:
    """return our implementation"""
    if engine == 'auto':
        engine = get_option('io.parquet.engine')
    if engine == 'auto':
        engine_classes = [PyArrowImpl, FastParquetImpl]
        error_msgs = ''
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += '\n - ' + str(err)
        raise ImportError(f"Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.\nA suitable version of pyarrow or fastparquet is required for parquet support.\nTrying to import the above resulted in these errors:{error_msgs}")
    if engine == 'pyarrow':
        return PyArrowImpl()
    elif engine == 'fastparquet':
        return FastParquetImpl()
    raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")


def _get_path_or_handle(
    path: Union[FilePath, Any],
    fs: Any,
    storage_options: Optional[StorageOptions] = None,
    mode: str = 'rb',
    is_dir: bool = False,
) -> Tuple[Any, Optional[IOHandles], Any]:
    """File handling for PyArrow."""
    path_or_handle: Any = stringify_path(path)
    if fs is not None:
        pa_fs = import_optional_dependency('pyarrow.fs', errors='ignore')
        fsspec = import_optional_dependency('fsspec', errors='ignore')
        if pa_fs is not None and isinstance(fs, pa_fs.FileSystem):
            if storage_options:
                raise NotImplementedError('storage_options not supported with a pyarrow FileSystem.')
        elif fsspec is not None and isinstance(fs, fsspec.spec.AbstractFileSystem):
            pass
        else:
            raise ValueError(f'filesystem must be a pyarrow or fsspec FileSystem, not a {type(fs).__name__}')
    if is_fsspec_url(path_or_handle) and fs is None:
        if storage_options is None:
            pa = import_optional_dependency('pyarrow')
            pa_fs = import_optional_dependency('pyarrow.fs')
            try:
                fs, path_or_handle = pa_fs.FileSystem.from_uri(path)
            except (TypeError, pa.ArrowInvalid):
                pass
        if fs is None:
            fsspec = import_optional_dependency('fsspec')
            fs, path_or_handle = fsspec.core.url_to_fs(path_or_handle, **storage_options or {})
    elif storage_options and (not is_url(path_or_handle) or mode != 'rb'):
        raise ValueError('storage_options passed with buffer, or non-supported URL')
    handles: Optional[IOHandles] = None
    if not fs and (not is_dir) and isinstance(path_or_handle, str) and (not os.path.isdir(path_or_handle)):
        handles = get_handle(path_or_handle, mode, is_text=False, storage_options=storage_options)
        fs = None
        path_or_handle = handles.handle
    return (path_or_handle, handles, fs)


class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None:
        if not isinstance(df, DataFrame):
            raise ValueError('to_parquet only supports IO with DataFrames')

    def write(
        self,
        df: DataFrame,
        path: Union[FilePath, WriteBuffer],
        compression: str,
        **kwargs: Any,
    ) -> None:
        raise AbstractMethodError(self)

    def read(
        self,
        path: Union[FilePath, ReadBuffer],
        columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DataFrame:
        raise AbstractMethodError(self)


class PyArrowImpl(BaseImpl):
    def __init__(self) -> None:
        import_optional_dependency('pyarrow', extra='pyarrow is required for parquet support.')
        import pyarrow.parquet  # noqa: F401
        import pandas.core.arrays.arrow.extension_types  # noqa: F401
        import pyarrow  # noqa: F401
        self.api = __import__('pyarrow')

    def write(
        self,
        df: DataFrame,
        path: Union[FilePath, WriteBuffer],
        compression: str = 'snappy',
        index: Optional[bool] = None,
        storage_options: Optional[StorageOptions] = None,
        partition_cols: Optional[Union[str, List[str]]] = None,
        filesystem: Any = None,
        **kwargs: Any,
    ) -> None:
        self.validate_dataframe(df)
        from_pandas_kwargs: Dict[str, Any] = {'schema': kwargs.pop('schema', None)}
        if index is not None:
            from_pandas_kwargs['preserve_index'] = index
        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)
        if df.attrs:
            df_metadata = {'PANDAS_ATTRS': json.dumps(df.attrs)}
            existing_metadata = table.schema.metadata or {}
            merged_metadata = {**existing_metadata, **df_metadata}
            table = table.replace_schema_metadata(merged_metadata)
        path_or_handle, handles, filesystem = _get_path_or_handle(
            path, filesystem, storage_options=storage_options, mode='wb', is_dir=partition_cols is not None
        )
        if isinstance(path_or_handle, io.BufferedWriter) and hasattr(path_or_handle, 'name') and isinstance(path_or_handle.name, (str, bytes)):
            if isinstance(path_or_handle.name, bytes):
                path_or_handle = path_or_handle.name.decode()
            else:
                path_or_handle = path_or_handle.name
        try:
            if partition_cols is not None:
                self.api.parquet.write_to_dataset(
                    table,
                    path_or_handle,
                    compression=compression,
                    partition_cols=partition_cols,
                    filesystem=filesystem,
                    **kwargs,
                )
            else:
                self.api.parquet.write_table(
                    table, path_or_handle, compression=compression, filesystem=filesystem, **kwargs
                )
        finally:
            if handles is not None:
                handles.close()

    def read(
        self,
        path: Union[FilePath, ReadBuffer],
        columns: Optional[List[str]] = None,
        filters: Any = None,
        dtype_backend: DtypeBackend = lib.no_default,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        to_pandas_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DataFrame:
        kwargs['use_pandas_metadata'] = True
        path_or_handle, handles, filesystem = _get_path_or_handle(
            path, filesystem, storage_options=storage_options, mode='rb'
        )
        try:
            pa_table = self.api.parquet.read_table(
                path_or_handle, columns=columns, filesystem=filesystem, filters=filters, **kwargs
            )
            with catch_warnings():
                filterwarnings('ignore', 'make_block is deprecated', DeprecationWarning)
                result = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend, to_pandas_kwargs=to_pandas_kwargs)
            if pa_table.schema.metadata:
                if b'PANDAS_ATTRS' in pa_table.schema.metadata:
                    df_metadata = pa_table.schema.metadata[b'PANDAS_ATTRS']
                    result.attrs = json.loads(df_metadata)
            return result
        finally:
            if handles is not None:
                handles.close()


class FastParquetImpl(BaseImpl):
    def __init__(self) -> None:
        fastparquet = import_optional_dependency('fastparquet', extra='fastparquet is required for parquet support.')
        self.api = fastparquet

    def write(
        self,
        df: DataFrame,
        path: Union[FilePath, WriteBuffer],
        compression: str = 'snappy',
        index: Optional[bool] = None,
        partition_cols: Optional[Union[str, List[str]]] = None,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        **kwargs: Any,
    ) -> None:
        self.validate_dataframe(df)
        if 'partition_on' in kwargs and partition_cols is not None:
            raise ValueError('Cannot use both partition_on and partition_cols. Use partition_cols for partitioning data')
        if 'partition_on' in kwargs:
            partition_cols = kwargs.pop('partition_on')
        if partition_cols is not None:
            kwargs['file_scheme'] = 'hive'
        if filesystem is not None:
            raise NotImplementedError('filesystem is not implemented for the fastparquet engine.')
        path = stringify_path(path)
        if is_fsspec_url(path):
            fsspec = import_optional_dependency('fsspec')
            kwargs['open_with'] = lambda path, _: fsspec.open(path, 'wb', **storage_options or {}).open()
        elif storage_options:
            raise ValueError('storage_options passed with file object or non-fsspec file path')
        with catch_warnings(record=True):
            self.api.write(
                path,
                df,
                compression=compression,
                write_index=index,
                partition_on=partition_cols,
                **kwargs,
            )

    def read(
        self,
        path: Union[FilePath, ReadBuffer],
        columns: Optional[List[str]] = None,
        filters: Any = None,
        storage_options: Optional[StorageOptions] = None,
        filesystem: Any = None,
        to_pandas_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DataFrame:
        parquet_kwargs: Dict[str, Any] = {}
        dtype_backend: DtypeBackend = kwargs.pop('dtype_backend', lib.no_default)
        parquet_kwargs['pandas_nulls'] = False
        if dtype_backend is not lib.no_default:
            raise ValueError("The 'dtype_backend' argument is not supported for the fastparquet engine")
        if filesystem is not None:
            raise NotImplementedError('filesystem is not implemented for the fastparquet engine.')
        if to_pandas_kwargs is not None:
            raise NotImplementedError('to_pandas_kwargs is not implemented for the fastparquet engine.')
        path = stringify_path(path)
        handles: Optional[IOHandles] = None
        if is_fsspec_url(path):
            fsspec = import_optional_dependency('fsspec')
            parquet_kwargs['fs'] = fsspec.open(path, 'rb', **storage_options or {}).fs
        elif isinstance(path, str) and (not os.path.isdir(path)):
            handles = get_handle(path, 'rb', is_text=False, storage_options=storage_options)
            path = handles.handle
        try:
            parquet_file = self.api.ParquetFile(path, **parquet_kwargs)
            with catch_warnings():
                filterwarnings('ignore', 'make_block is deprecated', DeprecationWarning)
                return parquet_file.to_pandas(columns=columns, filters=filters, **kwargs)
        finally:
            if handles is not None:
                handles.close()


@doc(storage_options=_shared_docs['storage_options'])
def to_parquet(
    df: DataFrame,
    path: Optional[Union[FilePath, WriteBuffer]] = None,
    engine: EngineType = 'auto',
    compression: str = 'snappy',
    index: Optional[bool] = None,
    storage_options: Optional[StorageOptions] = None,
    partition_cols: Optional[Union[str, List[str]]] = None,
    filesystem: Any = None,
    **kwargs: Any,
) -> Optional[bytes]:
    """
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, file-like object, or None, default None
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function. If None, the result is
        returned as bytes. If a string, it will be used as Root Directory path
        when writing a partitioned dataset. The engine fastparquet does not
        accept file-like objects.
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    compression : {'snappy', 'gzip', 'brotli', 'lz4', 'zstd', None},
        default 'snappy'. Name of the compression to use. Use ``None``
        for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.
    partition_cols : str or list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.
    {storage_options}

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    kwargs
        Additional keyword arguments passed to the engine.

    Returns
    -------
    bytes if no path argument is provided else None
    """
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]
    impl: BaseImpl = get_engine(engine)
    path_or_buf: Union[io.BytesIO, Union[FilePath, WriteBuffer]] = io.BytesIO() if path is None else path
    impl.write(
        df,
        path_or_buf,
        compression=compression,
        index=index,
        partition_cols=partition_cols,
        storage_options=storage_options,
        filesystem=filesystem,
        **kwargs,
    )
    if path is None:
        assert isinstance(path_or_buf, io.BytesIO)
        return path_or_buf.getvalue()
    else:
        return None


@doc(storage_options=_shared_docs['storage_options'])
def read_parquet(
    path: Union[FilePath, ReadBuffer],
    engine: EngineType = 'auto',
    columns: Optional[List[str]] = None,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: DtypeBackend = lib.no_default,
    filesystem: Any = None,
    filters: Any = None,
    to_pandas_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Load a parquet object from the file path, returning a DataFrame.

    The function automatically handles reading the data from a parquet file
    and creates a DataFrame with the appropriate structure.

    Parameters
    ----------
    path : str, path object or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function.
        The string could be a URL. Valid URL schemes include http, ftp, s3,
        gs, and file. For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``.
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    {storage_options}

        .. versionadded:: 1.3.0

    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    filters : List[Tuple] or List[List[Tuple]], default None
        To filter out data.
        Filter syntax: [[(column, op, val), ...],...]
        where op is [==, =, >, >=, <, <=, !=, in, not in]
        The innermost tuples are transposed into a set of filters applied
        through an `AND` operation.
        The outer list combines these sets of filters through an `OR`
        operation.
        A single list of tuples can also be used, meaning that no `OR`
        operation between set of filters is to be conducted.

        Using this argument will NOT result in row-wise filtering of the final
        partitions unless ``engine="pyarrow"`` is also specified.  For
        other engines, filtering is only performed at the partition level, that is,
        to prevent the loading of some row-groups and/or files.

        .. versionadded:: 2.1.0

    to_pandas_kwargs : dict | None, default None
        Keyword arguments to pass through to :func:`pyarrow.Table.to_pandas`
        when ``engine="pyarrow"``.

        .. versionadded:: 3.0.0

    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    DataFrame
        DataFrame based on parquet file.
    """
    impl: BaseImpl = get_engine(engine)
    check_dtype_backend(dtype_backend)
    return impl.read(
        path,
        columns=columns,
        filters=filters,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
        filesystem=filesystem,
        to_pandas_kwargs=to_pandas_kwargs,
        **kwargs,
    )