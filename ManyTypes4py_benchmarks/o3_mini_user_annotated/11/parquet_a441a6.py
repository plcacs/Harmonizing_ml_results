from __future__ import annotations

import io
import json
import os
from typing import Any, Literal, cast

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

from pandas._typing import DtypeBackend, FilePath, ReadBuffer, StorageOptions, WriteBuffer


def get_engine(engine: str) -> BaseImpl:
    if engine == "auto":
        engine = get_option("io.parquet.engine")

    if engine == "auto":
        engine_classes = [PyArrowImpl, FastParquetImpl]
        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'pyarrow', 'fastparquet'.\n"
            "A suitable version of "
            "pyarrow or fastparquet is required for parquet "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    if engine == "pyarrow":
        return PyArrowImpl()
    elif engine == "fastparquet":
        return FastParquetImpl()

    raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")


def _get_path_or_handle(
    path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
    fs: Any,
    storage_options: StorageOptions | None = None,
    mode: str = "rb",
    is_dir: bool = False,
) -> tuple[
    FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], IOHandles[bytes] | None, Any
]:
    path_or_handle: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes] = stringify_path(path)
    if fs is not None:
        pa_fs = import_optional_dependency("pyarrow.fs", errors="ignore")
        fsspec = import_optional_dependency("fsspec", errors="ignore")
        if pa_fs is not None and isinstance(fs, pa_fs.FileSystem):
            if storage_options:
                raise NotImplementedError(
                    "storage_options not supported with a pyarrow FileSystem."
                )
        elif fsspec is not None and isinstance(fs, fsspec.spec.AbstractFileSystem):
            pass
        else:
            raise ValueError(
                f"filesystem must be a pyarrow or fsspec FileSystem, "
                f"not a {type(fs).__name__}"
            )
    if is_fsspec_url(path_or_handle) and fs is None:
        if storage_options is None:
            pa = import_optional_dependency("pyarrow")
            pa_fs = import_optional_dependency("pyarrow.fs")
            try:
                fs, path_or_handle = pa_fs.FileSystem.from_uri(path)  # type: ignore
            except (TypeError, pa.ArrowInvalid):
                pass
        if fs is None:
            fsspec = import_optional_dependency("fsspec")
            fs, path_or_handle = fsspec.core.url_to_fs(
                path_or_handle, **(storage_options or {})
            )
    elif storage_options and (not is_url(path_or_handle) or mode != "rb"):
        raise ValueError("storage_options passed with buffer, or non-supported URL")

    handles: IOHandles[bytes] | None = None
    if (
        not fs
        and not is_dir
        and isinstance(path_or_handle, str)
        and not os.path.isdir(path_or_handle)
    ):
        handles = get_handle(
            path_or_handle, mode, is_text=False, storage_options=storage_options
        )
        fs = None
        path_or_handle = handles.handle
    return path_or_handle, handles, fs


class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None:
        if not isinstance(df, DataFrame):
            raise ValueError("to_parquet only supports IO with DataFrames")

    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: str | None,
        **kwargs: Any,
    ) -> None:
        raise AbstractMethodError(self)

    def read(
        self, path: FilePath | ReadBuffer[bytes], columns: list[str] | None = None, **kwargs: Any
    ) -> DataFrame:
        raise AbstractMethodError(self)


class PyArrowImpl(BaseImpl):
    def __init__(self) -> None:
        import_optional_dependency(
            "pyarrow", extra="pyarrow is required for parquet support."
        )
        import pyarrow.parquet
        import pandas.core.arrays.arrow.extension_types  # noqa: F401
        self.api: Any = pyarrow  # type: ignore

    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: str | None = "snappy",
        index: bool | None = None,
        storage_options: StorageOptions | None = None,
        partition_cols: list[str] | None = None,
        filesystem: Any = None,
        **kwargs: Any,
    ) -> None:
        self.validate_dataframe(df)
        from_pandas_kwargs: dict[str, Any] = {"schema": kwargs.pop("schema", None)}
        if index is not None:
            from_pandas_kwargs["preserve_index"] = index

        table: Any = self.api.Table.from_pandas(df, **from_pandas_kwargs)
        if df.attrs:
            df_metadata = {"PANDAS_ATTRS": json.dumps(df.attrs)}
            existing_metadata = table.schema.metadata or {}
            merged_metadata = {**existing_metadata, **df_metadata}
            table = table.replace_schema_metadata(merged_metadata)
        path_or_handle, handles, filesystem = _get_path_or_handle(
            path, filesystem, storage_options=storage_options, mode="wb", is_dir=partition_cols is not None
        )
        if (
            isinstance(path_or_handle, io.BufferedWriter)
            and hasattr(path_or_handle, "name")
            and isinstance(path_or_handle.name, (str, bytes))
        ):
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
                    table,
                    path_or_handle,
                    compression=compression,
                    filesystem=filesystem,
                    **kwargs,
                )
        finally:
            if handles is not None:
                handles.close()

    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = None,
        filters: list[tuple[Any, ...]] | list[list[tuple[Any, ...]]] | None = None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        storage_options: StorageOptions | None = None,
        filesystem: Any = None,
        to_pandas_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        kwargs["use_pandas_metadata"] = True
        path_or_handle, handles, filesystem = _get_path_or_handle(
            path, filesystem, storage_options=storage_options, mode="rb"
        )
        try:
            pa_table: Any = self.api.parquet.read_table(
                path_or_handle,
                columns=columns,
                filesystem=filesystem,
                filters=filters,
                **kwargs,
            )
            with catch_warnings():
                filterwarnings("ignore", "make_block is deprecated", DeprecationWarning)
                result: DataFrame = arrow_table_to_pandas(
                    pa_table,
                    dtype_backend=dtype_backend,
                    to_pandas_kwargs=to_pandas_kwargs,
                )
            if pa_table.schema.metadata:
                if b"PANDAS_ATTRS" in pa_table.schema.metadata:
                    df_metadata = pa_table.schema.metadata[b"PANDAS_ATTRS"]
                    result.attrs = json.loads(df_metadata)
            return result
        finally:
            if handles is not None:
                handles.close()


class FastParquetImpl(BaseImpl):
    def __init__(self) -> None:
        fastparquet = import_optional_dependency(
            "fastparquet", extra="fastparquet is required for parquet support."
        )
        self.api: Any = fastparquet

    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: Literal["snappy", "gzip", "brotli"] | None = "snappy",
        index: bool | None = None,
        partition_cols: list[str] | None = None,
        storage_options: StorageOptions | None = None,
        filesystem: Any = None,
        **kwargs: Any,
    ) -> None:
        self.validate_dataframe(df)
        if "partition_on" in kwargs and partition_cols is not None:
            raise ValueError(
                "Cannot use both partition_on and "
                "partition_cols. Use partition_cols for partitioning data"
            )
        if "partition_on" in kwargs:
            partition_cols = kwargs.pop("partition_on")
        if partition_cols is not None:
            kwargs["file_scheme"] = "hive"
        if filesystem is not None:
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )
        path_str: str = stringify_path(path)
        if is_fsspec_url(path_str):
            fsspec = import_optional_dependency("fsspec")
            kwargs["open_with"] = lambda path, _: fsspec.open(
                path, "wb", **(storage_options or {})
            ).open()
        elif storage_options:
            raise ValueError(
                "storage_options passed with file object or non-fsspec file path"
            )
        with catch_warnings(record=True):
            self.api.write(
                path_str,
                df,
                compression=compression,
                write_index=index,
                partition_on=partition_cols,
                **kwargs,
            )

    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = None,
        filters: list[tuple[Any, ...]] | list[list[tuple[Any, ...]]] | None = None,
        storage_options: StorageOptions | None = None,
        filesystem: Any = None,
        to_pandas_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        parquet_kwargs: dict[str, Any] = {}
        dtype_backend = kwargs.pop("dtype_backend", lib.no_default)
        parquet_kwargs["pandas_nulls"] = False
        if dtype_backend is not lib.no_default:
            raise ValueError(
                "The 'dtype_backend' argument is not supported for the "
                "fastparquet engine"
            )
        if filesystem is not None:
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )
        if to_pandas_kwargs is not None:
            raise NotImplementedError(
                "to_pandas_kwargs is not implemented for the fastparquet engine."
            )
        path_str: str = stringify_path(path)
        handles: IOHandles[bytes] | None = None
        if is_fsspec_url(path_str):
            fsspec = import_optional_dependency("fsspec")
            parquet_kwargs["fs"] = fsspec.open(path_str, "rb", **(storage_options or {})).fs
        elif isinstance(path_str, str) and not os.path.isdir(path_str):
            handles = get_handle(
                path_str, "rb", is_text=False, storage_options=storage_options
            )
            path_str = handles.handle
        try:
            parquet_file: Any = self.api.ParquetFile(path_str, **parquet_kwargs)
            with catch_warnings():
                filterwarnings("ignore", "make_block is deprecated", DeprecationWarning)
                return parquet_file.to_pandas(columns=columns, filters=filters, **kwargs)
        finally:
            if handles is not None:
                handles.close()


@doc(storage_options=_shared_docs["storage_options"])
def to_parquet(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = None,
    engine: str = "auto",
    compression: str | None = "snappy",
    index: bool | None = None,
    storage_options: StorageOptions | None = None,
    partition_cols: list[str] | None = None,
    filesystem: Any = None,
    **kwargs: Any,
) -> bytes | None:
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]
    impl: BaseImpl = get_engine(engine)
    path_or_buf: FilePath | WriteBuffer[bytes] = io.BytesIO() if path is None else path
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


@doc(storage_options=_shared_docs["storage_options"])
def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: str = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    filesystem: Any = None,
    filters: list[tuple[Any, ...]] | list[list[tuple[Any, ...]]] | None = None,
    to_pandas_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DataFrame:
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