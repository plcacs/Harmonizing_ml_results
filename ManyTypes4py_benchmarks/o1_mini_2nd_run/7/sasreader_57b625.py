"""
Read SAS sas7bdat or xport files.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, overload, Optional, Union
from pandas.util._decorators import doc
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import stringify_path

if TYPE_CHECKING:
    from collections.abc import Hashable
    from types import TracebackType
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer, Self
    from pandas import DataFrame

class SASReader(Iterator[DataFrame], ABC):
    """
    Abstract class for XportReader and SAS7BDATReader.
    """

    @abstractmethod
    def read(self, nrows: Optional[int] = None) -> DataFrame:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: int | None = ...,
    iterator: bool = ...,
    compression: CompressionOptions | str = ...,
) -> DataFrame | SASReader | XportReader:
    ...

@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: int | None = ...,
    iterator: bool = ...,
    compression: CompressionOptions | str = ...,
) -> DataFrame | SASReader | XportReader:
    ...

@doc(decompression_options=_shared_docs['decompression_options'] % 'filepath_or_buffer')
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions | str = 'infer',
) -> DataFrame | SASReader | XportReader:
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.sas7bdat``.
    format : str {{'xport', 'sas7bdat'}} or None
        If None, file format is inferred from file extension. If 'xport' or
        'sas7bdat', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    encoding : str, default is None
        Encoding for text data.  If None, text data are stored as raw bytes.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.
    iterator : bool, defaults to False
        If True, returns an iterator for reading the file incrementally.
    {decompression_options}

    Returns
    -------
    DataFrame, SAS7BDATReader, or XportReader
        DataFrame if iterator=False and chunksize=None, else SAS7BDATReader
        or XportReader, file format is inferred from file extension.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into a pandas DataFrame.
    read_excel : Read an Excel file into a pandas DataFrame.
    read_spss : Read an SPSS file into a pandas DataFrame.
    read_orc : Load an ORC object into a pandas DataFrame.
    read_feather : Load a feather-format object into a pandas DataFrame.

    Examples
    --------
    >>> df = pd.read_sas("sas_data.sas7bdat")  # doctest: +SKIP
    """
    if format is None:
        buffer_error_msg = 'If this is a buffer object rather than a string name, you must specify a format string'
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        if not isinstance(filepath_or_buffer, str):
            raise ValueError(buffer_error_msg)
        fname = filepath_or_buffer.lower()
        if '.xpt' in fname:
            format = 'xport'
        elif '.sas7bdat' in fname:
            format = 'sas7bdat'
        else:
            raise ValueError(f'unable to infer format of SAS file from filename: {fname!r}')
    if format.lower() == 'xport':
        from pandas.io.sas.sas_xport import XportReader
        reader: SASReader = XportReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    elif format.lower() == 'sas7bdat':
        from pandas.io.sas.sas7bdat import SAS7BDATReader
        reader = SAS7BDATReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    else:
        raise ValueError('unknown SAS format')
    if iterator or chunksize:
        return reader
    with reader:
        return reader.read()
