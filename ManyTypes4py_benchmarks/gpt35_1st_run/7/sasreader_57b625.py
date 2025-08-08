from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, overload
from pandas.util._decorators import doc
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import stringify_path
if TYPE_CHECKING:
    from collections.abc import Hashable
    from types import TracebackType
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer, Self
    from pandas import DataFrame

class SASReader(Iterator[DataFrame], ABC):
    @abstractmethod
    def read(self, nrows=None) -> DataFrame:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> SASReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

@overload
def read_sas(filepath_or_buffer: FilePath, *, format: str = ..., index: Hashable = ..., encoding: str = ..., chunksize: int = ..., iterator: bool = ..., compression: CompressionOptions = ...) -> DataFrame:
    ...

@overload
def read_sas(filepath_or_buffer: FilePath, *, format: str = ..., index: Hashable = ..., encoding: str = ..., chunksize: int = ..., iterator: bool = ..., compression: CompressionOptions = ...) -> DataFrame:
    ...

@doc(decompression_options=_shared_docs['decompression_options'] % 'filepath_or_buffer')
def read_sas(filepath_or_buffer: FilePath, *, format: str = None, index: Hashable = None, encoding: str = None, chunksize: int = None, iterator: bool = False, compression: CompressionOptions = 'infer') -> DataFrame:
    ...
