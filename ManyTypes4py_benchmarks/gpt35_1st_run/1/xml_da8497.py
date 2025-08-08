from __future__ import annotations
import io
from os import PathLike
from typing import TYPE_CHECKING, Any, List, Dict, Union
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError, ParserError
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_list_like
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle, infer_compression, is_fsspec_url, is_url, stringify_path
from pandas.io.parsers import TextParser
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from xml.etree.ElementTree import Element
    from lxml import etree
    from pandas._typing import CompressionOptions, ConvertersArg, DtypeArg, DtypeBackend, FilePath, ParseDatesArg, ReadBuffer, StorageOptions, XMLParsers
    from pandas import DataFrame

@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buffer')
class _XMLFrameParser:
    def __init__(self, path_or_buffer: Union[str, PathLike, Any], xpath: Union[str, Any], namespaces: Dict[str, str], elems_only: bool, attrs_only: bool, names: List[str], dtype: Dict[str, Any], converters: Dict[Union[int, str], Callable], parse_dates: Union[bool, List[Union[int, str], List[Union[int, str]], Dict[str, List[Union[int, str]]]], encoding: str, stylesheet: Union[str, Any], iterparse: Dict[str, List[Union[str]]], compression: CompressionOptions, storage_options: StorageOptions) -> None:
        ...

    def parse_data(self) -> None:
        ...

    def _parse_nodes(self, elems: List[Element]) -> List[Dict[str, Any]]:
        ...

    def _iterparse_nodes(self, iterparse: Dict[str, List[Union[str]]]) -> List[Dict[str, Any]]:
        ...

    def _validate_path(self) -> None:
        ...

    def _validate_names(self) -> None:
        ...

    def _parse_doc(self, raw_doc: Any) -> None:
        ...

class _EtreeFrameParser(_XMLFrameParser):
    def parse_data(self) -> None:
        ...

    def _validate_path(self) -> None:
        ...

    def _validate_names(self) -> None:
        ...

    def _parse_doc(self, raw_doc: Any) -> None:
        ...

class _LxmlFrameParser(_XMLFrameParser):
    def parse_data(self) -> None:
        ...

    def _validate_path(self) -> None:
        ...

    def _validate_names(self) -> None:
        ...

    def _parse_doc(self, raw_doc: Any) -> None:
        ...

    def _transform_doc(self) -> None:
        ...

def get_data_from_filepath(filepath_or_buffer: Union[str, PathLike, Any], encoding: str, compression: CompressionOptions, storage_options: StorageOptions) -> Any:
    ...

def preprocess_data(data: Any) -> Any:
    ...

def _data_to_frame(data: List[Dict[str, Any]], **kwargs: Any) -> DataFrame:
    ...

def _parse(path_or_buffer: Union[str, PathLike, Any], xpath: Union[str, Any], namespaces: Dict[str, str], elems_only: bool, attrs_only: bool, names: List[str], dtype: Dict[str, Any], converters: Dict[Union[int, str], Callable], parse_dates: Union[bool, List[Union[int, str], List[Union[int, str]], Dict[str, List[Union[int, str]]]], encoding: str, parser: str, stylesheet: Union[str, Any], iterparse: Dict[str, List[Union[str]]], compression: CompressionOptions, storage_options: StorageOptions, dtype_backend: DtypeBackend, **kwargs: Any) -> DataFrame:
    ...

@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buffer')
def read_xml(path_or_buffer: Union[str, PathLike, Any], *, xpath: str = './*', namespaces: Dict[str, str] = None, elems_only: bool = False, attrs_only: bool = False, names: List[str] = None, dtype: Dict[str, Any] = None, converters: Dict[Union[int, str], Callable] = None, parse_dates: Union[bool, List[Union[int, str], List[Union[int, str]], Dict[str, List[Union[int, str]]]] = None, encoding: str = 'utf-8', parser: str = 'lxml', stylesheet: Union[str, Any] = None, iterparse: Dict[str, List[Union[str]]] = None, compression: CompressionOptions = 'infer', storage_options: StorageOptions = None, dtype_backend: DtypeBackend = lib.no_default) -> DataFrame:
    ...
