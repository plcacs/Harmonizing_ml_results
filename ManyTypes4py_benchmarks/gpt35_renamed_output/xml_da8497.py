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
    def __init__(self, path_or_buffer: Union[str, PathLike, io.BufferedIOBase], xpath: str, namespaces: Dict[str, str], elems_only: bool, attrs_only: bool, names: List[str], dtype: Dict[str, Any], converters: Dict[Union[int, str], Callable], parse_dates: Union[bool, List[Union[int, str], List[List[Union[int, str]]], Dict[str, List[Union[int, str]]]], encoding: str, stylesheet: Union[str, io.BufferedIOBase], iterparse: Dict[str, List[str]], compression: CompressionOptions, storage_options: StorageOptions) -> None:
        self.path_or_buffer = path_or_buffer
        self.xpath = xpath
        self.namespaces = namespaces
        self.elems_only = elems_only
        self.attrs_only = attrs_only
        self.names = names
        self.dtype = dtype
        self.converters = converters
        self.parse_dates = parse_dates
        self.encoding = encoding
        self.stylesheet = stylesheet
        self.iterparse = iterparse
        self.compression = compression
        self.storage_options = storage_options

    def func_7l6ojxu6(self) -> Any:
        ...

    def func_3awbkhgx(self, elems: List[Element]) -> List[Dict[str, Any]]:
        ...

    def func_elvk2x6a(self, iterparse: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        ...

    def func_ob090zks(self) -> Any:
        ...

    def func_tjfk4tyd(self) -> Any:
        ...

    def func_pecre3gg(self, raw_doc: Union[str, bytes]) -> Any:
        ...


class _EtreeFrameParser(_XMLFrameParser):
    def func_7l6ojxu6(self) -> Any:
        ...

    def func_ob090zks(self) -> Any:
        ...

    def func_tjfk4tyd(self) -> Any:
        ...

    def func_pecre3gg(self, raw_doc: Union[str, bytes]) -> Any:
        ...


class _LxmlFrameParser(_XMLFrameParser):
    def func_7l6ojxu6(self) -> Any:
        ...

    def func_ob090zks(self) -> Any:
        ...

    def func_tjfk4tyd(self) -> Any:
        ...

    def func_pecre3gg(self, raw_doc: Union[str, bytes]) -> Any:
        ...

    def func_bkgj4avi(self) -> Any:
        ...


def func_qc51i70r(filepath_or_buffer: Union[str, PathLike], encoding: str, compression: CompressionOptions, storage_options: StorageOptions) -> Any:
    ...


def func_5d1iedzx(data: Union[str, bytes]) -> Union[io.StringIO, io.BytesIO]:
    ...


def func_z8x4jdv1(data: List[Dict[str, Any]], **kwargs: Any) -> DataFrame:
    ...


def func_va2nns90(path_or_buffer: Union[str, PathLike, io.BufferedIOBase], xpath: str, namespaces: Dict[str, str], elems_only: bool, attrs_only: bool, names: List[str], dtype: Dict[str, Any], converters: Dict[Union[int, str], Callable], parse_dates: Union[bool, List[Union[int, str], List[List[Union[int, str]]], Dict[str, List[Union[int, str]]]], encoding: str, parser: str, stylesheet: Union[str, io.BufferedIOBase], iterparse: Dict[str, List[str]], compression: CompressionOptions, storage_options: StorageOptions, dtype_backend: DtypeBackend, **kwargs: Any) -> DataFrame:
    ...


@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buffer')
def func_z6gu387h(path_or_buffer: Union[str, PathLike, io.BufferedIOBase], xpath: str = './*', namespaces: Dict[str, str] = None, elems_only: bool = False, attrs_only: bool = False, names: List[str] = None, dtype: Dict[str, Any] = None, converters: Dict[Union[int, str], Callable] = None, parse_dates: Union[bool, List[Union[int, str], List[List[Union[int, str]]], Dict[str, List[Union[int, str]]]] = None, encoding: str = 'utf-8', parser: str = 'lxml', stylesheet: Union[str, io.BufferedIOBase] = None, iterparse: Dict[str, List[str]] = None, compression: CompressionOptions = 'infer', storage_options: StorageOptions = None, dtype_backend: DtypeBackend = lib.no_default) -> DataFrame:
    ...
