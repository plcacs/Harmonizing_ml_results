from __future__ import annotations
import io
from os import PathLike
from typing import TYPE_CHECKING, Any, IO, Optional, Union, List, Dict, Callable

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
    from collections.abc import Sequence
    from xml.etree.ElementTree import Element
    from lxml import etree
    from pandas._typing import CompressionOptions, ConvertersArg, DtypeArg, DtypeBackend, FilePath, ParseDatesArg, ReadBuffer, StorageOptions, XMLParsers
    from pandas import DataFrame


@doc(storage_options=_shared_docs["storage_options"], decompression_options=_shared_docs["decompression_options"] % "path_or_buffer")
class _XMLFrameParser:
    """
    Internal subclass to parse XML into DataFrames.

    Parameters
    ----------
    path_or_buffer : a valid XML ``str``, path object or file-like object
    xpath : str or regex
    namespaces : dict
    elems_only : bool
    attrs_only : bool
    names : list
    dtype : dict
    converters : dict, optional
    parse_dates : bool or list of int or names or list of lists or dict
    encoding : str
    stylesheet : str or file-like
    iterparse : dict, optional
    compression : Any
    storage_options : dict, optional

    Notes
    -----
    To subclass this class effectively you must override the following methods:
        * :func:`parse_data`
        * :func:`_parse_nodes`
        * :func:`_iterparse_nodes`
        * :func:`_parse_doc`
        * :func:`_validate_names`
        * :func:`_validate_path`
    """

    def __init__(
        self,
        path_or_buffer: Union[str, PathLike[Any], IO[Any]],
        xpath: str,
        namespaces: Optional[Dict[str, str]],
        elems_only: bool,
        attrs_only: bool,
        names: Optional[List[str]],
        dtype: Optional[Dict[str, Any]],
        converters: Optional[Dict[Any, Callable[[Any], Any]]],
        parse_dates: Optional[Union[bool, List[Any]]],
        encoding: Optional[str],
        stylesheet: Optional[Union[str, IO[Any]]],
        iterparse: Optional[Dict[str, List[Any]]],
        compression: Optional[Union[str, bool]],
        storage_options: Optional[Dict[str, Any]]
    ) -> None:
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

    def parse_data(self) -> List[Dict[str, Any]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, parse and return specific nodes.
        """
        raise AbstractMethodError(self)

    def _parse_nodes(self, elems: List[Element]) -> List[Dict[str, Any]]:
        """
        Parse xml nodes.

        This method will parse the children and attributes of elements
        in ``xpath``, conditionally for only elements, only attributes
        or both while optionally renaming node names.
        """
        if self.elems_only and self.attrs_only:
            raise ValueError("Either element or attributes can be parsed not both.")
        if self.elems_only:
            if self.names:
                dicts = [
                    {
                        **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}),
                        **{nm: ch.text if ch.text else None for nm, ch in zip(self.names, el.findall("*"))},
                    }
                    for el in elems
                ]
            else:
                dicts = [
                    {ch.tag: ch.text if ch.text else None for ch in el.findall("*")} for el in elems
                ]
        elif self.attrs_only:
            dicts = [
                {k: v if v else None for k, v in el.attrib.items()} for el in elems
            ]
        elif self.names:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}),
                    **{nm: ch.text if ch.text else None for nm, ch in zip(self.names, el.findall("*"))},
                }
                for el in elems
            ]
        else:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}),
                    **{ch.tag: ch.text if ch.text else None for ch in el.findall("*")},
                }
                for el in elems
            ]
        dicts = [
            {k.split("}")[1] if "}" in k else k: v for k, v in d.items()} for d in dicts
        ]
        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]
        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]
        return dicts

    def _iterparse_nodes(self, iterparse: Callable[..., Any]) -> List[Dict[str, Any]]:
        """
        Iterparse xml nodes.

        This method will read in local disk, decompressed XML files for elements
        and underlying descendants using iterparse, a method to iterate through
        an XML tree without holding entire XML tree in memory.
        """
        dicts: List[Dict[str, Any]] = []
        row: Optional[Dict[str, Any]] = None
        if not isinstance(self.iterparse, dict):
            raise TypeError(f"{type(self.iterparse).__name__} is not a valid type for iterparse")
        row_node = next(iter(self.iterparse.keys())) if self.iterparse else ""
        if not is_list_like(self.iterparse[row_node]):
            raise TypeError(f"{type(self.iterparse[row_node])} is not a valid type for value in iterparse")
        if not hasattr(self.path_or_buffer, "read") and (
            not isinstance(self.path_or_buffer, (str, PathLike)) or is_url(self.path_or_buffer)
            or is_fsspec_url(self.path_or_buffer)
            or (isinstance(self.path_or_buffer, str) and self.path_or_buffer.startswith(("<?xml", "<")))
            or (infer_compression(self.path_or_buffer, "infer") is not None)
        ):
            raise ParserError("iterparse is designed for large XML files that are fully extracted on local disk and not as compressed files or online sources.")
        iterparse_repeats = len(self.iterparse[row_node]) != len(set(self.iterparse[row_node]))
        for event, elem in iterparse(self.path_or_buffer, events=("start", "end")):
            curr_elem = elem.tag.split("}")[1] if "}" in elem.tag else elem.tag
            if event == "start":
                if curr_elem == row_node:
                    row = {}
            if row is not None:
                if self.names and iterparse_repeats:
                    for col, nm in zip(self.iterparse[row_node], self.names):
                        if curr_elem == col:
                            elem_val = elem.text if elem.text else None
                            if elem_val not in row.values() and nm not in row:
                                row[nm] = elem_val
                        if col in elem.attrib:
                            if elem.attrib[col] not in row.values() and nm not in row:
                                row[nm] = elem.attrib[col]
                else:
                    for col in self.iterparse[row_node]:
                        if curr_elem == col:
                            row[col] = elem.text if elem.text else None
                        if col in elem.attrib:
                            row[col] = elem.attrib[col]
            if event == "end":
                if curr_elem == row_node and row is not None:
                    dicts.append(row)
                    row = None
                elem.clear()
                if hasattr(elem, "getprevious"):
                    while elem.getprevious() is not None and elem.getparent() is not None:
                        del elem.getparent()[0]
        if dicts == []:
            raise ParserError("No result from selected items in iterparse.")
        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]
        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]
        return dicts

    def _validate_path(self) -> List[Any]:
        """
        Validate ``xpath``.

        This method checks for syntax, evaluation, or empty nodes return.
        """
        raise AbstractMethodError(self)

    def _validate_names(self) -> None:
        """
        Validate names.

        This method will check if names is a list-like and aligns
        with length of parse nodes.
        """
        raise AbstractMethodError(self)

    def _parse_doc(self, raw_doc: Union[str, IO[Any]]) -> Any:
        """
        Build tree from path_or_buffer.

        This method will parse XML object into tree
        either from string/bytes or file location.
        """
        raise AbstractMethodError(self)


class _EtreeFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into DataFrames with the Python
    standard library XML module: `xml.etree.ElementTree`.
    """

    def parse_data(self) -> List[Dict[str, Any]]:
        from xml.etree.ElementTree import iterparse
        if self.stylesheet is not None:
            raise ValueError("To use stylesheet, you need lxml installed and selected as parser.")
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()
        else:
            elems = []  # This will not be used if iterparse is provided.
        self._validate_names()
        xml_dicts = self._parse_nodes(elems) if self.iterparse is None else self._iterparse_nodes(iterparse)
        return xml_dicts

    def _validate_path(self) -> List[Any]:
        msg = "xpath does not return any nodes or attributes. Be sure to specify in `xpath` the parent nodes of children and attributes to parse. If document uses namespaces denoted with xmlns, be sure to define namespaces and use them in xpath."
        try:
            elems = self.xml_doc.findall(self.xpath, namespaces=self.namespaces)
            children = [ch for el in elems for ch in el.findall("*")]
            attrs = {k: v for el in elems for k, v in el.attrib.items()}
            if elems is None:
                raise ValueError(msg)
            if elems is not None:
                if self.elems_only and children == []:
                    raise ValueError(msg)
                if self.attrs_only and attrs == {}:
                    raise ValueError(msg)
                if children == [] and attrs == {}:
                    raise ValueError(msg)
        except (KeyError, SyntaxError) as err:
            raise SyntaxError("You have used an incorrect or unsupported XPath expression for etree library or you used an undeclared namespace prefix.") from err
        return elems

    def _validate_names(self) -> None:
        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                parent = self.xml_doc.find(self.xpath, namespaces=self.namespaces)
                children = parent.findall("*") if parent is not None else []
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError("names does not match length of child elements in xpath.")
            else:
                raise TypeError(f"{type(self.names).__name__} is not a valid type for names")

    def _parse_doc(self, raw_doc: Union[str, IO[Any]]) -> Any:
        from xml.etree.ElementTree import XMLParser, parse
        handle_data = get_data_from_filepath(filepath_or_buffer=raw_doc, encoding=self.encoding, compression=self.compression, storage_options=self.storage_options)
        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            document = parse(xml_data, parser=curr_parser)
        return document.getroot()


class _LxmlFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into :class:`~pandas.DataFrame` with third-party
    full-featured XML library, ``lxml``, that supports
    ``XPath`` 1.0 and XSLT 1.0.
    """

    def parse_data(self) -> List[Dict[str, Any]]:
        from lxml.etree import iterparse
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            if self.stylesheet:
                self.xsl_doc = self._parse_doc(self.stylesheet)
                self.xml_doc = self._transform_doc()
            elems = self._validate_path()
        else:
            elems = []  # Not used when iterparse is provided.
        self._validate_names()
        xml_dicts = self._parse_nodes(elems) if self.iterparse is None else self._iterparse_nodes(iterparse)
        return xml_dicts

    def _validate_path(self) -> List[Any]:
        msg = "xpath does not return any nodes or attributes. Be sure to specify in `xpath` the parent nodes of children and attributes to parse. If document uses namespaces denoted with xmlns, be sure to define namespaces and use them in xpath."
        elems = self.xml_doc.xpath(self.xpath, namespaces=self.namespaces)
        children = [ch for el in elems for ch in el.xpath("*")]
        attrs = {k: v for el in elems for k, v in el.attrib.items()}
        if elems == []:
            raise ValueError(msg)
        if elems != []:
            if self.elems_only and children == []:
                raise ValueError(msg)
            if self.attrs_only and attrs == {}:
                raise ValueError(msg)
            if children == [] and attrs == {}:
                raise ValueError(msg)
        return elems

    def _validate_names(self) -> None:
        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                children = self.xml_doc.xpath(self.xpath + "[1]/*", namespaces=self.namespaces)
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError("names does not match length of child elements in xpath.")
            else:
                raise TypeError(f"{type(self.names).__name__} is not a valid type for names")

    def _parse_doc(self, raw_doc: Union[str, IO[Any]]) -> Any:
        from lxml.etree import XMLParser, fromstring, parse
        handle_data = get_data_from_filepath(filepath_or_buffer=raw_doc, encoding=self.encoding, compression=self.compression, storage_options=self.storage_options)
        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            if isinstance(xml_data, io.StringIO):
                if self.encoding is None:
                    raise TypeError("Can not pass encoding None when input is StringIO.")
                document = fromstring(xml_data.getvalue().encode(self.encoding), parser=curr_parser)
            else:
                document = parse(xml_data, parser=curr_parser)
        return document

    def _transform_doc(self) -> Any:
        """
        Transform original tree using stylesheet.

        This method will transform original xml using XSLT script into
        am ideally flatter xml document for easier parsing and migration
        to Data Frame.
        """
        from lxml.etree import XSLT
        transformer = XSLT(self.xsl_doc)
        new_doc = transformer(self.xml_doc)
        return new_doc


def get_data_from_filepath(
    filepath_or_buffer: Union[str, PathLike[Any], IO[Any]],
    encoding: Optional[str],
    compression: Optional[Union[str, bool]],
    storage_options: Optional[Dict[str, Any]]
) -> IO[Any]:
    """
    Extract raw XML data.
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    with get_handle(filepath_or_buffer, "r", encoding=encoding, compression=compression, storage_options=storage_options) as handle_obj:
        if hasattr(handle_obj.handle, "read"):
            data = handle_obj.handle.read()
            return preprocess_data(data)
        else:
            return handle_obj.handle


def preprocess_data(data: Union[str, bytes, IO[Any]]) -> IO[Any]:
    """
    Convert extracted raw data.
    """
    if isinstance(data, str):
        data = io.StringIO(data)
    elif isinstance(data, bytes):
        data = io.BytesIO(data)
    return data


def _data_to_frame(data: List[Dict[str, Any]], **kwargs: Any) -> DataFrame:
    """
    Convert parsed data to Data Frame.
    """
    tags = next(iter(data))
    nodes = [list(d.values()) for d in data]
    try:
        with TextParser(nodes, names=tags, **kwargs) as tp:
            return tp.read()
    except ParserError as err:
        raise ParserError("XML document may be too complex for import. Try to flatten document and use distinct element and attribute names.") from err


def _parse(
    path_or_buffer: Union[str, PathLike[Any], IO[Any]],
    xpath: str,
    namespaces: Optional[Dict[str, str]],
    elems_only: bool,
    attrs_only: bool,
    names: Optional[List[str]],
    dtype: Optional[Dict[str, Any]],
    converters: Optional[Dict[Any, Callable[[Any], Any]]],
    parse_dates: Optional[Union[bool, List[Any]]],
    encoding: Optional[str],
    parser: str,
    stylesheet: Optional[Union[str, IO[Any]]],
    iterparse: Optional[Dict[str, List[Any]]],
    compression: Optional[Union[str, bool]],
    storage_options: Optional[Dict[str, Any]],
    dtype_backend: Any = lib.no_default,
    **kwargs: Any
) -> DataFrame:
    """
    Call internal parsers.
    """
    if parser == "lxml":
        lxml = import_optional_dependency("lxml.etree", errors="ignore")
        if lxml is not None:
            p = _LxmlFrameParser(
                path_or_buffer,
                xpath,
                namespaces,
                elems_only,
                attrs_only,
                names,
                dtype,
                converters,
                parse_dates,
                encoding,
                stylesheet,
                iterparse,
                compression,
                storage_options,
            )
        else:
            raise ImportError("lxml not found, please install or use the etree parser.")
    elif parser == "etree":
        p = _EtreeFrameParser(
            path_or_buffer,
            xpath,
            namespaces,
            elems_only,
            attrs_only,
            names,
            dtype,
            converters,
            parse_dates,
            encoding,
            stylesheet,
            iterparse,
            compression,
            storage_options,
        )
    else:
        raise ValueError("Values for parser can only be lxml or etree.")
    data_dicts = p.parse_data()
    return _data_to_frame(data=data_dicts, dtype=dtype, converters=converters, parse_dates=parse_dates, dtype_backend=dtype_backend, **kwargs)


@doc(storage_options=_shared_docs["storage_options"], decompression_options=_shared_docs["decompression_options"] % "path_or_buffer")
def read_xml(
    path_or_buffer: Union[str, PathLike[Any], IO[Any]],
    *,
    xpath: str = "./*",
    namespaces: Optional[Dict[str, str]] = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    converters: Optional[Dict[Any, Callable[[Any], Any]]] = None,
    parse_dates: Optional[Union[bool, List[Any]]] = None,
    encoding: str = "utf-8",
    parser: str = "lxml",
    stylesheet: Optional[Union[str, IO[Any]]] = None,
    iterparse: Optional[Dict[str, List[Any]]] = None,
    compression: Optional[Union[str, bool]] = "infer",
    storage_options: Optional[Dict[str, Any]] = None,
    dtype_backend: Any = lib.no_default
) -> DataFrame:
    """
    Read XML document into a :class:`~pandas.DataFrame` object.
    """
    check_dtype_backend(dtype_backend)
    return _parse(
        path_or_buffer=path_or_buffer,
        xpath=xpath,
        namespaces=namespaces,
        elems_only=elems_only,
        attrs_only=attrs_only,
        names=names,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        encoding=encoding,
        parser=parser,
        stylesheet=stylesheet,
        iterparse=iterparse,
        compression=compression,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )