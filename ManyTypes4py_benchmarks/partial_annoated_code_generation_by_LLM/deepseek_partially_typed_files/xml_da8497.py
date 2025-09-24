"""
:mod:``pandas.io.xml`` is a module for reading XML.
"""
from __future__ import annotations
import io
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
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
    """
    Internal subclass to parse XML into DataFrames.

    Parameters
    ----------
    path_or_buffer : a valid JSON ``str``, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file.

    xpath : str or regex
        The ``XPath`` expression to parse required set of nodes for
        migration to :class:`~pandas.DataFrame`. ``etree`` supports limited ``XPath``.

    namespaces : dict
        The namespaces defined in XML document (``xmlns:namespace='URI'``)
        as dicts with key being namespace and value the URI.

    elems_only : bool
        Parse only the child elements at the specified ``xpath``.

    attrs_only : bool
        Parse only the attributes at the specified ``xpath``.

    names : list
        Column names for :class:`~pandas.DataFrame` of parsed XML data.

    dtype : dict
        Data type for data or columns. E.g. {{'a': np.float64,
        'b': np.int32, 'c': 'Int64'}}

        .. versionadded:: 1.5.0

    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels.

        .. versionadded:: 1.5.0

    parse_dates : bool or list of int or names or list of lists or dict
        Converts either index or select columns to datetimes

        .. versionadded:: 1.5.0

    encoding : str
        Encoding of xml object or document.

    stylesheet : str or file-like
        URL, file, file-like object, or a raw string containing XSLT,
        ``etree`` does not support XSLT but retained for consistency.

    iterparse : dict, optional
        Dict with row element as key and list of descendant elements
        and/or attributes as value to be retrieved in iterparsing of
        XML document.

        .. versionadded:: 1.5.0

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    See also
    --------
    pandas.io.xml._EtreeFrameParser
    pandas.io.xml._LxmlFrameParser

    Notes
    -----
    To subclass this class effectively you must override the following methods:`
        * :func:`parse_data`
        * :func:`_parse_nodes`
        * :func:`_iterparse_nodes`
        * :func:`_parse_doc`
        * :func:`_validate_names`
        * :func:`_validate_path`


    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(self, path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str], xpath: str, namespaces: Optional[Dict[str, str]], elems_only: bool, attrs_only: bool, names: Optional[Sequence[str]], dtype: Optional[DtypeArg], converters: Optional[ConvertersArg], parse_dates: Optional[ParseDatesArg], encoding: Optional[str], stylesheet: Optional[FilePath | ReadBuffer[bytes] | ReadBuffer[str]], iterparse: Optional[Dict[str, List[str]]], compression: CompressionOptions, storage_options: StorageOptions) -> None:
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
        self.compression: CompressionOptions = compression
        self.storage_options = storage_options

    def parse_data(self) -> List[Dict[str, Optional[str]]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, parse and return specific nodes.
        """
        raise AbstractMethodError(self)

    def _parse_nodes(self, elems: List[Any]) -> List[Dict[str, Optional[str]]]:
        """
        Parse xml nodes.

        This method will parse the children and attributes of elements
        in ``xpath``, conditionally for only elements, only attributes
        or both while optionally renaming node names.

        Raises
        ------
        ValueError
            * If only elements and only attributes are specified.

        Notes
        -----
        Namespace URIs will be removed from return node values. Also,
        elements with missing children or attributes compared to siblings
        will have optional keys filled with None values.
        """
        dicts: List[Dict[str, Optional[str]]]
        if self.elems_only and self.attrs_only:
            raise ValueError('Either element or attributes can be parsed not both.')
        if self.elems_only:
            if self.names:
                dicts = [{**({el.tag: el.text} if el.text and (not el.text.isspace()) else {}), **{nm: ch.text if ch.text else None for (nm, ch) in zip(self.names, el.findall('*'))}} for el in elems]
            else:
                dicts = [{ch.tag: ch.text if ch.text else None for ch in el.findall('*')} for el in elems]
        elif self.attrs_only:
            dicts = [{k: v if v else None for (k, v) in el.attrib.items()} for el in elems]
        elif self.names:
            dicts = [{**el.attrib, **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}), **{nm: ch.text if ch.text else None for (nm, ch) in zip(self.names, el.findall('*'))}} for el in elems]
        else:
            dicts = [{**el.attrib, **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}), **{ch.tag: ch.text if ch.text else None for ch in el.findall('*')}} for el in elems]
        dicts = [{k.split('}')[1] if '}' in k else k: v for (k, v) in d.items()} for d in dicts]
        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]
        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]
        return dicts

    def _iterparse_nodes(self, iterparse: Callable) -> List[Dict[str, Optional[str]]]:
        """
        Iterparse xml nodes.

        This method will read in local disk, decompressed XML files for elements
        and underlying descendants using iterparse, a method to iterate through
        an XML tree without holding entire XML tree in memory.

        Raises
        ------
        TypeError
            * If ``iterparse`` is not a dict or its dict value is not list-like.
        ParserError
            * If ``path_or_buffer`` is not a physical file on disk or file-like object.
            * If no data is returned from selected items in ``iterparse``.

        Notes
        -----
        Namespace URIs will be removed from return node values. Also,
        elements with missing children or attributes in submitted list
        will have optional keys filled with None values.
        """
        dicts: List[Dict[str, Optional[str]]] = []
        row: Optional[Dict[str, Optional[str]]] = None
        if not isinstance(self.iterparse, dict):
            raise TypeError(f'{type(self.iterparse).__name__} is not a valid type for iterparse')
        row_node = next(iter(self.iterparse.keys())) if self.iterparse else ''
        if not is_list_like(self.iterparse[row_node]):
            raise TypeError(f'{type(self.iterparse[row_node])} is not a valid type for value in iterparse')
        if not hasattr(self.path_or_buffer, 'read') and (not isinstance(self.path_or_buffer, (str, PathLike)) or is_url(self.path_or_buffer) or is_fsspec_url(self.path_or_buffer) or (isinstance(self.path_or_buffer, str) and self.path_or_buffer.startswith(('<?xml', '<'))) or (infer_compression(self.path_or_buffer, 'infer') is not None)):
            raise ParserError('iterparse is designed for large XML files that are fully extracted on local disk and not as compressed files or online sources.')
        iterparse_repeats = len(self.iterparse[row_node]) != len(set(self.iterparse[row_node]))
        for (event, elem) in iterparse(self.path_or_buffer, events=('start', 'end')):
            curr_elem = elem.tag.split('}')[1] if '}' in elem.tag else elem.tag
            if event == 'start':
                if curr_elem == row_node:
                    row = {}
            if row is not None:
                if self.names and iterparse_repeats:
                    for (col, nm) in zip(self.iterparse[row_node], self.names):
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
            if event == 'end':
                if curr_elem == row_node and row is not None:
                    dicts.append(row)
                    row = None
                elem.clear()
                if hasattr(elem, 'getprevious'):
                    while elem.getprevious() is not None and elem.getparent() is not None:
                        del elem.getparent()[0]
        if dicts == []:
            raise ParserError('No result from selected items in iterparse.')
        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]
        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]
        return dicts

    def _validate_path(self) -> List[Any]:
        """
        Validate ``xpath``.

        This method checks for syntax, evaluation, or empty nodes return.

        Raises
        ------
        SyntaxError
            * If xpah is not supported or issues with namespaces.

        ValueError
            * If xpah does not return any nodes.
        """
        raise AbstractMethodError(self)

    def _validate_names(self) -> None:
        """
        Validate names.

        This method will check if names is a list-like and aligns
        with length of parse nodes.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
        raise AbstractMethodError(self)

    def _parse_doc(self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]) -> Element | etree._Element:
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

    def parse_data(self) -> List[Dict[str, Optional[str]]]:
        from xml.etree.ElementTree import iterparse
        if self.stylesheet is not None:
            raise ValueError('To use stylesheet, you need lxml installed and selected as parser.')
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()
        self._validate_names()
        xml_dicts: List[Dict[str, Optional[str]]] = self._parse_nodes(elems) if self.iterparse is None else self._iterparse_nodes(iterparse)
        return xml_dicts

    def _validate_path(self) -> List[Any]:
        """
        Notes
        -----
        ``etree`` supports limited ``XPath``. If user attempts a more complex
        expression syntax error will raise.
        """
        msg = 'xpath does not return any nodes or attributes. Be sure to specify in `xpath` the parent nodes of children and attributes to parse. If document uses namespaces denoted with xmlns, be sure to define namespaces and use them in xpath.'
        try:
            elems = self.xml_doc.findall(self.xpath, namespaces=self.namespaces)
            children = [ch for el in elems for ch in el.findall('*')]
            attrs = {k: v for el in elems for (k, v) in el.attrib.items()}
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
            raise SyntaxError('You have used an incorrect or unsupported XPath expression for etree library or you used an undeclared namespace prefix.') from err
        return elems

    def _validate_names(self) -> None:
        children: List[Any]
        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                parent = self.xml_doc.find(self.xpath, namespaces=self.namespaces)
                children = parent.findall('*') if parent is not None else []
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError('names does not match length of child elements in xpath.')
            else:
                raise TypeError(f'{type(self.names).__name__} is not a valid type for names')

    def _parse_doc(self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]) -> Element:
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

    def parse_data(self) -> List[Dict[str, Optional[str]]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, optionally parse and run XSLT,
        and parse original or transformed XML and return specific nodes.
        """
        from lxml.etree import iterparse
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            if self.stylesheet:
                self.xsl_doc = self._parse_doc(self.stylesheet)
                self.xml_doc = self._transform_doc()
            elems = self._validate_path()
        self._validate_names()
        xml_dicts: List[Dict[str, Optional[str]]] = self._parse_nodes(elems) if self.iterparse is None else self._iterparse_nodes(iterparse)
        return xml_dicts

    def _validate_path(self) -> List[Any]:
        msg = 'xpath does not return any nodes or attributes. Be sure to specify in `xpath` the parent nodes of children and attributes to parse. If document uses namespaces denoted with xmlns, be sure to define namespaces and use them in xpath.'
        elems = self.xml_doc.xpath(self.xpath, namespaces=self.namespaces)
        children = [ch for el in elems for ch in el.xpath('*')]
        attrs = {k: v for el in elems for (k, v) in el.attrib.items()}
        if elems == []:
            raise ValueError(msg)
        if elems != []:
            if self.elems_only and children == []:
                raise ValueError(msg)
            if self.attrs_only and attrs == {}:
                raise ValueError(msg)
            if children ==