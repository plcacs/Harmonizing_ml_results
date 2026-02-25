from __future__ import annotations

import io
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    IO,
    Union,
    Callable,
    Sequence,
)

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    ParserError,
)
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import is_list_like

from pandas.core.shared_docs import _shared_docs

from pandas.io.common import (
    get_handle,
    infer_compression,
    is_fsspec_url,
    is_url,
    stringify_path,
)
from pandas.io.parsers import TextParser

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    from lxml import etree

    from pandas._typing import (
        CompressionOptions,
        ConvertersArg,
        DtypeArg,
        DtypeBackend,
        FilePath,
        ParseDatesArg,
        ReadBuffer,
        StorageOptions,
        XMLParsers,
    )

    from pandas import DataFrame


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",
)
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
    To subclass this class effectively you must override the following methods:
        * :func:`parse_data`
        * :func:`_parse_nodes`
        * :func:`_iterparse_nodes`
        * :func:`_parse_doc`
        * :func:`_validate_names`
        * :func:`_validate_path`
    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(
        self,
        path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
        xpath: str,
        namespaces: dict[str, str] | None,
        elems_only: bool,
        attrs_only: bool,
        names: Sequence[str] | None,
        dtype: DtypeArg | None,
        converters: ConvertersArg | None,
        parse_dates: ParseDatesArg | None,
        encoding: str | None,
        stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,
        iterparse: dict[str, list[str]] | None,
        compression: CompressionOptions,
        storage_options: StorageOptions,
    ) -> None:
        self.path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str] = path_or_buffer
        self.xpath: str = xpath
        self.namespaces: dict[str, str] | None = namespaces
        self.elems_only: bool = elems_only
        self.attrs_only: bool = attrs_only
        self.names: Sequence[str] | None = names
        self.dtype: DtypeArg | None = dtype
        self.converters: ConvertersArg | None = converters
        self.parse_dates: ParseDatesArg | None = parse_dates
        self.encoding: str | None = encoding
        self.stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = stylesheet
        self.iterparse: dict[str, list[str]] | None = iterparse
        self.compression: CompressionOptions = compression
        self.storage_options: StorageOptions = storage_options

    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, parse and return specific nodes.
        """
        raise AbstractMethodError(self)

    def _parse_nodes(self, elems: list[Any]) -> list[dict[str, str | None]]:
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
        if self.elems_only and self.attrs_only:
            raise ValueError("Either element or attributes can be parsed not both.")
        if self.elems_only:
            if self.names:
                dicts = [
                    {
                        **(
                            {el.tag: el.text}
                            if el.text and not el.text.isspace()
                            else {}
                        ),
                        **{
                            nm: ch.text if ch.text else None
                            for nm, ch in zip(self.names, el.findall("*"))
                        },
                    }
                    for el in elems
                ]
            else:
                dicts = [
                    {ch.tag: ch.text if ch.text else None for ch in el.findall("*")}
                    for el in elems
                ]
        elif self.attrs_only:
            dicts = [
                {k: v if v else None for k, v in el.attrib.items()} for el in elems
            ]
        elif self.names:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),
                    **{
                        nm: ch.text if ch.text else None
                        for nm, ch in zip(self.names, el.findall("*"))
                    },
                }
                for el in elems
            ]
        else:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),
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

    def _iterparse_nodes(self, iterparse: Callable[..., Any]) -> list[dict[str, str | None]]:
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
        dicts: list[dict[str, str | None]] = []
        row: dict[str, str | None] | None = None
        if not isinstance(self.iterparse, dict):
            raise TypeError(
                f"{type(self.iterparse).__name__} is not a valid type for iterparse"
            )
        row_node: str = next(iter(self.iterparse.keys())) if self.iterparse else ""
        if not is_list_like(self.iterparse[row_node]):
            raise TypeError(
                f"{type(self.iterparse[row_node])} is not a valid type for value in iterparse"
            )
        iterparse_repeats = len(self.iterparse[row_node]) != len(set(self.iterparse[row_node]))
        for event, elem in iterparse(self.path_or_buffer, events=("start", "end")):
            curr_elem: str = elem.tag.split("}")[1] if "}" in elem.tag else elem.tag
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
                    while (elem.getprevious() is not None and elem.getparent() is not None):
                        del elem.getparent()[0]
        if dicts == []:
            raise ParserError("No result from selected items in iterparse.")
        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]
        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]
        return dicts

    def _validate_path(self) -> list[Any]:
        """
        Validate ``xpath``.

        This method checks for syntax, evaluation, or empty nodes return.

        Raises
        ------
        SyntaxError
            * If xpath is not supported or issues with namespaces.

        ValueError
            * If xpath does not return any nodes.
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

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element | etree._Element:
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

    def parse_data(self) -> list[dict[str, str | None]]:
        from xml.etree.ElementTree import iterparse
        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()
        self._validate_names()
        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )
        return xml_dicts

    def _validate_path(self) -> list[Any]:
        msg: str = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )
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
            raise SyntaxError(
                "You have used an incorrect or unsupported XPath "
                "expression for etree library or you used an "
                "undeclared namespace prefix."
            ) from err
        return elems

    def _validate_names(self) -> None:
        children: list[Any]
        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                parent = self.xml_doc.find(self.xpath, namespaces=self.namespaces)
                children = parent.findall("*") if parent is not None else []
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element:
        from xml.etree.ElementTree import XMLParser, parse
        handle_data: Union[io.StringIO, io.BytesIO] = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )
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

    def parse_data(self) -> list[dict[str, str | None]]:
        from lxml.etree import iterparse
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            if self.stylesheet:
                self.xsl_doc = self._parse_doc(self.stylesheet)
                self.xml_doc = self._transform_doc()
            elems = self._validate_path()
        self._validate_names()
        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )
        return xml_dicts

    def _validate_path(self) -> list[Any]:
        msg: str = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )
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
        children: list[Any]
        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                children = self.xml_doc.xpath(
                    self.xpath + "[1]/*", namespaces=self.namespaces
                )
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> etree._Element:
        from lxml.etree import XMLParser, fromstring, parse
        handle_data: Union[io.StringIO, io.BytesIO] = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )
        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            if isinstance(xml_data, io.StringIO):
                if self.encoding is None:
                    raise TypeError(
                        "Can not pass encoding None when input is StringIO."
                    )
                document = fromstring(
                    xml_data.getvalue().encode(self.encoding), parser=curr_parser
                )
            else:
                document = parse(xml_data, parser=curr_parser)
        return document

    def _transform_doc(self) -> etree._XSLTResultTree:
        """
        Transform original tree using stylesheet.

        This method will transform original xml using XSLT script into
        an ideally flatter xml document for easier parsing and migration
        to Data Frame.
        """
        from lxml.etree import XSLT
        transformer = XSLT(self.xsl_doc)
        new_doc = transformer(self.xml_doc)
        return new_doc


def get_data_from_filepath(
    filepath_or_buffer: FilePath | bytes | ReadBuffer[bytes] | ReadBuffer[str],
    encoding: str | None,
    compression: CompressionOptions,
    storage_options: StorageOptions,
) -> Union[io.StringIO, io.BytesIO]:
    """
    Extract raw XML data.

    The method accepts two input types:
        1. filepath (string-like)
        2. file-like object (e.g. open file object, StringIO)
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    with get_handle(
        filepath_or_buffer,
        "r",
        encoding=encoding,
        compression=compression,
        storage_options=storage_options,
    ) as handle_obj:
        if hasattr(handle_obj.handle, "read"):
            return preprocess_data(handle_obj.handle.read())
        else:
            return handle_obj.handle  # type: ignore

def preprocess_data(data: Union[str, bytes]) -> Union[io.StringIO, io.BytesIO]:
    """
    Convert extracted raw data.

    This method will return underlying data of extracted XML content.
    The data either has a `read` attribute (e.g. a file object or a
    StringIO/BytesIO) or is a string or bytes that is an XML document.
    """
    if isinstance(data, str):
        data_io: io.StringIO = io.StringIO(data)
        return data_io
    elif isinstance(data, bytes):
        data_io: io.BytesIO = io.BytesIO(data)
        return data_io
    else:
        raise TypeError("Data type not supported.")

def _data_to_frame(data: list[dict[str, str | None]], **kwargs: Any) -> DataFrame:
    """
    Convert parsed data to Data Frame.

    This method will bind xml dictionary data of keys and values
    into named columns of Data Frame using the built-in TextParser
    class that builds a DataFrame and infers specific dtypes.
    """
    tags = next(iter(data))
    nodes = [list(d.values()) for d in data]
    try:
        with TextParser(nodes, names=tags, **kwargs) as tp:
            return tp.read()
    except ParserError as err:
        raise ParserError(
            "XML document may be too complex for import. "
            "Try to flatten document and use distinct "
            "element and attribute names."
        ) from err

def _parse(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    xpath: str,
    namespaces: dict[str, str] | None,
    elems_only: bool,
    attrs_only: bool,
    names: Sequence[str] | None,
    dtype: DtypeArg | None,
    converters: ConvertersArg | None,
    parse_dates: ParseDatesArg | None,
    encoding: str | None,
    parser: XMLParsers,
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,
    iterparse: dict[str, list[str]] | None,
    compression: CompressionOptions,
    storage_options: StorageOptions,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    **kwargs: Any,
) -> DataFrame:
    """
    Call internal parsers.

    This method will conditionally call internal parsers:
    LxmlFrameParser and/or EtreeParser.

    Raises
    ------
    ImportError
        * If lxml is not installed if selected as parser.
    ValueError
        * If parser is not lxml or etree.
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
    data_dicts: list[dict[str, str | None]] = p.parse_data()
    return _data_to_frame(
        data=data_dicts,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        dtype_backend=dtype_backend,
        **kwargs,
    )

@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",
)
def read_xml(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    xpath: str = "./*",
    namespaces: dict[str, str] | None = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Sequence[str] | None = None,
    dtype: DtypeArg | None = None,
    converters: ConvertersArg | None = None,
    parse_dates: ParseDatesArg | None = None,
    encoding: str | None = "utf-8",
    parser: XMLParsers = "lxml",
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = None,
    iterparse: dict[str, list[str]] | None = None,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame:
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