from __future__ import annotations
import io
from os import PathLike
from typing import Any, Callable, Dict, IO, List, Optional, Sequence, Union

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
    from xml.etree.ElementTree import Element
    from lxml import etree  # noqa: F401
    from pandas._typing import CompressionOptions, ConvertersArg, DtypeArg, DtypeBackend, FilePath, ParseDatesArg, ReadBuffer, StorageOptions, XMLParsers
    from pandas import DataFrame


@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buffer')
class _XMLFrameParser:
    def __init__(
        self,
        path_or_buffer: Union[str, PathLike[Any], IO[Any]],
        xpath: str,
        namespaces: Optional[Dict[str, str]],
        elems_only: bool,
        attrs_only: bool,
        names: Optional[Sequence[str]],
        dtype: Optional[Dict[Any, Any]],
        converters: Optional[Dict[Any, Callable[[Any], Any]]],
        parse_dates: Optional[Union[bool, Sequence[Any], Dict[Any, Any]]],
        encoding: str,
        stylesheet: Optional[Union[str, PathLike[Any], IO[Any]]],
        iterparse: Optional[Dict[str, Sequence[Any]]],
        compression: Optional[str],
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

    def _parse_nodes(self, elems: Sequence[Any]) -> List[Dict[str, Any]]:
        """
        Parse xml nodes.

        This method will parse the children and attributes of elements
        in ``xpath``, conditionally for only elements, only attributes
        or both while optionally renaming node names.

        Raises
        ------
        ValueError
            * If only elements and only attributes are specified.
        """
        if self.elems_only and self.attrs_only:
            raise ValueError('Either element or attributes can be parsed not both.')
        if self.elems_only:
            if self.names:
                dicts = [
                    {
                        **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}),
                        **{nm: ch.text if ch.text else None for nm, ch in zip(self.names, el.findall('*'))}
                    }
                    for el in elems
                ]
            else:
                dicts = [
                    {ch.tag: ch.text if ch.text else None for ch in el.findall('*')}
                    for el in elems
                ]
        elif self.attrs_only:
            dicts = [{k: v if v else None for k, v in el.attrib.items()} for el in elems]
        elif self.names:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}),
                    **{nm: ch.text if ch.text else None for nm, ch in zip(self.names, el.findall('*'))}
                }
                for el in elems
            ]
        else:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and (not el.text.isspace()) else {}),
                    **{ch.tag: ch.text if ch.text else None for ch in el.findall('*')}
                }
                for el in elems
            ]
        dicts = [{(k.split('}')[1] if '}' in k else k): v for k, v in d.items()} for d in dicts]
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

        Raises
        ------
        TypeError
            * If ``iterparse`` is not a dict or its dict value is not list-like.
        ParserError
            * If ``path_or_buffer`` is not a physical file on disk or file-like object.
            * If no data is returned from selected items in ``iterparse``.
        """
        dicts: List[Dict[str, Any]] = []
        row: Optional[Dict[str, Any]] = None
        if not isinstance(self.iterparse, dict):
            raise TypeError(f'{type(self.iterparse).__name__} is not a valid type for iterparse')
        row_node = next(iter(self.iterparse.keys())) if self.iterparse else ''
        if not is_list_like(self.iterparse[row_node]):
            raise TypeError(f'{type(self.iterparse[row_node])} is not a valid type for value in iterparse')
        if not hasattr(self.path_or_buffer, 'read') and (
            not isinstance(self.path_or_buffer, (str, PathLike))
            or is_url(self.path_or_buffer)
            or is_fsspec_url(self.path_or_buffer)
            or (isinstance(self.path_or_buffer, str) and self.path_or_buffer.startswith(('<?xml', '<')))
            or (infer_compression(self.path_or_buffer, 'infer') is not None)
        ):
            raise ParserError('iterparse is designed for large XML files that are fully extracted on local disk and not as compressed files or online sources.')
        iterparse_repeats: bool = len(self.iterparse[row_node]) != len(set(self.iterparse[row_node]))
        for event, elem in iterparse(self.path_or_buffer, events=('start', 'end')):
            curr_elem = elem.tag.split('}')[1] if '}' in elem.tag else elem.tag
            if event == 'start':
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

    def _validate_path(self) -> Any:
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

    def _parse_doc(self, raw_doc: Union[str, PathLike[Any], IO[Any]]) -> Any:
        """
        Build tree from path_or_buffer.

        This method will parse XML object into tree
        either from string/bytes or file location.
        """
        raise AbstractMethodError(self)


class _EtreeFrameParser(_XMLFrameParser):
    def parse_data(self) -> List[Dict[str, Any]]:
        from xml.etree.ElementTree import iterparse
        if self.stylesheet is not None:
            raise ValueError('To use stylesheet, you need lxml installed and selected as parser.')
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()
        else:
            elems = None
        self._validate_names()
        xml_dicts: List[Dict[str, Any]] = (
            self._parse_nodes(elems) if self.iterparse is None else self._iterparse_nodes(iterparse)
        )
        return xml_dicts

    def _validate_path(self) -> List[Element]:
        msg: str = ('xpath does not return any nodes or attributes. '
                    'Be sure to specify in `xpath` the parent nodes of children and attributes to parse. '
                    'If document uses namespaces denoted with xmlns, be sure to define namespaces and use them in xpath.')
        try:
            elems: List[Element] = self.xml_doc.findall(self.xpath, namespaces=self.namespaces)
            children = [ch for el in elems for ch in el.findall('*')]
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
            raise SyntaxError('You have used an incorrect or unsupported XPath expression for etree library or you used an undeclared namespace prefix.') from err
        return elems

    def _validate_names(self) -> None:
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

    def _parse_doc(self, raw_doc: Union[str, PathLike[Any], IO[Any]]) -> Element:
        from xml.etree.ElementTree import XMLParser, parse
        handle_data: IO[Any] = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options
        )
        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            document = parse(xml_data, parser=curr_parser)
        return document.getroot()


class _LxmlFrameParser(_XMLFrameParser):
    def parse_data(self) -> List[Dict[str, Any]]:
        from lxml.etree import iterparse
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            if self.stylesheet:
                self.xsl_doc = self._parse_doc(self.stylesheet)
                self.xml_doc = self._transform_doc()
            elems = self._validate_path()
        else:
            elems = None
        self._validate_names()
        xml_dicts: List[Dict[str, Any]] = (
            self._parse_nodes(elems) if self.iterparse is None else self._iterparse_nodes(iterparse)
        )
        return xml_dicts

    def _validate_path(self) -> List[Any]:
        msg: str = ('xpath does not return any nodes or attributes. '
                    'Be sure to specify in `xpath` the parent nodes of children and attributes to parse. '
                    'If document uses namespaces denoted with xmlns, be sure to define namespaces and use them in xpath.')
        elems = self.xml_doc.xpath(self.xpath, namespaces=self.namespaces)
        children = [ch for el in elems for ch in el.xpath('*')]
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
                children = self.xml_doc.xpath(self.xpath + '[1]/*', namespaces=self.namespaces)
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError('names does not match length of child elements in xpath.')
            else:
                raise TypeError(f'{type(self.names).__name__} is not a valid type for names')

    def _parse_doc(self, raw_doc: Union[str, PathLike[Any], IO[Any]]) -> Any:
        from lxml.etree import XMLParser, fromstring, parse
        handle_data: IO[Any] = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options
        )
        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            if isinstance(xml_data, io.StringIO):
                if self.encoding is None:
                    raise TypeError('Can not pass encoding None when input is StringIO.')
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
    encoding: str,
    compression: Optional[str],
    storage_options: Optional[Dict[str, Any]]
) -> IO[Any]:
    """
    Extract raw XML data.

    The method accepts two input types:
        1. filepath (string-like)
        2. file-like object (e.g. open file object, StringIO)
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    with get_handle(filepath_or_buffer, 'r', encoding=encoding, compression=compression, storage_options=storage_options) as handle_obj:
        if hasattr(handle_obj.handle, 'read'):
            return preprocess_data(handle_obj.handle.read())
        else:
            return handle_obj.handle


def preprocess_data(data: Union[str, bytes, IO[Any]]) -> IO[Any]:
    """
    Convert extracted raw data.

    This method will return underlying data of extracted XML content.
    The data either has a `read` attribute (e.g. a file object or a
    StringIO/BytesIO) or is a string or bytes that is an XML document.
    """
    if isinstance(data, str):
        data = io.StringIO(data)
    elif isinstance(data, bytes):
        data = io.BytesIO(data)
    return data


def _data_to_frame(data: List[Dict[str, Any]], **kwargs: Any) -> DataFrame:
    """
    Convert parsed data to Data Frame.

    This method will bind xml dictionary data of keys and values
    into named columns of Data Frame using the built-in TextParser
    class that build Data Frame and infers specific dtypes.
    """
    tags = next(iter(data))
    nodes = [list(d.values()) for d in data]
    try:
        with TextParser(nodes, names=tags, **kwargs) as tp:
            return tp.read()
    except ParserError as err:
        raise ParserError('XML document may be too complex for import. Try to flatten document and use distinct element and attribute names.') from err


def _parse(
    path_or_buffer: Union[str, PathLike[Any], IO[Any]],
    xpath: str,
    namespaces: Optional[Dict[str, str]],
    elems_only: bool,
    attrs_only: bool,
    names: Optional[Sequence[str]],
    dtype: Optional[Dict[Any, Any]],
    converters: Optional[Dict[Any, Callable[[Any], Any]]],
    parse_dates: Optional[Union[bool, Sequence[Any], Dict[Any, Any]]],
    encoding: str,
    parser: str,
    stylesheet: Optional[Union[str, PathLike[Any], IO[Any]]],
    iterparse: Optional[Dict[str, Sequence[Any]]],
    compression: Optional[str],
    storage_options: Optional[Dict[str, Any]],
    dtype_backend: DtypeBackend = lib.no_default,
    **kwargs: Any
) -> DataFrame:
    """
    Call internal parsers.

    This method will conditionally call internal parsers:
    LxmlFrameParser and/or EtreeParser.
    """
    if parser == 'lxml':
        lxml = import_optional_dependency('lxml.etree', errors='ignore')
        if lxml is not None:
            p = _LxmlFrameParser(path_or_buffer, xpath, namespaces, elems_only, attrs_only, names, dtype, converters, parse_dates, encoding, stylesheet, iterparse, compression, storage_options)
        else:
            raise ImportError('lxml not found, please install or use the etree parser.')
    elif parser == 'etree':
        p = _EtreeFrameParser(path_or_buffer, xpath, namespaces, elems_only, attrs_only, names, dtype, converters, parse_dates, encoding, stylesheet, iterparse, compression, storage_options)
    else:
        raise ValueError('Values for parser can only be lxml or etree.')
    data_dicts: List[Dict[str, Any]] = p.parse_data()
    return _data_to_frame(data=data_dicts, dtype=dtype, converters=converters, parse_dates=parse_dates, dtype_backend=dtype_backend, **kwargs)


@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buffer')
def read_xml(
    path_or_buffer: Union[str, PathLike[Any], IO[Any]],
    *,
    xpath: str = './*',
    namespaces: Optional[Dict[str, str]] = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Optional[Sequence[str]] = None,
    dtype: Optional[Dict[Any, Any]] = None,
    converters: Optional[Dict[Any, Callable[[Any], Any]]] = None,
    parse_dates: Optional[Union[bool, Sequence[Any], Dict[Any, Any]]] = None,
    encoding: str = 'utf-8',
    parser: str = 'lxml',
    stylesheet: Optional[Union[str, PathLike[Any], IO[Any]]] = None,
    iterparse: Optional[Dict[str, Sequence[Any]]] = None,
    compression: Optional[str] = 'infer',
    storage_options: Optional[Dict[str, Any]] = None,
    dtype_backend: DtypeBackend = lib.no_default
) -> DataFrame:
    """
    Read XML document into a :class:`~pandas.DataFrame` object.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    path_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a ``read()`` function. The string can be a path.
        The string can further be a URL. Valid URL schemes
        include http, ftp, s3, and file.

        .. deprecated:: 2.1.0
            Passing xml literal strings is deprecated.
            Wrap literal xml input in ``io.StringIO`` or ``io.BytesIO`` instead.

    xpath : str, optional, default './*'
        The ``XPath`` to parse required set of nodes for migration to
        :class:`~pandas.DataFrame`.``XPath`` should return a collection of elements
        and not a single element. Note: The ``etree`` parser supports limited ``XPath``
        expressions. For more complex ``XPath``, use ``lxml`` which requires
        installation.

    namespaces : dict, optional
        The namespaces defined in XML document as dicts with key being
        namespace prefix and value the URI. There is no need to include all
        namespaces in XML, only the ones used in ``xpath`` expression.
        Note: if XML document uses default namespace denoted as
        `xmlns='<URI>'` without a prefix, you must assign any temporary
        namespace prefix such as 'doc' to the URI in order to parse
        underlying nodes and/or attributes.

    elems_only : bool, optional, default False
        Parse only the child elements at the specified ``xpath``. By default,
        all child elements and non-empty text nodes are returned.

    attrs_only :  bool, optional, default False
        Parse only the attributes at the specified ``xpath``.
        By default, all attributes are returned.

    names :  list-like, optional
        Column names for DataFrame of parsed XML data. Use this parameter to
        rename original element names and distinguish same named elements and
        attributes.

    dtype : Type name or dict of column -> type, optional
        Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32,
        'c': 'Int64'}}
        Use `str` or `object` together with suitable `na_values` settings
        to preserve and not interpret dtype.
        If converters are specified, they will be applied INSTEAD
        of dtype conversion.

        .. versionadded:: 1.5.0

    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can either
        be integers or column labels.

        .. versionadded:: 1.5.0

    parse_dates : bool or list of int or names or list of lists or dict, default False
        Identifiers to parse index or columns to datetime. The behavior is as follows:

        * boolean. If True -> try parsing the index.
        * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
          each as a separate date column.
        * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
          a single date column.
        * dict, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
          result 'foo'

        .. versionadded:: 1.5.0

    encoding : str, optional, default 'utf-8'
        Encoding of XML document.

    parser : {{'lxml','etree'}}, default 'lxml'
        Parser module to use for retrieval of data. Only 'lxml' and
        'etree' are supported. With 'lxml' more complex ``XPath`` searches
        and ability to use XSLT stylesheet are supported.

    stylesheet : str, path object or file-like object
        A URL, file-like object, or a string path containing an XSLT script.
        This stylesheet should flatten complex, deeply nested XML documents
        for easier parsing. To use this feature you must have ``lxml`` module
        installed and specify 'lxml' as ``parser``. The ``xpath`` must
        reference nodes of transformed XML document generated after XSLT
        transformation and not the original XML document. Only XSLT 1.0
        scripts and not later versions is currently supported.

    iterparse : dict, optional
        The nodes or attributes to retrieve in iterparsing of XML document
        as a dict with key being the name of repeating element and value being
        list of elements or attribute names that are descendants of the repeated
        element. Note: If this option is used, it will replace ``xpath`` parsing
        and unlike ``xpath``, descendants do not need to relate to each other but can
        exist any where in document under the repeating element. This memory-
        efficient method should be used for very large XML files (500MB, 1GB, or 5GB+).
        For example, ``{{"row_element": ["child_elem", "attr", "grandchild_elem"]}}``.

        .. versionadded:: 1.5.0

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    Returns
    -------
    df
        A DataFrame.
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
        dtype_backend=dtype_backend
    )