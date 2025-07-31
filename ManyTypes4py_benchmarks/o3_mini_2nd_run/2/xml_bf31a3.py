from __future__ import annotations
import codecs
import io
from typing import TYPE_CHECKING, Any, final, Optional, Union, List, Dict
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import get_data_from_filepath

if TYPE_CHECKING:
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer, StorageOptions, WriteBuffer
    from pandas import DataFrame

@doc(storage_options=_shared_docs['storage_options'],
     compression_options=_shared_docs['compression_options'] % 'path_or_buffer')
class _BaseXMLFormatter:
    """
    Subclass for formatting data in XML.

    Parameters
    ----------
    path_or_buffer : str or file-like
        This can be either a string of raw XML, a valid URL,
        file or file-like object.

    index : bool
        Whether to include index in xml document.

    row_name : str
        Name for root of xml document. Default is 'data'.

    root_name : str
        Name for row elements of xml document. Default is 'row'.

    na_rep : str
        Missing data representation.

    attrs_cols : list
        List of columns to write as attributes in row element.

    elem_cols : list
        List of columns to write as children in row element.

    namespaces : dict
        The namespaces to define in XML document as dicts with key
        being namespace and value the URI.

    prefix : str
        The prefix for each element in XML document including root.

    encoding : str
        Encoding of xml object or document.

    xml_declaration : bool
        Whether to include xml declaration at top line item in xml.

    pretty_print : bool
        Whether to write xml document with line breaks and indentation.

    stylesheet : str or file-like
        A URL, file, file-like object, or a raw string containing XSLT.

    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    See also
    --------
    pandas.io.formats.xml.EtreeXMLFormatter
    pandas.io.formats.xml.LxmlXMLFormatter

    """
    def __init__(
        self,
        frame: DataFrame,
        path_or_buffer: Optional[Union[str, WriteBuffer]] = None,
        index: bool = True,
        root_name: str = 'data',
        row_name: str = 'row',
        na_rep: Optional[str] = None,
        attr_cols: Optional[List[Any]] = None,
        elem_cols: Optional[List[Any]] = None,
        namespaces: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        encoding: str = 'utf-8',
        xml_declaration: bool = True,
        pretty_print: bool = True,
        stylesheet: Optional[Union[str, Any]] = None,
        compression: Union[str, CompressionOptions] = 'infer',
        storage_options: Optional[StorageOptions] = None
    ) -> None:
        self.frame: DataFrame = frame
        self.path_or_buffer: Optional[Union[str, WriteBuffer]] = path_or_buffer
        self.index: bool = index
        self.root_name: str = root_name
        self.row_name: str = row_name
        self.na_rep: Optional[str] = na_rep
        self.attr_cols: Optional[List[Any]] = attr_cols
        self.elem_cols: Optional[List[Any]] = elem_cols
        self.namespaces: Optional[Dict[str, str]] = namespaces
        self.prefix: Optional[str] = prefix
        self.encoding: str = encoding
        self.xml_declaration: bool = xml_declaration
        self.pretty_print: bool = pretty_print
        self.stylesheet: Optional[Union[str, Any]] = stylesheet
        self.compression: Union[str, CompressionOptions] = compression
        self.storage_options: Optional[StorageOptions] = storage_options
        self.orig_cols: List[Any] = self.frame.columns.tolist()
        self.frame_dicts: Dict[Any, Dict[str, Any]] = self._process_dataframe()
        self._validate_columns()
        self._validate_encoding()
        self.prefix_uri: str = self._get_prefix_uri()
        self._handle_indexes()

    def _build_tree(self) -> bytes:
        """
        Build tree from data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        raise AbstractMethodError(self)

    @final
    def _validate_columns(self) -> None:
        """
        Validate elems_cols and attrs_cols.

        This method will check if columns is list-like.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
        if self.attr_cols and (not is_list_like(self.attr_cols)):
            raise TypeError(f'{type(self.attr_cols).__name__} is not a valid type for attr_cols')
        if self.elem_cols and (not is_list_like(self.elem_cols)):
            raise TypeError(f'{type(self.elem_cols).__name__} is not a valid type for elem_cols')

    @final
    def _validate_encoding(self) -> None:
        """
        Validate encoding.

        This method will check if encoding is among listed under codecs.

        Raises
        ------
        LookupError
            * If encoding is not available in codecs.
        """
        codecs.lookup(self.encoding)

    @final
    def _process_dataframe(self) -> Dict[Any, Dict[str, Any]]:
        """
        Adjust Data Frame to fit xml output.

        This method will adjust underlying data frame for xml output,
        including optionally replacing missing values and including indexes.
        """
        df = self.frame
        if self.index:
            df = df.reset_index()
        if self.na_rep is not None:
            df = df.fillna(self.na_rep)
        return df.to_dict(orient='index')

    @final
    def _handle_indexes(self) -> None:
        """
        Handle indexes.

        This method will add indexes into attr_cols or elem_cols.
        """
        if not self.index:
            return
        first_key = next(iter(self.frame_dicts))
        indexes = [x for x in self.frame_dicts[first_key].keys() if x not in self.orig_cols]
        if self.attr_cols:
            self.attr_cols = indexes + self.attr_cols
        if self.elem_cols:
            self.elem_cols = indexes + self.elem_cols

    def _get_prefix_uri(self) -> str:
        """
        Get uri of namespace prefix.

        This method retrieves corresponding URI to prefix in namespaces.

        Raises
        ------
        KeyError
            *If prefix is not included in namespace dict.
        """
        raise AbstractMethodError(self)

    @final
    def _other_namespaces(self) -> Dict[str, str]:
        """
        Define other namespaces.

        This method will build dictionary of namespaces attributes
        for root element, conditionally with optional namespaces and
        prefix.
        """
        nmsp_dict: Dict[str, str] = {}
        if self.namespaces:
            nmsp_dict = {
                f'xmlns{("" if p == "" else f":{p}")}': n
                for p, n in self.namespaces.items()
                if n != self.prefix_uri[1:-1]
            }
        return nmsp_dict

    @final
    def _build_attribs(self, d: Dict[str, Any], elem_row: Any) -> Any:
        """
        Create attributes of row.

        This method adds attributes using attr_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
        if not self.attr_cols:
            return elem_row
        for col in self.attr_cols:
            attr_name: str = self._get_flat_col_name(col)
            try:
                if not isna(d[col]):
                    elem_row.attrib[attr_name] = str(d[col])
            except KeyError as err:
                raise KeyError(f'no valid column, {col}') from err
        return elem_row

    @final
    def _get_flat_col_name(self, col: Any) -> str:
        flat_col = col
        if isinstance(col, tuple):
            flat_col = (
                ''.join([str(c) for c in col]).strip()
                if '' in col
                else '_'.join([str(c) for c in col]).strip()
            )
        return f'{self.prefix_uri}{flat_col}'

    @cache_readonly
    def _sub_element_cls(self) -> Any:
        raise AbstractMethodError(self)

    @final
    def _build_elems(self, d: Dict[str, Any], elem_row: Any) -> None:
        """
        Create child elements of row.

        This method adds child elements using elem_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
        sub_element_cls = self._sub_element_cls
        if not self.elem_cols:
            return
        for col in self.elem_cols:
            elem_name: str = self._get_flat_col_name(col)
            try:
                val: Optional[str] = None if isna(d[col]) or d[col] == '' else str(d[col])
                sub_element_cls(elem_row, elem_name).text = val
            except KeyError as err:
                raise KeyError(f'no valid column, {col}') from err

    @final
    def write_output(self) -> Optional[str]:
        xml_doc: bytes = self._build_tree()
        if self.path_or_buffer is not None:
            with get_handle(
                self.path_or_buffer,
                'wb',
                compression=self.compression,
                storage_options=self.storage_options,
                is_text=False
            ) as handles:
                handles.handle.write(xml_doc)
            return None
        else:
            return xml_doc.decode(self.encoding).rstrip()

class EtreeXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """
    def _build_tree(self) -> bytes:
        from xml.etree.ElementTree import Element, SubElement, tostring
        self.root = Element(f'{self.prefix_uri}{self.root_name}', attrib=self._other_namespaces())
        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f'{self.prefix_uri}{self.row_name}')
            if not self.attr_cols and (not self.elem_cols):
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)
            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)
        self.out_xml: bytes = tostring(
            self.root,
            method='xml',
            encoding=self.encoding,
            xml_declaration=self.xml_declaration
        )
        if self.pretty_print:
            self.out_xml = self._prettify_tree()
        if self.stylesheet is not None:
            raise ValueError('To use stylesheet, you need lxml installed and selected as parser.')
        return self.out_xml

    def _get_prefix_uri(self) -> str:
        from xml.etree.ElementTree import register_namespace
        uri: str = ''
        if self.namespaces:
            for p, n in self.namespaces.items():
                if isinstance(p, str) and isinstance(n, str):
                    register_namespace(p, n)
            if self.prefix:
                try:
                    uri = f'{{{self.namespaces[self.prefix]}}}'
                except KeyError as err:
                    raise KeyError(f'{self.prefix} is not included in namespaces') from err
            elif "" in self.namespaces:
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ''
        return uri

    @cache_readonly
    def _sub_element_cls(self) -> Any:
        from xml.etree.ElementTree import SubElement
        return SubElement

    def _prettify_tree(self) -> bytes:
        """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """
        from xml.dom.minidom import parseString
        dom = parseString(self.out_xml)
        return dom.toprettyxml(indent='  ', encoding=self.encoding)

class LxmlXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._convert_empty_str_key()

    def _build_tree(self) -> bytes:
        """
        Build tree from data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        from lxml.etree import Element, SubElement, tostring
        self.root = Element(f'{self.prefix_uri}{self.root_name}', nsmap=self.namespaces)
        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f'{self.prefix_uri}{self.row_name}')
            if not self.attr_cols and (not self.elem_cols):
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)
            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)
        self.out_xml: bytes = tostring(
            self.root,
            pretty_print=self.pretty_print,
            method='xml',
            encoding=self.encoding,
            xml_declaration=self.xml_declaration
        )
        if self.stylesheet is not None:
            self.out_xml = self._transform_doc()
        return self.out_xml

    def _convert_empty_str_key(self) -> None:
        """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """
        if self.namespaces and "" in self.namespaces.keys():
            self.namespaces[None] = self.namespaces.pop("", 'default')

    def _get_prefix_uri(self) -> str:
        uri: str = ''
        if self.namespaces:
            if self.prefix:
                try:
                    uri = f'{{{self.namespaces[self.prefix]}}}'
                except KeyError as err:
                    raise KeyError(f'{self.prefix} is not included in namespaces') from err
            elif "" in self.namespaces:
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ''
        return uri

    @cache_readonly
    def _sub_element_cls(self) -> Any:
        from lxml.etree import SubElement
        return SubElement

    def _transform_doc(self) -> bytes:
        """
        Parse stylesheet from file or buffer and run it.

        This method will parse stylesheet object into tree for parsing
        conditionally by its specific object type, then transforms
        original tree with XSLT script.
        """
        from lxml.etree import XSLT, XMLParser, fromstring, parse
        style_doc: Union[str, Any] = self.stylesheet
        assert style_doc is not None
        handle_data = get_data_from_filepath(
            filepath_or_buffer=style_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options
        )
        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            if isinstance(xml_data, io.StringIO):
                xsl_doc = fromstring(xml_data.getvalue().encode(self.encoding), parser=curr_parser)
            else:
                xsl_doc = parse(xml_data, parser=curr_parser)
        transformer = XSLT(xsl_doc)
        new_doc = transformer(self.root)
        return bytes(new_doc)