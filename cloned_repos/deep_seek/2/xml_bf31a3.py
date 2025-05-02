"""
:mod:`pandas.io.formats.xml` is a module for formatting data in XML.
"""
from __future__ import annotations
import codecs
import io
from typing import (
    TYPE_CHECKING,
    Any,
    final,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import get_data_from_filepath

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )
    from pandas import DataFrame
    from xml.etree.ElementTree import Element, SubElement
    from lxml.etree import _Element as LxmlElement, _XSLTResultTree

@doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'path_or_buffer')
class _BaseXMLFormatter:
    def __init__(
        self,
        frame: DataFrame,
        path_or_buffer: Optional[Union[str, WriteBuffer[bytes]]] = None,
        index: bool = True,
        root_name: str = 'data',
        row_name: str = 'row',
        na_rep: Optional[str] = None,
        attr_cols: Optional[List[Union[str, Tuple[str, ...]]]] = None,
        elem_cols: Optional[List[Union[str, Tuple[str, ...]]]] = None,
        namespaces: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        encoding: str = 'utf-8',
        xml_declaration: bool = True,
        pretty_print: bool = True,
        stylesheet: Optional[Union[str, ReadBuffer[bytes]] = None,
        compression: CompressionOptions = 'infer',
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        self.frame = frame
        self.path_or_buffer = path_or_buffer
        self.index = index
        self.root_name = root_name
        self.row_name = row_name
        self.na_rep = na_rep
        self.attr_cols = attr_cols
        self.elem_cols = elem_cols
        self.namespaces = namespaces if namespaces is not None else {}
        self.prefix = prefix
        self.encoding = encoding
        self.xml_declaration = xml_declaration
        self.pretty_print = pretty_print
        self.stylesheet = stylesheet
        self.compression = compression
        self.storage_options = storage_options
        self.orig_cols = self.frame.columns.tolist()
        self.frame_dicts = self._process_dataframe()
        self._validate_columns()
        self._validate_encoding()
        self.prefix_uri = self._get_prefix_uri()
        self._handle_indexes()

    def _build_tree(self) -> bytes:
        raise AbstractMethodError(self)

    @final
    def _validate_columns(self) -> None:
        if self.attr_cols and (not is_list_like(self.attr_cols)):
            raise TypeError(f'{type(self.attr_cols).__name__} is not a valid type for attr_cols')
        if self.elem_cols and (not is_list_like(self.elem_cols)):
            raise TypeError(f'{type(self.elem_cols).__name__} is not a valid type for elem_cols')

    @final
    def _validate_encoding(self) -> None:
        codecs.lookup(self.encoding)

    @final
    def _process_dataframe(self) -> Dict[int, Dict[str, Any]]:
        df = self.frame
        if self.index:
            df = df.reset_index()
        if self.na_rep is not None:
            df = df.fillna(self.na_rep)
        return cast(Dict[int, Dict[str, Any]], df.to_dict(orient='index'))

    @final
    def _handle_indexes(self) -> None:
        if not self.index:
            return
        first_key = next(iter(self.frame_dicts))
        indexes = [x for x in self.frame_dicts[first_key].keys() if x not in self.orig_cols]
        if self.attr_cols:
            self.attr_cols = indexes + self.attr_cols  # type: ignore[operator]
        if self.elem_cols:
            self.elem_cols = indexes + self.elem_cols  # type: ignore[operator]

    def _get_prefix_uri(self) -> str:
        raise AbstractMethodError(self)

    @final
    def _other_namespaces(self) -> Dict[str, str]:
        nmsp_dict: Dict[str, str] = {}
        if self.namespaces:
            nmsp_dict = {f'xmlns{(p if p == '' else f':{p}')}': n for p, n in self.namespaces.items() if n != self.prefix_uri[1:-1]}
        return nmsp_dict

    @final
    def _build_attribs(self, d: Dict[str, Any], elem_row: Union[Element, LxmlElement]) -> Union[Element, LxmlElement]:
        if not self.attr_cols:
            return elem_row
        for col in self.attr_cols:
            attr_name = self._get_flat_col_name(col)
            try:
                if not isna(d[col]):
                    elem_row.attrib[attr_name] = str(d[col])
            except KeyError as err:
                raise KeyError(f'no valid column, {col}') from err
        return elem_row

    @final
    def _get_flat_col_name(self, col: Union[str, Tuple[str, ...]]) -> str:
        flat_col = col
        if isinstance(col, tuple):
            flat_col = ''.join([str(c) for c in col]).strip() if '' in col else '_'.join([str(c) for c in col]).strip()
        return f'{self.prefix_uri}{flat_col}'

    @cache_readonly
    def _sub_element_cls(self) -> Any:
        raise AbstractMethodError(self)

    @final
    def _build_elems(self, d: Dict[str, Any], elem_row: Union[Element, LxmlElement]) -> None:
        sub_element_cls = self._sub_element_cls
        if not self.elem_cols:
            return
        for col in self.elem_cols:
            elem_name = self._get_flat_col_name(col)
            try:
                val = None if isna(d[col]) or d[col] == '' else str(d[col])
                sub_element_cls(elem_row, elem_name).text = val
            except KeyError as err:
                raise KeyError(f'no valid column, {col}') from err

    @final
    def write_output(self) -> Optional[str]:
        xml_doc = self._build_tree()
        if self.path_or_buffer is not None:
            with get_handle(self.path_or_buffer, 'wb', compression=self.compression, storage_options=self.storage_options, is_text=False) as handles:
                handles.handle.write(xml_doc)
            return None
        else:
            return xml_doc.decode(self.encoding).rstrip()

class EtreeXMLFormatter(_BaseXMLFormatter):
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
        self.out_xml = tostring(self.root, method='xml', encoding=self.encoding, xml_declaration=self.xml_declaration)
        if self.pretty_print:
            self.out_xml = self._prettify_tree()
        if self.stylesheet is not None:
            raise ValueError('To use stylesheet, you need lxml installed and selected as parser.')
        return self.out_xml

    def _get_prefix_uri(self) -> str:
        from xml.etree.ElementTree import register_namespace
        uri = ''
        if self.namespaces:
            for p, n in self.namespaces.items():
                if isinstance(p, str) and isinstance(n, str):
                    register_namespace(p, n)
            if self.prefix:
                try:
                    uri = f'{{{self.namespaces[self.prefix]}}}'
                except KeyError as err:
                    raise KeyError(f'{self.prefix} is not included in namespaces') from err
            elif '' in self.namespaces:
                uri = f'{{{self.namespaces['']}}}'
            else:
                uri = ''
        return uri

    @cache_readonly
    def _sub_element_cls(self) -> Any:
        from xml.etree.ElementTree import SubElement
        return SubElement

    def _prettify_tree(self) -> bytes:
        from xml.dom.minidom import parseString
        dom = parseString(self.out_xml)
        return dom.toprettyxml(indent='  ', encoding=self.encoding)

class LxmlXMLFormatter(_BaseXMLFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._convert_empty_str_key()

    def _build_tree(self) -> bytes:
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
        self.out_xml = tostring(self.root, pretty_print=self.pretty_print, method='xml', encoding=self.encoding, xml_declaration=self.xml_declaration)
        if self.stylesheet is not None:
            self.out_xml = self._transform_doc()
        return self.out_xml

    def _convert_empty_str_key(self) -> None:
        if self.namespaces and '' in self.namespaces.keys():
            self.namespaces[None] = self.namespaces.pop('', 'default')

    def _get_prefix_uri(self) -> str:
        uri = ''
        if self.namespaces:
            if self.prefix:
                try:
                    uri = f'{{{self.namespaces[self.prefix]}}}'
                except KeyError as err:
                    raise KeyError(f'{self.prefix} is not included in namespaces') from err
            elif '' in self.namespaces:
                uri = f'{{{self.namespaces['']}}}'
            else:
                uri = ''
        return uri

    @cache_readonly
    def _sub_element_cls(self) -> Any:
        from lxml.etree import SubElement
        return SubElement

    def _transform_doc(self) -> bytes:
        from lxml.etree import XSLT, XMLParser, fromstring, parse
        style_doc = self.stylesheet
        assert style_doc is not None
        handle_data = get_data_from_filepath(filepath_or_buffer=style_doc, encoding=self.encoding, compression=self.compression, storage_options=self.storage_options)
        with handle_data as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            if isinstance(xml_data, io.StringIO):
                xsl_doc = fromstring(xml_data.getvalue().encode(self.encoding), parser=curr_parser)
            else:
                xsl_doc = parse(xml_data, parser=curr_parser)
        transformer = XSLT(xsl_doc)
        new_doc = transformer(self.root)
        return bytes(new_doc)
