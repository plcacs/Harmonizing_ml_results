from __future__ import annotations
import codecs
import io
from typing import TYPE_CHECKING, Any, final, List, Dict, Tuple, Union
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

class _BaseXMLFormatter:
    def __init__(self, frame: DataFrame, path_or_buffer: Union[str, io.BufferedIOBase] = None, index: bool = True, root_name: str = 'data', row_name: str = 'row', na_rep: str = None, attr_cols: List[str] = None, elem_cols: List[str] = None, namespaces: Dict[str, str] = None, prefix: str = None, encoding: str = 'utf-8', xml_declaration: bool = True, pretty_print: bool = True, stylesheet: Union[str, io.BufferedIOBase] = None, compression: CompressionOptions = 'infer', storage_options: StorageOptions = None) -> None:
    def _build_tree(self) -> None:
    def _validate_columns(self) -> None:
    def _validate_encoding(self) -> None:
    def _process_dataframe(self) -> None:
    def _handle_indexes(self) -> None:
    def _get_prefix_uri(self) -> None:
    def _other_namespaces(self) -> None:
    def _build_attribs(self, d: Dict[str, Any], elem_row: Any) -> None:
    def _get_flat_col_name(self, col: Union[str, Tuple[str]]) -> None:
    def _sub_element_cls(self) -> None:
    def _build_elems(self, d: Dict[str, Any], elem_row: Any) -> None:
    def write_output(self) -> None:

class EtreeXMLFormatter(_BaseXMLFormatter):
    def _build_tree(self) -> None:
    def _get_prefix_uri(self) -> None:
    def _sub_element_cls(self) -> None:
    def _prettify_tree(self) -> None:

class LxmlXMLFormatter(_BaseXMLFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
    def _build_tree(self) -> None:
    def _convert_empty_str_key(self) -> None:
    def _get_prefix_uri(self) -> None:
    def _sub_element_cls(self) -> None:
    def _transform_doc(self) -> None:
