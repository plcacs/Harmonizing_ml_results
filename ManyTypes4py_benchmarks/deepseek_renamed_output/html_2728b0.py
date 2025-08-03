"""
:mod:`pandas.io.html` is a module containing functionality for dealing with
HTML IO.

"""
from __future__ import annotations
from collections import abc
import errno
import numbers
import os
import re
from re import Pattern
from typing import TYPE_CHECKING, Literal, cast, Any, Dict, List, Optional, Set, Tuple, Union, Iterable, Sequence
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError, EmptyDataError
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_list_like
from pandas import isna
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle, is_url, stringify_path, validate_header_arg
from pandas.io.formats.printing import pprint_thing
from pandas.io.parsers import TextParser
if TYPE_CHECKING:
    from collections.abc import Iterable as ABCIterable, Sequence as ABCSequence
    from pandas._typing import BaseBuffer, DtypeBackend, FilePath, HTMLFlavors, ReadBuffer, StorageOptions
    from pandas import DataFrame
    from bs4 import BeautifulSoup
    from lxml.etree import _Element as LxmlElement
    from lxml.html import HtmlElement

_RE_WHITESPACE: Pattern[str] = re.compile('[\\r\\n]+|\\s{2,}')


def func_0zljn5nz(s: str, regex: Pattern[str] = _RE_WHITESPACE) -> str:
    """
    Replace extra whitespace inside of a string with a single space.

    Parameters
    ----------
    s : str or unicode
        The string from which to remove extra whitespace.
    regex : re.Pattern
        The regular expression to use to remove extra whitespace.

    Returns
    -------
    subd : str or unicode
        `s` with all extra whitespace replaced with a single space.
    """
    return regex.sub(' ', s.strip())


def func_b2g385tr(skiprows: Union[int, slice, Sequence[int], None]) -> Union[int, Sequence[int]]:
    """
    Get an iterator given an integer, slice or container.

    Parameters
    ----------
    skiprows : int, slice, container
        The iterator to use to skip rows; can also be a slice.

    Raises
    ------
    TypeError
        * If `skiprows` is not a slice, integer, or Container

    Returns
    -------
    it : iterable
        A proper iterator to use to skip rows of a DataFrame.
    """
    if isinstance(skiprows, slice):
        start, step = skiprows.start or 0, skiprows.step or 1
        return list(range(start, skiprows.stop, step))
    elif isinstance(skiprows, numbers.Integral) or is_list_like(skiprows):
        return cast('int | Sequence[int]', skiprows)
    elif skiprows is None:
        return 0
    raise TypeError(
        f'{type(skiprows).__name__} is not a valid type for skipping rows')


def func_7fqleu5n(obj: Union[str, bytes, os.PathLike, IO[str]], encoding: Optional[str], storage_options: Optional[Dict[str, Any]]) -> str:
    """
    Try to read from a url, file or string.

    Parameters
    ----------
    obj : str, unicode, path object, or file-like object

    Returns
    -------
    raw_text : str
    """
    try:
        with get_handle(obj, 'r', encoding=encoding, storage_options=
            storage_options) as handles:
            return handles.handle.read()
    except OSError as err:
        if not is_url(obj):
            raise FileNotFoundError(
                f'[Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: {obj}'
                ) from err
        raise


class _HtmlFrameParser:
    """
    Base class for parsers that parse HTML into DataFrames.
    """
    def __init__(
        self,
        io: Union[str, bytes, os.PathLike, IO[str]],
        match: Pattern[str],
        attrs: Optional[Dict[str, str]],
        encoding: Optional[str],
        displayed_only: bool,
        extract_links: Optional[Literal["all", "header", "body", "footer"]],
        storage_options: Optional[Dict[str, Any]] = None
    ) -> None:
        self.io = io
        self.match = match
        self.attrs = attrs
        self.encoding = encoding
        self.displayed_only = displayed_only
        self.extract_links = extract_links
        self.storage_options = storage_options

    def func_kf23muyb(self) -> Iterable[Tuple[List[List[str]], List[List[str]], List[List[str]]]]:
        """
        Parse and return all tables from the DOM.

        Returns
        -------
        list of parsed (header, body, footer) tuples from tables.
        """
        tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        return (self._parse_thead_tbody_tfoot(table) for table in tables)

    def func_fjpn13wr(self, obj: Any, attr: str) -> Optional[str]:
        """
        Return the attribute value of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        attr : str or unicode
            The attribute, such as "colspan"

        Returns
        -------
        str or unicode
            The attribute value.
        """
        return obj.get(attr)

    def func_qf16iutu(self, obj: Any) -> Optional[str]:
        """
        Return a href if the DOM node contains a child <a> or None.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        href : str or unicode
            The href from the <a> child of the DOM node.
        """
        raise AbstractMethodError(self)

    def func_1hza7qh3(self, obj: Any) -> str:
        """
        Return the text of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        text : str or unicode
            The text from an individual DOM node.
        """
        raise AbstractMethodError(self)

    def func_anftxyg5(self, obj: Any) -> List[Any]:
        """
        Return the td elements from a row element.

        Parameters
        ----------
        obj : node-like
            A DOM <tr> node.

        Returns
        -------
        list of node-like
            These are the elements of each row, i.e., the columns.
        """
        raise AbstractMethodError(self)

    def func_v9s7h17c(self, table: Any) -> List[Any]:
        """
        Return the list of thead row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains zero or more thead elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        raise AbstractMethodError(self)

    def func_iu636man(self, table: Any) -> List[Any]:
        """
        Return the list of tbody row elements from the parsed table element.
        """
        raise AbstractMethodError(self)

    def func_9bawcksm(self, table: Any) -> List[Any]:
        """
        Return the list of tfoot row elements from the parsed table element.
        """
        raise AbstractMethodError(self)

    def func_nqhg6fw8(self, document: Any, match: Pattern[str], attrs: Optional[Dict[str, str]]) -> List[Any]:
        """
        Return all tables from the parsed DOM.
        """
        raise AbstractMethodError(self)

    def func_yqewg1wj(self, obj: Any, tag: str) -> bool:
        """
        Return whether an individual DOM node matches a tag
        """
        raise AbstractMethodError(self)

    def func_kecv8ior(self) -> Any:
        """
        Return a tree-like object that can be used to iterate over the DOM.
        """
        raise AbstractMethodError(self)

    def func_rqy5z4ue(self, table_html: Any) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """
        Given a table, return parsed header, body, and foot.
        """
        header_rows = self._parse_thead_tr(table_html)
        body_rows = self._parse_tbody_tr(table_html)
        footer_rows = self._parse_tfoot_tr(table_html)

        def func_u50xdlei(row: Any) -> bool:
            return all(self._equals_tag(t, 'th') for t in self._parse_td(row))
        if not header_rows:
            while body_rows and func_u50xdlei(body_rows[0]):
                header_rows.append(body_rows.pop(0))
        header, rem = self._expand_colspan_rowspan(header_rows, section=
            'header')
        body, rem = self._expand_colspan_rowspan(body_rows, section='body',
            remainder=rem, overflow=len(footer_rows) > 0)
        footer, _ = self._expand_colspan_rowspan(footer_rows, section=
            'footer', remainder=rem, overflow=False)
        return header, body, footer

    def func_dd3l7ffb(
        self,
        rows: List[Any],
        section: str,
        remainder: Optional[List[Tuple[int, Union[str, Tuple[str, Optional[str]]], int]]] = None,
        overflow: bool = True
    ) -> Tuple[List[List[Union[str, Tuple[str, Optional[str]]]]], List[Tuple[int, Union[str, Tuple[str, Optional[str]]], int]]]:
        """
        Given a list of <tr>s, return a list of text rows.
        """
        all_texts: List[List[Union[str, Tuple[str, Optional[str]]]]] = []
        remainder = remainder if remainder is not None else []
        for tr in rows:
            texts: List[Union[str, Tuple[str, Optional[str]]]] = []
            next_remainder: List[Tuple[int, Union[str, Tuple[str, Optional[str]]], int]]] = []
            index = 0
            tds = self._parse_td(tr)
            for td in tds:
                while remainder and remainder[0][0] <= index:
                    prev_i, prev_text, prev_rowspan = remainder.pop(0)
                    texts.append(prev_text)
                    if prev_rowspan > 1:
                        next_remainder.append((prev_i, prev_text, 
                            prev_rowspan - 1))
                    index += 1
                text = func_0zljn5nz(self._text_getter(td))
                if self.extract_links in ('all', section):
                    href = self._href_getter(td)
                    text = text, href
                rowspan = int(self._attr_getter(td, 'rowspan') or 1)
                colspan = int(self._attr_getter(td, 'colspan') or 1)
                for _ in range(colspan):
                    texts.append(text)
                    if rowspan > 1:
                        next_remainder.append((index, text, rowspan - 1))
                    index += 1
            for prev_i, prev_text, prev_rowspan in remainder:
                texts.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1)
                        )
            all_texts.append(texts)
            remainder = next_remainder
        if not overflow:
            while remainder:
                next_remainder = []
                texts = []
                for prev_i, prev_text, prev_rowspan in remainder:
                    texts.append(prev_text)
                    if prev_rowspan > 1:
                        next_remainder.append((prev_i, prev_text, 
                            prev_rowspan - 1))
                all_texts.append(texts)
                remainder = next_remainder
        return all_texts, remainder

    def func_f2vtmp1s(self, tbl_list: List[Any], attr_name: str) -> List[Any]:
        """
        Return list of tables, potentially removing hidden elements
        """
        if not self.displayed_only:
            return tbl_list
        return [x for x in tbl_list if 'display:none' not in getattr(x,
            attr_name).get('style', '').replace(' ', '')]


class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses BeautifulSoup under the hood.
    """
    def func_nqhg6fw8(self, document: BeautifulSoup, match: Pattern[str], attrs: Optional[Dict[str, str]]) -> List[BeautifulSoup]:
        element_name = 'table'
        tables = document.find_all(element_name, attrs=attrs)
        if not tables:
            raise ValueError('No tables found')
        result: List[BeautifulSoup] = []
        unique_tables: Set[BeautifulSoup] = set()
        tables = self._handle_hidden_tables(tables, 'attrs')
        for table in tables:
            if self.displayed_only:
                for elem in table.find_all('style'):
                    elem.decompose()
                for elem in table.find_all(style=re.compile('display:\\s*none')
                    ):
                    elem.decompose()
            if table not in unique_tables and table.find(string=match
                ) is not None:
                result.append(table)
            unique_tables.add(table)
        if not result:
            raise ValueError(
                f'No tables found matching pattern {match.pattern!r}')
        return result

    def func_qf16iutu(self, obj: BeautifulSoup) -> Optional[str]:
        a = obj.find('a', href=True)
        return None if not a else a['href']

    def func_1hza7qh3(self, obj: BeautifulSoup) -> str:
        return obj.text

    def func_yqewg1wj(self, obj: BeautifulSoup, tag: str) -> bool:
        return obj.name == tag

    def func_anftxyg5(self, row: BeautifulSoup) -> List[BeautifulSoup]:
        return row.find_all(('td', 'th'), recursive=False)

    def func_v9s7h17c(self, table: BeautifulSoup) -> List[BeautifulSoup]:
        return table.select('thead tr')

    def func_iu636man(self, table: BeautifulSoup) -> List[BeautifulSoup]:
        from_tbody = table.select('tbody tr')
        from_root = table.find_all('tr', recursive=False)
        return from_tbody + from_root

    def func_9bawcksm(self, table: BeautifulSoup) -> List[BeautifulSoup]:
        return table.select('tfoot tr')

    def func_frk0d5rl(self) -> str:
        raw_text = func_7fqleu5n(self.io, self.encoding, self.storage_options)
        if not raw_text:
            raise ValueError(f'No text parsed from document: {self.io}')
        return raw_text

    def func_kecv8ior(self) -> BeautifulSoup:
        from bs4 import BeautifulSoup
        bdoc = self._setup_build_doc()
        if isinstance(bdoc, bytes) and self.encoding is not None:
            udoc = bdoc.decode(self.encoding)
            from_encoding = None
        else:
            udoc = bdoc
            from_encoding = self.encoding
        soup = BeautifulSoup(udoc, features='html5lib', from_encoding=
            from_encoding)
        for br in soup.find_all('br'):
            br.replace_with('\n' + br.text)
        return soup


def func_qx2u5loj(attrs: Dict[str, str]) -> str:
    """
    Build an xpath expression to simulate bs4's ability to pass in kwargs to
    search for attributes when using the lxml parser.
    """
    if 'class_' in attrs:
        attrs['class'] = attrs.pop('class_')
    s = ' and '.join([f'@{k}={v!r}' for k, v in attrs.items()])
    return f'[{s}]'


_re_namespace: Dict[str, str] = {'re': 'http://exslt.org/regular-expressions'}


class _LxmlFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses lxml under the hood.
    """
    def func_qf16iutu(self, obj: HtmlElement) -> Optional[str]:
        href = obj.xpath('.//a/@href')
        return None if not href else href[0]

    def func_1hza7qh3(self, obj: HtmlElement) -> str:
        return obj.text_content()

    def func_anftxyg5(self, row: HtmlElement) -> List[HtmlElement]:
        return row.xpath('./td|./th')

    def func_nqhg6fw8(self, document: HtmlElement, match: Pattern[str], kwargs: Optional[Dict[str, str]]) -> List[HtmlElement]:
        pattern = match.pattern
        xpath_expr = f'//table[.//text()[re:test(., {pattern!r})]]'
        if kwargs:
            xpath_expr += func_qx2u5loj(kwargs)
        tables = document.xpath(xpath_expr, namespaces=_re_namespace)
        tables = self._handle_hidden_tables(tables, 'attrib')
        if self.displayed_only:
            for table in tables:
                for elem in table.xpath('.//style'):
                    elem.drop_tree()
                for elem in table.xpath('.//*[@style]'):
                    if 'display:none' in elem.attrib.get('style', '').replace(
                        ' ', ''):
                        elem.drop_tree()
        if not tables:
            raise ValueError(f'No tables found matching regex {pattern!r}')
        return tables

    def func_yqewg1wj(self, obj: HtmlElement, tag: str) -> bool:
        return obj.tag == tag

    def func_kecv8ior(self) -> HtmlElement:
        """
        Raises
        ------
        ValueError
            * If a URL that lxml cannot parse is passed.

        Exception
            * Any other ``Exception`` thrown. For example, trying to parse a
