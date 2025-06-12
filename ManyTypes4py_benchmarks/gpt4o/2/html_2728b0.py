from __future__ import annotations
from collections import abc
import errno
import numbers
import os
import re
from re import Pattern
from typing import TYPE_CHECKING, Literal, cast, Union, Optional, List, Tuple, Dict, Any, Iterator
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
    from collections.abc import Iterable, Sequence
    from pandas._typing import BaseBuffer, DtypeBackend, FilePath, HTMLFlavors, ReadBuffer, StorageOptions
    from pandas import DataFrame

_RE_WHITESPACE: Pattern = re.compile('[\\r\\n]+|\\s{2,}')


def _remove_whitespace(s: str, regex: Pattern = _RE_WHITESPACE) -> str:
    return regex.sub(' ', s.strip())


def _get_skiprows(skiprows: Union[int, slice, abc.Container]) -> Union[int, Sequence[int]]:
    if isinstance(skiprows, slice):
        start, step = (skiprows.start or 0, skiprows.step or 1)
        return list(range(start, skiprows.stop, step))
    elif isinstance(skiprows, numbers.Integral) or is_list_like(skiprows):
        return cast('int | Sequence[int]', skiprows)
    elif skiprows is None:
        return 0
    raise TypeError(f'{type(skiprows).__name__} is not a valid type for skipping rows')


def _read(obj: Union[str, os.PathLike, ReadBuffer[bytes], ReadBuffer[str]], encoding: Optional[str], storage_options: Optional[StorageOptions]) -> str:
    try:
        with get_handle(obj, 'r', encoding=encoding, storage_options=storage_options) as handles:
            return handles.handle.read()
    except OSError as err:
        if not is_url(obj):
            raise FileNotFoundError(f'[Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: {obj}') from err
        raise


class _HtmlFrameParser:
    def __init__(self, io: Union[str, os.PathLike, ReadBuffer[bytes], ReadBuffer[str]], match: Union[str, Pattern], attrs: Optional[Dict[str, str]], encoding: Optional[str], displayed_only: bool, extract_links: Optional[Literal["all", "header", "body", "footer"]], storage_options: Optional[StorageOptions] = None) -> None:
        self.io = io
        self.match = match
        self.attrs = attrs
        self.encoding = encoding
        self.displayed_only = displayed_only
        self.extract_links = extract_links
        self.storage_options = storage_options

    def parse_tables(self) -> Iterator[Tuple[List[List[str]], List[List[str]], List[List[str]]]]:
        tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        return (self._parse_thead_tbody_tfoot(table) for table in tables)

    def _attr_getter(self, obj: Any, attr: str) -> Optional[str]:
        return obj.get(attr)

    def _href_getter(self, obj: Any) -> Optional[str]:
        raise AbstractMethodError(self)

    def _text_getter(self, obj: Any) -> str:
        raise AbstractMethodError(self)

    def _parse_td(self, obj: Any) -> List[Any]:
        raise AbstractMethodError(self)

    def _parse_thead_tr(self, table: Any) -> List[Any]:
        raise AbstractMethodError(self)

    def _parse_tbody_tr(self, table: Any) -> List[Any]:
        raise AbstractMethodError(self)

    def _parse_tfoot_tr(self, table: Any) -> List[Any]:
        raise AbstractMethodError(self)

    def _parse_tables(self, document: Any, match: Union[str, Pattern], attrs: Optional[Dict[str, str]]) -> List[Any]:
        raise AbstractMethodError(self)

    def _equals_tag(self, obj: Any, tag: str) -> bool:
        raise AbstractMethodError(self)

    def _build_doc(self) -> Any:
        raise AbstractMethodError(self)

    def _parse_thead_tbody_tfoot(self, table_html: Any) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        header_rows = self._parse_thead_tr(table_html)
        body_rows = self._parse_tbody_tr(table_html)
        footer_rows = self._parse_tfoot_tr(table_html)

        def row_is_all_th(row: Any) -> bool:
            return all((self._equals_tag(t, 'th') for t in self._parse_td(row)))

        if not header_rows:
            while body_rows and row_is_all_th(body_rows[0]):
                header_rows.append(body_rows.pop(0))
        header, rem = self._expand_colspan_rowspan(header_rows, section='header')
        body, rem = self._expand_colspan_rowspan(body_rows, section='body', remainder=rem, overflow=len(footer_rows) > 0)
        footer, _ = self._expand_colspan_rowspan(footer_rows, section='footer', remainder=rem, overflow=False)
        return (header, body, footer)

    def _expand_colspan_rowspan(self, rows: List[Any], section: str, remainder: Optional[List[Tuple[int, Union[str, Tuple[str, Optional[str]]], int]]] = None, overflow: bool = True) -> Tuple[List[List[Union[str, Tuple[str, Optional[str]]]]], List[Tuple[int, Union[str, Tuple[str, Optional[str]]], int]]]:
        all_texts: List[List[Union[str, Tuple[str, Optional[str]]]]] = []
        remainder = remainder if remainder is not None else []
        for tr in rows:
            texts: List[Union[str, Tuple[str, Optional[str]]]] = []
            next_remainder: List[Tuple[int, Union[str, Tuple[str, Optional[str]]], int]] = []
            index = 0
            tds = self._parse_td(tr)
            for td in tds:
                while remainder and remainder[0][0] <= index:
                    prev_i, prev_text, prev_rowspan = remainder.pop(0)
                    texts.append(prev_text)
                    if prev_rowspan > 1:
                        next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
                    index += 1
                text = _remove_whitespace(self._text_getter(td))
                if self.extract_links in ('all', section):
                    href = self._href_getter(td)
                    text = (text, href)
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
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
            all_texts.append(texts)
            remainder = next_remainder
        if not overflow:
            while remainder:
                next_remainder = []
                texts = []
                for prev_i, prev_text, prev_rowspan in remainder:
                    texts.append(prev_text)
                    if prev_rowspan > 1:
                        next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
                all_texts.append(texts)
                remainder = next_remainder
        return (all_texts, remainder)

    def _handle_hidden_tables(self, tbl_list: List[Any], attr_name: str) -> List[Any]:
        if not self.displayed_only:
            return tbl_list
        return [x for x in tbl_list if 'display:none' not in getattr(x, attr_name).get('style', '').replace(' ', '')]


class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    def _parse_tables(self, document: Any, match: Union[str, Pattern], attrs: Optional[Dict[str, str]]) -> List[Any]:
        element_name = 'table'
        tables = document.find_all(element_name, attrs=attrs)
        if not tables:
            raise ValueError('No tables found')
        result = []
        unique_tables = set()
        tables = self._handle_hidden_tables(tables, 'attrs')
        for table in tables:
            if self.displayed_only:
                for elem in table.find_all('style'):
                    elem.decompose()
                for elem in table.find_all(style=re.compile('display:\\s*none')):
                    elem.decompose()
            if table not in unique_tables and table.find(string=match) is not None:
                result.append(table)
            unique_tables.add(table)
        if not result:
            raise ValueError(f'No tables found matching pattern {match.pattern!r}')
        return result

    def _href_getter(self, obj: Any) -> Optional[str]:
        a = obj.find('a', href=True)
        return None if not a else a['href']

    def _text_getter(self, obj: Any) -> str:
        return obj.text

    def _equals_tag(self, obj: Any, tag: str) -> bool:
        return obj.name == tag

    def _parse_td(self, row: Any) -> List[Any]:
        return row.find_all(('td', 'th'), recursive=False)

    def _parse_thead_tr(self, table: Any) -> List[Any]:
        return table.select('thead tr')

    def _parse_tbody_tr(self, table: Any) -> List[Any]:
        from_tbody = table.select('tbody tr')
        from_root = table.find_all('tr', recursive=False)
        return from_tbody + from_root

    def _parse_tfoot_tr(self, table: Any) -> List[Any]:
        return table.select('tfoot tr')

    def _setup_build_doc(self) -> str:
        raw_text = _read(self.io, self.encoding, self.storage_options)
        if not raw_text:
            raise ValueError(f'No text parsed from document: {self.io}')
        return raw_text

    def _build_doc(self) -> Any:
        from bs4 import BeautifulSoup
        bdoc = self._setup_build_doc()
        if isinstance(bdoc, bytes) and self.encoding is not None:
            udoc = bdoc.decode(self.encoding)
            from_encoding = None
        else:
            udoc = bdoc
            from_encoding = self.encoding
        soup = BeautifulSoup(udoc, features='html5lib', from_encoding=from_encoding)
        for br in soup.find_all('br'):
            br.replace_with('\n' + br.text)
        return soup


def _build_xpath_expr(attrs: Dict[str, str]) -> str:
    if 'class_' in attrs:
        attrs['class'] = attrs.pop('class_')
    s = ' and '.join([f'@{k}={v!r}' for k, v in attrs.items()])
    return f'[{s}]'


_re_namespace: Dict[str, str] = {'re': 'http://exslt.org/regular-expressions'}


class _LxmlFrameParser(_HtmlFrameParser):
    def _href_getter(self, obj: Any) -> Optional[str]:
        href = obj.xpath('.//a/@href')
        return None if not href else href[0]

    def _text_getter(self, obj: Any) -> str:
        return obj.text_content()

    def _parse_td(self, row: Any) -> List[Any]:
        return row.xpath('./td|./th')

    def _parse_tables(self, document: Any, match: Union[str, Pattern], kwargs: Optional[Dict[str, str]]) -> List[Any]:
        pattern = match.pattern
        xpath_expr = f'//table[.//text()[re:test(., {pattern!r})]]'
        if kwargs:
            xpath_expr += _build_xpath_expr(kwargs)
        tables = document.xpath(xpath_expr, namespaces=_re_namespace)
        tables = self._handle_hidden_tables(tables, 'attrib')
        if self.displayed_only:
            for table in tables:
                for elem in table.xpath('.//style'):
                    elem.drop_tree()
                for elem in table.xpath('.//*[@style]'):
                    if 'display:none' in elem.attrib.get('style', '').replace(' ', ''):
                        elem.drop_tree()
        if not tables:
            raise ValueError(f'No tables found matching regex {pattern!r}')
        return tables

    def _equals_tag(self, obj: Any, tag: str) -> bool:
        return obj.tag == tag

    def _build_doc(self) -> Any:
        from lxml.etree import XMLSyntaxError
        from lxml.html import HTMLParser, parse
        parser = HTMLParser(recover=True, encoding=self.encoding)
        if is_url(self.io):
            with get_handle(self.io, 'r', storage_options=self.storage_options) as f:
                r = parse(f.handle, parser=parser)
        else:
            try:
                r = parse(self.io, parser=parser)
            except OSError as err:
                raise FileNotFoundError(f'[Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: {self.io}') from err
        try:
            r = r.getroot()
        except AttributeError:
            pass
        else:
            if not hasattr(r, 'text_content'):
                raise XMLSyntaxError('no text parsed from document', 0, 0, 0)
        for br in r.xpath('*//br'):
            br.tail = '\n' + (br.tail or '')
        return r

    def _parse_thead_tr(self, table: Any) -> List[Any]:
        rows = []
        for thead in table.xpath('.//thead'):
            rows.extend(thead.xpath('./tr'))
            elements_at_root = thead.xpath('./td|./th')
            if elements_at_root:
                rows.append(thead)
        return rows

    def _parse_tbody_tr(self, table: Any) -> List[Any]:
        from_tbody = table.xpath('.//tbody//tr')
        from_root = table.xpath('./tr')
        return from_tbody + from_root

    def _parse_tfoot_tr(self, table: Any) -> List[Any]:
        return table.xpath('.//tfoot//tr')


def _expand_elements(body: List[List[str]]) -> None:
    data = [len(elem) for elem in body]
    lens = Series(data)
    lens_max = lens.max()
    not_max = lens[lens != lens_max]
    empty = ['']
    for ind, length in not_max.items():
        body[ind] += empty * (lens_max - length)


def _data_to_frame(**kwargs: Any) -> DataFrame:
    head, body, foot = kwargs.pop('data')
    header = kwargs.pop('header')
    kwargs['skiprows'] = _get_skiprows(kwargs['skiprows'])
    if head:
        body = head + body
        if header is None:
            if len(head) == 1:
                header = 0
            else:
                header = [i for i, row in enumerate(head) if any((text for text in row))]
    if foot:
        body += foot
    _expand_elements(body)
    with TextParser(body, header=header, **kwargs) as tp:
        return tp.read()


_valid_parsers: Dict[Optional[str], type] = {'lxml': _LxmlFrameParser, None: _LxmlFrameParser, 'html5lib': _BeautifulSoupHtml5LibFrameParser, 'bs4': _BeautifulSoupHtml5LibFrameParser}


def _parser_dispatch(flavor: Optional[str]) -> type:
    valid_parsers = list(_valid_parsers.keys())
    if flavor not in valid_parsers:
        raise ValueError(f'{flavor!r} is not a valid flavor, valid flavors are {valid_parsers}')
    if flavor in ('bs4', 'html5lib'):
        import_optional_dependency('html5lib')
        import_optional_dependency('bs4')
    else:
        import_optional_dependency('lxml.etree')
    return _valid_parsers[flavor]


def _print_as_set(s: set) -> str:
    arg = ', '.join([pprint_thing(el) for el in s])
    return f'{{{arg}}}'


def _validate_flavor(flavor: Optional[Union[str, Iterable[str]]]) -> Tuple[str, ...]:
    if flavor is None:
        flavor = ('lxml', 'bs4')
    elif isinstance(flavor, str):
        flavor = (flavor,)
    elif isinstance(flavor, abc.Iterable):
        if not all((isinstance(flav, str) for flav in flavor)):
            raise TypeError(f'Object of type {type(flavor).__name__!r} is not an iterable of strings')
    else:
        msg = repr(flavor) if isinstance(flavor, str) else str(flavor)
        msg += ' is not a valid flavor'
        raise ValueError(msg)
    flavor = tuple(flavor)
    valid_flavors = set(_valid_parsers)
    flavor_set = set(flavor)
    if not flavor_set & valid_flavors:
        raise ValueError(f'{_print_as_set(flavor_set)} is not a valid set of flavors, valid flavors are {_print_as_set(valid_flavors)}')
    return flavor


def _parse(flavor: Optional[Union[str, Iterable[str]]], io: Union[str, os.PathLike, ReadBuffer[bytes], ReadBuffer[str]], match: Union[str, Pattern], attrs: Optional[Dict[str, str]], encoding: Optional[str], displayed_only: bool, extract_links: Optional[Literal["all", "header", "body", "footer"]], storage_options: Optional[StorageOptions], **kwargs: Any) -> List[DataFrame]:
    flavor = _validate_flavor(flavor)
    compiled_match = re.compile(match)
    retained = None
    for flav in flavor:
        parser = _parser_dispatch(flav)
        p = parser(io, compiled_match, attrs, encoding, displayed_only, extract_links, storage_options)
        try:
            tables = p.parse_tables()
        except ValueError as caught:
            if hasattr(io, 'seekable') and io.seekable():
                io.seek(0)
            elif hasattr(io, 'seekable') and (not io.seekable()):
                raise ValueError(f"The flavor {flav} failed to parse your input. Since you passed a non-rewindable file object, we can't rewind it to try another parser. Try read_html() with a different flavor.") from caught
            retained = caught
        else:
            break
    else:
        assert retained is not None
        raise retained
    ret = []
    for table in tables:
        try:
            df = _data_to_frame(data=table, **kwargs)
            if extract_links in ('all', 'header') and isinstance(df.columns, MultiIndex):
                df.columns = Index(((col[0], None if isna(col[1]) else col[1]) for col in df.columns), tupleize_cols=False)
            ret.append(df)
        except EmptyDataError:
            continue
    return ret


@doc(storage_options=_shared_docs['storage_options'])
def read_html(io: Union[str, os.PathLike, ReadBuffer[bytes], ReadBuffer[str]], *, match: Union[str, Pattern] = '.+', flavor: Optional[Union[str, Iterable[str]]] = None, header: Optional[Union[int, List[int]]] = None, index_col: Optional[Union[int, List[int]]] = None, skiprows: Optional[Union[int, List[int], slice]] = None, attrs: Optional[Dict[str, str]] = None, parse_dates: bool = False, thousands: str = ',', encoding: Optional[str] = None, decimal: str = '.', converters: Optional[Dict[Union[int, str], Any]] = None, na_values: Optional[Iterable] = None, keep_default_na: bool = True, displayed_only: bool = True, extract_links: Optional[Literal["all", "header", "body", "footer"]] = None, dtype_backend: Union[str, lib.NoDefault] = lib.no_default, storage_options: Optional[StorageOptions] = None) -> List[DataFrame]:
    if isinstance(skiprows, numbers.Integral) and skiprows < 0:
        raise ValueError('cannot skip rows starting from the end of the data (you passed a negative value)')
    if extract_links not in [None, 'header', 'footer', 'body', 'all']:
        raise ValueError(f'`extract_links` must be one of {{None, "header", "footer", "body", "all"}}, got "{extract_links}"')
    validate_header_arg(header)
    check_dtype_backend(dtype_backend)
    io = stringify_path(io)
    return _parse(flavor=flavor, io=io, match=match, header=header, index_col=index_col, skiprows=skiprows, parse_dates=parse_dates, thousands=thousands, attrs=attrs, encoding=encoding, decimal=decimal, converters=converters, na_values=na_values, keep_default_na=keep_default_na, displayed_only=displayed_only, extract_links=extract_links, dtype_backend=dtype_backend, storage_options=storage_options)
