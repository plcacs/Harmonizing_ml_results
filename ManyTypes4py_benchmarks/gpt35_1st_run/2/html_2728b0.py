from __future__ import annotations
from collections.abc import Iterable, Sequence
import errno
import numbers
import os
import re
from re import Pattern
from typing import TYPE_CHECKING, Literal, cast
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
    from pandas._typing import BaseBuffer, DtypeBackend, FilePath, HTMLFlavors, ReadBuffer, StorageOptions
    from pandas import DataFrame

_RE_WHITESPACE: Pattern = re.compile('[\\r\\n]+|\\s{2,}')

def _remove_whitespace(s: str, regex: Pattern = _RE_WHITESPACE) -> str:
    """
    Replace extra whitespace inside of a string with a single space.

    Parameters
    ----------
    s : str
        The string from which to remove extra whitespace.
    regex : Pattern
        The regular expression to use to remove extra whitespace.

    Returns
    -------
    subd : str
        `s` with all extra whitespace replaced with a single space.
    """
    return regex.sub(' ', s.strip())

def _get_skiprows(skiprows: int | slice | Sequence) -> Iterable[int] | int:
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
        start, step = (skiprows.start or 0, skiprows.step or 1)
        return list(range(start, skiprows.stop, step))
    elif isinstance(skiprows, numbers.Integral) or is_list_like(skiprows):
        return cast('int | Sequence[int]', skiprows)
    elif skiprows is None:
        return 0
    raise TypeError(f'{type(skiprows).__name__} is not a valid type for skipping rows')

def _read(obj: str | os.PathLike | ReadBuffer, encoding: str, storage_options: StorageOptions) -> str:
    """
    Try to read from a url, file or string.

    Parameters
    ----------
    obj : str, os.PathLike, or file-like object
    encoding : str
        Encoding to be used by parser
    storage_options : StorageOptions

    Returns
    -------
    raw_text : str
    """
    try:
        with get_handle(obj, 'r', encoding=encoding, storage_options=storage_options) as handles:
            return handles.handle.read()
    except OSError as err:
        if not is_url(obj):
            raise FileNotFoundError(f'[Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: {obj}') from err
        raise

class _HtmlFrameParser:
    """
    Base class for parsers that parse HTML into DataFrames.

    Parameters
    ----------
    io : str or file-like
        This can be either a string path, a valid URL using the HTTP,
        FTP, or FILE protocols or a file-like object.

    match : str or regex
        The text to match in the document.

    attrs : dict
        List of HTML <table> element attributes to match.

    encoding : str
        Encoding to be used by parser

    displayed_only : bool
        Whether or not items with "display:none" should be ignored

    extract_links : {None, "all", "header", "body", "footer"}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    Attributes
    ----------
    io : str or file-like
        raw HTML, URL, or file-like object

    match : regex
        The text to match in the raw HTML

    attrs : dict-like
        A dictionary of valid table attributes to use to search for table
        elements.

    encoding : str
        Encoding to be used by parser

    displayed_only : bool
        Whether or not items with "display:none" should be ignored

    extract_links : {None, "all", "header", "body", "footer"}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    Notes
    -----
    To subclass this class effectively you must override the following methods:
        * :func:`_build_doc`
        * :func:`_attr_getter`
        * :func:`_href_getter`
        * :func:`_text_getter`
        * :func:`_parse_td`
        * :func:`_parse_thead_tr`
        * :func:`_parse_tbody_tr`
        * :func:`_parse_tfoot_tr`
        * :func:`_parse_tables`
        * :func:`_equals_tag`
    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(self, io: str | file-like, match: str | Pattern, attrs: dict, encoding: str, displayed_only: bool, extract_links: Literal[None, "all", "header", "body", "footer"], storage_options: StorageOptions | None):
        self.io = io
        self.match = match
        self.attrs = attrs
        self.encoding = encoding
        self.displayed_only = displayed_only
        self.extract_links = extract_links
        self.storage_options = storage_options

    def parse_tables(self) -> Iterable[tuple[list[list[str]], list[list[str]], list[list[str]]]]:
        """
        Parse and return all tables from the DOM.

        Returns
        -------
        list of parsed (header, body, footer) tuples from tables.
        """
        tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        return (self._parse_thead_tbody_tfoot(table) for table in tables)

    def _attr_getter(self, obj: node-like, attr: str) -> str:
        """
        Return the attribute value of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        attr : str
            The attribute, such as "colspan"

        Returns
        -------
        str
            The attribute value.
        """
        return obj.get(attr)

    def _href_getter(self, obj: node-like) -> str | None:
        """
        Return a href if the DOM node contains a child <a> or None.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        href : str | None
            The href from the <a> child of the DOM node.
        """
        raise AbstractMethodError(self)

    def _text_getter(self, obj: node-like) -> str:
        """
        Return the text of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        text : str
            The text from an individual DOM node.
        """
        raise AbstractMethodError(self)

    def _parse_td(self, obj: node-like) -> list[node-like]:
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

    def _parse_thead_tr(self, table: node-like) -> list[node-like]:
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

    def _parse_tbody_tr(self, table: node-like) -> list[node-like]:
        """
        Return the list of tbody row elements from the parsed table element.

        HTML5 table bodies consist of either 0 or more <tbody> elements (which
        only contain <tr> elements) or 0 or more <tr> elements. This method
        checks for both structures.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        raise AbstractMethodError(self)

    def _parse_tfoot_tr(self, table: node-like) -> list[node-like]:
        """
        Return the list of tfoot row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        raise AbstractMethodError(self)

    def _parse_tables(self, document: node-like, match: str | Pattern, attrs: dict) -> list[node-like]:
        """
        Return all tables from the parsed DOM.

        Parameters
        ----------
        document : the DOM from which to parse the table element.

        match : str | Pattern
            The text to search for in the DOM tree.

        attrs : dict
            A dictionary of table attributes that can be used to disambiguate
            multiple tables on a page.

        Raises
        ------
        ValueError : `match` does not match any text in the document.

        Returns
        -------
        list of node-like
            HTML <table> elements to be parsed into raw data.
        """
        raise AbstractMethodError(self)

    def _equals_tag(self, obj: node-like, tag: str) -> bool:
        """
        Return whether an individual DOM node matches a tag

        Parameters
        ----------
        obj : node-like
            A DOM node.

        tag : str
            Tag name to be checked for equality.

        Returns
        -------
        boolean
            Whether `obj`'s tag name is `tag`
        """
        raise AbstractMethodError(self)

    def _build_doc(self) -> node-like:
        """
        Return a tree-like object that can be used to iterate over the DOM.

        Returns
        -------
        node-like
            The DOM from which to parse the table element.
        """
        raise AbstractMethodError(self)

    def _parse_thead_tbody_tfoot(self, table_html: node-like) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
        """
        Given a table, return parsed header, body, and foot.

        Parameters
        ----------
        table_html : node-like

        Returns
        -------
        tuple of (header, body, footer), each a list of list-of-text rows.

        Notes
        -----
        Header and body are lists-of-lists. Top level list is a list of
        rows. Each row is a list of str text.

        Logic: Use <thead>, <tbody>, <tfoot> elements to identify
               header, body, and footer, otherwise:
               - Put all rows into body
               - Move rows from top of body to header only if
                 all elements inside row are <th>
               - Move rows from bottom of body to footer only if
                 all elements inside row are <th>
        """
        raise AbstractMethodError(self)

    def _expand_colspan_rowspan(self, rows: list[node-like], section: str, remainder: list[tuple[int, str | tuple, int]] | None = None, overflow: bool = True) -> tuple[list[list[str]], list[tuple[int, str | tuple, int]]]:
        """
        Given a list of <tr>s, return a list of text rows.

        Parameters
        ----------
        rows : list of node-like
            List of <tr>s
        section : str
            The section that the rows belong to (header, body or footer).
        remainder: list[tuple[int, str | tuple, int]] | None
            Any remainder from the expansion of previous section
        overflow: bool
            If true, return any partial rows as 'remainder'. If not, use up any
            partial rows. True by default.

        Returns
        -------
        list of list
            Each returned row is a list of str text, or tuple (text, link)
            if extract_links is not None.
        remainder
            Remaining partial rows if any. If overflow is False, an empty list
            is returned.

        Notes
        -----
        Any cell with ``rowspan`` or ``colspan`` will have its contents copied
        to subsequent cells.
        """
        raise AbstractMethodError(self)

    def _handle_hidden_tables(self, tbl_list: list[node-like], attr_name: str) -> list[node-like]:
        """
        Return list of tables, potentially removing hidden elements

        Parameters
        ----------
        tbl_list : list of node-like
            Type of list elements will vary depending upon parser used
        attr_name : str
            Name of the accessor for retrieving HTML attributes

        Returns
        -------
        list of node-like
            Return type matches `tbl_list`
        """
        raise AbstractMethodError(self)

class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses BeautifulSoup under the hood.

    See Also
    --------
    pandas.io.html._HtmlFrameParser
    pandas.io.html._LxmlFrameParser

    Notes
    -----
    Documentation strings for this class are in the base class
    :class:`pandas.io.html._HtmlFrameParser`.
    """

    def _parse_tables(self, document: node-like, match: str | Pattern, attrs: dict) -> list[node-like]:
        raise AbstractMethodError(self)

    def _href_getter(self, obj: node-like) -> str | None:
        raise AbstractMethodError(self)

    def _text_getter(self, obj: node-like) -> str:
        raise AbstractMethodError(self)

    def _equals_tag(self, obj: node-like, tag: str) -> bool:
        raise AbstractMethodError(self)

    def _parse_td(self, row: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

    def _parse_thead_tr(self, table: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

    def _parse_tbody_tr(self, table: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

    def _parse_tfoot_tr(self, table: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

    def _setup_build_doc(self) -> str:
        raise AbstractMethodError(self)

    def _build_doc(self) -> node-like:
        raise AbstractMethodError(self)

def _build_xpath_expr(attrs: dict) -> str:
    """
    Build an xpath expression to simulate bs4's ability to pass in kwargs to
    search for attributes when using the lxml parser.

    Parameters
    ----------
    attrs : dict
        A dict of HTML attributes. These are NOT checked for validity.

    Returns
    -------
    expr : str
        An XPath expression that checks for the given HTML attributes.
    """
    raise AbstractMethodError(self)

class _LxmlFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses lxml under the hood.

    Warning
    -------
    This parser can only handle HTTP, FTP, and FILE urls.

    See Also
    --------
    _HtmlFrameParser
    _BeautifulSoupLxmlFrameParser

    Notes
    -----
    Documentation strings for this class are in the base class
    :class:`_HtmlFrameParser`.
    """

    def _href_getter(self, obj: node-like) -> str | None:
        raise AbstractMethodError(self)

    def _text_getter(self, obj: node-like) -> str:
        raise AbstractMethodError(self)

    def _parse_td(self, row: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

    def _parse_tables(self, document: node-like, match: str | Pattern, kwargs: dict) -> list[node-like]:
        raise AbstractMethodError(self)

    def _equals_tag(self, obj: node-like, tag: str) -> bool:
        raise AbstractMethodError(self)

    def _build_doc(self) -> node-like:
        raise AbstractMethodError(self)

    def _parse_thead_tr(self, table: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

    def _parse_tbody_tr(self, table: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

    def _parse_tfoot_tr(self, table: node-like) -> list[node-like]:
        raise AbstractMethodError(self)

def _expand_elements(body: list[list[str]]) -> None:
    raise AbstractMethodError(self)

def _data_to_frame(**kwargs) -> DataFrame:
    raise AbstractMethodError(self)

def _parser_dispatch(flavor: Literal["lxml", "html5lib", "bs4"]) -> _HtmlFrameParser:
    raise AbstractMethodError(self)

def _print_as_set(s: set) -> str:
    raise AbstractMethodError(self)

def _validate_flavor(flavor: Literal["lxml", "html5lib", "bs4"]) -> tuple[str, ...]:
    raise AbstractMethodError(self)

def _parse(flavor: tuple[str, ...], io: str, match: str | Pattern, attrs: dict, encoding: str, displayed_only: bool, extract_links: Literal[None, "all", "header", "body", "footer"], storage_options: StorageOptions, **kwargs) -> list[DataFrame]:
    raise AbstractMethodError(self)

@doc(storage_options=_shared_docs['storage_options'])
def read_html(io: str, *, match: str | Pattern = '.+', flavor: Literal["lxml", "html5lib", "bs4"] | tuple[str, ...] | None = None, header: int | list | None = None, index_col: int | list | None = None, skiprows: int | slice | Sequence | None = None, attrs: dict | None = None, parse_dates: bool = False, thousands: str = ',', encoding: str | None = None, decimal: str = '.', converters: dict | None = None, na_values: Iterable | None = None, keep_default_na: bool = True, displayed_only: bool = True, extract_links: Literal[None, "all", "header", "body", "footer"] | None = None, dtype_backend: Literal['numpy_nullable', 'pyarrow'] = lib.no_default, storage_options: StorageOptions | None = None) -> list[DataFrame]:
    raise AbstractMethodError(self)
