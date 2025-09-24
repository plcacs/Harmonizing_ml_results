from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
import re
from typing import TYPE_CHECKING, Any, DefaultDict, Optional, TypedDict, Union, cast
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import is_complex, is_float, is_integer
from pandas.core.dtypes.generic import ABCSeries
from pandas import DataFrame, Index, IndexSlice, MultiIndex, Series, isna
from pandas.api.types import is_list_like
import pandas.core.common as com
if TYPE_CHECKING:
    from pandas._typing import Axis, Level
jinja2 = import_optional_dependency('jinja2', extra='DataFrame.style requires jinja2.')
from markupsafe import escape as escape_html
BaseFormatter = Union[str, Callable]
ExtFormatter = Union[BaseFormatter, dict[Any, Optional[BaseFormatter]]]
CSSPair = tuple[str, Union[str, float]]
CSSList = list[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties
CSSStyles = list[CSSDict]
Subset = Union[slice, Sequence, Index]

class StylerRenderer:
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """
    loader: Any = jinja2.PackageLoader('pandas', 'io/formats/templates')
    env: Any = jinja2.Environment(loader=loader, trim_blocks=True)
    template_html: Any = env.get_template('html.tpl')
    template_html_table: Any = env.get_template('html_table.tpl')
    template_html_style: Any = env.get_template('html_style.tpl')
    template_latex: Any = env.get_template('latex.tpl')
    template_typst: Any = env.get_template('typst.tpl')
    template_string: Any = env.get_template('string.tpl')

    def __init__(self, data: Union[DataFrame, Series], uuid: Optional[str] = None, uuid_len: int = 5, table_styles: Optional[CSSStyles] = None, table_attributes: Optional[str] = None, caption: Optional[str] = None, cell_ids: bool = True, precision: Optional[int] = None) -> None:
        if isinstance(data, Series):
            data = data.to_frame()
        if not isinstance(data, DataFrame):
            raise TypeError('``data`` must be a Series or DataFrame')
        self.data: DataFrame = data
        self.index: Index = data.index
        self.columns: Index = data.columns
        if not isinstance(uuid_len, int) or uuid_len < 0:
            raise TypeError('``uuid_len`` must be an integer in range [0, 32].')
        self.uuid: str = uuid or uuid4().hex[:min(32, uuid_len)]
        self.uuid_len: int = len(self.uuid)
        self.table_styles: Optional[CSSStyles] = table_styles
        self.table_attributes: Optional[str] = table_attributes
        self.caption: Optional[str] = caption
        self.cell_ids: bool = cell_ids
        self.css: dict[str, str] = {'row_heading': 'row_heading', 'col_heading': 'col_heading', 'index_name': 'index_name', 'col': 'col', 'row': 'row', 'col_trim': 'col_trim', 'row_trim': 'row_trim', 'level': 'level', 'data': 'data', 'blank': 'blank', 'foot': 'foot'}
        self.concatenated: list[StylerRenderer] = []
        self.hide_index_names: bool = False
        self.hide_column_names: bool = False
        self.hide_index_: list[bool] = [False] * self.index.nlevels
        self.hide_columns_: list[bool] = [False] * self.columns.nlevels
        self.hidden_rows: Sequence[int] = []
        self.hidden_columns: Sequence[int] = []
        self.ctx: DefaultDict[tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_index: DefaultDict[tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_columns: DefaultDict[tuple[int, int], CSSList] = defaultdict(list)
        self.cell_context: DefaultDict[tuple[int, int], str] = defaultdict(str)
        self._todo: list[tuple[Callable, tuple, dict]] = []
        self.tooltips: Optional[Tooltips] = None
        precision = get_option('styler.format.precision') if precision is None else precision
        self._display_funcs: DefaultDict[tuple[int, int], Callable[[Any], str]] = defaultdict(lambda : partial(_default_formatter, precision=precision))
        self._display_funcs_index: DefaultDict[tuple[int, int], Callable[[Any], str]] = defaultdict(lambda : partial(_default_formatter, precision=precision))
        self._display_funcs_index_names: DefaultDict[int, Callable[[Any], str]] = defaultdict(lambda : partial(_default_formatter, precision=precision))
        self._display_funcs_columns: DefaultDict[tuple[int, int], Callable[[Any], str]] = defaultdict(lambda : partial(_default_formatter, precision=precision))
        self._display_funcs_column_names: DefaultDict[int, Callable[[Any], str]] = defaultdict(lambda : partial(_default_formatter, precision=precision))

    def _render(self, sparse_index: bool, sparse_columns: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, blank: str = '') -> dict[str, Any]:
        """
        Computes and applies styles and then generates the general render dicts.

        Also extends the `ctx` and `ctx_index` attributes with those of concatenated
        stylers for use within `_translate_latex`
        """
        self._compute()
        dxs: list[dict[str, Any]] = []
        ctx_len: int = len(self.index)
        for (i, concatenated) in enumerate(self.concatenated):
            concatenated.hide_index_ = self.hide_index_
            concatenated.hidden_columns = self.hidden_columns
            foot: str = f"{self.css['foot']}{i}"
            concatenated.css = {**self.css, 'data': f'{foot}_data', 'row_heading': f'{foot}_row_heading', 'row': f'{foot}_row', 'foot': f'{foot}_foot'}
            dx: dict[str, Any] = concatenated._render(sparse_index, sparse_columns, max_rows, max_cols, blank)
            dxs.append(dx)
            for ((r, c), v) in concatenated.ctx.items():
                self.ctx[r + ctx_len, c] = v
            for ((r, c), v) in concatenated.ctx_index.items():
                self.ctx_index[r + ctx_len, c] = v
            ctx_len += len(concatenated.index)
        d: dict[str, Any] = self._translate(sparse_index, sparse_columns, max_rows, max_cols, blank, dxs)
        return d

    def _render_html(self, sparse_index: bool, sparse_columns: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, **kwargs: Any) -> str:
        """
        Renders the ``Styler`` including all applied styles to HTML.
        Generates a dict with necessary kwargs passed to jinja2 template.
        """
        d: dict[str, Any] = self._render(sparse_index, sparse_columns, max_rows, max_cols, '&nbsp;')
        d.update(kwargs)
        return self.template_html.render(**d, html_table_tpl=self.template_html_table, html_style_tpl=self.template_html_style)

    def _render_latex(self, sparse_index: bool, sparse_columns: bool, clines: Optional[str], **kwargs: Any) -> str:
        """
        Render a Styler in latex format
        """
        d: dict[str, Any] = self._render(sparse_index, sparse_columns, None, None)
        self._translate_latex(d, clines=clines)
        self.template_latex.globals['parse_wrap'] = _parse_latex_table_wrapping
        self.template_latex.globals['parse_table'] = _parse_latex_table_styles
        self.template_latex.globals['parse_cell'] = _parse_latex_cell_styles
        self.template_latex.globals['parse_header'] = _parse_latex_header_span
        d.update(kwargs)
        return self.template_latex.render(**d)

    def _render_typst(self, sparse_index: bool, sparse_columns: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, **kwargs: Any) -> str:
        """
        Render a Styler in typst format
        """
        d: dict[str, Any] = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_typst.render(**d)

    def _render_string(self, sparse_index: bool, sparse_columns: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, **kwargs: Any) -> str:
        """
        Render a Styler in string format
        """
        d: dict[str, Any] = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_string.render(**d)

    def _compute(self) -> StylerRenderer:
        """
        Execute the style functions built up in `self._todo`.

        Relies on the conventions that all style functions go through
        .apply or .map. The append styles to apply as tuples of

        (application method, *args, **kwargs)
        """
        self.ctx.clear()
        self.ctx_index.clear()
        self.ctx_columns.clear()
        r: StylerRenderer = self
        for (func, args, kwargs) in self._todo:
            r = func(self)(*args, **kwargs)
        return r

    def _translate(self, sparse_index: bool, sparse_cols: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, blank: str = '&nbsp;', dxs: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """
        Process Styler data and settings into a dict for template rendering.

        Convert data and settings from ``Styler`` attributes such as ``self.data``,
        ``self.tooltips`` including applying any methods in ``self._todo``.

        Parameters
        ----------
        sparse_index : bool
            Whether to sparsify the index or print all hierarchical index elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.index`.
        sparse_cols : bool
            Whether to sparsify the columns or print all hierarchical column elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.columns`.
        max_rows, max_cols : int, optional
            Specific max rows and cols. max_elements always take precedence in render.
        blank : str
            Entry to top-left blank cells.
        dxs : list[dict]
            The render dicts of the concatenated Stylers.

        Returns
        -------
        d : dict
            The following structure: {uuid, table_styles, caption, head, body,
            cellstyle, table_attributes}
        """
        if dxs is None:
            dxs = []
        self.css['blank_value'] = blank
        d: dict[str, Any] = {'uuid': self.uuid, 'table_styles': format_table_styles(self.table_styles or []), 'caption': self.caption}
        max_elements: int = get_option('styler.render.max_elements')
        max_rows = max_rows if max_rows else get_option('styler.render.max_rows')
        max_cols = max_cols if max_cols else get_option('styler.render.max_columns')
        (max_rows, max_cols) = _get_trimming_maximums(len(self.data.index), len(self.data.columns), max_elements, max_rows, max_cols)
        self.cellstyle_map_columns: DefaultDict[tuple[CSSPair, ...], list[str]] = defaultdict(list)
        head: list[list[dict[str, Any]]] = self._translate_header(sparse_cols, max_cols)
        d.update({'head': head})
        idx_lengths: dict[tuple[int, int], int] = _get_level_lengths(self.index, sparse_index, max_rows, self.hidden_rows)
        d.update({'index_lengths': idx_lengths})
        self.cellstyle_map: DefaultDict[tuple[CSSPair, ...], list[str]] = defaultdict(list)
        self.cellstyle_map_index: DefaultDict[tuple[CSSPair, ...], list[str]] = defaultdict(list)
        body: list[list[dict[str, Any]]] = self._translate_body(idx_lengths, max_rows, max_cols)
        d.update({'body': body})
        ctx_maps: dict[str, str] = {'cellstyle': 'cellstyle_map', 'cellstyle_index': 'cellstyle_map_index', 'cellstyle_columns': 'cellstyle_map_columns'}
        for (k, attr) in ctx_maps.items():
            map: list[dict[str, Any]] = [{'props': list(props), 'selectors': selectors} for (props, selectors) in getattr(self, attr).items()]
            d.update({k: map})
        for dx in dxs:
            d['body'].extend(dx['body'])
            d['cellstyle'].extend(dx['cellstyle'])
            d['cellstyle_index'].extend(dx['cellstyle_index'])
        table_attr: Optional[str] = self.table_attributes
        if not get_option('styler.html.mathjax'):
            table_attr = table_attr or ''
            if 'class="' in table_attr:
                table_attr = table_attr.replace('class="', 'class="tex2jax_ignore mathjax_ignore ')
            else:
                table_attr += ' class="tex2jax_ignore mathjax_ignore"'
        d.update({'table_attributes': table_attr})
        if self.tooltips:
            d = self.tooltips._translate(self, d)
        return d

    def _translate_header(self, sparsify_cols: bool, max_cols: int) -> list[list[dict[str, Any]]]:
        """
        Build each <tr> within table <head> as a list

        Using the structure:
             +----------------------------+---------------+---------------------------+
             |  index_blanks ...          | column_name_0 |  column_headers (level_0) |
          1) |       ..                   |       ..      |             ..            |
             |  index_blanks ...          | column_name_n |  column_headers (level_n) |
             +----------------------------+---------------+---------------------------+
          2) |  index_names (level_0 to level_n) ...      | column_blanks ...         |
             +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        sparsify_cols : bool
            Whether column_headers section will add colspan attributes (>1) to elements.
        max_cols : int
            Maximum number of columns to render. If exceeded will contain `...` filler.

        Returns
        -------
        head : list
            The associated HTML elements needed for template rendering.
        """
        col_lengths: dict[tuple[int, int], int] = _get_level_lengths(self.columns, sparsify_cols, max_cols, self.hidden_columns)
        clabels: list[list[Any]] = self.data.columns.tolist()
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))
        head: list[list[dict[str, Any]]] = []
        for (r, hide) in enumerate(self.hide_columns_):
            if hide or not clabels:
                continue
            header_row: list[dict[str, Any]] = self._generate_col_header_row((r, clabels), max_cols, col_lengths)
            head.append(header_row)
        if self.data.index.names and com.any_not_none(*self.data.index.names) and (not all(self.hide_index_)) and (not self.hide_index_names):
            index_names_row: list[dict[str, Any]] = self._generate_index_names_row(clabels, max_cols, col_lengths)
            head.append(index_names_row)
        return head

    def _generate_col_header_row(self, iter: Sequence[Any], max_cols: int, col_lengths: dict[tuple[int, int], int]) -> list[dict[str, Any]]:
        """
        Generate the row containing column headers:

         +----------------------------+---------------+---------------------------+
         |  index_blanks ...          | column_name_i |  column_headers (level_i) |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns
        col_lengths :
            c

        Returns
        -------
        list of elements
        """
        (r, clabels) = iter
        index_blanks: list[dict[str, Any]] = [_element('th', self.css['blank'], self.css['blank_value'], True)] * (self.index.nlevels - sum(self.hide_index_) - 1)
        name: Optional[Any] = self.data.columns.names[r]
        is_display: bool = name is not None and (not self.hide_column_names)
        value: Any = name if is_display else self.css['blank_value']
        display_value: Optional[str] = self._display_funcs_column_names[r](value) if is_display else None
        column_name: list[dict[str, Any]] = [_element('th', f"{self.css['blank']} {self.css['level']}{r}" if name is None else f"{self.css['index_name']} {self.css['