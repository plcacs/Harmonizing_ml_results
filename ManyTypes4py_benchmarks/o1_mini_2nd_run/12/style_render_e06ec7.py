from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
import re
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional, Tuple, Union
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
BaseFormatter = Union[str, Callable[[Any], Any]]
ExtFormatter = Union[BaseFormatter, Dict[Any, Optional[BaseFormatter]]]
CSSPair = Tuple[str, Union[str, float]]
CSSList = List[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    pass
CSSStyles = List[CSSDict]
Subset = Union[slice, Sequence[Any], Index]

class StylerRenderer:
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """
    loader: jinja2.PackageLoader = jinja2.PackageLoader('pandas', 'io/formats/templates')
    env: jinja2.Environment = jinja2.Environment(loader=loader, trim_blocks=True)
    template_html: jinja2.Template = env.get_template('html.tpl')
    template_html_table: jinja2.Template = env.get_template('html_table.tpl')
    template_html_style: jinja2.Template = env.get_template('html_style.tpl')
    template_latex: jinja2.Template = env.get_template('latex.tpl')
    template_typst: jinja2.Template = env.get_template('typst.tpl')
    template_string: jinja2.Template = env.get_template('string.tpl')

    def __init__(
        self,
        data: Union[Series, DataFrame],
        uuid: Optional[str] = None,
        uuid_len: int = 5,
        table_styles: Optional[List[Dict[str, Any]]] = None,
        table_attributes: Optional[str] = None,
        caption: Optional[str] = None,
        cell_ids: bool = True,
        precision: Optional[int] = None
    ) -> None:
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
        self.table_styles: Optional[List[Dict[str, Any]]] = table_styles
        self.table_attributes: Optional[str] = table_attributes
        self.caption: Optional[str] = caption
        self.cell_ids: bool = cell_ids
        self.css: Dict[str, str] = {
            'row_heading': 'row_heading',
            'col_heading': 'col_heading',
            'index_name': 'index_name',
            'col': 'col',
            'row': 'row',
            'col_trim': 'col_trim',
            'row_trim': 'row_trim',
            'level': 'level',
            'data': 'data',
            'blank': 'blank',
            'foot': 'foot'
        }
        self.concatenated: List[StylerRenderer] = []
        self.hide_index_names: bool = False
        self.hide_column_names: bool = False
        self.hide_index_: List[bool] = [False] * self.index.nlevels
        self.hide_columns_: List[bool] = [False] * self.columns.nlevels
        self.hidden_rows: List[int] = []
        self.hidden_columns: List[int] = []
        self.ctx: DefaultDict[Tuple[int, int], List[Any]] = defaultdict(list)
        self.ctx_index: DefaultDict[Tuple[int, int], List[Any]] = defaultdict(list)
        self.ctx_columns: DefaultDict[Tuple[int, int], List[Any]] = defaultdict(list)
        self.cell_context: DefaultDict[Tuple[int, int], str] = defaultdict(str)
        self._todo: List[Tuple[Callable[[StylerRenderer], Any], Tuple[Any, ...], Dict[str, Any]]] = []
        self.tooltips: Optional[Tooltips] = None
        precision = get_option('styler.format.precision') if precision is None else precision
        self._display_funcs: DefaultDict[Tuple[int, int], Callable[[Any], Any]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_index: DefaultDict[Tuple[int, int], Callable[[Any], Any]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_index_names: DefaultDict[Tuple[int, int], Callable[[Any], Any]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_columns: DefaultDict[Tuple[int, int], Callable[[Any], Any]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_column_names: DefaultDict[Tuple[int, int], Callable[[Any], Any]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )

    def _render(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        blank: str = ''
    ) -> Dict[str, Any]:
        """
        Computes and applies styles and then generates the general render dicts.

        Also extends the `ctx` and `ctx_index` attributes with those of concatenated
        stylers for use within `_translate_latex`
        """
        self._compute()
        dxs: List[Dict[str, Any]] = []
        ctx_len: int = len(self.index)
        for i, concatenated in enumerate(self.concatenated):
            concatenated.hide_index_ = self.hide_index_
            concatenated.hidden_columns = self.hidden_columns
            foot = f"{self.css['foot']}{i}"
            concatenated.css = {
                **self.css,
                'data': f"{foot}_data",
                'row_heading': f"{foot}_row_heading",
                'row': f"{foot}_row",
                'foot': f"{foot}_foot"
            }
            dx: Dict[str, Any] = concatenated._render(sparse_index, sparse_columns, max_rows, max_cols, blank)
            dxs.append(dx)
            for (r, c), v in concatenated.ctx.items():
                self.ctx[r + ctx_len, c] = v
            for (r, c), v in concatenated.ctx_index.items():
                self.ctx_index[r + ctx_len, c] = v
            ctx_len += len(concatenated.index)
        d: Dict[str, Any] = self._translate(sparse_index, sparse_columns, max_rows, max_cols, blank, dxs)
        return d

    def _render_html(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Renders the ``Styler`` including all applied styles to HTML.
        Generates a dict with necessary kwargs passed to jinja2 template.
        """
        d: Dict[str, Any] = self._render(sparse_index, sparse_columns, max_rows, max_cols, '&nbsp;')
        d.update(kwargs)
        return self.template_html.render(**d, html_table_tpl=self.template_html_table, html_style_tpl=self.template_html_style)

    def _render_latex(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        clines: Optional[str],
        **kwargs: Any
    ) -> str:
        """
        Render a Styler in latex format
        """
        d: Dict[str, Any] = self._render(sparse_index, sparse_columns, None, None)
        self._translate_latex(d, clines=clines)
        self.template_latex.globals['parse_wrap'] = _parse_latex_table_wrapping
        self.template_latex.globals['parse_table'] = _parse_latex_table_styles
        self.template_latex.globals['parse_cell'] = _parse_latex_cell_styles
        self.template_latex.globals['parse_header'] = _parse_latex_header_span
        d.update(kwargs)
        return self.template_latex.render(**d)

    def _render_typst(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Render a Styler in typst format
        """
        d: Dict[str, Any] = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_typst.render(**d)

    def _render_string(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Render a Styler in string format
        """
        d: Dict[str, Any] = self._render(sparse_index, sparse_columns, max_rows, max_cols)
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
        for func, args, kwargs in self._todo:
            r = func(self)(*args, **kwargs)
        return r

    def _translate(
        self,
        sparse_index: bool,
        sparse_cols: bool,
        max_rows: Optional[int],
        max_cols: Optional[int],
        blank: str,
        dxs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
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
        dxs : list[dict], optional
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
        d: Dict[str, Any] = {
            'uuid': self.uuid,
            'table_styles': format_table_styles(self.table_styles or []),
            'caption': self.caption
        }
        max_elements: int = get_option('styler.render.max_elements')
        max_rows = max_rows if max_rows else get_option('styler.render.max_rows')
        max_cols = max_cols if max_cols else get_option('styler.render.max_columns')
        max_rows, max_cols = _get_trimming_maximums(len(self.data.index), len(self.data.columns), max_elements, max_rows, max_cols)
        self.cellstyle_map_columns: DefaultDict[Tuple[Any, ...], List[str]] = defaultdict(list)
        head: List[List[Dict[str, Any]]] = self._translate_header(sparse_cols, max_cols)
        d.update({'head': head})
        idx_lengths: Dict[Tuple[int, int], int] = _get_level_lengths(self.index, sparse_index, max_rows, self.hidden_rows)
        d.update({'index_lengths': idx_lengths})
        self.cellstyle_map: DefaultDict[Tuple[Any, ...], List[str]] = defaultdict(list)
        self.cellstyle_map_index: DefaultDict[Tuple[Any, ...], List[str]] = defaultdict(list)
        body: List[List[Dict[str, Any]]] = self._translate_body(idx_lengths, max_rows, max_cols)
        d.update({'body': body})
        ctx_maps: Dict[str, str] = {
            'cellstyle': 'cellstyle_map',
            'cellstyle_index': 'cellstyle_map_index',
            'cellstyle_columns': 'cellstyle_map_columns'
        }
        for k, attr in ctx_maps.items():
            map_list: List[Dict[str, Any]] = [
                {'props': list(props), 'selectors': selectors}
                for props, selectors in getattr(self, attr).items()
            ]
            d.update({k: map_list})
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

    def _translate_header(
        self,
        sparsify_cols: bool,
        max_cols: int
    ) -> List[List[Dict[str, Any]]]:
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
        col_lengths: Dict[Tuple[int, int], int] = _get_level_lengths(self.columns, sparsify_cols, max_cols, self.hidden_columns)
        clabels: List[Any] = self.data.columns.tolist()
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))
        head: List[List[Dict[str, Any]]] = []
        for r, hide in enumerate(self.hide_columns_):
            if hide or not clabels:
                continue
            header_row: List[Dict[str, Any]] = self._generate_col_header_row((r, clabels), max_cols, col_lengths)
            head.append(header_row)
        if (
            self.data.index.names
            and com.any_not_none(*self.data.index.names)
            and (not all(self.hide_index_))
            and (not self.hide_index_names)
        ):
            index_names_row: List[Dict[str, Any]] = self._generate_index_names_row(clabels, max_cols, col_lengths)
            head.append(index_names_row)
        return head

    def _generate_col_header_row(
        self,
        iter: Tuple[int, List[Any]],
        max_cols: int,
        col_lengths: Dict[Tuple[int, int], int]
    ) -> List[Dict[str, Any]]:
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
        col_lengths : Dict[Tuple[int, int], int]
            c

        Returns
        -------
        list of elements
        """
        r, clabels = iter
        index_blanks: List[Dict[str, Any]] = [_element('th', self.css['blank'], self.css['blank_value'], True)] * (self.index.nlevels - sum(self.hide_index_) - 1)
        name: Optional[str] = self.data.columns.names[r]
        is_display: bool = name is not None and (not self.hide_column_names)
        value: str = name if is_display else self.css['blank_value']
        display_value: Optional[Any] = self._display_funcs_column_names[r](value) if is_display else None
        column_name: List[Dict[str, Any]] = [
            _element(
                'th',
                f"{self.css['blank']} {self.css['level']}{r}" if name is None else f"{self.css['index_name']} {self.css['level']}{r}",
                value,
                not all(self.hide_index_),
                display_value=display_value
            )
        ]
        column_headers: List[Dict[str, Any]] = []
        visible_col_count: int = 0
        for c, value in enumerate(clabels[r]):
            header_element_visible: bool = _is_visible(c, r, col_lengths)
            if header_element_visible:
                visible_col_count += col_lengths.get((r, c), 0)
            if self._check_trim(
                visible_col_count,
                max_cols,
                column_headers,
                'th',
                f"{self.css['col_heading']} {self.css['level']}{r} {self.css['col_trim']}"
            ):
                break
            header_element: Dict[str, Any] = _element(
                'th',
                f"{self.css['col_heading']} {self.css['level']}{r} {self.css['col']}{c}",
                value,
                header_element_visible,
                display_value=self._display_funcs_columns[r, c](value),
                attributes=f'colspan="{col_lengths.get((r, c), 0)}"' if col_lengths.get((r, c), 0) > 1 else ''
            )
            if self.cell_ids:
                header_element['id'] = f"{self.css['level']}{r}_col{c}"
            if header_element_visible and (r, c) in self.ctx_columns and self.ctx_columns[r, c]:
                header_element['id'] = f"{self.css['level']}{r}_col{c}"
                self.cellstyle_map_columns[tuple(self.ctx_columns[r, c])].append(f"{self.css['level']}{r}_col{c}")
            column_headers.append(header_element)
        return index_blanks + column_name + column_headers

    def _generate_index_names_row(
        self,
        iter: List[Any],
        max_cols: int,
        col_lengths: Dict[Tuple[int, int], int]
    ) -> List[Dict[str, Any]]:
        """
        Generate the row containing index names

         +----------------------------+---------------+---------------------------+
         |  index_names (level_0 to level_n) ...      | column_blanks ...         |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : List[Any]
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns

        Returns
        -------
        list of elements
        """
        clabels = iter
        index_names: List[Dict[str, Any]] = [
            _element(
                'th',
                f"{self.css['index_name']} {self.css['level']}{c}",
                self.css['blank_value'] if name is None else name,
                not self.hide_index_[c],
                display_value=None if name is None else self._display_funcs_index_names[c](name)
            )
            for c, name in enumerate(self.data.index.names)
        ]
        column_blanks: List[Dict[str, Any]] = []
        visible_col_count: int = 0
        if clabels:
            last_level: int = self.columns.nlevels - 1
            for c, value in enumerate(clabels[last_level]):
                header_element_visible: bool = _is_visible(c, last_level, col_lengths)
                if header_element_visible:
                    visible_col_count += 1
                if self._check_trim(
                    visible_col_count,
                    max_cols,
                    column_blanks,
                    'th',
                    f"{self.css['blank']} col{c} {self.css['col_trim']}",
                    self.css['blank_value']
                ):
                    break
                column_blanks.append(
                    _element(
                        'th',
                        f"{self.css['blank']} col{c} {self.css['col_trim']}",
                        self.css['blank_value'],
                        c not in self.hidden_columns
                    )
                )
        return index_names + column_blanks

    def _translate_body(
        self,
        idx_lengths: Dict[Tuple[int, int], int],
        max_rows: int,
        max_cols: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Build each <tr> within table <body> as a list

        Use the following structure:
          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Also add elements to the cellstyle_map for more efficient grouped elements in
        <style></style> block

        Parameters
        ----------
        sparsify_index : bool
            Whether index_headers section will add rowspan attributes (>1) to elements.

        Returns
        -------
        body : list
            The associated HTML elements needed for template rendering.
        """
        rlabels: List[Any] = self.data.index.tolist()
        if not isinstance(self.data.index, MultiIndex):
            rlabels = [[x] for x in rlabels]
        body: List[List[Dict[str, Any]]] = []
        visible_row_count: int = 0
        for r, row_tup in enumerate(self.data.itertuples(index=True, name=None)):
            if r in self.hidden_rows:
                continue
            visible_row_count += 1
            if self._check_trim(visible_row_count, max_rows, body, 'row'):
                break
            row_body: List[Dict[str, Any]] = self._generate_body_row((r, row_tup, rlabels), max_cols, idx_lengths)
            body.append(row_body)
        return body

    def _check_trim(
        self,
        count: int,
        max_: int,
        obj: List[Any],
        element: str,
        css: Optional[str] = None,
        value: str = '...'
    ) -> bool:
        """
        Indicates whether to break render loops and append a trimming indicator

        Parameters
        ----------
        count : int
            The loop count of previous visible items.
        max : int
            The allowable rendered items in the loop.
        obj : list
            The current render collection of the rendered items.
        element : str
            The type of element to append in the case a trimming indicator is needed.
        css : str, optional
            The css to add to the trimming indicator element.
        value : str, optional
            The value of the elements display if necessary.

        Returns
        -------
        result : bool
            Whether a trimming element was required and appended.
        """
        if count > max_:
            if element == 'row':
                obj.append(self._generate_trimmed_row(max_))
            else:
                obj.append(_element(element, css, value, True, attributes=''))
            return True
        return False

    def _generate_trimmed_row(self, max_cols: int) -> List[Dict[str, Any]]:
        """
        When a render has too many rows we generate a trimming row containing "..."

        Parameters
        ----------
        max_cols : int
            Number of permissible columns

        Returns
        -------
        list of elements
        """
        index_headers: List[Dict[str, Any]] = [
            _element(
                'th',
                f"{self.css['row_heading']} {self.css['level']}{c} {self.css['row_trim']}",
                '...',
                not self.hide_index_[c],
                attributes=''
            )
            for c in range(self.data.index.nlevels)
        ]
        data: List[Dict[str, Any]] = []
        visible_col_count: int = 0
        for c in range(len(self.columns)):
            data_element_visible: bool = c not in self.hidden_columns
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(
                visible_col_count,
                max_cols,
                data,
                'td',
                f"{self.css['data']} {self.css['row_trim']} {self.css['col_trim']}"
            ):
                break
            data.append(
                _element(
                    'td',
                    f"{self.css['data']} row{c} col{c} {self.css['row_trim']}",
                    '...',
                    data_element_visible,
                    attributes=''
                )
            )
        return index_headers + data

    def _generate_body_row(
        self,
        iter: Tuple[int, Tuple[Any, ...], List[List[Any]]],
        max_cols: int,
        idx_lengths: Dict[Tuple[int, int], int]
    ) -> List[Dict[str, Any]]:
        """
        Generate a regular row for the body section of appropriate format.

          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Iterable from outer scope: row number, row data tuple, row index labels.
        max_cols : int
            Number of permissible columns.
        idx_lengths : dict
            A map of the sparsification structure of the index

        Returns
        -------
            list of elements
        """
        r, row_tup, rlabels = iter
        index_headers: List[Dict[str, Any]] = []
        for c, value in enumerate(rlabels[r]):
            header_element_visible: bool = _is_visible(r, c, idx_lengths) and (not self.hide_index_[c])
            header_element: Dict[str, Any] = _element(
                'th',
                f"{self.css['row_heading']} {self.css['level']}{c} {self.css['row']}{r}",
                value,
                header_element_visible,
                display_value=self._display_funcs_index[r, c](value),
                attributes=f'rowspan="{idx_lengths.get((c, r), 0)}"' if idx_lengths.get((c, r), 0) > 1 else ''
            )
            if self.cell_ids:
                header_element['id'] = f"{self.css['level']}{c}_row{r}"
            if header_element_visible and (r, c) in self.ctx_index and self.ctx_index[r, c]:
                header_element['id'] = f"{self.css['level']}{c}_row{r}"
                self.cellstyle_map_index[tuple(self.ctx_index[r, c])].append(f"{self.css['level']}{c}_row{r}")
            index_headers.append(header_element)
        data: List[Dict[str, Any]] = []
        visible_col_count: int = 0
        for c, value in enumerate(row_tup[1:]):
            data_element_visible: bool = c not in self.hidden_columns and r not in self.hidden_rows
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(
                visible_col_count,
                max_cols,
                data,
                'td',
                f"{self.css['data']} row{r} col_trim"
            ):
                break
            cls: str = ''
            if (r, c) in self.cell_context:
                cls = ' ' + self.cell_context[r, c]
            data_element: Dict[str, Any] = _element(
                'td',
                f"{self.css['data']} row{r} col{c}{cls}",
                value,
                data_element_visible,
                attributes='',
                display_value=self._display_funcs[r, c](value)
            )
            if self.cell_ids:
                data_element['id'] = f"row{r}_col{c}"
            if data_element_visible and (r, c) in self.ctx and self.ctx[r, c]:
                data_element['id'] = f"row{r}_col{c}"
                self.cellstyle_map[tuple(self.ctx[r, c])].append(f"row{r}_col{c}")
            data.append(data_element)
        return index_headers + data

    def _translate_latex(
        self,
        d: Dict[str, Any],
        clines: Optional[str]
    ) -> None:
        """
        Post-process the default render dict for the LaTeX template format.

        Processing items included are:
          - Remove hidden columns from the non-headers part of the body.
          - Place cellstyles directly in td cells rather than use cellstyle_map.
          - Remove hidden indexes or reinsert missing th elements if part of multiindex
            or multirow sparsification (so that \\multirow and \\multicol work correctly).
        """
        index_levels: int = self.index.nlevels
        visible_index_level_n: int = max(1, index_levels - sum(self.hide_index_))
        d['head'] = [
            [
                {**col, 'cellstyle': self.ctx_columns[r, c - visible_index_level_n]}
                for c, col in enumerate(row)
                if col['is_visible']
            ]
            for r, row in enumerate(d['head'])
        ]

        def _concatenated_visible_rows(obj: StylerRenderer, n: int, row_indices: List[int]) -> int:
            """
            Extract all visible row indices recursively from concatenated stylers.
            """
            row_indices.extend([r + n for r in range(len(obj.index)) if r not in obj.hidden_rows])
            n += len(obj.index)
            for concatenated in obj.concatenated:
                n = _concatenated_visible_rows(concatenated, n, row_indices)
            return n

        def concatenated_visible_rows(obj: StylerRenderer) -> List[int]:
            row_indices: List[int] = []
            _concatenated_visible_rows(obj, 0, row_indices)
            return row_indices

        body: List[List[Dict[str, Any]]] = []
        for r, row in zip(concatenated_visible_rows(self), d['body']):
            if all(self.hide_index_):
                row_body_headers: List[Dict[str, Any]] = []
            else:
                row_body_headers = [
                    {
                        **col,
                        'display_value': col['display_value'] if col['is_visible'] else '',
                        'cellstyle': self.ctx_index[r, c]
                    }
                    for c, col in enumerate(row[:index_levels])
                    if col['type'] == 'th' and (not self.hide_index_[c])
                ]
            row_body_cells: List[Dict[str, Any]] = [
                {**col, 'cellstyle': self.ctx[r, c]}
                for c, col in enumerate(row[index_levels:])
                if col['is_visible'] and col['type'] == 'td'
            ]
            body.append(row_body_headers + row_body_cells)
        d['body'] = body
        if clines not in [None, 'all;data', 'all;index', 'skip-last;data', 'skip-last;index']:
            raise ValueError(f"`clines` value of {clines} is invalid. Should either be None or one of 'all;data', 'all;index', 'skip-last;data', 'skip-last;index'.")
        if clines is not None:
            data_len: int = len(row_body_cells) if 'data' in clines and d['body'] else 0
            d['clines']: DefaultDict[int, List[str]] = defaultdict(list)
            visible_row_indexes: List[int] = [r for r in range(len(self.data.index)) if r not in self.hidden_rows]
            visible_index_levels: List[int] = [i for i in range(index_levels) if not self.hide_index_[i]]
            for rn, r in enumerate(visible_row_indexes):
                for lvln, lvl in enumerate(visible_index_levels):
                    if lvln == index_levels - 1 and 'skip-last' in clines:
                        continue
                    idx_len: Optional[int] = d['index_lengths'].get((lvl, r), None)
                    if idx_len is not None:
                        d['clines'][rn + idx_len].append(f"\\cline{{{lvln + 1}-{len(visible_index_levels) + data_len}}}")

    def _translate_header(
        self,
        sparsify_cols: bool,
        max_cols: int
    ) -> List[List[Dict[str, Any]]]:
        # This method is already defined above with type annotations
        pass  # Placeholder

    def format(
        self,
        formatter: Optional[ExtFormatter] = None,
        subset: Optional[Subset] = None,
        na_rep: Optional[str] = None,
        precision: Optional[int] = None,
        decimal: str = '.',
        thousands: Optional[str] = None,
        escape: Optional[str] = None,
        hyperlinks: Optional[str] = None
    ) -> StylerRenderer:
        """
        Format the text display value of cells.

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.
        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.
        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers.
        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers.
        escape : str, optional
            Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\\`` in the cell display string with
            LaTeX-safe sequences.
            Use 'latex-math' to replace the characters the same way as in 'latex' mode,
            except for math substrings, which either are surrounded
            by two characters ``$`` or start with the character ``\\(`` and
            end with ``\\)``. Escaping is done before ``formatter``.
        hyperlinks : {"html", "latex"}, optional
            Convert string patterns containing https://, http://, ftp:// or www. to
            HTML <a> tags as clickable URL hyperlinks if "html", or LaTeX \\href
            commands if "latex".

        Returns
        -------
        Styler
            Returns itself for chaining.
        """
        if all(
            (
                formatter is None,
                subset is None,
                precision is None,
                decimal == '.',
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None
            )
        ):
            self._display_funcs.clear()
            return self
        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data: DataFrame = self.data.loc[subset]
        if not isinstance(formatter, dict):
            formatter = {col: formatter for col in data.columns}
        cis: List[int] = self.columns.get_indexer_for(data.columns)
        ris: List[int] = self.index.get_indexer_for(data.index)
        for ci in cis:
            format_func: Callable[[Any], Any] = _maybe_wrap_formatter(
                formatter.get(self.columns[ci]),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks
            )
            for ri in ris:
                self._display_funcs[ri, ci] = format_func
        return self

    def format_index(
        self,
        formatter: Optional[ExtFormatter] = None,
        axis: Union[int, str] = 0,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        na_rep: Optional[str] = None,
        precision: Optional[int] = None,
        decimal: str = '.',
        thousands: Optional[str] = None,
        escape: Optional[str] = None,
        hyperlinks: Optional[str] = None
    ) -> StylerRenderer:
        """
        Format the text display value of index labels or column headers.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        axis : {0, "index", 1, "columns"}
            Whether to apply the formatter to the index or column headers.
        level : int, str, list
            The level(s) over which to apply the generic formatter.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.
        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.
        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers.
        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers.
        escape : str, optional
            Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\\`` in the cell display string with
            LaTeX-safe sequences.
            Escaping is done before ``formatter``.
        hyperlinks : {"html", "latex"}, optional
            Convert string patterns containing https://, http://, ftp:// or www. to
            HTML <a> tags as clickable URL hyperlinks if "html", or LaTeX \\href
            commands if "latex".

        Returns
        -------
        Styler
            Returns itself for chaining.
        """
        axis_num: int = self.data._get_axis_number(axis)
        if axis_num == 0:
            display_funcs_, obj = (self._display_funcs_index, self.index)
        else:
            display_funcs_, obj = (self._display_funcs_columns, self.columns)
        levels_: List[int] = refactor_levels(level, obj)
        if all(
            (
                formatter is None,
                level is None,
                precision is None,
                decimal == '.',
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None
            )
        ):
            display_funcs_.clear()
            return self
        if not isinstance(formatter, dict):
            formatter = {lev: formatter for lev in levels_}
        else:
            formatter = {obj._get_level_number(lev): fmt for lev, fmt in formatter.items()}
        for lvl in levels_:
            format_func: Callable[[Any], Any] = _maybe_wrap_formatter(
                formatter.get(lvl),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks
            )
            for idx in (
                [(i, lvl) for i in range(len(obj))]
                if axis_num == 0
                else [(lvl, i) for i in range(len(obj))]
            ):
                display_funcs_[idx] = format_func
        return self

    def relabel_index(
        self,
        labels: Union[List[Any], Index],
        axis: Union[int, str] = 0,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None
    ) -> StylerRenderer:
        """
        Relabel the index, or column header, keys to display a set of specified values.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        labels : list-like or Index
            New labels to display. Must have same length as the underlying values not
            hidden.
        axis : {"index", 0, "columns", 1}
            Apply to the index or columns.
        level : int, str, list, optional
            The level(s) over which to apply the new labels. If `None` will apply
            to all levels of an Index or MultiIndex which are not hidden.

        Returns
        -------
        Styler
            Returns itself for chaining.
        """
        axis_num: int = self.data._get_axis_number(axis)
        if axis_num == 0:
            display_funcs_, obj = (self._display_funcs_index, self.index)
            hidden_labels, hidden_lvls = (self.hidden_rows, self.hide_index_)
        else:
            display_funcs_, obj = (self._display_funcs_columns, self.columns)
            hidden_labels, hidden_lvls = (self.hidden_columns, self.hide_columns_)
        visible_len: int = len(obj) - len(set(hidden_labels))
        if len(labels) != visible_len:
            raise ValueError(f'``labels`` must be of length equal to the number of visible labels along ``axis`` ({visible_len}).')
        if level is None:
            levels_: List[int] = [i for i in range(obj.nlevels) if not hidden_lvls[i]]
        else:
            levels_: List[int] = refactor_levels(level, obj)

        def alias_(x: Any, value: Any) -> Any:
            if isinstance(value, str):
                return value.format(x)
            return value

        visible_indices: List[int] = [i for i in range(len(obj)) if i not in hidden_labels]
        for ai, i in enumerate(visible_indices):
            if len(levels_) == 1:
                idx: Tuple[int, int] = (i, levels_[0]) if axis_num == 0 else (levels_[0], i)
                display_funcs_[idx] = partial(alias_, value=labels[ai])
            else:
                for aj, lvl in enumerate(levels_):
                    idx: Tuple[int, int] = (i, lvl) if axis_num == 0 else (lvl, i)
                    display_funcs_[idx] = partial(alias_, value=labels[ai][aj])
        return self

    def format_index_names(
        self,
        formatter: Optional[ExtFormatter] = None,
        axis: Union[int, str] = 0,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        na_rep: Optional[str] = None,
        precision: Optional[int] = None,
        decimal: str = '.',
        thousands: Optional[str] = None,
        escape: Optional[str] = None,
        hyperlinks: Optional[str] = None
    ) -> StylerRenderer:
        """
        Format the text display value of index names or column names.

        .. versionadded:: 3.0

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        axis : {0, "index", 1, "columns"}
            Whether to apply the formatter to the index or column headers.
        level : int, str, list
            The level(s) over which to apply the generic formatter.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.
        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.
        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers.
        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers.
        escape : str, optional
            Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\\`` in the cell display string with
            LaTeX-safe sequences.
            Escaping is done before ``formatter``.
        hyperlinks : {"html", "latex"}, optional
            Convert string patterns containing https://, http://, ftp:// or www. to
            HTML <a> tags as clickable URL hyperlinks if "html", or LaTeX \\href
            commands if "latex".

        Returns
        -------
        Styler
            Returns itself for chaining.
        """
        axis_num: int = self.data._get_axis_number(axis)
        if axis_num == 0:
            display_funcs_, obj = (self._display_funcs_index_names, self.index)
        else:
            display_funcs_, obj = (self._display_funcs_column_names, self.columns)
        levels_: List[int] = refactor_levels(level, obj)
        if all(
            (
                formatter is None,
                level is None,
                precision is None,
                decimal == '.',
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None
            )
        ):
            display_funcs_.clear()
            return self
        if not isinstance(formatter, dict):
            formatter = {lev: formatter for lev in levels_}
        else:
            formatter = {obj._get_level_number(lev): fmt for lev, fmt in formatter.items()}
        for lvl in levels_:
            format_func: Callable[[Any], Any] = _maybe_wrap_formatter(
                formatter.get(lvl),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks
            )
            display_funcs_[lvl] = format_func
        return self

def _element(
    html_element: str,
    html_class: str,
    value: Any,
    is_visible: bool,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Template to return container with information for a <td></td> or <th></th> element.
    """
    if 'display_value' not in kwargs or kwargs['display_value'] is None:
        kwargs['display_value'] = value
    return {'type': html_element, 'value': value, 'class': html_class, 'is_visible': is_visible, **kwargs}

def _get_trimming_maximums(
    rn: int,
    cn: int,
    max_elements: int,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    scaling_factor: float = 0.8
) -> Tuple[int, int]:
    """
    Recursively reduce the number of rows and columns to satisfy max elements.

    Parameters
    ----------
    rn, cn : int
        The number of input rows / columns
    max_elements : int
        The number of allowable elements
    max_rows, max_cols : int, optional
        Directly specify an initial maximum rows or columns before compression.
    scaling_factor : float
        Factor at which to reduce the number of rows / columns to fit.

    Returns
    -------
    rn, cn : tuple
        New rn and cn values that satisfy the max_elements constraint
    """

    def scale_down(rn_inner: int, cn_inner: int) -> Tuple[int, int]:
        if cn_inner >= rn_inner:
            return (rn_inner, int(cn_inner * scaling_factor))
        else:
            return (int(rn_inner * scaling_factor), cn_inner)
    if max_rows:
        rn = max_rows if rn > max_rows else rn
    if max_cols:
        cn = max_cols if cn > max_cols else cn
    while rn * cn > max_elements:
        rn, cn = scale_down(rn, cn)
    return (rn, cn)

def _get_level_lengths(
    index: Union[Index, MultiIndex],
    sparsify: bool,
    max_index: int,
    hidden_elements: Optional[Sequence[int]] = None
) -> Dict[Tuple[int, int], int]:
    """
    Given an index, find the level length for each element.

    Parameters
    ----------
    index : Index
        Index or columns to determine lengths of each element
    sparsify : bool
        Whether to hide or show each distinct element in a MultiIndex
    max_index : int
        The maximum number of elements to analyse along the index due to trimming
    hidden_elements : sequence of int, optional
        Index positions of elements hidden from display in the index affecting
        length

    Returns
    -------
    Dict[Tuple[int, int], int] :
        Result is a dictionary of (level, initial_position): span
    """
    if isinstance(index, MultiIndex):
        levels = index._format_multi(sparsify=lib.no_default, include_names=False)
    else:
        levels = index._format_flat(include_name=False)
    if hidden_elements is None:
        hidden_elements = []
    lengths: Dict[Tuple[int, int], int] = {}
    if not isinstance(index, MultiIndex):
        for i, value in enumerate(levels):
            if i not in hidden_elements:
                lengths[0, i] = 1
        return lengths
    for i, lvl in enumerate(levels):
        visible_row_count: int = 0
        for j, row in enumerate(lvl):
            if visible_row_count > max_index:
                break
            if not sparsify:
                if j not in hidden_elements:
                    lengths[i, j] = 1
                    visible_row_count += 1
            elif row is not lib.no_default and j not in hidden_elements:
                last_label = j
                lengths[i, last_label] = 1
                visible_row_count += 1
            elif row is not lib.no_default:
                last_label = j
                lengths[i, last_label] = 0
            elif j not in hidden_elements:
                visible_row_count += 1
                if visible_row_count > max_index:
                    break
                if lengths.get((i, last_label), 0) == 0:
                    last_label = j
                    lengths[i, last_label] = 1
                else:
                    lengths[i, last_label] += 1
    non_zero_lengths: Dict[Tuple[int, int], int] = {element: length for element, length in lengths.items() if length >= 1}
    return non_zero_lengths

def _is_visible(idx_row: int, idx_col: int, lengths: Dict[Tuple[int, int], int]) -> bool:
    """
    Index -> {(idx_row, idx_col): bool}).
    """
    return (idx_col, idx_row) in lengths

def format_table_styles(styles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    looks for multiple CSS selectors and separates them:
    [{'selector': 'td, th', 'props': 'a:v;'}]
        ---> [{'selector': 'td', 'props': 'a:v;'},
              {'selector': 'th', 'props': 'a:v;'}]
    """
    return [{'selector': selector, 'props': css_dict['props']} for css_dict in styles for selector in css_dict['selector'].split(',')]

def _default_formatter(x: Any, precision: int, thousands: bool = False) -> Union[str, Any]:
    """
    Format the display of a value

    Parameters
    ----------
    x : Any
        Input variable to be formatted
    precision : Int
        Floating point precision used if ``x`` is float or complex.
    thousands : bool, default False
        Whether to group digits with thousands separated with ",".

    Returns
    -------
    value : Any
        Matches input type, or string if input is float or complex or int with sep.
    """
    if is_float(x) or is_complex(x):
        return f"{x:,.{precision}f}" if thousands else f"{x:.{precision}f}"
    elif is_integer(x):
        return f"{x:,}" if thousands else str(x)
    return x

def _wrap_decimal_thousands(formatter: Callable[[Any], Any], decimal: str, thousands: Optional[str]) -> Callable[[Any], Any]:
    """
    Takes a string formatting function and wraps logic to deal with thousands and
    decimal parameters, in the case that they are non-standard and that the input
    is a (float, complex, int).
    """

    def wrapper(x: Any) -> Any:
        if is_float(x) or is_integer(x) or is_complex(x):
            if decimal != '.' and thousands is not None and (thousands != ','):
                return formatter(x).replace(',', '§_§-').replace('.', decimal).replace('§_§-', thousands)
            elif decimal != '.' and (thousands is None or thousands == ','):
                return formatter(x).replace('.', decimal)
            elif decimal == '.' and thousands is not None and (thousands != ','):
                return formatter(x).replace(',', thousands)
        return formatter(x)
    return wrapper

def _str_escape(x: Any, escape: str) -> Any:
    """if escaping: only use on str, else return input"""
    if isinstance(x, str):
        if escape == 'html':
            return escape_html(x)
        elif escape == 'latex':
            return _escape_latex(x)
        elif escape == 'latex-math':
            return _escape_latex_math(x)
        else:
            raise ValueError(f"`escape` only permitted in {{'html', 'latex', 'latex-math'}}, got {escape}")
    return x

def _render_href(x: Any, format: str) -> Any:
    """uses regex to detect a common URL pattern and converts to href tag in format."""
    if isinstance(x, str):
        if format == 'html':
            href: str = '<a href="{0}" target="_blank">{0}</a>'
        elif format == 'latex':
            href = '\\href{{{0}}}{{{0}}}'
        else:
            raise ValueError("``hyperlinks`` format can only be 'html' or 'latex'")
        pat: str = "((http|ftp)s?:\\/\\/|www.)[\\w/\\-?=%.:@]+\\.[\\w/\\-&?=%.,':;~!@#$*()\\[\\]]+"
        return re.sub(pat, lambda m: href.format(m.group(0)), x)
    return x

def _maybe_wrap_formatter(
    formatter: Optional[ExtFormatter] = None,
    na_rep: Optional[str] = None,
    precision: Optional[int] = None,
    decimal: str = '.',
    thousands: Optional[str] = None,
    escape: Optional[str] = None,
    hyperlinks: Optional[str] = None
) -> Callable[[Any], Any]:
    """
    Allows formatters to be expressed as str, callable or None, where None returns
    a default formatting function. wraps with na_rep, and precision where they are
    available.
    """
    if isinstance(formatter, str):
        func_0: Callable[[Any], Any] = lambda x: formatter.format(x)
    elif callable(formatter):
        func_0 = formatter
    elif formatter is None:
        precision = get_option('styler.format.precision') if precision is None else precision
        func_0 = partial(_default_formatter, precision=precision, thousands=thousands is not None)
    else:
        raise TypeError(f"'formatter' expected str or callable, got {type(formatter)}")
    if escape is not None:
        func_1: Callable[[Any], Any] = lambda x: func_0(_str_escape(x, escape=escape))
    else:
        func_1 = func_0
    if decimal != '.' or (thousands is not None and thousands != ','):
        func_2: Callable[[Any], Any] = _wrap_decimal_thousands(func_1, decimal=decimal, thousands=thousands)
    else:
        func_2 = func_1
    if hyperlinks is not None:
        func_3: Callable[[Any], Any] = lambda x: func_2(_render_href(x, format=hyperlinks))
    else:
        func_3 = func_2
    if na_rep is None:
        return func_3
    else:
        return lambda x: na_rep if isna(x) else func_3(x)

def non_reducing_slice(slice_: Any) -> Tuple[Any, ...]:
    """
    Ensure that a slice doesn't reduce to a Series or Scalar.

    Any user-passed `subset` should have this called on it
    to make sure we're always working with DataFrames.
    """
    kinds = (ABCSeries, np.ndarray, Index, list, str)
    if isinstance(slice_, kinds):
        slice_ = IndexSlice[:, slice_]

    def pred(part: Any) -> bool:
        """
        Returns
        -------
        bool
            True if slice does *not* reduce,
            False if `part` is a tuple.
        """
        if isinstance(part, tuple):
            return any((isinstance(s, slice) or is_list_like(s) for s in part))
        else:
            return isinstance(part, slice) or is_list_like(part)
    if not is_list_like(slice_):
        if not isinstance(slice_, slice):
            slice_ = [[slice_]]
        else:
            slice_ = [slice_]
    else:
        slice_ = [p if pred(p) else [p] for p in slice_]
    return tuple(slice_)

def maybe_convert_css_to_tuples(style: Union[str, CSSList]) -> CSSList:
    """
    Convert css-string to sequence of tuples format if needed.
    'color:red; border:1px solid black;' -> [('color', 'red'),
                                             ('border','1px solid red')]
    """
    if isinstance(style, str):
        if style and ':' not in style:
            raise ValueError(f"Styles supplied as string must follow CSS rule formats, for example 'attr: val;'. '{style}' was given.")
        s: List[str] = style.split(';')
        return [(x.split(':')[0].strip(), ':'.join(x.split(':')[1:]).strip()) for x in s if x.strip() != '']
    return style

def refactor_levels(level: Optional[Union[int, str, List[Union[int, str]]]], obj: Union[Index, MultiIndex]) -> List[int]:
    """
    Returns a consistent levels arg for use in ``hide_index`` or ``hide_columns``.

    Parameters
    ----------
    level : int, str, list
        Original ``level`` arg supplied to above methods.
    obj: Index or MultiIndex
        Either ``self.index`` or ``self.columns``

    Returns
    -------
    list : refactored arg with a list of levels to hide
    """
    if level is None:
        levels_ = list(range(obj.nlevels))
    elif isinstance(level, int):
        levels_ = [level]
    elif isinstance(level, str):
        levels_ = [obj._get_level_number(level)]
    elif isinstance(level, list):
        levels_ = [obj._get_level_number(lev) if not isinstance(lev, int) else lev for lev in level]
    else:
        raise ValueError('`level` must be of type `int`, `str` or list of such')
    return levels_

class Tooltips:
    """
    An extension to ``Styler`` that allows for and manipulates tooltips on hover
    of ``<td>`` cells in the HTML result.

    Parameters
    ----------
    css_name: str, default "pd-t"
        Name of the CSS class that controls visualisation of tooltips.
    css_props: list-like, default; see Notes
        List of (attr, value) tuples defining properties of the CSS class.
    tooltips: DataFrame, default empty
        DataFrame of strings aligned with underlying Styler data for tooltip
        display.
    as_title_attribute: bool, default False
        Flag to use title attribute based tooltips (True) or <span> based
        tooltips (False).
        Add the tooltip text as title attribute to resultant <td> element. If
        True, no CSS is generated and styling effects do not apply.

    Notes
    -----
    The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

    Hidden visibility is a key prerequisite to the hover functionality, and should
    always be included in any manual properties specification.
    """

    def __init__(
        self,
        css_props: List[Tuple[str, Union[str, int]]] = [
            ('visibility', 'hidden'),
            ('position', 'absolute'),
            ('z-index', 1),
            ('background-color', 'black'),
            ('color', 'white'),
            ('transform', 'translate(-20px, -20px)')
        ],
        css_name: str = 'pd-t',
        tooltips: DataFrame = DataFrame(),
        as_title_attribute: bool = False
    ) -> None:
        self.class_name: str = css_name
        self.class_properties: List[Tuple[str, Union[str, int]]] = css_props
        self.tt_data: DataFrame = tooltips
        self.table_styles: List[Dict[str, Any]] = []
        self.as_title_attribute: bool = as_title_attribute

    @property
    def _class_styles(self) -> List[Dict[str, Any]]:
        """
        Combine the ``_Tooltips`` CSS class name and CSS properties to the format
        required to extend the underlying ``Styler`` `table_styles` to allow
        tooltips to render in HTML.

        Returns
        -------
        styles : List
        """
        return [{'selector': f'.{self.class_name}', 'props': maybe_convert_css_to_tuples(self.class_properties)}]

    def _pseudo_css(
        self,
        uuid: str,
        name: str,
        row: int,
        col: int,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        For every table data-cell that has a valid tooltip (not None, NaN or
        empty string) must create two pseudo CSS entries for the specific
        <td> element id which are added to overall table styles:
        an on hover visibility change and a content change
        dependent upon the user's chosen display string.

        For example:
            [{"selector": "T__row1_col1:hover .pd-t",
             "props": [("visibility", "visible")]},
            {"selector": "T__row1_col1 .pd-t::after",
             "props": [("content", "Some Valid Text String")]}]

        Parameters
        ----------
        uuid: str
            The uuid of the Styler instance
        name: str
            The css-name of the class used for styling tooltips
        row : int
            The row index of the specified tooltip string data
        col : int
            The col index of the specified tooltip string data
        text : str
            The textual content of the tooltip to be displayed in HTML.

        Returns
        -------
        pseudo_css : List[Dict[str, Any]]
        """
        selector_id: str = '#T_' + uuid + '_row' + str(row) + '_col' + str(col)
        return [
            {
                'selector': selector_id + f':hover .{name}',
                'props': [('visibility', 'visible')]
            },
            {
                'selector': selector_id + f' .{name}::after',
                'props': [('content', f'"{text}"')]
            }
        ]

    def _translate(self, styler: StylerRenderer, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate the render dictionary to allow for tooltips:

        - Add ``<span>`` HTML element to each data cells ``display_value``. Ignores
          headers.
        - Add table level CSS styles to control pseudo classes.

        Parameters
        ----------
        styler_data : DataFrame
            Underlying ``Styler`` DataFrame used for reindexing.
        uuid : str
            The underlying ``Styler`` uuid for CSS id.
        d : dict
            The dictionary prior to final render

        Returns
        -------
        render_dict : Dict[str, Any]
        """
        self.tt_data = self.tt_data.reindex_like(styler.data)
        if self.tt_data.empty:
            return d
        mask: DataFrame = self.tt_data.isna() | self.tt_data.eq('')
        if not self.as_title_attribute:
            name: str = self.class_name
            self.table_styles = [
                style
                for sublist in [
                    self._pseudo_css(styler.uuid, name, i, j, str(self.tt_data.iloc[i, j]))
                    for i in range(len(self.tt_data.index))
                    for j in range(len(self.tt_data.columns))
                    if not (mask.iloc[i, j] or i in styler.hidden_rows or j in styler.hidden_columns)
                ]
                for style in sublist
            ]
            if self.table_styles:
                for row in d['body']:
                    for item in row:
                        if item['type'] == 'td':
                            item['display_value'] = str(item['display_value']) + f'<span class="{self.class_name}"></span>'
                d['table_styles'].extend(self._class_styles)
                d['table_styles'].extend(self.table_styles)
        else:
            index_offset: int = self.tt_data.index.nlevels
            body: List[List[Dict[str, Any]]] = d['body']
            for i in range(len(self.tt_data.index)):
                for j in range(len(self.tt_data.columns)):
                    if not mask.iloc[i, j] and i not in styler.hidden_rows and j not in styler.hidden_columns:
                        row: List[Dict[str, Any]] = body[i]
                        item: Dict[str, Any] = row[j + index_offset]
                        value: str = self.tt_data.iloc[i, j]
                        item['attributes'] += f' title="{value}"'
        return d

def _parse_latex_table_wrapping(table_styles: Optional[List[Dict[str, Any]]], caption: Optional[str]) -> bool:
    """
    Indicate whether LaTeX {tabular} should be wrapped with a {table} environment.

    Parses the `table_styles` and detects any selectors which must be included outside
    of {tabular}, i.e. indicating that wrapping must occur, and therefore return True,
    or if a caption exists and requires similar.
    """
    IGNORED_WRAPPERS = ['toprule', 'midrule', 'bottomrule', 'column_format']
    return (
        (table_styles is not None and any((d['selector'] not in IGNORED_WRAPPERS for d in table_styles)))
        or (caption is not None)
    )

def _parse_latex_table_styles(
    table_styles: List[Dict[str, Any]],
    selector: str
) -> Optional[str]:
    """
    Return the first 'props' 'value' from ``tables_styles`` identified by ``selector``.

    Examples
    --------
    >>> table_styles = [
    ...     {"selector": "foo", "props": [("attr", "value")]},
    ...     {"selector": "bar", "props": [("attr", "overwritten")]},
    ...     {"selector": "bar", "props": [("a1", "baz"), ("a2", "ignore")]},
    ... ]
    >>> _parse_latex_table_styles(table_styles, selector="bar")
    'baz'

    Notes
    -----
    The replacement of "§" with ":" is to avoid the CSS problem where ":" has structural
    significance and cannot be used in LaTeX labels, but is often required by them.
    """
    for style in reversed(table_styles):
        if style['selector'] == selector:
            return str(style['props'][0][1]).replace('§', ':')
    return None

def _parse_latex_cell_styles(
    latex_styles: List[Tuple[str, str]],
    display_value: str,
    convert_css: bool = False
) -> str:
    """
    Mutate the ``display_value`` string including LaTeX commands from ``latex_styles``.

    This method builds a recursive latex chain of commands based on the
    CSSList input, nested around ``display_value``.

    If a CSS style is given as ('<command>', '<options>') this is translated to
    '\\<command><options>{display_value}', and this value is treated as the
    display value for the next iteration.

    The most recent style forms the inner component, for example for styles:
    `[('c1', 'o1'), ('c2', 'o2')]` this returns: `\\c1o1{\\c2o2{display_value}}`

    Sometimes latex commands have to be wrapped with curly braces in different ways:
    We create some parsing flags to identify the different behaviours:

     - `--rwrap`        : `\\<command><options>{<display_value>}`
     - `--wrap`         : `{\\<command><options> <display_value>}`
     - `--nowrap`       : `\\<command><options> <display_value>`
     - `--lwrap`        : `{\\<command><options>} <display_value>`
     - `--dwrap`        : `{\\<command><options>}{<display_value>}`

    For example for styles:
    `[('c1', 'o1--wrap'), ('c2', 'o2')]` this returns: `{\\c1o1 \\c2o2{display_value}}
    """
    if convert_css:
        latex_styles = _parse_latex_css_conversion(latex_styles)
    for command, options in reversed(latex_styles):
        if '--wrap' in options:
            display_value = f"{{\\{command}{_parse_latex_options_strip(options, '--wrap')} {display_value}}}"
        elif '--nowrap' in options:
            display_value = f"\\{command}{_parse_latex_options_strip(options, '--nowrap')} {display_value}"
        elif '--lwrap' in options:
            display_value = f"{{\\{command}{_parse_latex_options_strip(options, '--lwrap')}}} {display_value}"
        elif '--rwrap' in options:
            display_value = f"\\{command}{_parse_latex_options_strip(options, '--rwrap')}{{{display_value}}}"
        elif '--dwrap' in options:
            display_value = f"{{\\{command}{_parse_latex_options_strip(options, '--dwrap')}}}{{{display_value}}}"
        else:
            display_value = f"\\{command}{options} {display_value}"
    return display_value

def _parse_latex_options_strip(value: str, arg: str) -> str:
    """
    Strip a css_value which may have latex wrapping arguments, css comment identifiers,
    and whitespaces, to a valid string for latex options parsing.

    For example: 'red /* --wrap */  ' --> 'red'
    """
    return value.replace(arg, '').replace('/*', '').replace('*/', '').strip()

def _parse_latex_css_conversion(styles: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Convert CSS (attribute,value) pairs to equivalent LaTeX (command,options) pairs.

    Ignore conversion if tagged with `--latex` option, skipped if no conversion found.
    """

    def font_weight(value: str, arg: str) -> Optional[Tuple[str, str]]:
        if value in ('bold', 'bolder'):
            return ('bfseries', f"{arg}")
        return None

    def font_style(value: str, arg: str) -> Optional[Tuple[str, str]]:
        if value == 'italic':
            return ('itshape', f"{arg}")
        if value == 'oblique':
            return ('slshape', f"{arg}")
        return None

    def color(
        value: str,
        user_arg: str,
        command: str,
        comm_arg: str
    ) -> Tuple[str, str]:
        """
        CSS colors have 5 formats to process:

         - 6 digit hex code: "#ff23ee"     --> [HTML]{FF23EE}
         - 3 digit hex code: "#f0e"        --> [HTML]{FF00EE}
         - rgba: rgba(128, 255, 0, 0.5)    --> [rgb]{0.502, 1.000, 0.000}
         - rgb: rgb(128, 255, 0,)          --> [rbg]{0.502, 1.000, 0.000}
         - string: red                     --> {red}

        Additionally rgb or rgba can be expressed in % which is also parsed.
        """
        arg = user_arg if user_arg != '' else comm_arg
        if value.startswith('#') and len(value) == 7:
            return (command, f'[HTML]{{{value[1:].upper()}}}{arg}')
        if value.startswith('#') and len(value) == 4:
            val = f"{value[1].upper() * 2}{value[2].upper() * 2}{value[3].upper() * 2}"
            return (command, f'[HTML]{{{val}}}{arg}')
        elif value.startswith('rgb'):
            r_match = re.findall(r'(?<=\()[0-9\s%]+(?=,)', value)
            r = float(r_match[0][:-1]) / 100 if '%' in r_match[0] else int(r_match[0]) / 255
            g_match = re.findall(r'(?<=,)[0-9\s%]+(?=,)', value)
            g = float(g_match[0][:-1]) / 100 if '%' in g_match[0] else int(g_match[0]) / 255
            if value[3] == 'a':
                b_match = re.findall(r'(?<=,)[0-9\s%]+(?=,)', value)[1]
            else:
                b_match = re.findall(r'(?<=,)[0-9\s%]+(?=\))', value)[0]
            b = float(b_match[:-1]) / 100 if '%' in b_match else int(b_match) / 255
            return (command, f'[rgb]{{{r:.3f}, {g:.3f}, {b:.3f}}}{arg}')
        else:
            return (command, f'{{{value}}}{arg}')

    CONVERTED_ATTRIBUTES: Dict[str, Callable[..., Optional[Tuple[str, str]]]] = {
        'font-weight': font_weight,
        'background-color': partial(color, command='cellcolor', comm_arg='--lwrap'),
        'color': partial(color, command='color', comm_arg=''),
        'font-style': font_style
    }
    latex_styles: List[Tuple[str, str]] = []
    for attribute, value in styles:
        if isinstance(value, str) and '--latex' in value:
            latex_styles.append((attribute, value.replace('--latex', '')))
        if attribute in CONVERTED_ATTRIBUTES:
            arg: str = ''
            for x in ['--wrap', '--nowrap', '--lwrap', '--dwrap', '--rwrap']:
                if x in value:
                    arg, value = (x, _parse_latex_options_strip(value, x))
                    break
            latex_style = CONVERTED_ATTRIBUTES[attribute](value, arg)
            if latex_style is not None:
                latex_styles.append(latex_style)
    return latex_styles

def _escape_latex(s: str) -> str:
    """
    Replace the characters ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``,
    ``~``, ``^``, and ``\\`` in the string with LaTeX-safe sequences.

    Use this if you need to display text that might contain such characters in LaTeX.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    return (
        s.replace('\\', 'ab2§=§8yz')
        .replace('ab2§=§8yz ', 'ab2§=§8yz\\space ')
        .replace('&', '\\&')
        .replace('%', '\\%')
        .replace('$', '\\$')
        .replace('#', '\\#')
        .replace('_', '\\_')
        .replace('{', '\\{')
        .replace('}', '\\}')
        .replace('~ ', '~\\space ')
        .replace('~', '\\textasciitilde ')
        .replace('^ ', '^\\space ')
        .replace('^', '\\textasciicircum ')
        .replace('ab2§=§8yz', '\\textbackslash ')
    )

def _math_mode_with_dollar(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``$`` and end with ``$``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace('\\$', 'rt8§=§7wz')
    pattern = re.compile(r'\$.*?\$')
    pos = 0
    ps = pattern.search(s, pos)
    res: List[str] = []
    while ps:
        res.append(_escape_latex(s[pos:ps.span()[0]]))
        res.append(ps.group())
        pos = ps.span()[1]
        ps = pattern.search(s, pos)
    res.append(_escape_latex(s[pos:len(s)]))
    return ''.join(res).replace('rt8§=§7wz', '\\$')

def _math_mode_with_parentheses(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``\\(`` and end with ``\\)``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace('\\(', 'LEFT§=§6yzLEFT').replace('\\)', 'RIGHTab5§=§RIGHT')
    res: List[str] = []
    for item in re.split(r'LEFT§=§6yz|ab5§=§RIGHT', s):
        if item.startswith('LEFT') and item.endswith('RIGHT'):
            res.append(item.replace('LEFT', '\\(').replace('RIGHT', '\\)'))
        elif 'LEFT' in item and 'RIGHT' in item:
            res.append(_escape_latex(item).replace('LEFT', '\\(').replace('RIGHT', '\\)'))
        else:
            res.append(_escape_latex(item).replace('LEFT', '\\textbackslash (').replace('RIGHT', '\\textbackslash )'))
    return ''.join(res)

def _escape_latex_math(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which either are surrounded
    by two characters ``$`` or start with the character ``\\(`` and end with ``\\)``,
    are preserved without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace('\\$', 'rt8§=§7wz')
    ps_d = re.compile(r'\$.*?\$').search(s, 0)
    ps_p = re.compile(r'\\(.*?)\\)').search(s, 0)
    mode: List[int] = []
    if ps_d:
        mode.append(ps_d.span()[0])
    if ps_p:
        mode.append(ps_p.span()[0])
    if len(mode) == 0:
        return _escape_latex(s.replace('rt8§=§7wz', '\\$'))
    if s[mode[0]] == '$':
        return _math_mode_with_dollar(s.replace('rt8§=§7wz', '\\$'))
    if s[mode[0] - 1:mode[0] + 1] == '\\(':
        return _math_mode_with_parentheses(s.replace('rt8§=§7wz', '\\$'))
    else:
        return _escape_latex(s.replace('rt8§=§7wz', '\\$'))

def _parse_latex_header_span(
    cell: Dict[str, Any],
    multirow_align: str,
    multicol_align: str,
    wrap: bool = False,
    convert_css: bool = False
) -> str:
    """
    Refactor the cell `display_value` if a 'colspan' or 'rowspan' attribute is present.

    'rowspan' and 'colspan' do not occur simultaneously. If they are detected then
    the `display_value` is altered to a LaTeX `multirow` or `multicol` command
    respectively, with the appropriate cell-span.

    ``wrap`` is used to enclose the `display_value` in braces which is needed for
    column headers using an siunitx package.

    Requires the package {multirow}, whereas multicol support is usually built in
    to the {tabular} environment.

    Examples
    --------
    >>> cell = {"cellstyle": "", "display_value": "text", "attributes": 'colspan="3"'}
    >>> _parse_latex_header_span(cell, "t", "c")
    '\\multicolumn{3}{c}{text}'
    """
    display_val: str = _parse_latex_cell_styles(cell['cellstyle'], cell['display_value'], convert_css)
    if 'attributes' in cell:
        attrs: str = cell['attributes']
        if 'colspan="' in attrs:
            colspan_str: str = attrs[attrs.find('colspan="') + 9:]
            colspan: int = int(colspan_str[:colspan_str.find('"')])
            if 'naive-l' == multicol_align:
                out: str = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks: str = ' & {}' if wrap else ' &'
                return out + blanks * (colspan - 1)
            elif 'naive-r' == multicol_align:
                out = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks = '{} & ' if wrap else '& '
                return blanks * (colspan - 1) + out
            return f"\\multicolumn{{{colspan}}}{{{multicol_align}}}{{{display_val}}}"
        elif 'rowspan="' in attrs:
            if multirow_align == 'naive':
                return display_val
            rowspan_str: str = attrs[attrs.find('rowspan="') + 9:]
            rowspan: int = int(rowspan_str[:rowspan_str.find('"')])
            return f"\\multirow[{multirow_align}]{{{rowspan}}}{{*}}{{{display_val}}}"
    if wrap:
        return f"{{{display_val}}}"
    else:
        return display_val

def _math_mode_with_dollar(s: str) -> str:
    # Already defined above
    pass  # Placeholder

def _ignore_redundant_calls():
    pass  # Placeholder

def _parse_latex_options_strip(value: str, arg: str) -> str:
    # Already defined above
    pass  # Placeholder

def _parse_latex_cell_styles(
    latex_styles: List[Tuple[str, str]],
    display_value: str,
    convert_css: bool = False
) -> str:
    # Already defined above with return type
    pass  # Placeholder

def _escape_latex_math(s: str) -> str:
    # Already defined above
    pass  # Placeholder

class Tooltips:
    # Already defined above with type annotations
    pass  # Placeholder

class StylerRenderer:
    # Other methods are defined and annotated above
    pass  # Placeholder
