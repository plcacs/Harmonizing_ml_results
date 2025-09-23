from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
import re
from typing import Any, DefaultDict, Optional, TypedDict, Union, List, Tuple
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

jinja2 = import_optional_dependency('jinja2', extra='DataFrame.style requires jinja2.')
from markupsafe import escape as escape_html

BaseFormatter = Union[str, Callable[[Any], str]]
ExtFormatter = Union[BaseFormatter, dict[Any, Optional[BaseFormatter]]]
CSSPair = Tuple[str, Union[str, float]]
CSSList = List[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties

CSSStyles = List[CSSDict]
Subset = Union[slice, Sequence, Index]

class StylerRenderer:
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """
    loader = jinja2.PackageLoader('pandas', 'io/formats/templates')
    env = jinja2.Environment(loader=loader, trim_blocks=True)
    template_html = env.get_template('html.tpl')
    template_html_table = env.get_template('html_table.tpl')
    template_html_style = env.get_template('html_style.tpl')
    template_latex = env.get_template('latex.tpl')
    template_typst = env.get_template('typst.tpl')
    template_string = env.get_template('string.tpl')

    def __init__(
        self,
        data: Union[DataFrame, Series],
        uuid: Optional[str] = None,
        uuid_len: int = 5,
        table_styles: Optional[List[dict[str, Any]]] = None,
        table_attributes: Optional[str] = None,
        caption: Optional[Any] = None,
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
        self.table_styles: Optional[List[dict[str, Any]]] = table_styles
        self.table_attributes: Optional[str] = table_attributes
        self.caption: Optional[Any] = caption
        self.cell_ids: bool = cell_ids
        self.css: dict[str, str] = {
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
        self.hidden_rows: Sequence[int] = []
        self.hidden_columns: Sequence[int] = []
        self.ctx: DefaultDict[Tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_index: DefaultDict[Tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_columns: DefaultDict[Tuple[int, int], CSSList] = defaultdict(list)
        self.cell_context: DefaultDict[Tuple[int, int], str] = defaultdict(str)
        self._todo: List[Tuple[Callable, Tuple[Any, ...], dict[str, Any]]] = []
        self.tooltips: Optional[Tooltips] = None
        precision = get_option('styler.format.precision') if precision is None else precision
        self._display_funcs: DefaultDict[Tuple[int, int], Callable[[Any], str]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_index: DefaultDict[Tuple[int, int], Callable[[Any], str]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_index_names: DefaultDict[int, Callable[[Any], str]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_columns: DefaultDict[Tuple[int, int], Callable[[Any], str]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )
        self._display_funcs_column_names: DefaultDict[int, Callable[[Any], str]] = defaultdict(
            lambda: partial(_default_formatter, precision=precision)
        )

    def _render(self, sparse_index: bool, sparse_columns: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, blank: str = '') -> dict[str, Any]:
        """
        Computes and applies styles and then generates the general render dicts.
        Also extends the `ctx` and `ctx_index` attributes with those of concatenated
        stylers for use within `_translate_latex`
        """
        self._compute()
        dxs: List[dict[str, Any]] = []
        ctx_len: int = len(self.index)
        for (i, concatenated) in enumerate(self.concatenated):
            concatenated.hide_index_ = self.hide_index_
            concatenated.hidden_columns = self.hidden_columns
            foot = f"{self.css['foot']}{i}"
            concatenated.css = {**self.css, 'data': f'{foot}_data', 'row_heading': f'{foot}_row_heading', 'row': f'{foot}_row', 'foot': f'{foot}_foot'}
            dx = concatenated._render(sparse_index, sparse_columns, max_rows, max_cols, blank)
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
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols, '&nbsp;')
        d.update(kwargs)
        return self.template_html.render(**d, html_table_tpl=self.template_html_table, html_style_tpl=self.template_html_style)

    def _render_latex(self, sparse_index: bool, sparse_columns: bool, clines: Optional[str], **kwargs: Any) -> str:
        """
        Render a Styler in latex format
        """
        d = self._render(sparse_index, sparse_columns, None, None)
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
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_typst.render(**d)

    def _render_string(self, sparse_index: bool, sparse_columns: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, **kwargs: Any) -> str:
        """
        Render a Styler in string format
        """
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
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

    def _translate(self, sparse_index: bool, sparse_cols: bool, max_rows: Optional[int] = None, max_cols: Optional[int] = None, blank: str = '&nbsp;', dxs: Optional[List[dict[str, Any]]] = None) -> dict[str, Any]:
        """
        Process Styler data and settings into a dict for template rendering.
        """
        if dxs is None:
            dxs = []
        self.css['blank_value'] = blank
        d: dict[str, Any] = {
            'uuid': self.uuid,
            'table_styles': format_table_styles(self.table_styles or []),
            'caption': self.caption
        }
        max_elements = get_option('styler.render.max_elements')
        max_rows = max_rows if max_rows else get_option('styler.render.max_rows')
        max_cols = max_cols if max_cols else get_option('styler.render.max_columns')
        (max_rows, max_cols) = _get_trimming_maximums(len(self.data.index), len(self.data.columns), max_elements, max_rows, max_cols)
        self.cellstyle_map_columns: DefaultDict[Tuple[CSSPair, ...], List[str]] = defaultdict(list)
        head = self._translate_header(sparse_cols, max_cols)
        d.update({'head': head})
        idx_lengths = _get_level_lengths(self.index, sparse_index, max_rows, self.hidden_rows)
        d.update({'index_lengths': idx_lengths})
        self.cellstyle_map: DefaultDict[Tuple[CSSPair, ...], List[str]] = defaultdict(list)
        self.cellstyle_map_index: DefaultDict[Tuple[CSSPair, ...], List[str]] = defaultdict(list)
        body: List[Any] = self._translate_body(idx_lengths, max_rows, max_cols)
        d.update({'body': body})
        ctx_maps = {'cellstyle': 'cellstyle_map', 'cellstyle_index': 'cellstyle_map_index', 'cellstyle_columns': 'cellstyle_map_columns'}
        for (k, attr) in ctx_maps.items():
            map_list = [{'props': list(props), 'selectors': selectors} for (props, selectors) in getattr(self, attr).items()]
            d.update({k: map_list})
        for dx in dxs:
            d['body'].extend(dx['body'])
            d['cellstyle'].extend(dx['cellstyle'])
            d['cellstyle_index'].extend(dx['cellstyle_index'])
        table_attr = self.table_attributes
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

    def _translate_header(self, sparsify_cols: bool, max_cols: int) -> List[Any]:
        """
        Build each <tr> within table <head> as a list.
        """
        col_lengths = _get_level_lengths(self.columns, sparsify_cols, max_cols, self.hidden_columns)
        clabels = self.data.columns.tolist()
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))
        head: List[Any] = []
        for (r, hide) in enumerate(self.hide_columns_):
            if hide or not clabels:
                continue
            header_row = self._generate_col_header_row((r, clabels), max_cols, col_lengths)
            head.append(header_row)
        if self.data.index.names and com.any_not_none(*self.data.index.names) and (not all(self.hide_index_)) and (not self.hide_index_names):
            index_names_row = self._generate_index_names_row(clabels, max_cols, col_lengths)
            head.append(index_names_row)
        return head

    def _generate_col_header_row(self, iter: Tuple[int, Any], max_cols: int, col_lengths: dict) -> List[Any]:
        """
        Generate the row containing column headers.
        """
        (r, clabels) = iter
        index_blanks = [_element('th', self.css['blank'], self.css['blank_value'], True)] * (self.index.nlevels - sum(self.hide_index_) - 1)
        name = self.data.columns.names[r]
        is_display = name is not None and (not self.hide_column_names)
        value = name if is_display else self.css['blank_value']
        display_value: Optional[str] = self._display_funcs_column_names[r](value) if is_display else None
        column_name = [_element('th', f"{self.css['blank']} {self.css['level']}{r}" if name is None else f"{self.css['index_name']} {self.css['level']}{r}", value, not all(self.hide_index_), display_value=display_value)]
        column_headers: List[Any] = []
        visible_col_count: int = 0
        for (c, value) in enumerate(clabels[r]):
            header_element_visible = _is_visible(c, r, col_lengths)
            if header_element_visible:
                visible_col_count += col_lengths.get((r, c), 0)
            if self._check_trim(visible_col_count, max_cols, column_headers, 'th', f"{self.css['col_heading']} {self.css['level']}{r} {self.css['col_trim']}"):
                break
            header_element = _element('th', f"{self.css['col_heading']} {self.css['level']}{r} {self.css['col']}{c}", value, header_element_visible, display_value=self._display_funcs_columns[r, c](value), attributes=(f'colspan="{col_lengths.get((r, c), 0)}"' if col_lengths.get((r, c), 0) > 1 else ''))
            if self.cell_ids:
                header_element['id'] = f"{self.css['level']}{r}_{self.css['col']}{c}"
            if header_element_visible and (r, c) in self.ctx_columns and self.ctx_columns[r, c]:
                header_element['id'] = f"{self.css['level']}{r}_{self.css['col']}{c}"
                self.cellstyle_map_columns[tuple(self.ctx_columns[r, c])].append(f"{self.css['level']}{r}_{self.css['col']}{c}")
            column_headers.append(header_element)
        return index_blanks + column_name + column_headers

    def _generate_index_names_row(self, iter: Any, max_cols: int, col_lengths: dict) -> List[Any]:
        """
        Generate the row containing index names.
        """
        clabels = iter
        index_names = [
            _element('th', f"{self.css['index_name']} {self.css['level']}{c}", self.css['blank_value'] if name is None else name, not self.hide_index_[c], display_value=(None if name is None else self._display_funcs_index_names[c](name)))
            for (c, name) in enumerate(self.data.index.names)
        ]
        column_blanks: List[Any] = []
        visible_col_count: int = 0
        if clabels:
            last_level = self.columns.nlevels - 1
            for (c, value) in enumerate(clabels[last_level]):
                header_element_visible = _is_visible(c, last_level, col_lengths)
                if header_element_visible:
                    visible_col_count += 1
                if self._check_trim(visible_col_count, max_cols, column_blanks, 'th', f"{self.css['blank']} {self.css['col']}{c} {self.css['col_trim']}", self.css['blank_value']):
                    break
                column_blanks.append(_element('th', f"{self.css['blank']} {self.css['col']}{c}", self.css['blank_value'], (c not in self.hidden_columns)))
        return index_names + column_blanks

    def _translate_body(self, idx_lengths: dict, max_rows: int, max_cols: int) -> List[Any]:
        """
        Build each <tr> within table <body> as a list.
        """
        rlabels = self.data.index.tolist()
        if not isinstance(self.data.index, MultiIndex):
            rlabels = [[x] for x in rlabels]
        body: List[Any] = []
        visible_row_count: int = 0
        for (r, row_tup) in [z for z in enumerate(self.data.itertuples()) if z[0] not in self.hidden_rows]:
            visible_row_count += 1
            if self._check_trim(visible_row_count, max_rows, body, 'row'):
                break
            body_row = self._generate_body_row((r, row_tup, rlabels), max_cols, idx_lengths)
            body.append(body_row)
        return body

    def _check_trim(self, count: int, max: int, obj: List[Any], element: str, css: Optional[str] = None, value: str = '...') -> bool:
        """
        Indicates whether to break render loops and append a trimming indicator.
        """
        if count > max:
            if element == 'row':
                obj.append(self._generate_trimmed_row(max))
            else:
                obj.append(_element(element, css if css is not None else '', value, True, attributes=''))
            return True
        return False

    def _generate_trimmed_row(self, max_cols: int) -> List[Any]:
        """
        When a render has too many rows we generate a trimming row containing "..."
        """
        index_headers = [_element('th', f"{self.css['row_heading']} {self.css['level']}{c} {self.css['row_trim']}", '...', not self.hide_index_[c], attributes='') for c in range(self.data.index.nlevels)]
        data: List[Any] = []
        visible_col_count: int = 0
        for (c, _) in enumerate(self.columns):
            data_element_visible = c not in self.hidden_columns
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(visible_col_count, max_cols, data, 'td', f"{self.css['data']} {self.css['row_trim']} {self.css['col_trim']}"):
                break
            data.append(_element('td', f"{self.css['data']} {self.css['col']}{c} {self.css['row_trim']}", '...', data_element_visible, attributes=''))
        return index_headers + data

    def _generate_body_row(self, iter: Tuple[int, Any, Any], max_cols: int, idx_lengths: dict) -> List[Any]:
        """
        Generate a regular row for the body section.
        """
        (r, row_tup, rlabels) = iter
        index_headers = []
        for (c, value) in enumerate(rlabels[r]):
            header_element_visible = _is_visible(r, c, idx_lengths) and (not self.hide_index_[c])
            header_element = _element('th', f"{self.css['row_heading']} {self.css['level']}{c} {self.css['row']}{r}", value, header_element_visible, display_value=self._display_funcs_index[r, c](value), attributes=(f'rowspan="{idx_lengths.get((c, r), 0)}"' if idx_lengths.get((c, r), 0) > 1 else ''))
            if self.cell_ids:
                header_element['id'] = f"{self.css['level']}{c}_{self.css['row']}{r}"
            if header_element_visible and (r, c) in self.ctx_index and self.ctx_index[r, c]:
                header_element['id'] = f"{self.css['level']}{c}_{self.css['row']}{r}"
                self.cellstyle_map_index[tuple(self.ctx_index[r, c])].append(f"{self.css['level']}{c}_{self.css['row']}{r}")
            index_headers.append(header_element)
        data: List[Any] = []
        visible_col_count: int = 0
        for (c, value) in enumerate(row_tup[1:]):
            data_element_visible = c not in self.hidden_columns and r not in self.hidden_rows
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(visible_col_count, max_cols, data, 'td', f"{self.css['data']} {self.css['row']}{r} {self.css['col_trim']}"):
                break
            cls: str = ''
            if (r, c) in self.cell_context:
                cls = ' ' + self.cell_context[r, c]
            data_element = _element('td', f"{self.css['data']} {self.css['row']}{r} {self.css['col']}{c}{cls}", value, data_element_visible, attributes='', display_value=self._display_funcs[r, c](value))
            if self.cell_ids:
                data_element['id'] = f"{self.css['row']}{r}_{self.css['col']}{c}"
            if data_element_visible and (r, c) in self.ctx and self.ctx[r, c]:
                data_element['id'] = f"{self.css['row']}{r}_{self.css['col']}{c}"
                self.cellstyle_map[tuple(self.ctx[r, c])].append(f"{self.css['row']}{r}_{self.css['col']}{c}")
            data.append(data_element)
        return index_headers + data

    def _translate_latex(self, d: dict[str, Any], clines: Optional[str]) -> None:
        """
        Post-process the default render dict for the LaTeX template format.
        """
        index_levels = self.index.nlevels
        visible_index_level_n = max(1, index_levels - sum(self.hide_index_))
        d['head'] = [[{**col, 'cellstyle': self.ctx_columns[r, c - visible_index_level_n]} for (c, col) in enumerate(row) if col['is_visible']] for (r, row) in enumerate(d['head'])]

        def _concatenated_visible_rows(obj: StylerRenderer, n: int, row_indices: List[int]) -> int:
            row_indices.extend([r + n for r in range(len(obj.index)) if r not in obj.hidden_rows])
            n += len(obj.index)
            for concatenated in obj.concatenated:
                n = _concatenated_visible_rows(concatenated, n, row_indices)
            return n

        def concatenated_visible_rows(obj: StylerRenderer) -> List[int]:
            row_indices: List[int] = []
            _concatenated_visible_rows(obj, 0, row_indices)
            return row_indices
        body: List[Any] = []
        for (r, row) in zip(concatenated_visible_rows(self), d['body']):
            if all(self.hide_index_):
                row_body_headers = []
            else:
                row_body_headers = [{**col, 'display_value': col['display_value'] if col['is_visible'] else '', 'cellstyle': self.ctx_index[r, c]} for (c, col) in enumerate(row[:index_levels]) if col['type'] == 'th' and (not self.hide_index_[c])]
            row_body_cells = [{**col, 'cellstyle': self.ctx[r, c]} for (c, col) in enumerate(row[index_levels:]) if col['is_visible'] and col['type'] == 'td']
            body.append(row_body_headers + row_body_cells)
        d['body'] = body
        if clines not in [None, 'all;data', 'all;index', 'skip-last;data', 'skip-last;index']:
            raise ValueError(f"`clines` value of {clines} is invalid. Should either be None or one of 'all;data', 'all;index', 'skip-last;data', 'skip-last;index'.")
        if clines is not None:
            data_len = len(row_body_cells) if 'data' in clines and d['body'] else 0
            d['clines'] = defaultdict(list)
            visible_row_indexes: List[int] = [r for r in range(len(self.data.index)) if r not in self.hidden_rows]
            visible_index_levels: List[int] = [i for i in range(index_levels) if not self.hide_index_[i]]
            for (rn, r) in enumerate(visible_row_indexes):
                for (lvln, lvl) in enumerate(visible_index_levels):
                    if lvl == index_levels - 1 and 'skip-last' in clines:
                        continue
                    idx_len = d['index_lengths'].get((lvl, r), None)
                    if idx_len is not None:
                        d['clines'][rn + idx_len].append(f'\\cline{{{lvln + 1}-{len(visible_index_levels) + data_len}}}')

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
        """
        if all((formatter is None, subset is None, precision is None, decimal == '.', thousands is None, na_rep is None, escape is None, hyperlinks is None)):
            self._display_funcs.clear()
            return self
        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]
        if not isinstance(formatter, dict):
            formatter = {col: formatter for col in data.columns}
        cis = self.columns.get_indexer_for(data.columns)
        ris = self.index.get_indexer_for(data.index)
        for ci in cis:
            format_func = _maybe_wrap_formatter(formatter.get(self.columns[ci]), na_rep=na_rep, precision=precision, decimal=decimal, thousands=thousands, escape=escape, hyperlinks=hyperlinks)
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
        """
        axis = self.data._get_axis_number(axis)
        if axis == 0:
            (display_funcs_, obj) = (self._display_funcs_index, self.index)
        else:
            (display_funcs_, obj) = (self._display_funcs_columns, self.columns)
        levels_ = refactor_levels(level, obj)
        if all((formatter is None, level is None, precision is None, decimal == '.', thousands is None, na_rep is None, escape is None, hyperlinks is None)):
            display_funcs_.clear()
            return self
        if not isinstance(formatter, dict):
            formatter = {level: formatter for level in levels_}
        else:
            formatter = {obj._get_level_number(level): formatter_ for (level, formatter_) in formatter.items()}
        for lvl in levels_:
            format_func = _maybe_wrap_formatter(formatter.get(lvl), na_rep=na_rep, precision=precision, decimal=decimal, thousands=thousands, escape=escape, hyperlinks=hyperlinks)
            for idx in [(i, lvl) if axis == 0 else (lvl, i) for i in range(len(obj))]:
                display_funcs_[idx] = format_func
        return self

    def relabel_index(self, labels: Union[Sequence, Index], axis: Union[int, str] = 0, level: Optional[Union[int, str, List[Union[int, str]]]] = None) -> StylerRenderer:
        """
        Relabel the index, or column header, keys to display a set of specified values.
        """
        axis = self.data._get_axis_number(axis)
        if axis == 0:
            (display_funcs_, obj) = (self._display_funcs_index, self.index)
            (hidden_labels, hidden_lvls) = (self.hidden_rows, self.hide_index_)
        else:
            (display_funcs_, obj) = (self._display_funcs_columns, self.columns)
            (hidden_labels, hidden_lvls) = (self.hidden_columns, self.hide_columns_)
        visible_len = len(obj) - len(set(hidden_labels))
        if len(labels) != visible_len:
            raise ValueError(f'``labels`` must be of length equal to the number of visible labels along ``axis`` ({visible_len}).')
        if level is None:
            level = [i for i in range(obj.nlevels) if not hidden_lvls[i]]
        levels_ = refactor_levels(level, obj)

        def alias_(x: Any, value: Any) -> Any:
            if isinstance(value, str):
                return value.format(x)
            return value
        for (ai, i) in enumerate([i for i in range(len(obj)) if i not in hidden_labels]):
            if len(levels_) == 1:
                idx = (i, levels_[0]) if axis == 0 else (levels_[0], i)
                display_funcs_[idx] = partial(alias_, value=labels[ai])
            else:
                for (aj, lvl) in enumerate(levels_):
                    idx = (i, lvl) if axis == 0 else (lvl, i)
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
        """
        axis = self.data._get_axis_number(axis)
        if axis == 0:
            (display_funcs_, obj) = (self._display_funcs_index_names, self.index)
        else:
            (display_funcs_, obj) = (self._display_funcs_column_names, self.columns)
        levels_ = refactor_levels(level, obj)
        if all((formatter is None, level is None, precision is None, decimal == '.', thousands is None, na_rep is None, escape is None, hyperlinks is None)):
            display_funcs_.clear()
            return self
        if not isinstance(formatter, dict):
            formatter = {level: formatter for level in levels_}
        else:
            formatter = {obj._get_level_number(level): formatter_ for (level, formatter_) in formatter.items()}
        for lvl in levels_:
            format_func = _maybe_wrap_formatter(formatter.get(lvl), na_rep=na_rep, precision=precision, decimal=decimal, thousands=thousands, escape=escape, hyperlinks=hyperlinks)
            display_funcs_[lvl] = format_func
        return self

def _element(html_element: str, html_class: str, value: Any, is_visible: bool, **kwargs: Any) -> dict[str, Any]:
    """
    Template to return container with information for a <td></td> or <th></th> element.
    """
    if 'display_value' not in kwargs or kwargs['display_value'] is None:
        kwargs['display_value'] = value
    return {'type': html_element, 'value': value, 'class': html_class, 'is_visible': is_visible, **kwargs}

def _get_trimming_maximums(rn: int, cn: int, max_elements: int, max_rows: Optional[int] = None, max_cols: Optional[int] = None, scaling_factor: float = 0.8) -> Tuple[int, int]:
    """
    Recursively reduce the number of rows and columns to satisfy max elements.
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
        (rn, cn) = scale_down(rn, cn)
    return (rn, cn)

def _get_level_lengths(index: Any, sparsify: bool, max_index: int, hidden_elements: Optional[Sequence[int]] = None) -> dict[Tuple[int, int], int]:
    """
    Given an index, find the level length for each element.
    """
    if isinstance(index, MultiIndex):
        levels = index._format_multi(sparsify=lib.no_default, include_names=False)
    else:
        levels = index._format_flat(include_name=False)
    if hidden_elements is None:
        hidden_elements = []
    lengths: dict[Tuple[int, int], int] = {}
    if not isinstance(index, MultiIndex):
        for (i, value) in enumerate(levels):
            if i not in hidden_elements:
                lengths[0, i] = 1
        return lengths
    for (i, lvl) in enumerate(levels):
        visible_row_count = 0
        for (j, row) in enumerate(lvl):
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
                if lengths[i, last_label] == 0:
                    last_label = j
                    lengths[i, last_label] = 1
                else:
                    lengths[i, last_label] += 1
    non_zero_lengths = {element: length for (element, length) in lengths.items() if length >= 1}
    return non_zero_lengths

def _is_visible(idx_row: int, idx_col: int, lengths: dict[Tuple[int, int], int]) -> bool:
    """
    Determine if an element is visible based on lengths.
    """
    return (idx_col, idx_row) in lengths

def format_table_styles(styles: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """
    Looks for multiple CSS selectors and separates them.
    """
    return [{'selector': selector, 'props': css_dict['props']} for css_dict in styles for selector in css_dict['selector'].split(',')]

def _default_formatter(x: Any, precision: int, thousands: bool = False) -> Any:
    """
    Format the display of a value.
    """
    if is_float(x) or is_complex(x):
        return f'{x:,.{precision}f}' if thousands else f'{x:.{precision}f}'
    elif is_integer(x):
        return f'{x:,}' if thousands else str(x)
    return x

def _wrap_decimal_thousands(formatter: Callable[[Any], str], decimal: str, thousands: str) -> Callable[[Any], str]:
    """
    Wrap a formatting function to deal with non-standard decimal and thousands.
    """
    def wrapper(x: Any) -> str:
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
    """
    Escape string characters if needed.
    """
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
    """
    Convert URLs in a string to hyperlinks.
    """
    if isinstance(x, str):
        if format == 'html':
            href = '<a href="{0}" target="_blank">{0}</a>'
        elif format == 'latex':
            href = '\\href{{{0}}}{{{0}}}'
        else:
            raise ValueError("``hyperlinks`` format can only be 'html' or 'latex'")
        pat = "((http|ftp)s?:\\/\\/|www.)[\\w/\\-?=%.:@]+\\.[\\w/\\-&?=%.,':;~!@#$*()\\[\\]]+"
        return re.sub(pat, lambda m: href.format(m.group(0)), x)
    return x

def _maybe_wrap_formatter(
    formatter: Optional[Union[str, Callable[[Any], str]]] = None,
    na_rep: Optional[str] = None,
    precision: Optional[int] = None,
    decimal: str = '.',
    thousands: Optional[str] = None,
    escape: Optional[str] = None,
    hyperlinks: Optional[str] = None
) -> Callable[[Any], Any]:
    """
    Wrap formatter to support na_rep, escape, and hyperlinks.
    """
    if isinstance(formatter, str):
        func_0: Callable[[Any], str] = lambda x: formatter.format(x)
    elif callable(formatter):
        func_0 = formatter
    elif formatter is None:
        precision = get_option('styler.format.precision') if precision is None else precision
        func_0 = partial(_default_formatter, precision=precision, thousands=thousands is not None)
    else:
        raise TypeError(f"'formatter' expected str or callable, got {type(formatter)}")
    if escape is not None:
        func_1 = lambda x: func_0(_str_escape(x, escape=escape))
    else:
        func_1 = func_0
    if decimal != '.' or (thousands is not None and thousands != ','):
        func_2 = _wrap_decimal_thousands(func_1, decimal=decimal, thousands=thousands)
    else:
        func_2 = func_1
    if hyperlinks is not None:
        func_3 = lambda x: func_2(_render_href(x, format=hyperlinks))
    else:
        func_3 = func_2
    if na_rep is None:
        return func_3
    else:
        return lambda x: na_rep if isna(x) is True else func_3(x)

def non_reducing_slice(slice_: Any) -> Tuple[Any, ...]:
    """
    Ensure that a slice doesn't reduce to a Series or Scalar.
    """
    kinds = (ABCSeries, np.ndarray, Index, list, str)
    if isinstance(slice_, kinds):
        slice_ = IndexSlice[:, slice_]

    def pred(part: Any) -> bool:
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
    Convert CSS-string to sequence of tuples format if needed.
    """
    if isinstance(style, str):
        if style and ':' not in style:
            raise ValueError(f"Styles supplied as string must follow CSS rule formats, for example 'attr: val;'. '{style}' was given.")
        s = style.split(';')
        return [(x.split(':')[0].strip(), ':'.join(x.split(':')[1:]).strip()) for x in s if x.strip() != '']
    return style

def refactor_levels(level: Optional[Union[int, str, List[Union[int, str]]]], obj: Any) -> List[int]:
    """
    Returns a consistent levels arg.
    """
    if level is None:
        levels_: List[int] = list(range(obj.nlevels))
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
    """
    def __init__(
        self,
        css_props: CSSProperties = [('visibility', 'hidden'), ('position', 'absolute'), ('z-index', 1), ('background-color', 'black'), ('color', 'white'), ('transform', 'translate(-20px, -20px)')],
        css_name: str = 'pd-t',
        tooltips: DataFrame = DataFrame(),
        as_title_attribute: bool = False
    ) -> None:
        self.class_name: str = css_name
        self.class_properties: CSSProperties = css_props
        self.tt_data: DataFrame = tooltips
        self.table_styles: CSSStyles = []
        self.as_title_attribute: bool = as_title_attribute

    @property
    def _class_styles(self) -> List[CSSDict]:
        """
        Combine the tooltip CSS class name and properties.
        """
        return [{'selector': f'.{self.class_name}', 'props': maybe_convert_css_to_tuples(self.class_properties)}]

    def _pseudo_css(self, uuid: str, name: str, row: int, col: int, text: str) -> List[CSSDict]:
        """
        Create pseudo CSS entries for tooltips.
        """
        selector_id = '#T_' + uuid + '_row' + str(row) + '_col' + str(col)
        return [
            {'selector': selector_id + f':hover .{name}', 'props': [('visibility', 'visible')]},
            {'selector': selector_id + f' .{name}::after', 'props': [('content', f'"{text}"')]}
        ]

    def _translate(self, styler: StylerRenderer, d: dict[str, Any]) -> dict[str, Any]:
        """
        Mutate the render dictionary to allow for tooltips.
        """
        self.tt_data = self.tt_data.reindex_like(styler.data)
        if self.tt_data.empty:
            return d
        mask = self.tt_data.isna() | self.tt_data.eq('')
        if not self.as_title_attribute:
            name = self.class_name
            self.table_styles = [
                style for sublist in [
                    self._pseudo_css(styler.uuid, name, i, j, str(self.tt_data.iloc[i, j]))
                    for i in range(len(self.tt_data.index))
                    for j in range(len(self.tt_data.columns))
                    if not (mask.iloc[i, j] or i in styler.hidden_rows or j in styler.hidden_columns)
                ] for style in sublist
            ]
            if self.table_styles:
                for row in d['body']:
                    for item in row:
                        if item['type'] == 'td':
                            item['display_value'] = str(item['display_value']) + f'<span class="{self.class_name}"></span>'
                d['table_styles'].extend(self._class_styles)
                d['table_styles'].extend(self.table_styles)
        else:
            index_offset = self.tt_data.index.nlevels
            body = d['body']
            for i in range(len(self.tt_data.index)):
                for j in range(len(self.tt_data.columns)):
                    if not mask.iloc[i, j] or i in styler.hidden_rows or j in styler.hidden_columns:
                        row = body[i]
                        item = row[j + index_offset]
                        value = self.tt_data.iloc[i, j]
                        item['attributes'] += f' title="{value}"'
        return d

def _parse_latex_table_wrapping(table_styles: Optional[List[dict[str, Any]]], caption: Optional[Any]) -> bool:
    """
    Indicate whether LaTeX {tabular} should be wrapped.
    """
    IGNORED_WRAPPERS = ['toprule', 'midrule', 'bottomrule', 'column_format']
    return table_styles is not None and any((d['selector'] not in IGNORED_WRAPPERS for d in table_styles)) or caption is not None

def _parse_latex_table_styles(table_styles: List[dict[str, Any]], selector: str) -> Optional[str]:
    """
    Return the first 'props' value from table_styles by selector.
    """
    for style in table_styles[::-1]:
        if style['selector'] == selector:
            return str(style['props'][0][1]).replace('§', ':')
    return None

def _parse_latex_cell_styles(latex_styles: CSSList, display_value: str, convert_css: bool = False) -> str:
    """
    Mutate the display_value string including LaTeX commands from latex_styles.
    """
    if convert_css:
        latex_styles = _parse_latex_css_conversion(latex_styles)
    for (command, options) in latex_styles[::-1]:
        formatter = {
            '--wrap': f'{{\\{command}--to_parse {display_value}}}',
            '--nowrap': f'\\{command}--to_parse {display_value}',
            '--lwrap': f'{{\\{command}--to_parse}} {display_value}',
            '--rwrap': f'\\{command}--to_parse{{{display_value}}}',
            '--dwrap': f'{{\\{command}--to_parse}}{{{display_value}}}'
        }
        display_value = f'\\{command}{options} {display_value}'
        for arg in ['--nowrap', '--wrap', '--lwrap', '--rwrap', '--dwrap']:
            if arg in str(options):
                display_value = formatter[arg].replace('--to_parse', _parse_latex_options_strip(value=options, arg=arg))
                break
    return display_value

def _parse_latex_header_span(cell: dict[str, Any], multirow_align: str, multicol_align: str, wrap: bool = False, convert_css: bool = False) -> str:
    """
    Refactor the cell display_value if a colspan or rowspan attribute is present.
    """
    display_val = _parse_latex_cell_styles(cell['cellstyle'], cell['display_value'], convert_css)
    if 'attributes' in cell:
        attrs = cell['attributes']
        if 'colspan="' in attrs:
            colspan_part = attrs[attrs.find('colspan="') + 9:]
            colspan = int(colspan_part[:colspan_part.find('"')])
            if 'naive-l' == multicol_align:
                out = f'{{{display_val}}}' if wrap else f'{display_val}'
                blanks = ' & {}' if wrap else ' &'
                return out + blanks * (colspan - 1)
            elif 'naive-r' == multicol_align:
                out = f'{{{display_val}}}' if wrap else f'{display_val}'
                blanks = '{} & ' if wrap else '& '
                return blanks * (colspan - 1) + out
            return f'\\multicolumn{{{colspan}}}{{{multicol_align}}}{{{display_val}}}'
        elif 'rowspan="' in attrs:
            if multirow_align == 'naive':
                return display_val
            rowspan_part = attrs[attrs.find('rowspan="') + 9:]
            rowspan = int(rowspan_part[:rowspan_part.find('"')])
            return f'\\multirow[{multirow_align}]{{{rowspan}}}{{*}}{{{display_val}}}'
    if wrap:
        return f'{{{display_val}}}'
    else:
        return display_val

def _parse_latex_options_strip(value: Any, arg: str) -> str:
    """
    Strip a css_value for LaTeX options parsing.
    """
    return str(value).replace(arg, '').replace('/*', '').replace('*/', '').strip()

def _parse_latex_css_conversion(styles: Union[str, CSSList]) -> CSSList:
    """
    Convert CSS (attribute,value) pairs to equivalent LaTeX (command,options) pairs.
    """
    def font_weight(value: str, arg: str) -> Optional[Tuple[str, str]]:
        if value in ('bold', 'bolder'):
            return ('bfseries', f'{arg}')
        return None

    def font_style(value: str, arg: str) -> Optional[Tuple[str, str]]:
        if value == 'italic':
            return ('itshape', f'{arg}')
        if value == 'oblique':
            return ('slshape', f'{arg}')
        return None

    def color(value: str, user_arg: str, command: str, comm_arg: str) -> Tuple[str, str]:
        arg_val = user_arg if user_arg != '' else comm_arg
        if value[0] == '#' and len(value) == 7:
            return (command, f'[HTML]{{{value[1:].upper()}}}{arg_val}')
        if value[0] == '#' and len(value) == 4:
            val = f'{value[1].upper() * 2}{value[2].upper() * 2}{value[3].upper() * 2}'
            return (command, f'[HTML]{{{val}}}{arg_val}')
        elif value[:3] == 'rgb':
            r = re.findall('(?<=\\()[0-9\\s%]+(?=,)', value)[0].strip()
            r_val = float(r[:-1]) / 100 if '%' in r else int(r) / 255
            g = re.findall('(?<=,)[0-9\\s%]+(?=,)', value)[0].strip()
            g_val = float(g[:-1]) / 100 if '%' in g else int(g) / 255
            if value[3] == 'a':
                b = re.findall('(?<=,)[0-9\\s%]+(?=,)', value)[1].strip()
            else:
                b = re.findall('(?<=,)[0-9\\s%]+(?=\\))', value)[0].strip()
            b_val = float(b[:-1]) / 100 if '%' in b else int(b) / 255
            return (command, f'[rgb]{{{r_val:.3f}, {g_val:.3f}, {b_val:.3f}}}{arg_val}')
        else:
            return (command, f'{{{value}}}{arg_val}')
    CONVERTED_ATTRIBUTES: dict[str, Callable[[str, str], Optional[Tuple[str, str]]]] = {
        'font-weight': font_weight,
        'background-color': lambda v, arg: color(v, arg, command='cellcolor', comm_arg='--lwrap'),
        'color': lambda v, arg: color(v, arg, command='color', comm_arg=''),
        'font-style': font_style
    }
    latex_styles: CSSList = []
    if isinstance(styles, str):
        styles = maybe_convert_css_to_tuples(styles)
    for (attribute, value) in styles:
        if isinstance(value, str) and '--latex' in value:
            latex_styles.append((attribute, value.replace('--latex', '')))
        if attribute in CONVERTED_ATTRIBUTES:
            arg_str = ''
            for x in ['--wrap', '--nowrap', '--lwrap', '--dwrap', '--rwrap']:
                if x in str(value):
                    (arg_str, value) = (x, _parse_latex_options_strip(value, x))
                    break
            latex_style = CONVERTED_ATTRIBUTES[attribute](value, arg_str)
            if latex_style is not None:
                latex_styles.extend([latex_style])
    return latex_styles

def _escape_latex(s: str) -> str:
    """
    Replace characters with LaTeX-safe sequences.
    """
    return s.replace('\\', 'ab2§=§8yz').replace('ab2§=§8yz ', 'ab2§=§8yz\\space ').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('~ ', '~\\space ').replace('~', '\\textasciitilde ').replace('^ ', '^\\space ').replace('^', '\\textasciicircum ').replace('ab2§=§8yz', '\\textbackslash ')

def _math_mode_with_dollar(s: str) -> str:
    """
    Preserve LaTeX math mode characters between $ symbols.
    """
    s = s.replace('\\$', 'rt8§=§7wz')
    pattern = re.compile('\\$.*?\\$')
    pos = 0
    res: List[str] = []
    ps = pattern.search(s, pos)
    while ps:
        res.append(_escape_latex(s[pos:ps.span()[0]]))
        res.append(ps.group())
        pos = ps.span()[1]
        ps = pattern.search(s, pos)
    res.append(_escape_latex(s[pos:len(s)]))
    return ''.join(res).replace('rt8§=§7wz', '\\$')

def _math_mode_with_parentheses(s: str) -> str:
    """
    Preserve LaTeX math mode characters between \\( and \\).
    """
    s = s.replace('\\(', 'LEFT§=§6yzLEFT').replace('\\)', 'RIGHTab5§=§RIGHT')
    res: List[str] = []
    for item in re.split('LEFT§=§6yz|ab5§=§RIGHT', s):
        if item.startswith('LEFT') and item.endswith('RIGHT'):
            res.append(item.replace('LEFT', '\\(').replace('RIGHT', '\\)'))
        elif 'LEFT' in item and 'RIGHT' in item:
            res.append(_escape_latex(item).replace('LEFT', '\\(').replace('RIGHT', '\\)'))
        else:
            res.append(_escape_latex(item).replace('LEFT', '\\textbackslash (').replace('RIGHT', '\\textbackslash )'))
    return ''.join(res)

def _escape_latex_math(s: str) -> str:
    """
    Escape LaTeX outside of math mode.
    """
    s = s.replace('\\$', 'rt8§=§7wz')
    ps_d = re.compile('\\$.*?\\$').search(s, 0)
    ps_p = re.compile('\\(.*?\\)').search(s, 0)
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