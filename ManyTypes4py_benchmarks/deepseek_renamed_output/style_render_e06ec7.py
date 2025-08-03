from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
import re
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional, Tuple, TypedDict, Union, cast
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

BaseFormatter = Union[str, Callable[..., str]]
ExtFormatter = Union[BaseFormatter, Dict[Any, Optional[BaseFormatter]]]
CSSPair = Tuple[str, Union[str, float]]
CSSList = List[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    pass
CSSStyles = List[CSSDict]
Subset = Union[slice, Sequence, Index]

class StylerRenderer:
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
        table_styles: Optional[CSSStyles] = None,
        table_attributes: Optional[str] = None,
        caption: Optional[str] = None,
        cell_ids: bool = True,
        precision: Optional[int] = None
    ) -> None:
        if isinstance(data, Series):
            data = data.to_frame()
        if not isinstance(data, DataFrame):
            raise TypeError('``data`` must be a Series or DataFrame')
        self.data = data
        self.index = data.index
        self.columns = data.columns
        if not isinstance(uuid_len, int) or uuid_len < 0:
            raise TypeError('``uuid_len`` must be an integer in range [0, 32].')
        self.uuid = uuid or uuid4().hex[:min(32, uuid_len)]
        self.uuid_len = len(self.uuid)
        self.table_styles = table_styles
        self.table_attributes = table_attributes
        self.caption = caption
        self.cell_ids = cell_ids
        self.css = {
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
        self.hide_index_names = False
        self.hide_column_names = False
        self.hide_index_ = [False] * self.index.nlevels
        self.hide_columns_ = [False] * self.columns.nlevels
        self.hidden_rows: List[int] = []
        self.hidden_columns: List[int] = []
        self.ctx: DefaultDict[Tuple[int, int], List[CSSPair]] = defaultdict(list)
        self.ctx_index: DefaultDict[Tuple[int, int], List[CSSPair]] = defaultdict(list)
        self.ctx_columns: DefaultDict[Tuple[int, int], List[CSSPair]] = defaultdict(list)
        self.cell_context: DefaultDict[Tuple[int, int], str] = defaultdict(str)
        self._todo: List[Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]] = []
        self.tooltips: Optional[Tooltips] = None
        precision = get_option('styler.format.precision') if precision is None else precision
        self._display_funcs: DefaultDict[Tuple[int, int], Callable[[Any], str]] = defaultdict(lambda: partial(_default_formatter, precision=precision))
        self._display_funcs_index: DefaultDict[Tuple[int, int], Callable[[Any], str]] = defaultdict(lambda: partial(_default_formatter, precision=precision))
        self._display_funcs_index_names: DefaultDict[int, Callable[[Any], str]] = defaultdict(lambda: partial(_default_formatter, precision=precision))
        self._display_funcs_columns: DefaultDict[Tuple[int, int], Callable[[Any], str]] = defaultdict(lambda: partial(_default_formatter, precision=precision))
        self._display_funcs_column_names: DefaultDict[int, Callable[[Any], str]] = defaultdict(lambda: partial(_default_formatter, precision=precision))

    def _render(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        blank: str = ''
    ) -> Dict[str, Any]:
        self._compute()
        dxs = []
        ctx_len = len(self.index)
        for i, concatenated in enumerate(self.concatenated):
            concatenated.hide_index_ = self.hide_index_
            concatenated.hidden_columns = self.hidden_columns
            foot = f'{self.css["foot"]}{i}'
            concatenated.css = {
                **self.css,
                'data': f'{foot}_data',
                'row_heading': f'{foot}_row_heading',
                'row': f'{foot}_row',
                'foot': f'{foot}_foot'
            }
            dx = concatenated._render(sparse_index, sparse_columns, max_rows, max_cols, blank)
            dxs.append(dx)
            for (r, c), v in concatenated.ctx.items():
                self.ctx[r + ctx_len, c] = v
            for (r, c), v in concatenated.ctx_index.items():
                self.ctx_index[r + ctx_len, c] = v
            ctx_len += len(concatenated.index)
        d = self._translate(sparse_index, sparse_columns, max_rows, max_cols, blank, dxs)
        return d

    def _render_html(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols, '&nbsp;')
        d.update(kwargs)
        return self.template_html.render(**d, html_table_tpl=self.template_html_table, html_style_tpl=self.template_html_style)

    def _render_latex(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        clines: Optional[str],
        **kwargs: Any
    ) -> str:
        d = self._render(sparse_index, sparse_columns, None, None)
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
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
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
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_string.render(**d)

    def _compute(self) -> StylerRenderer:
        self.ctx.clear()
        self.ctx_index.clear()
        self.ctx_columns.clear()
        r = self
        for func, args, kwargs in self._todo:
            r = func(self)(*args, **kwargs)
        return r

    def _translate(
        self,
        sparse_index: bool,
        sparse_cols: bool,
        max_rows: Optional[int],
        max_cols: Optional[int],
        blank: str = '&nbsp;',
        dxs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if dxs is None:
            dxs = []
        self.css['blank_value'] = blank
        d = {
            'uuid': self.uuid,
            'table_styles': format_table_styles(self.table_styles or []),
            'caption': self.caption
        }
        max_elements = get_option('styler.render.max_elements')
        max_rows = max_rows if max_rows else get_option('styler.render.max_rows')
        max_cols = max_cols if max_cols else get_option('styler.render.max_columns')
        max_rows, max_cols = _get_trimming_maximums(
            len(self.data.index), len(self.data.columns), max_elements, max_rows, max_cols)
        self.cellstyle_map_columns: DefaultDict[Tuple[CSSPair, ...], List[str]] = defaultdict(list)
        head = self._translate_header(sparse_cols, max_cols)
        d.update({'head': head})
        idx_lengths = _get_level_lengths(self.index, sparse_index, max_rows, self.hidden_rows)
        d.update({'index_lengths': idx_lengths})
        self.cellstyle_map: DefaultDict[Tuple[CSSPair, ...], List[str]] = defaultdict(list)
        self.cellstyle_map_index: DefaultDict[Tuple[CSSPair, ...], List[str]] = defaultdict(list)
        body = self._translate_body(idx_lengths, max_rows, max_cols)
        d.update({'body': body})
        ctx_maps = {
            'cellstyle': 'cellstyle_map',
            'cellstyle_index': 'cellstyle_map_index',
            'cellstyle_columns': 'cellstyle_map_columns'
        }
        for k, attr in ctx_maps.items():
            map = [{'props': list(props), 'selectors': selectors} for props, selectors in getattr(self, attr).items()]
            d.update({k: map})
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

    def _translate_header(self, sparsify_cols: bool, max_cols: int) -> List[List[Dict[str, Any]]]:
        col_lengths = _get_level_lengths(self.columns, sparsify_cols, max_cols, self.hidden_columns)
        clabels = self.data.columns.tolist()
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))
        head = []
        for r, hide in enumerate(self.hide_columns_):
            if hide or not clabels:
                continue
            header_row = self._generate_col_header_row((r, clabels), max_cols, col_lengths)
            head.append(header_row)
        if self.data.index.names and com.any_not_none(*self.data.index.names) and (not all(self.hide_index_)) and (not self.hide_index_names):
            index_names_row = self._generate_index_names_row(clabels, max_cols, col_lengths)
            head.append(index_names_row)
        return head

    def _generate_col_header_row(
        self,
        iter: Tuple[int, List[Tuple[Any, ...]]],
        max_cols: int,
        col_lengths: Dict[Tuple[int, int], int]
    ) -> List[Dict[str, Any]]:
        r, clabels = iter
        index_blanks = [_element('th', self.css['blank'], self.css['blank_value'], True)] * (self.index.nlevels - sum(self.hide_index_) - 1)
        name = self.data.columns.names[r]
        is_display = name is not None and (not self.hide_column_names)
        value = name if is_display else self.css['blank_value']
        display_value = self._display_funcs_column_names[r](value) if is_display else None
        column_name = [_element(
            'th',
            f'{self.css["blank"]} {self.css["level"]}{r}' if name is None else f'{self.css["index_name"]} {self.css["level"]}{r}',
            value,
            not all(self.hide_index_),
            display_value=display_value
        )]
        column_headers = []
        visible_col_count = 0
        for c, value in enumerate(clabels[r]):
            header_element_visible = _is_visible(c, r, col_lengths)
            if header_element_visible:
                visible_col_count += col_lengths.get((r, c), 0)
            if self._check_trim(visible_col_count, max_cols, column_headers, 'th', f'{self.css["col_heading"]} {self.css["level"]}{r} {self.css["col_trim"]}'):
                break
            header_element = _element(
                'th',
                f'{self.css["col_heading"]} {self.css["level"]}{r} {self.css["col"]}{c}',
                value,
                header_element_visible,
                display_value=self._display_funcs_columns[r, c](value),
                attributes=f'colspan="{col_lengths.get((r, c), 0)}"' if col_lengths.get((r, c), 0) > 1 else ''
            )
            if self.cell_ids:
                header_element['id'] = f'{self.css["level"]}{r}_{self.css["col"]}{c}'
            if header_element_visible and (r, c) in self.ctx_columns and self.ctx_columns[r, c]:
                header_element['id'] = f'{self.css["level"]}{r}_{self.css["col"]}{c}'
                self.cellstyle_map_columns[tuple(self.ctx_columns[r, c])].append(f'{self.css["level"]}{r}_{self.css["col"]}{c}')
            column_headers.append(header_element)
        return index_blanks + column_name + column_headers

    def _generate_index_names_row(
        self,
        iter: List[Tuple[Any, ...]],
        max_cols: int,
        col_lengths: Dict[Tuple[int, int], int]
    ) -> List[Dict[str, Any]]:
        clabels = iter
        index_names = [
            _element(
                'th',
                f'{self.css["index_name"]} {self.css["level"]}{c}',
                self.css['blank_value'] if name is None else name,
                not self.hide_index_[c],
                display_value=None if name is None else self._display_funcs_index_names[c](name)
            )
            for c, name in enumerate(self.data.index.names)
        ]
        column_blanks = []
        visible_col_count = 0
        if clabels:
            last_level = self.columns.nlevels - 1
            for c, value in enumerate(clabels[last_level]):
                header_element_visible = _is_visible(c, last_level, col_lengths)
                if header_element_visible:
                    visible_col_count += 1
                if self._check_trim(visible_col_count, max_cols, column_blanks, 'th', f'{self.css["blank"]} {self.css["col"]}{c} {self.css["col_trim"]}', self.css['blank_value']):
                    break
                column_blanks.append(_element('th', f'{self.css["blank"]} {self.css["col"]}{c}', self.css['blank_value'], c not in self.hidden_columns))
        return index_names + column_blanks

    def _translate_body(
        self,
        idx_lengths: Dict[Tuple[int, int], int],
        max_rows: int,
        max_cols: int
    ) -> List[List[Dict[str, Any]]]:
        rlabels = self.data.index.tolist()
        if not isinstance(self.data.index, MultiIndex):
            rlabels = [[x] for x in rlabels]
        body = []
        visible_row_count = 0
        for r, row_tup in [z for z in enumerate(self.data.itertuples()) if z[0] not in self.hidden_rows]:
            visible_row_count +=