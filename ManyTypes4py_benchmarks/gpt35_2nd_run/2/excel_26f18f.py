from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import itertools
import re
from typing import TYPE_CHECKING, Any, cast
import warnings
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Period, PeriodIndex
from pandas.io.formats.css import CSSResolver, CSSWarning
from pandas.io.formats.format import get_level_lengths
from pandas._typing import ExcelWriterMergeCells, FilePath, IndexLabel, StorageOptions, WriteExcelBuffer
from pandas import ExcelWriter

class ExcelCell:
    __fields__: tuple[str, str, Any, Any, Any, Any] = ('row', 'col', 'val', 'style', 'mergestart', 'mergeend')
    __slots__: tuple[str] = __fields__

    def __init__(self, row: Any, col: Any, val: Any, style: Any = None, mergestart: Any = None, mergeend: Any = None) -> None:
        self.row = row
        self.col = col
        self.val = val
        self.style = style
        self.mergestart = mergestart
        self.mergeend = mergeend

class CssExcelCell(ExcelCell):

    def __init__(self, row: Any, col: Any, val: Any, style: Any, css_styles: Any, css_row: Any, css_col: Any, css_converter: Any, **kwargs: Any) -> None:
        if css_styles and css_converter:
            declaration_dict = {prop.lower(): val for prop, val in css_styles[css_row, css_col]}
            unique_declarations = frozenset(declaration_dict.items())
            style = css_converter(unique_declarations)
        super().__init__(row=row, col=col, val=val, style=style, **kwargs)

class CSSToExcelConverter:
    NAMED_COLORS: dict[str, str] = CSS4_COLORS
    VERTICAL_MAP: dict[str, str] = {'top': 'top', 'text-top': 'top', 'middle': 'center', 'baseline': 'bottom', 'bottom': 'bottom', 'text-bottom': 'bottom'}
    BOLD_MAP: dict[str, Any] = {'bold': True, 'bolder': True, '600': True, '700': True, '800': True, '900': True, 'normal': False, 'lighter': False, '100': False, '200': False, '300': False, '400': False, '500': False}
    ITALIC_MAP: dict[str, Any] = {'normal': False, 'italic': True, 'oblique': True}
    FAMILY_MAP: dict[str, int] = {'serif': 1, 'sans-serif': 2, 'cursive': 4, 'fantasy': 5}
    BORDER_STYLE_MAP: dict[str, str] = {style.lower(): style for style in ['dashed', 'mediumDashDot', 'dashDotDot', 'hair', 'dotted', 'mediumDashDotDot', 'double', 'dashDot', 'slantDashDot', 'mediumDashed']}

    def __init__(self, inherited: str = None) -> None:
        if inherited is not None:
            self.inherited = self.compute_css(inherited)
        else:
            self.inherited = None
        self._call_cached = functools.cache(self._call_uncached)
    compute_css: CSSResolver = CSSResolver()

    def __call__(self, declarations: str | frozenset[tuple[str, str]]) -> dict:
        return self._call_cached(declarations)

    def _call_uncached(self, declarations: str | frozenset[tuple[str, str]]) -> dict:
        properties = self.compute_css(declarations, self.inherited)
        return self.build_xlstyle(properties)

    def build_xlstyle(self, props: dict) -> dict:
        out = {'alignment': self.build_alignment(props), 'border': self.build_border(props), 'fill': self.build_fill(props), 'font': self.build_font(props), 'number_format': self.build_number_format(props)}

        def remove_none(d: dict) -> None:
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    remove_none(v)
                    if not v:
                        del d[k]
        remove_none(out)
        return out

    def build_alignment(self, props: dict) -> dict:
        return {'horizontal': props.get('text-align'), 'vertical': self._get_vertical_alignment(props), 'wrap_text': self._get_is_wrap_text(props)}

    def _get_vertical_alignment(self, props: dict) -> str | None:
        vertical_align = props.get('vertical-align')
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)
        return None

    def _get_is_wrap_text(self, props: dict) -> bool | None:
        if props.get('white-space') is None:
            return None
        return bool(props['white-space'] not in ('nowrap', 'pre', 'pre-line'))

    def build_border(self, props: dict) -> dict:
        return {side: {'style': self._border_style(props.get(f'border-{side}-style'), props.get(f'border-{side}-width'), self.color_to_excel(props.get(f'border-{side}-color'))), 'color': self.color_to_excel(props.get(f'border-{side}-color'))} for side in ['top', 'right', 'bottom', 'left']}

    def _border_style(self, style: str, width: str, color: str) -> str | None:
        if width is None and style is None and (color is None):
            return None
        if width is None and style is None:
            return 'none'
        if style in ('none', 'hidden'):
            return 'none'
        width_name = self._get_width_name(width)
        if width_name is None:
            return 'none'
        if style in (None, 'groove', 'ridge', 'inset', 'outset', 'solid'):
            return width_name
        if style == 'double':
            return 'double'
        if style == 'dotted':
            if width_name in ('hair', 'thin'):
                return 'dotted'
            return 'mediumDashDotDot'
        if style == 'dashed':
            if width_name in ('hair', 'thin'):
                return 'dashed'
            return 'mediumDashed'
        elif style in self.BORDER_STYLE_MAP:
            return self.BORDER_STYLE_MAP[style]
        else:
            warnings.warn(f'Unhandled border style format: {style!r}', CSSWarning, stacklevel=find_stack_level())
            return 'none'

    def _get_width_name(self, width_input: str) -> str | None:
        width = self._width_to_float(width_input)
        if width < 1e-05:
            return None
        elif width < 1.3:
            return 'thin'
        elif width < 2.8:
            return 'medium'
        return 'thick'

    def _width_to_float(self, width: str) -> float:
        if width is None:
            width = '2pt'
        return self._pt_to_float(width)

    def _pt_to_float(self, pt_string: str) -> float:
        assert pt_string.endswith('pt')
        return float(pt_string.rstrip('pt'))

    def build_fill(self, props: dict) -> dict | None:
        fill_color = props.get('background-color')
        if fill_color not in (None, 'transparent', 'none'):
            return {'fgColor': self.color_to_excel(fill_color), 'patternType': 'solid'}

    def build_number_format(self, props: dict) -> dict:
        fc = props.get('number-format')
        fc = fc.replace('§', ';') if isinstance(fc, str) else fc
        return {'format_code': fc}

    def build_font(self, props: dict) -> dict:
        font_names = self._get_font_names(props)
        decoration = self._get_decoration(props)
        return {'name': font_names[0] if font_names else None, 'family': self._select_font_family(font_names), 'size': self._get_font_size(props), 'bold': self._get_is_bold(props), 'italic': self._get_is_italic(props), 'underline': 'single' if 'underline' in decoration else None, 'strike': 'line-through' in decoration or None, 'color': self.color_to_excel(props.get('color')), 'shadow': self._get_shadow(props)}

    def _get_is_bold(self, props: dict) -> bool | None:
        weight = props.get('font-weight')
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def _get_is_italic(self, props: dict) -> bool | None:
        font_style = props.get('font-style')
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def _get_decoration(self, props: dict) -> tuple[str]:
        decoration = props.get('text-decoration')
        if decoration is not None:
            return decoration.split()
        else:
            return ()

    def _get_shadow(self, props: dict) -> bool | None:
        if 'text-shadow' in props:
            return bool(re.search('^[^#(]*[1-9]', props['text-shadow']))
        return None

    def _get_font_names(self, props: dict) -> list[str]:
        font_names_tmp = re.findall('(?x)\n            (\n            "(?:[^"]|\\\\")+"\n            |\n            \'(?:[^\']|\\\\\')+\'\n            |\n            [^\'",]+\n            )(?=,|\\s*$)\n        ', props.get('font-family', ''))
        font_names = []
        for name in font_names_tmp:
            if name[:1] == '"':
                name = name[1:-1].replace('\\"', '"')
            elif name[:1] == "'":
                name = name[1:-1].replace("\\'", "'")
            else:
                name = name.strip()
            if name:
                font_names.append(name)
        return font_names

    def _get_font_size(self, props: dict) -> float | None:
        size = props.get('font-size')
        if size is None:
            return size
        return self._pt_to_float(size)

    def _select_font_family(self, font_names: list[str]) -> int | None:
        family = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break
        return family

    def color_to_excel(self, val: str) -> str | None:
        if val is None:
            return None
        if self._is_hex_color(val):
            return self._convert_hex_to_excel(val)
        try:
            return self.NAMED_COLORS[val]
        except KeyError:
            warnings.warn(f'Unhandled color format: {val!r}', CSSWarning, stacklevel=find_stack_level())
        return None

    def _is_hex_color(self, color_string: str) -> bool:
        return bool(color_string.startswith('#'))

    def _convert_hex_to_excel(self, color_string: str) -> str:
        code = color_string.lstrip('#')
        if self._is_shorthand_color(color_string):
            return (code[0] * 2 + code[1] * 2 + code[2] * 2).upper()
        else:
            return code.upper()

    def _is_shorthand_color(self, color_string: str) -> bool:
        code = color_string.lstrip('#')
        if len(code) == 3:
            return True
        elif len(code) == 6:
            return False
        else:
            raise ValueError(f'Unexpected color {color_string}')

class ExcelFormatter:
    max_rows: int = 2 ** 20
    max_cols: int = 2 ** 14

    def __init__(self, df: DataFrame, na_rep: str = '', float_format: str | None = None, cols: Sequence | None = None, header: bool | Sequence[str] = True, index: bool = True, index_label: str | Sequence | None = None, merge_cells: bool | str = False, inf_rep: str = 'inf', style_converter: Callable | None = None) -> None:
        self.rowcounter: int = 0
        self.na_rep: str = na_rep
        if not isinstance(df, DataFrame):
            self.styler = df
            self.styler._compute()
            df = df.data
            if style_converter is None:
                style_converter = CSSToExcelConverter()
            self.style_converter = style_converter
        else:
            self.styler = None
            self.style_converter = None
        self.df: DataFrame = df
        if cols is not None:
            if not len(Index(cols).intersection(df.columns)):
                raise KeyError('passes columns are not ALL present dataframe')
            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                raise KeyError("Not all names specified in 'columns' are found")
            self.df = df.reindex(columns=cols)
        self.columns: Index = self.df.columns
        self.float_format: str | None = float_format
        self.index: bool = index
        self.index_label: str | Sequence | None = index_label
        self.header: bool | Sequence[str] = header
        if not isinstance(merge_cells, bool) and merge_cells != 'columns':
            raise ValueError(f'Unexpected value for merge_cells={merge_cells!r}.')
        self.merge_cells: bool | str = merge_cells
        self.inf_rep: str = inf_rep

    def _format_value(self, val: Any) -> Any:
        if is_scalar(val) and missing.isna(val):
            val = self.na_rep
        elif is_float(val):
            if missing.isposinf_scalar(val):
                val = self.inf_rep
            elif missing.isneginf_scalar(val):
                val = f'-{self.inf_rep}'
            elif self.float_format is not None:
                val = float(self.float_format % val)
        if getattr(val, 'tzinfo', None) is not None:
            raise ValueError('Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.')
        return val

    def _format_header_mi(self) -> Iterable[ExcelCell]:
        if self.columns.nlevels > 1:
            if not self.index:
                raise NotImplementedError("Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.")
        if not (self._has_aliases or self.header):
            return
        columns = self.columns
        merge_columns = self.merge_cells in {True, 'columns'}
        level_strs = columns._format_multi(sparsify=merge_columns, include_names=False)
        level_lengths = get_level_lengths(level_strs)
        coloffset = 0
        lnum = 0
        if self.index and isinstance(self.df.index, MultiIndex):
            coloffset = self.df.index.nlevels - 1
        for lnum, name in enumerate(columns.names):
            yield ExcelCell(row=lnum, col=coloffset, val=name, style=None)
        for lnum, (spans, levels, level_codes) in enumerate(zip(level_lengths, columns.levels, columns.codes)):
            values = levels.take(level_codes)
            for i, span_val in spans.items():
                mergestart, mergeend = (None, None)
                if merge_columns and span_val > 1:
                    mergestart, mergeend = (lnum, coloffset + i + span_val)
                yield CssExcelCell(row=lnum, col=coloffset + i + 1, val=values[i], style=None, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=lnum, css_col=i, css_converter=self.style_converter, mergestart=mergestart, mergeend=mergeend)
        self.rowcounter = lnum

    def _format_header_regular(self) -> Iterable[CssExcelCell]:
        if self._has_aliases or self.header:
            self.rowcounter += 1
        if self.index:
            if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
                index_label = self.index_label[0]
            elif self.index_label and isinstance(self.index_label, str):
                index_label = self.index_label
            else:
                index_label = self.df.index.names[0]
            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1
            for colindex, colname in enumerate(self.columns):
                yield CssExcelCell(row=self.rowcounter, col=colindex, val=colname, style=None, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=0, css_col=colindex, css_converter=self.style_converter)

    def _format_header(self) -> Iterable[ExcelCell]:
        if isinstance(self.columns, MultiIndex):
            gen = self._format_header_mi()
        else:
            gen = self._format_header_regular()
        gen2 = ()
        if self.df.index.names:
            row = [x if x is not None else '' for x in self.df.index.names] + [''] * len(self.columns)
            if functools.reduce(lambda x, y: x and y, (x != '' for x in row)):
                gen2 = (ExcelCell(self.rowcounter, colindex, val, None) for colindex, val in enumerate(row))
                self.rowcounter += 1
        return itertools.chain(gen, gen2)

    def _format_body(self) -> Iterable[CssExcelCell]:
        if isinstance(self.df.index, MultiIndex):
            return self._format_hierarchical_rows()
        else:
            return self._format_regular_rows()

    def _format_regular_rows(self) -> Iterable[CssExcelCell]:
        if self._has_aliases or self.header:
            self.rowcounter += 1
        if self.index:
            if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
                index_labels = self.index_label
            else:
                index_labels = self.df.index.names
            if self.index_label and isinstance(self.index_label, str):
                index_labels = [self.index_label]
            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1
            if com.any_not_none(*index_labels) and self.header is not False:
                for cidx, name in enumerate(index_labels):
                    yield ExcelCell(self.rowcounter - 1, cidx, name, None)
            index_values = self.df.index
            if isinstance(self.df.index, PeriodIndex):
                index_values = self.df.index.to_timestamp()
            for idx, idxval in enumerate(index_values):
                yield CssExcelCell(row=self.rowcounter + idx, col=0, val=idxval, style=None, css_styles=getattr(self.styler, 'ctx_index', None), css_row=idx, css_col=0, css_converter=self.style_converter)
            coloffset = 1
        else:
            coloffset =