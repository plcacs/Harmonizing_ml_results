from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import itertools
import re
from typing import TYPE_CHECKING, Any, cast
import warnings
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Period, PeriodIndex
import pandas.core.common as com
from pandas.io.formats.css import CSSResolver, CSSWarning
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.style import Styler
from pandas.io.excel import ExcelWriter
from pandas.io.excel._base import ExcelFormatter
from pandas.io.excel._util import ExcelCell, CssExcelCell, CSSToExcelConverter

if TYPE_CHECKING:
    from pandas._typing import ExcelWriterMergeCells, FilePath, IndexLabel, StorageOptions, WriteExcelBuffer

class ExcelCell:
    __fields__: tuple[str, ...] = 'row', 'col', 'val', 'style', 'mergestart', 'mergeend'
    __slots__: tuple[str, ...] = __fields__

    def __init__(self, row: int, col: int, val: Any, style: Any = None, mergestart: Any = None, mergeend: Any = None) -> None:
        self.row = row
        self.col = col
        self.val = val
        self.style = style
        self.mergestart = mergestart
        self.mergeend = mergeend

class CssExcelCell(ExcelCell):

    def __init__(self, row: int, col: int, val: Any, style: Any, css_styles: Any, css_row: int, css_col: int, css_converter: Any, **kwargs: Any) -> None:
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

    def __call__(self, declarations: str | frozenset[tuple[str, str]]) -> dict:
        return self._call_cached(declarations)

    def func_rmlgtmg5(self, declarations: str | frozenset[tuple[str, str]]) -> dict:
        properties = self.compute_css(declarations, self.inherited)
        return self.build_xlstyle(properties)

    def func_sgriejwf(self, props: dict) -> dict:
        out: dict = {'alignment': self.build_alignment(props), 'border': self.build_border(props), 'fill': self.build_fill(props), 'font': self.build_font(props), 'number_format': self.build_number_format(props)}

        def func_5b7ijeux(d: dict) -> None:
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    func_5b7ijeux(v)
                    if not v:
                        del d[k]
        func_5b7ijeux(out)
        return out

    def func_jbytaw2g(self, props: dict) -> dict:
        return {'horizontal': props.get('text-align'), 'vertical': self._get_vertical_alignment(props), 'wrap_text': self._get_is_wrap_text(props)}

    def func_vbkjt72c(self, props: dict) -> str | None:
        vertical_align = props.get('vertical-align')
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)
        return None

    def func_1b0obxu1(self, props: dict) -> bool | None:
        if props.get('white-space') is None:
            return None
        return bool(props['white-space'] not in ('nowrap', 'pre', 'pre-line'))

    def func_mkdh0hd6(self, props: dict) -> dict:
        return {side: {'style': self._border_style(props.get(f'border-{side}-style'), props.get(f'border-{side}-width'), self.color_to_excel(props.get(f'border-{side}-color'))), 'color': self.color_to_excel(props.get(f'border-{side}-color'))} for side in ['top', 'right', 'bottom', 'left']}

    def func_va8m65mn(self, style: str, width: str, color: str) -> str | None:
        if width is None and style is None and color is None:
            return None
        if width is None and style is None:
            return 'none'
        if style in ('none', 'hidden'):
            return 'none'
        width_name = self._get_width_name(width)
        if width_name is None:
            return 'none'
        if style in ('groove', 'ridge', 'inset', 'outset', 'solid'):
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

    def func_adnrko8x(self, width_input: str) -> str | None:
        width = self._width_to_float(width_input)
        if width < 1e-05:
            return None
        elif width < 1.3:
            return 'thin'
        elif width < 2.8:
            return 'medium'
        return 'thick'

    def func_zz0d9sq9(self, width: str) -> float:
        if width is None:
            width = '2pt'
        return self._pt_to_float(width)

    def func_soqbczyw(self, pt_string: str) -> float:
        assert pt_string.endswith('pt')
        return float(pt_string.rstrip('pt'))

    def func_v0uilk4u(self, props: dict) -> dict:
        fill_color = props.get('background-color')
        if fill_color not in (None, 'transparent', 'none'):
            return {'fgColor': self.color_to_excel(fill_color), 'patternType': 'solid'}

    def func_iscntdlo(self, props: dict) -> dict:
        fc = props.get('number-format')
        fc = fc.replace('§', ';') if isinstance(fc, str) else fc
        return {'format_code': fc}

    def func_c7uhyu66(self, props: dict) -> dict:
        font_names = self._get_font_names(props)
        decoration = self._get_decoration(props)
        return {'name': font_names[0] if font_names else None, 'family': self._select_font_family(font_names), 'size': self._get_font_size(props), 'bold': self._get_is_bold(props), 'italic': self._get_is_italic(props), 'underline': 'single' if 'underline' in decoration else None, 'strike': 'line-through' in decoration or None, 'color': self.color_to_excel(props.get('color')), 'shadow': self._get_shadow(props)}

    def func_acxeoyi8(self, props: dict) -> bool | None:
        weight = props.get('font-weight')
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def func_a57kxoqy(self, props: dict) -> bool | None:
        font_style = props.get('font-style')
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def func_1ttvn4w1(self, props: dict) -> tuple[str, ...]:
        decoration = props.get('text-decoration')
        if decoration is not None:
            return decoration.split()
        else:
            return ()

    def func_7vgreki2(self, decoration: tuple[str, ...]) -> str | None:
        if 'underline' in decoration:
            return 'single'
        return None

    def func_1x54iaqz(self, props: dict) -> bool | None:
        if 'text-shadow' in props:
            return bool(re.search('^[^#(]*[1-9]', props['text-shadow']))
        return None

    def func_u74b97m2(self, props: dict) -> list[str]:
        font_names_tmp = re.findall("""(?x)
            (
            "(?:[^"]|\\\\")+"
            |
            '(?:[^']|\\\\')+'
            |
            [^'",]+
            )(?=,|\\s*$)
        """, props.get('font-family', ''))
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

    def func_70jra572(self, props: dict) -> float:
        size = props.get('font-size')
        if size is None:
            return size
        return self._pt_to_float(size)

    def func_p1oupn4n(self, font_names: list[str]) -> int | None:
        family = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break
        return family

    def func_6ogj28oz(self, val: str) -> str | None:
        if val is None:
            return None
        if self._is_hex_color(val):
            return self._convert_hex_to_excel(val)
        try:
            return self.NAMED_COLORS[val]
        except KeyError:
            warnings.warn(f'Unhandled color format: {val!r}', CSSWarning, stacklevel=find_stack_level())
        return None

    def func_l6prgbjl(self, color_string: str) -> bool:
        return color_string.startswith('#')

    def func_z3m5d6qm(self, color_string: str) -> str:
        code = color_string.lstrip('#')
        if self._is_shorthand_color(color_string):
            return (code[0] * 2 + code[1] * 2 + code[2] * 2).upper()
        else:
            return code.upper()

    def func_jbxcwzim(self, color_string: str) -> bool:
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

    def __init__(self, df: DataFrame | Styler, na_rep: str = '', float_format: str | None = None, cols: Sequence | None = None, header: bool | Sequence[str] = True, index: bool = True, index_label: str | Sequence | None = None, merge_cells: bool | str = False, inf_rep: str = 'inf', style_converter: Callable | None = None) -> None:
        self.rowcounter: int = 0
        self.na_rep: str = na_rep
        if not isinstance(df, DataFrame):
            self.styler: Styler = df
            self.styler._compute()
            df = df.data
            if style_converter is None:
                style_converter = CSSToExcelConverter()
            self.style_converter: Callable = style_converter
        else:
            self.styler: None = None
            self.style_converter: None = None
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

    def func_hh53u5dk(self, val: Any) -> Any:
        if is_scalar(val) and com.isna(val):
            val = self.na_rep
        elif com.isfloat(val):
            if com.isposinf_scalar(val):
                val = self.inf_rep
            elif com.isneginf_scalar(val):
                val = f'-{self.inf_rep}'
            elif self.float_format is not None:
                val = float(self.float_format % val)
        if getattr(val, 'tzinfo', None) is not None:
            raise ValueError('Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.')
        return val

    def func_p3b82rxh(self) -> Iterable[ExcelCell]:
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
                mergestart, mergeend = None, None
                if merge_columns and span_val > 1:
                    mergestart, mergeend = lnum, coloffset + i + span_val
                yield CssExcelCell(row=lnum, col=coloffset + i + 1, val=values[i], style=None, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=lnum, css_col=i, css_converter=self.style_converter, mergestart=mergestart, mergeend=mergeend)
        self.rowcounter = lnum

    def func_vx3cr1t9(self) -> Iterable[ExcelCell]:
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
            if index_label and self.header is not False:
                yield ExcelCell(self.rowcounter - 1, 0, index_label, None)
            index_values = self.df.index
            if isinstance(self.df.index, PeriodIndex):
                index_values = self.df.index.to_timestamp()
            for idx, idxval in enumerate(index_values):
                yield CssExcelCell(row=self.rowcounter + idx, col=0, val=idxval, style=None, css_styles=getattr(self.styler, 'ctx_index', None), css_row=idx, css_col=0, css_converter=self.style_converter)
            coloffset = 1
        else:
            coloffset = 0
        yield from self._generate_body(coloffset)

    def func_qo9j9gx0(self) -> Iterable[ExcelCell]:
        if self._has_aliases or self.header:
            self.rowcounter += 1
        gcolidx = 0
        if self.index:
            index_labels = self.df.index.names
            if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
                index_labels = self.index_label
            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1
            if com.any_not_none(*index_labels) and self.header is not False:
                for cidx, name in enumerate(index_labels):
                    yield ExcelCell(self.rowcounter - 1, cidx, name, None)
            if self.merge_cells and self.merge_cells != 'columns':
                level_strs = self.df.index._format_multi(sparsify=True, include_names=False)
                for spans, levels, level_codes in zip(level_strs, self.df.index.levels, self.df