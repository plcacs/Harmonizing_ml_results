"""
Utilities for conversion to writer-agnostic Excel representation.
"""
from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import itertools
import re
from typing import TYPE_CHECKING, Any, cast, Optional, Union, Tuple, List, Dict, Set, FrozenSet
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import is_float, is_scalar
from pandas import DataFrame, Index, MultiIndex, Period, PeriodIndex
import pandas.core.common as com
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import CSSResolver, CSSWarning
from pandas.io.formats.format import get_level_lengths
if TYPE_CHECKING:
    from pandas._typing import ExcelWriterMergeCells, FilePath, IndexLabel, StorageOptions, WriteExcelBuffer
    from pandas import ExcelWriter

class ExcelCell:
    __fields__: Tuple[str, ...] = ('row', 'col', 'val', 'style', 'mergestart', 'mergeend')
    __slots__: Tuple[str, ...] = __fields__

    def __init__(self, row: int, col: int, val: Any, style: Optional[Any] = None, mergestart: Optional[int] = None, mergeend: Optional[int] = None) -> None:
        self.row: int = row
        self.col: int = col
        self.val: Any = val
        self.style: Optional[Any] = style
        self.mergestart: Optional[int] = mergestart
        self.mergeend: Optional[int] = mergeend

class CssExcelCell(ExcelCell):

    def __init__(self, row: int, col: int, val: Any, style: Optional[Any], css_styles: Optional[Any], css_row: int, css_col: int, css_converter: Optional[Callable], **kwargs: Any) -> None:
        if css_styles and css_converter:
            declaration_dict: Dict[str, str] = {prop.lower(): val for (prop, val) in css_styles[css_row, css_col]}
            unique_declarations: FrozenSet[Tuple[str, str]] = frozenset(declaration_dict.items())
            style = css_converter(unique_declarations)
        super().__init__(row=row, col=col, val=val, style=style, **kwargs)

class CSSToExcelConverter:
    """
    A callable for converting CSS declarations to ExcelWriter styles

    Supports parts of CSS 2.2, with minimal CSS 3.0 support (e.g. text-shadow),
    focusing on font styling, backgrounds, borders and alignment.

    Operates by first computing CSS styles in a fairly generic
    way (see :meth:`compute_css`) then determining Excel style
    properties from CSS properties (see :meth:`build_xlstyle`).

    Parameters
    ----------
    inherited : str, optional
        CSS declarations understood to be the containing scope for the
        CSS processed by :meth:`__call__`.
    """
    NAMED_COLORS: Dict[str, str] = CSS4_COLORS
    VERTICAL_MAP: Dict[str, str] = {'top': 'top', 'text-top': 'top', 'middle': 'center', 'baseline': 'bottom', 'bottom': 'bottom', 'text-bottom': 'bottom'}
    BOLD_MAP: Dict[str, bool] = {'bold': True, 'bolder': True, '600': True, '700': True, '800': True, '900': True, 'normal': False, 'lighter': False, '100': False, '200': False, '300': False, '400': False, '500': False}
    ITALIC_MAP: Dict[str, bool] = {'normal': False, 'italic': True, 'oblique': True}
    FAMILY_MAP: Dict[str, int] = {'serif': 1, 'sans-serif': 2, 'cursive': 4, 'fantasy': 5}
    BORDER_STYLE_MAP: Dict[str, str] = {style.lower(): style for style in ['dashed', 'mediumDashDot', 'dashDotDot', 'hair', 'dotted', 'mediumDashDotDot', 'double', 'dashDot', 'slantDashDot', 'mediumDashed']}
    inherited: Optional[Dict[str, str]]

    def __init__(self, inherited: Optional[str] = None) -> None:
        if inherited is not None:
            self.inherited = self.compute_css(inherited)
        else:
            self.inherited = None
        self._call_cached: Callable = functools.cache(self._call_uncached)
    compute_css: Callable = CSSResolver()

    def __call__(self, declarations: Union[str, FrozenSet[Tuple[str, str]]]) -> Dict[str, Any]:
        """
        Convert CSS declarations to ExcelWriter style.

        Parameters
        ----------
        declarations : str | frozenset[tuple[str, str]]
            CSS string or set of CSS declaration tuples.
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}

        Returns
        -------
        xlstyle : dict
            A style as interpreted by ExcelWriter when found in
            ExcelCell.style.
        """
        return self._call_cached(declarations)

    def _call_uncached(self, declarations: Union[str, FrozenSet[Tuple[str, str]]]) -> Dict[str, Any]:
        properties: Dict[str, str] = self.compute_css(declarations, self.inherited)
        return self.build_xlstyle(properties)

    def build_xlstyle(self, props: Mapping[str, str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {'alignment': self.build_alignment(props), 'border': self.build_border(props), 'fill': self.build_fill(props), 'font': self.build_font(props), 'number_format': self.build_number_format(props)}

        def remove_none(d: Dict[str, Any]) -> None:
            """Remove key where value is None, through nested dicts"""
            for (k, v) in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    remove_none(v)
                    if not v:
                        del d[k]
        remove_none(out)
        return out

    def build_alignment(self, props: Mapping[str, str]) -> Dict[str, Optional[Union[str, bool]]]:
        return {'horizontal': props.get('text-align'), 'vertical': self._get_vertical_alignment(props), 'wrap_text': self._get_is_wrap_text(props)}

    def _get_vertical_alignment(self, props: Mapping[str, str]) -> Optional[str]:
        vertical_align: Optional[str] = props.get('vertical-align')
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)
        return None

    def _get_is_wrap_text(self, props: Mapping[str, str]) -> Optional[bool]:
        if props.get('white-space') is None:
            return None
        return bool(props['white-space'] not in ('nowrap', 'pre', 'pre-line'))

    def build_border(self, props: Mapping[str, str]) -> Dict[str, Dict[str, Optional[str]]]:
        return {side: {'style': self._border_style(props.get(f'border-{side}-style'), props.get(f'border-{side}-width'), self.color_to_excel(props.get(f'border-{side}-color'))), 'color': self.color_to_excel(props.get(f'border-{side}-color'))} for side in ['top', 'right', 'bottom', 'left']}

    def _border_style(self, style: Optional[str], width: Optional[str], color: Optional[str]) -> Optional[str]:
        if width is None and style is None and (color is None):
            return None
        if width is None and style is None:
            return 'none'
        if style in ('none', 'hidden'):
            return 'none'
        width_name: Optional[str] = self._get_width_name(width)
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

    def _get_width_name(self, width_input: Optional[str]) -> Optional[str]:
        width: float = self._width_to_float(width_input)
        if width < 1e-05:
            return None
        elif width < 1.3:
            return 'thin'
        elif width < 2.8:
            return 'medium'
        return 'thick'

    def _width_to_float(self, width: Optional[str]) -> float:
        if width is None:
            width = '2pt'
        return self._pt_to_float(width)

    def _pt_to_float(self, pt_string: str) -> float:
        assert pt_string.endswith('pt')
        return float(pt_string.rstrip('pt'))

    def build_fill(self, props: Mapping[str, str]) -> Optional[Dict[str, str]]:
        fill_color: Optional[str] = props.get('background-color')
        if fill_color not in (None, 'transparent', 'none'):
            return {'fgColor': self.color_to_excel(fill_color), 'patternType': 'solid'}
        return None

    def build_number_format(self, props: Mapping[str, str]) -> Dict[str, Optional[str]]:
        fc: Optional[str] = props.get('number-format')
        if isinstance(fc, str):
            fc = fc.replace('§', ';')
        return {'format_code': fc}

    def build_font(self, props: Mapping[str, str]) -> Dict[str, Optional[Union[bool, float, str]]]:
        font_names: List[str] = self._get_font_names(props)
        decoration: Sequence[str] = self._get_decoration(props)
        return {'name': font_names[0] if font_names else None, 'family': self._select_font_family(font_names), 'size': self._get_font_size(props), 'bold': self._get_is_bold(props), 'italic': self._get_is_italic(props), 'underline': 'single' if 'underline' in decoration else None, 'strike': 'line-through' in decoration or None, 'color': self.color_to_excel(props.get('color')), 'shadow': self._get_shadow(props)}

    def _get_is_bold(self, props: Mapping[str, str]) -> Optional[bool]:
        weight: Optional[str] = props.get('font-weight')
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def _get_is_italic(self, props: Mapping[str, str]) -> Optional[bool]:
        font_style: Optional[str] = props.get('font-style')
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def _get_decoration(self, props: Mapping[str, str]) -> Sequence[str]:
        decoration: Optional[str] = props.get('text-decoration')
        if decoration is not None:
            return decoration.split()
        else:
            return ()

    def _get_underline(self, decoration: Sequence[str]) -> Optional[str]:
        if 'underline' in decoration:
            return 'single'
        return None

    def _get_shadow(self, props: Mapping[str, str]) -> Optional[bool]:
        if 'text-shadow' in props:
            return bool(re.search('^[^#(]*[1-9]', props['text-shadow']))
        return None

    def _get_font_names(self, props: Mapping[str, str]) -> List[str]:
        font_names_tmp: List[str] = re.findall('(?x)\n            (\n            "(?:[^"]|\\\\")+"\n            |\n            \'(?:[^\']|\\\\\')+\'\n            |\n            [^\'",]+\n            )(?=,|\\s*$)\n        ', props.get('font-family', ''))
        font_names: List[str] = []
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

    def _get_font_size(self, props: Mapping[str, str]) -> Optional[float]:
        size: Optional[str] = props.get('font-size')
        if size is None:
            return size
        return self._pt_to_float(size)

    def _select_font_family(self, font_names: Sequence[str]) -> Optional[int]:
        family: Optional[int] = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break
        return family

    def color_to_excel(self, val: Optional[str]) -> Optional[str]:
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
        code: str = color_string.lstrip('#')
        if self._is_shorthand_color(color_string):
            return (code[0] * 2 + code[1] * 2 + code[2] * 2).upper()
        else:
            return code.upper()

    def _is_shorthand_color(self, color_string: str) -> bool:
        """Check if color code is shorthand.

        #FFF is a shorthand as opposed to full #FFFFFF.
        """
        code: str = color_string.lstrip('#')
        if len(code) == 3:
            return True
        elif len(code) == 6:
            return False
        else:
            raise ValueError(f'Unexpected color {color_string}')

class ExcelFormatter:
    """
    Class for formatting a DataFrame to a list of ExcelCells,

    Parameters
    ----------
    df : DataFrame or Styler
    na_rep: na representation
    float_format : str, default None
        Format string for floating point numbers
    cols : sequence, optional
        Columns to write
    header : bool or sequence of str, default True
        Write out column names. If a list of string is given it is
        assumed to be aliases for the column names
    index : bool, default True
        output row names (index)
    index_label : str or sequence, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the DataFrame uses MultiIndex.
    merge_cells : bool or 'columns', default False
        Format MultiIndex column headers and Hierarchical Rows as merged cells
        if True. Merge MultiIndex column headers only if 'columns'.
        .. versionchanged:: 3.0.0
            Added the 'columns' option.
    inf_rep : str, default `'inf'`
        representation for np.inf values (which aren't representable in Excel)
        A `'-'` sign will be added in front of -inf.
    style_converter : callable, optional
        This translates Styler styles (CSS) into ExcelWriter styles.
        Defaults to ``CSSToExcelConverter()``.
        It should have signature css_declarations string -> excel style.
        This is only called for body cells.
    """
    max_rows: int = 2 ** 20
    max_cols: int = 2 ** 14

    def __init__(self, df: Any, na_rep: str = '', float_format: Optional[str] = None, cols: Optional[Sequence[Hashable]] = None, header: Union[Sequence[Hashable], bool] = True, index: bool = True, index_label: Optional[IndexLabel] = None, merge_cells: ExcelWriterMergeCells = False, inf_rep: str = 'inf', style_converter: Optional[Callable] = None) -> None:
        self.rowcounter: int = 0
        self.na_rep: str = na_rep
        if not isinstance(df, DataFrame):
            self.styler: Optional[Any] = df
            self.styler._compute()
            df = df.data
            if style_converter is None:
                style_converter = CSSToExcelConverter()
            self.style_converter: Optional[Callable] = style_converter
        else:
            self.styler: Optional[Any] = None
            self.style_converter: Optional[Callable] = None
        self.df: DataFrame = df
        if cols is not None:
            if not len(Index(cols).intersection(df.columns)):
                raise KeyError('passes columns are not ALL present dataframe')
            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                raise KeyError("Not all names specified in 'columns' are found")
            self.df = df.reindex(columns=cols)
        self.columns: Any = self.df.columns
        self.float_format: Optional[str] = float_format