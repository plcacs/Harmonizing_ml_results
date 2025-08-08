"""
Utilities for conversion to writer-agnostic Excel representation.
"""
from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import itertools
import re
from typing import TYPE_CHECKING, Any, cast, Optional, Union, Tuple, List, Dict, FrozenSet
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
    __fields__ = 'row', 'col', 'val', 'style', 'mergestart', 'mergeend'
    __slots__ = __fields__

    def __init__(
        self,
        row: int,
        col: int,
        val: Any,
        style: Optional[Dict[str, Any]] = None,
        mergestart: Optional[int] = None,
        mergeend: Optional[int] = None
    ) -> None:
        self.row = row
        self.col = col
        self.val = val
        self.style = style
        self.mergestart = mergestart
        self.mergeend = mergeend


class CssExcelCell(ExcelCell):
    def __init__(
        self,
        row: int,
        col: int,
        val: Any,
        style: Optional[Dict[str, Any]],
        css_styles: Optional[Dict[Tuple[int, int], List[Tuple[str, str]]],
        css_row: int,
        css_col: int,
        css_converter: Optional[Callable[[FrozenSet[Tuple[str, str]]], Dict[str, Any]],
        **kwargs: Any
    ) -> None:
        if css_styles and css_converter:
            declaration_dict = {prop.lower(): val for prop, val in css_styles[css_row, css_col]}
            unique_declarations = frozenset(declaration_dict.items())
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
    NAMED_COLORS = CSS4_COLORS
    VERTICAL_MAP = {'top': 'top', 'text-top': 'top', 'middle': 'center',
        'baseline': 'bottom', 'bottom': 'bottom', 'text-bottom': 'bottom'}
    BOLD_MAP = {'bold': True, 'bolder': True, '600': True, '700': True,
        '800': True, '900': True, 'normal': False, 'lighter': False, '100':
        False, '200': False, '300': False, '400': False, '500': False}
    ITALIC_MAP = {'normal': False, 'italic': True, 'oblique': True}
    FAMILY_MAP = {'serif': 1, 'sans-serif': 2, 'cursive': 4, 'fantasy': 5}
    BORDER_STYLE_MAP = {style.lower(): style for style in ['dashed',
        'mediumDashDot', 'dashDotDot', 'hair', 'dotted', 'mediumDashDotDot',
        'double', 'dashDot', 'slantDashDot', 'mediumDashed']}

    def __init__(self, inherited: Optional[str] = None) -> None:
        if inherited is not None:
            self.inherited = self.compute_css(inherited)
        else:
            self.inherited = None
        self._call_cached = functools.cache(self._call_uncached)
    compute_css = CSSResolver()

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

    def func_rmlgtmg5(self, declarations: Union[str, FrozenSet[Tuple[str, str]]]) -> Dict[str, Any]:
        properties = self.compute_css(declarations, self.inherited)
        return self.build_xlstyle(properties)

    def func_sgriejwf(self, props: Dict[str, Any]) -> Dict[str, Any]:
        out = {'alignment': self.build_alignment(props), 'border': self.
            build_border(props), 'fill': self.build_fill(props), 'font':
            self.build_font(props), 'number_format': self.
            build_number_format(props)}

        def func_5b7ijeux(d: Dict[str, Any]) -> None:
            """Remove key where value is None, through nested dicts"""
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    func_5b7ijeux(v)
                    if not v:
                        del d[k]
        func_5b7ijeux(out)
        return out

    def func_jbytaw2g(self, props: Dict[str, Any]) -> Dict[str, Optional[str]]:
        return {'horizontal': props.get('text-align'), 'vertical': self.
            _get_vertical_alignment(props), 'wrap_text': self.
            _get_is_wrap_text(props)}

    def func_vbkjt72c(self, props: Dict[str, Any]) -> Optional[str]:
        vertical_align = props.get('vertical-align')
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)
        return None

    def func_1b0obxu1(self, props: Dict[str, Any]) -> Optional[bool]:
        if props.get('white-space') is None:
            return None
        return bool(props['white-space'] not in ('nowrap', 'pre', 'pre-line'))

    def func_mkdh0hd6(self, props: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {side: {'style': self._border_style(props.get(
            f'border-{side}-style'), props.get(f'border-{side}-width'),
            'color': self.color_to_excel(props.get(f'border-{side}-color'))
            } for side in ['top', 'right', 'bottom', 'left']}

    def func_va8m65mn(self, style: Optional[str], width: Optional[str], color: Optional[str]) -> Optional[str]:
        if width is None and style is None and color is None:
            return None
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
            warnings.warn(f'Unhandled border style format: {style!r}',
                CSSWarning, stacklevel=find_stack_level())
            return 'none'

    def func_adnrko8x(self, width_input: Optional[str]) -> Optional[str]:
        width = self._width_to_float(width_input)
        if width < 1e-05:
            return None
        elif width < 1.3:
            return 'thin'
        elif width < 2.8:
            return 'medium'
        return 'thick'

    def func_zz0d9sq9(self, width: Optional[str]) -> float:
        if width is None:
            width = '2pt'
        return self._pt_to_float(width)

    def func_soqbczyw(self, pt_string: str) -> float:
        assert pt_string.endswith('pt')
        return float(pt_string.rstrip('pt'))

    def func_v0uilk4u(self, props: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fill_color = props.get('background-color')
        if fill_color not in (None, 'transparent', 'none'):
            return {'fgColor': self.color_to_excel(fill_color),
                'patternType': 'solid'}
        return None

    def func_iscntdlo(self, props: Dict[str, Any]) -> Dict[str, Any]:
        fc = props.get('number-format')
        fc = fc.replace('§', ';') if isinstance(fc, str) else fc
        return {'format_code': fc}

    def func_c7uhyu66(self, props: Dict[str, Any]) -> Dict[str, Any]:
        font_names = self._get_font_names(props)
        decoration = self._get_decoration(props)
        return {'name': font_names[0] if font_names else None, 'family':
            self._select_font_family(font_names), 'size': self.
            _get_font_size(props), 'bold': self._get_is_bold(props),
            'italic': self._get_is_italic(props), 'underline': 'single' if 
            'underline' in decoration else None, 'strike': 'line-through' in
            decoration or None, 'color': self.color_to_excel(props.get(
            'color')), 'shadow': self._get_shadow(props)}

    def func_acxeoyi8(self, props: Dict[str, Any]) -> Optional[bool]:
        weight = props.get('font-weight')
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def func_a57kxoqy(self, props: Dict[str, Any]) -> Optional[bool]:
        font_style = props.get('font-style')
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def func_1ttvn4w1(self, props: Dict[str, Any]) -> Tuple[str, ...]:
        decoration = props.get('text-decoration')
        if decoration is not None:
            return tuple(decoration.split())
        else:
            return ()

    def func_7vgreki2(self, decoration: Tuple[str, ...]) -> Optional[str]:
        if 'underline' in decoration:
            return 'single'
        return None

    def func_1x54iaqz(self, props: Dict[str, Any]) -> Optional[bool]:
        if 'text-shadow' in props:
            return bool(re.search('^[^#(]*[1-9]', props['text-shadow']))
        return None

    def func_u74b97m2(self, props: Dict[str, Any]) -> List[str]:
        font_names_tmp = re.findall(
            """(?x)
            (
            "(?:[^"]|\\\\")+"
            |
            '(?:[^']|\\\\')+'
            |
            [^'",]+
            )(?=,|\\s*$)
        """
            , props.get('font-family', ''))
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

    def func_70jra572(self, props: Dict[str, Any]) -> Optional[float]:
        size = props.get('font-size')
        if size is None:
            return size
        return self._pt_to_float(size)

    def func_p1oupn4n(self, font_names: List[str]) -> Optional[int]:
        family = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break
        return family

    def func_6ogj28oz(self, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        if self._is_hex_color(val):
            return self._convert_hex_to_excel(val)
        try:
            return self.NAMED_COLORS[val]
        except KeyError:
            warnings.warn(f'Unhandled color format: {val!r}', CSSWarning,
                stacklevel=find_stack_level())
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
        """Check if color code is shorthand.

        #FFF is a shorthand as opposed to full #FFFFFF.
        """
        code = color_string.lstrip('#')
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
    max_rows = 2 ** 20
    max_cols = 2 ** 14

    def __init__(
        self,
        df: Union[DataFrame, Any],  # Any is for Styler
        na_rep: str = '',
        float_format: Optional[str] = None,
        cols: Optional[Sequence[Hashable]] = None,
        header: Union[bool, Sequence[str]] = True,
        index: bool = True,
        index_label: Optional[Union[str, Sequence[str]]] = None,
        merge_cells: Union[bool, str] = False,
        inf_rep: str = 'inf',
        style_converter: Optional[Callable[[FrozenSet[Tuple[str, str]]], Dict[str, Any]]] = None
    ) -> None:
        self.rowcounter = 0
        self.na_rep = na_rep
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
        self.df = df
        if cols is not None:
            if not len(Index(cols).intersection(df.columns)):
                raise KeyError('passes columns are not ALL present dataframe')
            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                raise KeyError("Not all names specified in 'columns' are found")
            self.df = df.reindex(columns=cols)
        self.columns = self.df.columns
        self.float_format = float_format
        self.index = index
        self.index_label = index_label
        self.header = header
        if not isinstance(merge_cells, bool) and merge_cells != 'columns':
            raise ValueError(f'Unexpected value for merge_cells={merge_cells!r