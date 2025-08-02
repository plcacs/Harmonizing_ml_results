"""
Utilities for conversion to writer-agnostic Excel representation.
"""
from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import itertools
import re
from typing import TYPE_CHECKING, Any, cast
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
    from pandas._typing import (
        ExcelWriterMergeCells,
        FilePath,
        IndexLabel,
        StorageOptions,
        WriteExcelBuffer,
    )
    from pandas import ExcelWriter


class ExcelCell:
    __fields__ = 'row', 'col', 'val', 'style', 'mergestart', 'mergeend'
    __slots__ = __fields__

    def __init__(
        self,
        row: int,
        col: int,
        val: Any,
        style: Any = None,
        mergestart: Any = None,
        mergeend: Any = None,
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
        style: Any,
        css_styles: Any,
        css_row: int,
        css_col: int,
        css_converter: Callable[[frozenset[tuple[str, str]]], dict],
        **kwargs: Any,
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
    NAMED_COLORS: Mapping[str, str] = CSS4_COLORS
    VERTICAL_MAP: Mapping[str, str] = {
        'top': 'top',
        'text-top': 'top',
        'middle': 'center',
        'baseline': 'bottom',
        'bottom': 'bottom',
        'text-bottom': 'bottom',
    }
    BOLD_MAP: Mapping[str, bool] = {
        'bold': True,
        'bolder': True,
        '600': True,
        '700': True,
        '800': True,
        '900': True,
        'normal': False,
        'lighter': False,
        '100': False,
        '200': False,
        '300': False,
        '400': False,
        '500': False,
    }
    ITALIC_MAP: Mapping[str, bool] = {
        'normal': False,
        'italic': True,
        'oblique': True,
    }
    FAMILY_MAP: Mapping[str, int] = {
        'serif': 1,
        'sans-serif': 2,
        'cursive': 4,
        'fantasy': 5,
    }
    BORDER_STYLE_MAP: Mapping[str, str] = {
        style.lower(): style for style in [
            'dashed',
            'mediumDashDot',
            'dashDotDot',
            'hair',
            'dotted',
            'mediumDashDotDot',
            'double',
            'dashDot',
            'slantDashDot',
            'mediumDashed',
        ]
    }

    def __init__(self, inherited: str | None = None) -> None:
        if inherited is not None:
            self.inherited = self.compute_css(inherited)
        else:
            self.inherited = None
        self._call_cached: Callable[[frozenset[tuple[str, str]] | str], dict] = functools.cache(
            self.func_icykwmqj
        )
    compute_css: CSSResolver = CSSResolver()

    def __call__(self, declarations: str | frozenset[tuple[str, str]]) -> dict:
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

    def func_icykwmqj(self, declarations: str | frozenset[tuple[str, str]]) -> dict:
        properties = self.compute_css(declarations, self.inherited)
        return self.func_1fzwl654(properties)

    def func_1fzwl654(self, props: dict) -> dict:
        out: dict = {
            'alignment': self.func_513nlezs(props),
            'border': self.func_nowr9ijq(props),
            'fill': self.func_1ct7bjj2(props),
            'font': self.func_pg3fobvx(props),
            'number_format': self.func_40uk97ed(props),
        }

        def func_mkjj7xgu(d: dict) -> None:
            """Remove key where value is None, through nested dicts"""
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    func_mkjj7xgu(v)
                    if not v:
                        del d[k]

        func_mkjj7xgu(out)
        return out

    def func_513nlezs(self, props: dict) -> dict:
        return {
            'horizontal': props.get('text-align'),
            'vertical': self.func_vw7rptci(props),
            'wrap_text': self.func_c80cn3ml(props),
        }

    def func_vw7rptci(self, props: dict) -> str | None:
        vertical_align = props.get('vertical-align')
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)
        return None

    def func_c80cn3ml(self, props: dict) -> bool | None:
        if props.get('white-space') is None:
            return None
        return bool(props['white-space'] not in ('nowrap', 'pre', 'pre-line'))

    def func_nowr9ijq(self, props: dict) -> dict[str, dict[str, Any]] | None:
        return {
            side: {
                'style': self.func_blzsbztb(
                    props.get(f'border-{side}-style'),
                    props.get(f'border-{side}-width'),
                    self.func_mk2ecwzn(props.get(f'border-{side}-color')),
                ),
                'color': self.func_mk2ecwzn(props.get(f'border-{side}-color')),
            }
            for side in ['top', 'right', 'bottom', 'left']
        }

    def func_blzsbztb(
        self, style: str | None, width: str | None, color: str | None
    ) -> str | None:
        if width is None and style is None and color is None:
            return None
        if width is None and style is None:
            return 'none'
        if style in ('none', 'hidden'):
            return 'none'
        width_name = self.func_4d95rcmz(width)
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
            warnings.warn(
                f'Unhandled border style format: {style!r}',
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return 'none'

    def func_4d95rcmz(self, width_input: str | None) -> str | None:
        width = self.func_2bwl7kzu(width_input)
        if width < 1e-05:
            return None
        elif width < 1.3:
            return 'thin'
        elif width < 2.8:
            return 'medium'
        return 'thick'

    def func_2bwl7kzu(self, width: str | None) -> float:
        if width is None:
            width = '2pt'
        return self.func_w405qnch(width)

    def func_w405qnch(self, pt_string: str) -> float:
        assert pt_string.endswith('pt')
        return float(pt_string.rstrip('pt'))

    def func_1ct7bjj2(self, props: dict) -> dict[str, Any] | None:
        fill_color = props.get('background-color')
        if fill_color not in (None, 'transparent', 'none'):
            return {
                'fgColor': self.func_mk2ecwzn(fill_color),
                'patternType': 'solid',
            }
        return None

    def func_40uk97ed(self, props: dict) -> dict[str, Any] | None:
        fc = props.get('number-format')
        fc = fc.replace('§', ';') if isinstance(fc, str) else fc
        return {'format_code': fc}

    def func_pg3fobvx(self, props: dict) -> dict[str, Any] | None:
        font_names = self.func_1vyq98w5(props)
        decoration = self.func_0fqkvxra(props)
        return {
            'name': font_names[0] if font_names else None,
            'family': self.func_llqyctcm(font_names),
            'size': self.func_lc0lik1u(props),
            'bold': self.func_tfeytb84(props),
            'italic': self.func_7nfnieeq(props),
            'underline': 'single' if 'underline' in decoration else None,
            'strike': 'line-through' in decoration or None,
            'color': self.func_mk2ecwzn(props.get('color')),
            'shadow': self.func_9u8m5w5i(props),
        }

    def func_tfeytb84(self, props: dict) -> bool | None:
        weight = props.get('font-weight')
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def func_7nfnieeq(self, props: dict) -> bool | None:
        font_style = props.get('font-style')
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def func_0fqkvxra(self, props: dict) -> tuple[str, ...]:
        decoration = props.get('text-decoration')
        if decoration is not None:
            return tuple(decoration.split())
        else:
            return ()

    def func_f1k5wsfb(self, decoration: tuple[str, ...]) -> str | None:
        if 'underline' in decoration:
            return 'single'
        return None

    def func_9u8m5w5i(self, props: dict) -> bool | None:
        if 'text-shadow' in props:
            return bool(re.search('^[^#(]*[1-9]', props['text-shadow']))
        return None

    def func_1vyq98w5(self, props: dict) -> list[str]:
        font_names_tmp = re.findall(
            r"""(?x)
                (
                    "(?:[^"]|\\")+" 
                    |
                    '(?:[^']|\\')+' 
                    |
                    [^'",]+
                )(?=,|\s*$)
            """,
            props.get('font-family', ''),
        )
        font_names: list[str] = []
        for name in font_names_tmp:
            if name.startswith('"'):
                name = name[1:-1].replace('\\"', '"')
            elif name.startswith("'"):
                name = name[1:-1].replace("\\'", "'")
            else:
                name = name.strip()
            if name:
                font_names.append(name)
        return font_names

    def func_lc0lik1u(self, props: dict) -> float | None:
        size = props.get('font-size')
        if size is None:
            return None
        return self.func_vaf0z87k(size)

    def func_llqyctcm(self, font_names: list[str]) -> int | None:
        family: int | None = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break
        return family

    def func_mk2ecwzn(self, val: Any) -> str | None:
        if val is None:
            return None
        if self.func_gk1mjh51(val):
            return self.func_sdljejuu(val)
        try:
            return self.NAMED_COLORS[val]
        except KeyError:
            warnings.warn(
                f'Unhandled color format: {val!r}',
                CSSWarning,
                stacklevel=find_stack_level(),
            )
        return None

    def func_gk1mjh51(self, color_string: str) -> bool:
        return color_string.startswith('#')

    def func_sdljejuu(self, color_string: str) -> str:
        code = color_string.lstrip('#')
        if self.func_vaf0z87k(color_string):
            code = (code[0] * 2 + code[1] * 2 + code[2] * 2).upper()
        else:
            code = code.upper()
        return code

    def func_vaf0z87k(self, color_string: str) -> bool:
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
    max_rows: int = 2 ** 20
    max_cols: int = 2 ** 14

    def __init__(
        self,
        df: DataFrame | Any,  # Styler
        na_rep: str = '',
        float_format: str | None = None,
        cols: Sequence[Any] | None = None,
        header: bool | Sequence[str] = True,
        index: bool = True,
        index_label: str | Sequence[str] | None = None,
        merge_cells: bool | str = False,
        inf_rep: str = 'inf',
        style_converter: Callable[[frozenset[tuple[str, str]]], dict] | None = None,
    ) -> None:
        self.rowcounter: int = 0
        self.na_rep: str = na_rep
        if not isinstance(df, DataFrame):
            self.styler: Any = df
            self.styler._compute()
            df = df.data
            if style_converter is None:
                style_converter = CSSToExcelConverter()
            self.style_converter: Callable[[frozenset[tuple[str, str]]], dict] | None = style_converter
        else:
            self.styler = None
            self.style_converter = None
        self.df: DataFrame = df
        if cols is not None:
            if not len(Index(cols).intersection(df.columns)):
                raise KeyError('passed columns are not ALL present dataframe')
            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                raise KeyError("Not all names specified in 'columns' are found")
            self.df = df.reindex(columns=cols)
        self.columns: Index = self.df.columns
        self.float_format: str | None = float_format
        self.index: bool = index
        self.index_label: str | Sequence[str] | None = index_label
        self.header: bool | Sequence[str] = header
        if not isinstance(merge_cells, bool) and merge_cells != 'columns':
            raise ValueError(f'Unexpected value for merge_cells={merge_cells!r}.')
        self.merge_cells: bool | str = merge_cells
        self.inf_rep: str = inf_rep

    def func_180xztqr(self, val: Any) -> Any:
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
            raise ValueError(
                'Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.'
            )
        return val

    def func_234zp6bb(self) -> Iterable[ExcelCell]:
        if self.columns.nlevels > 1:
            if not self.index:
                raise NotImplementedError(
                    "Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented."
                )
        if not (self.func_z5z1zg7h or self.header):
            return
        columns = self.columns
        merge_columns: bool = self.merge_cells in {True, 'columns'}
        level_strs = columns._format_multi(sparsify=merge_columns, include_names=False)
        level_lengths = get_level_lengths(level_strs)
        coloffset: int = 0
        lnum: int = 0
        if self.index and isinstance(self.df.index, MultiIndex):
            coloffset = self.df.index.nlevels - 1
        for lnum, name in enumerate(columns.names):
            yield ExcelCell(row=lnum, col=coloffset, val=name, style=None)
        for lnum, (spans, levels, level_codes) in enumerate(zip(level_lengths, columns.levels, columns.codes)):
            values = levels.take(level_codes)
            for i, span_val in spans.items():
                mergestart: int | None = None
                mergeend: int | None = None
                if merge_columns and span_val > 1:
                    mergestart = lnum
                    mergeend = coloffset + i + span_val
                yield CssExcelCell(
                    row=lnum,
                    col=coloffset + i + 1,
                    val=values[i],
                    style=None,
                    css_styles=getattr(self.styler, 'ctx_columns', None),
                    css_row=lnum,
                    css_col=i,
                    css_converter=self.style_converter,
                    mergestart=mergestart,
                    mergeend=mergeend,
                )
        self.rowcounter = lnum

    def func_xyjbim3a(self) -> Iterable[CssExcelCell]:
        if self.func_z5z1zg7h or self.header:
            coloffset: int = 0
            if self.index:
                coloffset = 1
                if isinstance(self.df.index, MultiIndex):
                    coloffset = len(self.df.index.names)
            colnames = self.columns
            if self.func_z5z1zg7h:
                self.header = cast(Sequence[str], self.header)
                if len(self.header) != len(self.columns):
                    raise ValueError(
                        f'Writing {len(self.columns)} cols but got {len(self.header)} aliases'
                    )
                colnames = self.header
            for colindex, colname in enumerate(colnames):
                yield CssExcelCell(
                    row=self.rowcounter,
                    col=colindex + coloffset,
                    val=colname,
                    style=None,
                    css_styles=getattr(self.styler, 'ctx_columns', None),
                    css_row=0,
                    css_col=colindex,
                    css_converter=self.style_converter,
                )

    def func_svkk9a45(self) -> Iterable[ExcelCell | CssExcelCell]:
        if isinstance(self.columns, MultiIndex):
            gen = self.func_234zp6bb()
        else:
            gen = self.func_xyjbim3a()
        gen2: Iterable[ExcelCell] = ()
        if self.df.index.names:
            row: list[str] = [x if x is not None else '' for x in self.df.index.names] + [''] * len(self.columns)
            if all(x != '' for x in row):
                gen2 = (
                    ExcelCell(self.rowcounter, colindex, val, None)
                    for colindex, val in enumerate(row)
                )
                self.rowcounter += 1
        return itertools.chain(gen, gen2)

    def func_85yve4em(self) -> Iterable[CssExcelCell]:
        if isinstance(self.df.index, MultiIndex):
            return self._format_hierarchical_rows()
        else:
            return self._format_regular_rows()

    def func_3b1zjjhx(self) -> Iterable[ExcelCell | CssExcelCell]:
        if self.func_z5z1zg7h or self.header:
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
                yield CssExcelCell(
                    row=self.rowcounter + idx,
                    col=0,
                    val=idxval,
                    style=None,
                    css_styles=getattr(self.styler, 'ctx_index', None),
                    css_row=idx,
                    css_col=0,
                    css_converter=self.style_converter,
                )
            coloffset: int = 1
        else:
            coloffset = 0
        yield from self.func_59fehsbz()

    def func_59fehsbz(self) -> Iterable[ExcelCell | CssExcelCell]:
        if self.func_z5z1zg7h or self.header:
            self.rowcounter += 1
        gcolidx: int = 0
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
                level_lengths = get_level_lengths(level_strs)
                for spans, levels, level_codes in zip(level_lengths, self.df.index.levels, self.df.index.codes):
                    values = levels.take(level_codes, allow_fill=levels._can_hold_na, fill_value=levels._na_value)
                    if isinstance(values[0], Period):
                        values = values.to_timestamp()
                    for i, span_val in spans.items():
                        mergestart: int | None = None
                        mergeend: int | None = None
                        if span_val > 1:
                            mergestart = self.rowcounter + i + span_val - 1
                            mergeend = gcolidx
                        yield CssExcelCell(
                            row=self.rowcounter + i,
                            col=gcolidx,
                            val=values[i],
                            style=None,
                            css_styles=getattr(self.styler, 'ctx_index', None),
                            css_row=i,
                            css_col=gcolidx,
                            css_converter=self.style_converter,
                            mergestart=mergestart,
                            mergeend=mergeend,
                        )
                    gcolidx += 1
            else:
                for indexcolvals in zip(*self.df.index):
                    for idx, indexcolval in enumerate(indexcolvals):
                        if isinstance(indexcolval, Period):
                            indexcolval = indexcolval.to_timestamp()
                        yield CssExcelCell(
                            row=self.rowcounter + idx,
                            col=gcolidx,
                            val=indexcolval,
                            style=None,
                            css_styles=getattr(self.styler, 'ctx_index', None),
                            css_row=idx,
                            css_col=gcolidx,
                            css_converter=self.style_converter,
                        )
                    gcolidx += 1
        yield from self._generate_body(gcolidx)

    @property
    def func_z5z1zg7h(self) -> bool:
        """Whether the aliases for column names are present."""
        return is_list_like(self.header)

    def func_i1fpbfxk(self, coloffset: int) -> Iterable[CssExcelCell]:
        for colidx in range(len(self.columns)):
            series = self.df.iloc[:, colidx]
            for i, val in enumerate(series):
                yield CssExcelCell(
                    row=self.rowcounter + i,
                    col=colidx + coloffset,
                    val=val,
                    style=None,
                    css_styles=getattr(self.styler, 'ctx', None),
                    css_row=i,
                    css_col=colidx,
                    css_converter=self.style_converter,
                )

    def func_ptnsim1a(self) -> Iterable[ExcelCell | CssExcelCell]:
        for cell in itertools.chain(self.func_234zp6bb(), self.func_i1fpbfxk(0)):
            cell.val = self.func_180xztqr(cell.val)
            yield cell

    @doc(storage_options=_shared_docs['storage_options'])
    def func_xtm0t17q(
        self,
        writer: ExcelWriter | FilePath,
        sheet_name: str = 'Sheet1',
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
        engine: str | None = None,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame
        startrow :
            upper left cell row to dump data frame
        startcol :
            upper left cell column to dump data frame
        freeze_panes : tuple of integer (length 2), default None
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen
        engine : string, default None
            write engine to use if writer is a path - you can also set this
            via the options ``io.excel.xlsx.writer``,
            or ``io.excel.xlsm.writer``.

        {storage_options}

        engine_kwargs: dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        from pandas.io.excel import ExcelWriter
        num_rows: int
        num_cols: int
        num_rows, num_cols = self.df.shape
        if num_rows > self.max_rows or num_cols > self.max_cols:
            raise ValueError(
                f'This sheet is too large! Your sheet size is: {num_rows}, {num_cols} Max sheet size is: {self.max_rows}, {self.max_cols}'
            )
        if engine_kwargs is None:
            engine_kwargs = {}
        formatted_cells: Iterable[ExcelCell | CssExcelCell] = self.func_ptnsim1a()
        if isinstance(writer, ExcelWriter):
            need_save: bool = False
        else:
            writer = ExcelWriter(
                writer,
                engine=engine,
                storage_options=storage_options,
                engine_kwargs=engine_kwargs,
            )
            need_save = True
        try:
            writer._write_cells(
                formatted_cells,
                sheet_name,
                startrow=startrow,
                startcol=startcol,
                freeze_panes=freeze_panes,
            )
        finally:
            if need_save:
                writer.close()
