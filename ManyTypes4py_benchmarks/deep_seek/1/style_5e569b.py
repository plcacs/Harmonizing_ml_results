from __future__ import annotations
import copy
from functools import partial
import operator
import textwrap
from typing import (
    TYPE_CHECKING, 
    overload, 
    Any, 
    Optional, 
    Union, 
    List, 
    Dict, 
    Tuple, 
    Callable, 
    Sequence, 
    Hashable, 
    Literal, 
    cast
)
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import Substitution, doc
import pandas as pd
from pandas import IndexSlice, RangeIndex
import pandas.core.common as com
from pandas.core.frame import DataFrame, Series
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer

jinja2 = import_optional_dependency('jinja2', extra='DataFrame.style requires jinja2.')
from pandas.io.formats.style_render import (
    CSSProperties, 
    CSSStyles, 
    ExtFormatter, 
    StylerRenderer, 
    Subset, 
    Tooltips, 
    format_table_styles, 
    maybe_convert_css_to_tuples, 
    non_reducing_slice, 
    refactor_levels
)

if TYPE_CHECKING:
    from collections.abc import Callable as CallableABC, Hashable as HashableABC, Sequence as SequenceABC
    from matplotlib.colors import Colormap
    from pandas._typing import (
        Any as AnyType, 
        Axis, 
        AxisInt, 
        Concatenate, 
        ExcelWriterMergeCells, 
        FilePath, 
        IndexLabel, 
        IntervalClosedType, 
        Level, 
        P, 
        QuantileInterpolation, 
        Scalar, 
        Self, 
        StorageOptions, 
        T, 
        WriteBuffer, 
        WriteExcelBuffer
    )
    from pandas import ExcelWriter

subset_args = 'subset : label, array-like, IndexSlice, optional\n            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input\n            or single key, to `DataFrame.loc[:, <subset>]` where the columns are\n            prioritised, to limit ``data`` to *before* applying the function.'
properties_args = 'props : str, default None\n           CSS properties to use for highlighting. If ``props`` is given, ``color``\n           is not used.'
coloring_args = "color : str, default '{default}'\n           Background color to use for highlighting."
buffering_args = 'buf : str, path object, file-like object, optional\n         String, path object (implementing ``os.PathLike[str]``), or file-like\n         object implementing a string ``write()`` function. If ``None``, the result is\n         returned as a string.'
encoding_args = 'encoding : str, optional\n              Character encoding setting for file output (and meta tags if available).\n              Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8".'

class Styler(StylerRenderer):
    def __init__(
        self, 
        data: Union[DataFrame, Series], 
        precision: Optional[int] = None, 
        table_styles: Optional[List[Dict[str, Any]]] = None, 
        uuid: Optional[str] = None, 
        caption: Optional[Union[str, Tuple[str, str]]] = None, 
        table_attributes: Optional[str] = None, 
        cell_ids: bool = True, 
        na_rep: Optional[str] = None, 
        uuid_len: int = 5, 
        decimal: Optional[str] = None, 
        thousands: Optional[str] = None, 
        escape: Optional[str] = None, 
        formatter: Optional[Union[str, Callable, Dict]] = None
    ) -> None:
        super().__init__(
            data=data, 
            uuid=uuid, 
            uuid_len=uuid_len, 
            table_styles=table_styles, 
            table_attributes=table_attributes, 
            caption=caption, 
            cell_ids=cell_ids, 
            precision=precision
        )
        thousands = thousands or get_option('styler.format.thousands')
        decimal = decimal or get_option('styler.format.decimal')
        na_rep = na_rep or get_option('styler.format.na_rep')
        escape = escape or get_option('styler.format.escape')
        formatter = formatter or get_option('styler.format.formatter')
        self.format(
            formatter=formatter, 
            precision=precision, 
            na_rep=na_rep, 
            escape=escape, 
            decimal=decimal, 
            thousands=thousands
        )

    def concat(self, other: Styler) -> Styler:
        if not isinstance(other, Styler):
            raise TypeError('`other` must be of type `Styler`')
        if not self.data.columns.equals(other.data.columns):
            raise ValueError('`other.data` must have same columns as `Styler.data`')
        if not self.data.index.nlevels == other.data.index.nlevels:
            raise ValueError('number of index levels must be same in `other` as in `Styler`. See documentation for suggestions.')
        self.concatenated.append(other)
        return self

    def _repr_html_(self) -> Optional[str]:
        if get_option('styler.render.repr') == 'html':
            return self.to_html()
        return None

    def _repr_latex_(self) -> Optional[str]:
        if get_option('styler.render.repr') == 'latex':
            return self.to_latex()
        return None

    def set_tooltips(
        self, 
        ttips: DataFrame, 
        props: Optional[Union[str, List[Tuple[str, str]]]] = None, 
        css_class: Optional[str] = None, 
        as_title_attribute: bool = False
    ) -> Styler:
        if not self.cell_ids:
            raise NotImplementedError("Tooltips can only render with 'cell_ids' is True.")
        if not ttips.index.is_unique or not ttips.columns.is_unique:
            raise KeyError('Tooltips render only if `ttips` has unique index and columns.')
        if self.tooltips is None:
            self.tooltips = Tooltips()
        self.tooltips.tt_data = ttips
        if not as_title_attribute:
            if props:
                self.tooltips.class_properties = props
            if css_class:
                self.tooltips.class_name = css_class
        else:
            self.tooltips.as_title_attribute = as_title_attribute
        return self

    @doc(NDFrame.to_excel, klass='Styler', storage_options=_shared_docs['storage_options'], storage_options_versionadded='1.5.0', encoding_parameter=textwrap.dedent('        encoding : str or None, default None\n            Unused parameter, present for compatibility.\n        '), verbose_parameter=textwrap.dedent('        verbose : str, default True\n            Optional unused parameter, present for compatibility.\n        '), extra_parameters='')
    def to_excel(
        self, 
        excel_writer: Union[FilePath, WriteExcelBuffer, ExcelWriter], 
        sheet_name: str = 'Sheet1', 
        na_rep: str = '', 
        float_format: Optional[str] = None, 
        columns: Optional[Sequence[Hashable]] = None, 
        header: Union[bool, Sequence[str]] = True, 
        index: bool = True, 
        index_label: Optional[Union[Hashable, Sequence[Hashable]]] = None, 
        startrow: int = 0, 
        startcol: int = 0, 
        engine: Optional[str] = None, 
        merge_cells: bool = True, 
        encoding: Optional[str] = None, 
        inf_rep: str = 'inf', 
        verbose: bool = True, 
        freeze_panes: Optional[Tuple[int, int]]] = None, 
        storage_options: Optional[Dict[str, Any]]] = None
    ) -> None:
        from pandas.io.formats.excel import ExcelFormatter
        formatter = ExcelFormatter(
            self, 
            na_rep=na_rep, 
            cols=columns, 
            header=header, 
            float_format=float_format, 
            index=index, 
            index_label=index_label, 
            merge_cells=merge_cells, 
            inf_rep=inf_rep
        )
        formatter.write(
            excel_writer, 
            sheet_name=sheet_name, 
            startrow=startrow, 
            startcol=startcol, 
            freeze_panes=freeze_panes, 
            engine=engine, 
            storage_options=storage_options
        )

    @overload
    def to_latex(
        self, 
        buf: Union[str, WriteBuffer[str]], 
        *, 
        column_format: Optional[str] = ..., 
        position: Optional[str] = ..., 
        position_float: Optional[Literal["centering", "raggedleft", "raggedright"]] = ..., 
        hrules: Optional[bool] = ..., 
        clines: Optional[str] = ..., 
        label: Optional[str] = ..., 
        caption: Optional[Union[str, Tuple[str, str]]] = ..., 
        sparse_index: Optional[bool] = ..., 
        sparse_columns: Optional[bool] = ..., 
        multirow_align: Optional[str] = ..., 
        multicol_align: Optional[str] = ..., 
        siunitx: bool = ..., 
        environment: Optional[str] = ..., 
        encoding: Optional[str] = ..., 
        convert_css: bool = ...
    ) -> None:
        ...

    @overload
    def to_latex(
        self, 
        buf: None = ..., 
        *, 
        column_format: Optional[str] = ..., 
        position: Optional[str] = ..., 
        position_float: Optional[Literal["centering", "raggedleft", "raggedright"]] = ..., 
        hrules: Optional[bool] = ..., 
        clines: Optional[str] = ..., 
        label: Optional[str] = ..., 
        caption: Optional[Union[str, Tuple[str, str]]] = ..., 
        sparse_index: Optional[bool] = ..., 
        sparse_columns: Optional[bool] = ..., 
        multirow_align: Optional[str] = ..., 
        multicol_align: Optional[str] = ..., 
        siunitx: bool = ..., 
        environment: Optional[str] = ..., 
        encoding: Optional[str] = ..., 
        convert_css: bool = ...
    ) -> str:
        ...

    def to_latex(
        self, 
        buf: Optional[Union[str, WriteBuffer[str]]] = None, 
        *, 
        column_format: Optional[str] = None, 
        position: Optional[str] = None, 
        position_float: Optional[Literal["centering", "raggedleft", "raggedright"]] = None, 
        hrules: Optional[bool] = None, 
        clines: Optional[str] = None, 
        label: Optional[str] = None, 
        caption: Optional[Union[str, Tuple[str, str]]] = None, 
        sparse_index: Optional[bool] = None, 
        sparse_columns: Optional[bool] = None, 
        multirow_align: Optional[str] = None, 
        multicol_align: Optional[str] = None, 
        siunitx: bool = False, 
        environment: Optional[str] = None, 
        encoding: Optional[str] = None, 
        convert_css: bool = False
    ) -> Optional[str]:
        obj = self._copy(deepcopy=True)
        table_selectors = [style['selector'] for style in self.table_styles] if self.table_styles is not None else []
        if column_format is not None:
            obj.set_table_styles([{'selector': 'column_format', 'props': f':{column_format}'}], overwrite=False)
        elif 'column_format' in table_selectors:
            pass
        else:
            _original_columns = self.data.columns
            self.data.columns = RangeIndex(stop=len(self.data.columns))
            numeric_cols = self.data._get_numeric_data().columns.to_list()
            self.data.columns = _original_columns
            column_format = ''
            for level in range(self.index.nlevels):
                column_format += '' if self.hide_index_[level] else 'l'
            for ci, _ in enumerate(self.data.columns):
                if ci not in self.hidden_columns:
                    column_format += ('r' if not siunitx else 'S') if ci in numeric_cols else 'l'
            obj.set_table_styles([{'selector': 'column_format', 'props': f':{column_format}'}], overwrite=False)
        if position:
            obj.set_table_styles([{'selector': 'position', 'props': f':{position}'}], overwrite=False)
        if position_float:
            if environment == 'longtable':
                raise ValueError("`position_float` cannot be used in 'longtable' `environment`")
            if position_float not in ['raggedright', 'raggedleft', 'centering']:
                raise ValueError(f"`position_float` should be one of 'raggedright', 'raggedleft', 'centering', got: '{position_float}'")
            obj.set_table_styles([{'selector': 'position_float', 'props': f':{position_float}'}], overwrite=False)
        hrules = get_option('styler.latex.hrules') if hrules is None else hrules
        if hrules:
            obj.set_table_styles([{'selector': 'toprule', 'props': ':toprule'}, {'selector': 'midrule', 'props': ':midrule'}, {'selector': 'bottomrule', 'props': ':bottomrule'}], overwrite=False)
        if label:
            obj.set_table_styles([{'selector': 'label', 'props': f':{{{label.replace(":", "§")}}}'}], overwrite=False)
        if caption:
            obj.set_caption(caption)
        if sparse_index is None:
            sparse_index = get_option('styler.sparse.index')
        if sparse_columns is None:
            sparse_columns = get_option('styler.sparse.columns')
        environment = environment or get_option('styler.latex.environment')
        multicol_align = multicol_align or get_option('styler.latex.multicol_align')
        multirow_align = multirow_align or get_option('styler.latex.multirow_align')
        latex = obj._render_latex(
            sparse_index=sparse_index, 
            sparse_columns=sparse_columns, 
            multirow_align=multirow_align, 
            multicol_align=multicol_align, 
            environment=environment, 
            convert_css=convert_css, 
            siunitx=siunitx, 
            clines=clines
        )
        encoding = encoding or get_option('styler.render.encoding') if isinstance(buf, str) else encoding
        return save_to_buffer(latex, buf=buf, encoding=encoding)

    @overload
    def to_typst(
        self, 
        buf: Union[str, WriteBuffer[str]], 
        *, 
        encoding: Optional[str] = ..., 
        sparse_index: Optional[bool] = ..., 
        sparse_columns: Optional[bool] = ..., 
        max_rows: Optional[int] = ..., 
        max_columns: Optional[int] = ...
    ) -> None:
        ...

    @overload
    def to_typst(
        self, 
        buf: None = ..., 
        *, 
        encoding: Optional[str] = ..., 
        sparse_index: Optional[bool] = ..., 
        sparse_columns: Optional[bool] = ..., 
        max_rows: Optional[int] = ..., 
        max_columns: Optional[int] = ...
    ) -> str:
        ...

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_typst(
        self, 
        buf: Optional[Union[str, WriteBuffer[str]]] = None, 
        *, 
        encoding: Optional[str] = None, 
        sparse_index: Optional[bool] = None, 
        sparse_columns: Optional[bool] = None, 
        max_rows: Optional[int] = None, 
        max_columns: Optional[int] = None, 
        delimiter: str = ' '
    ) -> Optional[str]:
        obj = self._copy(deepcopy=True)
        if sparse_index is None:
            sparse_index = get_option('styler.sparse.index')
        if sparse_columns is None:
            sparse_columns = get_option('styler.sparse.columns')
        text = obj._render_typst(
            sparse_columns=sparse_columns, 
            sparse_index=sparse_index, 
            max_rows=max_rows, 
            max_cols=max_columns
        )
        return save_to_buffer(text, buf=buf, encoding=encoding if buf is not None else None)

    @overload
    def to_html(
        self, 
        buf: Union[str, WriteBuffer[str]], 
        *, 
        table_uuid: Optional[str] = ..., 
        table_attributes: Optional[str] = ..., 
        sparse_index: Optional[bool] = ..., 
        sparse_columns: Optional[bool] = ..., 
        bold_headers: bool = ..., 
        caption: Optional[Union[str, Tuple[str, str]]] = ..., 
        max_rows: Optional[int] = ..., 
        max_columns: Optional[int] = ..., 
        encoding: Optional[str] = ..., 
        doctype_html: bool = ..., 
        exclude_styles: bool = ..., 
        **kwargs: Any
    ) -> None:
        ...

    @overload
    def to_html(
        self, 
        buf: None = ..., 
        *, 
        table_uuid: Optional[str] = ..., 
        table_attributes: Optional[str] = ..., 
        sparse_index: Optional[bool] = ..., 
        sparse_columns: Optional[bool] = ..., 
        bold_headers: bool = ..., 
        caption: Optional[Union[str, Tuple[str, str]]] = ..., 
        max_rows: Optional[int] = ..., 
        max_columns: Optional[int] = ..., 
        encoding: Optional[str] = ..., 
        doctype_html: bool = ..., 
        exclude_styles: bool