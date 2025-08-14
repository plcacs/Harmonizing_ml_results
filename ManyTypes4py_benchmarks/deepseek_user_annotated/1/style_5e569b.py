from __future__ import annotations

import copy
from functools import partial
import operator
import textwrap
from typing import (
    TYPE_CHECKING,
    overload,
    Any,
    Callable,
    Hashable,
    Sequence,
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Literal,
    cast,
    TypeVar,
    Generic,
    Type,
    Mapping,
    Iterable,
    Iterator,
    Set,
    FrozenSet,
    MutableMapping,
    Deque,
    DefaultDict,
    Counter,
    ChainMap,
    Awaitable,
    Coroutine,
    AsyncIterable,
    AsyncIterator,
    AsyncGenerator,
    NamedTuple,
    NoReturn,
    Pattern,
    Match,
    IO,
    TextIO,
    BinaryIO,
    AnyStr,
    TypeAlias,
    Final,
    Protocol,
    runtime_checkable,
    TypedDict,
    NewType,
    get_type_hints,
    get_args,
    get_origin,
    Annotated,
    ClassVar,
    ForwardRef,
    ParamSpec,
    Concatenate,
    Self,
)

import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
    Substitution,
    doc,
)
import pandas as pd
from pandas import (
    IndexSlice,
    RangeIndex,
)
import pandas.core.common as com
from pandas.core.frame import (
    DataFrame,
    Series,
)
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )
    from matplotlib.colors import Colormap
    from pandas._typing import (
        Any,
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
        WriteExcelBuffer,
    )
    from pandas import ExcelWriter

jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")

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
    refactor_levels,
)

T = TypeVar('T')
P = ParamSpec('P')

subset_args = """subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function."""

properties_args = """props : str, default None
           CSS properties to use for highlighting. If ``props`` is given, ``color``
           is not used."""

coloring_args = """color : str, default '{default}'
           Background color to use for highlighting."""

buffering_args = """buf : str, path object, file-like object, optional
         String, path object (implementing ``os.PathLike[str]``), or file-like
         object implementing a string ``write()`` function. If ``None``, the result is
         returned as a string."""

encoding_args = """encoding : str, optional
              Character encoding setting for file output (and meta tags if available).
              Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8"."""

class Styler(StylerRenderer):
    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: str | tuple | list | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str | None = None,
        thousands: str | None = None,
        escape: str | None = None,
        formatter: ExtFormatter | None = None,
    ) -> None:
        super().__init__(
            data=data,
            uuid=uuid,
            uuid_len=uuid_len,
            table_styles=table_styles,
            table_attributes=table_attributes,
            caption=caption,
            cell_ids=cell_ids,
            precision=precision,
        )
        thousands = thousands or get_option("styler.format.thousands")
        decimal = decimal or get_option("styler.format.decimal")
        na_rep = na_rep or get_option("styler.format.na_rep")
        escape = escape or get_option("styler.format.escape")
        formatter = formatter or get_option("styler.format.formatter")
        self.format(
            formatter=formatter,
            precision=precision,
            na_rep=na_rep,
            escape=escape,
            decimal=decimal,
            thousands=thousands,
        )

    def concat(self, other: Styler) -> Styler:
        if not isinstance(other, Styler):
            raise TypeError("`other` must be of type `Styler`")
        if not self.data.columns.equals(other.data.columns):
            raise ValueError("`other.data` must have same columns as `Styler.data`")
        if not self.data.index.nlevels == other.data.index.nlevels:
            raise ValueError(
                "number of index levels must be same in `other` "
                "as in `Styler`. See documentation for suggestions."
            )
        self.concatenated.append(other)
        return self

    def _repr_html_(self) -> str | None:
        if get_option("styler.render.repr") == "html":
            return self.to_html()
        return None

    def _repr_latex_(self) -> str | None:
        if get_option("styler.render.repr") == "latex":
            return self.to_latex()
        return None

    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = None,
        css_class: str | None = None,
        as_title_attribute: bool = False,
    ) -> Styler:
        if not self.cell_ids:
            raise NotImplementedError(
                "Tooltips can only render with 'cell_ids' is True."
            )
        if not ttips.index.is_unique or not ttips.columns.is_unique:
            raise KeyError(
                "Tooltips render only if `ttips` has unique index and columns."
            )
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

    @overload
    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: str | None = None,
        merge_cells: ExcelWriterMergeCells = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None: ...

    @overload
    def to_excel(
        self,
        excel_writer: None = ...,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: str | None = None,
        merge_cells: ExcelWriterMergeCells = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None: ...

    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter | None = None,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: str | None = None,
        merge_cells: ExcelWriterMergeCells = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
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
            inf_rep=inf_rep,
        )
        formatter.write(
            excel_writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            freeze_panes=freeze_panes,
            engine=engine,
            storage_options=storage_options,
        )

    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> None: ...

    @overload
    def to_latex(
        self,
        buf: None = ...,
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> str: ...

    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: str | None = None,
        hrules: bool | None = None,
        clines: str | None = None,
        label: str | None = None,
        caption: str | tuple | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: str | None = None,
        multicol_align: str | None = None,
        siunitx: bool = False,
        environment: str | None = None,
        encoding: str | None = None,
        convert_css: bool = False,
    ) -> str | None:
        obj = self._copy(deepcopy=True)
        table_selectors = (
            [style["selector"] for style in self.table_styles]
            if self.table_styles is not None
            else []
        )
        if column_format is not None:
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )
        elif "column_format" in table_selectors:
            pass
        else:
            _original_columns = self.data.columns
            self.data.columns = RangeIndex(stop=len(self.data.columns))
            numeric_cols = self.data._get_numeric_data().columns.to_list()
            self.data.columns = _original_columns
            column_format = ""
            for level in range(self.index.nlevels):
                column_format += "" if self.hide_index_[level] else "l"
            for ci, _ in enumerate(self.data.columns):
                if ci not in self.hidden_columns:
                    column_format += (
                        ("r" if not siunitx else "S") if ci in numeric_cols else "l"
                    )
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )
        if position:
            obj.set_table_styles(
                [{"selector": "position", "props": f":{position}"}],
                overwrite=False,
            )
        if position_float:
            if environment == "longtable":
                raise ValueError(
                    "`position_float` cannot be used in 'longtable' `environment`"
                )
            if position_float not in ["raggedright", "raggedleft", "centering"]:
                raise ValueError(
                    f"`position_float` should be one of "
                    f"'raggedright', 'raggedleft', 'centering', "
                    f"got: '{position_float}'"
                )
            obj.set_table_styles(
                [{"selector": "position_float", "props": f":{position_float}"}],
                overwrite=False,
            )
        hrules = get_option("styler.latex.hrules") if hrules is None else hrules
        if hrules:
            obj.set_table_styles(
                [
                    {"selector": "toprule", "props": ":toprule"},
                    {"selector": "midrule", "props": ":midrule"},
                    {"selector": "bottomrule", "props": ":bottomrule"},
                ],
                overwrite=False,
            )
        if label:
            obj.set_table_styles(
                [{"selector": "label", "props": f":{{{label.replace(':', '§')}}}"}],
                overwrite=False,
            )
        if caption:
            obj.set_caption(caption)
        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")
        environment = environment or get_option("styler.latex.environment")
        multicol_align = multicol_align or get_option("styler.latex.multicol_align")
        multirow_align = multirow_align or get_option("styler.latex.multirow_align")
        latex = obj._render_latex(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            multirow_align=multirow_align,
            multicol_align=multicol_align,
            environment=environment,
            convert_css=convert_css,
            siunitx=siunitx,
            clines=clines,
        )
        encoding = (
            (encoding or get_option("styler.render.encoding"))
            if isinstance(buf, str)
            else encoding
        )
        return save_to_buffer(latex, buf=buf, encoding=encoding)

    @overload
    def to_typst(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
    ) -> None: ...

    @overload
    def to_typst(
        self,
        buf: None = ...,
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
    ) -> str: ...

    def to_typst(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
    ) -> str | None:
        obj = self._copy(deepcopy=True)
        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")
        text = obj._render_typst(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
        )
        return save_to_buffer(
            text, buf=buf, encoding=(encoding if buf is not None else None)
        )

    @overload
    def to_html(
        self