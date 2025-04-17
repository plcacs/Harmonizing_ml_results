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
    Union,
    Optional,
    List,
    Tuple,
    Dict,
    cast,
    TypeVar,
    Concatenate,
    Literal,
    Mapping,
    Iterable,
    Iterator,
    Set,
    Type,
    Generic,
    overload,
    Protocol,
    runtime_checkable,
    TypedDict,
    final,
    NoReturn,
    NewType,
    get_type_hints,
    get_origin,
    get_args,
    TypeAlias,
    ParamSpec,
    Self,
    final,
    NoReturn,
    NewType,
    get_type_hints,
    get_origin,
    get_args,
    TypeAlias,
    ParamSpec,
    Self,
    final,
    NoReturn,
    NewType,
    get_type_hints,
    get_origin,
    get_args,
    TypeAlias,
    ParamSpec,
    Self,
)

import numpy as np
import pandas as pd
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
    Substitution,
    doc,
)
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


####
# Shared Doc Strings

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

#
###


class Styler(StylerRenderer):
    r"""
    Helps style a DataFrame or Series according to the data with HTML and CSS.

    This class provides methods for styling and formatting a Pandas DataFrame or Series.
    The styled output can be rendered as HTML or LaTeX, and it supports CSS-based
    styling, allowing users to control colors, font styles, and other visual aspects of
    tabular data. It is particularly useful for presenting DataFrame objects in a
    Jupyter Notebook environment or when exporting styled tables for reports and

    Parameters
    ----------
    data : Series or DataFrame
        Data to be styled - either a Series or DataFrame.
    precision : int, optional
        Precision to round floats to. If not given defaults to
        ``pandas.options.styler.format.precision``.

        .. versionchanged:: 1.4.0
    table_styles : list-like, default None
        List of {selector: (attr, value)} dicts; see Notes.
    uuid : str, default None
        A unique identifier to avoid CSS collisions; generated automatically.
    caption : str, tuple, default None
        String caption to attach to the table. Tuple only used for LaTeX dual captions.
    table_attributes : str, default None
        Items that show up in the opening ``<table>`` tag
        in addition to automatic (by default) id.
    cell_ids : bool, default True
        If True, each cell will have an ``id`` attribute in their HTML tag.
        The ``id`` takes the form ``T_<uuid>_row<num_row>_col<num_col>``
        where ``<uuid>`` is the unique identifier, ``<num_row>`` is the row
        number and ``<num_col>`` is the column number.
    na_rep : str, optional
        Representation for missing values.
        If ``na_rep`` is None, no special formatting is applied, and falls back to
        ``pandas.options.styler.format.na_rep``.

    uuid_len : int, default 5
        If ``uuid`` is not specified, the length of the ``uuid`` to randomly generate
        expressed in hex characters, in range [0, 32].
    decimal : str, optional
        Character used as decimal separator for floats, complex and integers. If not
        given uses ``pandas.options.styler.format.decimal``.

        .. versionadded:: 1.3.0

    thousands : str, optional, default None
        Character used as thousands separator for floats, complex and integers. If not
        given uses ``pandas.options.styler.format.thousands``.

        .. versionadded:: 1.3.0

    escape : str, optional
        Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
        in cell display string with HTML-safe sequences.

        Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
        ``{``, ``}``, ``~``, ``^``, and ``\`` in the cell display string with
        LaTeX-safe sequences. Use 'latex-math' to replace the characters
        the same way as in 'latex' mode, except for math substrings,
        which either are surrounded by two characters ``$`` or start with
        the character ``\(`` and end with ``\)``.
        If not given uses ``pandas.options.styler.format.escape``.

        .. versionadded:: 1.3.0
    formatter : str, callable, dict, optional
        Object to define how values are displayed. See ``Styler.format``. If not given
        uses ``pandas.options.styler.format.formatter``.

        .. versionadded:: 1.4.0

    Attributes
    ----------
    env : Jinja2 jinja2.Environment
    template_html : Jinja2 Template
    template_html_table : Jinja2 Template
    template_html_style : Jinja2 Template
    template_latex : Jinja2 Template
    loader : Jinja2 Loader

    See Also
    --------
    DataFrame.style : Return a Styler object containing methods for building
        a styled HTML representation for the DataFrame.

    Notes
    -----
    .. warning::

       ``Styler`` is primarily intended for use on safe input that you control.
       When using ``Styler`` on untrusted, user-provided input to serve HTML,
       you should set ``escape="html"`` to prevent security vulnerabilities.
       See the Jinja2 documentation on escaping HTML for more.

    Most styling will be done by passing style functions into
    ``Styler.apply`` or ``Styler.map``. Style functions should
    return values with strings containing CSS ``'attr: value'`` that will
    be applied to the indicated cells.

    If using in the Jupyter notebook, Styler has defined a ``_repr_html_``
    to automatically render itself. Otherwise call Styler.to_html to get
    the generated HTML.

    CSS classes are attached to the generated HTML

    * Index and Column names include ``index_name`` and ``level<k>``
      where `k` is its level in a MultiIndex
    * Index label cells include

      * ``row_heading``
      * ``row<n>`` where `n` is the numeric position of the row
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Column label cells include

      * ``col_heading``
      * ``col<n>`` where `n` is the numeric position of the column
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Blank cells include ``blank``
    * Data cells include ``data``
    * Trimmed cells include ``col_trim`` or ``row_trim``.

    Any, or all, or these classes can be renamed by using the ``css_class_names``
    argument in ``Styler.set_table_styles``, giving a value such as
    *{"row": "MY_ROW_CLASS", "col_trim": "", "row_trim": ""}*.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     [[1.0, 2.0, 3.0], [4, 5, 6]], index=["a", "b"], columns=["A", "B", "C"]
    ... )
    >>> pd.io.formats.style.Styler(
    ...     df, precision=2, caption="My table"
    ... )  # doctest: +SKIP

    Please see:
    `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
    """

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

        # validate ordered args
        thousands = thousands or get_option("styler.format.thousands")
        decimal = decimal or get_option("styler.format.decimal")
        na_rep = na_rep or get_option("styler.format.na_rep")
        escape = escape or get_option("styler.format.escape")
        formatter = formatter or get_option("styler.format.formatter")
        # precision is handled by superclass as default for performance

        self.format(
            formatter=formatter,
            precision=precision,
            na_rep=na_rep,
            escape=escape,
            decimal=decimal,
            thousands=thousands,
        )

    def concat(self, other: Styler) -> Styler:
        """
        Append another Styler to combine the output into a single table.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        other : Styler
            The other Styler object which has already been styled and formatted. The
            data for this Styler must have the same columns as the original, and the
            number of index levels must also be the same to render correctly.

        Returns
        -------
        Styler
            Instance of class with specified Styler appended.

        See Also
        --------
        Styler.clear : Reset the ``Styler``, removing any previously applied styles.
        Styler.export : Export the styles applied to the current Styler.

        Notes
        -----
        The purpose of this method is to extend existing styled dataframes with other
        metrics that may be useful but may not conform to the original's structure.
        For example adding a sub total row, or displaying metrics such as means,
        variance or counts.

        Styles that are applied using the ``apply``, ``map``, ``apply_index``
        and ``map_index``, and formatting applied with ``format`` and
        ``format_index`` will be preserved.

        .. warning::
            Only the output methods ``to_html``, ``to_string`` and ``to_latex``
            currently work with concatenated Stylers.

            Other output methods, including ``to_excel``, **do not** work with
            concatenated Stylers.

        The following should be noted:

          - ``table_styles``, ``table_attributes``, ``caption`` and ``uuid`` are all
            inherited from the original Styler and not ``other``.
          - hidden columns and hidden index levels will be inherited from the
            original Styler
          - ``css`` will be inherited from the original Styler, and the value of
            keys ``data``, ``row_heading`` and ``row`` will be prepended with
            ``foot0_``. If more concats are chained, their styles will be prepended
            with ``foot1_``, ''foot_2'', etc., and if a concatenated style have
            another concatenated style, the second style will be prepended with
            ``foot{parent}_foot{child}_``.

        A common use case is to concatenate user defined functions with
        ``DataFrame.agg`` or with described statistics via ``DataFrame.describe``.
        See examples.

        Examples
        --------
        A common use case is adding totals rows, or otherwise, via methods calculated
        in ``DataFrame.agg``.

        >>> df = pd.DataFrame(
        ...     [[4, 6], [1, 9], [3, 4], [5, 5], [9, 6]],
        ...     columns=["Mike", "Jim"],
        ...     index=["Mon", "Tue", "Wed", "Thurs", "Fri"],
        ... )
        >>> styler = df.style.concat(df.agg(["sum"]).style)  # doctest: +SKIP

        .. figure:: ../../_static/style/footer_simple.png

        Since the concatenated object is a Styler the existing functionality can be
        used to conditionally format it as well as the original.

        >>> descriptors = df.agg(["sum", "mean", lambda s: s.dtype])
        >>> descriptors.index = ["Total", "Average", "dtype"]
        >>> other = (
        ...     descriptors.style.highlight_max(
        ...         axis=1, subset=(["Total", "Average"], slice(None))
        ...     .format(subset=("Average", slice(None)), precision=2, decimal=",")
        ...     .map(lambda v: "font-weight: bold;")
        ... )
        >>> styler = df.style.highlight_max(color="salmon").set_table_styles(
        ...     [{"selector": ".foot_row0", "props": "border-top: 1px solid black;"}]
        ... )
        >>> styler.concat(other)  # doctest: +SKIP

        .. figure:: ../../_static/style/footer_extended.png

        When ``other`` has fewer index levels than the original Styler it is possible
        to extend the index in ``other``, with placeholder levels.

        >>> df = pd.DataFrame(
        ...     [[1], [2]], index=pd.MultiIndex.from_product([[0], [1, 2]])
        ... )
        >>> descriptors = df.agg(["sum"])
        >>> descriptors.index = pd.MultiIndex.from_product([[""], descriptors.index])
        >>> df.style.concat(descriptors.style)  # doctest: +SKIP
        """
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
        """
        Hooks into Jupyter notebook rich display system, which calls _repr_html_ by
        default if an object is returned at the end of a cell.
        """
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
        """
        Set the DataFrame of strings on ``Styler`` generating ``:hover`` tooltips.

        These string based tooltips are only applicable to ``<td>`` HTML elements,
        and cannot be used for column or index headers.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        ttips : DataFrame
            DataFrame containing strings that will be translated to tooltips, mapped
            by identical column and index values that must exist on the underlying
            Styler data. None, NaN values, and empty strings will be ignored and
            not affect the rendered HTML.
        props : list-like or str, optional
            List of (attr, value) tuples or a valid CSS string. If ``None`` adopts
            the internal default values described in notes.
        css_class : str, optional
            Name of the tooltip class used in CSS, should conform to HTML standards.
            Only useful if integrating tooltips with external CSS. If ``None`` uses the
            internal default value 'pd-t'.
        as_title_attribute : bool, default False
            Add the tooltip text as title attribute to resultant <td> element. If True
            then props and css_class arguments are ignored.

        Returns
        -------
        Styler
            Instance of class with DataFrame set for strings on ``Styler``
                generating ``:hover`` tooltips.

        See Also
        --------
        Styler.set_table_attributes : Set the table attributes added to the
            ``<table>`` HTML element.
        Styler.set_table_styles : Set the table styles included within the
            ``<style>`` HTML element.

        Notes
        -----
        Tooltips are created by adding `<span class="pd-t"></span>` to each data cell
        and then manipulating the table level CSS to attach pseudo hover and pseudo
        after selectors to produce the required the results.

        The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

        The property 'visibility: hidden;' is a key prerequisite to the hover
        functionality, and should always be included in any manual properties
        specification, using the ``props`` argument.

        Tooltips are not designed to be efficient, and can add large amounts of
        additional HTML for larger tables, since they also require that ``cell_ids``
        is forced to `True`.

        If multiline tooltips are required, or if styling is not required and/or
        space is of concern, then utilizing as_title_attribute as True will store
        the tooltip on the <td> title attribute. This will cause no CSS
        to be generated nor will the <span> elements. Storing tooltips through
        the title attribute will mean that tooltip styling effects do not apply.

        Examples
        --------
        Basic application

        >>> df = pd.DataFrame(data=[[0, 1], [2, 3]])
        >>> ttips = pd.DataFrame(
        ...     data=[["Min", ""], [np.nan, "Max"]], columns=df.columns, index=df.index
        ... )
        >>> s = df.style.set_tooltips(ttips).to_html()

        Optionally controlling the tooltip visual display

        >>> df.style.set_tooltips(
        ...     ttips,
        ...     css_class="tt-add",
        ...     props=[
        ...         ("visibility", "hidden"),
        ...         ("position", "absolute"),
        ...         ("z-index", 1),
        ...     ],
        ... )  # doctest: +SKIP
        >>> df.style.set_tooltips(
        ...     ttips,
        ...     css_class="tt-add",
        ...     props="visibility:hidden; position:absolute; z-index:1;",
        ... )
        ... # doctest: +SKIP

        Multiline tooltips with smaller size footprint

        >>> df.style.set_tooltips(ttips, as_title_attribute=True)  # doctest: +SKIP
        """
        if not self.cell_ids:
            # tooltips not optimised for individual cell check. requires reasonable
            # redesign and more extensive code for a feature that might be rarely used.
            raise NotImplementedError(
                "Tooltips can only render with 'cell_ids' is True."
            )
        if not ttips.index.is_unique or not ttips.columns.is_unique:
            raise KeyError(
                "Tooltips render only if `ttips` has unique index and columns."
            )
        if self.tooltips is None:  # create a default instance if necessary
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

    @doc(
        NDFrame.to_excel,
        klass="Styler",
        storage_options=_shared_docs["storage_options"],
        storage_options_versionadded="1.5.0",
        encoding_parameter=textwrap.dedent(
            """\
        encoding : str or None, default None
            Unused parameter, present for compatibility.
        """
        ),
        verbose_parameter=textwrap.dedent(
            """\
        verbose : str, default True
            Optional unused parameter, present for compatibility.
        """
        ),
        extra_parameters="",
    )
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
        r"""
        Write Styler to a file, buffer or string in LaTeX format.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        column_format : str, optional
            The LaTeX column specification placed in location:

            \\begin{tabular}{<column_format>}

            Defaults to 'l' for index and
            non-numeric data columns, and, for numeric data columns,
            to 'r' by default, or 'S' if ``siunitx`` is ``True``.
        position : str, optional
            The LaTeX positional argument (e.g. 'h!') for tables, placed in location:

            ``\\begin{table}[<position>]``.
        position_float : {"centering", "raggedleft", "raggedright"}, optional
            The LaTeX float command placed in location:

            \\begin{table}[<position>]

            \\<position_float>

            Cannot be used if ``environment`` is "longtable".
        hrules : bool
            Set to `True` to add \\toprule, \\midrule and \\bottomrule from the
            {booktabs} LaTeX package.
            Defaults to ``pandas.options.styler.latex.hrules``, which is `False`.

            .. versionchanged:: 1.4.0
        clines : str, optional
            Use to control adding \\cline commands for the index labels separation.
            Possible values are:

              - `None`: no cline commands are added (default).
              - `"all;data"`: a cline is added for every index value extending the
                width of the table, including data entries.
              - `"all;index"`: as above with lines extending only the width of the
                index entries.
              - `"skip-last;data"`: a cline is added for each index value except the
                last level (which is never sparsified), extending the widtn of the
                table.
              - `"skip-last;index"`: as above with lines extending only the width of the
                index entries.

            .. versionadded:: 1.4.0
        label : str, optional
            The LaTeX label included as: \\label{<label>}.
            This is used with \\ref{<label>} in the main .tex file.
        caption : str, tuple, optional
            If string, the LaTeX table caption included as: \\caption{<caption>}.
            If tuple, i.e ("full caption", "short caption"), the caption included
            as: \\caption[<caption[1]>]{<caption[0]>}.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index``, which is `True`.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns``, which
            is `True`.
        multirow_align : {"c", "t", "b", "naive"}, optional
            If sparsifying hierarchical MultiIndexes whether to align text centrally,
            at the top or bottom using the multirow package. If not given defaults to
            ``pandas.options.styler.latex.multirow_align``, which is `"c"`.
            If "naive" is given renders without multirow.

            .. versionchanged:: 1.4.0
        multicol_align : {"r", "c", "l", "naive-l", "naive-r"}, optional
            If sparsifying hierarchical MultiIndex columns whether to align text at
            the left, centrally, or at the right. If not given defaults to
            ``pandas.options.styler.latex.multicol_align``, which is "r".
            If a naive option is given renders without multicol.
            Pipe decorators can also be added to non-naive values to draw vertical
            rules, e.g. "\|r" will draw a rule on the left side of right aligned merged
            cells.

            .. versionchanged:: 1.4.0
        siunitx : bool, default False
            Set to ``True`` to structure LaTeX compatible with the {siunitx} package.
        environment : str, optional
            If given, the environment that will replace 'table' in ``\\begin{table}``.
            If 'longtable' is specified then a more suitable template is
            rendered. If not given defaults to
            ``pandas.options.styler.latex.environment``, which is `None`.

            .. versionadded:: 1.4.0
        encoding : str, optional
            Character encoding setting. Defaults
            to ``pandas.options.styler.render.encoding``, which is "utf-8".
        convert_css : bool, default False
            Convert simple cell-styles from CSS to LaTeX format. Any CSS not found in
            conversion table is dropped. A style can be forced by adding option
            `--latex`. See notes.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        Styler.format: Format the text display value of cells.

        Notes
        -----
        **Latex Packages**

        For the following features we recommend the following LaTeX inclusions:

        ===================== ==========================================================
        Feature               Inclusion
        ===================== ==========================================================
        sparse columns        none: included within default {tabular} environment
        sparse rows           \\usepackage{multirow}
        hrules                \\usepackage{booktabs}
        colors                \\usepackage[table]{xcolor}
        siunitx               \\usepackage{siunitx}
        bold (with siunitx)   | \\usepackage{etoolbox}
                              | \\robustify\\bfseries
                              | \\sisetup{detect-all = true}  *(within {document})*
        italic (with siunitx) | \\usepackage{etoolbox}
                              | \\robustify\\itshape
                              | \\sisetup{detect-all = true}  *(within {document})*
        environment           \\usepackage{longtable} if arg is "longtable"
                              | or any other relevant environment package
        hyperlinks            \\usepackage{hyperref}
        ===================== ==========================================================

        **Cell Styles**

        LaTeX styling can only be rendered if the accompanying styling functions have
        been constructed with appropriate LaTeX commands. All styling
        functionality is built around the concept of a CSS ``(<attribute>, <value>)``
        pair (see `Table Visualization <../../user_guide/style.ipynb>`_), and this
        should be replaced by a LaTeX
        ``(<command>, <options>)`` approach. Each cell will be styled individually
        using nested LaTeX commands with their accompanied options.

        For example the following code will highlight and bold a cell in HTML-CSS:

        >>> df = pd.DataFrame([[1, 2], [3, 4]])
        >>> s = df.style.highlight_max(axis=None,
        ...                            props='background-color:red; font-weight:bold;')
        >>> s.to_html()  # doctest: +SKIP

        The equivalent using LaTeX only commands is the following:

        >>> s = df.style.highlight_max(axis=None,
        ...                            props='cellcolor:{red}; bfseries: ;')
        >>> s.to_latex()  # doctest: +SKIP

        Internally these structured LaTeX ``(<command>, <options>)`` pairs
        are translated to the
        ``display_value`` with the default structure:
        ``\<command><options> <display_value>``.
        Where there are multiple commands the latter is nested recursively, so that
        the above example highlighted cell is rendered as
        ``\cellcolor{red} \bfseries 4``.

        Occasionally this format does not suit the applied command, or
        combination of LaTeX packages that is in use, so additional flags can be
        added to the ``<options>``, within the tuple, to result in different
        positions of required braces (the **default** being the same as ``--nowrap``):

        =================================== ============================================
        Tuple Format                           Output Structure
        =================================== ============================================
        (<command>,<options>)               \\<command><options> <display_value>
        (<command>,<options> ``--nowrap``)  \\<command><options> <display_value>
        (<command