"""
Module for applying conditional formatting to DataFrames and Series.
"""
from __future__ import annotations
import copy
from functools import partial
import operator
import textwrap
from typing import TYPE_CHECKING, overload, Callable, Hashable, Sequence, Any, Optional, Union, List, Tuple, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, IndexSlice, RangeIndex
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import Substitution, doc
from pandas.core.common import maybe_convert_css_to_tuples
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import CSSProperties, CSSStyles, ExtFormatter, StylerRenderer, Subset, Tooltips, format_table_styles, maybe_convert_css_to_tuples, non_reducing_slice, refactor_levels

jinja2 = import_optional_dependency('jinja2', extra='DataFrame.style requires jinja2.')

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
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

subset_args = 'subset : label, array-like, IndexSlice, optional\n            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input\n            or single key, to `DataFrame.loc[:, <subset>]` where the columns are\n            prioritised, to limit ``data`` to *before* applying the function.'
properties_args = 'props : str, default None\n           CSS properties to use for highlighting. If ``props`` is given, ``color``\n           is not used.'
coloring_args = "color : str, default '{default}'\n           Background color to use for highlighting."
buffering_args = 'buf : str, path object, file-like object, optional\n         String, path object (implementing ``os.PathLike[str]``), or file-like\n         object implementing a string ``write()`` function. If ``None``, the result is\n         returned as a string.'
encoding_args = 'encoding : str, optional\n              Character encoding setting for file output (and meta tags if available).\n              Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8".'

class Styler(StylerRenderer):
    """
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
        ``{``, ``}``, ``~``, ``^``, and ``\\`` in the cell display string with
        LaTeX-safe sequences. Use 'latex-math' to replace the characters
        the same way as in 'latex' mode, except for math substrings,
        which either are surrounded by two characters ``$`` or start with
        the character ``\\(`` and end with ``\\)``.
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
        data: Union[Series, DataFrame],
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
        formatter: Optional[Union[str, Callable, Dict]] = None,
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
        ...     )
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
            raise TypeError('`other` must be of type `Styler`')
        if not self.data.columns.equals(other.data.columns):
            raise ValueError('`other.data` must have same columns as `Styler.data`')
        if not self.data.index.nlevels == other.data.index.nlevels:
            raise ValueError('number of index levels must be same in `other` as in `Styler`. See documentation for suggestions.')
        self.concatenated.append(other)
        return self

    def _repr_html_(self) -> Optional[str]:
        """
        Hooks into Jupyter notebook rich display system, which calls _repr_html_ by
        default if an object is returned at the end of a cell.
        """
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
        props: Optional[Union[List[Tuple[str, str]], str]] = None,
        css_class: Optional[str] = None,
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
        excel_writer: Union[str, ExcelWriter, WriteExcelBuffer],
        sheet_name: str = 'Sheet1',
        na_rep: str = '',
        float_format: Optional[str] = None,
        columns: Optional[Sequence[Union[str, int, Hashable]]] = None,
        header: bool = True,
        index: bool = True,
        index_label: Optional[Union[str, Sequence[Hashable]]] = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Optional[str] = None,
        merge_cells: bool = True,
        encoding: Optional[str] = None,
        inf_rep: str = 'inf',
        verbose: bool = True,
        freeze_panes: Optional[Tuple[int, int]] = None,
        storage_options: Optional[StorageOptions] = None,
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
        buf: Union[str, Any],
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
        multirow_align: Optional[Literal["c", "t", "b", "naive"]] = ...,
        multicol_align: Optional[Literal["r", "c", "l", "naive-l", "naive-r"]] = ...,
        siunitx: Literal[False] = ...,
        environment: Optional[str] = ...,
        encoding: Optional[str] = ...,
        convert_css: bool = ...,
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
        multirow_align: Optional[Literal["c", "t", "b", "naive"]] = ...,
        multicol_align: Optional[Literal["r", "c", "l", "naive-l", "naive-r"]] = ...,
        siunitx: Literal[False] = ...,
        environment: Optional[str] = ...,
        encoding: Optional[str] = ...,
        convert_css: bool = ...,
    ) -> str:
        ...

    def to_latex(
        self,
        buf: Optional[Union[str, Any]] = None,
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
        multirow_align: Optional[Literal["c", "t", "b", "naive"]] = None,
        multicol_align: Optional[Literal["r", "c", "l", "naive-l", "naive-r"]] = None,
        siunitx: bool = False,
        environment: Optional[str] = None,
        encoding: Optional[str] = None,
        convert_css: bool = False,
    ) -> Optional[str]:
        """
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

            \\\\begin{tabular}{<column_format>}

            Defaults to 'l' for index and
            non-numeric data columns, and, for numeric data columns,
            to 'r' by default, or 'S' if ``siunitx`` is ``True``.
        position : str, optional
            The LaTeX positional argument (e.g. 'h!') for tables, placed in location:

            ``\\\\begin{table}[<position>]``.
        position_float : {"centering", "raggedleft", "raggedright"}, optional
            The LaTeX float command placed in location:

            \\\\begin{table}[<position>]

            \\\\<position_float>

            Cannot be used if ``environment`` is "longtable".
        hrules : bool
            Set to `True` to add \\\\toprule, \\\\midrule and \\\\bottomrule from the
            {booktabs} LaTeX package.
            Defaults to ``pandas.options.styler.latex.hrules``, which is `False`.

            .. versionchanged:: 1.4.0
        clines : str, optional
            Use to control adding \\\\cline commands for the index labels separation.
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
            The LaTeX label included as: \\\\label{<label>}.
            This is used with \\\\ref{<label>} in the main .tex file.
        caption : str, tuple, optional
            If string, the LaTeX table caption included as: \\\\caption{<caption>}.
            If tuple, i.e ("full caption", "short caption"), the caption included
            as: \\\\caption[<caption[1]>]{<caption[0]>}.
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
            rules, e.g. "\\|r" will draw a rule on the left side of right aligned merged
            cells.

            .. versionchanged:: 1.4.0
        siunitx : bool, default False
            Set to ``True`` to structure LaTeX compatible with the {siunitx} package.
        environment : str, optional
            If given, the environment that will replace 'table' in ``\\\\begin{table}``.
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
        sparse rows           \\\\usepackage{multirow}
        hrules                \\\\usepackage{booktabs}
        colors                \\\\usepackage[table]{xcolor}
        siunitx               \\\\usepackage{siunitx}
        bold (with siunitx)   | \\\\usepackage{etoolbox}
                              | \\\\robustify\\\\bfseries
                              | \\\\sisetup{detect-all = true}  *(within {document})*
        italic (with siunitx) | \\\\usepackage{etoolbox}
                              | \\\\robustify\\\\itshape
                              | \\\\sisetup{detect-all = true}  *(within {document})*
        environment           \\\\usepackage{longtable} if arg is "longtable"
                              | or any other relevant environment package
        hyperlinks            \\\\usepackage{hyperref}
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
        ``\\<command><options> <display_value>``.
        Where there are multiple commands the latter is nested recursively, so that
        the above example highlighted cell is rendered as
        ``\\cellcolor{red} \\bfseries 4``.

        Occasionally this format does not suit the applied command, or
        combination of LaTeX packages that is in use, so additional flags can be
        added to the ``<options>``, within the tuple, to result in different
        positions of required braces (the **default** being the same as ``--nowrap``):

        =================================== ============================================
        Tuple Format                           Output Structure
        =================================== ============================================
        (<command>,<options>)               \\\\<command><options> <display_value>
        (<command>,<options> ``--nowrap``)  \\\\<command><options> <display_value>
        (<command>,<options> ``--rwrap``)   \\\\<command><options>{<display_value>}
        (<command>,<options> ``--wrap``)    {\\\\<command><options> <display_value>}
        (<command>,<options> ``--lwrap``)   {\\\\<command><options>} <display_value>
        (<command>,<options> ``--dwrap``)   {\\\\<command><options>}{<display_value>}
        =================================== ============================================

        For example the `textbf` command for font-weight
        should always be used with `--rwrap` so ``('textbf', '--rwrap')`` will render a
        working cell, wrapped with braces, as ``\\textbf{<display_value>}``.

        A more comprehensive example is as follows:

        >>> df = pd.DataFrame([[1, 2.2, "dogs"], [3, 4.4, "cats"], [2, 6.6, "cows"]],
        ...                   index=["ix1", "ix2", "ix3"],
        ...                   columns=["Integers", "Floats", "Strings"])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red}; textit:--rwrap; textbf:--rwrap;'
        ... )
        >>> s.to_latex()  # doctest: +SKIP

        .. figure:: ../../_static/style/latex_1.png

        **Table Styles**

        Internally Styler uses its ``table_styles`` object to parse the
        ``column_format``, ``position``, ``position_float``, and ``label``
        input arguments. These arguments are added to table styles in the format:

        .. code-block:: python

            set_table_styles([
                {"selector": "column_format", "props": f":{column_format};"},
                {"selector": "position", "props": f":{position};"},
                {"selector": "position_float", "props": f":{position_float};"},
                {"selector": "label", "props": f":{{{label.replace(':','§')}}};"}
            ], overwrite=False)

        Exception is made for the ``hrules`` argument which, in fact, controls all three
        commands: ``toprule``, ``bottomrule`` and ``midrule`` simultaneously. Instead of
        setting ``hrules`` to ``True``, it is also possible to set each
        individual rule definition, by manually setting the ``table_styles``,
        for example below we set a regular ``toprule``, set an ``hline`` for
        ``bottomrule`` and exclude the ``midrule``:

        .. code-block:: python

            set_table_styles([
                {'selector': 'toprule', 'props': ':toprule;'},
                {'selector': 'bottomrule', 'props': ':hline;'},
            ], overwrite=False)

        If other ``commands`` are added to table styles they will be detected, and
        positioned immediately above the '\\\\begin{tabular}' command. For example to
        add odd and even row coloring, from the {colortbl} package, in format
        ``\\rowcolors{1}{pink}{red}``, use:

        .. code-block:: python

            set_table_styles([
                {'selector': 'rowcolors', 'props': ':{1}{pink}{red};'}
            ], overwrite=False)

        A more comprehensive example using these arguments is as follows:

        >>> df.columns = pd.MultiIndex.from_tuples([
        ...     ("Numeric", "Integers"),
        ...     ("Numeric", "Floats"),
        ...     ("Non-Numeric", "Strings")
        ... ])
        >>> df.index = pd.MultiIndex.from_tuples([
        ...     ("L0", "ix1"), ("L0", "ix2"), ("L1", "ix3")
        ... ])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
        ... )
        >>> s.to_latex(
        ...     column_format="rrrrr", position="h", position_float="centering",
        ...     hrules=True, label="table:5", caption="Styled LaTeX Table",
        ...     multirow_align="t", multicol_align="r"
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/latex_2.png

        **Formatting**

        To format values :meth:`Styler.format` should be used prior to calling
        `Styler.to_latex`, as well as other methods such as :meth:`Styler.hide`
        for example:

        >>> s.clear()
        >>> s.table_styles = []
        >>> s.caption = None
        >>> s.format({
        ...     ("Numeric", "Integers"): '\\\\${}',
        ...     ("Numeric", "Floats"): '{:.3f}',
        ...     ("Non-Numeric", "Strings"): str.upper
        ... })  # doctest: +SKIP
                        Numeric      Non-Numeric
                  Integers   Floats    Strings
        L0    ix1       $1   2.200      DOGS
              ix2       $3   4.400      CATS
        L1    ix3       $2   6.600      COWS

        >>> s.to_latex()  # doctest: +SKIP
        \\begin{tabular}{llrrl}
        {} & {} & \\multicolumn{2}{r}{Numeric} & {Non-Numeric} \\\\
        {} & {} & {Integers} & {Floats} & {Strings} \\\\
        \\multirow[c]{2}{*}{L0} & ix1 & \\\\$1 & 2.200 & DOGS \\\\
         & ix2 & \\$3 & 4.400 & CATS \\\\
        L1 & ix3 & \\$2 & 6.600 & COWS \\\\
        \\end{tabular}

        **CSS Conversion**

        This method can convert a Styler constructured with HTML-CSS to LaTeX using
        the following limited conversions.

        ================== ==================== ============= ==========================
        CSS Attribute      CSS value            LaTeX Command LaTeX Options
        ================== ==================== ============= ==========================
        font-weight        | bold               | bfseries
                           | bolder             | bfseries
        font-style         | italic             | itshape
                           | oblique            | slshape
        background-color   | red                cellcolor     | {red}--lwrap
                           | #fe01ea                          | [HTML]{FE01EA}--lwrap
                           | #f0e                             | [HTML]{FF00EE}--lwrap
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}--lwrap
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}--lwrap
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}--lwrap
        color              | red                color         | {red}
                           | #fe01ea                          | [HTML]{FE01EA}
                           | #f0e                             | [HTML]{FF00EE}
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}
        ================== ==================== ============= ==========================

        It is also possible to add user-defined LaTeX only styles to a HTML-CSS Styler
        using the ``--latex`` flag, and to add LaTeX parsing options that the
        converter will detect within a CSS-comment.

        >>> df = pd.DataFrame([[1]])
        >>> df.style.set_properties(
        ...     **{"font-weight": "bold /* --dwrap */", "Huge": "--latex--rwrap"}
        ... ).to_latex(convert_css=True)  # doctest: +SKIP
        \\begin{tabular}{lr}
        {} & {0} \\\\
        0 & {\\bfseries}{\\Huge{1}} \\\\
        \\end{tabular}

        Examples
        --------
        Below we give a complete step by step example adding some advanced features
        and noting some common gotchas.

        First we create the DataFrame and Styler as usual, including MultiIndex rows
        and columns, which allow for more advanced formatting options:

        >>> cidx = pd.MultiIndex.from_arrays([
        ...     ["Equity", "Equity", "Equity", "Equity",
        ...      "Stats", "Stats", "Stats", "Stats", "Rating"],
        ...     ["Energy", "Energy", "Consumer", "Consumer", "", "", "", "", ""],
        ...     ["BP", "Shell", "H&M", "Unilever",
        ...      "Std Dev", "Variance", "52w High", "52w Low", ""]
        ... ])
        >>> iidx = pd.MultiIndex.from_arrays([
        ...     ["Equity", "Equity", "Equity", "Equity"],
        ...     ["Energy", "Energy", "Consumer", "Consumer"],
        ...     ["BP", "Shell", "H&M", "Unilever"]
        ... ])
        >>> styler = pd.DataFrame([
        ...     [1, 0.8, 0.66, 0.72, 32.1678, 32.1678**2, 335.12, 240.89, "Buy"],
        ...     [0.8, 1.0, 0.69, 0.79, 1.876, 1.876**2, 14.12, 19.78, "Hold"],
        ...     [0.66, 0.69, 1.0, 0.86, 7, 7**2, 210.9, 140.6, "Buy"],
        ...     [0.72, 0.79, 0.86, 1.0, 213.76, 213.76**2, 2807, 3678, "Sell"],
        ... ], columns=cidx, index=iidx).style

        Second we will format the display and, since our table is quite wide, will
        hide the repeated level-0 of the index:

        >>> (styler.format(subset="Equity", precision=2)
        ...     .format(subset="Stats", precision=1, thousands=",")
        ...     .format(subset="Rating", formatter=str.upper)
        ...     .format_index(escape="latex", axis=1)
        ...     .format_index(escape="latex", axis=0)
        ...     .hide(level=0, axis=0))  # doctest: +SKIP

        Note that one of the string entries of the index and column headers is "H&M".
        Without applying the `escape="latex"` option to the `format_index` method the
        resultant LaTeX will fail to render, and the error returned is quite
        difficult to debug. Using the appropriate escape the "&" is converted to "\\\\&".

        Thirdly we will apply some (CSS-HTML) styles to our object. We will use a
        builtin method and also define our own method to highlight the stock
        recommendation:

        >>> def rating_color(v: str) -> str:
        ...     if v == "Buy":
        ...         color = "#33ff85"
        ...     elif v == "Sell":
        ...         color = "#ff5933"
        ...     else:
        ...         color = "#ffdd33"
        ...     return f"color: {color}; font-weight: bold;"
        >>> (styler.background_gradient(cmap="inferno", subset="Equity", vmin=0, vmax=1)
        ...     .map(rating_color, subset="Rating"))  # doctest: +SKIP

        All the above styles will work with HTML (see below) and LaTeX upon conversion:

        .. figure:: ../../_static/style/latex_stocks_html.png

        However, we finally want to add one LaTeX only style
        (from the {graphicx} package), that is not easy to convert from CSS and
        pandas does not support it. Notice the `--latex` flag used here,
        as well as `--rwrap` to ensure this is formatted correctly and
        not ignored upon conversion.

        >>> styler.map_index(
        ...     lambda v: "rotatebox:{45}--rwrap--latex;", level=2, axis=1
        ... )  # doctest: +SKIP

        Finally we render our LaTeX adding in other options as required:

        >>> styler.to_latex(
        ...     caption="Selected stock correlation and simple statistics.",
        ...     clines="skip-last;data",
        ...     convert_css=True,
        ...     position_float="centering",
        ...     multicol_align="|c|",
        ...     hrules=True,
        ... )  # doctest: +SKIP
        \\begin{table}
        \\centering
        \\caption{Selected stock correlation and simple statistics.}
        \\begin{tabular}{llrrrrrrrrl}
        \\toprule
         &  & \\multicolumn{4}{|c|}{Equity} & \\multicolumn{4}{|c|}{Stats} & Rating \\\\
         &  & \\multicolumn{2}{|c|}{Energy} & \\multicolumn{2}{|c|}{Consumer} &
        \\multicolumn{4}{|c|}{} &  \\\\
         &  & \\rotatebox{45}{BP} & \\rotatebox{45}{Shell} & \\rotatebox{45}{H\\&M} &
        \\rotatebox{45}{Unilever} & \\rotatebox{45}{Std Dev} & \\rotatebox{45}{Variance} &
        \\rotatebox{45}{52w High} & \\rotatebox{45}{52w Low} & \\rotatebox{45}{} \\\\
        \\midrule
        \\multirow[c]{2}{*}{Energy} & BP & {\\cellcolor[HTML]{FCFFA4}}
        \\color[HTML]{000000} 1.00 & {\\cellcolor[HTML]{FCA50A}} \\color[HTML]{000000}
        0.80 & {\\cellcolor[HTML]{EB6628}} \\color[HTML]{F1F1F1} 0.66 &
        {\\cellcolor[HTML]{F68013}} \\color[HTML]{F1F1F1} 0.72 & 32.2 & 1,034.8 & 335.1
        & 240.9 & \\color[HTML]{33FF85} \\bfseries BUY \\\\
         & Shell & {\\cellcolor[HTML]{FCA50A}} \\color[HTML]{000000} 0.80 &
        {\\cellcolor[HTML]{FCFFA4}} \\color[HTML]{000000} 1.00 &
        {\\cellcolor[HTML]{F1731D}} \\color[HTML]{F1F1F1} 0.69 &
        {\\cellcolor[HTML]{FCA108}} \\color[HTML]{000000} 0.79 & 1.9 & 3.5 & 14.1 &
        19.8 & \\color[HTML]{FFDD33} \\bfseries HOLD \\\\
        \\cline{1-11}
        \\multirow[c]{2}{*}{Consumer} & H\\&M & {\\cellcolor[HTML]{EB6628}}
        \\color[HTML]{F1F1F1} 0.66 & {\\cellcolor[HTML]{F1731D}} \\color[HTML]{F1F1F1}
        0.69 & {\\cellcolor[HTML]{FCFFA4}} \\color[HTML]{000000} 1.00 &
        {\\cellcolor[HTML]{FAC42A}} \\color[HTML]{000000} 0.86 & 7.0 & 49.0 & 210.9 &
        140.6 & \\color[HTML]{33FF85} \\bfseries BUY \\\\
         & Unilever & {\\cellcolor[HTML]{F68013}} \\color[HTML]{F1F1F1} 0.72 &
        {\\cellcolor[HTML]{FCA108}} \\color[HTML]{000000} 0.79 &
        {\\cellcolor[HTML]{FAC42A}} \\color[HTML]{000000} 0.86 &
        {\\cellcolor[HTML]{FCFFA4}} \\color[HTML]{000000} 1.00 & 213.8 & 45,693.3 &
        2,807.0 & 3,678.0 & \\color[HTML]{FF5933} \\bfseries SELL \\\\
        \\cline{1-11}
        \\bottomrule
        \\end{tabular}
        \\end{table}

        .. figure:: ../../_static/style/latex_stocks.png
        """
        obj = self._copy(deep=True)
        table_selectors: List[str] = [style['selector'] for style in self.table_styles] if self.table_styles is not None else []
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
            obj.set_table_styles([
                {'selector': 'toprule', 'props': ':toprule'},
                {'selector': 'midrule', 'props': ':midrule'},
                {'selector': 'bottomrule', 'props': ':bottomrule'},
            ], overwrite=False)
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
        latex: str = obj._render_latex(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            multirow_align=multirow_align,
            multicol_align=multicol_align,
            environment=environment,
            convert_css=convert_css,
            siunitx=siunitx,
            clines=clines,
        )
        encoding = encoding or (get_option('styler.render.encoding') if isinstance(buf, str) else None)
        return save_to_buffer(latex, buf=buf, encoding=encoding)

    @overload
    def to_typst(
        self,
        buf: Union[str, Any],
        *,
        encoding: Optional[str] = ...,
        sparse_index: Optional[bool] = ...,
        sparse_columns: Optional[bool] = ...,
        max_rows: Optional[int] = ...,
        max_columns: Optional[int] = ...,
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
        max_columns: Optional[int] = ...,
    ) -> str:
        ...

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_typst(
        self,
        buf: Optional[Union[str, Any]] = None,
        *,
        encoding: Optional[str] = None,
        sparse_index: Optional[bool] = None,
        sparse_columns: Optional[bool] = None,
        max_rows: Optional[int] = None,
        max_columns: Optional[int] = None,
    ) -> Optional[str]:
        """
        Write Styler to a file, buffer or string in Typst format.

        .. versionadded:: 3.0.0

        Parameters
        ----------
        %(buf)s
        %(encoding)s
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows``, which is None.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_typst : Write a DataFrame to a file,
            buffer or string in Typst format.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.to_typst()  # doctest: +SKIP

        .. code-block:: typst

            #table(
              columns: 3,
              [], [A], [B],

              [0], [1], [3],
              [1], [2], [4],
            )
        """
        obj = self._copy(deep=True)
        if sparse_index is None:
            sparse_index = get_option('styler.sparse.index')
        if sparse_columns is None:
            sparse_columns = get_option('styler.sparse.columns')
        text: str = obj._render_typst(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
        )
        return save_to_buffer(text, buf=buf, encoding=(encoding if buf is not None else None))

    @overload
    def to_html(
        self,
        buf: Union[str, Any],
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
        **kwargs: Any,
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
        exclude_styles: bool = ...,
        **kwargs: Any,
    ) -> str:
        ...

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_html(
        self,
        buf: Optional[Union[str, Any]] = None,
        *,
        table_uuid: Optional[str] = None,
        table_attributes: Optional[str] = None,
        sparse_index: Optional[bool] = None,
        sparse_columns: Optional[bool] = None,
        bold_headers: bool = False,
        caption: Optional[Union[str, Tuple[str, str]]] = None,
        max_rows: Optional[int] = None,
        max_columns: Optional[int] = None,
        encoding: Optional[str] = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Write Styler to a file, buffer or string in HTML-CSS format.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        %(buf)s
        table_uuid : str, optional
            Id attribute assigned to the <table> HTML element in the format:

            ``<table id="T_<table_uuid>" ..>``

            If not given uses Styler's initially assigned value.
        table_attributes : str, optional
            Attributes to assign within the `<table>` HTML element in the format:

            ``<table .. <table_attributes> >``

            If not given defaults to Styler's preexisting value.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.

            .. versionadded:: 1.4.0
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.

            .. versionadded:: 1.4.0
        bold_headers : bool, optional
            Adds "font-weight: bold;" as a CSS property to table style header cells.

            .. versionadded:: 1.4.0
        caption : str, optional
            Set, or overwrite, the caption on Styler before rendering.

            .. versionadded:: 1.4.0
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows/max_columns``.

            .. versionadded:: 1.4.0
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).

            .. versionadded:: 1.4.0
        %(encoding)s
        doctype_html : bool, default False
            Whether to output a fully structured HTML file including all
            HTML elements, or just the core ``<style>`` and ``<table>`` elements.
        exclude_styles : bool, default False
            Whether to include the ``<style>`` element and all associated element
            ``class`` and ``id`` identifiers, or solely the ``<table>`` element without
            styling identifiers.
        **kwargs : Any
            Any additional keyword arguments are passed through to the jinja2
            ``self.template.render`` process. This is useful when you need to provide
            additional variables for a custom template.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_html: Write a DataFrame to a file, buffer or string in HTML format.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> print(df.style.to_html())  # doctest: +SKIP
        <style type="text/css">
        </style>
        <table id="T_1e78e">
          <thead>
            <tr>
              <th class="blank level0" >&nbsp;</th>
              <th id="T_1e78e_level0_col0" class="col_heading level0 col0" >A</th>
              <th id="T_1e78e_level0_col1" class="col_heading level0 col1" >B</th>
            </tr>
        ...
        """

        obj = self._copy(deep=True)
        if table_uuid:
            obj.set_uuid(table_uuid)
        if table_attributes:
            obj.set_table_attributes(table_attributes)
        if sparse_index is None:
            sparse_index = get_option('styler.sparse.index')
        if sparse_columns is None:
            sparse_columns = get_option('styler.sparse.columns')
        if bold_headers:
            obj.set_table_styles([{'selector': 'th', 'props': 'font-weight: bold;'}], overwrite=False)
        if caption is not None:
            obj.set_caption(caption)
        html: str = obj._render_html(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            max_rows=max_rows,
            max_cols=max_columns,
            exclude_styles=exclude_styles,
            encoding=(encoding or get_option('styler.render.encoding')),
            doctype_html=doctype_html,
            **kwargs,
        )
        return save_to_buffer(html, buf=buf, encoding=(encoding if buf is not None else None))

    @overload
    def to_string(
        self,
        buf: Union[str, Any],
        *,
        encoding: Optional[str] = ...,
        sparse_index: Optional[bool] = ...,
        sparse_columns: Optional[bool] = ...,
        max_rows: Optional[int] = ...,
        max_columns: Optional[int] = ...,
        delimiter: str = ...,
    ) -> None:
        ...

    @overload
    def to_string(
        self,
        buf: None = ...,
        *,
        encoding: Optional[str] = ...,
        sparse_index: Optional[bool] = ...,
        sparse_columns: Optional[bool] = ...,
        max_rows: Optional[int] = ...,
        max_columns: Optional[int] = ...,
        delimiter: str = ...,
    ) -> str:
        ...

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_string(
        self,
        buf: Optional[Union[str, Any]] = None,
        *,
        encoding: Optional[str] = None,
        sparse_index: Optional[bool] = None,
        sparse_columns: Optional[bool] = None,
        max_rows: Optional[int] = None,
        max_columns: Optional[int] = None,
        delimiter: str = ' ',
    ) -> Optional[str]:
        """
        Write Styler to a file, buffer or string in text format.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        %(buf)s
        %(encoding)s
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows``, which is None.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).
        delimiter : str, default single space
            The separator between data elements.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_string : Render a DataFrame to a console-friendly tabular output.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.to_string()
        ' A B\\n0 1 3\\n1 2 4\\n'
        """
        obj = self._copy(deep=True)
        if sparse_index is None:
            sparse_index = get_option('styler.sparse.index')
        if sparse_columns is None:
            sparse_columns = get_option('styler.sparse.columns')
        text: str = obj._render_string(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
            delimiter=delimiter,
        )
        return save_to_buffer(text, buf=buf, encoding=(encoding if buf is not None else None))

    def set_td_classes(self, classes: DataFrame) -> Styler:
        """
        Set the ``class`` attribute of ``<td>`` HTML elements.

        Parameters
        ----------
        classes : DataFrame
            DataFrame containing strings that will be translated to CSS classes,
            mapped by identical column and index key values that must exist on the
            underlying Styler data. None, NaN values, and empty strings will
            be ignored and not affect the rendered HTML.

        Returns
        -------
        Styler
            Instance of class with ``class`` attribute set for ``<td>``
                HTML elements.

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Notes
        -----
        Can be used in combination with ``Styler.set_table_styles`` to define an
        internal CSS solution without reference to external CSS files.

        Examples
        --------
        >>> df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        >>> classes = pd.DataFrame(
        ...     [["min-val red", "", "blue"], ["red", None, "blue max-val"]],
        ...     index=df.index,
        ...     columns=df.columns,
        ... )
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Using `MultiIndex` columns and a `classes` `DataFrame` as a subset of the
        underlying,

        >>> df = pd.DataFrame(
        ...     [[1, 2], [3, 4]],
        ...     index=["a", "b"],
        ...     columns=[["level0", "level0"], ["level1a", "level1b"]],
        ... )
        >>> classes = pd.DataFrame(
        ...     ["min-val"], index=["a"], columns=[["level0"], ["level1a"]]
        ... )
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Form of the output with new additional css classes,

        >>> from pandas.io.formats.style import Styler
        >>> df = pd.DataFrame([[1]])
        >>> css = pd.DataFrame([["other-class"]])
        >>> s = Styler(df, uuid="_", cell_ids=False).set_td_classes(css)
        >>> s.hide(axis=0).to_html()  # doctest: +SKIP
        '<style type="text/css"></style>'
        '<table id="T__">'
        '  <thead>'
        '    <tr><th class="col_heading level0 col0" >0</th></tr>'
        '  </thead>'
        '  <tbody>'
        '    <tr><td class="data row0 col0 other-class" >1</td></tr>'
        '  </tbody>'
        '</table>'
        """
        if not classes.index.is_unique or not classes.columns.is_unique:
            raise KeyError('Classes render only if `classes` has unique index and columns.')
        classes = classes.reindex_like(self.data)
        for r, row_tup in enumerate(classes.itertuples(index=True, name=None)):
            for c, value in enumerate(row_tup):
                if c == 0:
                    continue
                if not (pd.isna(value) or value == ''):
                    self.cell_context[r - 1, c - 1] = str(value)
        return self

    def _update_ctx(self, attrs: DataFrame) -> None:
        """
        Update the state of the ``Styler`` for data cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : DataFrame
            should contain strings of '<property>: <value>;<prop2>: <val2>'
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        """
        if not self.index.is_unique or not self.columns.is_unique:
            raise KeyError('`Styler.apply` and `.map` are not compatible with non-unique index or columns.')
        for cn in attrs.columns:
            j: int = self.columns.get_loc(cn)
            ser: Series = attrs[cn]
            for rn, c in ser.items():
                if not (c or pd.isna(c)):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                i: int = self.index.get_loc(rn)
                self.ctx[i, j].extend(css_list)

    def _update_ctx_header(self, attrs: Union[Series, DataFrame], axis: int) -> None:
        """
        Update the state of the ``Styler`` for header cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : Series
            Should contain strings of '<property>: <value>;<prop2>: <val2>', and an
            integer index.
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        axis : int
            Identifies whether the ctx object being updated is the index or columns
        """
        for j in attrs.columns:
            ser: Series = attrs[j]
            for i, c in ser.items():
                if not (c or pd.isna(c)):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                if axis == 0:
                    self.ctx_index[i, j].extend(css_list)
                else:
                    self.ctx_columns[j, i].extend(css_list)

    def _copy(self, deepcopy: bool = False) -> Styler:
        """
        Copies a Styler, allowing for deepcopy or shallow copy

        Copying a Styler aims to recreate a new Styler object which contains the same
        data and styles as the original.

        Data dependent attributes [copied and NOT exported]:
          - formatting (._display_funcs)
          - hidden index values or column values (.hidden_rows, .hidden_columns)
          - tooltips
          - cell_context (cell css classes)
          - ctx (cell css styles)
          - caption
          - concatenated stylers

        Non-data dependent attributes [copied and exported]:
          - css
          - hidden index state and hidden columns state (.hide_index_, .hide_columns_)
          - table_attributes
          - table_styles
          - applied styles (_todo)
        """
        styler = type(self)(self.data)
        shallow = [
            'hide_index_',
            'hide_columns_',
            'hide_column_names',
            'hide_index_names',
            'table_attributes',
            'cell_ids',
            'caption',
            'uuid',
            'uuid_len',
            'template_latex',
            'template_html_style',
            'template_html_table',
            'template_html',
        ]
        deep_attrs = [
            'css',
            'concatenated',
            '_display_funcs',
            '_display_funcs_index',
            '_display_funcs_columns',
            '_display_funcs_index_names',
            '_display_funcs_column_names',
            'hidden_rows',
            'hidden_columns',
            'ctx',
            'ctx_index',
            'ctx_columns',
            'cell_context',
            '_todo',
            'table_styles',
            'tooltips',
        ]
        for attr in shallow:
            setattr(styler, attr, getattr(self, attr))
        for attr in deep_attrs:
            val = getattr(self, attr)
            setattr(styler, attr, copy.deepcopy(val) if deepcopy else val)
        return styler

    def __copy__(self) -> Styler:
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo: Any) -> Styler:
        return self._copy(deepcopy=True)

    def clear(self) -> None:
        """
        Reset the ``Styler``, removing any previously applied styles.

        Returns None.

        See Also
        --------
        Styler.apply : Apply a CSS-styling function column-wise, row-wise,
            or table-wise.
        Styler.export : Export the styles applied to the current Styler.
        Styler.map : Apply a CSS-styling function elementwise.
        Styler.use : Set the styles on the current Styler.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, np.nan]})

        After any added style:

        >>> df.style.highlight_null(color="yellow")  # doctest: +SKIP

        Remove it with:

        >>> df.style.clear()  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        clean_copy = Styler(self.data, uuid=self.uuid)
        clean_attrs = [a for a in clean_copy.__dict__ if not callable(a)]
        self_attrs = [a for a in self.__dict__ if not callable(a)]
        for attr in clean_attrs:
            setattr(self, attr, getattr(clean_copy, attr))
        for attr in set(self_attrs).difference(clean_attrs):
            delattr(self, attr)

    def _apply(
        self,
        func: Callable[[Union[Series, DataFrame]], Union[Series, DataFrame, np.ndarray]],
        axis: Union[int, str, None] = 0,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        **kwargs: Any,
    ) -> Styler:
        """
        Internal apply method.

        To be used with the apply and applymap functions.
        """
        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]
        if data.empty:
            result: Union[DataFrame, Series] = DataFrame()
        elif axis is None:
            result = func(data, **kwargs)
            if not isinstance(result, (DataFrame, np.ndarray)):
                raise TypeError(
                    f'Function {func!r} must return a DataFrame or ndarray when passed to `Styler.apply` with axis=None'
                )
            if isinstance(result, np.ndarray) and data.shape != result.shape:
                raise ValueError(
                    f'Function {func!r} returned ndarray with wrong shape.\nResult has shape: {result.shape}\nExpected shape: {data.shape}'
                )
            if isinstance(result, np.ndarray):
                result = DataFrame(result, index=data.index, columns=data.columns)
        else:
            axis_num = self.data._get_axis_number(axis)
            if axis_num == 0:
                result = data.apply(func, axis=0, **kwargs)
            else:
                result = data.T.apply(func, axis=0, **kwargs).T
        if isinstance(result, Series):
            raise ValueError(
                f'Function {func!r} resulted in the apply method collapsing to a Series.\nUsually, this is the result of the function returning a single value, instead of list-like.'
            )
        msg = (
            f'Function {func!r} created invalid {{0}} labels.\nUsually, this is the result of the function returning a {("Series" if axis is not None else "DataFrame")} which contains invalid labels, or returning an incorrectly shaped, list-like object which cannot be mapped to labels, possibly due to applying the function along the wrong axis.\nResult {{0}} has shape: {{1}}\nExpected {{0}} shape:   {{2}}'
        )
        if not all(result.index.isin(data.index)):
            raise ValueError(msg.format('index', result.index.shape, data.index.shape))
        if not all(result.columns.isin(data.columns)):
            raise ValueError(msg.format('columns', result.columns.shape, data.columns.shape))
        self._update_ctx(result)
        return self

    @doc(name='apply', wise='row-wise', alt='map', altwise='elementwise', func='take a Series and return a string array of the same length', input_note='the index as a Series, if an Index, or a level of a MultiIndex', output_note='an identically sized array of CSS styles as strings', var='label', ret='np.where(label == "B", "background-color: yellow;", "")', ret2='["background-color: yellow;" if "x" in v else "" for v in label]')
    def apply(
        self,
        func: Callable[..., Union[Series, DataFrame, np.ndarray]],
        axis: Union[int, str, None] = 0,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        **kwargs: Any,
    ) -> Styler:
        """
        Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a Series if ``axis`` in [0,1] and return a list-like
            object of same length, or a Series, not necessarily of same length, with
            valid index labels considering ``subset``.
            ``func`` should take a DataFrame if ``axis`` is ``None`` and return either
            an ndarray with the same shape or a DataFrame, not necessarily of the same
            shape, with valid index and columns labels considering ``subset``.

            .. versionchanged:: 1.3.0

            .. versionchanged:: 1.4.0

        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(subset)s
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
            Instance of class with CSS applied to its HTML representation.

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        This is similar to ``DataFrame.apply``, except that ``axis=None``
        applies the function to the entire DataFrame at once,
        rather than column-wise or row-wise.

        Examples
        --------
        >>> def highlight_max(x, color):
        ...     return np.where(x == np.nanmax(x.to_numpy()), f"color: {color};", None)
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.apply(highlight_max, color="red")  # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="blue", axis=1)  # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="green", axis=None)  # doctest: +SKIP

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.apply(highlight_max, color="red", subset="A")
        ... # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="red", subset=["A", "B"])
        ... # doctest: +SKIP

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.apply(highlight_max, color="red", subset=([0, 1, 2], slice(None)))
        ... # doctest: +SKIP
        >>> df.style.apply(highlight_max, color="red", subset=(slice(0, 5, 2), "A"))
        ... # doctest: +SKIP

        Using a function which returns a Series / DataFrame of unequal length but
        containing valid index labels

        >>> df = pd.DataFrame([[1, 2], [3, 4], [4, 6]], index=["A1", "A2", "Total"])
        >>> total_style = pd.Series("font-weight: bold;", index=["Total"])
        >>> df.style.apply(lambda s: total_style)  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        self._todo.append((lambda instance: instance._apply, (func, axis, subset), kwargs))
        return self

    def _apply_index(
        self,
        func: Callable[[Series], Union[Series, List[str], np.ndarray]],
        axis: Union[int, str],
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        method: str = 'apply',
    ) -> Styler:
        """
        Internal apply_index method.

        To be used with the apply_index and map_index functions.
        """
        axis_num = self.data._get_axis_number(axis)
        obj = self.data.index if axis_num == 0 else self.data.columns
        levels_ = refactor_levels(level, obj)
        data = DataFrame(obj.to_list()).loc[:, levels_]
        if method == 'apply':
            result: Union[Series, DataFrame] = data.apply(func, axis=0, **kwargs)
        elif method == 'map':
            result: Union[Series, DataFrame] = data.map(func, **kwargs)
        self._update_ctx_header(result, axis_num)
        return self

    @doc(apply_index, this='map', wise='elementwise', alt='apply', altwise='level-wise', func='take a scalar and return a string', input_note='an index value, if an Index, or a level value of a MultiIndex', output_note='CSS styles as a string', var='label', ret='"background-color: yellow;" if label == "B" else None', ret2='"background-color: yellow;" if "x" in label else None')
    def map_index(
        self,
        func: Callable[[Any], Optional[str]],
        axis: Union[int, str],
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Styler:
        """
        Apply a CSS-styling function to headers elementwise.

        Updates the HTML representation with the result.

        .. versionadded:: 1.4.0

        .. versionadded:: 2.1.0
           Styler.applymap_index was deprecated and renamed to Styler.map_index.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        axis : {0, 1, "index", "columns"}
            The headers over which to apply the function.
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
            Instance of class with CSS-styling function applied elementwise.

        See Also
        --------
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        Each input to ``func`` will be an index value, if an Index, or a level value of a MultiIndex. The output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or, if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        Basic usage to conditionally highlight values in the index.

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"])
        >>> def color_b(label: str) -> Optional[str]:
        ...     return "background-color: yellow;" if label == "B" else None
        >>> df.style.map_index(color_b, axis="index")  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead1.png

        Selectively applying to specific levels of MultiIndex columns.

        >>> midx = pd.MultiIndex.from_product([["ix", "jy"], [0, 1], ["x3", "z4"]])
        >>> df = pd.DataFrame([np.arange(8)], columns=midx)
        >>> def highlight_x(label: str) -> Optional[str]:
        ...     return "background-color: yellow;" if "x" in label else None
        >>> df.style.map_index(
        ...     highlight_x, axis="columns", level=[0, 2]
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead2.png
        """
        self._todo.append(
            (lambda instance: instance._apply_index, (func, axis, level, 'map'), kwargs)
        )
        return self

    def _map(
        self,
        func: Callable[[Any], Optional[str]],
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        **kwargs: Any,
    ) -> Styler:
        func = partial(func, **kwargs)
        if subset is None:
            subset = IndexSlice[:]
        subset = non_reducing_slice(subset)
        result = self.data.loc[subset].map(func)
        self._update_ctx(result)
        return self

    @Substitution(subset=subset_args)
    def map(
        self,
        func: Callable[[Any], Optional[str]],
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        **kwargs: Any,
    ) -> Styler:
        """
        Apply a CSS-styling function elementwise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        %(subset)s
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
            Instance of class with CSS-styling function applied elementwise.

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        >>> def color_negative(v: float, color: str) -> Optional[str]:
        ...     return f"color: {color};" if v < 0 else None
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.map(color_negative, color="red")  # doctest: +SKIP

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.map(color_negative, color="red", subset="A")
        ... # doctest: +SKIP
        >>> df.style.map(color_negative, color="red", subset=["A", "B"])
        ... # doctest: +SKIP

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.map(
        ...     color_negative, color="red", subset=([0, 1, 2], slice(None))
        ... )  # doctest: +SKIP
        >>> df.style.map(color_negative, color="red", subset=(slice(0, 5, 2), "A"))
        ... # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        self._todo.append(
            (lambda instance: instance._map, (func, subset), kwargs)
        )
        return self

    def set_table_attributes(self, attributes: str) -> Styler:
        """
        Set the table attributes added to the ``<table>`` HTML element.

        These are items in addition to automatic (by default) ``id`` attribute.

        Parameters
        ----------
        attributes : str
            Table attributes to be added to the ``<table>`` HTML element.

        Returns
        -------
        Styler
            Instance of class with specified table attributes set.

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
        >>> df.style.set_table_attributes('class="pure-table"')  # doctest: +SKIP
        # ... <table class="pure-table"> ...
        """
        self.table_attributes = attributes
        return self

    def export(self) -> Dict[str, Any]:
        """
        Export the styles applied to the current Styler.

        Can be applied to a second Styler with ``Styler.use``.

        Returns
        -------
        dict
            Contains data-independent (exportable) styles applied to current Styler.

        See Also
        --------
        Styler.use: Set the styles on the current Styler.
        Styler.copy: Create a copy of the current Styler.

        Notes
        -----
        This method is designed to copy non-data dependent attributes of
        one Styler to another. It differs from ``Styler.copy`` where data and
        data dependent attributes are also copied.

        The following items are exported since they are not generally data dependent:

          - Styling functions added by the ``apply`` and ``map``
          - Whether axes and names are hidden from the display, if unambiguous.
          - Table attributes
          - Table styles

        The following attributes are considered data dependent and therefore not
        exported:

          - Caption
          - UUID
          - Tooltips
          - Any hidden rows or columns identified by Index labels
          - Any formatting applied using ``Styler.format``
          - Any CSS classes added using ``Styler.set_td_classes``

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        return {
            'apply': copy.copy(self._todo),
            'table_attributes': self.table_attributes,
            'table_styles': copy.copy(self.table_styles),
            'hide_index': all(self.hide_index_),
            'hide_columns': all(self.hide_columns_),
            'hide_index_names': self.hide_index_names,
            'hide_column_names': self.hide_column_names,
            'css': copy.copy(self.css),
        }

    def use(self, styles: Dict[str, Any]) -> Styler:
        """
        Set the styles on the current Styler.

        Possibly uses styles from ``Styler.export``.

        Parameters
        ----------
        styles : dict(str, Any)
            List of attributes to add to Styler. Dict keys should contain only:
              - "apply": list of styler functions, typically added with ``apply`` or
                ``map``.
              - "table_attributes": HTML attributes, typically added with
                ``set_table_attributes``.
              - "table_styles": CSS selectors and properties, typically added with
                ``set_table_styles``.
              - "hide_index":  whether the index is hidden, typically added with
                ``hide_index``, or a boolean list for hidden levels.
              - "hide_columns": whether column headers are hidden, typically added with
                ``hide_columns``, or a boolean list for hidden levels.
              - "hide_index_names": whether index names are hidden.
              - "hide_column_names": whether column header names are hidden.
              - "css": the css class names used.

        Returns
        -------
        Styler
            Instance of class with defined styler attributes added.

        See Also
        --------
        Styler.export : Export the non data dependent attributes to the current Styler.

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        self._todo.extend(styles.get('apply', []))
        table_attributes = self.table_attributes or ''
        obj_table_atts = '' if styles.get('table_attributes') is None else str(styles.get('table_attributes'))
        self.set_table_attributes((table_attributes + ' ' + obj_table_atts).strip())
        if styles.get('table_styles'):
            self.set_table_styles(styles.get('table_styles'), overwrite=False)
        for obj in ['index', 'columns']:
            hide_obj = styles.get(f'hide_{obj}')
            if hide_obj is not None:
                if isinstance(hide_obj, bool):
                    n = getattr(self, obj).nlevels
                    setattr(self, f'hide_{obj}_', [hide_obj] * n)
                else:
                    setattr(self, f'hide_{obj}_', hide_obj)
        self.hide_index_names = styles.get('hide_index_names', False)
        self.hide_column_names = styles.get('hide_column_names', False)
        if styles.get('css'):
            self.css = styles.get('css')
        return self

    def set_uuid(self, uuid: str) -> Styler:
        """
        Set the uuid applied to ``id`` attributes of HTML elements.

        Parameters
        ----------
        uuid : str
            The uuid to be applied to ``id`` attributes of HTML elements.

        Returns
        -------
        Styler
            Instance of class with specified uuid for `id` attributes set.

        See Also
        --------
        Styler.set_caption : Set the text added to a ``<caption>`` HTML element.
        Styler.set_td_classes : Set the ``class`` attribute of ``<td>`` HTML elements.
        Styler.set_tooltips : Set the DataFrame of strings on ``Styler`` generating
            ``:hover`` tooltips.

        Notes
        -----
        Almost all HTML elements within the table, and including the ``<table>`` element
        are assigned ``id`` attributes. The format is ``T_uuid_<extra>`` where
        ``<extra>`` is typically a more specific identifier, such as ``row1_col2``.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["c1", "c2"])

        You can get the `id` attributes with the following:

        >>> print((df).style.to_html())  # doctest: +SKIP

        To add a title to column `c1`, its `id` is T_20a7d_level0_col0:

        >>> df.style.set_uuid("T_20a7d_level0_col0").set_caption("Test")
        ... # doctest: +SKIP

        Please see:
        `Table visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        self.uuid = uuid
        return self

    def set_caption(self, caption: Union[str, Tuple[str, str]]) -> Styler:
        """
        Set the text added to a ``<caption>`` HTML element.

        Parameters
        ----------
        caption : str, tuple, list
            For HTML output either the string input is used or the first element of the
            tuple. For LaTeX the string input provides a caption and the additional
            tuple input allows for full captions and short captions, in that order.

        Returns
        -------
        Styler
            Instance of class with text set for ``<caption>`` HTML element.

        See Also
        --------
        Styler.set_td_classes : Set the ``class`` attribute of ``<td>`` HTML elements.
        Styler.set_tooltips : Set the DataFrame of strings on ``Styler`` generating
            ``:hover`` tooltips.
        Styler.set_uuid : Set the uuid applied to ``id`` attributes of HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.set_caption("test")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        msg = '`caption` must be either a string or 2-tuple of strings.'
        if isinstance(caption, (list, tuple)):
            if len(caption) != 2 or not isinstance(caption[0], str) or (not isinstance(caption[1], str)):
                raise ValueError(msg)
        elif not isinstance(caption, str):
            raise ValueError(msg)
        self.caption = caption
        return self

    def set_sticky(
        self,
        axis: Union[int, str],
        pixel_size: Optional[int] = None,
        levels: Optional[Union[int, str, List[Union[int, str]]]] = None,
    ) -> Styler:
        """
        Add CSS to permanently display the index or column headers in a scrolling frame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to make the index or column headers sticky.
        pixel_size : int, optional
            Required to configure the width of index cells or the height of column
            header cells when sticking a MultiIndex (or with a named Index).
            Defaults to 75 and 25 respectively.
        levels : int, str, list, optional
            If ``axis`` is a MultiIndex the specific levels to stick. If ``None`` will
            stick all levels.

        Returns
        -------
        Styler
            Instance of class with CSS set for permanently displaying headers
                in scrolling frame.

        See Also
        --------
        Styler.set_properties : Set defined CSS-properties to each ``<td>``
            HTML element for the given subset.

        Notes
        -----
        .. warning::
           This method only works with the output methods ``to_html``, ``to_string``
           and ``to_latex``.

           Other output methods, including ``to_excel``, ignore this hiding method
           and will display all data.

        This method has multiple functionality depending upon the combination
        of the ``subset``, ``level`` and ``names`` arguments (see examples). The
        ``axis`` argument is used only to control whether the method is applied to row
        or column headers:

        .. list-table:: Argument combinations
           :widths: 10 20 10 60
           :header-rows: 1

           * - ``subset``
             - ``level``
             - ``names``
             - Effect
           * - None
             - None
             - False
             - The axis-Index is hidden entirely.
           * - None
             - None
             - True
             - Only the axis-Index names are hidden.
           * - None
             - Int, Str, List
             - False
             - Specified axis-MultiIndex levels are hidden entirely.
           * - None
             - Int, Str, List
             - True
             - Specified axis-MultiIndex levels are hidden entirely and the names of
               remaining axis-MultiIndex levels.
           * - Subset
             - None
             - False
             - The specified data rows/columns are hidden, but the axis-Index itself,
               and names, remain unchanged.
           * - Subset
             - None
             - True
             - The specified data rows/columns and axis-Index names are hidden, but
               the axis-Index itself remains unchanged.
           * - Subset
             - Int, Str, List
             - Boolean
             - ValueError: cannot supply ``subset`` and ``level`` simultaneously.

        Note this method only hides the identified elements so can be chained to hide
        multiple elements in sequence.

        Examples
        --------
        Simple application hiding specific rows:

        >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=["a", "b", "c"])
        >>> df.style.hide(["a", "b"])  # doctest: +SKIP
             0    1
        c    5    6

        Hide the index and retain the data values:

        >>> midx = pd.MultiIndex.from_product([["x", "y"], ["a", "b", "c"]])
        >>> df = pd.DataFrame(np.random.randn(6, 6), index=midx, columns=midx)
        >>> df.style.format("{:.1f}").hide()  # doctest: +SKIP
                            x                    y
              a      b      c      a      b      c
            0.1    0.0    0.4    1.3    0.6   -1.4
            0.7    1.0    1.3    1.5   -0.0   -0.2
            1.4   -0.8    1.6   -0.2   -0.4   -0.3
            0.4    1.0   -0.2   -0.8   -1.2    1.1
           -0.6    1.2    1.8    1.9    0.3    0.3
            0.8    0.5   -0.3    1.2    2.2   -0.8

        Hide specific rows in a MultiIndex but retain the index:

        >>> df.style.format("{:.1f}").hide(subset=(slice(None), ["a", "c"]))
        ... # doctest: +SKIP
                                     x                    y
                       a      b      c      a      b      c
            x   b    0.7    1.0    1.3    1.5   -0.0   -0.2
            y   b   -0.6    1.2    1.8    1.9    0.3    0.3

        Hide specific rows and the index through chaining:

        >>> df.style.format("{:.1f}").hide(subset=(slice(None), ["a", "c"])).hide()
        ... # doctest: +SKIP
                             x                    y
               a      b      c      a      b      c
            0.7    1.0    1.3    1.5   -0.0   -0.2
           -0.6    1.2    1.8    1.9    0.3    0.3

        Hide a specific level:

        >>> df.style.format("{:,.1f}").hide(level=1)  # doctest: +SKIP
                                 x                    y
                   a      b      c      a      b      c
            x    0.1    0.0    0.4    1.3    0.6   -1.4
                0.7    1.0    1.3    1.5   -0.0   -0.2
                1.4   -0.8    1.6   -0.2   -0.4   -0.3
            y    0.4    1.0   -0.2   -0.8   -1.2    1.1
                -0.6    1.2    1.8    1.9    0.3    0.3
                 0.8    0.5   -0.3    1.2    2.2   -0.8

        Hiding just the index level names:

        >>> df.index.names = ["lev0", "lev1"]
        >>> df.style.format("{:,.1f}").hide(names=True)  # doctest: +SKIP
                                     x                    y
                       a      b      c      a      b      c
            x   a    0.1    0.0    0.4    1.3    0.6   -1.4
                b    0.7    1.0    1.3    1.5   -0.0   -0.2
                c    1.4   -0.8    1.6   -0.2   -0.4   -0.3
            y   a    0.4    1.0   -0.2   -0.8   -1.2    1.1
                b   -0.6    1.2    1.8    1.9    0.3    0.3
                c    0.8    0.5   -0.3    1.2    2.2   -0.8

        Examples all produce equivalently transposed effects with ``axis="columns"``.
        """
        axis_num = self.data._get_axis_number(axis)
        if axis_num == 0:
            obj, objs, alt = ('index', 'index', 'rows')
        else:
            obj, objs, alt = ('column', 'columns', 'columns')
        if level is not None and subset is not None:
            raise ValueError('`subset` and `level` cannot be passed simultaneously')
        if subset is None:
            if level is None and getattr(self, f'hide_{obj}_names'):
                setattr(self, f'hide_{obj}_names', True)
                return self
            levels_ = refactor_levels(level, getattr(self, objs)) if level is not None else []
            if level is None and not getattr(self, f'hide_{obj}_names'):
                levels_ = []
            setattr(
                self,
                f'hide_{objs}_',
                [lev in levels_ for lev in range(getattr(self, objs).nlevels)],
            )
        else:
            if axis_num == 0:
                subset_ = IndexSlice[subset, :]
            else:
                subset_ = IndexSlice[:, subset]
            subset = non_reducing_slice(subset_)
            hide = self.data.loc[subset]
            h_els = getattr(self, objs).get_indexer_for(hide.index if axis_num == 0 else hide.columns)
            setattr(self, f'hidden_{alt}', h_els)
        if getattr(self, f'hide_{obj}_names', False):
            setattr(self, f'hide_{obj}_names', True)
        return self

    def _get_numeric_subset_default(self) -> pd.Index:
        return self.data.columns[self.data.select_dtypes(include=np.number).columns]

    @doc(name='background', alt='text', image_prefix='bg', text_threshold='text_color_threshold : float or int\n\n            Luminance threshold for determining text color in [0, 1]. Facilitates text\n\n            visibility across varying background colors. All text is dark if 0, and\n\n            light if 1, defaults to 0.408.')
    @Substitution(subset=subset_args)
    def background_gradient(
        self,
        cmap: Union[str, Colormap] = 'PuBu',
        low: float = 0,
        high: float = 0,
        axis: Union[int, str] = 0,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        text_color_threshold: float = 0.408,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        gmap: Optional[Union[Series, DataFrame, np.ndarray, List[List[float]], List[float]]] = None,
    ) -> Styler:
        """
        Color the background in a gradient style.

        The background color is determined according
        to the data in each column, row or frame, or by a given
        gradient map. Requires matplotlib.

        Parameters
        ----------
        %(subset)s
        cmap : str or colormap
            Matplotlib colormap.
        low : float
            Compress the color range at the low end. This is a multiple of the data
            range to extend below the minimum; good values usually in [0, 1],
            defaults to 0.
        high : float
            Compress the color range at the high end. This is a multiple of the data
            range to extend above the maximum; good values usually in [0, 1],
            defaults to 0.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(subset)s
        text_color_threshold : float or int
            Luminance threshold for determining text color in [0, 1]. Facilitates text
            visibility across varying background colors. All text is dark if 0, and
            light if 1, defaults to 0.408.
        vmin : float, optional
            Minimum data value that corresponds to colormap minimum value.
            If not specified the minimum value of the data (or gmap) will be used.
        vmax : float, optional
            Maximum data value that corresponds to colormap maximum value.
            If not specified the maximum value of the data (or gmap) will be used.
        gmap : array-like, optional
            Gradient map for determining the background colors. If not supplied
            will use the underlying data from rows, columns or frame. If given as an
            ndarray or list-like must be an identical shape to the underlying data
            considering ``axis`` and ``subset``. If given as DataFrame or Series must
            have same index and column labels considering ``axis`` and ``subset``.
            If supplied, ``vmin`` and ``vmax`` should be given relative to this
            gradient map.

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler
            Instance of class with background colored in gradient style.

        See Also
        --------
        Styler.text_gradient: Color the text in a gradient style.

        Notes
        -----
        When using ``low`` and ``high`` the range
        of the gradient, given by the data if ``gmap`` is not given or by ``gmap``,
        is extended at the low end effectively by
        `map.min - low * map.range` and at the high end by
        `map.max + high * map.range` before the colors are normalized and determined.

        If combining with ``vmin`` and ``vmax`` the `map.min`, `map.max` and
        `map.range` are replaced by values according to the values derived from
        ``vmin`` and ``vmax``.

        This method will preselect numeric columns and ignore non-numeric columns
        unless a ``gmap`` is supplied in which case no preselection occurs.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     columns=["City", "Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ...     data=[
        ...         ["Stockholm", 21.6, 5.0, 3.2],
        ...         ["Oslo", 22.4, 13.3, 3.1],
        ...         ["Copenhagen", 24.5, 0.0, 6.7],
        ...     ],
        ... )

        Shading the values column-wise, with ``axis=0``, preselecting numeric columns

        >>> df.style.background_gradient(axis=0)  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_ax0.png

        Shading all values collectively using ``axis=None``

        >>> df.style.background_gradient(axis=None)  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone.png

        Compress the color map from the both ``low`` and ``high`` ends

        >>> df.style.background_gradient(axis=None, low=0.75, high=1.0)  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone_lowhigh.png

        Manually setting ``vmin`` and ``vmax`` gradient thresholds

        >>> df.style.background_gradient(axis=None, vmin=6.7, vmax=21.6)  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone_vminvmax.png

        Setting a ``gmap`` and applying to all columns with another ``cmap``

        >>> df.style.background_gradient(axis=0, gmap=df['Temp (c)'], cmap='YlOrRd')
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/bg_gmap.png

        Setting the gradient map for a dataframe (i.e. ``axis=None``), we need to
        explicitly state ``subset`` to match the ``gmap`` shape

        >>> gmap = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        >>> df.style.background_gradient(
        ...     axis=None, gmap=gmap,
        ...     cmap='YlOrRd', subset=['Temp (c)', 'Rain (mm)', 'Wind (m/s)']
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/bg_axNone_gmap.png

        """
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()
        self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            text_color_threshold=text_color_threshold,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
        )
        return self

    def text_gradient(
        self,
        cmap: Union[str, Colormap] = 'PuBu',
        low: float = 0,
        high: float = 0,
        axis: Union[int, str] = 0,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        gmap: Optional[Union[Series, DataFrame, np.ndarray, List[List[float]], List[float]]] = None,
    ) -> Styler:
        """
        Color the text in a gradient style.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        %(subset)s
        cmap : str or colormap
            Matplotlib colormap.
        low : float
            Compress the color range at the low end. This is a multiple of the data
            range to extend below the minimum; good values usually in [0, 1],
            defaults to 0.
        high : float
            Compress the color range at the high end. This is a multiple of the data
            range to extend above the maximum; good values usually in [0, 1],
            defaults to 0.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(subset)s
        vmin : float, optional
            Minimum data value that corresponds to colormap minimum value.
            If not specified the minimum value of the data (or gmap) will be used.
        vmax : float, optional
            Maximum data value that corresponds to colormap maximum value.
            If not specified the maximum value of the data (or gmap) will be used.
        gmap : array-like, optional
            Gradient map for determining the text colors. If not supplied
            will use the underlying data from rows, columns or frame. If given as an
            ndarray or list-like must be an identical shape to the underlying data
            considering ``axis`` and ``subset``. If given as DataFrame or Series must
            have same index and column labels considering ``axis`` and ``subset``.
            If supplied, ``vmin`` and ``vmax`` should be given relative to this
            gradient map.

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler
            Instance of class with text colored in gradient style.

        See Also
        --------
        Styler.background_gradient: Color the background in a gradient style.

        Notes
        -----
        When using ``low`` and ``high`` the range
        of the gradient, given by the data if ``gmap`` is not given or by ``gmap``,
        is extended at the low end effectively by
        `map.min - low * map.range` and at the high end by
        `map.max + high * map.range` before the colors are normalized and determined.

        If combining with ``vmin`` and ``vmax`` the `map.min`, `map.max` and
        `map.range` are replaced by values according to the values derived from
        ``vmin`` and ``vmax``.

        This method will preselect numeric columns and ignore non-numeric columns
        unless a ``gmap`` is supplied in which case no preselection occurs.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     columns=["City", "Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ...     data=[
        ...         ["Stockholm", 21.6, 5.0, 3.2],
        ...         ["Oslo", 22.4, 13.3, 3.1],
        ...         ["Copenhagen", 24.5, 0.0, 6.7],
        ...     ],
        ... )

        Shading the text values column-wise, with ``axis=0``, preselecting numeric columns

        >>> df.style.text_gradient(axis=0)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_ax0.png

        Shading all text values collectively using ``axis=None``

        >>> df.style.text_gradient(axis=None)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone.png

        Compress the color map from the both ``low`` and ``high`` ends

        >>> df.style.text_gradient(axis=None, low=0.75, high=1.0)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone_lowhigh.png

        Manually setting ``vmin`` and ``vmax`` gradient thresholds

        >>> df.style.text_gradient(axis=None, vmin=6.7, vmax=21.6)  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone_vminvmax.png

        Setting a ``gmap`` and applying to all columns with another ``cmap``

        >>> df.style.text_gradient(axis=0, gmap=df['Temp (c)'], cmap='YlOrRd')
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/tg_gmap.png

        Setting the gradient map for a dataframe (i.e. ``axis=None``), we need to
        explicitly state ``subset`` to match the ``gmap`` shape

        >>> gmap = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        >>> df.style.text_gradient(
        ...     axis=None, gmap=gmap,
        ...     cmap='YlOrRd', subset=['Temp (c)', 'Rain (mm)', 'Wind (m/s)']
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/tg_axNone_gmap.png
        """
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()
        self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            text_color_threshold=0.408,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
            text_only=True,
        )
        return self

    def highlight_null(
        self,
        color: str = 'red',
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        props: Optional[str] = None,
    ) -> Styler:
        """
        Highlight missing values with a style.

        Parameters
        ----------
        %(color)s

            .. versionadded:: 1.5.0

        %(subset)s

        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler
            Instance of class where null values are highlighted with given style.

        See Also
        --------
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, np.nan]})
        >>> df.style.highlight_null(color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        def f(data: DataFrame, props: str) -> np.ndarray:
            return np.where(pd.isna(data).to_numpy(), props, '')

        if props is None:
            props = f'background-color: {color};'
        return self.apply(func=f, axis=None, subset=subset, props=props)

    def highlight_max(
        self,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        color: str = 'yellow',
        axis: Union[int, str] = 0,
        props: Optional[str] = None,
    ) -> Styler:
        """
        Highlight the maximum with a style.

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler
            Instance of class where max value is highlighted in given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [2, 1], "B": [3, 4]})
        >>> df.style.highlight_max(color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        if props is None:
            props = f'background-color: {color};'
        return self.apply(partial(_highlight_value, op='max'), axis=axis, subset=subset, props=props)

    def highlight_min(
        self,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        color: str = 'yellow',
        axis: Union[int, str] = 0,
        props: Optional[str] = None,
    ) -> Styler:
        """
        Highlight the minimum with a style.

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler
            Instance of class where min value is highlighted in given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [2, 1], "B": [3, 4]})
        >>> df.style.highlight_min(color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        if props is None:
            props = f'background-color: {color};'
        return self.apply(partial(_highlight_value, op='min'), axis=axis, subset=subset, props=props)

    def highlight_between(
        self,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        color: str = 'yellow',
        axis: Union[int, str] = 0,
        left: Optional[float] = None,
        right: Optional[float] = None,
        inclusive: str = 'both',
        props: Optional[str] = None,
    ) -> Styler:
        """
        Highlight a defined range with a style.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis along which to determine and highlight quantiles. If ``None`` quantiles
            are measured over the entire DataFrame. See examples.
        left : scalar or datetime-like, or sequence or array-like, default None
            Left bound for defining the range.
        right : scalar or datetime-like, or sequence or array-like, default None
            Right bound for defining the range.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether bounds are closed or open.
        %(props)s

        Returns
        -------
        Styler
            Instance of class with range highlighted in given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Notes
        -----
        .. warning::
           This method only works with the output methods ``to_html``, ``to_string``
           and ``to_latex``.

           Other output methods, including ``to_excel``, ignore this hiding method
           and will display all data.

        This method has multiple functionality depending upon the combination
        of the ``subset``, ``level`` and ``names`` arguments (see examples). The
        ``axis`` argument is used only to control whether the method is applied to row
        or column headers.

        .. list-table:: Argument combinations
           :widths: 10 20 10 60
           :header-rows: 1

           * - ``subset``
             - ``level``
             - ``names``
             - Effect
           * - None
             - None
             - False
             - The axis-Index is hidden entirely.
           * - None
             - None
             - True
             - Only the axis-Index names are hidden.
           * - None
             - Int, Str, List
             - False
             - Specified axis-MultiIndex levels are hidden entirely.
           * - None
             - Int, Str, List
             - True
             - Specified axis-MultiIndex levels are hidden entirely and the names of
               remaining axis-MultiIndex levels.
           * - Subset
             - None
             - False
             - The specified data rows/columns are hidden, but the axis-Index itself,
               and names, remain unchanged.
           * - Subset
             - None
             - True
             - The specified data rows/columns and axis-Index names are hidden, but
               the axis-Index itself remains unchanged.
           * - Subset
             - Int, Str, List
             - Boolean
             - ValueError: cannot supply ``subset`` and ``level`` simultaneously.

        Note this method only hides the identified elements so can be chained to hide
        multiple elements in sequence.

        Examples
        --------
        Basic usage

        >>> df = pd.DataFrame(
        ...     {
        ...         "One": [1.2, 1.6, 1.5],
        ...         "Two": [2.9, 2.1, 2.5],
        ...         "Three": [3.1, 3.2, 3.8],
        ...     }
        ... )
        >>> df.style.highlight_between(left=2.1, right=2.9)  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_basic.png

        Using a range input sequence along an ``axis``, in this case setting a ``left``
        and ``right`` for each column individually

        >>> df.style.highlight_between(
        ...     left=[1.4, 2.4, 3.4], right=[1.6, 2.6, 3.6], axis=1, color="#fffd75"
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_seq.png

        Using ``axis=None`` and providing the ``left`` argument as an array that
        matches the input DataFrame, with a constant ``right``

        >>> df.style.highlight_between(
        ...     left=[[2, 2, 3], [2, 2, 3], [3, 3, 3]],
        ...     right=3.5,
        ...     axis=None,
        ...     color="#fffd75",
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_axNone.png

        Using ``props`` instead of default background coloring

        >>> df.style.highlight_between(
        ...     left=1.5, right=3.5, props="font-weight:bold;color:#e83e8c"
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_props.png
        """
        if props is None:
            props = f'background-color: {color};'
        return self.apply(
            _highlight_between,
            axis=axis,
            subset=subset,
            props=props,
            left=left,
            right=right,
            inclusive=inclusive,
        )

    def highlight_quantile(
        self,
        subset: Optional[Union[Hashable, Sequence[Hashable], Tuple[Union[Hashable, slice], Union[Hashable, slice]]]] = None,
        color: str = 'yellow',
        axis: Union[int, str] = 0,
        q_left: float = 0.0,
        q_right: float = 1.0,
        interpolation: str = 'linear',
        inclusive: str = 'both',
        props: Optional[str] = None,
    ) -> Styler:
        """
        Highlight values defined by a quantile with a style.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis along which to determine and highlight quantiles. If ``None`` quantiles
            are measured over the entire DataFrame. See examples.
        q_left : float, default 0
            Left bound, in [0, q_right), for the target quantile range.
        q_right : float, default 1
            Right bound, in (q_left, 1], for the target quantile range.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Argument passed to ``Series.quantile`` or ``DataFrame.quantile`` for
            quantile estimation.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether quantile bounds are closed or open.
        %(props)s

        Returns
        -------
        Styler
            Instance of class where values in quantile highlighted with given style.

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.

        Notes
        -----
        This function does not work with ``str`` dtypes.

        Examples
        --------
        Using ``axis=None`` and apply a quantile to all collective data

        >>> df = pd.DataFrame(np.arange(10).reshape(2, 5) + 1)
        >>> df.style.highlight_quantile(axis=None, q_left=0.8, color="#fffd75")
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/hq_axNone.png

        Or highlight quantiles row-wise or column-wise, in this case by row-wise

        >>> df.style.highlight_quantile(axis=1, q_left=0.8, color="#fffd75")
        ... # doctest: +SKIP

        .. figure:: ../../_static/style/hq_ax1.png

        Use ``props`` instead of default background coloring

        >>> df.style.highlight_quantile(
        ...     axis=None,
        ...     q_left=0.2,
        ...     q_right=0.8,
        ...     props="font-weight:bold;color:#e83e8c",
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/hq_props.png
        """
        subset_ = slice(None) if subset is None else subset
        subset_ = non_reducing_slice(subset_)
        data = self.data.loc[subset_]
        quantiles = [q_left, q_right]
        if axis is None:
            q = Series(data.to_numpy().ravel()).quantile(q=quantiles, interpolation=interpolation)
            axis_apply = None
        else:
            axis_num = self.data._get_axis_number(axis)
            q = data.quantile(axis=axis_num, numeric_only=False, q=quantiles, interpolation=interpolation)
            axis_apply = 1 - axis_num
        if props is None:
            props = f'background-color: {color};'
        return self.apply(
            _highlight_between,
            axis=axis_apply,
            subset=subset,
            props=props,
            left=q.iloc[0],
            right=q.iloc[1],
            inclusive=inclusive,
        )

    @classmethod
    def from_custom_template(
        cls,
        searchpath: Union[str, List[str]],
        html_table: Optional[str] = None,
        html_style: Optional[str] = None,
    ) -> type[Styler]:
        """
        Factory function for creating a subclass of ``Styler``.

        Uses custom templates and Jinja environment.

        .. versionchanged:: 1.3.0

        Parameters
        ----------
        searchpath : str or list
            Path or paths of directories containing the templates.
        html_table : str
            Name of your custom template to replace the html_table template.

            .. versionadded:: 1.3.0

        html_style : str
            Name of your custom template to replace the html_style template.

            .. versionadded:: 1.3.0

        Returns
        -------
        MyStyler : subclass of Styler
            Has the correct ``env``,``template_html``, ``template_html_table`` and
            ``template_html_style`` class attributes set.

        See Also
        --------
        Styler.export : Export the styles applied to the current Styler.
        Styler.use : Set the styles on the current Styler.

        Examples
        --------
        >>> from pandas.io.formats.style import Styler
        >>> EasyStyler = Styler.from_custom_template(
        ...     "path/to/template",
        ...     "template.tpl",
        ... )  # doctest: +SKIP
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> EasyStyler(df)  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

        class MyStyler(cls):
            env: jinja2.Environment = jinja2.Environment(loader=loader)
            if html_table:
                template_html_table: jinja2.Template = env.get_template(html_table)
            if html_style:
                template_html_style: jinja2.Template = env.get_template(html_style)

        return MyStyler

    @overload
    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        ...

    def pipe(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply ``func(self, *args, **kwargs)``, and return the result.

        Parameters
        ----------
        func : function
            Function to apply to the Styler.  Alternatively, a
            ``(callable, keyword)`` tuple where ``keyword`` is a string
            indicating the keyword of ``callable`` that expects the Styler.
        *args : optional
            Arguments passed to `func`.
        **kwargs : optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object :
            The value returned by ``func``.

        See Also
        --------
        DataFrame.pipe : Analogous method for DataFrame.
        Styler.apply : Apply a CSS-styling function column-wise, row-wise, or
            table-wise.

        Notes
        -----
        Like :meth:`DataFrame.pipe`, this method can simplify the
        application of several user-defined functions to a styler.  Instead
        of writing:

        .. code-block:: python

            f(g(df.style.format(precision=3), arg1=a), arg2=b, arg3=c)

        users can write:

        .. code-block:: python

            (df.style.format(precision=3).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c))

        In particular, this allows users to define functions that take a
        styler object, along with other parameters, and return the styler after
        making styling changes (such as calling :meth:`Styler.apply` or
        :meth:`Styler.set_properties`).

        Examples
        --------

        **Common Use**

        A common usage pattern is to pre-define styling operations which
        can be easily applied to a generic styler in a single ``pipe`` call.

        >>> def some_highlights(styler: Styler, min_color: str = "red", max_color: str = "blue") -> Styler:
        ...     styler.highlight_min(color=min_color, axis=None)
        ...     styler.highlight_max(color=max_color, axis=None)
        ...     styler.highlight_null()
        ...     return styler
        >>> df = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> df.style.pipe(some_highlights, min_color="green")  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_hl.png

        Since the method returns a ``Styler`` object it can be chained with other
        methods as if applying the underlying highlighters directly.

        >>> (
        ...     df.style.format("{:.1f}")
        ...     .pipe(some_highlights, min_color="green")
        ...     .highlight_between(left=2, right=5)
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_hl2.png

        **Advanced Use**

        Sometimes it may be necessary to pre-define styling functions, but in the case
        where those functions rely on the styler, data or context. Since
        ``Styler.use`` and ``Styler.export`` are designed to be non-data dependent,
        they cannot be used for this purpose. Additionally the ``Styler.apply``
        and ``Styler.format`` type methods are not context aware, so a solution
        is to use ``pipe`` to dynamically wrap this functionality.

        Suppose we want to code a generic styling function that highlights the final
        level of a MultiIndex. The number of levels in the Index is dynamic so we
        need the ``Styler`` context to define the level.

        >>> def highlight_last_level(styler: Styler) -> Styler:
        ...     styler.apply_index(
        ...         lambda v: "background-color: pink; color: yellow",
        ...         axis="columns",
        ...         level=styler.columns.nlevels - 1,
        ...     )
        ...     return styler
        >>> df.columns = pd.MultiIndex.from_product([["A", "B"], ["X", "Y"]])
        >>> df.style.pipe(highlight_last_level)  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_applymap.png

        Additionally suppose we want to highlight a column header if there is any
        missing data in that column.
        In this case we need the data object itself to determine the effect on the
        column headers.

        >>> def highlight_header_missing(styler: Styler, level: int) -> Styler:
        ...     def dynamic_highlight(s: Series) -> np.ndarray:
        ...         return np.where(
        ...             styler.data.isna().any(), "background-color: red;", ""
        ...         )
        ...
        ...     styler.apply_index(dynamic_highlight, axis=1, level=level)
        ...     return styler
        >>> df.style.pipe(highlight_header_missing, level=1)  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_applydata.png
        """
        return com.pipe(self, func, *args, **kwargs)

    def _get_numeric_subset_default(self) -> pd.Index:
        return self.data.columns[self.data.select_dtypes(include=np.number).columns]

def _validate_apply_axis_arg(
    arg: Union[Sequence[Any], Series, DataFrame, Any],
    arg_name: str,
    dtype: Optional[np.dtype],
    data: Union[Series, DataFrame],
) -> np.ndarray:
    """
    For the apply-type methods, ``axis=None`` creates ``data`` as DataFrame, and for
    ``axis=[1,0]`` it creates a Series. Where ``arg`` is expected as an element
    of some operator with ``data`` we must make sure that the two are compatible shapes,
    or raise.

    Parameters
    ----------
    arg : sequence, Series or DataFrame
        the user input arg
    arg_name : string
        name of the arg for use in error messages
    dtype : numpy dtype, optional
        forced numpy dtype if given
    data : Series or DataFrame
        underling subset of Styler data on which operations are performed

    Returns
    -------
    ndarray
    """
    dtype_dict = {'dtype': dtype} if dtype else {}
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(f"'{arg_name}' is a Series but underlying data for operations is a DataFrame since 'axis=None'")
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(f"'{arg_name}' is a DataFrame but underlying data for operations is a Series with 'axis in [0,1]'")
    if isinstance(arg, (Series, DataFrame)):
        arg = arg.reindex_like(data).to_numpy(**dtype_dict)
    else:
        arg = np.asarray(arg, **dtype_dict)
        assert isinstance(arg, np.ndarray)
        if arg.shape != data.shape:
            raise ValueError(
                f"supplied '{arg_name}' is not correct shape for data over selected 'axis': got {arg.shape}, expected {data.shape}"
            )
    return arg

def _background_gradient(
    data: Union[Series, DataFrame],
    cmap: Union[str, Colormap] = 'PuBu',
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gmap: Optional[Union[Series, DataFrame, np.ndarray, List[List[float]], List[float]]] = None,
    text_only: bool = False,
) -> Union[List[str], List[List[str]], DataFrame]:
    """
    Color background in a range according to the data or a gradient map
    """
    if gmap is None:
        gmap = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        gmap = _validate_apply_axis_arg(gmap, 'gmap', float, data)
    smin = np.nanmin(gmap) if vmin is None else vmin
    smax = np.nanmax(gmap) if vmax is None else vmax
    rng = smax - smin
    _matplotlib = import_optional_dependency('matplotlib', extra='Styler.background_gradient requires matplotlib.')
    norm = _matplotlib.colors.Normalize(smin - rng * low, smax + rng * high)
    if cmap is None:
        rgbas = _matplotlib.colormaps[_matplotlib.rcParams['image.cmap']](norm(gmap))
    else:
        rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap))

    def relative_luminance(rgba: Tuple[float, ...]) -> float:
        """
        Calculate relative luminance of a color.

        The calculation adheres to the W3C standards
        (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

        Parameters
        ----------
        color : rgb or rgba tuple

        Returns
        -------
        float
            The relative luminance as a value from 0 to 1
        """
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
            for x in rgba[:3]
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba: Tuple[float, ...], text_only: bool) -> str:
        if not text_only:
            dark = relative_luminance(rgba) < text_color_threshold
            text_color: str = '#f1f1f1' if dark else '#000000'
            return f'background-color: {_matplotlib.colors.rgb2hex(rgba)};color: {text_color};'
        else:
            return f'color: {_matplotlib.colors.rgb2hex(rgba)};'

    if isinstance(data, pd.Series):
        return [css(rgba, text_only) for rgba in rgbas]
    else:
        return DataFrame(
            [[css(rgba, text_only) for rgba in row] for row in rgbas],
            index=data.index,
            columns=data.columns,
        )

def _highlight_between(
    data: Union[Series, DataFrame],
    props: str,
    left: Optional[float] = None,
    right: Optional[float] = None,
    inclusive: str = 'both',
) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    """
    if isinstance(left, (list, tuple, np.ndarray)):
        left = _validate_apply_axis_arg(left, 'left', None, data)
    if isinstance(right, (list, tuple, np.ndarray)):
        right = _validate_apply_axis_arg(right, 'right', None, data)
    if inclusive == 'both':
        ops = (operator.ge, operator.le)
    elif inclusive == 'neither':
        ops = (operator.gt, operator.lt)
    elif inclusive == 'left':
        ops = (operator.ge, operator.lt)
    elif inclusive == 'right':
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(f"'inclusive' values can be 'both', 'left', 'right', or 'neither' got {inclusive}")
    g_left = ops[0](data, left) if left is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(g_left, (DataFrame, Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = ops[1](data, right) if right is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(l_right, (DataFrame, Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, '')

def _highlight_value(
    data: Union[Series, DataFrame],
    op: str,
    props: str,
) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, DataFrame):
        value = getattr(value, op)(skipna=True)
    cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, '')
