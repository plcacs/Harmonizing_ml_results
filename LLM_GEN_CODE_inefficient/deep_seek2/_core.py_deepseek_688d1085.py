from __future__ import annotations

import importlib
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Union,
    Sequence,
    Dict,
    List,
    Tuple,
    Callable,
    Any,
    Hashable,
    TypeVar,
    overload,
)

from pandas._config import get_option

from pandas.util._decorators import (
    Appender,
    Substitution,
)

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )
    import types

    from matplotlib.axes import Axes
    import numpy as np

    from pandas._typing import IndexLabel

    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby.generic import DataFrameGroupBy


def holds_integer(column: Index) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}


def hist_series(
    self: Series,
    by: Optional[Any] = None,
    ax: Optional[Axes] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    figsize: Optional[Tuple[int, int]] = None,
    bins: Union[int, Sequence[int]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any,
) -> Axes:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(
        self,
        by=by,
        ax=ax,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        figsize=figsize,
        bins=bins,
        legend=legend,
        **kwargs,
    )


def hist_frame(
    data: DataFrame,
    column: Optional[IndexLabel] = None,
    by: Optional[Any] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    ax: Optional[Axes] = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    layout: Optional[Tuple[int, int]] = None,
    bins: Union[int, Sequence[int]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any,
) -> Union[Axes, np.ndarray]:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(
        data,
        column=column,
        by=by,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        ax=ax,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
        legend=legend,
        bins=bins,
        **kwargs,
    )


_boxplot_doc: str = """
Make a box plot from DataFrame columns.

Make a box-and-whisker plot from DataFrame columns, optionally grouped
by some other columns. A box plot is a method for graphically depicting
groups of numerical data through their quartiles.
The box extends from the Q1 to Q3 quartile values of the data,
with a line at the median (Q2). The whiskers extend from the edges
of box to show the range of the data. By default, they extend no more than
`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest
data point within that interval. Outliers are plotted as separate dots.

For further details see
Wikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.

Parameters
----------
%(data)s\
column : str or list of str, optional
    Column name or list of names, or vector.
    Can be any valid input to :meth:`pandas.DataFrame.groupby`.
by : str or array-like, optional
    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.
    One box-plot will be done per value of columns in `by`.
ax : object of class matplotlib.axes.Axes, optional
    The matplotlib axes to be used by boxplot.
fontsize : float or str
    Tick label font size in points or as a string (e.g., `large`).
rot : float, default 0
    The rotation angle of labels (in degrees)
    with respect to the screen coordinate system.
grid : bool, default True
    Setting this to True will show the grid.
figsize : A tuple (width, height) in inches
    The size of the figure to create in matplotlib.
layout : tuple (rows, columns), optional
    For example, (3, 5) will display the subplots
    using 3 rows and 5 columns, starting from the top-left.
return_type : {'axes', 'dict', 'both'} or None, default 'axes'
    The kind of object to return. The default is ``axes``.

    * 'axes' returns the matplotlib axes the boxplot is drawn on.
    * 'dict' returns a dictionary whose values are the matplotlib
      Lines of the boxplot.
    * 'both' returns a namedtuple with the axes and dict.
    * when grouping with ``by``, a Series mapping columns to
      ``return_type`` is returned.

      If ``return_type`` is `None`, a NumPy array
      of axes with the same shape as ``layout`` is returned.
%(backend)s\

**kwargs
    All other plotting keyword arguments to be passed to
    :func:`matplotlib.pyplot.boxplot`.

Returns
-------
result
    See Notes.

See Also
--------
Series.plot.hist: Make a histogram.
matplotlib.pyplot.boxplot : Matplotlib equivalent plot.

Notes
-----
The return type depends on the `return_type` parameter:

* 'axes' : object of class matplotlib.axes.Axes
* 'dict' : dict of matplotlib.lines.Line2D objects
* 'both' : a namedtuple with structure (ax, lines)

For data grouped with ``by``, return a Series of the above or a numpy
array:

* :class:`~pandas.Series`
* :class:`~numpy.array` (for ``return_type = None``)

Use ``return_type='dict'`` when you want to tweak the appearance
of the lines after plotting. In this case a dict containing the Lines
making up the boxes, caps, fliers, medians, and whiskers is returned.

Examples
--------

Boxplots can be created for every column in the dataframe
by ``df.boxplot()`` or indicating the columns to be used:

.. plot::
    :context: close-figs

    >>> np.random.seed(1234)
    >>> df = pd.DataFrame(np.random.randn(10, 4),
    ...                   columns=['Col1', 'Col2', 'Col3', 'Col4'])
    >>> boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])  # doctest: +SKIP

Boxplots of variables distributions grouped by the values of a third
variable can be created using the option ``by``. For instance:

.. plot::
    :context: close-figs

    >>> df = pd.DataFrame(np.random.randn(10, 2),
    ...                   columns=['Col1', 'Col2'])
    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
    ...                      'B', 'B', 'B', 'B', 'B'])
    >>> boxplot = df.boxplot(by='X')

A list of strings (i.e. ``['X', 'Y']``) can be passed to boxplot
in order to group the data by combination of the variables in the x-axis:

.. plot::
    :context: close-figs

    >>> df = pd.DataFrame(np.random.randn(10, 3),
    ...                   columns=['Col1', 'Col2', 'Col3'])
    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
    ...                      'B', 'B', 'B', 'B', 'B'])
    >>> df['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A',
    ...                      'B', 'A', 'B', 'A', 'B'])
    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])

The layout of boxplot can be adjusted giving a tuple to ``layout``:

.. plot::
    :context: close-figs

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      layout=(2, 1))

Additional formatting can be done to the boxplot, like suppressing the grid
(``grid=False``), rotating the labels in the x-axis (i.e. ``rot=45``)
or changing the fontsize (i.e. ``fontsize=15``):

.. plot::
    :context: close-figs

    >>> boxplot = df.boxplot(grid=False, rot=45, fontsize=15)  # doctest: +SKIP

The parameter ``return_type`` can be used to select the type of element
returned by `boxplot`.  When ``return_type='axes'`` is selected,
the matplotlib axes on which the boxplot is drawn are returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')
    >>> type(boxplot)
    <class 'matplotlib.axes._axes.Axes'>

When grouping with ``by``, a Series mapping columns to ``return_type``
is returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      return_type='axes')
    >>> type(boxplot)
    <class 'pandas.Series'>

If ``return_type`` is `None`, a NumPy array of axes with the same shape
as ``layout`` is returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      return_type=None)
    >>> type(boxplot)
    <class 'numpy.ndarray'>
"""

_backend_doc: str = """\
backend : str, default None
    Backend to use instead of the backend specified in the option
    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
    specify the ``plotting.backend`` for the whole session, set
    ``pd.options.plotting.backend``.
"""


_bar_or_line_doc: str = """
        Parameters
        ----------
        x : label or position, optional
            Allows plotting of one column versus another. If not specified,
            the index of the DataFrame is used.
        y : label or position, optional
            Allows plotting of one column versus another. If not specified,
            all numerical columns are used.
        color : str, array-like, or dict, optional
            The color for each of the DataFrame's columns. Possible values are:

            - A single color string referred to by name, RGB or RGBA code,
                for instance 'red' or '#a98d19'.

            - A sequence of color strings referred to by name, RGB or RGBA
                code, which will be used for each column recursively. For
                instance ['green','yellow'] each column's %(kind)s will be filled in
                green or yellow, alternatively. If there is only a single column to
                be plotted, then only the first color from the color list will be
                used.

            - A dict of the form {column name : color}, so that each column will be
                colored accordingly. For example, if your columns are called `a` and
                `b`, then passing {'a': 'green', 'b': 'red'} will color %(kind)ss for
                column `a` in green and %(kind)ss for column `b` in red.

        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            An ndarray is returned with one :class:`matplotlib.axes.Axes`
            per column when ``subplots=True``.
"""


@Substitution(data="data : DataFrame\n    The data to visualize.\n", backend="")
@Appender(_boxplot_doc)
def boxplot(
    data: DataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[str] = None,
    **kwargs: Any,
) -> Union[Axes, np.ndarray]:
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.boxplot(
        data,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


@Substitution(data="", backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: DataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs: Any,
) -> Union[Axes, np.ndarray]:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


def boxplot_frame_groupby(
    grouped: DataFrameGroupBy,
    subplots: bool = True,
    column: Optional[Union[str, List[str]]] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = True,
    backend: Optional[str] = None,
    **kwargs: Any,
) -> Union[Dict[Any, Any], Axes]:
    """
    Make box plots from DataFrameGroupBy data.

    Parameters
    ----------
    grouped : DataFrameGroupBy
        The grouped DataFrame object over which to create the box plots.
    subplots : bool
        * ``False`` - no subplots will be used
        * ``True`` - create a subplot for each group.
    column : column name or list of names, or vector
        Can be any valid input to groupby.
    fontsize : float or str
        Font size for the labels.
    rot : float
        Rotation angle of labels (in degrees) on the x-axis.
    grid : bool
        Whether to show grid lines on the plot.
    ax : Matplotlib axis object, default None
        The axes on which to draw the plots. If None, uses the current axes.
    figsize : tuple of (float, float)
        The figure size in inches (width, height).
    layout : tuple (optional)
        The layout of the plot: (rows, columns).
    sharex : bool, default False
        Whether x-axes will be shared among subplots.
    sharey : bool, default True
        Whether y-axes will be shared among subplots.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        All other plotting keyword arguments to be passed to
        matplotlib's boxplot function.

    Returns
    -------
    dict or DataFrame.boxplot return value
        If ``subplots=True``, returns a dictionary of group keys to the boxplot
        return values. If ``subplots=False``, returns the boxplot return value
        of a single DataFrame.

    See Also
    --------
    DataFrame.boxplot : Create a box plot from a DataFrame.
    Series.plot : Plot a Series.

    Examples
    --------
    You can create boxplots for grouped data and show them as separate subplots:

    .. plot::
        :context: close-figs

        >>> import itertools
        >>> tuples = [t for t in itertools.product(range(1000), range(4))]
        >>> index = pd.MultiIndex.from_tuples(tuples, names=["lvl0", "lvl1"])
        >>> data = np.random.randn(len(index), 4)
        >>> df = pd.DataFrame(data, columns=list("ABCD"), index=index)
        >>> grouped = df.groupby(level="lvl1")
        >>> grouped.boxplot(rot=45, fontsize=12, figsize=(8, 10))  # doctest: +SKIP

    The ``subplots=False`` option shows the boxplots in a single figure.

    .. plot::
        :context: close-figs

        >>> grouped.boxplot(subplots=False, rot=45, fontsize=12)  # doctest: +SKIP
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame_groupby(
        grouped,
        subplots=subplots,
        column=column,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        ax=ax,
        figsize=figsize,
        layout=layout,
        sharex=sharex,
        sharey=sharey,
        **kwargs,
    )


class PlotAccessor(PandasObject):
    """
    Make plots of Series or DataFrame.

    Uses the backend specified by the
    option ``plotting.backend``. By default, matplotlib is used.

    Parameters
    ----------
    data : Series or DataFrame
        The object for which the method is called.

    Attributes
    ----------
    x : label or position, default None
        Only used if data is a DataFrame.
    y : label, position or list of label, positions, default None
        Allows plotting of one column versus another. Only used if data is a
        DataFrame.
    kind : str
        The kind of plot to produce:

        - 'line' : line plot (default)
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : boxplot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot (DataFrame only)
        - 'hexbin' : hexbin plot (DataFrame only)
    ax : matplotlib axes object, default None
        An axes of the current figure.
    subplots : bool or sequence of iterables, default False
        Whether to group columns into subplots:

        - ``False`` : No subplots will be used
        - ``True`` : Make separate subplots for each column.
        - sequence of iterables of column labels: Create a subplot for each
          group of columns. For example `[('a', 'c'), ('b', 'd')]` will
          create 2 subplots: one with columns 'a' and 'c', and one
          with columns 'b' and 'd'. Remaining columns that aren't specified
          will be plotted in additional subplots (one per column).

          .. versionadded:: 1.5.0

    sharex : bool, default True if ax is None else False
        In case ``subplots=True``, share x axis and set some x axis labels
        to invisible; defaults to True if ax is None otherwise False if
        an ax is passed in; Be aware, that passing in both an ax and
        ``sharex=True`` will alter all x axis labels for all axis in a figure.
    sharey : bool, default False
        In case ``subplots=True``, share y axis and set some y axis labels to invisible.
    layout : tuple, optional
        (rows, columns) for the layout of subplots.
    figsize : a tuple (width, height) in inches
        Size of a figure object.
    use_index : bool, default True
        Use index as ticks for x axis.
    title : str or list
        Title to use for the plot. If a string is passed, print the string
        at the top of the figure. If a list is passed and `subplots` is
        True, print each item in the list above the corresponding subplot.
    grid : bool, default None (matlab style default)
        Axis grid lines.
    legend : bool or {'reverse'}
        Place legend on axis subplots.
    style : list or dict
        The matplotlib line style per column.
    logx : bool or 'sym', default False
        Use log scaling or symlog scaling on x axis.

    logy : bool or 'sym' default False
        Use log scaling or symlog scaling on y axis.

    loglog : bool or 'sym', default False
        Use log scaling or symlog scaling on both x and y axes.

    xticks : sequence
        Values to use for the xticks.
    yticks : sequence
        Values to use for the yticks.
    xlim : 2-tuple/list
        Set the x limits of the current axes.
    ylim : 2-tuple/list
        Set the y limits of the current axes.
    xlabel : label, optional
        Name to use for the xlabel on x-axis. Default uses index name as xlabel, or the
        x-column name for planar plots.

        .. versionchanged:: 2.0.0

            Now applicable to histograms.

    ylabel : label, optional
        Name to use for the ylabel on y-axis. Default will show no ylabel, or the
        y-column name for planar plots.

        .. versionchanged:: 2.0.0

            Now applicable to histograms.

    rot : float, default None
        Rotation for ticks (xticks for vertical, yticks for horizontal
        plots).
    fontsize : float, default None
        Font size for xticks and yticks.
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that
        name from matplotlib.
    colorbar : bool, optional
        If True, plot colorbar (only relevant for 'scatter' and 'hexbin'
        plots).
    position : float
        Specify relative alignments for bar plot layout.
        From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
        (center).
    table : bool, Series or DataFrame, default False
        If True, draw a table using the data in the DataFrame and the data
        will be transposed to meet matplotlib's default layout.
        If a Series or DataFrame is passed, use passed data to draw a
        table.
    yerr : DataFrame, Series, array-like, dict and str
        See :ref:`Plotting with Error Bars <visualization.errorbars>` for
        detail.
    xerr : DataFrame, Series, array-like, dict and str
        Equivalent to yerr.
    stacked : bool, default False in line and bar plots, and True in area plot
        If True, create stacked plot.
    secondary_y : bool or sequence, default False
        Whether to plot on the secondary y-axis if a list/tuple, which
        columns to plot on secondary y-axis.
    mark_right : bool, default True
        When using a secondary_y axis, automatically mark the column
        labels with "(right)" in the legend.
    include_bool : bool, default is False
        If True, boolean values can be plotted.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        If the backend is not the default matplotlib one, the return value
        will be the object returned by the backend.

    See Also
    --------
    matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.
    DataFrame.hist : Make a histogram.
    DataFrame.boxplot : Make a box plot.
    DataFrame.plot.scatter : Make a scatter plot with varying marker
        point size and color.
    DataFrame.plot.hexbin : Make a hexagonal binning plot of
        two variables.
    DataFrame.plot.kde : Make Kernel Density Estimate plot using
        Gaussian kernels.
    DataFrame.plot.area : Make a stacked area plot.
    DataFrame.plot.bar : Make a bar plot.
    DataFrame.plot.barh : Make a horizontal bar plot.

    Notes
    -----
    - See matplotlib documentation online for more on this subject
    - If `kind` = 'bar' or 'barh', you can specify relative alignments
      for bar plot layout by `position` keyword.
      From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
      (center)

    Examples
    --------
    For Series:

    .. plot::
        :context: close-figs

        >>> ser = pd.Series([1, 2, 3, 3])
        >>> plot = ser.plot(kind="hist", title="My plot")

    For DataFrame:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame(
        ...     {
        ...         "length": [1.5, 0.5, 1.2, 0.9, 3],
        ...         "width": [0.7, 0.2, 0.15, 0.2, 1.1],
        ...     },
        ...     index=["pig", "rabbit", "duck", "chicken", "horse"],
        ... )
        >>> plot = df.plot(title="DataFrame Plot")

    For SeriesGroupBy:

    .. plot::
        :context: close-figs

        >>> lst = [-1, -2, -3, 1, 2, 3]
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> plot = ser.groupby(lambda x: x > 0).plot(title="SeriesGroupBy Plot")

    For DataFrameGroupBy:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["A", "B", "A", "B"]})
        >>> plot = df.groupby("col2").plot(kind="bar", title="DataFrameGroupBy Plot")
    """

    _common_kinds: Tuple[str, ...] = ("line", "bar", "barh", "kde", "density", "area", "hist", "box")
    _series_kinds: Tuple[str, ...] = ("pie",)
    _dataframe_kinds: Tuple[str, ...] = ("scatter", "hexbin")
    _kind_aliases: Dict[str, str] = {"density": "kde"}
    _all_kinds: Tuple[str, ...] = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Union[Series, DataFrame]) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(backend_name: str, data: Union[Series, DataFrame], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Optional[Hashable], Optional[Hashable], str, Dict[str, Any]]:
        """
        This function makes calls to this accessor `__call__` method compatible
        with the previous `SeriesPlotMethods.__call__` and
        `DataFramePlotMethods.__call__`. Those had slightly different
        signatures, since `DataFramePlotMethods` accepted `x` and `y`
        parameters.
        """
        if isinstance(data, ABCSeries):
            arg_def = [
                ("kind", "line"),
                ("ax", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", False),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("label", None),
                ("secondary_y", False),
                ("xlabel", None),
                ("ylabel", None),
            ]
        elif isinstance(data, ABCDataFrame):
            arg_def = [
                ("x", None),
                ("y", None),
                ("kind", "line"),
                ("ax", None),
                ("subplots", False),
                ("sharex", None),
                ("sharey", False),
                ("layout", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", True),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("secondary_y", False),
                ("xlabel", None),
                ("ylabel", None),
            ]
        else:
            raise TypeError(
                f"Called plot accessor for type {type(data).__name__}, "
                "expected Series or DataFrame"
            )

        if args and isinstance(data, ABCSeries):
            positional_args = str(args)[1:-1]
            keyword_args = ", ".join(
                [f"{name}={value!r}" for (name, _), value in zip(arg_def, args)]
            )
            msg = (
                "`Series.plot()` should not be called with positional "
                "arguments, only keyword arguments. The order of "
                "positional arguments will change in the future. "
                f"Use `Series.plot({keyword_args})` instead of "
                f"`Series.plot({positional_args})`."
            )
            raise TypeError(msg)

        pos_args = {name: value for (name, _), value in zip(arg_def, args)}
        if backend_name == "pandas.plotting._matplotlib":
            kwargs = dict(arg_def, **pos_args, **kwargs)
        else:
            kwargs = dict(pos_args, **kwargs)

        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        kind = kwargs.pop("kind", "line")
        return x, y, kind, kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plot_backend = _get_plot_backend(kwargs.pop("backend", None))

        x, y, kind, kwargs = self._get_call_args(
            plot_backend.__name__, self._parent, args, kwargs
        )

        kind = self._kind_aliases.get(kind, kind)

        # when using another backend, get out of the way
        if plot_backend.__name__ != "pandas.plotting._matplotlib":
            return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)

        if kind not in self._all_kinds:
            raise ValueError(
                f"{kind} is not a valid plot kind Valid plot kinds: {self._all_kinds}"
            )

        data = self._parent

        if isinstance(data, ABCSeries):
            kwargs["reuse_plot"] = True

        if kind in self._dataframe_kinds:
            if isinstance(data, ABCDataFrame):
                return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
            else:
                raise ValueError(f"plot kind {kind} can only be used for data frames")
        elif kind in self._series_kinds:
            if isinstance(data, ABCDataFrame):
                if y is None and kwargs.get("subplots") is False:
                    raise ValueError(
                        f"{kind} requires either y column or 'subplots=True'"
                    )
                if y is not None:
                    if is_integer(y) and not holds_integer(data.columns):
                        y = data.columns[y]
                    # converted to series actually. copy to not modify
                    data = data[y].copy(deep=False)
                    data.index.name = y
        elif isinstance(data, ABCDataFrame):
            data_cols = data.columns
            if x is not None:
                if is_integer(x) and not holds_integer(data.columns):
                    x = data_cols[x]
                elif not isinstance(data[x], ABCSeries):
                    raise ValueError("x must be a label or position")
                data = data.set_index(x)
            if y is not None:
                # check if we have y as int or list of ints
                int_ylist = is_list_like(y) and all(is_integer(c) for c in y)
                int_y_arg = is_integer(y) or int_ylist
                if int_y_arg and not holds_integer(data.columns):
                    y = data_cols[y]

                label_kw = kwargs["label"] if "label" in kwargs else False
                for kw in ["xerr", "yerr"]:
                    if kw in kwargs and (
                        isinstance(kwargs[kw], str) or is_integer(kwargs[kw])
                    ):
                        try:
                            kwargs[kw] = data[kwargs[kw]]
                        except (IndexError, KeyError, TypeError):
                            pass

                data = data[y]

                if isinstance(data, ABCSeries):
                    label_name = label_kw or y
                    data.name = label_name
                else:
                    # error: Argument 1 to "len" has incompatible type "Any | bool";
                    # expected "Sized"  [arg-type]
                    match = is_list_like(label_kw) and len(label_kw) == len(y)  # type: ignore[arg-type]
                    if label_kw and not match:
                        raise ValueError(
                           