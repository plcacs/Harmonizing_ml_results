from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Literal, Any, Optional, Union, Tuple, List, Dict, Callable, Sequence, Hashable
from pandas._config import get_option
from pandas.util._decorators import Appender, Substitution
from pandas.core.dtypes.common import is_integer, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.base import PandasObject
if TYPE_CHECKING:
    from collections.abc import Callable as CallableABC, Hashable as HashableABC, Sequence as SequenceABC
    import types
    from matplotlib.axes import Axes
    import numpy as np
    from pandas._typing import IndexLabel
    from pandas import DataFrame, Index, Series
    from pandas.core.groupby.generic import DataFrameGroupBy

def holds_integer(column: Any) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}

def hist_series(self: Any, by: Optional[Any] = None, ax: Optional[Any] = None, grid: bool = True, xlabelsize: Optional[int] = None, xrot: Optional[float] = None, ylabelsize: Optional[int] = None, yrot: Optional[float] = None, figsize: Optional[Tuple[float, float]] = None, bins: Union[int, Sequence[float]] = 10, backend: Optional[str] = None, legend: bool = False, **kwargs: Any) -> Any:
    """
    Draw histogram of the input series using matplotlib.

    Parameters
    ----------
    by : object, optional
        If passed, then used to form histograms for separate groups.
    ax : matplotlib axis object
        If not passed, uses gca().
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels.
    figsize : tuple, default None
        Figure size in inches by default.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    legend : bool, default False
        Whether to show the legend.

    **kwargs
        To be passed to the actual plotting function.

    Returns
    -------
    matplotlib.axes.Axes
        A histogram plot.

    See Also
    --------
    matplotlib.axes.Axes.hist : Plot a histogram using matplotlib.

    Examples
    --------
    For Series:

    .. plot::
        :context: close-figs

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> hist = ser.hist()

    For Groupby:

    .. plot::
        :context: close-figs

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> hist = ser.groupby(level=0).hist()
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(self, by=by, ax=ax, grid=grid, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot, figsize=figsize, bins=bins, legend=legend, **kwargs)

def hist_frame(data: Any, column: Optional[Union[str, Sequence[str]]] = None, by: Optional[Any] = None, grid: bool = True, xlabelsize: Optional[int] = None, xrot: Optional[float] = None, ylabelsize: Optional[int] = None, yrot: Optional[float] = None, ax: Optional[Any] = None, sharex: bool = False, sharey: bool = False, figsize: Optional[Tuple[float, float]] = None, layout: Optional[Tuple[int, int]] = None, bins: Union[int, Sequence[float]] = 10, backend: Optional[str] = None, legend: bool = False, **kwargs: Any) -> Any:
    """
    Make a histogram of the DataFrame's columns.

    A `histogram`_ is a representation of the distribution of data.
    This function calls :meth:`matplotlib.pyplot.hist`, on each series in
    the DataFrame, resulting in one histogram per column.

    .. _histogram: https://en.wikipedia.org/wiki/Histogram

    Parameters
    ----------
    data : DataFrame
        The pandas object holding the data.
    column : str or sequence, optional
        If passed, will be used to limit data to a subset of columns.
    by : object, optional
        If passed, then used to form histograms for separate groups.
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels. For example, a value of 90 displays the
        x labels rotated 90 degrees clockwise.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels. For example, a value of 90 displays the
        y labels rotated 90 degrees clockwise.
    ax : Matplotlib axes object, default None
        The axes to plot the histogram on.
    sharex : bool, default True if ax is None else False
        In case subplots=True, share x axis and set some x axis labels to
        invisible; defaults to True if ax is None otherwise False if an ax
        is passed in.
        Note that passing in both an ax and sharex=True will alter all x axis
        labels for all subplots in a figure.
    sharey : bool, default False
        In case subplots=True, share y axis and set some y axis labels to
        invisible.
    figsize : tuple, optional
        The size in inches of the figure to create. Uses the value in
        `matplotlib.rcParams` by default.
    layout : tuple, optional
        Tuple of (rows, columns) for the layout of the histograms.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.

    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.

    legend : bool, default False
        Whether to show the legend.

    **kwargs
        All other plotting keyword arguments to be passed to
        :meth:`matplotlib.pyplot.hist`.

    Returns
    -------
    matplotlib.Axes or numpy.ndarray of them
        Returns a AxesSubplot object a numpy array of AxesSubplot objects.

    See Also
    --------
    matplotlib.pyplot.hist : Plot a histogram using matplotlib.

    Examples
    --------
    This example draws a histogram based on the length and width of
    some animals, displayed in three bins

    .. plot::
        :context: close-figs

        >>> data = {
        ...     "length": [1.5, 0.5, 1.2, 0.9, 3],
        ...     "width": [0.7, 0.2, 0.15, 0.2, 1.1],
        ... }
        >>> index = ["pig", "rabbit", "duck", "chicken", "horse"]
        >>> df = pd.DataFrame(data, index=index)
        >>> hist = df.hist(bins=3)
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(data, column=column, by=by, grid=grid, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot, ax=ax, sharex=sharex, sharey=sharey, figsize=figsize, layout=layout, legend=legend, bins=bins, **kwargs)

_boxplot_doc = "\nMake a box plot from DataFrame columns.\n\nMake a box-and-whisker plot from DataFrame columns, optionally grouped\nby some other columns. A box plot is a method for graphically depicting\ngroups of numerical data through their quartiles.\nThe box extends from the Q1 to Q3 quartile values of the data,\nwith a line at the median (Q2). The whiskers extend from the edges\nof box to show the range of the data. By default, they extend no more than\n`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest\ndata point within that interval. Outliers are plotted as separate dots.\n\nFor further details see\nWikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.\n\nParameters\n----------\n%(data)scolumn : str or list of str, optional\n    Column name or list of names, or vector.\n    Can be any valid input to :meth:`pandas.DataFrame.groupby`.\nby : str or array-like, optional\n    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.\n    One box-plot will be done per value of columns in `by`.\nax : object of class matplotlib.axes.Axes, optional\n    The matplotlib axes to be used by boxplot.\nfontsize : float or str\n    Tick label font size in points or as a string (e.g., `large`).\nrot : float, default 0\n    The rotation angle of labels (in degrees)\n    with respect to the screen coordinate system.\ngrid : bool, default True\n    Setting this to True will show the grid.\nfigsize : A tuple (width, height) in inches\n    The size of the figure to create in matplotlib.\nlayout : tuple (rows, columns), optional\n    For example, (3, 5) will display the subplots\n    using 3 rows and 5 columns, starting from the top-left.\nreturn_type : {'axes', 'dict', 'both'} or None, default 'axes'\n    The kind of object to return. The default is ``axes``.\n\n    * 'axes' returns the matplotlib axes the boxplot is drawn on.\n    * 'dict' returns a dictionary whose values are the matplotlib\n      Lines of the boxplot.\n    * 'both' returns a namedtuple with the axes and dict.\n    * when grouping with ``by``, a Series mapping columns to\n      ``return_type`` is returned.\n\n      If ``return_type`` is `None`, a NumPy array\n      of axes with the same shape as ``layout`` is returned.\n%(backend)s\n**kwargs\n    All other plotting keyword arguments to be passed to\n    :func:`matplotlib.pyplot.boxplot`.\n\nReturns\n-------\nresult\n    See Notes.\n\nSee Also\n--------\nSeries.plot.hist: Make a histogram.\nmatplotlib.pyplot.boxplot : Matplotlib equivalent plot.\n\nNotes\n-----\nThe return type depends on the `return_type` parameter:\n\n* 'axes' : object of class matplotlib.axes.Axes\n* 'dict' : dict of matplotlib.lines.Line2D objects\n* 'both' : a namedtuple with structure (ax, lines)\n\nFor data grouped with ``by``, return a Series of the above or a numpy\narray:\n\n* :class:`~pandas.Series`\n* :class:`~numpy.array` (for ``return_type = None``)\n\nUse ``return_type='dict'`` when you want to tweak the appearance\nof the lines after plotting. In this case a dict containing the Lines\nmaking up the boxes, caps, fliers, medians, and whiskers is returned.\n\nExamples\n--------\n\nBoxplots can be created for every column in the dataframe\nby ``df.boxplot()`` or indicating the columns to be used:\n\n.. plot::\n    :context: close-figs\n\n    >>> np.random.seed(1234)\n    >>> df = pd.DataFrame(np.random.randn(10, 4),\n    ...                   columns=['Col1', 'Col2', 'Col3', 'Col4'])\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])  # doctest: +SKIP\n\nBoxplots of variables distributions grouped by the values of a third\nvariable can be created using the option ``by``. For instance:\n\n.. plot::\n    :context: close-figs\n\n    >>> df = pd.DataFrame(np.random.randn(10, 2),\n    ...                   columns=['Col1', 'Col2'])\n    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',\n    ...                      'B', 'B', 'B', 'B', 'B'])\n    >>> boxplot = df.boxplot(by='X')\n\nA list of strings (i.e. ``['X', 'Y']``) can be passed to boxplot\nin order to group the data by combination of the variables in the x-axis:\n\n.. plot::\n    :context: close-figs\n\n    >>> df = pd.DataFrame(np.random.randn(10, 3),\n    ...                   columns=['Col1', 'Col2', 'Col3'])\n    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',\n    ...                      'B', 'B', 'B', 'B', 'B'])\n    >>> df['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A',\n    ...                      'B', 'A', 'B', 'A', 'B'])\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])\n\nThe layout of boxplot can be adjusted giving a tuple to ``layout``:\n\n.. plot::\n    :context: close-figs\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      layout=(2, 1))\n\nAdditional formatting can be done to the boxplot, like suppressing the grid\n(``grid=False``), rotating the labels in the x-axis (i.e. ``rot=45``)\nor changing the fontsize (i.e. ``fontsize=15``):\n\n.. plot::\n    :context: close-figs\n\n    >>> boxplot = df.boxplot(grid=False, rot=45, fontsize=15)  # doctest: +SKIP\n\nThe parameter ``return_type`` can be used to select the type of element\nreturned by `boxplot`.  When ``return_type='axes'`` is selected,\nthe matplotlib axes on which the boxplot is drawn are returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')\n    >>> type(boxplot)\n    <class 'matplotlib.axes._axes.Axes'>\n\nWhen grouping with ``by``, a Series mapping columns to ``return_type``\nis returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      return_type='axes')\n    >>> type(boxplot)\n    <class 'pandas.Series'>\n\nIf ``return_type`` is `None`, a NumPy array of axes with the same shape\nas ``layout`` is returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      return_type=None)\n    >>> type(boxplot)\n    <class 'numpy.ndarray'>\n"
_backend_doc = "backend : str, default None\n    Backend to use instead of the backend specified in the option\n    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to\n    specify the ``plotting.backend`` for the whole session, set\n    ``pd.options.plotting.backend``.\n"
_bar_or_line_doc = "\n        Parameters\n        ----------\n        x : label or position, optional\n            Allows plotting of one column versus another. If not specified,\n            the index of the DataFrame is used.\n        y : label or position, optional\n            Allows plotting of one column versus another. If not specified,\n            all numerical columns are used.\n        color : str, array-like, or dict, optional\n            The color for each of the DataFrame's columns. Possible values are:\n\n            - A single color string referred to by name, RGB or RGBA code,\n                for instance 'red' or '#a98d19'.\n\n            - A sequence of color strings referred to by name, RGB or RGBA\n                code, which will be used for each column recursively. For\n                instance ['green','yellow'] each column's %(kind)s will be filled in\n                green or yellow, alternatively. If there is only a single column