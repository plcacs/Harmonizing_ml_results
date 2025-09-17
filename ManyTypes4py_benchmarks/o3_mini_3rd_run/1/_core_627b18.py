from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Literal, Any, Union, Sequence, Optional, Tuple, Dict, Callable, Iterable
from types import ModuleType
from pandas._config import get_option
from pandas.util._decorators import Appender, Substitution
from pandas.core.dtypes.common import is_integer, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.base import PandasObject

if TYPE_CHECKING:
    from collections.abc import Hashable
    import types
    from matplotlib.axes import Axes
    import numpy as np
    from pandas._typing import IndexLabel
    from pandas import DataFrame, Index, Series
    from pandas.core.groupby.generic import DataFrameGroupBy

def holds_integer(column: Any) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}

def hist_series(
    self: Any,
    by: Optional[Any] = None,
    ax: Optional["Axes"] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    bins: Union[int, Sequence[int]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any
) -> "Axes":
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
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(self, by=by, ax=ax, grid=grid, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot, figsize=figsize, bins=bins, legend=legend, **kwargs)

def hist_frame(
    data: "DataFrame",
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Any] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    ax: Optional["Axes"] = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    bins: Union[int, Sequence[int]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any
) -> Union["Axes", "np.ndarray"]:
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
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(data, column=column, by=by, grid=grid, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot, ax=ax, sharex=sharex, sharey=sharey, figsize=figsize, layout=layout, legend=legend, bins=bins, **kwargs)

_boxplot_doc = "\nMake a box plot from DataFrame columns.\n\nMake a box-and-whisker plot from DataFrame columns, optionally grouped\nby some other columns. A box plot is a method for graphically depicting\ngroups of numerical data through their quartiles.\nThe box extends from the Q1 to Q3 quartile values of the data,\nwith a line at the median (Q2). The whiskers extend from the edges\nof box to show the range of the data. By default, they extend no more than\n`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest\ndata point within that interval. Outliers are plotted as separate dots.\n\nFor further details see\nWikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.\n\nParameters\n----------\n%(data)scolumn : str or list of str, optional\n    Column name or list of names, or vector.\n    Can be any valid input to :meth:`pandas.DataFrame.groupby`.\nby : str or array-like, optional\n    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.\n    One box-plot will be done per value of columns in `by`.\nax : object of class matplotlib.axes.Axes, optional\n    The matplotlib axes to be used by boxplot.\nfontsize : float or str\n    Tick label font size in points or as a string (e.g., `large`).\nrot : float, default 0\n    The rotation angle of labels (in degrees)\n    with respect to the screen coordinate system.\ngrid : bool, default True\n    Setting this to True will show the grid.\nfigsize : A tuple (width, height) in inches\n    The size of the figure to create in matplotlib.\nlayout : tuple (rows, columns), optional\n    For example, (3, 5) will display the subplots\n    using 3 rows and 5 columns, starting from the top-left.\nreturn_type : {'axes', 'dict', 'both'} or None, default 'axes'\n    The kind of object to return. The default is ``axes``.\n\n    * 'axes' returns the matplotlib axes the boxplot is drawn on.\n    * 'dict' returns a dictionary whose values are the matplotlib\n      Lines of the boxplot.\n    * 'both' returns a namedtuple with the axes and dict.\n    * when grouping with ``by``, a Series mapping columns to\n      ``return_type`` is returned.\n\n      If ``return_type`` is `None`, a NumPy array\n      of axes with the same shape as ``layout`` is returned.\n%(backend)s\n**kwargs\n    All other plotting keyword arguments to be passed to\n    :func:`matplotlib.pyplot.boxplot`.\n\nReturns\n-------\nresult\n    See Notes.\n\nSee Also\n--------\nSeries.plot.hist: Make a histogram.\nmatplotlib.pyplot.boxplot : Matplotlib equivalent plot.\n\nNotes\n-----\nThe return type depends on the `return_type` parameter:\n\n* 'axes' : object of class matplotlib.axes.Axes\n* 'dict' : dict of matplotlib.lines.Line2D objects\n* 'both' : a namedtuple with structure (ax, lines)\n\nFor data grouped with ``by``, return a Series of the above or a numpy\narray:\n\n* :class:`~pandas.Series`\n* :class:`~numpy.array` (for ``return_type = None``)\n\nUse ``return_type='dict'`` when you want to tweak the appearance\nof the lines after plotting. In this case a dict containing the Lines\nmaking up the boxes, caps, fliers, medians, and whiskers is returned.\n"
_backend_doc = "backend : str, default None\n    Backend to use instead of the backend specified in the option\n    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to\n    specify the ``plotting.backend`` for the whole session, set\n    ``pd.options.plotting.backend``.\n"
_bar_or_line_doc = "\n        Parameters\n        ----------\n        x : label or position, optional\n            Allows plotting of one column versus another. If not specified,\n            the index of the DataFrame is used.\n        y : label or position, optional\n            Allows plotting of one column versus another. If not specified,\n            all numerical columns are used.\n        color : str, array-like, or dict, optional\n            The color for each of the DataFrame's columns. Possible values are:\n\n            - A single color string referred to by name, RGB or RGBA code,\n                for instance 'red' or '#a98d19'.\n\n            - A sequence of color strings referred to by name, RGB or RGBA\n                code, which will be used for each column recursively. For\n                instance ['green','yellow'] each column's %(kind)s will be filled in\n                green or yellow, alternatively. If there is only a single column to\n                be plotted, then only the first color from the color list will be\n                used.\n\n            - A dict of the form {column name : color}, so that each column will be\n                colored accordingly. For example, if your columns are called `a` and\n                `b`, then passing {'a': 'green', 'b': 'red'} will color %(kind)ss for\n                column `a` in green and %(kind)ss for column `b` in red.\n\n        **kwargs\n            Additional keyword arguments are documented in\n            :meth:`DataFrame.plot`.\n\n        Returns\n        -------\n        matplotlib.axes.Axes or np.ndarray of them\n            An ndarray is returned with one :class:`matplotlib.axes.Axes`\n            per column when ``subplots=True``.\n"
@Substitution(data='data : DataFrame\n    The data to visualize.\n', backend='')
@Appender(_boxplot_doc)
def boxplot(
    data: "DataFrame",
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[Any]]] = None,
    ax: Optional["Axes"] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = None,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.boxplot(data, column=column, by=by, ax=ax, fontsize=fontsize, rot=rot, grid=grid, figsize=figsize, layout=layout, return_type=return_type, **kwargs)

@Substitution(data='', backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: "DataFrame",
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[Any]]] = None,
    ax: Optional["Axes"] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = None,
    backend: Optional[str] = None,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame(self, column=column, by=by, ax=ax, fontsize=fontsize, rot=rot, grid=grid, figsize=figsize, layout=layout, return_type=return_type, **kwargs)

def boxplot_frame_groupby(
    grouped: "DataFrameGroupBy",
    subplots: bool = True,
    column: Optional[Union[str, Sequence[str]]] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    ax: Optional["Axes"] = None,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = True,
    backend: Optional[str] = None,
    **kwargs: Any
) -> Any:
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
        ``plotting.backend``.
    **kwargs
        All other plotting keyword arguments to be passed to
        matplotlib's boxplot function.

    Returns
    -------
    dict or DataFrame.boxplot return value
        If ``subplots=True``, returns a dictionary of group keys to the boxplot
        return values. If ``subplots=False``, returns the boxplot return value
        of a single DataFrame.
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame_groupby(grouped, subplots=subplots, column=column, fontsize=fontsize, rot=rot, grid=grid, ax=ax, figsize=figsize, layout=layout, sharex=sharex, sharey=sharey, **kwargs)

class PlotAccessor(PandasObject):
    """
    Make plots of Series or DataFrame.

    Uses the backend specified by the
    option ``plotting.backend``. By default, matplotlib is used.

    Parameters
    ----------
    data : Series or DataFrame
        The object for which the method is called.
    """
    def __init__(self, data: Union["Series", "DataFrame"]) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(backend_name: str, data: Union["Series", "DataFrame"], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any, str, Dict[str, Any]]:
        """
        This function makes calls to this accessor `__call__` method compatible
        with the previous `SeriesPlotMethods.__call__` and
        `DataFramePlotMethods.__call__`. Those had slightly different
        signatures, since `DataFramePlotMethods` accepted `x` and `y`
        parameters.
        """
        if isinstance(data, ABCSeries):
            arg_def = [('kind', 'line'), ('ax', None), ('figsize', None), ('use_index', True), ('title', None), ('grid', None), ('legend', False), ('style', None), ('logx', False), ('logy', False), ('loglog', False), ('xticks', None), ('yticks', None), ('xlim', None), ('ylim', None), ('rot', None), ('fontsize', None), ('colormap', None), ('table', False), ('yerr', None), ('xerr', None), ('label', None), ('secondary_y', False), ('xlabel', None), ('ylabel', None)]
        elif isinstance(data, ABCDataFrame):
            arg_def = [('x', None), ('y', None), ('kind', 'line'), ('ax', None), ('subplots', False), ('sharex', None), ('sharey', False), ('layout', None), ('figsize', None), ('use_index', True), ('title', None), ('grid', None), ('legend', True), ('style', None), ('logx', False), ('logy', False), ('loglog', False), ('xticks', None), ('yticks', None), ('xlim', None), ('ylim', None), ('rot', None), ('fontsize', None), ('colormap', None), ('table', False), ('yerr', None), ('xerr', None), ('secondary_y', False), ('xlabel', None), ('ylabel', None)]
        else:
            raise TypeError(f'Called plot accessor for type {type(data).__name__}, expected Series or DataFrame')
        if args and isinstance(data, ABCSeries):
            positional_args = str(args)[1:-1]
            keyword_args = ', '.join([f'{name}={value!r}' for (name, _), value in zip(arg_def, args)])
            msg = f'`Series.plot()` should not be called with positional arguments, only keyword arguments. The order of positional arguments will change in the future. Use `Series.plot({keyword_args})` instead of `Series.plot({positional_args})`.'
            raise TypeError(msg)
        pos_args = {name: value for (name, _), value in zip(arg_def, args)}
        if backend_name == 'pandas.plotting._matplotlib':
            kwargs = dict(arg_def, **pos_args, **kwargs)
        else:
            kwargs = dict(pos_args, **kwargs)
        x = kwargs.pop('x', None)
        y = kwargs.pop('y', None)
        kind = kwargs.pop('kind', 'line')
        return (x, y, kind, kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plot_backend = _get_plot_backend(kwargs.pop('backend', None))
        x, y, kind, kwargs = self._get_call_args(plot_backend.__name__, self._parent, args, kwargs)
        kind = {'density': 'kde'}.get(kind, kind)
        if plot_backend.__name__ != 'pandas.plotting._matplotlib':
            return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)
        if kind not in ('line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box', 'pie', 'scatter', 'hexbin'):
            raise ValueError(f'{kind} is not a valid plot kind Valid plot kinds: {("line", "bar", "barh", "kde", "density", "area", "hist", "box", "pie", "scatter", "hexbin")}')
        data = self._parent
        if isinstance(data, ABCSeries):
            kwargs['reuse_plot'] = True
        if kind in ('scatter', 'hexbin'):
            if isinstance(data, ABCDataFrame):
                return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
            else:
                raise ValueError(f'plot kind {kind} can only be used for data frames')
        elif kind in ('pie',):
            if isinstance(data, ABCDataFrame):
                if y is None and kwargs.get('subplots') is False:
                    raise ValueError(f"{kind} requires either y column or 'subplots=True'")
                if y is not None:
                    if is_integer(y) and (not holds_integer(data.columns)):
                        y = data.columns[y]
                    data = data[y].copy(deep=False)
                    data.index.name = y
        elif isinstance(data, ABCDataFrame):
            data_cols = data.columns
            if x is not None:
                if is_integer(x) and (not holds_integer(data.columns)):
                    x = data_cols[x]
                elif not isinstance(data[x], ABCSeries):
                    raise ValueError('x must be a label or position')
                data = data.set_index(x)
            if y is not None:
                int_ylist = is_list_like(y) and all((is_integer(c) for c in y))
                int_y_arg = is_integer(y) or int_ylist
                if int_y_arg and (not holds_integer(data.columns)):
                    y = data_cols[y]
                label_kw = kwargs['label'] if 'label' in kwargs else False
                for kw in ['xerr', 'yerr']:
                    if kw in kwargs and (isinstance(kwargs[kw], str) or is_integer(kwargs[kw])):
                        try:
                            kwargs[kw] = data[kwargs[kw]]
                        except (IndexError, KeyError, TypeError):
                            pass
                data = data[y]
                if isinstance(data, ABCSeries):
                    label_name = label_kw or y
                    data.name = label_name
                else:
                    match = is_list_like(label_kw) and len(label_kw) == len(y)
                    if label_kw and (not match):
                        raise ValueError('label should be list-like and same length as y')
                    label_name = label_kw or data.columns
                    data.columns = label_name
        return plot_backend.plot(data, kind=kind, **kwargs)
    __call__.__doc__ = __doc__

    @Appender('\n        See Also\n        --------\n        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.\n\n        Examples\n        --------\n\n        .. plot::\n            :context: close-figs\n\n            >>> s = pd.Series([1, 3, 2])\n            >>> s.plot.line()  # doctest: +SKIP\n\n        .. plot::\n            :context: close-figs\n\n            The following example shows the populations for some animals\n            over the years.\n\n            >>> df = pd.DataFrame({\n            ...     'pig': [20, 18, 489, 675, 1776],\n            ...     'horse': [4, 25, 281, 600, 1900]\n            ... }, index=[1990, 1997, 2003, 2009, 2014])\n            >>> lines = df.plot.line()\n\n        .. plot::\n           :context: close-figs\n\n           An example with subplots, so an array of axes is returned.\n\n           >>> axes = df.plot.line(subplots=True)\n           >>> type(axes)\n           <class 'numpy.ndarray'>\n\n        .. plot::\n           :context: close-figs\n\n           Let's repeat the same example, but specifying colors for\n           each column (in this case, for each animal).\n\n           >>> axes = df.plot.line(\n           ...     subplots=True, color={'pig': 'pink', 'horse': '#742802'}\n           ... )\n    ')
    @Substitution(kind='line')
    @Appender(_bar_or_line_doc)
    def line(self, x: Optional[Any] = None, y: Optional[Any] = None, color: Optional[Any] = None, **kwargs: Any) -> Any:
        """
        Plot Series or DataFrame as lines.

        This function is useful to plot lines using DataFrame's values
        as coordinates.
        """
        if color is not None:
            kwargs['color'] = color
        return self(kind='line', x=x, y=y, **kwargs)

    @Appender('\n        See Also\n        --------\n        DataFrame.plot.barh : Horizontal bar plot.\n        DataFrame.plot : Make plots of a DataFrame.\n        matplotlib.pyplot.bar : Make a bar plot with matplotlib.\n\n        Examples\n        --------\n        Basic plot.\n\n        .. plot::\n            :context: close-figs\n\n            >>> df = pd.DataFrame({"lab": ["A", "B", "C"], "val": [10, 30, 20]})\n            >>> ax = df.plot.bar(x="lab", y="val", rot=0)\n\n        Plot a whole dataframe to a bar plot. Each column is assigned a\n        distinct color, and each row is nested in a group along the\n        horizontal axis.\n    ')
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def bar(self, x: Optional[Any] = None, y: Optional[Any] = None, color: Optional[Any] = None, **kwargs: Any) -> Any:
        """
        Vertical bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent.
        """
        if color is not None:
            kwargs['color'] = color
        return self(kind='bar', x=x, y=y, **kwargs)

    @Appender('\n        See Also\n        --------\n        DataFrame.plot.bar : Vertical bar plot.\n        DataFrame.plot : Make plots of DataFrame using matplotlib.\n        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.\n\n        Examples\n        --------\n        Basic example\n\n        .. plot::\n            :context: close-figs\n\n            >>> df = pd.DataFrame({"lab": ["A", "B", "C"], "val": [10, 30, 20]})\n            >>> ax = df.plot.barh(x="lab", y="val")\n    ')
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def barh(self, x: Optional[Any] = None, y: Optional[Any] = None, color: Optional[Any] = None, **kwargs: Any) -> Any:
        """
        Make a horizontal bar plot.

        A horizontal bar plot is a plot that presents quantitative data with
        rectangular bars with lengths proportional to the values that they
        represent.
        """
        if color is not None:
            kwargs['color'] = color
        return self(kind='barh', x=x, y=y, **kwargs)

    def box(self, by: Optional[Union[str, Sequence[Any]]] = None, **kwargs: Any) -> Any:
        """
        Make a box plot of the DataFrame columns.
        """
        return self(kind='box', by=by, **kwargs)

    def hist(self, by: Optional[Union[str, Sequence[Any]]] = None, bins: Union[int, Sequence[int]] = 10, **kwargs: Any) -> Any:
        """
        Draw one histogram of the DataFrame's columns.
        """
        return self(kind='hist', by=by, bins=bins, **kwargs)

    def kde(self, bw_method: Optional[Union[str, float, Callable]] = None, ind: Optional[Union["np.ndarray", int]] = None, weights: Optional["np.ndarray"] = None, **kwargs: Any) -> Any:
        """
        Generate Kernel Density Estimate plot using Gaussian kernels.
        """
        return self(kind='kde', bw_method=bw_method, ind=ind, weights=weights, **kwargs)
    density = kde

    def area(self, x: Optional[Any] = None, y: Optional[Any] = None, stacked: bool = True, **kwargs: Any) -> Any:
        """
        Draw a stacked area plot.
        """
        return self(kind='area', x=x, y=y, stacked=stacked, **kwargs)

    def pie(self, y: Optional[Union[int, str]] = None, **kwargs: Any) -> Any:
        """
        Generate a pie plot.
        """
        if y is not None:
            kwargs['y'] = y
        if isinstance(self._parent, ABCDataFrame) and kwargs.get('y', None) is None and (not kwargs.get('subplots', False)):
            raise ValueError("pie requires either y column or 'subplots=True'")
        return self(kind='pie', **kwargs)

    def scatter(self, x: Union[int, str], y: Union[int, str], s: Optional[Any] = None, c: Optional[Any] = None, **kwargs: Any) -> Any:
        """
        Create a scatter plot with varying marker point size and color.
        """
        return self(kind='scatter', x=x, y=y, s=s, c=c, **kwargs)

    def hexbin(self, x: Union[int, str], y: Union[int, str], C: Optional[Union[int, str]] = None, reduce_C_function: Optional[Callable] = None, gridsize: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: Any) -> Any:
        """
        Generate a hexagonal binning plot.
        """
        if reduce_C_function is not None:
            kwargs['reduce_C_function'] = reduce_C_function
        if gridsize is not None:
            kwargs['gridsize'] = gridsize
        return self(kind='hexbin', x=x, y=y, C=C, **kwargs)

_backends: Dict[str, ModuleType] = {}

def _load_backend(backend: str) -> ModuleType:
    """
    Load a pandas plotting backend.

    Parameters
    ----------
    backend : str
        The identifier for the backend. Either an entrypoint item registered
        with importlib.metadata, "matplotlib", or a module name.

    Returns
    -------
    types.ModuleType
        The imported backend.
    """
    from importlib.metadata import entry_points
    if backend == 'matplotlib':
        try:
            module = importlib.import_module('pandas.plotting._matplotlib')
        except ImportError:
            raise ImportError('matplotlib is required for plotting when the default backend "matplotlib" is selected.') from None
        return module
    found_backend = False
    eps = entry_points()
    key = 'pandas_plotting_backends'
    if hasattr(eps, 'select'):
        entry = eps.select(group=key)
    else:
        entry = eps.get(key, ())
    for entry_point in entry:
        found_backend = entry_point.name == backend
        if found_backend:
            module = entry_point.load()
            break
    if not found_backend:
        try:
            module = importlib.import_module(backend)
            found_backend = True
        except ImportError:
            pass
    if found_backend:
        if hasattr(module, 'plot'):
            return module
    raise ValueError(f"Could not find plotting backend '{backend}'. Ensure that you've installed the package providing the '{backend}' entrypoint, or that the package has a top-level `.plot` method.")

def _get_plot_backend(backend: Optional[str] = None) -> ModuleType:
    """
    Return the plotting backend to use.

    Parameters
    ----------
    backend : str, optional
        The backend identifier.

    Returns
    -------
    ModuleType
        The module providing the plotting backend.
    """
    backend_str: str = backend or get_option('plotting.backend')
    if backend_str in _backends:
        return _backends[backend_str]
    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module