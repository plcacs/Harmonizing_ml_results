from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Literal, Any, Optional, Union, Tuple, List, Dict, Callable, Sequence, Hashable, TypeVar
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

T = TypeVar('T')

def holds_integer(column: Any) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}

def hist_series(
    self: Any,
    by: Optional[Any] = None,
    ax: Optional[Any] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    bins: Union[int, Sequence[float]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(
        self, by=by, ax=ax, grid=grid, xlabelsize=xlabelsize, xrot=xrot,
        ylabelsize=ylabelsize, yrot=yrot, figsize=figsize, bins=bins,
        legend=legend, **kwargs
    )

def hist_frame(
    data: Any,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Any] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    ax: Optional[Any] = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    bins: Union[int, Sequence[float]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(
        data, column=column, by=by, grid=grid, xlabelsize=xlabelsize, xrot=xrot,
        ylabelsize=ylabelsize, yrot=yrot, ax=ax, sharex=sharex, sharey=sharey,
        figsize=figsize, layout=layout, legend=legend, bins=bins, **kwargs
    )

_boxplot_doc: str = "\nMake a box plot from DataFrame columns.\n\nMake a box-and-whisker plot from DataFrame columns, optionally grouped\nby some other columns. A box plot is a method for graphically depicting\ngroups of numerical data through their quartiles.\nThe box extends from the Q1 to Q3 quartile values of the data,\nwith a line at the median (Q2). The whiskers extend from the edges\nof box to show the range of the data. By default, they extend no more than\n`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest\ndata point within that interval. Outliers are plotted as separate dots.\n\nFor further details see\nWikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.\n\nParameters\n----------\n%(data)scolumn : str or list of str, optional\n    Column name or list of names, or vector.\n    Can be any valid input to :meth:`pandas.DataFrame.groupby`.\nby : str or array-like, optional\n    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.\n    One box-plot will be done per value of columns in `by`.\nax : object of class matplotlib.axes.Axes, optional\n    The matplotlib axes to be used by boxplot.\nfontsize : float or str\n    Tick label font size in points or as a string (e.g., `large`).\nrot : float, default 0\n    The rotation angle of labels (in degrees)\n    with respect to the screen coordinate system.\ngrid : bool, default True\n    Setting this to True will show the grid.\nfigsize : A tuple (width, height) in inches\n    The size of the figure to create in matplotlib.\nlayout : tuple (rows, columns), optional\n    For example, (3, 5) will display the subplots\n    using 3 rows and 5 columns, starting from the top-left.\nreturn_type : {'axes', 'dict', 'both'} or None, default 'axes'\n    The kind of object to return. The default is ``axes``.\n\n    * 'axes' returns the matplotlib axes the boxplot is drawn on.\n    * 'dict' returns a dictionary whose values are the matplotlib\n      Lines of the boxplot.\n    * 'both' returns a namedtuple with the axes and dict.\n    * when grouping with ``by``, a Series mapping columns to\n      ``return_type`` is returned.\n\n      If ``return_type`` is `None`, a NumPy array\n      of axes with the same shape as ``layout`` is returned.\n%(backend)s\n**kwargs\n    All other plotting keyword arguments to be passed to\n    :func:`matplotlib.pyplot.boxplot`.\n\nReturns\n-------\nresult\n    See Notes.\n\nSee Also\n--------\nSeries.plot.hist: Make a histogram.\nmatplotlib.pyplot.boxplot : Matplotlib equivalent plot.\n\nNotes\n-----\nThe return type depends on the `return_type` parameter:\n\n* 'axes' : object of class matplotlib.axes.Axes\n* 'dict' : dict of matplotlib.lines.Line2D objects\n* 'both' : a namedtuple with structure (ax, lines)\n\nFor data grouped with ``by``, return a Series of the above or a numpy\narray:\n\n* :class:`~pandas.Series`\n* :class:`~numpy.array` (for ``return_type = None``)\n\nUse ``return_type='dict'`` when you want to tweak the appearance\nof the lines after plotting. In this case a dict containing the Lines\nmaking up the boxes, caps, fliers, medians, and whiskers is returned.\n\nExamples\n--------\n\nBoxplots can be created for every column in the dataframe\nby ``df.boxplot()`` or indicating the columns to be used:\n\n.. plot::\n    :context: close-figs\n\n    >>> np.random.seed(1234)\n    >>> df = pd.DataFrame(np.random.randn(10, 4),\n    ...                   columns=['Col1', 'Col2', 'Col3', 'Col4'])\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])  # doctest: +SKIP\n\nBoxplots of variables distributions grouped by the values of a third\nvariable can be created using the option ``by``. For instance:\n\n.. plot::\n    :context: close-figs\n\n    >>> df = pd.DataFrame(np.random.randn(10, 2),\n    ...                   columns=['Col1', 'Col2'])\n    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',\n    ...                      'B', 'B', 'B', 'B', 'B'])\n    >>> boxplot = df.boxplot(by='X')\n\nA list of strings (i.e. ``['X', 'Y']``) can be passed to boxplot\nin order to group the data by combination of the variables in the x-axis:\n\n.. plot::\n    :context: close-figs\n\n    >>> df = pd.DataFrame(np.random.randn(10, 3),\n    ...                   columns=['Col1', 'Col2', 'Col3'])\n    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',\n    ...                      'B', 'B', 'B', 'B', 'B'])\n    >>> df['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A',\n    ...                      'B', 'A', 'B', 'A', 'B'])\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])\n\nThe layout of boxplot can be adjusted giving a tuple to ``layout``:\n\n.. plot::\n    :context: close-figs\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      layout=(2, 1))\n\nAdditional formatting can be done to the boxplot, like suppressing the grid\n(``grid=False``), rotating the labels in the x-axis (i.e. ``rot=45``)\nor changing the fontsize (i.e. ``fontsize=15``):\n\n.. plot::\n    :context: close-figs\n\n    >>> boxplot = df.boxplot(grid=False, rot=45, fontsize=15)  # doctest: +SKIP\n\nThe parameter ``return_type`` can be used to select the type of element\nreturned by `boxplot`.  When ``return_type='axes'`` is selected,\nthe matplotlib axes on which the boxplot is drawn are returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')\n    >>> type(boxplot)\n    <class 'matplotlib.axes._axes.Axes'>\n\nWhen grouping with ``by``, a Series mapping columns to ``return_type``\nis returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      return_type='axes')\n    >>> type(boxplot)\n    <class 'pandas.Series'>\n\nIf ``return_type`` is `None`, a NumPy array of axes with the same shape\nas ``layout`` is returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      return_type=None)\n    >>> type(boxplot)\n    <class 'numpy.ndarray'>\n"
_backend_doc: str = "backend : str, default None\n    Backend to use instead of the backend specified in the option\n    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to\n    specify the ``plotting.backend`` for the whole session, set\n    ``pd.options.plotting.backend``.\n"
_bar_or_line_doc: str = "\n        Parameters\n        ----------\n        x : label or position, optional\n            Allows plotting of one column versus another. If not specified,\n            the index of the DataFrame is used.\n        y : label, position or list of label, positions, default None\n            Allows plotting of one column versus another. If not specified,\n            all numerical columns are used.\n        color : str, array-like, or dict, optional\n            The color for each of the DataFrame's columns. Possible values are:\n\n            - A single color string referred to by name, RGB or RGBA code,\n                for instance 'red' or '#a98d19'.\n\n            - A sequence of color strings referred to by name, RGB or RGBA\n                code, which will be used for each column recursively. For\n                instance ['green','yellow'] each column's %(kind)s will be filled in\n                green or yellow, alternatively. If there is only a single column to\n                be plotted, then only the first color from the color list will be\n                used.\n\n            - A dict of the form {column name : color}, so that each column will be\n                colored accordingly. For example, if your columns are called `a` and\n                `b`, then passing {'a': 'green', 'b': 'red'} will color %(kind)ss for\n                column `a` in green and %(kind)ss for column `b` in red.\n\n        **kwargs\n            Additional keyword arguments are documented in\n            :meth:`DataFrame.plot`.\n\n        Returns\n        -------\n        matplotlib.axes.Axes or np.ndarray of them\n            An ndarray is returned with one :class:`matplotlib.axes.Axes`\n            per column when ``subplots=True``.\n"

@Substitution(data='data : DataFrame\n    The data to visualize.\n', backend='')
@Appender(_boxplot_doc)
def boxplot(
    data: Any,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Any] = None,
    ax: Optional[Any] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = None,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.boxplot(
        data, column=column, by=by, ax=ax, fontsize=fontsize, rot=rot,
        grid=grid, figsize=figsize, layout=layout, return_type=return_type, **kwargs
    )

@Substitution(data='', backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: Any,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Any] = None,
    ax: Optional[Any] = None,
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
    return plot_backend.boxplot_frame(
        self, column=column, by=by, ax=ax, fontsize=fontsize, rot=rot,
        grid=grid, figsize=figsize, layout=layout, return_type=return_type, **kwargs
    )

def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: Optional[Union[str, List[str]]] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = True,
    backend: Optional[str] = None,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame_groupby(
        grouped, subplots=subplots, column=column, fontsize=fontsize, rot=rot,
        grid=grid, ax=ax, figsize=figsize, layout=layout, sharex=sharex,
        sharey=sharey, **kwargs
    )

class PlotAccessor(PandasObject):
    _common_kinds: Tuple[str, ...] = ('line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box')
    _series_kinds: Tuple[str, ...] = ('pie',)
    _dataframe_kinds: Tuple[str, ...] = ('scatter', 'hexbin')
    _kind_aliases: Dict[str, str] = {'density': 'kde'}
    _all_kinds: Tuple[str, ...] = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Any) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(
        backend_name: str,
        data: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any]
    ) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any]]:
        if isinstance(data, ABCSeries):
            arg_def = [('kind', 'line'), ('ax', None), ('figsize', None), ('use_index', True), ('title', None), ('grid', None), ('legend', False), ('style', None), ('logx', False), ('logy', False), ('loglog', False), ('xticks', None), ('yticks', None), ('xlim', None), ('ylim', None), ('rot', None), ('fontsize', None), ('colormap', None), ('table', False), ('yerr', None), ('xerr', None), ('label', None), ('secondary_y', False), ('xlabel', None), ('ylabel', None)]
        elif isinstance(data, ABCDataFrame):
            arg_def = [('x', None), ('y', None), ('kind', 'line'), ('ax', None), ('subplots', False), ('sharex', None), ('sharey', False), ('layout', None), ('fig