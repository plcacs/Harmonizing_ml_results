from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Literal, Optional, Union, Sequence, Tuple, Any, Dict, List
from pandas._config import get_option
from pandas.util._decorators import Appender, Substitution
from pandas.core.dtypes.common import is_integer, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.base import PandasObject

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    import types
    from matplotlib.axes import Axes
    import numpy as np
    from pandas._typing import IndexLabel
    from pandas import DataFrame, Index, Series
    from pandas.core.groupby.generic import DataFrameGroupBy
    from types import ModuleType

def holds_integer(column: ABCSeries) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}

def hist_series(
    self: ABCSeries,
    by: Optional[Hashable] = None,
    ax: Optional[Axes] = None,
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
) -> Axes:
    """
    Draw histogram of the input series using matplotlib.
    """
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
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Hashable] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    ax: Optional[Axes] = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    bins: Union[int, Sequence[float]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any
) -> Union[Axes, np.ndarray]:
    """
    Make a histogram of the DataFrame's columns.
    """
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

_boxplot_doc = "\nMake a box plot from DataFrame columns.\n\nMake a box-and-whisker plot from DataFrame columns, optionally grouped\nby some other columns. A box plot is a method for graphically depicting\ngroups of numerical data through their quartiles.\nThe box extends from the Q1 to Q3 quartile values of the data,\nwith a line at the median (Q2). The whiskers extend from the edges\nof box to show the range of the data. By default, they extend no more than\n`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest\ndata point within that interval. Outliers are plotted as separate dots.\n\nFor further details see\nWikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.\n\nParameters\n----------\n%(data)scolumn : str or list of str, optional\n    Column name or list of names, or vector.\n    Can be any valid input to :meth:`pandas.DataFrame.groupby`.\nby : str or array-like, optional\n    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.\n    One box-plot will be done per value of columns in `by`.\nax : object of class matplotlib.axes.Axes, optional\n    The matplotlib axes to be used by boxplot.\nfontsize : float or str\n    Tick label font size in points or as a string (e.g., `large`).\nrot : float, default 0\n    The rotation angle of labels (in degrees)\n    with respect to the screen coordinate system.\ngrid : bool, default True\n    Setting this to True will show the grid.\nfigsize : A tuple (width, height) in inches\n    The size of the figure to create in matplotlib.\nlayout : tuple (rows, columns), optional\n    For example, (3, 5) will display the subplots\n    using 3 rows and 5 columns, starting from the top-left.\nreturn_type : {'axes', 'dict', 'both'} or None, default 'axes'\n    The kind of object to return. The default is ``axes``.\n\n    * 'axes' returns the matplotlib axes the boxplot is drawn on.\n    * 'dict' returns a dictionary whose values are the matplotlib\n      Lines of the boxplot.\n    * 'both' returns a namedtuple with the axes and dict.\n    * when grouping with ``by``, a Series mapping columns to\n      ``return_type`` is returned.\n\n      If ``return_type`` is `None`, a NumPy array\n      of axes with the same shape as ``layout`` is returned.\n%(backend)s\n**kwargs\n    All other plotting keyword arguments to be passed to\n    :func:`matplotlib.pyplot.boxplot`.\n\nReturns\n-------\nresult\n    See Notes.\n\nSee Also\n--------\nSeries.plot.hist: Make a histogram.\nmatplotlib.pyplot.boxplot : Matplotlib equivalent plot.\n\nNotes\n-----\nThe return type depends on the `return_type` parameter:\n\n* 'axes' : object of class matplotlib.axes.Axes\n* 'dict' : dict of matplotlib.lines.Line2D objects\n* 'both' : a namedtuple with structure (ax, lines)\n\nFor data grouped with ``by``, return a Series of the above or a numpy\narray:\n\n* :class:`~pandas.Series`\n* :class:`~numpy.array` (for ``return_type = None``)\n\nUse ``return_type='dict'`` when you want to tweak the appearance\nof the lines after plotting. In this case a dict containing the Lines\nmaking up the boxes, caps, fliers, medians, and whiskers is returned.\n\nExamples\n--------\n\nBoxplots can be created for every column in the dataframe\nby ``df.boxplot()`` or indicating the columns to be used:\n\n.. plot::\n    :context: close-figs\n\n    >>> np.random.seed(1234)\n    >>> df = pd.DataFrame(np.random.randn(10, 4),\n    ...                   columns=['Col1', 'Col2', 'Col3', 'Col4'])\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])  # doctest: +SKIP\n\nBoxplots of variables distributions grouped by the values of a third\nvariable can be created using the option ``by``. For instance:\n\n.. plot::\n    :context: close-figs\n\n    >>> df = pd.DataFrame(np.random.randn(10, 2),\n    ...                   columns=['Col1', 'Col2'])\n    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',\n    ...                      'B', 'B', 'B', 'B', 'B'])\n    >>> boxplot = df.boxplot(by='X')\n\nA list of strings (i.e. ``['X', 'Y']``) can be passed to boxplot\nin order to group the data by combination of the variables in the x-axis:\n\n.. plot::\n    :context: close-figs\n\n    >>> df = pd.DataFrame(np.random.randn(10, 3),\n    ...                   columns=['Col1', 'Col2', 'Col3'])\n    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',\n    ...                      'B', 'B', 'B', 'B', 'B'])\n    >>> df['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A',\n    ...                      'B', 'A', 'B', 'A', 'B'])\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])\n\nThe layout of boxplot can be adjusted giving a tuple to ``layout``:\n\n.. plot::\n    :context: close-figs\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      layout=(2, 1))\n\nAdditional formatting can be done to the boxplot, like suppressing the grid\n(``grid=False``), rotating the labels in the x-axis (i.e. ``rot=45``)\nor changing the fontsize (i.e. ``fontsize=15``):\n\n.. plot::\n    :context: close-figs\n\n    >>> boxplot = df.boxplot(grid=False, rot=45, fontsize=15)  # doctest: +SKIP\n\nThe parameter ``return_type`` can be used to select the type of element\nreturned by `boxplot`.  When ``return_type='axes'`` is selected,\nthe matplotlib axes on which the boxplot is drawn are returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')\n    >>> type(boxplot)\n    <class 'matplotlib.axes._axes.Axes'>\n\nWhen grouping with ``by``, a Series mapping columns to ``return_type``\nis returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      return_type='axes')\n    >>> type(boxplot)\n    <class 'pandas.Series'>\n\nIf ``return_type`` is `None`, a NumPy array of axes with the same shape\nas ``layout`` is returned:\n\n    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',\n    ...                      return_type=None)\n    >>> type(boxplot)\n    <class 'numpy.ndarray'>\n"

_backend_doc = "backend : str, default None\n    Backend to use instead of the backend specified in the option\n    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to\n    specify the ``plotting.backend`` for the whole session, set\n    ``pd.options.plotting.backend``.\n"

_bar_or_line_doc = "\n        Parameters\n        ----------\n        x : label or position, optional\n            Allows plotting of one column versus another. If not specified,\n            the index of the DataFrame is used.\n        y : label or position, optional\n            Allows plotting of one column versus another. If not specified,\n            all numerical columns are used.\n        color : str, array-like, or dict, optional\n            The color for each of the DataFrame's columns. Possible values are:\n\n            - A single color string referred to by name, RGB or RGBA code,\n                for instance 'red' or '#a98d19'.\n\n            - A sequence of color strings referred to by name, RGB or RGBA\n                code, which will be used for each column recursively. For\n                instance ['green','yellow'] each column's %(kind)s will be filled in\n                green or yellow, alternatively. If there is only a single column to\n                be plotted, then only the first color from the color list will be\n                used.\n\n            - A dict of the form {column name : color}, so that each column will be\n                colored accordingly. For example, if your columns are called `a` and\n                `b`, then passing {'a': 'green', 'b': 'red'} will color %(kind)ss for\n                column `a` in green and %(kind)ss for column `b` in red.\n\n        **kwargs\n            Additional keyword arguments are documented in\n            :meth:`DataFrame.plot`.\n\n        Returns\n        -------\n        matplotlib.axes.Axes or np.ndarray of them\n            An ndarray is returned with one :class:`matplotlib.axes.Axes`\n            per column when ``subplots=True``.\n"

@Substitution(data='data : DataFrame\n    The data to visualize.\n', backend='')
@Appender(_boxplot_doc)
def boxplot(
    data: DataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Hashable] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[Union[int, float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = None,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend('matplotlib')
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

@Substitution(data='', backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: ABCDataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Hashable] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[Union[int, float, str]] = None,
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
    fontsize: Optional[Union[int, float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = True,
    backend: Optional[str] = None,
    **kwargs: Any
) -> Union[Dict[Any, Any], Any]:
    """
    Make box plots from DataFrameGroupBy data.
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
    """

    _common_kinds = ('line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box')
    _series_kinds = ('pie',)
    _dataframe_kinds = ('scatter', 'hexbin')
    _kind_aliases = {'density': 'kde'}
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Union[ABCSeries, ABCDataFrame]) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(
        backend_name: str,
        data: Union[ABCSeries, ABCDataFrame],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Optional[Union[str, int, List[str], List[int]]], Optional[Union[str, int, List[str], List[int]]], str, Dict[str, Any]]:
        """
        This function makes calls to this accessor `__call__` method compatible
        with the previous `SeriesPlotMethods.__call__` and
        `DataFramePlotMethods.__call__`. Those had slightly different
        signatures, since `DataFramePlotMethods` accepted `x` and `y`
        parameters.
        """
        if isinstance(data, ABCSeries):
            arg_def = [
                ('kind', 'line'),
                ('ax', None),
                ('figsize', None),
                ('use_index', True),
                ('title', None),
                ('grid', None),
                ('legend', False),
                ('style', None),
                ('logx', False),
                ('logy', False),
                ('loglog', False),
                ('xticks', None),
                ('yticks', None),
                ('xlim', None),
                ('ylim', None),
                ('rot', None),
                ('fontsize', None),
                ('colormap', None),
                ('table', False),
                ('yerr', None),
                ('xerr', None),
                ('label', None),
                ('secondary_y', False),
                ('xlabel', None),
                ('ylabel', None),
            ]
        elif isinstance(data, ABCDataFrame):
            arg_def = [
                ('x', None),
                ('y', None),
                ('kind', 'line'),
                ('ax', None),
                ('subplots', False),
                ('sharex', None),
                ('sharey', False),
                ('layout', None),
                ('figsize', None),
                ('use_index', True),
                ('title', None),
                ('grid', None),
                ('legend', True),
                ('style', None),
                ('logx', False),
                ('logy', False),
                ('loglog', False),
                ('xticks', None),
                ('yticks', None),
                ('xlim', None),
                ('ylim', None),
                ('rot', None),
                ('fontsize', None),
                ('colormap', None),
                ('table', False),
                ('yerr', None),
                ('xerr', None),
                ('secondary_y', False),
                ('xlabel', None),
                ('ylabel', None),
            ]
        else:
            raise TypeError(
                f'Called plot accessor for type {type(data).__name__}, expected Series or DataFrame'
            )
        if args and isinstance(data, ABCSeries):
            positional_args = str(args)[1:-1]
            keyword_args = ', '.join(
                [f'{name}={value!r}' for (name, _), value in zip(arg_def, args)]
            )
            msg = (
                f'`Series.plot()` should not be called with positional arguments, '
                f'only keyword arguments. The order of positional arguments will change in the future. '
                f'Use `Series.plot({keyword_args})` instead of `Series.plot({positional_args})`.'
            )
            raise TypeError(msg)
        pos_args = {name: value for (name, _), value in zip(arg_def, args)}
        if backend_name == 'pandas.plotting._matplotlib':
            combined_kwargs = {**dict(arg_def), **pos_args, **kwargs}
        else:
            combined_kwargs = {**pos_args, **kwargs}
        x = combined_kwargs.pop('x', None)
        y = combined_kwargs.pop('y', None)
        kind = combined_kwargs.pop('kind', 'line')
        return (x, y, kind, combined_kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plot_backend = _get_plot_backend(kwargs.pop('backend', None))
        x, y, kind, kwargs = self._get_call_args(plot_backend.__name__, self._parent, args, kwargs)
        kind = self._kind_aliases.get(kind, kind)
        if plot_backend.__name__ != 'pandas.plotting._matplotlib':
            return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)
        if kind not in self._all_kinds:
            raise ValueError(f'{kind} is not a valid plot kind Valid plot kinds: {self._all_kinds}')
        data = self._parent
        if isinstance(data, ABCSeries):
            kwargs['reuse_plot'] = True
        if kind in self._dataframe_kinds:
            if isinstance(data, ABCDataFrame):
                return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
            else:
                raise ValueError(f'plot kind {kind} can only be used for data frames')
        elif kind in self._series_kinds:
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

    @Appender(
        '\n        See Also\n        --------\n        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.\n\n        Examples\n        --------\n\n        .. plot::\n            :context: close-figs\n\n            >>> s = pd.Series([1, 3, 2])\n            >>> s.plot.line()  # doctest: +SKIP\n\n        .. plot::\n            :context: close-figs\n\n            The following example shows the populations for some animals\n            over the years.\n\n            >>> df = pd.DataFrame({\n            ...     \'pig\': [20, 18, 489, 675, 1776],\n            ...     \'horse\': [4, 25, 281, 600, 1900]\n            ... }, index=[1990, 1997, 2003, 2009, 2014])\n            >>> lines = df.plot.line()\n\n        .. plot::\n           :context: close-figs\n\n           An example with subplots, so an array of axes is returned.\n\n           >>> axes = df.plot.line(subplots=True)\n           >>> type(axes)\n           <class \'numpy.ndarray\'>\n\n        .. plot::\n           :context: close-figs\n\n           Let\'s repeat the same example, but specifying colors for\n           each column (in this case, for each animal).\n\n           >>> axes = df.plot.line(\n           ...     subplots=True, color={"pig": "pink", "horse": "#742802"}\n           ... )\n\n        .. plot::\n            :context: close-figs\n\n            The following example shows the relationship between both\n            populations.\n\n            >>> lines = df.plot.line(x=\'pig\', y=\'horse\')\n        '
    )
    @Substitution(kind='line')
    @Appender(_bar_or_line_doc)
    def line(
        self,
        x: Optional[Union[str, int]] = None,
        y: Optional[Union[str, int, List[str], List[int]]] = None,
        color: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Plot Series or DataFrame as lines.

        This function is useful to plot lines using DataFrame's values
        as coordinates.
        """
        if color is not None:
            kwargs['color'] = color
        return self(kind='line', x=x, y=y, **kwargs)

    @Appender(
        '\n        See Also\n        --------\n        DataFrame.plot.barh : Horizontal bar plot.\n        DataFrame.plot : Make plots of a DataFrame.\n        matplotlib.pyplot.bar : Make a bar plot with matplotlib.\n\n        Examples\n        --------\n        Basic plot.\n\n        .. plot::\n            :context: close-figs\n\n            >>> df = pd.DataFrame({\'lab\': [\'A\', \'B\', \'C\'], \'val\': [10, 30, 20]})\n            >>> ax = df.plot.bar(x=\'lab\', y=\'val\', rot=0)\n\n        Plot a whole dataframe to a bar plot. Each column is assigned a\n        distinct color, and each row is nested in a group along the\n        horizontal axis.\n\n        .. plot::\n            :context: close-figs\n\n            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]\n            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]\n            >>> index = [\'snail\', \'pig\', \'elephant\',\n            ...          \'rabbit\', \'giraffe\', \'coyote\', \'horse\']\n            >>> df = pd.DataFrame({\'speed\': speed,\n            ...                    \'lifespan\': lifespan}, index=index)\n            >>> ax = df.plot.bar(rot=0)\n\n        Plot stacked bar charts for the DataFrame\n\n        .. plot::\n            :context: close-figs\n\n            >>> ax = df.plot.bar(stacked=True)\n\n        Instead of nesting, the figure can be split by column with\n        ``subplots=True``. In this case, a :class:`numpy.ndarray` of\n        :class:`matplotlib.axes.Axes` are returned.\n\n        .. plot::\n            :context: close-figs\n\n            >>> axes = df.plot.bar(rot=0, subplots=True)\n            >>> axes[1].legend(loc=2)  # doctest: +SKIP\n\n        If you don\'t like the default colours, you can specify how you\'d\n        like each column to be colored.\n\n        .. plot::\n            :context: close-figs\n\n            >>> axes = df.plot.bar(\n            ...     rot=0, subplots=True, color={"speed": "red", "lifespan": "green"}\n            ... )\n            >>> axes[1].legend(loc=2)  # doctest: +SKIP\n\n        Plot a single column.\n\n        .. plot::\n            :context: close-figs\n\n            >>> ax = df.plot.bar(y=\'speed\', rot=0)\n\n        Plot only selected categories for the DataFrame.\n\n        .. plot::\n            :context: close-figs\n\n            >>> ax = df.plot.bar(x=\'lifespan\', rot=0)\n    '
    )
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def bar(
        self,
        x: Optional[Union[str, int]] = None,
        y: Optional[Union[str, int, List[str], List[int]]] = None,
        color: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Vertical bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
        """
        if color is not None:
            kwargs['color'] = color
        return self(kind='bar', x=x, y=y, **kwargs)

    @Appender(
        '\n        See Also\n        --------\n        DataFrame.plot.bar : Vertical bar plot.\n        DataFrame.plot : Make plots of DataFrame using matplotlib.\n        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.\n\n        Examples\n        --------\n        Basic example\n\n        .. plot::\n            :context: close-figs\n\n            >>> df = pd.DataFrame({\'lab\': [\'A\', \'B\', \'C\'], \'val\': [10, 30, 20]})\n            >>> ax = df.plot.barh(x=\'lab\', y=\'val\')\n\n        Plot a whole DataFrame to a horizontal bar plot\n\n        .. plot::\n            :context: close-figs\n\n            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]\n            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]\n            >>> index = [\'snail\', \'pig\', \'elephant\',\n            ...          \'rabbit\', \'giraffe\', \'coyote\', \'horse\']\n            >>> df = pd.DataFrame({\'speed\': speed,\n            ...                    \'lifespan\': lifespan}, index=index)\n            >>> ax = df.plot.barh()\n\n        Plot stacked barh charts for the DataFrame\n\n        .. plot::\n            :context: close-figs\n\n            >>> ax = df.plot.barh(stacked=True)\n\n        We can specify colors for each column\n\n        .. plot::\n            :context: close-figs\n\n            >>> ax = df.plot.barh(color={"speed": "red", "lifespan": "green"})\n\n        Plot a column of the DataFrame to a horizontal bar plot\n\n        .. plot::\n            :context: close-figs\n\n            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]\n            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]\n            >>> index = [\'snail\', \'pig\', \'elephant\',\n            ...          \'rabbit\', \'giraffe\', \'coyote\', \'horse\']\n            >>> df = pd.DataFrame({\'speed\': speed,\n            ...                    \'lifespan\': lifespan}, index=index)\n            >>> ax = df.plot.barh(y=\'speed\')\n\n        Plot DataFrame versus the desired column\n\n        .. plot::\n            :context: close-figs\n\n            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]\n            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]\n            >>> index = [\'snail\', \'pig\', \'elephant\',\n            ...          \'rabbit\', \'giraffe\', \'coyote\', \'horse\']\n            >>> df = pd.DataFrame({\'speed\': speed,\n            ...                    \'lifespan\': lifespan}, index=index)\n            >>> ax = df.plot.barh(x=\'lifespan\')\n    '
    )
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def barh(
        self,
        x: Optional[Union[str, int]] = None,
        y: Optional[Union[str, int, List[str], List[int]]] = None,
        color: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Make a horizontal bar plot.

        A horizontal bar plot is a plot that presents quantitative data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
        """
        if color is not None:
            kwargs['color'] = color
        return self(kind='barh', x=x, y=y, **kwargs)

    def box(
        self,
        by: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Make a box plot of the DataFrame columns.

        A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.
        The box extends from the Q1 to Q3 quartile values of the data,
        with a line at the median (Q2). The whiskers extend from the edges
        of box to show the range of the data. The position of the whiskers
        is set by default to 1.5*IQR (IQR = Q3 - Q1) from the edges of the
        box. Outlier points are those past the end of the whiskers.

        For further details see Wikipedia's
        entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

        A consideration when using this chart is that the box and the whiskers
        can overlap, which is very common when plotting small sets of data.

        Parameters
        ----------
        by : str or sequence
            Column in the DataFrame to group by.

            .. versionchanged:: 1.4.0

               Previously, `by` is silently ignore and makes no groupings

        **kwargs
            Additional keywords are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them
            The matplotlib axes containing the box plot.

        See Also
        --------
        DataFrame.boxplot: Another method to draw a box plot.
        Series.plot.box: Draw a box plot from a Series object.
        matplotlib.pyplot.boxplot: Draw a box plot in matplotlib.

        Examples
        --------
        Draw a box plot from a DataFrame with four columns of randomly
        generated data.

        .. plot::
            :context: close-figs

            >>> data = np.random.randn(25, 4)
            >>> df = pd.DataFrame(data, columns=list("ABCD"))
            >>> ax = df.plot.box()

        You can also generate groupings if you specify the `by` parameter (which
        can take a column name, or a list or tuple of column names):

        .. versionchanged:: 1.4.0

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.box(column="age", by="gender", figsize=(10, 8))
        """
        return self(kind='box', by=by, **kwargs)

    def hist(
        self,
        by: Optional[Union[str, Sequence[str]]] = None,
        bins: int = 10,
        **kwargs: Any
    ) -> Axes:
        """
        Draw one histogram of the DataFrame's columns.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.
        This is useful when the DataFrame's Series are in a similar scale.

        Parameters
        ----------
        by : str or sequence, optional
            Column in the DataFrame to group by.

            .. versionchanged:: 1.4.0

               Previously, `by` is silently ignore and makes no groupings

        bins : int, default 10
            Number of histogram bins to be used.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Return a histogram plot.

        See Also
        --------
        DataFrame.hist : Draw histograms per DataFrame's Series.
        Series.hist : Draw a histogram with Series' data.

        Examples
        --------
        When we roll a die 6000 times, we expect to get each value around 1000
        times. But when we roll two dice and sum the result, the distribution
        is going to be quite different. A histogram illustrates those
        distributions.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.randint(1, 7, 6000), columns=["one"])
            >>> df["two"] = df["one"] + np.random.randint(1, 7, 6000)
            >>> ax = df.plot.hist(bins=12, alpha=0.5)

        A grouped histogram can be generated by providing the parameter `by` (which
        can be a column name, or a list of column names):

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.hist(column=["age"], by="gender", figsize=(10, 8))
        """
        return self(kind='hist', by=by, bins=bins, **kwargs)

    def kde(
        self,
        bw_method: Optional[Union[str, float, Callable[[Any], Any]]] = None,
        ind: Optional[Union[np.ndarray, int]] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Union[Axes, np.ndarray]:
        """
        Generate Kernel Density Estimate plot using Gaussian kernels.

        In statistics, `kernel density estimation`_ (KDE) is a non-parametric
        way to estimate the probability density function (PDF) of a random
        variable. This function uses Gaussian kernels and includes automatic
        bandwidth determination.

        .. _kernel density estimation:
            https://en.wikipedia.org/wiki/Kernel_density_estimation

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth. This can be
            'scott', 'silverman', a scalar constant or a callable.
            If None (default), 'scott' is used.
            See :class:`scipy.stats.gaussian_kde` for more information.
        ind : NumPy array or int, optional
            Evaluation points for the estimated PDF. If None (default),
            1000 equally spaced points are used. If `ind` is a NumPy array, the
            KDE is evaluated at the points passed. If `ind` is an integer,
            `ind` number of equally spaced points are used.
        weights : NumPy array, optional
            Weights of datapoints. This must be the same shape as datapoints.
            If None (default), the samples are assumed to be equally weighted.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them
            The matplotlib axes containing the KDE plot.

        See Also
        --------
        scipy.stats.gaussian_kde : Representation of a kernel-density
            estimate using Gaussian kernels. This is the function used
            internally to estimate the PDF.

        Examples
        --------
        Given a Series of points randomly sampled from an unknown
        distribution, estimate its PDF using KDE with automatic
        bandwidth determination and plot the results, evaluating them at
        1000 equally spaced points (default):

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 2, 2.5, 3, 3.5, 4, 5])
            >>> ax = s.plot.kde()

        A scalar bandwidth can be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(bw_method=3)

        Finally, the `ind` parameter determines the evaluation points for the
        plot of the estimated PDF:

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(ind=[1, 2, 3, 4, 5])

        For DataFrame, it works in the same way:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {
            ...         "x": [1, 2, 2.5, 3, 3.5, 4, 5],
            ...         "y": [4, 4, 4.5, 5, 5.5, 6, 6],
            ...     }
            ... )
            >>> ax = df.plot.kde()

        A scalar bandwidth can be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(bw_method=3)

        Finally, the `ind` parameter determines the evaluation points for the
        plot of the estimated PDF:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(ind=[1, 2, 3, 4, 5, 6])
        """
        return self(kind='kde', bw_method=bw_method, ind=ind, weights=weights, **kwargs)
    density = kde

    def area(
        self,
        x: Optional[Union[str, int]] = None,
        y: Optional[Union[str, int, List[str], List[int]]] = None,
        stacked: bool = True,
        **kwargs: Any
    ) -> Union[Axes, np.ndarray]:
        """
        Draw a stacked area plot.

        An area plot displays quantitative data visually.
        This function wraps the matplotlib area function.

        Parameters
        ----------
        x : label or position, optional
            Coordinates for the X axis. By default uses the index.
        y : label or position, optional
            Column to plot. By default uses all columns.
        stacked : bool, default True
            Area plots are stacked by default. Set to False to create a
            unstacked plot.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Area plot, or array of area plots if subplots is True.

        See Also
        --------
        DataFrame.plot : Make plots of DataFrame using matplotlib.

        Examples
        --------
        Draw an area plot based on basic business metrics:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {
            ...         "sales": [3, 2, 3, 9, 10, 6],
            ...         "signups": [5, 5, 6, 12, 14, 13],
            ...         "visits": [20, 42, 28, 62, 81, 50],
            ...     },
            ...     index=pd.date_range(
            ...         start="2018/01/01", end="2018/07/01", freq="ME"
            ...     ),
            ... )
            >>> ax = df.plot.area()

        Area plots are stacked by default. To produce an unstacked plot,
        pass ``stacked=False``:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(stacked=False)

        Draw an area plot for a single column:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(y="sales")

        Draw with a different `x`:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {
            ...         "sales": [3, 2, 3],
            ...         "visits": [20, 42, 28],
            ...         "day": [1, 2, 3],
            ...     }
            ... )
            >>> ax = df.plot.area(x="day")
        """
        return self(kind='area', x=x, y=y, stacked=stacked, **kwargs)

    def pie(
        self,
        y: Optional[Union[str, int]] = None,
        **kwargs: Any
    ) -> Union[Axes, np.ndarray]:
        """
        Generate a pie plot.

        A pie plot is a proportional representation of the numerical data in a
        column. This function wraps :meth:`matplotlib.pyplot.pie` for the
        specified column. If no column reference is passed and
        ``subplots=True`` a pie plot is drawn for each numerical column
        independently.
        """
        if y is not None:
            kwargs['y'] = y
        if isinstance(self._parent, ABCDataFrame) and kwargs.get('y', None) is None and (not kwargs.get('subplots', False)):
            raise ValueError("pie requires either y column or 'subplots=True'")
        return self(kind='pie', **kwargs)

    def scatter(
        self,
        x: Union[str, int],
        y: Union[str, int],
        s: Optional[Union[str, float, Sequence[float]]] = None,
        c: Optional[Union[str, int, Sequence[str]]] = None,
        **kwargs: Any
    ) -> Axes:
        """
        Create a scatter plot with varying marker point size and color.

        The coordinates of each point are defined by two dataframe columns and
        filled circles are used to represent each point. This kind of plot is
        useful to see complex correlations between two variables. Points could
        be for instance natural 2D coordinates like longitude and latitude in
        a map or, in general, any pair of metrics that can be plotted against
        each other.
        """
        return self(kind='scatter', x=x, y=y, s=s, c=c, **kwargs)

    def hexbin(
        self,
        x: Union[str, int],
        y: Union[str, int],
        C: Optional[Union[str, int]] = None,
        reduce_C_function: Optional[Callable[[Any], Any]] = None,
        gridsize: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs: Any
    ) -> Axes:
        """
        Generate a hexagonal binning plot.

        Generate a hexagonal binning plot of `x` versus `y`. If `C` is `None`
        (the default), this is a histogram of the number of occurrences
        of the observations at ``(x[i], y[i])``.

        If `C` is specified, specifies values at given coordinates
        ``(x[i], y[i])``. These values are accumulated for each hexagonal
        bin and then reduced according to `reduce_C_function`,
        having as default the NumPy's mean function (:meth:`numpy.mean`).
        (If `C` is specified, it must also be a 1-D sequence
        of the same length as `x` and `y`, or a column label.)

        Parameters
        ----------
        x : int or str
            The column label or position for x points.
        y : int or str
            The column label or position for y points.
        C : int or str, optional
            The column label or position for the value of `(x, y)` point.
        reduce_C_function : callable, default `np.mean`
            Function of one argument that reduces all the values in a bin to
            a single number (e.g. `np.mean`, `np.max`, `np.sum`, `np.std`).
        gridsize : int or tuple of (int, int), default 100
            The number of hexagons in the x-direction.
            The corresponding number of hexagons in the y-direction is
            chosen in a way that the hexagons are approximately regular.
            Alternatively, gridsize can be a tuple with two elements
            specifying the number of hexagons in the x-direction and the
            y-direction.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.Axes
            The matplotlib ``Axes`` on which the hexbin is plotted.

        See Also
        --------
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.hexbin : Hexagonal binning plot using matplotlib,
            the matplotlib function that is used under the hood.

        Examples
        --------
        The following examples are generated with random data from
        a normal distribution.

        .. plot::
            :context: close-figs

            >>> n = 10000
            >>> df = pd.DataFrame({"x": np.random.randn(n), "y": np.random.randn(n)})
            >>> ax = df.plot.hexbin(x="x", y="y", gridsize=20)

        The next example uses `C` and `np.sum` as `reduce_C_function`.
        Note that `'observations'` values ranges from 1 to 5 but the result
        plot shows values up to more than 25. This is because of the
        `reduce_C_function`.

        .. plot::
            :context: close-figs

            >>> n = 500
            >>> df = pd.DataFrame(
            ...     {
            ...         "coord_x": np.random.uniform(-3, 3, size=n),
            ...         "coord_y": np.random.uniform(30, 50, size=n),
            ...         "observations": np.random.randint(1, 5, size=n),
            ...     }
            ... )
            >>> ax = df.plot.hexbin(
            ...     x="coord_x",
            ...     y="coord_y",
            ...     C="observations",
            ...     reduce_C_function=np.sum,
            ...     gridsize=10,
            ...     cmap="viridis",
            ... )
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
            raise ImportError(
                'matplotlib is required for plotting when the default backend "matplotlib" is selected.'
            ) from None
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
    raise ValueError(
        f"Could not find plotting backend '{backend}'. Ensure that you've installed the package providing the '{backend}' entrypoint, or that the package has a top-level `.plot` method."
    )

def _get_plot_backend(backend: Optional[str] = None) -> ModuleType:
    """
    Return the plotting backend to use (e.g. `pandas.plotting._matplotlib`).

    The plotting system of pandas uses matplotlib by default, but the idea here
    is that it can also work with other third-party backends. This function
    returns the module which provides a top-level `.plot` method that will
    actually do the plotting. The backend is specified from a string, which
    either comes from the keyword argument `backend`, or, if not specified, from
    the option `pandas.options.plotting.backend`. All the rest of the code in
    this file uses the backend specified there for the plotting.

    The backend is imported lazily, as matplotlib is a soft dependency, and
    pandas can be used without it being installed.

    Notes
    -----
    Modifies `_backends` with imported backend as a side effect.
    """
    backend_str = backend or get_option('plotting.backend')
    if backend_str in _backends:
        return _backends[backend_str]
    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module
