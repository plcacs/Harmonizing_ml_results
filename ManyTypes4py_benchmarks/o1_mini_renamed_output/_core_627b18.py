from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Literal, Any, Iterable, Optional, Union, Callable
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


def func_zo1u1t9h(column: Series) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}


def func_215yl7tf(
    self: Series,
    by: Hashable | Iterable[Hashable] | None = None,
    ax: Axes | None = None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot: float | None = None,
    ylabelsize: int | None = None,
    yrot: float | None = None,
    figsize: tuple[float, float] | None = None,
    bins: int | Sequence[float] = 10,
    backend: str | None = None,
    legend: bool = False,
    **kwargs: Any,
) -> Axes:
    """
    Draw histogram of the input series using matplotlib.

    [Docstring omitted for brevity]
    """
    plot_backend = func_mgku6wig(backend)
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


def func_3dafmopj(
    data: DataFrame,
    column: str | Sequence[str] | None = None,
    by: Hashable | Iterable[Hashable] | None = None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot: float | None = None,
    ylabelsize: int | None = None,
    yrot: float | None = None,
    ax: Axes | None = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    bins: int | Sequence[float] = 10,
    backend: str | None = None,
    legend: bool = False,
    **kwargs: Any,
) -> Axes | np.ndarray:
    """
    Make a histogram of the DataFrame's columns.

    [Docstring omitted for brevity]
    """
    plot_backend = func_mgku6wig(backend)
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


_boxplot_doc = """
Make a box plot from DataFrame columns.

[Docstring omitted for brevity]
"""

_backend_doc = """backend : str, default None
    Backend to use instead of the backend specified in the option
    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
    specify the ``plotting.backend`` for the whole session, set
    ``pd.options.plotting.backend``.
"""

_bar_or_line_doc = """
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


@Substitution(
    data="""data : DataFrame
        The data to visualize.
""",
    backend='',
)
@Appender(_boxplot_doc)
def func_l2ezyeld(
    data: DataFrame,
    column: str | Sequence[str] | None = None,
    by: Hashable | Iterable[Hashable] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: Literal['axes', 'dict', 'both'] | None = None,
    **kwargs: Any,
) -> Union[Axes, dict, Any]:
    plot_backend = func_mgku6wig('matplotlib')
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


@Substitution(
    data='',
    backend=_backend_doc,
)
@Appender(_boxplot_doc)
def func_72mgchco(
    self: DataFrame,
    column: str | Sequence[str] | None = None,
    by: Hashable | Iterable[Hashable] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: Literal['axes', 'dict', 'both'] | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> Union[Axes, dict, Any]:
    plot_backend = func_mgku6wig(backend)
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


def func_wqrfmjwv(
    grouped: DataFrameGroupBy,
    subplots: bool = True,
    column: str | Sequence[str] | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    sharex: bool = False,
    sharey: bool = True,
    backend: str | None = None,
    **kwargs: Any,
) -> Union[dict, Any]:
    """
    Make box plots from DataFrameGroupBy data.

    [Docstring omitted for brevity]
    """
    plot_backend = func_mgku6wig(backend)
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

    [Docstring omitted for brevity]
    """
    _common_kinds: tuple[str, ...] = ('line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box')
    _series_kinds: tuple[str, ...] = ('pie',)
    _dataframe_kinds: tuple[str, ...] = ('scatter', 'hexbin')
    _kind_aliases: dict[str, str] = {'density': 'kde'}
    _all_kinds: tuple[str, ...] = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Union[Series, DataFrame]):
        self._parent = data

    @staticmethod
    def func_3zwuu1sl(
        backend_name: str,
        data: Union[Series, DataFrame],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Optional[Any], Optional[Any], str, dict[str, Any]]:
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
            keyword_args = ', '.join([f'{name}={value!r}' for (name, _), value in zip(arg_def, args)])
            msg = (
                f'`Series.plot()` should not be called with positional arguments, only keyword arguments. The order of positional arguments will change in the future. Use `Series.plot({keyword_args})` instead of `Series.plot({positional_args})`.'
            )
            raise TypeError(msg)
        pos_args = {name: value for (name, _), value in zip(arg_def, args)}
        if backend_name == 'pandas.plotting._matplotlib':
            kwargs = {**dict(arg_def), **pos_args, **kwargs}
        else:
            kwargs = {**pos_args, **kwargs}
        x = kwargs.pop('x', None)
        y = kwargs.pop('y', None)
        kind = kwargs.pop('kind', 'line')
        return x, y, kind, kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plot_backend = func_mgku6wig(kwargs.pop('backend', None))
        x, y, kind, kwargs = self.func_3zwuu1sl(plot_backend.__name__, self._parent, args, kwargs)
        kind = self._kind_aliases.get(kind, kind)
        if plot_backend.__name__ != 'pandas.plotting._matplotlib':
            return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)
        if kind not in self._all_kinds:
            raise ValueError(
                f'{kind} is not a valid plot kind Valid plot kinds: {self._all_kinds}'
            )
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
                    if is_integer(y) and not func_zo1u1t9h(data.columns):
                        y = data.columns[y]
                    data = data[y].copy(deep=False)
                    data.index.name = y
        elif isinstance(data, ABCDataFrame):
            data_cols = data.columns
            if x is not None:
                if is_integer(x) and not func_zo1u1t9h(data.columns):
                    x = data_cols[x]
                elif not isinstance(data[x], ABCSeries):
                    raise ValueError('x must be a label or position')
                data = data.set_index(x)
            if y is not None:
                int_ylist = is_list_like(y) and all(is_integer(c) for c in y)
                int_y_arg = is_integer(y) or int_ylist
                if int_y_arg and not func_zo1u1t9h(data.columns):
                    y = data_cols[y]
                label_kw = kwargs.get('label', False)
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
                    if label_kw and not match:
                        raise ValueError('label should be list-like and same length as y')
                    label_name = label_kw or data.columns
                    data.columns = label_name
        return plot_backend.plot(data, kind=kind, **kwargs)
    __call__.__doc__ = __doc__

    @Appender(
        """
        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Examples
        --------

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 3, 2])
            >>> s.plot.line()  # doctest: +SKIP

        .. plot::
            :context: close-figs

            The following example shows the populations for some animals
            over the years.

            >>> df = pd.DataFrame({
            ...     'pig': [20, 18, 489, 675, 1776],
            ...     'horse': [4, 25, 281, 600, 1900]
            ... }, index=[1990, 1997, 2003, 2009, 2014])
            >>> lines = df.plot.line()

        .. plot::
           :context: close-figs

           An example with subplots, so an array of axes is returned.

           >>> axes = df.plot.line(subplots=True)
           >>> type(axes)
           <class 'numpy.ndarray'>

        .. plot::
           :context: close-figs

           Let's repeat the same example, but specifying colors for
           each column (in this case, for each animal).

           >>> axes = df.plot.line(
           ...     subplots=True, color={"pig": "pink", "horse": "#742802"}
           ... )
           >>> axes[1].legend(loc=2)  # doctest: +SKIP

        .. plot::
            :context: close-figs

            The following example shows the relationship between both
            populations.

            >>> lines = df.plot.line(x='pig', y='horse')
        """
    )
    @Substitution(kind='line')
    @Appender(_bar_or_line_doc)
    def func_q1h6jvai(
        self,
        x: str | int | None = None,
        y: str | int | list[str | int] | None = None,
        color: str | list[str] | dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Axes | np.ndarray:
        """
        Plot Series or DataFrame as lines.

        This function is useful to plot lines using DataFrame's values
        as coordinates.
        """
        if color is not None:
            kwargs['color'] = color
        return self(kind='line', x=x, y=y, **kwargs)

    @Appender(
        """
        See Also
        --------
        DataFrame.plot.barh : Horizontal bar plot.
        DataFrame.plot : Make plots of a DataFrame using matplotlib.
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.bar(x='lab', y='val', rot=0)

        Plot a whole dataframe to a bar plot. Each column is assigned a
        distinct color, and each row is nested in a group along the
        horizontal axis.

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.bar(rot=0)

        Plot stacked bar charts for the DataFrame

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(stacked=True)

        Instead of nesting, the figure can be split by column with
        ``subplots=True``. In this case, a :class:`numpy.ndarray` of
        :class:`matplotlib.axes.Axes` are returned.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(rot=0, subplots=True)
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        If you don't like the default colours, you can specify how you'd
        like each column to be colored.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(
            ...     rot=0, subplots=True, color={"speed": "red", "lifespan": "green"}
            ... )
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        Plot a single column.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(y='speed', rot=0)

        Plot only selected categories for the DataFrame.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(x='lifespan', rot=0)
        """
    )
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def func_3n6mvd2k(
        self,
        x: str | int | None = None,
        y: str | int | list[str | int] | None = None,
        color: str | list[str] | dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Axes | np.ndarray:
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
        """
        See Also
        --------
        DataFrame.plot.bar : Vertical bar plot.
        DataFrame.plot : Make plots of DataFrame using matplotlib.
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.bar(x='lab', y='val', rot=0)

        Plot a whole dataframe to a bar plot. Each column is assigned a
        distinct color, and each row is nested in a group along the
        horizontal axis.

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.bar(rot=0)

        Plot stacked bar charts for the DataFrame

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(stacked=True)

        Instead of nesting, the figure can be split by column with
        ``subplots=True``. In this case, a :class:`numpy.ndarray` of
        :class:`matplotlib.axes.Axes` are returned.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(rot=0, subplots=True)
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        If you don't like the default colours, you can specify how you'd
        like each column to be colored.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(
            ...     rot=0, subplots=True, color={"speed": "red", "lifespan": "green"}
            ... )
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        Plot a single column.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(y='speed', rot=0)

        Plot only selected categories for the DataFrame.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(x='lifespan', rot=0)
        """
    )
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def func_2zjbrkvc(
        self,
        x: str | int | None = None,
        y: str | int | list[str | int] | None = None,
        color: str | list[str] | dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Axes | np.ndarray:
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

    def func_rglbi4ew(
        self,
        by: Hashable | Iterable[Hashable] | None = None,
        **kwargs: Any,
    ) -> Axes | np.ndarray | dict[str, Any]:
        """
        Make a box plot of the DataFrame columns.

        [Docstring omitted for brevity]
        """
        return self(kind='box', by=by, **kwargs)

    def func_byaivru6(
        self,
        by: Hashable | Iterable[Hashable] | None = None,
        bins: int = 10,
        **kwargs: Any,
    ) -> Axes:
        """
        Draw one histogram of the DataFrame's columns.

        [Docstring omitted for brevity]
        """
        return self(kind='hist', by=by, bins=bins, **kwargs)

    def func_vs7v1ljw(
        self,
        bw_method: str | float | Callable[[Any], Any] | None = None,
        ind: Optional[np.ndarray | int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Axes | np.ndarray:
        """
        Generate Kernel Density Estimate plot using Gaussian kernels.

        [Docstring omitted for brevity]
        """
        return self(kind='kde', bw_method=bw_method, ind=ind, weights=weights, **kwargs)

    density = kde

    def func_vjd5w44a(
        self,
        x: str | int | None = None,
        y: str | int | None = None,
        stacked: bool = True,
        **kwargs: Any,
    ) -> Axes | np.ndarray:
        """
        Draw a stacked area plot.

        [Docstring omitted for brevity]
        """
        return self(kind='area', x=x, y=y, stacked=stacked, **kwargs)

    def func_vx77995m(
        self,
        y: str | int | None = None,
        **kwargs: Any,
    ) -> Axes | np.ndarray:
        """
        Generate a pie plot.

        [Docstring omitted for brevity]
        """
        if y is not None:
            kwargs['y'] = y
        if isinstance(self._parent, ABCDataFrame) and kwargs.get('y') is None and not kwargs.get('subplots', False):
            raise ValueError("pie requires either y column or 'subplots=True'")
        return self(kind='pie', **kwargs)

    def func_gjucj9oo(
        self,
        x: str | int,
        y: str | int,
        s: str | float | Sequence[float] | None = None,
        c: str | int | Sequence[str | int] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Create a scatter plot with varying marker point size and color.

        [Docstring omitted for brevity]
        """
        return self(kind='scatter', x=x, y=y, s=s, c=c, **kwargs)

    def func_3haebn14(
        self,
        x: str | int,
        y: str | int,
        C: str | int | None = None,
        reduce_C_function: Callable[[Any], Any] | None = None,
        gridsize: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Generate a hexagonal binning plot.

        [Docstring omitted for brevity]
        """
        if reduce_C_function is not None:
            kwargs['reduce_C_function'] = reduce_C_function
        if gridsize is not None:
            kwargs['gridsize'] = gridsize
        return self(kind='hexbin', x=x, y=y, C=C, **kwargs)


_backends: dict[str, types.ModuleType] = {}


def func_hvpiplqe(backend: str) -> types.ModuleType:
    """
    Load a pandas plotting backend.

    [Docstring omitted for brevity]
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


def func_mgku6wig(backend: Optional[str] = None) -> types.ModuleType:
    """
    Return the plotting backend to use (e.g. `pandas.plotting._matplotlib`).

    [Docstring omitted for brevity]
    """
    backend_str: str | None = backend or get_option('plotting.backend')
    if backend_str in _backends:
        return _backends[backend_str]
    module = func_hvpiplqe(backend_str)
    _backends[backend_str] = module
    return module
