from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Literal, Any, Optional, Union, Dict, List, Tuple, Sequence, Callable, Hashable
from pandas._config import get_option
from pandas.util._decorators import Appender, Substitution
from pandas.core.dtypes.common import is_integer, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.base import PandasObject
if TYPE_CHECKING:
    from collections.abc import Callable as CallableType, Hashable as HashableType, Sequence as SequenceType
    import types
    from matplotlib.axes import Axes
    import numpy as np
    from pandas._typing import IndexLabel
    from pandas import DataFrame, Index, Series
    from pandas.core.groupby.generic import DataFrameGroupBy


def func_zo1u1t9h(column: Any) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}


def func_215yl7tf(
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


def func_3dafmopj(
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
        data, column=column, by=by, grid=grid, xlabelsize=xlabelsize,
        xrot=xrot, ylabelsize=ylabelsize, yrot=yrot, ax=ax, sharex=sharex,
        sharey=sharey, figsize=figsize, layout=layout, legend=legend,
        bins=bins, **kwargs
    )


@Substitution(data="""data : DataFrame
    The data to visualize.
""",
    backend='')
@Appender(_boxplot_doc)
def func_l2ezyeld(
    data: Any,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, Sequence[Any]]] = None,
    ax: Optional[Any] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = 'axes',
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.boxplot(
        data, column=column, by=by, ax=ax, fontsize=fontsize, rot=rot,
        grid=grid, figsize=figsize, layout=layout, return_type=return_type,
        **kwargs
    )


@Substitution(data='', backend=_backend_doc)
@Appender(_boxplot_doc)
def func_72mgchco(
    self: Any,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, Sequence[Any]]] = None,
    ax: Optional[Any] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = 'axes',
    backend: Optional[str] = None,
    **kwargs: Any
) -> Any:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame(
        self, column=column, by=by, ax=ax, fontsize=fontsize, rot=rot,
        grid=grid, figsize=figsize, layout=layout, return_type=return_type,
        **kwargs
    )


def func_wqrfmjwv(
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
) -> Union[Dict[Any, Any], Any]:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame_groupby(
        grouped, subplots=subplots, column=column, fontsize=fontsize,
        rot=rot, grid=grid, ax=ax, figsize=figsize, layout=layout,
        sharex=sharex, sharey=sharey, **kwargs
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
    def func_3zwuu1sl(backend_name: str, data: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any, str, Dict[str, Any]]:
        if isinstance(data, ABCSeries):
            arg_def = [('kind', 'line'), ('ax', None), ('figsize', None),
                      ('use_index', True), ('title', None), ('grid', None),
                      ('legend', False), ('style', None), ('logx', False),
                      ('logy', False), ('loglog', False), ('xticks', None),
                      ('yticks', None), ('xlim', None), ('ylim', None),
                      ('rot', None), ('fontsize', None), ('colormap', None),
                      ('table', False), ('yerr', None), ('xerr', None),
                      ('label', None), ('secondary_y', False), ('xlabel', None),
                      ('ylabel', None)]
        elif isinstance(data, ABCDataFrame):
            arg_def = [('x', None), ('y', None), ('kind', 'line'), ('ax', None),
                      ('subplots', False), ('sharex', None), ('sharey', False),
                      ('layout', None), ('figsize', None), ('use_index', True),
                      ('title', None), ('grid', None), ('legend', True),
                      ('style', None), ('logx', False), ('logy', False),
                      ('loglog', False), ('xticks', None), ('yticks', None),
                      ('xlim', None), ('ylim', None), ('rot', None),
                      ('fontsize', None), ('colormap', None), ('table', False),
                      ('yerr', None), ('xerr', None), ('secondary_y', False),
                      ('xlabel', None), ('ylabel', None)]
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
            kwargs = dict(arg_def, **pos_args, **kwargs)
        else:
            kwargs = dict(pos_args, **kwargs)
        x = kwargs.pop('x', None)
        y = kwargs.pop('y', None)
        kind = kwargs.pop('kind', 'line')
        return x, y, kind, kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plot_backend = _get_plot_backend(kwargs.pop('backend', None))
        x, y, kind, kwargs = self._get_call_args(plot_backend.__name__, self._parent, args, kwargs)
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
                raise ValueError(
                    f'plot kind {kind} can only be used for data frames')
        elif kind in self._series_kinds:
            if isinstance(data, ABCDataFrame):
                if y is None and kwargs.get('subplots') is False:
                    raise ValueError(
                        f"{kind} requires either y column or 'subplots=True'")
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
                    if label_kw and not match:
                        raise ValueError(
                            'label should be list-like and same length as y')
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

        .. plot::
            :context: close-figs

            The following example shows the relationship between both
            populations.

            >>> lines = df.plot.line(x='pig', y='horse')
        """
    )
    @Substitution(kind='line')
    @Appender(_bar_or_line_doc)
    def func_q1h6jvai(self, x: Optional[Any] = None, y: Optional[Any] = None, color: Optional[Any] = None, **kwargs: Any) -> Any:
        if color is not None:
            kwargs['color'] = color
        return self(kind='line', x=x, y=y, **kwargs)

    @Appender(
        """
        See Also
        --------
        DataFrame.plot.barh : Horizontal bar plot.
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.bar : Make a bar plot with matplotlib.

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
    @Substitution