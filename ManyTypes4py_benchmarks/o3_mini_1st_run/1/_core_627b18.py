from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union, Tuple
from pandas._config import get_option
from pandas.util._decorators import Appender, Substitution
from pandas.core.dtypes.common import is_integer, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.base import PandasObject

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    import types
    from matplotlib.axes import Axes
    import numpy as np
    from pandas._typing import IndexLabel
    from pandas import DataFrame, Index, Series
    from pandas.core.groupby.generic import DataFrameGroupBy

def holds_integer(column: Any) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}

def hist_series(
    self: Union[ABCSeries, Any],
    by: Any = None,
    ax: Optional[Axes] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    figsize: Optional[Tuple[Any, ...]] = None,
    bins: Union[int, Sequence[Any]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any
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
        **kwargs
    )

def hist_frame(
    data: Any,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Any = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    ax: Optional[Any] = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: Optional[Tuple[Any, ...]] = None,
    layout: Optional[Tuple[Any, ...]] = None,
    bins: Union[int, Sequence[Any]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any
) -> Union[Axes, "np.ndarray"]:
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
        **kwargs
    )

@Substitution(data='data : DataFrame\n    The data to visualize.\n', backend='')
@Appender(_boxplot_doc)
def boxplot(
    data: Any,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[Any]]] = None,
    ax: Optional[Any] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[Any, ...]] = None,
    layout: Optional[Tuple[Any, ...]] = None,
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
        **kwargs
    )

@Substitution(data='', backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: Any,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[Any]]] = None,
    ax: Optional[Any] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[Tuple[Any, ...]] = None,
    layout: Optional[Tuple[Any, ...]] = None,
    return_type: Optional[Union[str, None]] = None,
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
        **kwargs
    )

def boxplot_frame_groupby(
    grouped: DataFrameGroupBy,
    subplots: bool = True,
    column: Optional[Union[str, Sequence[str]]] = None,
    fontsize: Optional[Union[float, str]] = None,
    rot: float = 0,
    grid: bool = True,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[Any, ...]] = None,
    layout: Optional[Tuple[Any, ...]] = None,
    sharex: bool = False,
    sharey: bool = True,
    backend: Optional[str] = None,
    **kwargs: Any
) -> Any:
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
        **kwargs
    )

class PlotAccessor(PandasObject):
    def __init__(self, data: Union[ABCSeries, ABCDataFrame]) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(
        backend_name: str,
        data: Union[ABCSeries, ABCDataFrame],
        args: Sequence[Any],
        kwargs: Dict[str, Any]
    ) -> Tuple[Optional[Any], Optional[Any], str, Dict[str, Any]]:
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
            raise TypeError(f'Called plot accessor for type {type(data).__name__}, expected Series or DataFrame')
        if args and isinstance(data, ABCSeries):
            positional_args = str(args)[1:-1]
            keyword_args = ', '.join([f'{name}={value!r}' for (name, _), value in zip(arg_def, args)])
            msg = (
                f'`Series.plot()` should not be called with positional arguments, only keyword arguments. '
                f'The order of positional arguments will change in the future. Use '
                f'`Series.plot({keyword_args})` instead of `Series.plot({positional_args})`.'
            )
            raise TypeError(msg)
        pos_args = {name: value for (name, _), value in zip(arg_def, args)}
        if backend_name == 'pandas.plotting._matplotlib':
            kwargs = dict(arg_def, **pos_args, **kwargs)  # type: ignore[arg-type]
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
        _all_kinds = ('line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box', 'pie', 'scatter', 'hexbin')
        _dataframe_kinds = ('scatter', 'hexbin')
        _series_kinds = ('pie',)
        if kind not in _all_kinds:
            raise ValueError(f'{kind} is not a valid plot kind Valid plot kinds: {_all_kinds}')
        data = self._parent
        if isinstance(data, ABCSeries):
            kwargs['reuse_plot'] = True
        if kind in _dataframe_kinds:
            if isinstance(data, ABCDataFrame):
                return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
            else:
                raise ValueError(f'plot kind {kind} can only be used for data frames')
        elif kind in _series_kinds:
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
                int_ylist = is_list_like(y) and all(is_integer(c) for c in y)
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
        if color is not None:
            kwargs['color'] = color
        return self(kind='line', x=x, y=y, **kwargs)

    @Appender('\n        See Also\n        --------\n        DataFrame.plot.barh : Horizontal bar plot.\n        DataFrame.plot : Make plots of a DataFrame.\n        matplotlib.pyplot.bar : Make a bar plot with matplotlib.\n\n        Examples\n        --------\n        Basic plot.\n\n        .. plot::\n            :context: close-figs\n\n            >>> df = pd.DataFrame({\'lab\': [\'A\', \'B\', \'C\'], \'val\': [10, 30, 20]})\n            >>> ax = df.plot.bar(x=\'lab\', y=\'val\', rot=0)\n    ')
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def bar(self, x: Optional[Any] = None, y: Optional[Any] = None, color: Optional[Any] = None, **kwargs: Any) -> Any:
        if color is not None:
            kwargs['color'] = color
        return self(kind='bar', x=x, y=y, **kwargs)

    @Appender('\n        See Also\n        --------\n        DataFrame.plot.bar : Vertical bar plot.\n        DataFrame.plot : Make plots of DataFrame using matplotlib.\n        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.\n\n        Examples\n        --------\n        Basic example\n\n        .. plot::\n            :context: close-figs\n\n            >>> df = pd.DataFrame({\'lab\': [\'A\', \'B\', \'C\'], \'val\': [10, 30, 20]})\n            >>> ax = df.plot.barh(x=\'lab\', y=\'val\')\n    ')
    @Substitution(kind='bar')
    @Appender(_bar_or_line_doc)
    def barh(self, x: Optional[Any] = None, y: Optional[Any] = None, color: Optional[Any] = None, **kwargs: Any) -> Any:
        if color is not None:
            kwargs['color'] = color
        return self(kind='barh', x=x, y=y, **kwargs)

    def box(self, by: Optional[Any] = None, **kwargs: Any) -> Any:
        return self(kind='box', by=by, **kwargs)

    def hist(self, by: Optional[Any] = None, bins: Union[int, Sequence[Any]] = 10, **kwargs: Any) -> Axes:
        return self(kind='hist', by=by, bins=bins, **kwargs)

    def kde(self, bw_method: Optional[Union[str, float, Callable]] = None, ind: Optional[Union["np.ndarray", int]] = None, weights: Optional["np.ndarray"] = None, **kwargs: Any) -> Any:
        return self(kind='kde', bw_method=bw_method, ind=ind, weights=weights, **kwargs)
    density = kde

    def area(self, x: Optional[Any] = None, y: Optional[Any] = None, stacked: bool = True, **kwargs: Any) -> Any:
        return self(kind='area', x=x, y=y, stacked=stacked, **kwargs)

    def pie(self, y: Optional[Union[int, str]] = None, **kwargs: Any) -> Any:
        if y is not None:
            kwargs['y'] = y
        if isinstance(self._parent, ABCDataFrame) and kwargs.get('y', None) is None and (not kwargs.get('subplots', False)):
            raise ValueError("pie requires either y column or 'subplots=True'")
        return self(kind='pie', **kwargs)

    def scatter(self, x: Union[int, str], y: Union[int, str], s: Optional[Any] = None, c: Optional[Any] = None, **kwargs: Any) -> Any:
        return self(kind='scatter', x=x, y=y, s=s, c=c, **kwargs)

    def hexbin(self, x: Union[int, str], y: Union[int, str], C: Optional[Union[int, str]] = None, reduce_C_function: Optional[Callable] = None, gridsize: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: Any) -> Axes:
        if reduce_C_function is not None:
            kwargs['reduce_C_function'] = reduce_C_function
        if gridsize is not None:
            kwargs['gridsize'] = gridsize
        return self(kind='hexbin', x=x, y=y, C=C, **kwargs)

_backends: Dict[str, Any] = {}

def _load_backend(backend: str) -> Any:
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

def _get_plot_backend(backend: Optional[str] = None) -> Any:
    backend_str = backend or get_option('plotting.backend')
    if backend_str in _backends:
        return _backends[backend_str]
    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module