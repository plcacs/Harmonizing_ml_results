from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Generator, Mapping
from pandas.plotting._core import _get_plot_backend
if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.table import Table
    import numpy as np
    from pandas import DataFrame, Series

def table(ax, data, **kwargs: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.table(ax=ax, data=data, rowLabels=None, colLabels=None, **kwargs)

def register():
    plot_backend = _get_plot_backend('matplotlib')
    plot_backend.register()

def deregister():
    plot_backend = _get_plot_backend('matplotlib')
    plot_backend.deregister()

def scatter_matrix(frame, alpha=0.5, figsize=None, ax=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwargs: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.scatter_matrix(frame=frame, alpha=alpha, figsize=figsize, ax=ax, grid=grid, diagonal=diagonal, marker=marker, density_kwds=density_kwds, hist_kwds=hist_kwds, range_padding=range_padding, **kwargs)

def radviz(frame, class_column, ax=None, color=None, colormap=None, **kwds: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.radviz(frame=frame, class_column=class_column, ax=ax, color=color, colormap=colormap, **kwds)

def andrews_curves(frame, class_column, ax=None, samples=200, color=None, colormap=None, **kwargs: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.andrews_curves(frame=frame, class_column=class_column, ax=ax, samples=samples, color=color, colormap=colormap, **kwargs)

def bootstrap_plot(series, fig=None, size=50, samples=500, **kwds: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.bootstrap_plot(series=series, fig=fig, size=size, samples=samples, **kwds)

def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None, use_columns=False, xticks=None, colormap=None, axvlines=True, axvlines_kwds=None, sort_labels=False, **kwargs: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.parallel_coordinates(frame=frame, class_column=class_column, cols=cols, ax=ax, color=color, use_columns=use_columns, xticks=xticks, colormap=colormap, axvlines=axvlines, axvlines_kwds=axvlines_kwds, sort_labels=sort_labels, **kwargs)

def lag_plot(series, lag=1, ax=None, **kwds: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.lag_plot(series=series, lag=lag, ax=ax, **kwds)

def autocorrelation_plot(series, ax=None, **kwargs: Any):
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.autocorrelation_plot(series=series, ax=ax, **kwargs)

class _Options(dict):
    _ALIASES = {'x_compat': 'xaxis.compat'}
    _DEFAULT_KEYS = ['xaxis.compat']

    def __init__(self):
        super().__setitem__('xaxis.compat', False)

    def __getitem__(self, key):
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(f'{key} is not a valid pandas plotting option')
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        key = self._get_canonical_key(key)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        key = self._get_canonical_key(key)
        if key in self._DEFAULT_KEYS:
            raise ValueError(f'Cannot remove default parameter {key}')
        super().__delitem__(key)

    def __contains__(self, key):
        key = self._get_canonical_key(key)
        return super().__contains__(key)

    def reset(self):
        self.__init__()

    def _get_canonical_key(self, key):
        return self._ALIASES.get(key, key)

    @contextmanager
    def use(self, key, value):
        old_value = self[key]
        try:
            self[key] = value
            yield self
        finally:
            self[key] = old_value
plot_params = _Options()