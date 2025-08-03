from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, final, Optional, Union, List, Tuple, Dict, Set, Callable, TypeVar, Generic
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_any_real_numeric_dtype, is_bool, is_float, is_float_dtype, is_hashable, is_integer, is_integer_dtype, is_iterator, is_list_like, is_number, is_numeric_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCDatetimeIndex, ABCIndex, ABCMultiIndex, ABCPeriodIndex, ABCSeries
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import decorate_axes, format_dateaxis, maybe_convert_index, maybe_resample, use_dynamic_x
from pandas.plotting._matplotlib.tools import create_subplots, flatten_axes, format_date_labels, get_all_lines, get_xlim, handle_shared_axes

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from pandas._typing import IndexLabel, NDFrameT, PlottingOrientation, npt
    from pandas import DataFrame, Index, Series

T = TypeVar('T')

def holds_integer(column: Any) -> bool:
    return column.inferred_type in {'integer', 'mixed-integer'}

def _color_in_style(style: str) -> bool:
    """
    Check if there is a color letter in the style string.
    """
    return not set(mpl.colors.BASE_COLORS).isdisjoint(style)

class MPLPlot(ABC):
    """
    Base class for assembling a pandas plot using matplotlib

    Parameters
    ----------
    data :

    """

    @property
    @abstractmethod
    def _kind(self) -> str:
        """Specify kind str. Must be overridden in child class"""
        raise NotImplementedError
    _layout_type: str = 'vertical'
    _default_rot: int = 0

    @property
    def orientation(self) -> Optional[str]:
        return None

    def __init__(
        self,
        data: Any,
        kind: Optional[str] = None,
        by: Optional[Any] = None,
        subplots: Union[bool, Iterable[Any]] = False,
        sharex: Optional[bool] = None,
        sharey: bool = False,
        use_index: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        grid: Optional[bool] = None,
        legend: Union[bool, str] = True,
        rot: Optional[int] = None,
        ax: Optional[Any] = None,
        fig: Optional[Any] = None,
        title: Optional[Union[str, List[str]]] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xticks: Optional[Sequence[float]] = None,
        yticks: Optional[Sequence[float]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        fontsize: Optional[int] = None,
        secondary_y: Union[bool, Sequence[Any]] = False,
        colormap: Optional[Any] = None,
        table: Union[bool, Any] = False,
        layout: Optional[Tuple[int, int]] = None,
        include_bool: bool = False,
        column: Optional[Union[str, List[str]]] = None,
        *,
        logx: Union[bool, str, None] = False,
        logy: Union[bool, str, None] = False,
        loglog: Union[bool, str, None] = False,
        mark_right: bool = True,
        stacked: bool = False,
        label: Optional[str] = None,
        style: Optional[Union[str, List[str], Dict[Any, str]]] = None,
        **kwds: Any
    ) -> None:
        if by in ([], ()):
            raise ValueError('No group keys passed!')
        self.by = com.maybe_make_list(by)
        if isinstance(data, ABCDataFrame):
            if column:
                self.columns = com.maybe_make_list(column)
            elif self.by is None:
                self.columns = [col for col in data.columns if is_numeric_dtype(data[col])]
            else:
                self.columns = [col for col in data.columns if col not in self.by and is_numeric_dtype(data[col])]
        if self.by is not None and self._kind == 'hist':
            self._grouped = data.groupby(unpack_single_str_list(self.by))
        self.kind = kind
        self.subplots = type(self)._validate_subplots_kwarg(subplots, data, kind=self._kind)
        self.sharex = type(self)._validate_sharex(sharex, ax, by)
        self.sharey = sharey
        self.figsize = figsize
        self.layout = layout
        self.xticks = xticks
        self.yticks = yticks
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.use_index = use_index
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fontsize = fontsize
        if rot is not None:
            self.rot = rot
            self._rot_set = True
        else:
            self._rot_set = False
            self.rot = self._default_rot
        if grid is None:
            grid = False if secondary_y else mpl.rcParams['axes.grid']
        self.grid = grid
        self.legend = legend
        self.legend_handles: List[Any] = []
        self.legend_labels: List[str] = []
        self.logx = type(self)._validate_log_kwd('logx', logx)
        self.logy = type(self)._validate_log_kwd('logy', logy)
        self.loglog = type(self)._validate_log_kwd('loglog', loglog)
        self.label = label
        self.style = style
        self.mark_right = mark_right
        self.stacked = stacked
        self.ax = ax
        xerr = kwds.pop('xerr', None)
        yerr = kwds.pop('yerr', None)
        nseries = self._get_nseries(data)
        xerr, data = type(self)._parse_errorbars('xerr', xerr, data, nseries)
        yerr, data = type(self)._parse_errorbars('yerr', yerr, data, nseries)
        self.errors: Dict[str, Any] = {'xerr': xerr, 'yerr': yerr}
        self.data = data
        if not isinstance(secondary_y, (bool, tuple, list, np.ndarray, ABCIndex)):
            secondary_y = [secondary_y]
        self.secondary_y = secondary_y
        if 'cmap' in kwds and colormap:
            raise TypeError('Only specify one of `cmap` and `colormap`.')
        if 'cmap' in kwds:
            self.colormap = kwds.pop('cmap')
        else:
            self.colormap = colormap
        self.table = table
        self.include_bool = include_bool
        self.kwds = kwds
        color = kwds.pop('color', lib.no_default)
        self.color = self._validate_color_args(color, self.colormap)
        assert 'color' not in self.kwds
        self.data = self._ensure_frame(self.data)

    @final
    @staticmethod
    def _validate_sharex(sharex: Optional[bool], ax: Optional[Any], by: Optional[Any]) -> bool:
        if sharex is None:
            if ax is None and by is None:
                sharex = True
            else:
                sharex = False
        elif not is_bool(sharex):
            raise TypeError('sharex must be a bool or None')
        return bool(sharex)

    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: Any) -> Union[bool, str, None]:
        if value is None or isinstance(value, bool) or (isinstance(value, str) and value == 'sym'):
            return value
        raise ValueError(f"keyword '{kwd}' should be bool, None, or 'sym', not '{value}'")

    @final
    @staticmethod
    def _validate_subplots_kwarg(subplots: Union[bool, Iterable[Any]], data: Any, kind: str) -> Union[bool, List[Tuple[int, ...]]]:
        """
        Validate the subplots parameter

        - check type and content
        - check for duplicate columns
        - check for invalid column names
        - convert column names into indices
        - add missing columns in a group of their own
        See comments in code below for more details.

        Parameters
        ----------
        subplots : subplots parameters as passed to PlotAccessor

        Returns
        -------
        validated subplots : a bool or a list of tuples of column indices. Columns
        in the same tuple will be grouped together in the resulting plot.
        """
        if isinstance(subplots, bool):
            return subplots
        elif not isinstance(subplots, Iterable):
            raise ValueError('subplots should be a bool or an iterable')
        supported_kinds = ('line', 'bar', 'barh', 'hist', 'kde', 'density', 'area', 'pie')
        if kind not in supported_kinds:
            raise ValueError(f'When subplots is an iterable, kind must be one of {", ".join(supported_kinds)}. Got {kind}.')
        if isinstance(data, ABCSeries):
            raise NotImplementedError('An iterable subplots for a Series is not supported.')
        columns = data.columns
        if isinstance(columns, ABCMultiIndex):
            raise NotImplementedError('An iterable subplots for a DataFrame with a MultiIndex column is not supported.')
        if columns.nunique() != len(columns):
            raise NotImplementedError('An iterable subplots for a DataFrame with non-unique column labels is not supported.')
        out: List[Tuple[int, ...]] = []
        seen_columns: Set[Any] = set()
        for group in subplots:
            if not is_list_like(group):
                raise ValueError('When subplots is an iterable, each entry should be a list/tuple of column names.')
            idx_locs = columns.get_indexer_for(group)
            if (idx_locs == -1).any():
                bad_labels = np.extract(idx_locs == -1, group)
                raise ValueError(f'Column label(s) {list(bad_labels)} not found in the DataFrame.')
            unique_columns = set(group)
            duplicates = seen_columns.intersection(unique_columns)
            if duplicates:
                raise ValueError(f'Each column should be in only one subplot. Columns {duplicates} were found in multiple subplots.')
            seen_columns = seen_columns.union(unique_columns)
            out.append(tuple(idx_locs))
        unseen_columns = columns.difference(seen_columns)
        for column in unseen_columns:
            idx_loc = columns.get_loc(column)
            out.append((idx_loc,))
        return out

    def _validate_color_args(self, color: Any, colormap: Optional[Any]) -> Optional[Any]:
        if color is lib.no_default:
            if 'colors' in self.kwds and colormap is not None:
                warnings.warn("'color' and 'colormap' cannot be used simultaneously. Using 'color'", stacklevel=find_stack_level())
            return None
        if self.nseries == 1 and color is not None and (not is_list_like(color)):
            color = [color]
        if isinstance(color, tuple) and self.nseries == 1 and (len(color) in (3, 4)):
            color = [color]
        if colormap is not None:
            warnings.warn("'color' and 'colormap' cannot be used simultaneously. Using 'color'", stacklevel=find_stack_level())
        if self.style is not None:
            if isinstance(self.style, dict):
                styles = [self.style[col] for col in self.columns if col in self.style]
            elif is_list_like(self.style):
                styles = self.style
            else:
                styles = [self.style]
            for s in styles:
                if _color_in_style(s):
                    raise ValueError("Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol")
        return color

    @final
    @staticmethod
    def _iter_data(data: Any) -> Iterator[Tuple[Any, np.ndarray]]:
        for col, values in data.items():
            yield (col, np.asarray(values.values))

    def _get_nseries(self, data: Any) -> int:
        if data.ndim == 1:
            return 1
        elif self.by is not None and self._kind == 'hist':
            return len(self._grouped)
        elif self.by is not None and self._kind == 'box':
            return len(self.columns)
        else:
            return data.shape[1]

    @final
    @property
    def nseries(self) -> int:
        return self._get_nseries(self.data)

    @final
    def generate(self) -> None:
        self._compute_plot_data()
        fig = self.fig
        self._make_plot(fig)
        self._add_table()
        self._make_legend()
        self._adorn_subplots(fig)
        for ax in self.axes:
            self._post_plot_logic_common(ax)
            self._post_plot_logic(ax, self.data)

    @final
    @staticmethod
    def _has_plotted_object(ax: Any) -> bool:
        """check whether ax has data"""
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    @final
    def _maybe_right_yaxis(self, ax: Any, axes_num: int) -> Any:
        if not self.on_right(axes_num):
            return self._get_ax_layer(ax)
        if hasattr(ax, 'right_ax'):
            return ax.right_ax
        elif hasattr(ax, 'left_ax'):
            return ax
        else:
            orig_ax, new_ax = (ax, ax.twinx())
            new_ax._get_lines = orig_ax._get_lines
            new_ax._get_patches_for_fill = orig_ax._get_patches_for_fill
            orig_ax.right_ax, new_ax.left_ax = (new_ax, orig_ax)
            if not self._has_plotted_object(orig_ax):
                orig_ax.get_yaxis().set_visible(False)
            if self.logy is True or self.loglog is True:
                new_ax.set_yscale('log')
            elif self.logy == 'sym' or self.loglog == 'sym':
                new_ax.set_yscale('symlog')
            return new_ax

    @final
    @cache_readonly
    def fig(self) -> Any:
        return self._axes_and_fig[1]

    @final
    @cache_readonly
    def axes(self) -> Sequence[Any]:
        return self._axes_and_fig[0]

    @final
    @cache_readonly
    def _axes_and_fig(self) -> Tuple[Sequence[Any], Any]:
        import matplotlib.pyplot as plt
        if self.subplots:
            naxes = self.nseries if isinstance(self.subplots, bool) else len(self.subplots)
            fig, axes = create_subplots(naxes=naxes, sharex=self.sharex, sharey=self.sharey, figsize=self.figsize, ax=self.ax, layout=self.layout, layout_type=self._layout_type)
        elif self.ax is None:
            fig = plt.figure(figsize=self.figsize)
            axes = fig.add_subplot(111)
        else:
            fig = self.ax.get_figure()
            if self.figsize is not None:
                fig.set_size_inches(self.figsize)
            axes = self.ax
        axes = np.fromiter(flatten_axes(axes), dtype=object)
        if self.logx is True or self.loglog is True:
            [a.set_xscale('log') for a in axes]
        elif self.logx == 'sym' or self.loglog == 'sym':
            [a.set_xscale('symlog') for a in axes]
        if self.logy is True or self.loglog is True:
            [a.set_yscale('log') for a in axes]
        elif self.logy == 'sym' or self.loglog == 'sym':
            [a.set_yscale('symlog') for a in axes]
        axes_seq = cast(Sequence['Axes'], axes)
        return (axes_seq, fig)

    @property
    def result(self) -> Any:
        """
        Return result axes
        """
        if self.subplots:
            if self.layout is not None and (not is_list_like(self.ax)):
                return self.axes.reshape(*self.layout)
            else:
                return self.axes
        else:
            sec_true = isinstance(self.secondary_y, bool) and self.secondary_y
            all_sec = is_list_like(self.secondary_y) and len(self.secondary_y