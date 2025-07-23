from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, final, Optional, Union, Tuple, List, Dict
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_bool,
    is_float,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_iterator,
    is_list_like,
    is_number,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
    decorate_axes,
    format_dateaxis,
    maybe_convert_index,
    maybe_resample,
    use_dynamic_x,
)
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    format_date_labels,
    get_all_lines,
    get_xlim,
    handle_shared_axes,
)
if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from pandas._typing import IndexLabel, NDFrameT, PlottingOrientation, npt
    from pandas import DataFrame, Index, Series


def holds_integer(column: Series) -> bool:
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

    _layout_type: Literal['vertical', 'single', 'horizontal'] = 'vertical'
    _default_rot: int = 0

    @property
    def orientation(self) -> Optional[str]:
        return None

    def __init__(
        self,
        data: NDFrameT,
        kind: Optional[str] = None,
        by: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        subplots: bool = False,
        sharex: Optional[bool] = None,
        sharey: Union[bool, Literal['reverse']] = False,
        use_index: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        grid: Optional[bool] = None,
        legend: Union[bool, Literal['reverse']] = True,
        rot: Optional[Union[int, float]] = None,
        ax: Optional[Axes] = None,
        fig: Optional[Figure] = None,
        title: Optional[Union[str, List[str]]] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xticks: Optional[List[float]] = None,
        yticks: Optional[List[float]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        fontsize: Optional[float] = None,
        secondary_y: Union[bool, List[Hashable], Tuple[Hashable, ...], np.ndarray, Index] = False,
        colormap: Optional[str] = None,
        table: Union[bool, Any] = False,
        layout: Optional[Tuple[int, int]] = None,
        include_bool: bool = False,
        column: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        *,
        logx: Union[bool, Literal['sym']] = False,
        logy: Union[bool, Literal['sym']] = False,
        loglog: Union[bool, Literal['sym']] = False,
        mark_right: bool = True,
        stacked: bool = False,
        label: Optional[str] = None,
        style: Optional[Union[str, List[str], Dict[str, str]]] = None,
        **kwds: Any,
    ) -> None:
        if by in ([], ()):
            raise ValueError('No group keys passed!')
        self.by: Optional[List[str]] = com.maybe_make_list(by)
        if isinstance(data, ABCDataFrame):
            if column:
                self.columns: List[str] = com.maybe_make_list(column)
            elif self.by is None:
                self.columns = [col for col in data.columns if is_numeric_dtype(data[col])]
            else:
                self.columns = [col for col in data.columns if col not in self.by and is_numeric_dtype(data[col])]
        if self.by is not None and self._kind == 'hist':
            self._grouped: Any = data.groupby(unpack_single_str_list(self.by))
        self.kind: Optional[str] = kind
        self.subplots: Union[bool, List[Tuple[int, ...]]] = type(self)._validate_subplots_kwarg(subplots, data, kind=self._kind)
        self.sharex: bool = type(self)._validate_sharex(sharex, ax, by)
        self.sharey: bool = sharey
        self.figsize: Optional[Tuple[float, float]] = figsize
        self.layout: Optional[Tuple[int, int]] = layout
        self.xticks: Optional[List[float]] = xticks
        self.yticks: Optional[List[float]] = yticks
        self.xlim: Optional[Tuple[float, float]] = xlim
        self.ylim: Optional[Tuple[float, float]] = ylim
        self.title: Optional[Union[str, List[str]]] = title
        self.use_index: bool = use_index
        self.xlabel: Optional[str] = xlabel
        self.ylabel: Optional[str] = ylabel
        self.fontsize: Optional[float] = fontsize
        if rot is not None:
            self.rot: Union[int, float] = rot
            self._rot_set: bool = True
        else:
            self._rot_set = False
            self.rot = self._default_rot
        if grid is None:
            grid = False if secondary_y else mpl.rcParams['axes.grid']
        self.grid: bool = grid
        self.legend: Union[bool, Literal['reverse']] = legend
        self.legend_handles: List[Any] = []
        self.legend_labels: List[str] = []
        self.logx: Union[bool, Literal['sym']] = type(self)._validate_log_kwd('logx', logx)
        self.logy: Union[bool, Literal['sym']] = type(self)._validate_log_kwd('logy', logy)
        self.loglog: Union[bool, Literal['sym']] = type(self)._validate_log_kwd('loglog', loglog)
        self.label: Optional[str] = label
        self.style: Optional[Union[str, List[str], Dict[str, str]]] = style
        self.mark_right: bool = mark_right
        self.stacked: bool = stacked
        self.ax: Optional[Axes] = ax
        xerr: Optional[Union[Series, np.ndarray, DataFrame, Dict[str, Any], str, float]] = kwds.pop('xerr', None)
        yerr: Optional[Union[Series, np.ndarray, DataFrame, Dict[str, Any], str, float]] = kwds.pop('yerr', None)
        nseries: int = self._get_nseries(data)
        xerr, data = type(self)._parse_errorbars('xerr', xerr, data, nseries)
        yerr, data = type(self)._parse_errorbars('yerr', yerr, data, nseries)
        self.errors: Dict[str, Optional[Union[np.ndarray, DataFrame, Dict[str, Any]]]] = {'xerr': xerr, 'yerr': yerr}
        self.data: NDFrameT = data
        if not isinstance(secondary_y, (bool, tuple, list, np.ndarray, ABCIndex)):
            secondary_y = [secondary_y]
        self.secondary_y: Union[bool, List[Hashable], Tuple[Hashable, ...], np.ndarray, Index] = secondary_y
        if 'cmap' in kwds and colormap:
            raise TypeError('Only specify one of `cmap` and `colormap`.')
        if 'cmap' in kwds:
            self.colormap: Optional[str] = kwds.pop('cmap')
        else:
            self.colormap = colormap
        self.table: Union[bool, Any] = table
        self.include_bool: bool = include_bool
        self.kwds: Dict[str, Any] = kwds
        color: Union[Any, mpl.colors.ColorLike] = kwds.pop('color', lib.no_default)
        self.color: Optional[Union[List[Any], Dict[Any, Any]]] = self._validate_color_args(color, self.colormap)
        assert 'color' not in self.kwds
        self.data = self._ensure_frame(self.data)

    @final
    @staticmethod
    def _validate_sharex(sharex: Optional[bool], ax: Optional[Axes], by: Optional[Union[str, List[str], Tuple[str, ...]]]) -> bool:
        if sharex is None:
            if ax is None and by is None:
                sharex = True
            else:
                sharex = False
        elif not is_bool(sharex):
            raise TypeError('sharex must be a bool or None')
        return bool(sharex)

    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: Union[bool, Literal['sym'], None]) -> Union[bool, Literal['sym'], None]:
        if value is None or isinstance(value, bool) or (isinstance(value, str) and value == 'sym'):
            return value
        raise ValueError(f"keyword '{kwd}' should be bool, None, or 'sym', not '{value}'")

    @final
    @staticmethod
    def _validate_subplots_kwarg(
        subplots: Union[bool, Iterable[Union[str, List[str], Tuple[str, ...]]]],
        data: NDFrameT,
        kind: str,
    ) -> Union[bool, List[Tuple[int, ...]]]:
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
        seen_columns: set = set()
        for group in subplots:
            if not is_list_like(group):
                raise ValueError('When subplots is an iterable, each entry should be a list/tuple of column names.')
            idx_locs = columns.get_indexer_for(group)
            if (idx_locs == -1).any():
                bad_labels = list(np.extract(idx_locs == -1, group))
                raise ValueError(f'Column label(s) {bad_labels} not found in the DataFrame.')
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

    def _validate_color_args(self, color: Any, colormap: Optional[str]) -> Optional[Union[List[Any], Dict[Any, Any]]]:
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
    def _iter_data(data: DataFrame) -> Iterator[Tuple[str, np.ndarray]]:
        for col, values in data.items():
            yield (col, np.asarray(values.values))

    def _get_nseries(self, data: NDFrameT) -> int:
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
        fig: Figure = self.fig
        self._make_plot(fig)
        self._add_table()
        self._make_legend()
        self._adorn_subplots(fig)
        for ax in self.axes:
            self._post_plot_logic_common(ax)
            self._post_plot_logic(ax, self.data)

    @final
    @staticmethod
    def _has_plotted_object(ax: Axes) -> bool:
        """check whether ax has data"""
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    @final
    def _maybe_right_yaxis(self, ax: Axes, axes_num: int) -> Axes:
        if not self.on_right(axes_num):
            return self._get_ax_layer(ax)
        if hasattr(ax, 'right_ax'):
            return cast(Axes, ax.right_ax)
        elif hasattr(ax, 'left_ax'):
            return ax
        else:
            orig_ax = ax
            new_ax = ax.twinx()
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
    def fig(self) -> Figure:
        return self._axes_and_fig[1]

    @final
    @cache_readonly
    def axes(self) -> Sequence[Axes]:
        return self._axes_and_fig[0]

    @final
    @cache_readonly
    def _axes_and_fig(self) -> Tuple[Sequence[Axes], Figure]:
        import matplotlib.pyplot as plt
        if self.subplots:
            naxes = self.nseries if isinstance(self.subplots, bool) else len(self.subplots)
            fig, axes = create_subplots(
                naxes=naxes,
                sharex=self.sharex,
                sharey=self.sharey,
                figsize=self.figsize,
                ax=self.ax,
                layout=self.layout,
                layout_type=self._layout_type,
            )
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
            for a in axes:
                a.set_xscale('log')
        elif self.logx == 'sym' or self.loglog == 'sym':
            for a in axes:
                a.set_xscale('symlog')
        if self.logy is True or self.loglog is True:
            for a in axes:
                a.set_yscale('log')
        elif self.logy == 'sym' or self.loglog == 'sym':
            for a in axes:
                a.set_yscale('symlog')
        axes_seq: Sequence[Axes] = cast(Sequence[Axes], axes)
        return (axes_seq, fig)

    @property
    def result(self) -> Union[Axes, np.ndarray]:
        """
        Return result axes
        """
        if self.subplots:
            if self.layout is not None and (not is_list_like(self.ax)):
                return self.axes.reshape(*self.layout)
            else:
                return self.axes
        else:
            sec_true: bool = isinstance(self.secondary_y, bool) and self.secondary_y
            all_sec: bool = is_list_like(self.secondary_y) and len(self.secondary_y) == self.nseries
            if sec_true or all_sec:
                return self._get_ax_layer(self.axes[0], primary=False)
            else:
                return self.axes[0]

    @final
    @staticmethod
    def _convert_to_ndarray(data: Series) -> Union[np.ndarray, Series]:
        if isinstance(data.dtype, CategoricalDtype):
            return data
        if (is_integer_dtype(data.dtype) or is_float_dtype(data.dtype)) and isinstance(data.dtype, ExtensionDtype):
            return data.to_numpy(dtype='float', na_value=np.nan)
        if len(data) > 0:
            return np.asarray(data)
        return data

    @final
    def _ensure_frame(self, data: NDFrameT) -> DataFrame:
        if isinstance(data, ABCSeries):
            label = self.label
            if label is None and data.name is None:
                label = ''
            if label is None:
                data = data.to_frame()
            else:
                data = data.to_frame(name=label)
        elif self._kind in ('hist', 'box'):
            cols = self.columns if self.by is None else self.columns + self.by
            data = data.loc[:, cols]
        return data

    @final
    def _compute_plot_data(self) -> None:
        data = self.data
        if self.by is not None:
            self.subplots = True
            data = reconstruct_data_with_by(self.data, by=self.by, cols=self.columns)
        data = data.infer_objects()
        include_type: List[Union[type, str]] = [np.number, 'datetime', 'datetimetz', 'timedelta']
        if self.include_bool is True:
            include_type.append(np.bool_)
        exclude_type: Optional[List[str]] = None
        if self._kind == 'box':
            include_type = [np.number]
            exclude_type = ['timedelta']
        if self._kind == 'scatter':
            include_type.extend(['object', 'category', 'string'])
        numeric_data: DataFrame = data.select_dtypes(include=include_type, exclude=exclude_type)
        is_empty: bool = numeric_data.shape[-1] == 0
        if is_empty:
            raise TypeError('no numeric data to plot')
        self.data = numeric_data.apply(self._convert_to_ndarray)

    @abstractmethod
    def _make_plot(self, fig: Figure) -> None:
        raise AbstractMethodError(self)

    @final
    def _add_table(self) -> None:
        if self.table is False:
            return
        elif self.table is True:
            data = self.data.transpose()
        else:
            data = self.table
        ax = self._get_ax(0)
        tools.table(ax, data)

    @final
    def _post_plot_logic_common(self, ax: Axes) -> None:
        """Common post process for each axes"""
        if self.orientation == 'vertical' or self.orientation is None:
            type(self)._apply_axis_properties(ax.xaxis, rot=self.rot, fontsize=self.fontsize)
            type(self)._apply_axis_properties(ax.yaxis, fontsize=self.fontsize)
            if hasattr(ax, 'right_ax'):
                type(self)._apply_axis_properties(ax.right_ax.yaxis, fontsize=self.fontsize)
        elif self.orientation == 'horizontal':
            type(self)._apply_axis_properties(ax.yaxis, rot=self.rot, fontsize=self.fontsize)
            type(self)._apply_axis_properties(ax.xaxis, fontsize=self.fontsize)
            if hasattr(ax, 'right_ax'):
                type(self)._apply_axis_properties(ax.right_ax.yaxis, fontsize=self.fontsize)
        else:
            raise ValueError

    @abstractmethod
    def _post_plot_logic(self, ax: Axes, data: DataFrame) -> None:
        """Post process for each axes. Overridden in child classes"""

    @final
    def _adorn_subplots(self, fig: Figure) -> None:
        """Common post process unrelated to data"""
        if len(self.axes) > 0:
            all_axes: List[Axes] = self._get_subplots(fig)
            nrows, ncols = self._get_axes_layout(fig)
            handle_shared_axes(
                axarr=all_axes,
                nplots=len(all_axes),
                naxes=nrows * ncols,
                nrows=nrows,
                ncols=ncols,
                sharex=self.sharex,
                sharey=self.sharey,
            )
        for ax in self.axes:
            ax = getattr(ax, 'right_ax', ax)
            if self.yticks is not None:
                ax.set_yticks(self.yticks)
            if self.xticks is not None:
                ax.set_xticks(self.xticks)
            if self.ylim is not None:
                ax.set_ylim(self.ylim)
            if self.xlim is not None:
                ax.set_xlim(self.xlim)
            if self.ylabel is not None:
                ax.set_ylabel(pprint_thing(self.ylabel))
            ax.grid(self.grid)
        if self.title:
            if self.subplots:
                if is_list_like(self.title):
                    if len(self.title) != self.nseries:
                        raise ValueError(
                            f'The length of `title` must equal the number of columns if using `title` of type `list` and `subplots=True`.\nlength of title = {len(self.title)}\nnumber of columns = {self.nseries}'
                        )
                    for ax, title in zip(self.axes, self.title):
                        ax.set_title(title)
                else:
                    fig.suptitle(self.title)
            else:
                if is_list_like(self.title):
                    msg = 'Using `title` of type `list` is not supported unless `subplots=True` is passed'
                    raise ValueError(msg)
                self.axes[0].set_title(self.title)

    @final
    @staticmethod
    def _apply_axis_properties(axis: Axis, rot: Optional[Union[int, float]] = None, fontsize: Optional[float] = None) -> None:
        """
        Tick creation within matplotlib is reasonably expensive and is
        internally deferred until accessed as Ticks are created/destroyed
        multiple times per draw. It's therefore beneficial for us to avoid
        accessing unless we will act on the Tick.
        """
        if rot is not None or fontsize is not None:
            labels = axis.get_majorticklabels() + axis.get_minorticklabels()
            for label in labels:
                if rot is not None:
                    label.set_rotation(rot)
                if fontsize is not None:
                    label.set_fontsize(fontsize)

    @final
    @property
    def legend_title(self) -> Optional[str]:
        if not isinstance(self.data.columns, ABCMultiIndex):
            name = self.data.columns.name
            if name is not None:
                name = pprint_thing(name)
            return name
        else:
            stringified = map(pprint_thing, self.data.columns.names)
            return ','.join(stringified)

    @final
    def _mark_right_label(self, label: str, index: int) -> str:
        """
        Append ``(right)`` to the label of a line if it's plotted on the right axis.

        Note that ``(right)`` is only appended when ``subplots=False``.
        """
        if not self.subplots and self.mark_right and self.on_right(index):
            label += ' (right)'
        return label

    @final
    def _append_legend_handles_labels(self, handle: Any, label: str) -> None:
        """
        Append current handle and label to ``legend_handles`` and ``legend_labels``.

        These will be used to make the legend.
        """
        self.legend_handles.append(handle)
        self.legend_labels.append(label)

    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])
        handles: List[Any] = []
        labels: List[str] = []
        title: str = ''
        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                if Version(mpl.__version__) < Version('3.7'):
                    handles = leg.legendHandles
                else:
                    handles = leg.legend_handles
                labels = [x.get_text() for x in leg.get_texts()]
            if self.legend:
                if self.legend == 'reverse':
                    handles += list(reversed(self.legend_handles))
                    labels += list(reversed(self.legend_labels))
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels
                if self.legend_title is not None:
                    title = self.legend_title
            if len(handles) > 0:
                ax.legend(handles, labels, loc='best', title=title)
        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'No artists with labels found to put in legend.', UserWarning)
                        ax.legend(loc='best')

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes) -> Tuple[Axes, Optional[Any]]:
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()
        other_ax = getattr(ax, 'left_ax', None) or getattr(ax, 'right_ax', None)
        other_leg: Optional[Any] = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return (ax, leg)

    _need_to_set_index: bool = False

    @final
    def _get_xticks(self) -> List[Union[int, float]]:
        index = self.data.index
        is_datetype: bool = index.inferred_type in ('datetime', 'date', 'datetime64', 'time')
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                x: List[Union[float, int]] = index.to_timestamp()._mpl_repr()
            elif is_any_real_numeric_dtype(index.dtype):
                x = index._mpl_repr()
            elif isinstance(index, ABCDatetimeIndex) or is_datetype:
                x = index._mpl_repr()
            else:
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            x = list(range(len(index)))
        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls,
        ax: Axes,
        x: Union[List[Union[int, float]], np.ndarray, ABCIndex],
        y: Union[List[float], np.ndarray],
        style: Optional[str] = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> List[Line2D]:
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)
        if isinstance(x, ABCIndex):
            x = x._mpl_repr()
        if is_errorbar:
            if 'xerr' in kwds:
                kwds['xerr'] = np.array(kwds.get('xerr'))
            if 'yerr' in kwds:
                kwds['yerr'] = np.array(kwds.get('yerr'))
            return ax.errorbar(x, y, **kwds)
        else:
            args: Tuple[Any, ...]
            if style is not None:
                args = (x, y, style)
            else:
                args = (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self) -> Optional[str]:
        """Specify whether xlabel/ylabel should be used to override index name"""
        return self.xlabel

    @final
    def _get_index_name(self) -> Optional[str]:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ','.join([pprint_thing(x) for x in name])
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)
        return name

    @final
    @classmethod
    def _get_ax_layer(cls, ax: Axes, primary: bool = True) -> Axes:
        """get left (primary) or right (secondary) axes"""
        if primary:
            return getattr(ax, 'left_ax', ax)
        else:
            return getattr(ax, 'right_ax', ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
        if isinstance(self.subplots, list):
            return next(
                (
                    group_idx
                    for group_idx, group in enumerate(self.subplots)
                    if col_idx in group
                )
            )
        else:
            return col_idx

    @final
    def _get_ax(self, i: int) -> Axes:
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            self.axes[i] = ax
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)
        ax.get_yaxis().set_visible(True)
        return ax

    @final
    def on_right(self, i: int) -> bool:
        if isinstance(self.secondary_y, bool):
            return self.secondary_y
        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y
        return False

    @final
    def _apply_style_colors(
        self,
        colors: Union[List[Any], Dict[Any, Any]],
        kwds: Dict[str, Any],
        col_num: int,
        label: str,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style: Optional[str] = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style
        has_color: bool = 'color' in kwds or self.colormap is not None
        nocolor_style: bool = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds['color'] = colors[label]
            else:
                kwds['color'] = colors[col_num % len(colors)]
        return (style, kwds)

    def _get_colors(
        self,
        num_colors: Optional[int] = None,
        color_kwds: str = 'color',
    ) -> Union[List[Any], Dict[Any, Any]]:
        if num_colors is None:
            num_colors = self.nseries
        if color_kwds == 'color':
            color = self.color
        else:
            color = self.kwds.get(color_kwds)
        return get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=color)

    @final
    @staticmethod
    def _parse_errorbars(
        label: str,
        err: Optional[Union[Series, np.ndarray, DataFrame, Dict[str, Any], str, float]],
        data: NDFrameT,
        nseries: int,
    ) -> Tuple[Optional[Union[np.ndarray, DataFrame, Dict[str, Any]]], NDFrameT]:
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
        if err is None:
            return (None, data)

        def match_labels(data: NDFrameT, e: Series) -> Series:
            e = e.reindex(data.index)
            return e

        if isinstance(err, ABCDataFrame):
            err = match_labels(data, err)
        elif isinstance(err, dict):
            pass
        elif isinstance(err, ABCSeries):
            err = match_labels(data, err)
            err = np.atleast_2d(err)
            err = np.tile(err, (nseries, 1))
        elif isinstance(err, str):
            evalues = data[err].values
            data = data[data.columns.drop(err)]
            err = np.atleast_2d(evalues)
            err = np.tile(err, (nseries, 1))
        elif is_list_like(err):
            if is_iterator(err):
                err = np.atleast_2d(list(err))
            else:
                err = np.atleast_2d(err)
            err_shape = err.shape
            if isinstance(data, ABCSeries) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape (2, {len(data)})')
            elif isinstance(data, ABCDataFrame) and err.ndim == 3:
                if err_shape[0] != nseries or err_shape[1] != 2 or err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape ({nseries}, 2, {len(data)})')
            if len(err) == 1:
                err = np.tile(err, (nseries, 1))
        elif is_number(err):
            err = np.tile([err], (nseries, len(data)))
        else:
            msg = f'No valid {label} detected'
            raise ValueError(msg)
        return (err, data)

    @final
    def _get_errorbars(
        self,
        label: Optional[str] = None,
        index: Optional[int] = None,
        xerr: bool = True,
        yerr: bool = True,
    ) -> Dict[str, Any]:
        errors: Dict[str, Any] = {}
        for kw, flag in zip(['xerr', 'yerr'], [xerr, yerr]):
            if flag:
                err = self.errors[kw]
                if isinstance(err, (ABCDataFrame, dict)):
                    if label is not None and label in err.keys():
                        err = err[label]
                    else:
                        err = None
                elif index is not None and err is not None:
                    err = err[index]
                if err is not None:
                    errors[kw] = err
        return errors

    @final
    def _get_subplots(self, fig: Figure) -> List[Axes]:
        if Version(mpl.__version__) < Version('3.8'):
            Klass = mpl.axes.Subplot
        else:
            Klass = mpl.axes.Axes
        return [ax for ax in fig.get_axes() if isinstance(ax, Klass) and ax.get_subplotspec() is not None]

    @final
    def _get_axes_layout(self, fig: Figure) -> Tuple[int, int]:
        axes = self._get_subplots(fig)
        x_set: set = set()
        y_set: set = set()
        for ax in axes:
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))

    @final
    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])
        handles: List[Any] = []
        labels: List[str] = []
        title: str = ''
        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                if Version(mpl.__version__) < Version('3.7'):
                    handles = leg.legendHandles  # type: ignore
                else:
                    handles = leg.legend_handles  # type: ignore
                labels = [x.get_text() for x in leg.get_texts()]
            if self.legend:
                if self.legend == 'reverse':
                    handles += list(reversed(self.legend_handles))
                    labels += list(reversed(self.legend_labels))
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels
                if self.legend_title is not None:
                    title = self.legend_title
            if len(handles) > 0:
                ax.legend(handles, labels, loc='best', title=title)
        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'No artists with labels found to put in legend.', UserWarning)
                        ax.legend(loc='best')

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes) -> Tuple[Axes, Optional[Any]]:
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()
        other_ax: Optional[Axes] = getattr(ax, 'left_ax', None) or getattr(ax, 'right_ax', None)
        other_leg: Optional[Any] = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return (ax, leg)

    _need_to_set_index: bool = False

    @final
    def _get_xticks(self) -> List[Union[int, float]]:
        index = self.data.index
        is_datetype: bool = index.inferred_type in ('datetime', 'date', 'datetime64', 'time')
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                x: List[Union[int, float]] = index.to_timestamp()._mpl_repr()
            elif is_any_real_numeric_dtype(index.dtype):
                x = index._mpl_repr()
            elif isinstance(index, ABCDatetimeIndex) or is_datetype:
                x = index._mpl_repr()
            else:
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            x = list(range(len(index)))
        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls,
        ax: Axes,
        x: Union[List[Union[int, float]], np.ndarray, ABCIndex],
        y: Union[List[float], np.ndarray],
        style: Optional[str] = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> List[mpl.lines.Line2D]:
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)
        if isinstance(x, ABCIndex):
            x = x._mpl_repr()
        if is_errorbar:
            if 'xerr' in kwds:
                kwds['xerr'] = np.array(kwds.get('xerr'))
            if 'yerr' in kwds:
                kwds['yerr'] = np.array(kwds.get('yerr'))
            return ax.errorbar(x, y, **kwds)
        else:
            args: Tuple[Any, ...]
            if style is not None:
                args = (x, y, style)
            else:
                args = (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self) -> Optional[str]:
        """Specify whether xlabel/ylabel should be used to override index name"""
        return self.xlabel

    @final
    def _get_index_name(self) -> Optional[str]:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ','.join([pprint_thing(x) for x in name])
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)
        return name

    @final
    @classmethod
    def _get_ax_layer(cls, ax: Axes, primary: bool = True) -> Axes:
        """get left (primary) or right (secondary) axes"""
        if primary:
            return getattr(ax, 'left_ax', ax)
        else:
            return getattr(ax, 'right_ax', ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
        if isinstance(self.subplots, list):
            return next(
                (
                    group_idx
                    for group_idx, group in enumerate(self.subplots)
                    if col_idx in group
                )
            )
        else:
            return col_idx

    @final
    def _get_ax(self, i: int) -> Axes:
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax: Axes = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            self.axes[i] = ax
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)
        ax.get_yaxis().set_visible(True)
        return ax

    @final
    def on_right(self, i: int) -> bool:
        if isinstance(self.secondary_y, bool):
            return self.secondary_y
        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y
        return False

    @final
    def _apply_style_colors(
        self,
        colors: Union[List[Any], Dict[Any, Any]],
        kwds: Dict[str, Any],
        col_num: int,
        label: str,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style: Optional[str] = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style
        has_color: bool = 'color' in kwds or self.colormap is not None
        nocolor_style: bool = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds['color'] = colors[label]
            else:
                kwds['color'] = colors[col_num % len(colors)]
        return (style, kwds)

    def _get_colors(
        self,
        num_colors: Optional[int] = None,
        color_kwds: str = 'color',
    ) -> Union[List[Any], Dict[Any, Any]]:
        if num_colors is None:
            num_colors = self.nseries
        if color_kwds == 'color':
            color = self.color
        else:
            color = self.kwds.get(color_kwds)
        return get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=color)

    @final
    @staticmethod
    def _parse_errorbars(
        label: str,
        err: Optional[Union[Series, np.ndarray, DataFrame, Dict[str, Any], str, float]],
        data: NDFrameT,
        nseries: int,
    ) -> Tuple[Optional[Union[np.ndarray, DataFrame, Dict[str, Any]]], NDFrameT]:
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
        if err is None:
            return (None, data)

        def match_labels(data: NDFrameT, e: Series) -> Series:
            e = e.reindex(data.index)
            return e

        if isinstance(err, ABCDataFrame):
            err = match_labels(data, err)
        elif isinstance(err, dict):
            pass
        elif isinstance(err, ABCSeries):
            err = match_labels(data, err)
            err = np.atleast_2d(err)
            err = np.tile(err, (nseries, 1))
        elif isinstance(err, str):
            evalues = data[err].values
            data = data[data.columns.drop(err)]
            err = np.atleast_2d(evalues)
            err = np.tile(err, (nseries, 1))
        elif is_list_like(err):
            if is_iterator(err):
                err = np.atleast_2d(list(err))
            else:
                err = np.atleast_2d(err)
            err_shape = err.shape
            if isinstance(data, ABCSeries) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape (2, {len(data)})')
            elif isinstance(data, ABCDataFrame) and err.ndim == 3:
                if err_shape[0] != nseries or err_shape[1] != 2 or err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape ({nseries}, 2, {len(data)})')
            if len(err) == 1:
                err = np.tile(err, (nseries, 1))
        elif is_number(err):
            err = np.tile([err], (nseries, len(data)))
        else:
            msg = f'No valid {label} detected'
            raise ValueError(msg)
        return (err, data)

    @final
    def _get_errorbars(
        self,
        label: Optional[str] = None,
        index: Optional[int] = None,
        xerr: bool = True,
        yerr: bool = True,
    ) -> Dict[str, Any]:
        errors: Dict[str, Any] = {}
        for kw, flag in zip(['xerr', 'yerr'], [xerr, yerr]):
            if flag:
                err = self.errors[kw]
                if isinstance(err, (ABCDataFrame, dict)):
                    if label is not None and label in err.keys():
                        err = err[label]
                    else:
                        err = None
                elif index is not None and err is not None:
                    err = err[index]
                if err is not None:
                    errors[kw] = err
        return errors

    @final
    def _get_subplots(self, fig: Figure) -> List[Axes]:
        if Version(mpl.__version__) < Version('3.8'):
            Klass = mpl.axes.Subplot
        else:
            Klass = mpl.axes.Axes
        return [ax for ax in fig.get_axes() if isinstance(ax, Klass) and ax.get_subplotspec() is not None]

    @final
    def _get_axes_layout(self, fig: Figure) -> Tuple[int, int]:
        axes = self._get_subplots(fig)
        x_set: set = set()
        y_set: set = set()
        for ax in axes:
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))

    @final
    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])
        handles: List[Any] = []
        labels: List[str] = []
        title: str = ''
        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                if Version(mpl.__version__) < Version('3.7'):
                    handles = leg.legendHandles  # type: ignore
                else:
                    handles = leg.legend_handles  # type: ignore
                labels = [x.get_text() for x in leg.get_texts()]
            if self.legend:
                if self.legend == 'reverse':
                    handles += list(reversed(self.legend_handles))
                    labels += list(reversed(self.legend_labels))
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels
                if self.legend_title is not None:
                    title = self.legend_title
            if len(handles) > 0:
                ax.legend(handles, labels, loc='best', title=title)
        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'No artists with labels found to put in legend.', UserWarning)
                        ax.legend(loc='best')

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes) -> Tuple[Axes, Optional[Any]]:
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()
        other_ax: Optional[Axes] = getattr(ax, 'left_ax', None) or getattr(ax, 'right_ax', None)
        other_leg: Optional[Any] = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return (ax, leg)

    _need_to_set_index: bool = False

    @final
    def _get_xticks(self) -> List[Union[int, float]]:
        index = self.data.index
        is_datetype: bool = index.inferred_type in ('datetime', 'date', 'datetime64', 'time')
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                x: List[Union[int, float]] = index.to_timestamp()._mpl_repr()
            elif is_any_real_numeric_dtype(index.dtype):
                x = index._mpl_repr()
            elif isinstance(index, ABCDatetimeIndex) or is_datetype:
                x = index._mpl_repr()
            else:
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            x = list(range(len(index)))
        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls,
        ax: Axes,
        x: Union[List[Union[int, float]], np.ndarray, ABCIndex],
        y: Union[List[float], np.ndarray],
        style: Optional[str] = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> List[mpl.lines.Line2D]:
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)
        if isinstance(x, ABCIndex):
            x = x._mpl_repr()
        if is_errorbar:
            if 'xerr' in kwds:
                kwds['xerr'] = np.array(kwds.get('xerr'))
            if 'yerr' in kwds:
                kwds['yerr'] = np.array(kwds.get('yerr'))
            return ax.errorbar(x, y, **kwds)
        else:
            args: Tuple[Any, ...]
            if style is not None:
                args = (x, y, style)
            else:
                args = (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self) -> Optional[str]:
        """Specify whether xlabel/ylabel should be used to override index name"""
        return self.xlabel

    @final
    def _get_index_name(self) -> Optional[str]:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ','.join([pprint_thing(x) for x in name])
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)
        return name

    @final
    @classmethod
    def _get_ax_layer(cls, ax: Axes, primary: bool = True) -> Axes:
        """get left (primary) or right (secondary) axes"""
        if primary:
            return getattr(ax, 'left_ax', ax)
        else:
            return getattr(ax, 'right_ax', ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
        if isinstance(self.subplots, list):
            return next(
                (
                    group_idx
                    for group_idx, group in enumerate(self.subplots)
                    if col_idx in group
                )
            )
        else:
            return col_idx

    @final
    def _get_ax(self, i: int) -> Axes:
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax: Axes = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            self.axes[i] = ax
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)
        ax.get_yaxis().set_visible(True)
        return ax

    @final
    def on_right(self, i: int) -> bool:
        if isinstance(self.secondary_y, bool):
            return self.secondary_y
        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y
        return False

    @final
    def _apply_style_colors(
        self,
        colors: Union[List[Any], Dict[Any, Any]],
        kwds: Dict[str, Any],
        col_num: int,
        label: str,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style: Optional[str] = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style
        has_color: bool = 'color' in kwds or self.colormap is not None
        nocolor_style: bool = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds['color'] = colors[label]
            else:
                kwds['color'] = colors[col_num % len(colors)]
        return (style, kwds)

    def _get_colors(
        self,
        num_colors: Optional[int] = None,
        color_kwds: str = 'color',
    ) -> Union[List[Any], Dict[Any, Any]]:
        if num_colors is None:
            num_colors = self.nseries
        if color_kwds == 'color':
            color = self.color
        else:
            color = self.kwds.get(color_kwds)
        return get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=color)

    @final
    @staticmethod
    def _parse_errorbars(
        label: str,
        err: Optional[Union[Series, np.ndarray, DataFrame, Dict[str, Any], str, float]],
        data: NDFrameT,
        nseries: int,
    ) -> Tuple[Optional[Union[np.ndarray, DataFrame, Dict[str, Any]]], NDFrameT]:
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
        if err is None:
            return (None, data)

        def match_labels(data: NDFrameT, e: Series) -> Series:
            e = e.reindex(data.index)
            return e

        if isinstance(err, ABCDataFrame):
            err = match_labels(data, err)
        elif isinstance(err, dict):
            pass
        elif isinstance(err, ABCSeries):
            err = match_labels(data, err)
            err = np.atleast_2d(err)
            err = np.tile(err, (nseries, 1))
        elif isinstance(err, str):
            evalues = data[err].values
            data = data[data.columns.drop(err)]
            err = np.atleast_2d(evalues)
            err = np.tile(err, (nseries, 1))
        elif is_list_like(err):
            if is_iterator(err):
                err = np.atleast_2d(list(err))
            else:
                err = np.atleast_2d(err)
            err_shape = err.shape
            if isinstance(data, ABCSeries) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape (2, {len(data)})')
            elif isinstance(data, ABCDataFrame) and err.ndim == 3:
                if err_shape[0] != nseries or err_shape[1] != 2 or err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape ({nseries}, 2, {len(data)})')
            if len(err) == 1:
                err = np.tile(err, (nseries, 1))
        elif is_number(err):
            err = np.tile([err], (nseries, len(data)))
        else:
            msg = f'No valid {label} detected'
            raise ValueError(msg)
        return (err, data)

    @final
    def _get_errorbars(
        self,
        label: Optional[str] = None,
        index: Optional[int] = None,
        xerr: bool = True,
        yerr: bool = True,
    ) -> Dict[str, Any]:
        errors: Dict[str, Any] = {}
        for kw, flag in zip(['xerr', 'yerr'], [xerr, yerr]):
            if flag:
                err = self.errors[kw]
                if isinstance(err, (ABCDataFrame, dict)):
                    if label is not None and label in err.keys():
                        err = err[label]
                    else:
                        err = None
                elif index is not None and err is not None:
                    err = err[index]
                if err is not None:
                    errors[kw] = err
        return errors

    @final
    def _get_subplots(self, fig: Figure) -> List[Axes]:
        if Version(mpl.__version__) < Version('3.8'):
            Klass = mpl.axes.Subplot
        else:
            Klass = mpl.axes.Axes
        return [ax for ax in fig.get_axes() if isinstance(ax, Klass) and ax.get_subplotspec() is not None]

    @final
    def _get_axes_layout(self, fig: Figure) -> Tuple[int, int]:
        axes = self._get_subplots(fig)
        x_set: set = set()
        y_set: set = set()
        for ax in axes:
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))

    @final
    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])
        handles: List[Any] = []
        labels: List[str] = []
        title: str = ''
        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                if Version(mpl.__version__) < Version('3.7'):
                    handles = leg.legendHandles  # type: ignore
                else:
                    handles = leg.legend_handles  # type: ignore
                labels = [x.get_text() for x in leg.get_texts()]
            if self.legend:
                if self.legend == 'reverse':
                    handles += list(reversed(self.legend_handles))
                    labels += list(reversed(self.legend_labels))
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels
                if self.legend_title is not None:
                    title = self.legend_title
            if len(handles) > 0:
                ax.legend(handles, labels, loc='best', title=title)
        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'No artists with labels found to put in legend.', UserWarning)
                        ax.legend(loc='best')

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes) -> Tuple[Axes, Optional[Any]]:
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()
        other_ax: Optional[Axes] = getattr(ax, 'left_ax', None) or getattr(ax, 'right_ax', None)
        other_leg: Optional[Any] = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return (ax, leg)

    _need_to_set_index: bool = False

    @final
    def _get_xticks(self) -> List[Union[int, float]]:
        index = self.data.index
        is_datetype: bool = index.inferred_type in ('datetime', 'date', 'datetime64', 'time')
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                x: List[Union[int, float]] = index.to_timestamp()._mpl_repr()
            elif is_any_real_numeric_dtype(index.dtype):
                x = index._mpl_repr()
            elif isinstance(index, ABCDatetimeIndex) or is_datetype:
                x = index._mpl_repr()
            else:
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            x = list(range(len(index)))
        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls,
        ax: Axes,
        x: Union[List[Union[int, float]], np.ndarray, ABCIndex],
        y: Union[List[float], np.ndarray],
        style: Optional[str] = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> List[mpl.lines.Line2D]:
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)
        if isinstance(x, ABCIndex):
            x = x._mpl_repr()
        if is_errorbar:
            if 'xerr' in kwds:
                kwds['xerr'] = np.array(kwds.get('xerr'))
            if 'yerr' in kwds:
                kwds['yerr'] = np.array(kwds.get('yerr'))
            return ax.errorbar(x, y, **kwds)
        else:
            args: Tuple[Any, ...]
            if style is not None:
                args = (x, y, style)
            else:
                args = (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self) -> Optional[str]:
        """Specify whether xlabel/ylabel should be used to override index name"""
        return self.xlabel

    @final
    def _get_index_name(self) -> Optional[str]:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ','.join([pprint_thing(x) for x in name])
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)
        return name

    @final
    @classmethod
    def _get_ax_layer(cls, ax: Axes, primary: bool = True) -> Axes:
        """get left (primary) or right (secondary) axes"""
        if primary:
            return getattr(ax, 'left_ax', ax)
        else:
            return getattr(ax, 'right_ax', ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
        if isinstance(self.subplots, list):
            return next(
                (
                    group_idx
                    for group_idx, group in enumerate(self.subplots)
                    if col_idx in group
                )
            )
        else:
            return col_idx

    @final
    def _get_ax(self, i: int) -> Axes:
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax: Axes = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            self.axes[i] = ax
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)
        ax.get_yaxis().set_visible(True)
        return ax

    @final
    def on_right(self, i: int) -> bool:
        if isinstance(self.secondary_y, bool):
            return self.secondary_y
        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y
        return False

    @final
    def _apply_style_colors(
        self,
        colors: Union[List[Any], Dict[Any, Any]],
        kwds: Dict[str, Any],
        col_num: int,
        label: str,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style: Optional[str] = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style
        has_color: bool = 'color' in kwds or self.colormap is not None
        nocolor_style: bool = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds['color'] = colors[label]
            else:
                kwds['color'] = colors[col_num % len(colors)]
        return (style, kwds)

    def _get_colors(
        self,
        num_colors: Optional[int] = None,
        color_kwds: str = 'color',
    ) -> Union[List[Any], Dict[Any, Any]]:
        if num_colors is None:
            num_colors = self.nseries
        if color_kwds == 'color':
            color = self.color
        else:
            color = self.kwds.get(color_kwds)
        return get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=color)

    @final
    @staticmethod
    def _parse_errorbars(
        label: str,
        err: Optional[Union[Series, np.ndarray, DataFrame, Dict[str, Any], str, float]],
        data: NDFrameT,
        nseries: int,
    ) -> Tuple[Optional[Union[np.ndarray, DataFrame, Dict[str, Any]]], NDFrameT]:
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
        if err is None:
            return (None, data)

        def match_labels(data: NDFrameT, e: Series) -> Series:
            e = e.reindex(data.index)
            return e

        if isinstance(err, ABCDataFrame):
            err = match_labels(data, err)
        elif isinstance(err, dict):
            pass
        elif isinstance(err, ABCSeries):
            err = match_labels(data, err)
            err = np.atleast_2d(err)
            err = np.tile(err, (nseries, 1))
        elif isinstance(err, str):
            evalues = data[err].values
            data = data[data.columns.drop(err)]
            err = np.atleast_2d(evalues)
            err = np.tile(err, (nseries, 1))
        elif is_list_like(err):
            if is_iterator(err):
                err = np.atleast_2d(list(err))
            else:
                err = np.atleast_2d(err)
            err_shape = err.shape
            if isinstance(data, ABCSeries) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape (2, {len(data)})')
            elif isinstance(data, ABCDataFrame) and err.ndim == 3:
                if err_shape[0] != nseries or err_shape[1] != 2 or err_shape[2] != len(data):
                    raise ValueError(f'Asymmetrical error bars should be provided with the shape ({nseries}, 2, {len(data)})')
            if len(err) == 1:
                err = np.tile(err, (nseries, 1))
        elif is_number(err):
            err = np.tile([err], (nseries, len(data)))
        else:
            msg = f'No valid {label} detected'
            raise ValueError(msg)
        return (err, data)

    @final
    def _get_errorbars(
        self,
        label: Optional[str] = None,
        index: Optional[int] = None,
        xerr: bool = True,
        yerr: bool = True,
    ) -> Dict[str, Any]:
        errors: Dict[str, Any] = {}
        for kw, flag in zip(['xerr', 'yerr'], [xerr, yerr]):
            if flag:
                err = self.errors[kw]
                if isinstance(err, (ABCDataFrame, dict)):
                    if label is not None and label in err.keys():
                        err = err[label]
                    else:
                        err = None
                elif index is not None and err is not None:
                    err = err[index]
                if err is not None:
                    errors[kw] = err
        return errors

    @final
    def _get_subplots(self, fig: Figure) -> List[Axes]:
        if Version(mpl.__version__) < Version('3.8'):
            Klass = mpl.axes.Subplot
        else:
            Klass = mpl.axes.Axes
        return [ax for ax in fig.get_axes() if isinstance(ax, Klass) and ax.get_subplotspec() is not None]

    @final
    def _get_axes_layout(self, fig: Figure) -> Tuple[int, int]:
        axes = self._get_subplots(fig)
        x_set: set = set()
        y_set: set = set()
        for ax in axes:
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))
