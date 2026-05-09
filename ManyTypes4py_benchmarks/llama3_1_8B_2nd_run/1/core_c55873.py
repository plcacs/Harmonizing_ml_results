from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, final
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

def holds_integer(column: Sequence[Any]) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}

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
    _layout_type: str = "vertical"
    _default_rot: int = 0

    @property
    def orientation(self) -> PlottingOrientation:
        return None

    def __init__(
        self,
        data: NDFrameT,
        kind: str | None = None,
        by: Iterable[str | IndexLabel] | None = None,
        subplots: bool | Iterable[str] | None = False,
        sharex: bool | None = None,
        sharey: bool = False,
        use_index: bool = True,
        figsize: tuple[float, float] | None = None,
        grid: bool | None = None,
        legend: bool | str = True,
        rot: int | None = None,
        ax: Axes | None = None,
        fig: Figure | None = None,
        title: str | Iterable[str] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        xticks: Iterable[Any] | None = None,
        yticks: Iterable[Any] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        fontsize: int | None = None,
        secondary_y: bool | Iterable[Any] | None = False,
        colormap: str | None = None,
        table: bool | str | None = False,
        layout: tuple[int, int] | None = None,
        include_bool: bool = False,
        column: Iterable[str | IndexLabel] | None = None,
        *,
        logx: bool = False,
        logy: bool = False,
        loglog: bool = False,
        mark_right: bool = True,
        stacked: bool = False,
        label: str | None = None,
        style: str | dict[str, str] | None = None,
        **kwds: Any,
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
        if self.by is not None and self._kind == "hist":
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
            grid = False if secondary_y else mpl.rcParams["axes.grid"]
        self.grid = grid
        self.legend = legend
        self.legend_handles = []
        self.legend_labels = []
        self.logx = type(self)._validate_log_kwd("logx", logx)
        self.logy = type(self)._validate_log_kwd("logy", logy)
        self.loglog = type(self)._validate_log_kwd("loglog", loglog)
        self.label = label
        self.style = style
        self.mark_right = mark_right
        self.stacked = stacked
        self.ax = ax
        xerr = kwds.pop("xerr", None)
        yerr = kwds.pop("yerr", None)
        nseries = self._get_nseries(data)
        xerr, data = type(self)._parse_errorbars("xerr", xerr, data, nseries)
        yerr, data = type(self)._parse_errorbars("yerr", yerr, data, nseries)
        self.errors = {"xerr": xerr, "yerr": yerr}
        self.data = data
        if not isinstance(secondary_y, (bool, tuple, list, np.ndarray, ABCIndex)):
            secondary_y = [secondary_y]
        self.secondary_y = secondary_y
        if "cmap" in kwds and colormap:
            raise TypeError("Only specify one of `cmap` and `colormap`.")
        if "cmap" in kwds:
            self.colormap = kwds.pop("cmap")
        else:
            self.colormap = colormap
        self.table = table
        self.include_bool = include_bool
        self.kwds = kwds
        color = kwds.pop("color", lib.no_default)
        self.color = self._validate_color_args(color, self.colormap)
        assert "color" not in self.kwds
        self.data = self._ensure_frame(self.data)

    @final
    @staticmethod
    def _validate_sharex(sharex: bool | None, ax: Axes | None, by: Iterable[str | IndexLabel] | None) -> bool:
        if sharex is None:
            if ax is None and by is None:
                sharex = True
            else:
                sharex = False
        elif not is_bool(sharex):
            raise TypeError("sharex must be a bool or None")
        return bool(sharex)

    @classmethod
    def _validate_log_kwd(
        cls, kwd: str, value: bool | str | None
    ) -> bool | str | None:
        if value is None or isinstance(value, bool) or (isinstance(value, str) and value == "sym"):
            return value
        raise ValueError(f"keyword '{kwd}' should be bool, None, or 'sym', not '{value}'")

    @final
    @staticmethod
    def _validate_subplots_kwarg(
        subplots: bool | Iterable[str], data: NDFrameT, kind: str
    ) -> bool | Iterable[tuple[int, ...]]:
        """
        Validate the subplots parameter

        - check type and content
        - check for duplicate columns
        - check for invalid column names
        - convert column names into indices
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
            raise ValueError("subplots should be a bool or an iterable")
        supported_kinds = ("line", "bar", "barh", "hist", "kde", "density", "area", "pie")
        if kind not in supported_kinds:
            raise ValueError(
                f'When subplots is an iterable, kind must be one of {", ".join(supported_kinds)}. Got {kind}.'
            )
        if isinstance(data, ABCSeries):
            raise NotImplementedError("An iterable subplots for a Series is not supported.")
        columns = data.columns
        if isinstance(columns, ABCMultiIndex):
            raise NotImplementedError("An iterable subplots for a DataFrame with a MultiIndex column is not supported.")
        if columns.nunique() != len(columns):
            raise NotImplementedError(
                "An iterable subplots for a DataFrame with non-unique column labels is not supported."
            )
        out: list[tuple[int, ...]] = []
        seen_columns: set[str] = set()
        for group in subplots:
            if not is_list_like(group):
                raise ValueError("When subplots is an iterable, each entry should be a list/tuple of column names.")
            idx_locs = columns.get_indexer_for(group)
            if (idx_locs == -1).any():
                bad_labels = np.extract(idx_locs == -1, group)
                raise ValueError(f'Column label(s) {list(bad_labels)} not found in the DataFrame.')
            unique_columns = set(group)
            duplicates = seen_columns.intersection(unique_columns)
            if duplicates:
                raise ValueError(
                    f'Each column should be in only one subplot. Columns {duplicates} were found in multiple subplots.'
                )
            seen_columns = seen_columns.union(unique_columns)
            out.append(tuple(idx_locs))
        unseen_columns = columns.difference(seen_columns)
        for column in unseen_columns:
            idx_loc = columns.get_loc(column)
            out.append((idx_loc,))
        return out

    def _validate_color_args(
        self, color: str | None, colormap: str | None
    ) -> str | None:
        if color is lib.no_default:
            if "colors" in self.kwds and colormap is not None:
                warnings.warn(
                    "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                    stacklevel=find_stack_level(),
                )
            return None
        if self.nseries == 1 and color is not None and (not is_list_like(color)):
            color = [color]
        if isinstance(color, tuple) and self.nseries == 1 and (len(color) in (3, 4)):
            color = [color]
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )
        if self.style is not None:
            if isinstance(self.style, dict):
                styles = [self.style[col] for col in self.columns if col in self.style]
            elif is_list_like(self.style):
                styles = self.style
            else:
                styles = [self.style]
            for s in styles:
                if _color_in_style(s):
                    raise ValueError(
                        "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
                    )
        return color

    @final
    @staticmethod
    def _iter_data(data: NDFrameT) -> Iterator[tuple[str, np.ndarray]]:
        for col, values in data.items():
            yield (col, np.asarray(values.values))

    def _get_nseries(self, data: NDFrameT) -> int:
        if data.ndim == 1:
            return 1
        elif self.by is not None and self._kind == "hist":
            return len(self._grouped)
        elif self.by is not None and self._kind == "box":
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
    def _has_plotted_object(ax: Axes) -> bool:
        """check whether ax has data"""
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    @final
    def _maybe_right_yaxis(
        self, ax: Axes, axes_num: int
    ) -> Axes:
        if not self.on_right(axes_num):
            return self._get_ax_layer(ax)
        if hasattr(ax, "right_ax"):
            return ax.right_ax
        elif hasattr(ax, "left_ax"):
            return ax
        else:
            orig_ax, new_ax = (ax, ax.twinx())
            new_ax._get_lines = orig_ax._get_lines
            new_ax._get_patches_for_fill = orig_ax._get_patches_for_fill
            orig_ax.right_ax, new_ax.left_ax = (new_ax, orig_ax)
            if not self._has_plotted_object(orig_ax):
                orig_ax.get_yaxis().set_visible(False)
            if self.logy is True or self.loglog is True:
                new_ax.set_yscale("log")
            elif self.logy == "sym" or self.loglog == "sym":
                new_ax.set_yscale("symlog")
            return new_ax

    @final
    @cache_readonly
    def fig(self) -> Figure:
        return self._axes_and_fig[1]

    @final
    @cache_readonly
    def axes(self) -> tuple[Axes, ...]:
        return self._axes_and_fig[0]

    @final
    @cache_readonly
    def _axes_and_fig(self) -> tuple[tuple[Axes, ...], Figure]:
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
            [a.set_xscale("log") for a in axes]
        elif self.logx == "sym" or self.loglog == "sym":
            [a.set_xscale("symlog") for a in axes]
        if self.logy is True or self.loglog is True:
            [a.set_yscale("log") for a in axes]
        elif self.logy == "sym" or self.loglog == "sym":
            [a.set_yscale("symlog") for a in axes]
        axes_seq: Sequence[Axes] = cast(Sequence[Axes], axes)
        return (axes_seq, fig)

    @property
    def result(self) -> Axes | tuple[Axes, ...]:
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
            all_sec = is_list_like(self.secondary_y) and len(self.secondary_y) == self.nseries
            if sec_true or all_sec:
                return self._get_ax_layer(self.axes[0], primary=False)
            else:
                return self.axes[0]

    @final
    @staticmethod
    def _convert_to_ndarray(data: NDFrameT) -> np.ndarray:
        if isinstance(data.dtype, CategoricalDtype):
            return data
        if (
            is_integer_dtype(data.dtype)
            or is_float_dtype(data.dtype)
        ) and isinstance(data.dtype, ExtensionDtype):
            return data.to_numpy(dtype="float", na_value=np.nan)
        if len(data) > 0:
            return np.asarray(data)
        return data

    @final
    def _ensure_frame(self, data: NDFrameT) -> NDFrameT:
        if isinstance(data, ABCSeries):
            label = self.label
            if label is None and data.name is None:
                label = ""
            if label is None:
                data = data.to_frame()
            else:
                data = data.to_frame(name=label)
        elif self._kind in ("hist", "box"):
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
        include_type = [np.number, "datetime", "datetimetz", "timedelta"]
        if self.include_bool is True:
            include_type.append(np.bool_)
        exclude_type = None
        if self._kind == "box":
            include_type = [np.number]
            exclude_type = ["timedelta"]
        if self._kind == "scatter":
            include_type.extend(["object", "category", "string"])
        numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)
        is_empty = numeric_data.shape[-1] == 0
        if is_empty:
            raise TypeError("no numeric data to plot")
        self.data = numeric_data.apply(type(self)._convert_to_ndarray)

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
        if self.orientation == "vertical" or self.orientation is None:
            type(self)._apply_axis_properties(ax.xaxis, rot=self.rot, fontsize=self.fontsize)
            type(self)._apply_axis_properties(ax.yaxis, fontsize=self.fontsize)
            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(ax.right_ax.yaxis, fontsize=self.fontsize)
        elif self.orientation == "horizontal":
            type(self)._apply_axis_properties(ax.yaxis, rot=self.rot, fontsize=self.fontsize)
            type(self)._apply_axis_properties(ax.xaxis, fontsize=self.fontsize)
            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(ax.right_ax.yaxis, fontsize=self.fontsize)
        else:
            raise ValueError

    @abstractmethod
    def _post_plot_logic(self, ax: Axes, data: NDFrameT) -> None:
        """Post process for each axes. Overridden in child classes"""

    @final
    def _adorn_subplots(self, fig: Figure) -> None:
        """Common post process unrelated to data"""
        if len(self.axes) > 0:
            all_axes = self._get_subplots(fig)
            nrows, ncols = self._get_axes_layout(fig)
            handle_shared_axes(axarr=all_axes, nplots=len(all_axes), naxes=nrows * ncols, nrows=nrows, ncols=ncols, sharex=self.sharex, sharey=self.sharey)
        for ax in self.axes:
            ax = getattr(ax, "right_ax", ax)
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
                            f'The length of `title` must equal the number of columns if using `title` of type `list` and `subplots=True`.\n'
                            f"length of title = {len(self.title)}\n"
                            f"number of columns = {self.nseries}"
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
    def _apply_axis_properties(
        axis: Axis, rot: int | None = None, fontsize: int | None = None
    ) -> None:
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
    def legend_title(self) -> str | None:
        if not isinstance(self.data.columns, ABCMultiIndex):
            name = self.data.columns.name
            if name is not None:
                name = pprint_thing(name)
            return name
        else:
            stringified = map(pprint_thing, self.data.columns.names)
            return ",".join(stringified)

    @final
    def _mark_right_label(self, label: str, index: int) -> str:
        """
        Append ``(right)`` to the label of a line if it's plotted on the right axis.

        Note that ``(right)`` is only appended when ``subplots=False``.
        """
        if not self.subplots and self.mark_right and self.on_right(index):
            label += " (right)"
        return label

    @final
    def _append_legend_handles_labels(self, handle: Artist, label: str) -> None:
        """
        Append current handle and label to ``legend_handles`` and ``legend_labels``.

        These will be used to make the legend.
        """
        self.legend_handles.append(handle)
        self.legend_labels.append(label)

    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])
        handles = []
        labels = []
        title = ""
        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                if Version(mpl.__version__) < Version("3.7"):
                    handles = leg.legendHandles
                else:
                    handles = leg.legend_handles
                labels = [x.get_text() for x in leg.get_texts()]
            if self.legend:
                if self.legend == "reverse":
                    handles += reversed(self.legend_handles)
                    labels += reversed(self.legend_labels)
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels
                if self.legend_title is not None:
                    title = self.legend_title
            if len(handles) > 0:
                ax.legend(handles, labels, loc="best", title=title)
        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            "No artists with labels found to put in legend.",
                            UserWarning,
                        )
                        ax.legend(loc="best")

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes) -> tuple[Axes, Artist | None]:
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()
        other_ax = getattr(ax, "left_ax", None) or getattr(ax, "right_ax", None)
        other_leg = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return (ax, leg)

    @final
    def _get_xticks(self) -> list[Any]:
        index = self.data.index
        is_datetype = index.inferred_type in ("datetime", "date", "datetime64", "time")
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                x = index.to_timestamp()._mpl_repr()
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
        x: np.ndarray,
        y: np.ndarray,
        style: str | None = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> tuple[Artist, np.ndarray]:
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)
        if isinstance(x, ABCIndex):
            x = x._mpl_repr()
        if is_errorbar:
            if "xerr" in kwds:
                kwds["xerr"] = np.array(kwds.get("xerr"))
            if "yerr" in kwds:
                kwds["yerr"] = np.array(kwds.get("yerr"))
            return ax.errorbar(x, y, **kwds)
        else:
            args = (x, y, style) if style is not None else (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self) -> str | None:
        """Specify whether xlabel/ylabel should be used to override index name"""
        return self.xlabel

    @final
    def _get_index_name(self) -> str | None:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ",".join([pprint_thing(x) for x in name])
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
            return getattr(ax, "left_ax", ax)
        else:
            return getattr(ax, "right_ax", ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
        if isinstance(self.subplots, list):
            return next(
                (group_idx for group_idx, group in enumerate(self.subplots) if col_idx in group),
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

    @final
    def _apply_style_colors(
        self,
        colors: str | dict[str, str] | None,
        kwds: dict[str, Any],
        col_num: int,
        label: str,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style = None
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
        has_color = "color" in kwds or self.colormap is not None
        nocolor_style = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[col_num % len(colors)]
        return (style, kwds)

    def _get_colors(
        self,
        num_colors: int | None = None,
        color_kwds: str = "color",
    ) -> str | list[str] | dict[str, str]:
        if num_colors is None:
            num_colors = self.nseries
        if color_kwds == "color":
            color = self.color
        else:
            color = self.kwds.get(color_kwds)
        return get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=color)

    @final
    @staticmethod
    def _parse_errorbars(
        label: str,
        err: str | np.ndarray | ABCSeries | dict[str, Any] | None,
        data: NDFrameT,
        nseries: int,
    ) -> tuple[np.ndarray | None, NDFrameT]:
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

        def match_labels(data: NDFrameT, e: ABCSeries | np.ndarray) -> ABCSeries | np.ndarray:
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
                    raise ValueError(
                        f'Asymmetrical error bars should be provided with the shape (2, {len(data)})'
                    )
            elif isinstance(data, ABCDataFrame) and err.ndim == 3:
                if err_shape[0] != nseries or err_shape[1] != 2 or err_shape[2] != len(data):
                    raise ValueError(
                        f'Asymmetrical error bars should be provided with the shape ({nseries}, 2, {len(data)})'
                    )
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
        label: str | None = None,
        index: int | None = None,
        xerr: bool = True,
        yerr: bool = True,
    ) -> dict[str, np.ndarray]:
        errors: dict[str, np.ndarray] = {}
        for kw, flag in zip(["xerr", "yerr"], [xerr, yerr]):
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
    def _get_subplots(self, fig: Figure) -> list[Axes]:
        if Version(mpl.__version__) < Version("3.8"):
            Klass = mpl.axes.Subplot
        else:
            Klass = mpl.axes.Axes
        return [ax for ax in fig.get_axes() if isinstance(ax, Klass) and ax.get_subplotspec() is not None]

    @final
    def _get_axes_layout(self, fig: Figure) -> tuple[int, int]:
        axes = self._get_subplots(fig)
        x_set = set()
        y_set = set()
        for ax in axes:
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))

class PlanePlot(MPLPlot, ABC):
    """
    Abstract class for plotting on plane, currently scatter and hexbin.
    """
    _layout_type: str = "single"

    def __init__(self, data: NDFrameT, x: str, y: str, **kwargs: Any) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(self._kind + " requires an x and y column")
        if is_integer(x) and (not holds_integer(self.data.columns)):
            x = self.data.columns[x]
        if is_integer(y) and (not holds_integer(self.data.columns)):
            y = self.data.columns[y]
        self.x = x
        self.y = y

    @final
    def _get_nseries(self, data: NDFrameT) -> int:
        return 1

    @final
    def _post_plot_logic(self, ax: Axes, data: NDFrameT) -> None:
        x, y = (self.x, self.y)
        xlabel = self.xlabel if self.xlabel is not None else pprint_thing(x)
        ylabel = self.ylabel if self.ylabel is not None else pprint_thing(y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @final
    def _plot_colorbar(self, ax: Axes, *, fig: Figure, **kwds: Any) -> Artist:
        img = ax.collections[-1]
        return fig.colorbar(img, ax=ax, **kwds)

class ScatterPlot(PlanePlot):

    @property
    def _kind(self) -> str:
        return "scatter"

    def __init__(
        self,
        data: NDFrameT,
        x: str,
        y: str,
        s: int | str | None = None,
        c: str | int | None = None,
        *,
        colorbar: bool | str | None = lib.no_default,
        norm: None | str | dict[str, Any] | Callable[[np.ndarray], np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        if s is None:
            s = 20
        elif is_hashable(s) and s in data.columns:
            s = data[s]
        self.s = s
        self.colorbar = colorbar
        self.norm = norm
        super().__init__(data, x, y, **kwargs)
        if is_integer(c) and (not holds_integer(self.data.columns)):
            c = self.data.columns[c]
        self.c = c

    def _make_plot(self, fig: Figure) -> None:
        x, y, c, data = (self.x, self.y, self.c, self.data)
        ax = self.axes[0]
        c_is_column = is_hashable(c) and c in self.data.columns
        color_by_categorical = c_is_column and isinstance(self.data[c].dtype, CategoricalDtype)
        color = self.color
        c_values = self._get_c_values(color, color_by_categorical, c_is_column)
        norm, cmap = self._get_norm_and_cmap(c_values, color_by_categorical)
        cb = self._get_colorbar(c_values, c_is_column)
        if self.legend:
            label = self.label
        else:
            label = None
        create_colors = not self._are_valid_colors(c_values)
        if create_colors:
            color_mapping = self._get_color_mapping(c_values)
            c_values = [color_mapping[s] for s in c_values]
            ax.legend(handles=[mpl.patches.Circle((0, 0), facecolor=c, label=s) for s, c in color_mapping.items()])
        scatter = ax.scatter(data[x].values, data[y].values, c=c_values, label=label, cmap=cmap, norm=norm, s=self.s, **self.kwds)
        if cb:
            cbar_label = c if c_is_column else ""
            cbar = self._plot_colorbar(ax, fig=fig, label=cbar_label)
            if color_by_categorical:
                n_cats = len(self.data[c].cat.categories)
                cbar.set_ticks(np.linspace(0.5, n_cats - 0.5, n_cats))
                cbar.ax.set_yticklabels(self.data[c].cat.categories)
        if label is not None:
            self._append_legend_handles_labels(scatter, label)
        errors_x = self._get_errorbars(label=x, index=0, yerr=False)
        errors_y = self._get_errorbars(label=y, index=0, xerr=False)
        if len(errors_x) > 0 or len(errors_y) > 0:
            err_kwds = dict(errors_x, **errors_y)
            err_kwds["ecolor"] = scatter.get_facecolor()[0]
            ax.errorbar(data[x].values, data[y].values, linestyle="none", **err_kwds)

    def _get_c_values(
        self,
        color: str | None,
        color_by_categorical: bool,
        c_is_column: bool,
    ) -> np.ndarray | str | list[str] | None:
        c = self.c
        if c is not None and color is not None:
            raise TypeError("Specify exactly one of `c` and `color`")
        if c is None and color is None:
            c_values = mpl.rcParams["patch.facecolor"]
        elif color is not None:
            c_values = color
        elif color_by_categorical:
            c_values = self.data[c].cat.codes
        elif c_is_column:
            c_values = self.data[c].values
        else:
            c_values = c
        return c_values

    def _are_valid_colors(self, c_values: np.ndarray | str | list[str] | None) -> bool:
        unique = np.unique(c_values)
        try:
            if len(c_values) and all((isinstance(c, str) for c in unique)):
                mpl.colors.to_rgba_array(unique)
            return True
        except (TypeError, ValueError) as _:
            return False

    def _get_color_mapping(self, c_values: np.ndarray | str | list[str] | None) -> dict[str, str]:
        unique = np.unique(c_values)
        n_colors = len(unique)
        cmap = mpl.colormaps.get_cmap(self.colormap)
        colors = cmap(np.linspace(0, 1, n_colors))
        return dict(zip(unique, colors))

    def _get_norm_and_cmap(
        self,
        c_values: np.ndarray | str | list[str] | None,
        color_by_categorical: bool,
    ) -> tuple[None | str | dict[str, Any] | Callable[[np.ndarray], np.ndarray], None | str | dict[str, Any]]:
        c = self.c
        if self.colormap is not None:
            cmap = mpl.colormaps.get_cmap(self.colormap)
        elif not isinstance(c_values, str) and is_integer_dtype(c_values):
            cmap = mpl.colormaps["Greys"]
        else:
            cmap = None
        if color_by_categorical and cmap is not None:
            n_cats = len(self.data[c].cat.categories)
            cmap = mpl.colors.ListedColormap([cmap(i) for i in range(cmap.N)])
            bounds = np.linspace(0, n_cats, n_cats + 1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = self.norm
        return (norm, cmap)

    def _get_colorbar(
        self,
        c_values: np.ndarray | str | list[str] | None,
        c_is_column: bool,
    ) -> bool:
        plot_colorbar = self.colormap or c_is_column
        cb = self.colorbar
        if cb is lib.no_default:
            return is_numeric_dtype(c_values) and plot_colorbar
        return cb

class HexBinPlot(PlanePlot):

    @property
    def _kind(self) -> str:
        return "hexbin"

    def __init__(
        self,
        data: NDFrameT,
        x: str,
        y: str,
        C: str | int | None = None,
        *,
        colorbar: bool | str | None = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(data, x, y, **kwargs)
        if is_integer(C) and (not holds_integer(self.data.columns)):
            C = self.data.columns[C]
        self.C = C
        self.colorbar = colorbar
        if len(self.data[self.x]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires x column to be numeric")
        if len(self.data[self.y]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires y column to be numeric")

    def _make_plot(self, fig: Figure) -> None:
        x, y, data, C = (self.x, self.y, self.data, self.C)
        ax = self.axes[0]
        cmap = self.colormap or "BuGn"
        cmap = mpl.colormaps.get_cmap(cmap)
        cb = self.colorbar
        if C is None:
            c_values = None
        else:
            c_values = data[C].values
        ax.hexbin(data[x].values, data[y].values, C=c_values, cmap=cmap, **self.kwds)
        if cb:
            self._plot_colorbar(ax, fig=fig)

    def _make_legend(self) -> None:
        pass

class LinePlot(MPLPlot):
    _default_rot: int = 0

    @property
    def orientation(self) -> PlottingOrientation:
        return "vertical"

    @property
    def _kind(self) -> str:
        return "line"

    def __init__(self, data: NDFrameT, **kwargs: Any) -> None:
        from pandas.plotting import plot_params

        MPLPlot.__init__(self, data, **kwargs)
        if self.stacked:
            self.data = self.data.fillna(value=0)
        self.x_compat = plot_params["x_compat"]
        if "x_compat" in self.kwds:
            self.x_compat = bool(self.kwds.pop("x_compat"))

    @final
    def _is_ts_plot(self) -> bool:
        return not self.x_compat and self.use_index and self._use_dynamic_x()

    @final
    def _use_dynamic_x(self) -> bool:
        return use_dynamic_x(self._get_ax(0), self.data)

    def _make_plot(self, fig: Figure) -> None:
        if self._is_ts_plot():
            data = maybe_convert_index(self._get_ax(0), self.data)
            x = data.index
            plotf = self._ts_plot
            it = data.items()
        else:
            x = self._get_xticks()
            plotf = self._plot
            it = self._iter_data(data=self.data)
        stacking_id = self._get_stacking_id()
        is_errorbar = com.any_not_none(*self.errors.values())
        colors = self._get_colors()
        for i, (label, y) in enumerate(it):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color
            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label
            newlines = plotf(ax, x, y, style=style, column_num=i, stacking_id=stacking_id, is_errorbar=is_errorbar, **kwds)
            self._append_legend_handles_labels(newlines[0], label)
            if self._is_ts_plot():
                lines = get_all_lines(ax)
                left, right = get_xlim(lines)
                ax.set_xlim(left, right)

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        style: str | None = None,
        column_num: int = 0,
        stacking_id: int | None = None,
        **kwds: Any,
    ) -> tuple[Artist, np.ndarray]:
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        lines = MPLPlot._plot(ax, x, y_values, style=style, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        return lines

    @final
    def _ts_plot(self, ax: Axes, x: np.ndarray, data: ABCSeries, style: str | None = None, **kwds: Any) -> tuple[Artist, np.ndarray]:
        freq, data = maybe_resample(data, ax, kwds)
        decorate_axes(ax, freq)
        if hasattr(ax, "left_ax"):
            decorate_axes(ax.left_ax, freq)
        if hasattr(ax, "right_ax"):
            decorate_axes(ax.right_ax, freq)
        ax._plot_data.append((data, self._kind, kwds))
        lines = self._plot(ax, data.index, np.asarray(data.values), style=style, **kwds)
        format_dateaxis(ax, ax.freq, data.index)
        return lines

    @final
    def _get_stacking_id(self) -> int | None:
        if self.stacked:
            return id(self.data)
        else:
            return None

    @final
    @classmethod
    def _initialize_stacker(cls, ax: Axes, stacking_id: int | None, n: int) -> None:
        if stacking_id is None:
            return
        if not hasattr(ax, "_stacker_pos_prior"):
            ax._stacker_pos_prior = {}
        if not hasattr(ax, "_stacker_neg_prior"):
            ax._stacker_neg_prior = {}
        ax._stacker_pos_prior[stacking_id] = np.zeros(n)
        ax._stacker_neg_prior[stacking_id] = np.zeros(n)

    @final
    @classmethod
    def _get_stacked_values(cls, ax: Axes, stacking_id: int | None, values: np.ndarray, label: str) -> np.ndarray:
        if stacking_id is None:
            return values
        if not hasattr(ax, "_stacker_pos_prior"):
            cls._initialize_stacker(ax, stacking_id, len(values))
        if (values >= 0).all():
            return ax._stacker_pos_prior[stacking_id] + values
        elif (values <= 0).all():
            return ax._stacker_neg_prior[stacking_id] + values
        raise ValueError(
            f"When stacked is True, each column must be either all positive or all negative. Column '{label}' contains both positive and negative values"
        )

    @final
    @classmethod
    def _update_stacker(cls, ax: Axes, stacking_id: int | None, values: np.ndarray) -> None:
        if stacking_id is None:
            return
        if (values >= 0).all():
            ax._stacker_pos_prior[stacking_id] += values
        elif (values <= 0).all():
            ax._stacker_neg_prior[stacking_id] += values

    def _post_plot_logic(self, ax: Axes, data: NDFrameT) -> None:
        def get_label(i: int) -> str:
            if is_float(i) and i.is_integer():
                i = int(i)
            try:
                return pprint_thing(data.index[i])
            except Exception:
                return ""

        if self._need_to_set_index:
            xticks = ax.get_xticks()
            xticklabels = [get_label(x) for x in xticks]
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks))
            ax.set_xticklabels(xticklabels)
        condition = not self._use_dynamic_x() and (data.index._is_all_dates and self.use_index) and (not self.subplots or (self.subplots and self.sharex))
        index_name = self._get_index_name()
        if condition:
            if not self._rot_set:
                self.rot = 30
            format_date_labels(ax, rot=self.rot)
        if index_name is not None and self.use_index:
            ax.set_xlabel(index_name)

class AreaPlot(LinePlot):

    @property
    def _kind(self) -> str:
        return "area"

    def __init__(self, data: NDFrameT, **kwargs: Any) -> None:
        kwargs.setdefault("stacked", True)
        data = data.fillna(value=0)
        LinePlot.__init__(self, data, **kwargs)
        if not self.stacked:
            self.kwds.setdefault("alpha", 0.5)
        if self.logy or self.loglog:
            raise ValueError("Log-y scales are not supported in area plot")

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        style: str | None = None,
        column_num: int = 0,
        stacking_id: int | None = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> tuple[Artist, np.ndarray]:
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)
        xdata, y_values = lines[0].get_data(orig=False)
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            start = ax._stacker_pos_prior[stacking_id]
        elif (y <= 0).all():
            start = ax._stacker_neg_prior[stacking_id]
        else:
            start = np.zeros(len(y))
        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()
        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        res = [rect]
        return res

    def _post_plot_logic(self, ax: Axes, data: NDFrameT) -> None:
        LinePlot._post_plot_logic(self, ax, data)
        is_shared_y = len(list(ax.get_shared_y_axes())) > 0
        if self.ylim is None and (not is_shared_y):
            if (data >= 0).all().all():
                ax.set_ylim(0, None)
            elif (data <= 0).all().all():
                ax.set_ylim(None, 0)

class BarPlot(MPLPlot):

    @property
    def _kind(self) -> str:
        return "bar"
    _default_rot: int = 90

    @property
    def orientation(self) -> PlottingOrientation:
        return "vertical"

    def __init__(
        self,
        data: NDFrameT,
        *,
        align: str = "center",
        bottom: np.ndarray | None = None,
        left: np.ndarray | None = None,
        width: float = 0.5,
        position: float = 0.5,
        log: bool = False,
        **kwargs: Any,
    ) -> None:
        self._is_series = isinstance(data, ABCSeries)
        self.bar_width = width
        self._align = align
        self._position = position
        self.tick_pos = np.arange(len(data))
        if is_list_like(bottom):
            bottom = np.array(bottom)
        if is_list_like(left):
            left = np.array(left)
        self.bottom = bottom
        self.left = left
        self.log = log
        MPLPlot.__init__(self, data, **kwargs)

    @cache_readonly
    def ax_pos(self) -> np.ndarray:
        return self.tick_pos - self.tickoffset

    @cache_readonly
    def tickoffset(self) -> float:
        if self.stacked or self.subplots:
            return self.bar_width * self._position
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return self.bar_width * (self._position - 0.5) + w * 0.5
        else:
            return self.bar_width * self._position

    @cache_readonly
    def lim_offset(self) -> float:
        if self.stacked or self.subplots:
            if self._align == "edge":
                return self.bar_width / 2
            else:
                return 0
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return w * 0.5
        else:
            return 0

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        w: float,
        start: float = 0,
        log: bool = False,
        **kwds: Any,
    ) -> Artist:
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors()
        ncolors = len(colors)
        pos_prior = neg_prior = np.zeros(len(self.data))
        K = self.nseries
        data = self.data.fillna(0)
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self._is_series:
                kwds["color"] = colors
            elif isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[i % ncolors]
            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            if ("yerr" in kwds or "xerr" in kwds) and kwds.get("ecolor") is None:
                kwds["ecolor"] = mpl.rcParams["xtick.color"]
            start = 0
            if self.log and (y >= 1).all():
                start = 1
            start = start + self._start_base
            kwds["align"] = self._align
            if self.subplots:
                w = self.bar_width / 2
                rect = self._plot(ax, self.ax_pos + w, y, self.bar_width, start=start, label=label, log=self.log, **kwds)
                ax.set_title(label)
            elif self.stacked:
                mask = y >= 0
                start = np.where(mask, pos_prior, neg_prior) + self._start_base
                w = self.bar_width / 2
                rect = self._plot(ax, self.ax_pos + w, y, self.bar_width, start=start, label=label, log=self.log, **kwds)
                pos_prior = pos_prior + np.where(mask, y, 0)
                neg_prior = neg_prior + np.where(mask, 0, y)
            else:
                w = self.bar_width / K
                rect = self._plot(ax, self.ax_pos + (i + 0.5) * w, y, w, start=start, label=label, log=self.log, **kwds)
            self._append_legend_handles_labels(rect, label)

    def _post_plot_logic(self, ax: Axes, data: NDFrameT) -> None:
        if self.use_index:
            str_index = [pprint_thing(key) for key in data.index]
        else:
            str_index = [pprint_thing(key) for key in range(data.shape[0])]
        s_edge = self.ax_pos[0] - 0.25 + self.lim_offset
        e_edge = self.ax_pos[-1] + 0.25 + self.bar_width + self.lim_offset
        self._decorate_ticks(ax, self._get_index_name(), str_index, s_edge, e_edge)

    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        ax.set_xlim((start_edge, end_edge))
        if self.xticks is not None:
            ax.set_xticks(np.array(self.xticks))
        else:
            ax.set_xticks(self.tick_pos)
            ax.set_xticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_xlabel(name)

class BarhPlot(BarPlot):

    @property
    def _kind(self) -> str:
        return "barh"
    _default_rot: int = 0

    @property
    def orientation(self) -> PlottingOrientation:
        return "horizontal"

    @property
    def _start_base(self) -> np.ndarray:
        return self.left

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        w: float,
        start: float = 0,
        log: bool = False,
        **kwds: Any,
    ) -> Artist:
        return ax.barh(x, y, w, left=start, log=log, **kwds)

    def _get_custom_index_name(self) -> str | None:
        return self.ylabel

    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        ax.set_ylim((start_edge, end_edge))
        ax.set_yticks(self.tick_pos)
        ax.set_yticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_ylabel(name)
        ax.set_xlabel(self.xlabel)

class PiePlot(MPLPlot):

    @property
    def _kind(self) -> str:
        return "pie"
    _layout_type: str = "horizontal"

    def __init__(self, data: NDFrameT, kind: str | None = None, **kwargs: Any) -> None:
        data = data.fillna(value=0)
        lt_zero = data < 0
        if isinstance(data, ABCDataFrame) and lt_zero.any().any():
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        elif isinstance(data, ABCSeries) and lt_zero.any():
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        MPLPlot.__init__(self, data, kind=kind, **kwargs)

    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: bool | str | None) -> bool | str | None:
        super()._validate_log_kwd(kwd=kwd, value=value)
        if value is not False:
            warnings.warn(f"PiePlot ignores the '{kwd}' keyword", UserWarning, stacklevel=find_stack_level())
        return False

    def _validate_color_args(self, color: str | None, colormap: str | None) -> None:
        return None

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors(num_colors=len(self.data), color_kwds="colors")
        self.kwds.setdefault("colors", colors)
        for i, (label, y) in enumerate(self._iter_data(data=self.data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()

            def blank_labeler(label: str, value: float) -> str:
                if value == 0:
                    return ""
                else:
                    return label

            idx = [pprint_thing(v) for v in self.data.index]
            labels = kwds.pop("labels", idx)
            if labels is not None:
                blabels = [blank_labeler(left, value) for left, value in zip(labels, y)]
            else:
                blabels = None
            results = ax.pie(y, labels=blabels, **kwds)
            if kwds.get("autopct", None) is not None:
                patches, texts, autotexts = results
            else:
                patches, texts = results
                autotexts = []
            if self.fontsize is not None:
                for t in texts + autotexts:
                    t.set_fontsize(self.fontsize)
            leglabels = labels if labels is not None else idx
            for _patch, _leglabel in zip(patches, leglabels):
                self._append_legend_handles_labels(_patch, _leglabel)

    def _post_plot_logic(self, ax: Axes, data: NDFrameT) -> None:
        pass
