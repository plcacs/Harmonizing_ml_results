```python
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
    Union,
    Optional,
    Callable,
)
import warnings

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray, npt

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
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
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
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.container import BarContainer
    from matplotlib.quiver import Barbs

    from pandas._typing import (
        IndexLabel,
        NDFrameT,
        PlottingOrientation,
        npt,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


def holds_integer(column: Index) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}


def _color_in_style(style: str) -> bool:
    return not set(mpl.colors.BASE_COLORS).isdisjoint(style)


class MPLPlot(ABC):
    _layout_type: str = "vertical"
    _default_rot: int = 0
    _kind: str
    orientation: Optional[str] = None
    data: DataFrame
    errors: dict[str, Any]
    secondary_y: Union[bool, tuple, list, np.ndarray, ABCIndex]
    colormap: Optional[Union[str, Colormap]]
    color: Optional[Union[str, list[str], dict[Hashable, str]]]
    kwds: dict[str, Any]

    @property
    @abstractmethod
    def _kind(self) -> str:
        raise NotImplementedError

    def __init__(
        self,
        data: Union[DataFrame, Series],
        kind: Optional[str] = None,
        by: Optional[IndexLabel] = None,
        subplots: Union[bool, Sequence[Sequence[str]]] = False,
        sharex: Optional[bool] = None,
        sharey: bool = False,
        use_index: bool = True,
        figsize: Optional[tuple[float, float]] = None,
        grid: Optional[bool] = None,
        legend: Union[bool, str] = True,
        rot: Optional[int] = None,
        ax: Optional[Union[Axes, Sequence[Axes]]] = None,
        fig: Optional[Figure] = None,
        title: Optional[Union[str, list[str]]] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        xticks: Optional[Sequence[float]] = None,
        yticks: Optional[Sequence[float]] = None,
        xlabel: Optional[Hashable] = None,
        ylabel: Optional[Hashable] = None,
        fontsize: Optional[int] = None,
        secondary_y: Union[bool, tuple, list, np.ndarray, ABCIndex] = False,
        colormap: Optional[Union[str, Colormap]] = None,
        table: bool = False,
        layout: Optional[tuple[int, int]] = None,
        include_bool: bool = False,
        column: Optional[IndexLabel] = None,
        *,
        logx: Union[bool, None, Literal["sym"]] = False,
        logy: Union[bool, None, Literal["sym"]] = False,
        loglog: Union[bool, None, Literal["sym"]] = False,
        mark_right: bool = True,
        stacked: bool = False,
        label: Optional[Hashable] = None,
        style: Optional[Union[list[str], dict[Hashable, str], str]] = None,
        **kwds: Any,
    ) -> None:
        self.by = com.maybe_make_list(by)
        if isinstance(data, ABCDataFrame):
            if column:
                self.columns = com.maybe_make_list(column)
            elif self.by is None:
                self.columns = [col for col in data.columns if is_numeric_dtype(data[col])]
            else:
                self.columns = [
                    col for col in data.columns
                    if col not in self.by and is_numeric_dtype(data[col])
                ]
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
        self.rot = self._default_rot if rot is None else rot
        self._rot_set = rot is not None
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
        xerr, data = type(self)._parse_errorbars("xerr", kwds.pop("xerr", None), data, self._get_nseries(data))
        yerr, data = type(self)._parse_errorbars("yerr", kwds.pop("yerr", None), data, self._get_nseries(data))
        self.errors = {"xerr": xerr, "yerr": yerr}
        self.data = data
        self.secondary_y = secondary_y
        self.colormap = colormap
        self.table = table
        self.include_bool = include_bool
        self.kwds = kwds
        color = kwds.pop("color", lib.no_default)
        self.color = self._validate_color_args(color, self.colormap)
        self.data = self._ensure_frame(self.data)

    @final
    @staticmethod
    def _validate_sharex(sharex: Optional[bool], ax: Optional[Axes], by: Optional[IndexLabel]) -> bool:
        return bool(sharex if sharex is not None else (ax is None and by is None))

    @classmethod
    def _validate_log_kwd(
        cls,
        kwd: str,
        value: Union[bool, None, Literal["sym"]],
    ) -> Union[bool, None, Literal["sym"]]:
        return value

    @final
    @staticmethod
    def _validate_subplots_kwarg(
        subplots: Union[bool, Sequence[Sequence[str]]], data: Union[Series, DataFrame], kind: str
    ) -> Union[bool, list[tuple[int, ...]]]:
        return subplots if isinstance(subplots, bool) else []

    def _validate_color_args(self, color: Any, colormap: Optional[Union[str, Colormap]]) -> Optional[Union[str, list[str], dict[Hashable, str]]]:
        return None

    @final
    def _iter_data(self, data: Union[DataFrame, dict[Hashable, Union[Series, DataFrame]]]) -> Iterator[tuple[Hashable, NDArray]]:
        for col, values in data.items():
            yield col, np.asarray(values.values)

    def _get_nseries(self, data: Union[Series, DataFrame]) -> int:
        return 1

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
        return len(ax.lines) + len(ax.artists) + len(ax.containers) > 0

    @final
    def _maybe_right_yaxis(self, ax: Axes, axes_num: int) -> Axes:
        return ax

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
    def _axes_and_fig(self) -> tuple[Sequence[Axes], Figure]:
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
            axes = [fig.add_subplot(111)]
        else:
            fig = self.ax.get_figure()
            axes = [self.ax]
        return cast(Sequence[Axes], axes), fig

    @final
    @staticmethod
    def _convert_to_ndarray(data: Series) -> NDArray:
        return data.to_numpy()

    @final
    def _ensure_frame(self, data: Union[DataFrame, Series]) -> DataFrame:
        return data.to_frame()

    def _compute_plot_data(self) -> None:
        self.data = self.data.infer_objects()

    def _make_plot(self, fig: Figure) -> None:
        pass

    @final
    def _add_table(self) -> None:
        pass

    @final
    def _post_plot_logic_common(self, ax: Axes) -> None:
        pass

    @abstractmethod
    def _post_plot_logic(self, ax: Axes, data: Union[DataFrame, Series]) -> None:
        pass

    @final
    def _adorn_subplots(self, fig: Figure) -> None:
        pass

    @final
    @staticmethod
    def _apply_axis_properties(axis: Axis, rot: Optional[int] = None, fontsize: Optional[int] = None) -> None:
        pass

    @final
    @property
    def legend_title(self) -> Optional[str]:
        return None

    @final
    def _mark_right_label(self, label: str, index: int) -> str:
        return label

    @final
    def _append_legend_handles_labels(self, handle: Artist, label: str) -> None:
        pass

    def _make_legend(self) -> None:
        pass

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes) -> tuple[Axes, Optional[Artist]]:
        return ax, None

    @final
    def _get_xticks(self) -> Union[list[int], NDArray]:
        return []

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls,
        ax: Axes,
        x: Union[Index, NDArray],
        y: NDArray,
        style: Optional[str] = None,
        is_errorbar: bool = False,
        **kwds: Any,
    ) -> list[Artist]:
        return []

    @final
    def _get_index_name(self) -> Optional[str]:
        return None

    @final
    @classmethod
    def _get_ax_layer(cls, ax: Axes, primary: bool = True) -> Axes:
        return ax

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        return 0

    @final
    def _get_ax(self, i: int) -> Axes:
        return self.axes[0]

    @final
    def on_right(self, i: int) -> bool:
        return False

    @final
    def _apply_style_colors(
        self,
        colors: Union[str, list[str], 
        kwds: dict[str, Any],
        col_num: int,
        label: str,
    ) -> tuple[Optional[str], dict[str, Any]]:
        return None, kwds

    def _get_colors(
        self,
        num_colors: Optional[int] = None,
        color_kwds: str = "color",
    ) -> Union[str, list[str], dict[Hashable, str]]:
        return []

    @final
    @staticmethod
    def _parse_errorbars(
        label: str,
        err: Any,
        data: NDFrameT,
        nseries: int,
    ) -> tuple[Any, NDFrameT]:
        return None, data

    @final
    def _get_errorbars(
        self, label: Optional[str] = None, index: Optional[int] = None, xerr: bool = True, yerr: bool = True
    ) -> dict[str, Any]:
        return {}

    @final
    def _get_subplots(self, fig: Figure) -> list[Axes]:
        return []

    @final
    def _get_axes_layout(self, fig: Figure) -> tuple[int, int]:
        return (0, 0)


class PlanePlot(MPLPlot, ABC):
    _layout_type: str = "single"

    def __init__(
        self,
        data: DataFrame,
        x: Hashable,
        y: Hashable,
        **kwargs: Any,
    ) -> None:
        super().__init__(data, **kwargs)
        self.x = x
        self.y = y

    @final
    def _post_plot_logic(self, ax: Axes, data: DataFrame) -> None:
        pass

    @final
    def _plot_colorbar(self, ax: Axes, *, fig: Figure, **kwds: Any) -> Optional[mpl.colorbar.Colorbar]:
        return None


class ScatterPlot(PlanePlot):
    _kind: Literal["scatter"] = "scatter"

    def __init__(
        self,
        data: DataFrame,
        x: Hashable,
        y: Hashable,
        s: Optional[Union[int, float, str]] = None,
        c: Optional[Union[Hashable, NDArray]] = None,
        *,
        colorbar: Union[bool, lib.NoDefault] = lib.no_default,
        norm: Optional[Normalize] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data, x, y, **kwargs)
        self.s = s if s is not None else 20
        self.colorbar = colorbar
        self.norm = norm
        self.c = c

    def _make_plot(self, fig: Figure) -> None:
        pass


class HexBinPlot(PlanePlot):
    _kind: Literal["hexbin"] = "hexbin"

    def __init__(
        self,
        data: DataFrame,
        x: Hashable,
        y: Hashable,
        C: Optional[Hashable] = None,
        *,
        colorbar: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(data, x, y, **kwargs)
        self.C = C
        self.colorbar = colorbar

    def _make_plot(self, fig: Figure) -> None:
        pass


class LinePlot(MPLPlot):
    _default_rot: int = 0
    orientation: PlottingOrientation = "vertical"
    _kind: Literal["line"] = "line"

    def __init__(self, data: Union[DataFrame, Series], **kwargs: Any) -> None:
        super().__init__(data, **kwargs)

    @classmethod
    def