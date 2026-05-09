from distutils.version import LooseVersion
import matplotlib as mat
import numpy as np
import pandas as pd
from matplotlib.axes._base import _process_plot_format
from pandas.core.dtypes.inference import is_list_like
from pandas.io.formats.printing import pprint_thing
from databricks.koalas.plot import TopNPlotBase, SampledPlotBase, HistogramPlotBase, BoxPlotBase, unsupported_function, KdePlotBase

if LooseVersion(pd.__version__) < LooseVersion('0.25'):
    from pandas.plotting._core import _all_kinds, BarPlot as PandasBarPlot, BoxPlot as PandasBoxPlot, HistPlot as PandasHistPlot, MPLPlot as PandasMPLPlot, PiePlot as PandasPiePlot, AreaPlot as PandasAreaPlot, LinePlot as PandasLinePlot, BarhPlot as PandasBarhPlot, ScatterPlot as PandasScatterPlot, KdePlot as PandasKdePlot
else:
    from pandas.plotting._matplotlib import BarPlot as PandasBarPlot, BoxPlot as PandasBoxPlot, HistPlot as PandasHistPlot, PiePlot as PandasPiePlot, AreaPlot as PandasAreaPlot, LinePlot as PandasLinePlot, BarhPlot as PandasBarhPlot, ScatterPlot as PandasScatterPlot, KdePlot as PandasKdePlot
    from pandas.plotting._core import PlotAccessor
    from pandas.plotting._matplotlib.core import MPLPlot as PandasMPLPlot
    _all_kinds = PlotAccessor._all_kinds


class KoalasBarPlot(PandasBarPlot, TopNPlotBase):

    def __init__(self, data: pd.Series, **kwargs: dict) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _plot(self, ax: mat.axes.Axes, x: list, y: list, w: list, start: int = 0, log: bool = False, **kwds: dict) -> mat.axes.Axes:
        self.set_result_text(ax)
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)


class KoalasBoxPlot(PandasBoxPlot, BoxPlotBase):

    def boxplot(self, ax: mat.axes.Axes, bxpstats: list, notch: bool = None, sym: str = None, vert: bool = None, whis: float = None, positions: list = None, widths: list = None, patch_artist: bool = None, bootstrap: int = None, usermedians: list = None, conf_intervals: list = None, meanline: bool = None, showmeans: bool = None, showcaps: bool = None, showbox: bool = None, showfliers: bool = None, boxprops: dict = None, labels: list = None, flierprops: dict = None, medianprops: dict = None, meanprops: dict = None, capprops: dict = None, whiskerprops: dict = None, manage_ticks: bool = None, manage_xticks: bool = None, autorange: bool = False, zorder: int = None, precision: float = None) -> list:
        ...

    def _plot(self, ax: mat.axes.Axes, bxpstats: list, column_num: int = None, return_type: str = 'axes', **kwds: dict) -> tuple:
        ...

    def _compute_plot_data(self) -> None:
        ...

    def _make_plot(self) -> None:
        ...

    @staticmethod
    def rc_defaults(notch: bool = None, vert: bool = None, whis: float = None, patch_artist: bool = None, bootstrap: int = None, meanline: bool = None, showmeans: bool = None, showcaps: bool = None, showbox: bool = None, showfliers: bool = None, **kwargs: dict) -> dict:
        ...


class KoalasHistPlot(PandasHistPlot, HistogramPlotBase):

    def _args_adjust(self) -> None:
        ...

    def _compute_plot_data(self) -> None:
        ...

    def _make_plot(self) -> None:
        ...

    @classmethod
    def _plot(cls, ax: mat.axes.Axes, y: list, style: str = None, bins: list = None, bottom: int = 0, column_num: int = 0, stacking_id: int = None, **kwds: dict) -> list:
        ...


class KoalasPiePlot(PandasPiePlot, TopNPlotBase):

    def __init__(self, data: pd.Series, **kwargs: dict) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self) -> None:
        ...


class KoalasAreaPlot(PandasAreaPlot, SampledPlotBase):

    def __init__(self, data: pd.Series, **kwargs: dict) -> None:
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self) -> None:
        ...


class KoalasLinePlot(PandasLinePlot, SampledPlotBase):

    def __init__(self, data: pd.Series, **kwargs: dict) -> None:
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self) -> None:
        ...


class KoalasBarhPlot(PandasBarhPlot, TopNPlotBase):

    def __init__(self, data: pd.Series, **kwargs: dict) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self) -> None:
        ...


class KoalasScatterPlot(PandasScatterPlot, TopNPlotBase):

    def __init__(self, data: pd.Series, x: str, y: str, **kwargs: dict) -> None:
        super().__init__(self.get_top_n(data), x, y, **kwargs)

    def _make_plot(self) -> None:
        ...


class KoalasKdePlot(PandasKdePlot, KdePlotBase):

    def _compute_plot_data(self) -> None:
        ...

    def _make_plot(self) -> None:
        ...

    def _get_ind(self, y: pd.Series) -> list:
        ...

    @classmethod
    def _plot(cls, ax: mat.axes.Axes, y: list, style: str = None, bw_method: str = None, ind: list = None, column_num: int = None, stacking_id: int = None, **kwds: dict) -> list:
        ...


_klasses = [KoalasHistPlot, KoalasBarPlot, KoalasBoxPlot, KoalasPiePlot, KoalasAreaPlot, KoalasLinePlot, KoalasBarhPlot, KoalasScatterPlot, KoalasKdePlot]
_plot_klass = {getattr(klass, '_kind'): klass for klass in _klasses}
_common_kinds = {'area', 'bar', 'barh', 'box', 'hist', 'kde', 'line', 'pie'}
_series_kinds = _common_kinds.union(set())
_dataframe_kinds = _common_kinds.union({'scatter', 'hexbin'})
_koalas_all_kinds = _common_kinds.union(_series_kinds).union(_dataframe_kinds)


def plot_koalas(data: pd.DataFrame, kind: str, **kwargs: dict) -> mat.axes.Axes:
    ...


def plot_series(data: pd.Series, kind: str = 'line', ax: mat.axes.Axes = None, figsize: tuple = None, use_index: bool = True, title: str = None, grid: bool = None, legend: bool = False, style: list = None, logx: bool = False, logy: bool = False, loglog: bool = False, xticks: list = None, yticks: list = None, xlim: list = None, ylim: list = None, rot: int = None, fontsize: int = None, colormap: str = None, table: bool = False, yerr: pd.DataFrame = None, xerr: pd.DataFrame = None, label: str = None, secondary_y: bool = False, **kwds: dict) -> mat.axes.Axes:
    ...


def plot_frame(data: pd.DataFrame, x: str = None, y: str = None, kind: str = 'line', ax: mat.axes.Axes = None, subplots: bool = False, sharex: bool = None, sharey: bool = False, layout: tuple = None, figsize: tuple = None, use_index: bool = True, title: str = None, grid: bool = None, legend: bool = True, style: list = None, logx: bool = False, logy: bool = False, loglog: bool = False, xticks: list = None, yticks: list = None, xlim: list = None, ylim: list = None, rot: int = None, fontsize: int = None, colormap: str = None, table: bool = False, yerr: pd.DataFrame = None, xerr: pd.DataFrame = None, secondary_y: bool = False, sort_columns: bool = False, **kwds: dict) -> mat.axes.Axes:
    ...


def _plot(data: pd.DataFrame, x: str = None, y: str = None, subplots: bool = False, ax: mat.axes.Axes = None, kind: str = 'line', **kwds: dict) -> mat.axes.Axes:
    ...
