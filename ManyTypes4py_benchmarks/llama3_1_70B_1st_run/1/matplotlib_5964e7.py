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

    def __init__(self, data: pd.Series, **kwargs) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _plot(self, ax: mat.axes.Axes, x: np.ndarray, y: np.ndarray, w: np.ndarray, start: int = 0, log: bool = False, **kwds) -> mat.axes.Axes:
        self.set_result_text(ax)
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)

class KoalasBoxPlot(PandasBoxPlot, BoxPlotBase):

    def boxplot(self, ax: mat.axes.Axes, bxpstats: list, notch: bool = None, sym: str = None, vert: bool = None, whis: float = None, positions: np.ndarray = None, widths: np.ndarray = None, patch_artist: bool = None, bootstrap: int = None, usermedians: np.ndarray = None, conf_intervals: np.ndarray = None, meanline: bool = None, showmeans: bool = None, showcaps: bool = None, showbox: bool = None, showfliers: bool = None, boxprops: dict = None, labels: list = None, flierprops: dict = None, medianprops: dict = None, meanprops: dict = None, capprops: dict = None, whiskerprops: dict = None, manage_ticks: bool = None, manage_xticks: bool = None, autorange: bool = False, zorder: int = None, precision: float = None):
        # ...

    def _plot(self, ax: mat.axes.Axes, bxpstats: list, column_num: int = None, return_type: str = 'axes', **kwds) -> tuple:
        bp = self.boxplot(ax, bxpstats, **kwds)
        if return_type == 'dict':
            return (bp, bp)
        elif return_type == 'both':
            return (self.BP(ax=ax, lines=bp), bp)
        else:
            return (ax, bp)

    def _compute_plot_data(self) -> None:
        colname = self.data.name
        spark_column_name = self.data._internal.spark_column_name_for(self.data._column_label)
        data = self.data
        self.kwds.update(KoalasBoxPlot.rc_defaults(**self.kwds))
        showfliers = self.kwds.get('showfliers', False)
        whis = self.kwds.get('whis', 1.5)
        labels = self.kwds.get('labels', [colname])
        precision = self.kwds.get('precision', 0.01)
        col_stats, col_fences = BoxPlotBase.compute_stats(data, spark_column_name, whis, precision)
        outliers = BoxPlotBase.outliers(data, spark_column_name, *col_fences)
        whiskers = BoxPlotBase.calc_whiskers(spark_column_name, outliers)
        if showfliers:
            fliers = BoxPlotBase.get_fliers(spark_column_name, outliers, whiskers[0])
        else:
            fliers = []
        stats = []
        item = {'mean': col_stats['mean'], 'med': col_stats['med'], 'q1': col_stats['q1'], 'q3': col_stats['q3'], 'whislo': whiskers[0], 'whishi': whiskers[1], 'fliers': fliers, 'label': labels[0]}
        stats.append(item)
        self.data = {labels[0]: stats}

    def _make_plot(self) -> None:
        bxpstats = list(self.data.values())[0]
        ax = self._get_ax(0)
        kwds = self.kwds.copy()
        for stats in bxpstats:
            if len(stats['fliers']) > 1000:
                stats['fliers'] = stats['fliers'][:1000]
                ax.text(1, 1, 'showing top 1,000 fliers only', size=6, ha='right', va='bottom', transform=ax.transAxes)
        ret, bp = self._plot(ax, bxpstats, column_num=0, return_type=self.return_type, **kwds)
        self.maybe_color_bp(bp)
        self._return_obj = ret
        labels = [l for l, _ in self.data.items()]
        labels = [pprint_thing(l) for l in labels]
        if not self.use_index:
            labels = [pprint_thing(key) for key in range(len(labels))]
        self._set_ticklabels(ax, labels)

    @staticmethod
    def rc_defaults(notch: bool = None, vert: bool = None, whis: float = None, patch_artist: bool = None, bootstrap: int = None, meanline: bool = None, showmeans: bool = None, showcaps: bool = None, showbox: bool = None, showfliers: bool = None, **kwargs) -> dict:
        if whis is None:
            whis = mat.rcParams['boxplot.whiskers']
        if bootstrap is None:
            bootstrap = mat.rcParams['boxplot.bootstrap']
        if notch is None:
            notch = mat.rcParams['boxplot.notch']
        if vert is None:
            vert = mat.rcParams['boxplot.vertical']
        if patch_artist is None:
            patch_artist = mat.rcParams['boxplot.patchartist']
        if meanline is None:
            meanline = mat.rcParams['boxplot.meanline']
        if showmeans is None:
            showmeans = mat.rcParams['boxplot.showmeans']
        if showcaps is None:
            showcaps = mat.rcParams['boxplot.showcaps']
        if showbox is None:
            showbox = mat.rcParams['boxplot.showbox']
        if showfliers is None:
            showfliers = mat.rcParams['boxplot.showfliers']
        return dict(whis=whis, bootstrap=bootstrap, notch=notch, vert=vert, patch_artist=patch_artist, meanline=meanline, showmeans=showmeans, showcaps=showcaps, showbox=showbox, showfliers=showfliers)

class KoalasHistPlot(PandasHistPlot, HistogramPlotBase):

    def _args_adjust(self) -> None:
        if is_list_like(self.bottom):
            self.bottom = np.array(self.bottom)

    def _compute_plot_data(self) -> None:
        self.data, self.bins = HistogramPlotBase.prepare_hist_data(self.data, self.bins)

    def _make_plot(self) -> None:
        colors = self._get_colors(num_colors=1)
        stacking_id = self._get_stacking_id()
        output_series = HistogramPlotBase.compute_hist(self.data, self.bins)
        for (i, label), y in zip(enumerate(self.data._internal.column_labels), output_series):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            label = pprint_thing(label if len(label) > 1 else label[0])
            kwds['label'] = label
            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds['style'] = style
            kwds = self._make_plot_keywords(kwds, y)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            self._add_legend_handle(artists[0], label, index=i)

    @classmethod
    def _plot(cls, ax: mat.axes.Axes, y: np.ndarray, style: str = None, bins: np.ndarray = None, bottom: int = 0, column_num: int = 0, stacking_id: int = None, **kwds) -> list:
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)
        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds['label'])
        n, bins, patches = ax.hist(bins[:-1], bins=bins, bottom=bottom, weights=y, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

class KoalasPiePlot(PandasPiePlot, TopNPlotBase):

    def __init__(self, data: pd.Series, **kwargs) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasAreaPlot(PandasAreaPlot, SampledPlotBase):

    def __init__(self, data: pd.Series, **kwargs) -> None:
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasLinePlot(PandasLinePlot, SampledPlotBase):

    def __init__(self, data: pd.Series, **kwargs) -> None:
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasBarhPlot(PandasBarhPlot, TopNPlotBase):

    def __init__(self, data: pd.Series, **kwargs) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasScatterPlot(PandasScatterPlot, TopNPlotBase):

    def __init__(self, data: pd.Series, x: pd.Series, y: pd.Series, **kwargs) -> None:
        super().__init__(self.get_top_n(data), x, y, **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasKdePlot(PandasKdePlot, KdePlotBase):

    def _compute_plot_data(self) -> None:
        self.data = KdePlotBase.prepare_kde_data(self.data)

    def _make_plot(self) -> None:
        colors = self._get_colors(num_colors=1)
        stacking_id = self._get_stacking_id()
        sdf = self.data._internal.spark_frame
        for i, label in enumerate(self.data._internal.column_labels):
            y = sdf.select(self.data._internal.spark_column_for(label))
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            label = pprint_thing(label if len(label) > 1 else label[0])
            kwds['label'] = label
            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds['style'] = style
            kwds = self._make_plot_keywords(kwds, y)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            self._add_legend_handle(artists[0], label, index=i)

    def _get_ind(self, y: pd.Series) -> np.ndarray:
        return KdePlotBase.get_ind(y, self.ind)

    @classmethod
    def _plot(cls, ax: mat.axes.Axes, y: pd.Series, style: str = None, bw_method: str = None, ind: np.ndarray = None, column_num: int = None, stacking_id: int = None, **kwds) -> list:
        y = KdePlotBase.compute_kde(y, bw_method=bw_method, ind=ind)
        lines = PandasMPLPlot._plot(ax, ind, y, style=style, **kwds)
        return lines

_klasses = [KoalasHistPlot, KoalasBarPlot, KoalasBoxPlot, KoalasPiePlot, KoalasAreaPlot, KoalasLinePlot, KoalasBarhPlot, KoalasScatterPlot, KoalasKdePlot]
_plot_klass = {getattr(klass, '_kind'): klass for klass in _klasses}
_common_kinds = {'area', 'bar', 'barh', 'box', 'hist', 'kde', 'line', 'pie'}
_series_kinds = _common_kinds.union(set())
_dataframe_kinds = _common_kinds.union({'scatter', 'hexbin'})
_koalas_all_kinds = _common_kinds.union(_series_kinds).union(_dataframe_kinds)

def plot_koalas(data: pd.Series | pd.DataFrame, kind: str, **kwargs) -> mat.axes.Axes | np.ndarray:
    if kind not in _koalas_all_kinds:
        raise ValueError('{} is not a valid plot kind'.format(kind))
    from databricks.koalas import DataFrame, Series
    if isinstance(data, Series):
        if kind not in _series_kinds:
            return unsupported_function(class_name='pd.Series', method_name=kind)()
        return plot_series(data=data, kind=kind, **kwargs)
    elif isinstance(data, DataFrame):
        if kind not in _dataframe_kinds:
            return unsupported_function(class_name='pd.DataFrame', method_name=kind)()
        return plot_frame(data=data, kind=kind, **kwargs)

def plot_series(data: pd.Series, kind: str = 'line', ax: mat.axes.Axes = None, figsize: tuple = None, use_index: bool = True, title: str | list = None, grid: bool = None, legend: bool | str = False, style: list | dict = None, logx: bool = False, logy: bool = False, loglog: bool = False, xticks: list = None, yticks: list = None, xlim: list = None, ylim: list = None, rot: int = None, fontsize: int = None, colormap: str | mat.colors.Colormap = None, table: bool | pd.Series | pd.DataFrame = False, yerr: pd.DataFrame | pd.Series | np.ndarray | dict | str = None, xerr: pd.DataFrame | pd.Series | np.ndarray | dict | str = None, label: str = None, secondary_y: bool | list = False, **kwds) -> mat.axes.Axes | np.ndarray:
    import matplotlib.pyplot as plt
    if ax is None and len(plt.get_fignums()) > 0:
        with plt.rc_context():
            ax = plt.gca()
        ax = PandasMPLPlot._get_ax_layer(ax)
    return _plot(data, kind=kind, ax=ax, figsize=figsize, use_index=use_index, title=title, grid=grid, legend=legend, style=style, logx=logx, logy=logy, loglog=loglog, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, rot=rot, fontsize=fontsize, colormap=colormap, table=table, yerr=yerr, xerr=xerr, label=label, secondary_y=secondary_y, **kwds)

def plot_frame(data: pd.DataFrame, x: str | int = None, y: str | int = None, kind: str = 'line', ax: mat.axes.Axes = None, subplots: bool = False, sharex: bool = None, sharey: bool = False, layout: tuple = None, figsize: tuple = None, use_index: bool = True, title: str | list = None, grid: bool = None, legend: bool | str = True, style: list | dict = None, logx: bool = False, logy: bool = False, loglog: bool = False, xticks: list = None, yticks: list = None, xlim: list = None, ylim: list = None, rot: int = None, fontsize: int = None, colormap: str | mat.colors.Colormap = None, table: bool | pd.Series | pd.DataFrame = False, yerr: pd.DataFrame | pd.Series | np.ndarray | dict | str = None, xerr: pd.DataFrame | pd.Series | np.ndarray | dict | str = None, secondary_y: bool | list = False, sort_columns: bool = False, **kwds) -> mat.axes.Axes | np.ndarray:
    return _plot(data, x=x, y=y, subplots=subplots, ax=ax, kind=kind, figsize=figsize, use_index=use_index, title=title, grid=grid, legend=legend, subplots=subplots, style=style, logx=logx, logy=logy, loglog=loglog, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, rot=rot, fontsize=fontsize, colormap=colormap, table=table, yerr=yerr, xerr=xerr, sharex=sharex, sharey=sharey, secondary_y=secondary_y, layout=layout, sort_columns=sort_columns, **kwds)

def _plot(data: pd.Series | pd.DataFrame, x: str | int = None, y: str | int = None, subplots: bool = False, ax: mat.axes.Axes = None, kind: str = 'line', **kwds) -> mat.axes.Axes | np.ndarray:
    from databricks.koalas import DataFrame
    kind = kind.lower().strip()
    kind = {'density': 'kde'}.get(kind, kind)
    if kind in _all_kinds:
        klass = _plot_klass[kind]
    else:
        raise ValueError('%r is not a valid plot kind' % kind)
    if kind in ('scatter', 'hexbin'):
        plot_obj = klass(data, x, y, subplots=subplots, ax=ax, kind=kind, **kwds)
    else:
        if isinstance(data, DataFrame):
            if x is not None:
                data = data.set_index(x)
            if y is not None:
                data = data[y]
        plot_obj = klass(data, subplots=subplots, ax=ax, kind=kind, **kwds)
    plot_obj.generate()
    plot_obj.draw()
    return plot_obj.result
