from distutils.version import LooseVersion
from typing import Any, Dict, List, Optional, Union, Tuple
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

from matplotlib.axes import Axes

class KoalasBarPlot(PandasBarPlot, TopNPlotBase):

    def __init__(self, data: Any, **kwargs: Any) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _plot(self, ax: Axes, x: Any, y: Any, w: Any, start: float = 0, log: bool = False, **kwds: Any) -> Any:
        self.set_result_text(ax)
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)

class KoalasBoxPlot(PandasBoxPlot, BoxPlotBase):

    def boxplot(self, ax: Axes, bxpstats: List[Dict[str, Any]], notch: Optional[Any] = None, sym: Optional[str] = None, vert: Optional[Any] = None, whis: Optional[Any] = None, positions: Optional[Any] = None, widths: Optional[Any] = None, patch_artist: Optional[bool] = None, bootstrap: Optional[Any] = None, usermedians: Optional[Any] = None, conf_intervals: Optional[Any] = None, meanline: Optional[Any] = None, showmeans: Optional[Any] = None, showcaps: Optional[Any] = None, showbox: Optional[Any] = None, showfliers: Optional[Any] = None, boxprops: Optional[Dict[str, Any]] = None, labels: Optional[Any] = None, flierprops: Optional[Dict[str, Any]] = None, medianprops: Optional[Dict[str, Any]] = None, meanprops: Optional[Dict[str, Any]] = None, capprops: Optional[Dict[str, Any]] = None, whiskerprops: Optional[Dict[str, Any]] = None, manage_ticks: Optional[bool] = None, manage_xticks: Optional[bool] = None, autorange: bool = False, zorder: Optional[Any] = None, precision: Optional[Any] = None) -> Any:

        def update_dict(dictionary: Optional[Dict[str, Any]], rc_name: str, properties: List[str]) -> Dict[str, Any]:
            """ Loads properties in the dictionary from rc file if not already
            in the dictionary"""
            rc_str = 'boxplot.{0}.{1}'
            if dictionary is None:
                dictionary = dict()
            for prop_dict in properties:
                dictionary.setdefault(prop_dict, mat.rcParams[rc_str.format(rc_name, prop_dict)])
            return dictionary
        flier_props: List[str] = ['color', 'marker', 'markerfacecolor', 'markeredgecolor', 'markersize', 'linestyle', 'linewidth']
        default_props: List[str] = ['color', 'linewidth', 'linestyle']
        boxprops = update_dict(boxprops, 'boxprops', default_props)
        whiskerprops = update_dict(whiskerprops, 'whiskerprops', default_props)
        capprops = update_dict(capprops, 'capprops', default_props)
        medianprops = update_dict(medianprops, 'medianprops', default_props)
        meanprops = update_dict(meanprops, 'meanprops', default_props)
        flierprops = update_dict(flierprops, 'flierprops', flier_props)
        if patch_artist:
            boxprops['linestyle'] = 'solid'
            boxprops['edgecolor'] = boxprops.pop('color')
        if sym is not None:
            if sym == '':
                flierprops = dict(linestyle='none', marker='', color='none')
                showfliers = False
            else:
                _, marker, color = _process_plot_format(sym)
                if marker is not None:
                    flierprops['marker'] = marker
                if color is not None:
                    flierprops['color'] = color
                    flierprops['markerfacecolor'] = color
                    flierprops['markeredgecolor'] = color
        if usermedians is not None:
            if len(np.ravel(usermedians)) != len(bxpstats) or np.shape(usermedians)[0] != len(bxpstats):
                raise ValueError('usermedians length not compatible with x')
            else:
                for stats, med in zip(bxpstats, usermedians):
                    if med is not None:
                        stats['med'] = med
        if conf_intervals is not None:
            if np.shape(conf_intervals)[0] != len(bxpstats):
                err_mess = 'conf_intervals length not compatible with x'
                raise ValueError(err_mess)
            else:
                for stats, ci in zip(bxpstats, conf_intervals):
                    if ci is not None:
                        if len(ci) != 2:
                            raise ValueError('each confidence interval must have two values')
                        else:
                            if ci[0] is not None:
                                stats['cilo'] = ci[0]
                            if ci[1] is not None:
                                stats['cihi'] = ci[1]
        should_manage_ticks: bool = True
        if manage_xticks is not None:
            should_manage_ticks = manage_xticks
        if manage_ticks is not None:
            should_manage_ticks = manage_ticks
        if LooseVersion(mat.__version__) < LooseVersion('3.1.0'):
            extra_args = {'manage_xticks': should_manage_ticks}
        else:
            extra_args = {'manage_ticks': should_manage_ticks}
        artists = ax.bxp(bxpstats, positions=positions, widths=widths, vert=vert, patch_artist=patch_artist, shownotches=notch, showmeans=showmeans, showcaps=showcaps, showbox=showbox, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, meanprops=meanprops, meanline=meanline, showfliers=showfliers, capprops=capprops, whiskerprops=whiskerprops, zorder=zorder, **extra_args)
        return artists

    def _plot(self, ax: Axes, bxpstats: List[Dict[str, Any]], column_num: Optional[int] = None, return_type: str = 'axes', **kwds: Any) -> Any:
        bp = self.boxplot(ax, bxpstats, **kwds)
        if return_type == 'dict':
            return (bp, bp)
        elif return_type == 'both':
            return (self.BP(ax=ax, lines=bp), bp)
        else:
            return (ax, bp)

    def _compute_plot_data(self) -> None:
        colname: Any = self.data.name
        spark_column_name: Any = self.data._internal.spark_column_name_for(self.data._column_label)
        data: Any = self.data
        self.kwds.update(KoalasBoxPlot.rc_defaults(**self.kwds))
        showfliers: Any = self.kwds.get('showfliers', False)
        whis: Any = self.kwds.get('whis', 1.5)
        labels: Any = self.kwds.get('labels', [colname])
        precision: Any = self.kwds.get('precision', 0.01)
        col_stats, col_fences = BoxPlotBase.compute_stats(data, spark_column_name, whis, precision)
        outliers = BoxPlotBase.outliers(data, spark_column_name, *col_fences)
        whiskers = BoxPlotBase.calc_whiskers(spark_column_name, outliers)
        if showfliers:
            fliers = BoxPlotBase.get_fliers(spark_column_name, outliers, whiskers[0])
        else:
            fliers = []
        stats: List[Dict[str, Any]] = []
        item: Dict[str, Any] = {'mean': col_stats['mean'], 'med': col_stats['med'], 'q1': col_stats['q1'], 'q3': col_stats['q3'], 'whislo': whiskers[0], 'whishi': whiskers[1], 'fliers': fliers, 'label': labels[0]}
        stats.append(item)
        self.data = {labels[0]: stats}

    def _make_plot(self) -> None:
        bxpstats: List[Dict[str, Any]] = list(self.data.values())[0]
        ax: Axes = self._get_ax(0)
        kwds: Dict[str, Any] = self.kwds.copy()
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
    def rc_defaults(notch: Optional[Any] = None, vert: Optional[Any] = None, whis: Optional[Any] = None, patch_artist: Optional[Any] = None, bootstrap: Optional[Any] = None, meanline: Optional[Any] = None, showmeans: Optional[Any] = None, showcaps: Optional[Any] = None, showbox: Optional[Any] = None, showfliers: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
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
            ax: Axes = self._get_ax(i)
            kwds: Dict[str, Any] = self.kwds.copy()
            label_str = pprint_thing(label if len(label) > 1 else label[0])
            kwds['label'] = label_str
            style, kwds = self._apply_style_colors(colors, kwds, i, label_str)
            if style is not None:
                kwds['style'] = style
            kwds = self._make_plot_keywords(kwds, y)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            self._add_legend_handle(artists[0], label_str, index=i)

    @classmethod
    def _plot(cls, ax: Axes, y: Any, style: Optional[Any] = None, bins: Optional[Any] = None, bottom: float = 0, column_num: int = 0, stacking_id: Any = None, **kwds: Any) -> Any:
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)
        base = np.zeros(len(bins) - 1)
        bottom_val = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds['label'])
        n, bins_out, patches = ax.hist(bins[:-1], bins=bins, bottom=bottom_val, weights=y, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

class KoalasPiePlot(PandasPiePlot, TopNPlotBase):

    def __init__(self, data: Any, **kwargs: Any) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasAreaPlot(PandasAreaPlot, SampledPlotBase):

    def __init__(self, data: Any, **kwargs: Any) -> None:
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasLinePlot(PandasLinePlot, SampledPlotBase):

    def __init__(self, data: Any, **kwargs: Any) -> None:
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasBarhPlot(PandasBarhPlot, TopNPlotBase):

    def __init__(self, data: Any, **kwargs: Any) -> None:
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self) -> None:
        self.set_result_text(self._get_ax(0))
        super()._make_plot()

class KoalasScatterPlot(PandasScatterPlot, TopNPlotBase):

    def __init__(self, data: Any, x: Any, y: Any, **kwargs: Any) -> None:
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
            ax: Axes = self._get_ax(i)
            kwds: Dict[str, Any] = self.kwds.copy()
            label_str = pprint_thing(label if len(label) > 1 else label[0])
            kwds['label'] = label_str
            style, kwds = self._apply_style_colors(colors, kwds, i, label_str)
            if style is not None:
                kwds['style'] = style
            kwds = self._make_plot_keywords(kwds, y)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            self._add_legend_handle(artists[0], label_str, index=i)

    def _get_ind(self, y: Any) -> Any:
        return KdePlotBase.get_ind(y, self.ind)

    @classmethod
    def _plot(cls, ax: Axes, y: Any, style: Optional[Any] = None, bw_method: Optional[Any] = None, ind: Optional[Any] = None, column_num: Optional[int] = None, stacking_id: Any = None, **kwds: Any) -> Any:
        y_vals = KdePlotBase.compute_kde(y, bw_method=bw_method, ind=ind)
        lines = PandasMPLPlot._plot(ax, ind, y_vals, style=style, **kwds)
        return lines

_klasses = [KoalasHistPlot, KoalasBarPlot, KoalasBoxPlot, KoalasPiePlot, KoalasAreaPlot, KoalasLinePlot, KoalasBarhPlot, KoalasScatterPlot, KoalasKdePlot]
_plot_klass: Dict[str, Any] = {getattr(klass, '_kind'): klass for klass in _klasses}
_common_kinds: set = {'area', 'bar', 'barh', 'box', 'hist', 'kde', 'line', 'pie'}
_series_kinds = _common_kinds.union(set())
_dataframe_kinds = _common_kinds.union({'scatter', 'hexbin'})
_koalas_all_kinds = _common_kinds.union(_series_kinds).union(_dataframe_kinds)

def plot_koalas(data: Any, kind: str, **kwargs: Any) -> Any:
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

def plot_series(data: Any, kind: str = 'line', ax: Optional[Any] = None, figsize: Optional[Tuple[float, float]] = None, use_index: bool = True, title: Optional[Any] = None, grid: Optional[bool] = None, legend: Union[bool, str] = False, style: Optional[Any] = None, logx: bool = False, logy: bool = False, loglog: bool = False, xticks: Optional[Any] = None, yticks: Optional[Any] = None, xlim: Optional[Union[List[float], Tuple[float, float]]] = None, ylim: Optional[Union[List[float], Tuple[float, float]]] = None, rot: Optional[int] = None, fontsize: Optional[int] = None, colormap: Optional[Any] = None, table: Union[bool, Any] = False, yerr: Optional[Any] = None, xerr: Optional[Any] = None, label: Optional[Any] = None, secondary_y: bool = False, **kwds: Any) -> Any:
    """
    Make plots of Series using matplotlib / pylab.
    """
    import matplotlib.pyplot as plt
    if ax is None and len(plt.get_fignums()) > 0:
        with plt.rc_context():
            ax = plt.gca()
        ax = PandasMPLPlot._get_ax_layer(ax)
    return _plot(data, kind=kind, ax=ax, figsize=figsize, use_index=use_index, title=title, grid=grid, legend=legend, style=style, logx=logx, logy=logy, loglog=loglog, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, rot=rot, fontsize=fontsize, colormap=colormap, table=table, yerr=yerr, xerr=xerr, label=label, secondary_y=secondary_y, **kwds)

def plot_frame(data: Any, x: Optional[Any] = None, y: Optional[Any] = None, kind: str = 'line', ax: Optional[Any] = None, subplots: Optional[Any] = None, sharex: Optional[bool] = None, sharey: bool = False, layout: Optional[Any] = None, figsize: Optional[Tuple[float, float]] = None, use_index: bool = True, title: Optional[Any] = None, grid: Optional[bool] = None, legend: Union[bool, str] = True, style: Optional[Any] = None, logx: bool = False, logy: bool = False, loglog: bool = False, xticks: Optional[Any] = None, yticks: Optional[Any] = None, xlim: Optional[Union[List[float], Tuple[float, float]]] = None, ylim: Optional[Union[List[float], Tuple[float, float]]] = None, rot: Optional[int] = None, fontsize: Optional[int] = None, colormap: Optional[Any] = None, table: Union[bool, Any] = False, yerr: Optional[Any] = None, xerr: Optional[Any] = None, secondary_y: bool = False, sort_columns: bool = False, **kwds: Any) -> Any:
    """
    Make plots of DataFrames using matplotlib / pylab.
    """
    return _plot(data, kind=kind, x=x, y=y, ax=ax, figsize=figsize, use_index=use_index, title=title, grid=grid, legend=legend, subplots=subplots, style=style, logx=logx, logy=logy, loglog=loglog, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, rot=rot, fontsize=fontsize, colormap=colormap, table=table, yerr=yerr, xerr=xerr, sharex=sharex, sharey=sharey, secondary_y=secondary_y, layout=layout, sort_columns=sort_columns, **kwds)

def _plot(data: Any, x: Optional[Any] = None, y: Optional[Any] = None, subplots: bool = False, ax: Optional[Any] = None, kind: str = 'line', **kwds: Any) -> Any:
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