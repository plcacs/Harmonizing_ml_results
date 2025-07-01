"""
Module consolidating common testing functions for checking plotting.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, List, Tuple, Dict
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
if TYPE_CHECKING:
    from collections.abc import Sequence
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist

def _check_legend_labels(axes: Union[Axes, Sequence[Axes]], labels: Optional[Sequence[str]] = None, visible: bool = True) -> None:
    if visible and labels is None:
        raise ValueError('labels must be specified when visible is True')
    axes = _flatten_visible(axes)
    for ax in axes:
        if visible:
            assert ax.get_legend() is not None
            _check_text_labels(ax.get_legend().get_texts(), labels)
        else:
            assert ax.get_legend() is None

def _check_legend_marker(ax: Axes, expected_markers: Optional[Sequence[str]] = None, visible: bool = True) -> None:
    if visible and expected_markers is None:
        raise ValueError('Markers must be specified when visible is True')
    if visible:
        handles, _ = ax.get_legend_handles_labels()
        markers = [handle.get_marker() for handle in handles]
        assert markers == expected_markers
    else:
        assert ax.get_legend() is None

def _check_data(xp: Axes, rs: Axes) -> None:
    xp_lines = xp.get_lines()
    rs_lines = rs.get_lines()
    assert len(xp_lines) == len(rs_lines)
    for xpl, rsl in zip(xp_lines, rs_lines):
        xpdata = xpl.get_xydata()
        rsdata = rsl.get_xydata()
        tm.assert_almost_equal(xpdata, rsdata)

def _check_visible(collections: Union[Artist, Sequence[Artist]], visible: bool = True) -> None:
    from matplotlib.collections import Collection
    if not isinstance(collections, Collection) and (not is_list_like(collections)):
        collections = [collections]
    for patch in collections:
        assert patch.get_visible() == visible

def _check_patches_all_filled(axes: Union[Axes, Sequence[Axes]], filled: bool = True) -> None:
    axes = _flatten_visible(axes)
    for ax in axes:
        for patch in ax.patches:
            assert patch.fill == filled

def _get_colors_mapped(series: Series, colors: Sequence[str]) -> List[str]:
    unique = series.unique()
    mapped = dict(zip(unique, colors))
    return [mapped[v] for v in series.values]

def _check_colors(collections: Sequence[Artist], linecolors: Optional[Sequence[str]] = None, facecolors: Optional[Sequence[str]] = None, mapping: Optional[Series] = None) -> None:
    from matplotlib import colors
    from matplotlib.collections import Collection, LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    conv = colors.ColorConverter
    if linecolors is not None:
        if mapping is not None:
            linecolors = _get_colors_mapped(mapping, linecolors)
            linecolors = linecolors[:len(collections)]
        assert len(collections) == len(linecolors)
        for patch, color in zip(collections, linecolors):
            if isinstance(patch, Line2D):
                result = patch.get_color()
                result = conv.to_rgba(result)
            elif isinstance(patch, (PolyCollection, LineCollection)):
                result = tuple(patch.get_edgecolor()[0])
            else:
                result = patch.get_edgecolor()
            expected = conv.to_rgba(color)
            assert result == expected
    if facecolors is not None:
        if mapping is not None:
            facecolors = _get_colors_mapped(mapping, facecolors)
            facecolors = facecolors[:len(collections)]
        assert len(collections) == len(facecolors)
        for patch, color in zip(collections, facecolors):
            if isinstance(patch, Collection):
                result = patch.get_facecolor()[0]
            else:
                result = patch.get_facecolor()
            if isinstance(result, np.ndarray):
                result = tuple(result)
            expected = conv.to_rgba(color)
            assert result == expected

def _check_text_labels(texts: Union[Artist, Sequence[Artist]], expected: Union[str, Sequence[str]]) -> None:
    if not is_list_like(texts):
        assert texts.get_text() == expected
    else:
        labels = [t.get_text() for t in texts]
        assert len(labels) == len(expected)
        for label, e in zip(labels, expected):
            assert label == e

def _check_ticks_props(axes: Union[Axes, Sequence[Axes]], xlabelsize: Optional[float] = None, xrot: Optional[float] = None, ylabelsize: Optional[float] = None, yrot: Optional[float] = None) -> None:
    from matplotlib.ticker import NullFormatter
    axes = _flatten_visible(axes)
    for ax in axes:
        if xlabelsize is not None or xrot is not None:
            if isinstance(ax.xaxis.get_minor_formatter(), NullFormatter):
                labels = ax.get_xticklabels()
            else:
                labels = ax.get_xticklabels() + ax.get_xticklabels(minor=True)
            for label in labels:
                if xlabelsize is not None:
                    tm.assert_almost_equal(label.get_fontsize(), xlabelsize)
                if xrot is not None:
                    tm.assert_almost_equal(label.get_rotation(), xrot)
        if ylabelsize is not None or yrot is not None:
            if isinstance(ax.yaxis.get_minor_formatter(), NullFormatter):
                labels = ax.get_yticklabels()
            else:
                labels = ax.get_yticklabels() + ax.get_yticklabels(minor=True)
            for label in labels:
                if ylabelsize is not None:
                    tm.assert_almost_equal(label.get_fontsize(), ylabelsize)
                if yrot is not None:
                    tm.assert_almost_equal(label.get_rotation(), yrot)

def _check_ax_scales(axes: Union[Axes, Sequence[Axes]], xaxis: str = 'linear', yaxis: str = 'linear') -> None:
    axes = _flatten_visible(axes)
    for ax in axes:
        assert ax.xaxis.get_scale() == xaxis
        assert ax.yaxis.get_scale() == yaxis

def _check_axes_shape(axes: Union[Axes, Sequence[Axes]], axes_num: Optional[int] = None, layout: Optional[Tuple[int, int]] = None, figsize: Optional[Tuple[float, float]] = None) -> None:
    from pandas.plotting._matplotlib.tools import flatten_axes
    if figsize is None:
        figsize = (6.4, 4.8)
    visible_axes = _flatten_visible(axes)
    if axes_num is not None:
        assert len(visible_axes) == axes_num
        for ax in visible_axes:
            assert len(ax.get_children()) > 0
    if layout is not None:
        x_set = set()
        y_set = set()
        for ax in flatten_axes(axes):
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        result = (len(y_set), len(x_set))
        assert result == layout
    tm.assert_numpy_array_equal(visible_axes[0].figure.get_size_inches(), np.array(figsize, dtype=np.float64))

def _flatten_visible(axes: Union[Axes, Sequence[Axes]]) -> List[Axes]:
    from pandas.plotting._matplotlib.tools import flatten_axes
    axes_ndarray = flatten_axes(axes)
    axes = [ax for ax in axes_ndarray if ax.get_visible()]
    return axes

def _check_has_errorbars(axes: Union[Axes, Sequence[Axes]], xerr: int = 0, yerr: int = 0) -> None:
    axes = _flatten_visible(axes)
    for ax in axes:
        containers = ax.containers
        xerr_count = 0
        yerr_count = 0
        for c in containers:
            has_xerr = getattr(c, 'has_xerr', False)
            has_yerr = getattr(c, 'has_yerr', False)
            if has_xerr:
                xerr_count += 1
            if has_yerr:
                yerr_count += 1
        assert xerr == xerr_count
        assert yerr == yerr_count

def _check_box_return_type(returned: Union[Series, Axes, Tuple[Axes, Dict[str, Artist]], Dict[str, Artist]], return_type: Optional[str], expected_keys: Optional[Sequence[str]] = None, check_ax_title: bool = True) -> None:
    from matplotlib.axes import Axes
    types = {'dict': dict, 'axes': Axes, 'both': tuple}
    if expected_keys is None:
        if return_type is None:
            return_type = 'dict'
        assert isinstance(returned, types[return_type])
        if return_type == 'both':
            assert isinstance(returned.ax, Axes)
            assert isinstance(returned.lines, dict)
    else:
        if return_type is None:
            for r in _flatten_visible(returned):
                assert isinstance(r, Axes)
            return
        assert isinstance(returned, Series)
        assert sorted(returned.keys()) == sorted(expected_keys)
        for key, value in returned.items():
            assert isinstance(value, types[return_type])
            if return_type == 'axes':
                if check_ax_title:
                    assert value.get_title() == key
            elif return_type == 'both':
                if check_ax_title:
                    assert value.ax.get_title() == key
                assert isinstance(value.ax, Axes)
                assert isinstance(value.lines, dict)
            elif return_type == 'dict':
                line = value['medians'][0]
                axes = line.axes
                if check_ax_title:
                    assert axes.get_title() == key
            else:
                raise AssertionError

def _check_grid_settings(obj: pd.DataFrame, kinds: Sequence[str], kws: Optional[Dict] = None) -> None:
    import matplotlib as mpl

    def is_grid_on() -> bool:
        xticks = mpl.pyplot.gca().xaxis.get_major_ticks()
        yticks = mpl.pyplot.gca().yaxis.get_major_ticks()
        xoff = all((not g.gridline.get_visible() for g in xticks))
        yoff = all((not g.gridline.get_visible() for g in yticks))
        return not (xoff and yoff)
    if kws is None:
        kws = {}
    spndx = 1
    for kind in kinds:
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc('axes', grid=False)
        obj.plot(kind=kind, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc('axes', grid=True)
        obj.plot(kind=kind, grid=False, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()
        if kind not in ['pie', 'hexbin', 'scatter']:
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc('axes', grid=True)
            obj.plot(kind=kind, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc('axes', grid=False)
            obj.plot(kind=kind, grid=True, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()

def _unpack_cycler(rcParams: Dict, field: str = 'color') -> List[str]:
    return [v[field] for v in rcParams['axes.prop_cycle']]

def get_x_axis(ax: Axes) -> List[Axes]:
    return ax._shared_axes['x']

def get_y_axis(ax: Axes) -> List[Axes]:
    return ax._shared_axes['y']

def assert_is_valid_plot_return_object(objs: Union[Series, np.ndarray, Artist, Tuple, Dict]) -> None:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    if isinstance(objs, (Series, np.ndarray)):
        if isinstance(objs, Series):
            objs = objs._values
        for el in objs.reshape(-1):
            msg = f"one of 'objs' is not a matplotlib Axes instance, type encountered {type(el).__name__!r}"
            assert isinstance(el, (Axes, dict)), msg
    else:
        msg = f"objs is neither an ndarray of Artist instances nor a single ArtistArtist instance, tuple, or dict, 'objs' is a {type(objs).__name__!r}"
        assert isinstance(objs, (Artist, tuple, dict)), msg

def _check_plot_works(f: callable, default_axes: bool = False, **kwargs) -> Union[Artist, Tuple, Dict]:
    import matplotlib.pyplot as plt
    if default_axes:
        gen_plots = _gen_default_plot
    else:
        gen_plots = _gen_two_subplots
    ret = None
    fig = kwargs.get('figure', plt.gcf())
    fig.clf()
    for ret in gen_plots(f, fig, **kwargs):
        assert_is_valid_plot_return_object(ret)
    return ret

def _gen_default_plot(f: callable, fig: plt.Figure, **kwargs) -> Artist:
    yield f(**kwargs)

def _gen_two_subplots(f: callable, fig: plt.Figure, **kwargs) -> Artist:
    if 'ax' not in kwargs:
        fig.add_subplot(211)
    yield f(**kwargs)
    if f is pd.plotting.bootstrap_plot:
        assert 'ax' not in kwargs
    else:
        kwargs['ax'] = fig.add_subplot(212)
    yield f(**kwargs)
