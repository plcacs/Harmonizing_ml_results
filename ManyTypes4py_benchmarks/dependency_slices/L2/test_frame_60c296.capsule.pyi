from typing import Any

# === Third-party dependency: mpl_toolkits.axes_grid1 ===
# Used symbols: make_axes_locatable

# === Third-party dependency: mpl_toolkits.axes_grid1.inset_locator ===
def inset_axes(parent_axes, width, height, loc = ..., bbox_to_anchor = ..., bbox_transform = ..., axes_class = ..., axes_kwargs = ..., borderpad = ...) -> Any: ...

# === Third-party dependency: numpy ===
# Used symbols: __version__, abs, arange, array, delete, eye, float64, int64, nan, nanmax, nanmin, ones, random, repeat, std, zeros

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import PeriodIndex
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import bdate_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
def external_error_raised(expected_exception: type[Exception]) -> ContextManager: ...
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.dtypes.api ===
# re-export: from pandas.core.dtypes.common import is_list_like

# === Internal dependency: pandas.io.formats.printing ===
def pprint_thing(thing: object, _nest_lvl: int = ..., escape_chars: EscapeChars | None = ..., default_escapes: bool = ..., quote_strings: bool = ..., max_seq_items: int | None = ...) -> str: ...

# === Internal dependency: pandas.plotting ===
# re-export: from pandas.plotting._core import PlotAccessor
# re-export: from pandas.plotting._misc import plot_params
# re-export: from pandas.plotting._misc import table

# === Internal dependency: pandas.tests.plotting.common ===
def _check_legend_labels(axes, labels = ..., visible = ...) -> Any: ...
def _check_data(xp, rs) -> Any: ...
def _check_visible(collections, visible = ...) -> Any: ...
def _check_colors(collections, linecolors = ..., facecolors = ..., mapping = ...) -> Any: ...
def _check_text_labels(texts, expected) -> Any: ...
def _check_ticks_props(axes, xlabelsize = ..., xrot = ..., ylabelsize = ..., yrot = ...) -> Any: ...
def _check_ax_scales(axes, xaxis = ..., yaxis = ...) -> Any: ...
def _check_axes_shape(axes, axes_num = ..., layout = ..., figsize = ...) -> Any: ...
def _check_has_errorbars(axes, xerr = ..., yerr = ...) -> Any: ...
def _check_box_return_type(returned, return_type, expected_keys = ..., check_ax_title = ...) -> Any: ...
def _check_grid_settings(obj, kinds, kws = ...) -> Any: ...
def get_y_axis(ax) -> Any: ...
def _check_plot_works(f, default_axes = ..., **kwargs) -> Any: ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, param, raises