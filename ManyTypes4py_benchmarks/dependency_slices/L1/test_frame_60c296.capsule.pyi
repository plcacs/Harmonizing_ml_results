from typing import Any

# === Third-party dependency: mpl_toolkits.axes_grid1 ===
# Used symbols: make_axes_locatable

# === Third-party dependency: mpl_toolkits.axes_grid1.inset_locator ===
def inset_axes(parent_axes, width, height, loc = ..., bbox_to_anchor = ..., bbox_transform = ..., axes_class = ..., axes_kwargs = ..., borderpad = ...) -> Any: ...

# === Third-party dependency: numpy ===
# Used symbols: __version__, abs, arange, array, delete, eye, float64, int64, nan, nanmax, nanmin, ones, random, repeat, std, zeros

# === Internal dependency: pandas ===
from pandas._config import option_context
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import MultiIndex
from pandas.core.api import PeriodIndex
from pandas.core.api import period_range
from pandas.core.api import date_range
from pandas.core.api import bdate_range
from pandas.core.api import to_datetime
from pandas.core.api import array
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
def external_error_raised(expected_exception): ...
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.dtypes.api ===
from pandas.core.dtypes.common import is_list_like

# === Internal dependency: pandas.io.formats.printing ===
def pprint_thing(thing, _nest_lvl=..., escape_chars=..., default_escapes=..., quote_strings=..., max_seq_items=...): ...

# === Internal dependency: pandas.plotting ===
from pandas.plotting._core import PlotAccessor
from pandas.plotting._misc import plot_params
from pandas.plotting._misc import table

# === Internal dependency: pandas.tests.plotting.common ===
def _check_legend_labels(axes, labels=..., visible=...): ...
def _check_data(xp, rs): ...
def _check_visible(collections, visible=...): ...
def _check_colors(collections, linecolors=..., facecolors=..., mapping=...): ...
def _check_text_labels(texts, expected): ...
def _check_ticks_props(axes, xlabelsize=..., xrot=..., ylabelsize=..., yrot=...): ...
def _check_ax_scales(axes, xaxis=..., yaxis=...): ...
def _check_axes_shape(axes, axes_num=..., layout=..., figsize=...): ...
def _check_has_errorbars(axes, xerr=..., yerr=...): ...
def _check_box_return_type(returned, return_type, expected_keys=..., check_ax_title=...): ...
def _check_grid_settings(obj, kinds, kws=...): ...
def get_y_axis(ax): ...
def _check_plot_works(f, default_axes=..., **kwargs): ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package, min_version=...): ...

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, param, raises