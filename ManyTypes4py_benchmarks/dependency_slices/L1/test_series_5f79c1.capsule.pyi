# === Third-party dependency: numpy ===
# Used symbols: abs, arange, array, delete, float64, int_, isnan, linspace, nan, nanmax, nanmin, random, vstack

# === Internal dependency: pandas ===
from pandas.core.api import period_range
from pandas.core.api import Timedelta
from pandas.core.api import timedelta_range
from pandas.core.api import date_range
from pandas.core.api import bdate_range
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
def external_error_raised(expected_exception): ...
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.compat ===
def is_platform_linux(): ...

# === Internal dependency: pandas.compat.numpy ===
_np_version = np.__version__
_nlv = Version(...)
np_version_gte1p24 = _nlv >= Version('1.24')

# === Internal dependency: pandas.plotting ===
from pandas.plotting._core import PlotAccessor

# === Internal dependency: pandas.plotting._matplotlib.converter ===
class DatetimeConverter(mdates.DateConverter):
    ...

# === Internal dependency: pandas.plotting._matplotlib.style ===
def get_standard_colors(num_colors, colormap=..., color_type=..., *, color): ...
def get_standard_colors(num_colors, colormap=..., color_type=..., *, color=...): ...

# === Internal dependency: pandas.tests.plotting.common ===
def _check_legend_labels(axes, labels=..., visible=...): ...
def _check_colors(collections, linecolors=..., facecolors=..., mapping=...): ...
def _check_text_labels(texts, expected): ...
def _check_ticks_props(axes, xlabelsize=..., xrot=..., ylabelsize=..., yrot=...): ...
def _check_ax_scales(axes, xaxis=..., yaxis=...): ...
def _check_axes_shape(axes, axes_num=..., layout=..., figsize=...): ...
def _check_has_errorbars(axes, xerr=..., yerr=...): ...
def _check_grid_settings(obj, kinds, kws=...): ...
def _unpack_cycler(rcParams, field=...): ...
def get_y_axis(ax): ...
def _check_plot_works(f, default_axes=..., **kwargs): ...

# === Internal dependency: pandas.tseries.offsets ===
from pandas._libs.tslibs.offsets import CustomBusinessDay

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package, min_version=...): ...

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises