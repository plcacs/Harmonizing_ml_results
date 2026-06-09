from typing import Any

# === Third-party dependency: matplotlib ===
def rc_context(rc = ..., fname = ...) -> Any: ...

# === Third-party dependency: matplotlib.axes ===
# Used symbols: Axes

# === Third-party dependency: matplotlib.lines ===
class Line2D(Artist): ...

# === Third-party dependency: matplotlib.pyplot ===
def draw_if_interactive(*args, **kwargs) -> Any: ...
def gca() -> Axes: ...

# === Third-party dependency: numpy ===
# Used symbols: append, array, asarray, nan, ndarray, take

# === Internal dependency: pandas ===
from pandas.core.api import MultiIndex
from pandas.core.api import Series
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._libs.lib ===
class _NoDefault(Enum):
    no_default = Ellipsis
no_default = _NoDefault.no_default

# === Internal dependency: pandas.core.common ===
def convert_to_list_like(values): ...

# === Internal dependency: pandas.core.dtypes.common ===
from pandas.core.dtypes.inference import is_dict_like

# === Internal dependency: pandas.core.dtypes.generic ===
def create_pandas_abc_type(name, attr, comp): ...
ABCSeries = cast(...)

# === Internal dependency: pandas.core.dtypes.missing ===
def remove_na_arraylike(arr): ...

# === Internal dependency: pandas.io.formats.printing ===
def pprint_thing(thing, _nest_lvl=..., escape_chars=..., default_escapes=..., quote_strings=..., max_seq_items=...): ...

# === Internal dependency: pandas.plotting._matplotlib.core ===
class MPLPlot(ABC): ...
class LinePlot(MPLPlot):
    def orientation(self): ...
    def _kind(self): ...

# === Internal dependency: pandas.plotting._matplotlib.groupby ===
def create_iter_data_given_by(data, kind=...): ...

# === Internal dependency: pandas.plotting._matplotlib.style ===
def get_standard_colors(num_colors, colormap=..., color_type=..., *, color): ...
def get_standard_colors(num_colors, colormap=..., color_type=..., *, color=...): ...

# === Internal dependency: pandas.plotting._matplotlib.tools ===
def maybe_adjust_figure(fig, *args, **kwargs): ...
def create_subplots(naxes, sharex=..., sharey=..., squeeze=..., subplot_kw=..., ax=..., layout=..., layout_type=..., **fig_kw): ...
def flatten_axes(axes): ...

# === Internal dependency: pandas.util._decorators ===
from pandas._libs.properties import cache_readonly

# === Internal dependency: pandas.util._exceptions ===
def find_stack_level(): ...