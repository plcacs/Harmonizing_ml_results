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
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._libs.lib ===
no_default: Final

# === Internal dependency: pandas.core.common ===
def convert_to_list_like(values: Hashable | Iterable | AnyArrayLike) -> list | AnyArrayLike: ...

# === Internal dependency: pandas.core.dtypes.common ===
# re-export: from pandas.core.dtypes.inference import is_dict_like

# === Internal dependency: pandas.core.dtypes.generic ===
ABCSeries: cast

# === Internal dependency: pandas.core.dtypes.missing ===
def remove_na_arraylike(arr: Series | Index | np.ndarray) -> Any: ...

# === Internal dependency: pandas.io.formats.printing ===
def pprint_thing(thing: object, _nest_lvl: int = ..., escape_chars: EscapeChars | None = ..., default_escapes: bool = ..., quote_strings: bool = ..., max_seq_items: int | None = ...) -> str: ...

# === Internal dependency: pandas.plotting._matplotlib.core ===
class MPLPlot(ABC): ...
class LinePlot(MPLPlot):
    def orientation(self) -> PlottingOrientation: ...
    def _kind(self) -> Literal['line', 'area', 'hist', 'kde', 'box']: ...

# === Internal dependency: pandas.plotting._matplotlib.groupby ===
def create_iter_data_given_by(data: DataFrame, kind: str = ...) -> dict[Hashable, DataFrame | Series]: ...

# === Internal dependency: pandas.plotting._matplotlib.style ===
def get_standard_colors(num_colors: int, colormap: Colormap | None = ..., color_type: str = ..., *, color: dict[str, Color]) -> dict[str, Color]: ...
def get_standard_colors(num_colors: int, colormap: Colormap | None = ..., color_type: str = ..., *, color: Color | Sequence[Color] | None = ...) -> list[Color]: ...
def get_standard_colors(num_colors: int, colormap: Colormap | None = ..., color_type: str = ..., *, color: dict[str, Color] | Color | Sequence[Color] | None = ...) -> dict[str, Color] | list[Color]: ...

# === Internal dependency: pandas.plotting._matplotlib.tools ===
def maybe_adjust_figure(fig: Figure, *args, **kwargs) -> None: ...
def create_subplots(naxes: int, sharex: bool = ..., sharey: bool = ..., squeeze: bool = ..., subplot_kw = ..., ax = ..., layout = ..., layout_type: str = ..., **fig_kw) -> Any: ...
def flatten_axes(axes: Axes | Iterable[Axes]) -> Generator[Axes, None, None]: ...

# === Internal dependency: pandas.util._decorators ===
# re-export: from pandas._libs.properties import cache_readonly

# === Internal dependency: pandas.util._exceptions ===
def find_stack_level() -> int: ...