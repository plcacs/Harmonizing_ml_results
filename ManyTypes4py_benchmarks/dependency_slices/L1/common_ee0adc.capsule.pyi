from typing import Any

# === Third-party dependency: matplotlib ===
def rc(group, **kwargs) -> Any: ...

# === Third-party dependency: matplotlib.artist ===
class Artist: ...

# === Third-party dependency: matplotlib.axes ===
# Used symbols: Axes

# === Third-party dependency: matplotlib.collections ===
class Collection(mcolorizer.ColorizingArtist): ...
class PolyCollection(_CollectionWithSizes): ...
class LineCollection(Collection): ...

# === Third-party dependency: matplotlib.lines ===
class Line2D(Artist): ...

# === Third-party dependency: matplotlib.pyplot ===
def gcf() -> Figure: ...

# === Third-party dependency: matplotlib.ticker ===
class NullFormatter(Formatter): ...

# === Third-party dependency: numpy ===
# Used symbols: array, float64, ndarray

# === Internal dependency: pandas ===
from pandas.core.api import Series
from pandas import plotting

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.core.dtypes.api ===
from pandas.core.dtypes.common import is_list_like

# === Internal dependency: pandas.plotting._matplotlib.tools ===
def flatten_axes(axes): ...