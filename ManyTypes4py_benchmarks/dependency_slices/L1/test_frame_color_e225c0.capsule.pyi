from typing import Any

# === Third-party dependency: cycler ===
def cycler(arg: Cycler[K, V]) -> Cycler[K, V]: ...
def cycler(**kwargs: Iterable[V]) -> Cycler[str, V]: ...
def cycler(label: K, itr: Iterable[V]) -> Cycler[K, V]: ...
def cycler(*args, **kwargs) -> Any: ...

# === Third-party dependency: numpy ===
# Used symbols: arange, array, float64, isclose, linspace, random, unique

# === Internal dependency: pandas ===
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.tests.plotting.common ===
def _check_colors(collections, linecolors=..., facecolors=..., mapping=...): ...
def _unpack_cycler(rcParams, field=...): ...
def _check_plot_works(f, default_axes=..., **kwargs): ...

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises