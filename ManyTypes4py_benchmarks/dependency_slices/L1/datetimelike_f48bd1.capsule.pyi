from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: array_equal, asarray, int64, intp, ndarray, str_

# === Internal dependency: pandas ===
from pandas.core.api import PeriodIndex

# === Internal dependency: pandas._libs ===
from pandas._libs.tslibs import NaT
from pandas._libs.tslibs import Timedelta

# === Internal dependency: pandas._libs.lib ===
def maybe_indices_to_slice(indices, max_len): ...

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.dtypes import Resolution
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.offsets import Tick
from pandas._libs.tslibs.offsets import to_offset

# === Internal dependency: pandas._libs.tslibs.parsing ===
def parse_datetime_string_with_reso(date_string, freq=..., dayfirst=..., yearfirst=...): ...

# === Internal dependency: pandas._typing ===
Self: Any
npt: Any

# === Internal dependency: pandas.compat.numpy.function ===
class CompatValidator:
    def __init__(self, defaults, fname=..., method=..., max_fname_arg_count=...): ...
TAKE_DEFAULTS = {}
validate_take = CompatValidator(...)

# === Internal dependency: pandas.core.arrays ===
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.period import PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray

# === Internal dependency: pandas.core.arrays.datetimelike ===
class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray): ...

# === Internal dependency: pandas.core.common ===
def asarray_tuplesafe(values, dtype=...): ...

# === Internal dependency: pandas.core.dtypes.common ===
from pandas.core.dtypes.inference import is_integer
from pandas.core.dtypes.inference import is_list_like

# === Internal dependency: pandas.core.dtypes.concat ===
def concat_compat(to_concat, axis=..., ea_compat_axis=...): ...

# === Internal dependency: pandas.core.dtypes.dtypes ===
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype): ...
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...

# === Internal dependency: pandas.core.indexes.base ===
class Index(IndexOpsMixin, PandasObject):
    ...
_index_doc_kwargs = {'klass': 'Index', 'inplace': '', 'target_klass': 'Index', 'raises_section': '', 'unique': 'Index', 'duplicated': 'np.ndarray'}
_index_shared_docs = {}

# === Internal dependency: pandas.core.indexes.extension ===
class NDArrayBackedExtensionIndex(ExtensionIndex):
    ...

# === Internal dependency: pandas.core.indexes.range ===
class RangeIndex(Index): ...

# === Internal dependency: pandas.core.tools.timedeltas ===
def to_timedelta(arg, unit=..., errors=...): ...

# === Internal dependency: pandas.errors ===
class NullFrequencyError(ValueError): ...
class InvalidIndexError(Exception): ...

# === Internal dependency: pandas.util._decorators ===
def doc(*docstrings, **params): ...
class Appender: ...
from pandas._libs.properties import cache_readonly