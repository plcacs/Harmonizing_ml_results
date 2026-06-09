from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: asarray, int64, ndarray, object_, timedelta64

# === Internal dependency: pandas._libs.index ===
class PeriodEngine(Int64Engine): ...

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.dtypes import Resolution
from pandas._libs.tslibs.nattype import NaT
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.offsets import Tick

# === Internal dependency: pandas._libs.tslibs.dtypes ===
OFFSET_TO_PERIOD_FREQSTR = ...

# === Internal dependency: pandas._typing ===
Self: Any
npt: Any

# === Internal dependency: pandas.core.arrays.period ===
class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):
    def __init__(self, values, dtype=..., copy=...): ...
def raise_on_incompatible(left, right): ...
def period_array(data, freq=..., copy=...): ...
def validate_dtype_freq(dtype, freq): ...

# === Internal dependency: pandas.core.common ===
def count_not_none(*args): ...

# === Internal dependency: pandas.core.dtypes.common ===
from pandas.core.dtypes.inference import is_integer

# === Internal dependency: pandas.core.dtypes.dtypes ===
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    ...

# === Internal dependency: pandas.core.dtypes.generic ===
def create_pandas_abc_type(name, attr, comp): ...
ABCSeries = cast(...)

# === Internal dependency: pandas.core.dtypes.missing ===
def is_valid_na_for_dtype(obj, dtype): ...

# === Internal dependency: pandas.core.indexes.base ===
def maybe_extract_name(name, obj, cls): ...
_index_doc_kwargs = {'klass': 'Index', 'inplace': '', 'target_klass': 'Index', 'raises_section': '', 'unique': 'Index', 'duplicated': 'np.ndarray'}

# === Internal dependency: pandas.core.indexes.datetimelike ===
class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    def freq(self): ...
    def freq(self, value): ...
    def asi8(self): ...
    def freqstr(self): ...
    def _resolution_obj(self): ...
    def _formatter_func(self): ...

# === Internal dependency: pandas.core.indexes.datetimes ===
class DatetimeIndex(DatetimeTimedeltaMixin): ...
from pandas.core.indexes.base import Index

# === Internal dependency: pandas.core.indexes.extension ===
def inherit_names(names, delegate, cache=..., wrap=...): ...

# === Internal dependency: pandas.util._decorators ===
def doc(*docstrings, **params): ...
from pandas._libs.properties import cache_readonly