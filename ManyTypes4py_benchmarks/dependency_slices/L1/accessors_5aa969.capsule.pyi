# === Third-party dependency: numpy ===
# Used symbols: asarray, ndarray

# === Internal dependency: pandas ===
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._libs.lib ===
def is_np_dtype(dtype, kinds=...): ...

# === Internal dependency: pandas.core.accessor ===
class PandasDelegate:
    ...
def delegate_names(delegate, accessors, typ, overwrite=..., accessor_mapping=..., raise_on_missing=...): ...

# === Internal dependency: pandas.core.arrays ===
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.period import PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray

# === Internal dependency: pandas.core.arrays.arrow.array ===
class ArrowExtensionArray(OpsMixin, ExtensionArraySupportsAnyAll, ArrowStringArrayMixin, BaseStringArrayMethods): ...

# === Internal dependency: pandas.core.base ===
class PandasObject(DirNamesMixin):
    def _constructor(self): ...
class NoNewAttributesMixin:
    ...

# === Internal dependency: pandas.core.dtypes.common ===
def is_integer_dtype(arr_or_dtype): ...
from pandas.core.dtypes.inference import is_list_like

# === Internal dependency: pandas.core.dtypes.dtypes ===
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype): ...
class DatetimeTZDtype(PandasExtensionDtype): ...
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...
class ArrowDtype(StorageExtensionDtype): ...

# === Internal dependency: pandas.core.dtypes.generic ===
def create_pandas_abc_type(name, attr, comp): ...
ABCSeries = cast(...)

# === Internal dependency: pandas.core.indexes.datetimes ===
class DatetimeIndex(DatetimeTimedeltaMixin): ...

# === Internal dependency: pandas.core.indexes.timedeltas ===
class TimedeltaIndex(DatetimeTimedeltaMixin): ...

# === Internal dependency: pandas.util._exceptions ===
def find_stack_level(): ...