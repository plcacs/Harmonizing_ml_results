# === Third-party dependency: numpy ===
# Used symbols: asarray, nan, ndarray

# === Internal dependency: pandas._libs.lib ===
class _NoDefault(Enum):
    no_default = Ellipsis
no_default = _NoDefault.no_default

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.np_datetime import is_supported_dtype

# === Internal dependency: pandas._typing ===
PythonScalar = Union[str, float, bool]
PandasScalar = Union['Period', 'Timestamp', 'Timedelta', 'Interval']
Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, date]

# === Internal dependency: pandas.compat.numpy.function ===
class CompatValidator:
    def __init__(self, defaults, fname=..., method=..., max_fname_arg_count=...): ...
ALLANY_DEFAULTS = {}
validate_all = CompatValidator(...)
validate_any = CompatValidator(...)
MINMAX_DEFAULTS = {'axis': None, 'dtype': None, 'out': None, 'keepdims': False}
validate_min = CompatValidator(...)
validate_max = CompatValidator(...)
STAT_FUNC_DEFAULTS = {}
SUM_DEFAULTS = STAT_FUNC_DEFAULTS.copy(...)
PROD_DEFAULTS = SUM_DEFAULTS.copy(...)
MEAN_DEFAULTS = SUM_DEFAULTS.copy(...)
MEDIAN_DEFAULTS = STAT_FUNC_DEFAULTS.copy(...)
validate_sum = CompatValidator(...)
validate_prod = CompatValidator(...)
validate_mean = CompatValidator(...)
validate_median = CompatValidator(...)
STAT_DDOF_FUNC_DEFAULTS = {}
validate_stat_ddof_func = CompatValidator(...)

# === Internal dependency: pandas.core.arraylike ===
class OpsMixin:
    ...
def dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs): ...
def dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs): ...
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

# === Internal dependency: pandas.core.arrays ===
from pandas.core.arrays.timedeltas import TimedeltaArray

# === Internal dependency: pandas.core.arrays._mixins ===
class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    def __getitem__(self, key): ...

# === Internal dependency: pandas.core.construction ===
def ensure_wrapped_if_datetimelike(arr): ...

# === Internal dependency: pandas.core.dtypes.astype ===
def astype_array(values, dtype, copy=...): ...

# === Internal dependency: pandas.core.dtypes.cast ===
def construct_1d_object_array_from_listlike(values): ...

# === Internal dependency: pandas.core.dtypes.common ===
def pandas_dtype(dtype): ...

# === Internal dependency: pandas.core.dtypes.dtypes ===
class NumpyEADtype(ExtensionDtype):
    def __init__(self, dtype): ...

# === Internal dependency: pandas.core.missing ===
def clean_fill_method(method, *, allow_nearest=...): ...
def clean_fill_method(method, *, allow_nearest): ...
def interpolate_2d_inplace(data, index, axis, method=..., limit=..., limit_direction=..., limit_area=..., fill_value=..., mask=..., **kwargs): ...
def pad_or_backfill_inplace(values, method=..., axis=..., limit=..., limit_area=...): ...

# === Internal dependency: pandas.core.nanops ===
def nanany(values, *, axis=..., skipna=..., mask=...): ...
def nanall(values, *, axis=..., skipna=..., mask=...): ...
def nansum(values, *, axis=..., skipna=..., min_count=..., mask=...): ...
def nanmean(values, *, axis=..., skipna=..., mask=...): ...
def nanmedian(values, *, axis=..., skipna=..., mask=...): ...
def nanstd(values, *, axis=..., skipna=..., ddof=..., mask=...): ...
def nanvar(values, *, axis=..., skipna=..., ddof=..., mask=...): ...
def nansem(values, *, axis=..., skipna=..., ddof=..., mask=...): ...
def _nanminmax(meth, fill_value_typ): ...
def nanskew(values, *, axis=..., skipna=..., mask=...): ...
def nankurt(values, *, axis=..., skipna=..., mask=...): ...
def nanprod(values, *, axis=..., skipna=..., min_count=..., mask=...): ...
nanmin = _nanminmax(...)
nanmax = _nanminmax(...)

# === Internal dependency: pandas.core.ops ===
from pandas.core.ops.array_ops import get_array_op
from pandas.core.ops.array_ops import maybe_prepare_scalar_for_op
from pandas.core.roperator import rdivmod

# === Internal dependency: pandas.core.strings.object_array ===
class ObjectStringArrayMixin(BaseStringArrayMethods):
    ...