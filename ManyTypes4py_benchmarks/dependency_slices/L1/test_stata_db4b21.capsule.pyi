from typing import Any

# === Internal dependency: io ===
BytesIO: Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, bool_, datetime64, double, dtype, finfo, float32, float64, inf, int16, int32, int64, int8, nan, random, reshape, uint16, uint32, uint64, uint8, zeros

# === Internal dependency: pandas ===
from pandas.core.api import Int8Dtype
from pandas.core.api import Int16Dtype
from pandas.core.api import Int32Dtype
from pandas.core.api import Int64Dtype
from pandas.core.api import UInt8Dtype
from pandas.core.api import UInt16Dtype
from pandas.core.api import UInt32Dtype
from pandas.core.api import UInt64Dtype
from pandas.core.api import CategoricalDtype
from pandas.core.api import BooleanDtype
from pandas.core.api import NA
from pandas.core.api import Index
from pandas.core.api import RangeIndex
from pandas.core.api import NaT
from pandas.core.api import date_range
from pandas.core.api import Categorical
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._io import round_trip_pathlib
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_dict_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal
from pandas._testing.contexts import decompress_file

# === Internal dependency: pandas.core.frame ===
class DataFrame(NDFrame, OpsMixin):
    def __init__(self, data=..., index=..., columns=..., dtype=..., copy=...): ...
    def to_stata(self, path, *, convert_dates=..., write_index=..., byteorder=..., time_stamp=..., data_label=..., variable_labels=..., version=..., convert_strl=..., compression=..., storage_options=..., value_labels=...): ...
    def rename(self, mapper=..., *, index=..., columns=..., axis=..., copy=..., inplace, level=..., errors=...): ...
    def rename(self, mapper=..., *, index=..., columns=..., axis=..., copy=..., inplace=..., level=..., errors=...): ...
    def set_index(self, keys, *, drop=..., append=..., inplace=..., verify_integrity=...): ...
    def set_index(self, keys, *, drop=..., append=..., inplace, verify_integrity=...): ...
from pandas.core.series import Series

# === Internal dependency: pandas.io.stata ===
class StataMissingValue:
    def __init__(self, value): ...
    def string(self): ...
class StataReader(StataParser, abc.Iterator): ...
def read_stata(filepath_or_buffer, *, convert_dates=..., convert_categoricals=..., index_col=..., convert_missing=..., preserve_dtypes=..., columns=..., order_categoricals=..., chunksize=..., iterator=..., compression=..., storage_options=...): ...
class StataWriter(StataParser):
    def __init__(self, fname, data, convert_dates=..., write_index=..., byteorder=..., time_stamp=..., data_label=..., variable_labels=..., compression=..., storage_options=..., *, value_labels=...): ...
    def write_file(self): ...
class StataWriterUTF8(StataWriter117):
    def __init__(self, fname, data, convert_dates=..., write_index=..., byteorder=..., time_stamp=..., data_label=..., variable_labels=..., convert_strl=..., version=..., compression=..., storage_options=..., *, value_labels=...): ...
from pandas.errors import CategoricalConversionWarning
from pandas.errors import InvalidColumnName
from pandas.errors import PossiblePrecisionLoss
from pandas.errors import ValueLabelTypeMismatch

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package, min_version=...): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises