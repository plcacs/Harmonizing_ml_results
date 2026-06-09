from typing import Any

# === Internal dependency: io ===
BytesIO: Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, bool_, datetime64, double, dtype, finfo, float32, float64, inf, int16, int32, int64, int8, nan, random, reshape, uint16, uint32, uint64, uint8, zeros

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Int8Dtype
# re-export: from pandas.core.api import Int16Dtype
# re-export: from pandas.core.api import Int32Dtype
# re-export: from pandas.core.api import Int64Dtype
# re-export: from pandas.core.api import UInt8Dtype
# re-export: from pandas.core.api import UInt16Dtype
# re-export: from pandas.core.api import UInt32Dtype
# re-export: from pandas.core.api import UInt64Dtype
# re-export: from pandas.core.api import CategoricalDtype
# re-export: from pandas.core.api import BooleanDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import RangeIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._io import round_trip_pathlib
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_dict_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal
# re-export: from pandas._testing.contexts import decompress_file

# === Internal dependency: pandas.core.frame ===
class DataFrame(NDFrame, OpsMixin):
    def __init__(self, data = ..., index: Axes | None = ..., columns: Axes | None = ..., dtype: Dtype | None = ..., copy: bool | None = ...) -> None: ...
    def to_stata(self, path: FilePath | WriteBuffer[bytes], *, convert_dates: dict[Hashable, str] | None = ..., write_index: bool = ..., byteorder: ToStataByteorder | None = ..., time_stamp: datetime.datetime | None = ..., data_label: str | None = ..., variable_labels: dict[Hashable, str] | None = ..., version: int | None = ..., convert_strl: Sequence[Hashable] | None = ..., compression: CompressionOptions = ..., storage_options: StorageOptions | None = ..., value_labels: dict[Hashable, dict[float, str]] | None = ...) -> None: ...
    def rename(self, mapper: Renamer | None = ..., *, index: Renamer | None = ..., columns: Renamer | None = ..., axis: Axis | None = ..., copy: bool | lib.NoDefault = ..., inplace: Literal[True], level: Level = ..., errors: IgnoreRaise = ...) -> None: ...
    def rename(self, mapper: Renamer | None = ..., *, index: Renamer | None = ..., columns: Renamer | None = ..., axis: Axis | None = ..., copy: bool | lib.NoDefault = ..., inplace: Literal[False] = ..., level: Level = ..., errors: IgnoreRaise = ...) -> DataFrame: ...
    def rename(self, mapper: Renamer | None = ..., *, index: Renamer | None = ..., columns: Renamer | None = ..., axis: Axis | None = ..., copy: bool | lib.NoDefault = ..., inplace: bool = ..., level: Level = ..., errors: IgnoreRaise = ...) -> DataFrame | None: ...
    def rename(self, mapper: Renamer | None = ..., *, index: Renamer | None = ..., columns: Renamer | None = ..., axis: Axis | None = ..., copy: bool | lib.NoDefault = ..., inplace: bool = ..., level: Level | None = ..., errors: IgnoreRaise = ...) -> DataFrame | None: ...
    def set_index(self, keys, *, drop: bool = ..., append: bool = ..., inplace: Literal[False] = ..., verify_integrity: bool = ...) -> DataFrame: ...
    def set_index(self, keys, *, drop: bool = ..., append: bool = ..., inplace: Literal[True], verify_integrity: bool = ...) -> None: ...
    def set_index(self, keys, *, drop: bool = ..., append: bool = ..., inplace: bool = ..., verify_integrity: bool = ...) -> DataFrame | None: ...
# re-export: from pandas.core.series import Series

# === Internal dependency: pandas.io.stata ===
class StataMissingValue:
    def __init__(self, value: float) -> None: ...
    def string(self) -> str: ...
class StataReader(StataParser, abc.Iterator): ...
def read_stata(filepath_or_buffer: FilePath | ReadBuffer[bytes], *, convert_dates: bool = ..., convert_categoricals: bool = ..., index_col: str | None = ..., convert_missing: bool = ..., preserve_dtypes: bool = ..., columns: Sequence[str] | None = ..., order_categoricals: bool = ..., chunksize: int | None = ..., iterator: bool = ..., compression: CompressionOptions = ..., storage_options: StorageOptions | None = ...) -> DataFrame | StataReader: ...
class StataWriter(StataParser):
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None = ..., write_index: bool = ..., byteorder: str | None = ..., time_stamp: datetime | None = ..., data_label: str | None = ..., variable_labels: dict[Hashable, str] | None = ..., compression: CompressionOptions = ..., storage_options: StorageOptions | None = ..., *, value_labels: dict[Hashable, dict[float, str]] | None = ...) -> None: ...
    def write_file(self) -> None: ...
class StataWriterUTF8(StataWriter117):
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None = ..., write_index: bool = ..., byteorder: str | None = ..., time_stamp: datetime | None = ..., data_label: str | None = ..., variable_labels: dict[Hashable, str] | None = ..., convert_strl: Sequence[Hashable] | None = ..., version: int | None = ..., compression: CompressionOptions = ..., storage_options: StorageOptions | None = ..., *, value_labels: dict[Hashable, dict[float, str]] | None = ...) -> None: ...
# re-export: from pandas.errors import CategoricalConversionWarning
# re-export: from pandas.errors import InvalidColumnName
# re-export: from pandas.errors import PossiblePrecisionLoss
# re-export: from pandas.errors import ValueLabelTypeMismatch

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises