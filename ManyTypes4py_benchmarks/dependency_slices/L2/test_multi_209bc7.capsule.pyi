# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, float32, int32, int64, nan, random, tile

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import RangeIndex
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.reshape.concat ===
def concat(objs: Iterable[DataFrame] | Mapping[HashableT, DataFrame], *, axis: Literal[0, 'index'] = ..., join: str = ..., ignore_index: bool = ..., keys: Iterable[Hashable] | None = ..., levels = ..., names: list[HashableT] | None = ..., verify_integrity: bool = ..., sort: bool = ..., copy: bool | lib.NoDefault = ...) -> DataFrame: ...
def concat(objs: Iterable[Series] | Mapping[HashableT, Series], *, axis: Literal[0, 'index'] = ..., join: str = ..., ignore_index: bool = ..., keys: Iterable[Hashable] | None = ..., levels = ..., names: list[HashableT] | None = ..., verify_integrity: bool = ..., sort: bool = ..., copy: bool | lib.NoDefault = ...) -> Series: ...
def concat(objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame], *, axis: Literal[0, 'index'] = ..., join: str = ..., ignore_index: bool = ..., keys: Iterable[Hashable] | None = ..., levels = ..., names: list[HashableT] | None = ..., verify_integrity: bool = ..., sort: bool = ..., copy: bool | lib.NoDefault = ...) -> DataFrame | Series: ...
def concat(objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame], *, axis: Literal[1, 'columns'], join: str = ..., ignore_index: bool = ..., keys: Iterable[Hashable] | None = ..., levels = ..., names: list[HashableT] | None = ..., verify_integrity: bool = ..., sort: bool = ..., copy: bool | lib.NoDefault = ...) -> DataFrame: ...
def concat(objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame], *, axis: Axis = ..., join: str = ..., ignore_index: bool = ..., keys: Iterable[Hashable] | None = ..., levels = ..., names: list[HashableT] | None = ..., verify_integrity: bool = ..., sort: bool = ..., copy: bool | lib.NoDefault = ...) -> DataFrame | Series: ...

# === Internal dependency: pandas.core.reshape.merge ===
def merge(left: DataFrame | Series, right: DataFrame | Series, how: MergeHow = ..., on: IndexLabel | AnyArrayLike | None = ..., left_on: IndexLabel | AnyArrayLike | None = ..., right_on: IndexLabel | AnyArrayLike | None = ..., left_index: bool = ..., right_index: bool = ..., sort: bool = ..., suffixes: Suffixes = ..., copy: bool | lib.NoDefault = ..., indicator: str | bool = ..., validate: str | None = ...) -> DataFrame: ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises