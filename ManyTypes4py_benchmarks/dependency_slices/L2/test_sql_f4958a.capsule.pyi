from typing import Any

# === Third-party dependency: adbc_driver_manager ===
# Used symbols: ProgrammingError

# === Third-party dependency: adbc_driver_postgresql ===
# Used symbols: dbapi

# === Third-party dependency: adbc_driver_sqlite ===
# Used symbols: dbapi

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, bool_, datetime64, dtype, float32, float64, floating, inf, int32, int64, integer, nan, ones, random, round, str_

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import ArrowDtype
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import isna
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import DateOffset
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import to_timedelta
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat
# re-export: from pandas.io.api import read_sql

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._libs.lib ===
no_default: Final

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.arrays ===
# re-export: from pandas.core.arrays import ArrowExtensionArray

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat.pyarrow import pa_version_under14p1

# === Internal dependency: pandas.compat._optional ===
def import_optional_dependency(name: str, extra: str = ..., min_version: str | None = ..., *, errors: Literal['raise'] = ...) -> types.ModuleType: ...
def import_optional_dependency(name: str, extra: str = ..., min_version: str | None = ..., *, errors: Literal['warn', 'ignore']) -> types.ModuleType | None: ...
def import_optional_dependency(name: str, extra: str = ..., min_version: str | None = ..., *, errors: Literal['raise', 'warn', 'ignore'] = ...) -> types.ModuleType | None: ...

# === Internal dependency: pandas.io.sql ===
def read_sql_table(table_name: str, con, schema = ..., index_col: str | list[str] | None = ..., coerce_float = ..., parse_dates: list[str] | dict[str, str] | None = ..., columns: list[str] | None = ..., chunksize: None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame: ...
def read_sql_table(table_name: str, con, schema = ..., index_col: str | list[str] | None = ..., coerce_float = ..., parse_dates: list[str] | dict[str, str] | None = ..., columns: list[str] | None = ..., chunksize: int = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> Iterator[DataFrame]: ...
def read_sql_table(table_name: str, con, schema: str | None = ..., index_col: str | list[str] | None = ..., coerce_float: bool = ..., parse_dates: list[str] | dict[str, str] | None = ..., columns: list[str] | None = ..., chunksize: int | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame | Iterator[DataFrame]: ...
def read_sql_query(sql, con, index_col: str | list[str] | None = ..., coerce_float = ..., params: list[Any] | Mapping[str, Any] | None = ..., parse_dates: list[str] | dict[str, str] | None = ..., chunksize: None = ..., dtype: DtypeArg | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame: ...
def read_sql_query(sql, con, index_col: str | list[str] | None = ..., coerce_float = ..., params: list[Any] | Mapping[str, Any] | None = ..., parse_dates: list[str] | dict[str, str] | None = ..., chunksize: int = ..., dtype: DtypeArg | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> Iterator[DataFrame]: ...
def read_sql_query(sql, con, index_col: str | list[str] | None = ..., coerce_float: bool = ..., params: list[Any] | Mapping[str, Any] | None = ..., parse_dates: list[str] | dict[str, str] | None = ..., chunksize: int | None = ..., dtype: DtypeArg | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame | Iterator[DataFrame]: ...
def read_sql(sql, con, index_col: str | list[str] | None = ..., coerce_float = ..., params = ..., parse_dates = ..., columns: list[str] = ..., chunksize: None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., dtype: DtypeArg | None = ...) -> DataFrame: ...
def read_sql(sql, con, index_col: str | list[str] | None = ..., coerce_float = ..., params = ..., parse_dates = ..., columns: list[str] = ..., chunksize: int = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., dtype: DtypeArg | None = ...) -> Iterator[DataFrame]: ...
def read_sql(sql, con, index_col: str | list[str] | None = ..., coerce_float: bool = ..., params = ..., parse_dates = ..., columns: list[str] | None = ..., chunksize: int | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., dtype: DtypeArg | None = ...) -> DataFrame | Iterator[DataFrame]: ...
def to_sql(frame, name: str, con, schema: str | None = ..., if_exists: Literal['fail', 'replace', 'append'] = ..., index: bool = ..., index_label: IndexLabel | None = ..., chunksize: int | None = ..., dtype: DtypeArg | None = ..., method: Literal['multi'] | Callable | None = ..., engine: str = ..., **engine_kwargs) -> int | None: ...
def has_table(table_name: str, con, schema: str | None = ...) -> bool: ...
def pandasSQL_builder(con, schema: str | None = ..., need_transaction: bool = ...) -> PandasSQL: ...
class SQLTable(PandasObject):
    def __init__(self, name: str, pandas_sql_engine, frame = ..., index: bool | str | list[str] | None = ..., if_exists: Literal['fail', 'replace', 'append'] = ..., prefix: str = ..., index_label = ..., schema = ..., keys = ..., dtype: DtypeArg | None = ...) -> None: ...
class SQLAlchemyEngine(BaseEngine): ...
def get_engine(engine: str) -> BaseEngine: ...
class SQLDatabase(PandasSQL): ...
def _get_valid_sqlite_name(name: object) -> str: ...
class SQLiteTable(SQLTable):
    def __init__(self, *args, **kwargs) -> None: ...
    def sql_schema(self) -> str: ...
class SQLiteDatabase(PandasSQL):
    def __init__(self, con) -> None: ...
def get_schema(frame, name: str, keys = ..., con = ..., dtype: DtypeArg | None = ..., schema: str | None = ...) -> str: ...
# re-export: from pandas.errors import DatabaseError
table_exists = has_table

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_installed(package: str) -> pytest.MarkDecorator: ...

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises, skip

# === Third-party dependency: sqlalchemy ===
# re-export: from .engine import create_engine as create_engine
# re-export: from .inspection import inspect as inspect
# re-export: from .schema import Column as Column
# re-export: from .schema import MetaData as MetaData
# re-export: from .schema import Table as Table
# re-export: from .sql.expression import bindparam as bindparam
# re-export: from .sql.expression import insert as insert
# re-export: from .sql.expression import select as select
# re-export: from .sql.expression import text as text
# re-export: from .types import BigInteger as BigInteger
# re-export: from .types import Boolean as Boolean
# re-export: from .types import DateTime as DateTime
# re-export: from .types import Double as Double
# re-export: from .types import Float as Float
# re-export: from .types import Integer as Integer
# re-export: from .types import String as String
# re-export: from .types import TEXT as TEXT
# re-export: from .types import TIMESTAMP as TIMESTAMP
# re-export: from .types import Unicode as Unicode
__version__: str

# === Third-party dependency: sqlalchemy.dialects.mysql ===
# Used symbols: insert

# === Third-party dependency: sqlalchemy.dialects.postgresql ===
# Used symbols: insert

# === Third-party dependency: sqlalchemy.engine ===
# Used symbols: Engine

# === Third-party dependency: sqlalchemy.orm ===
# Used symbols: Session, declarative_base, sessionmaker

# === Third-party dependency: sqlalchemy.schema ===
# Used symbols: MetaData

# === Third-party dependency: sqlalchemy.sql ===
# Used symbols: text