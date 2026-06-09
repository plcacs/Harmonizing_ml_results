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
from pandas._config import option_context
from pandas.core.api import ArrowDtype
from pandas.core.api import StringDtype
from pandas.core.api import NA
from pandas.core.api import isna
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import DateOffset
from pandas.core.api import to_datetime
from pandas.core.api import to_timedelta
from pandas.core.api import array
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas.io.api import read_sql

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._libs.lib ===
class _NoDefault(Enum):
    no_default = Ellipsis
no_default = _NoDefault.no_default

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal
from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.arrays ===
from pandas.core.arrays import ArrowExtensionArray

# === Internal dependency: pandas.compat ===
from pandas.compat.pyarrow import pa_version_under14p1

# === Internal dependency: pandas.compat._optional ===
def import_optional_dependency(name, extra=..., min_version=..., *, errors=...): ...
def import_optional_dependency(name, extra=..., min_version=..., *, errors): ...

# === Internal dependency: pandas.io.sql ===
def read_sql_table(table_name, con, schema=..., index_col=..., coerce_float=..., parse_dates=..., columns=..., chunksize=..., dtype_backend=...): ...
def read_sql_query(sql, con, index_col=..., coerce_float=..., params=..., parse_dates=..., chunksize=..., dtype=..., dtype_backend=...): ...
def read_sql(sql, con, index_col=..., coerce_float=..., params=..., parse_dates=..., columns=..., chunksize=..., dtype_backend=..., dtype=...): ...
def to_sql(frame, name, con, schema=..., if_exists=..., index=..., index_label=..., chunksize=..., dtype=..., method=..., engine=..., **engine_kwargs): ...
def has_table(table_name, con, schema=...): ...
def pandasSQL_builder(con, schema=..., need_transaction=...): ...
class SQLTable(PandasObject):
    def __init__(self, name, pandas_sql_engine, frame=..., index=..., if_exists=..., prefix=..., index_label=..., schema=..., keys=..., dtype=...): ...
class SQLAlchemyEngine(BaseEngine): ...
def get_engine(engine): ...
class SQLDatabase(PandasSQL): ...
def _get_valid_sqlite_name(name): ...
class SQLiteTable(SQLTable):
    def __init__(self, *args, **kwargs): ...
    def sql_schema(self): ...
class SQLiteDatabase(PandasSQL):
    def __init__(self, con): ...
def get_schema(frame, name, keys=..., con=..., dtype=..., schema=...): ...
from pandas.errors import DatabaseError
table_exists = has_table

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_installed(package): ...

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