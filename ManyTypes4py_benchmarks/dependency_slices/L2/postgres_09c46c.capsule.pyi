from typing import Any

# === Third-party dependency: flask_babel ===
def gettext(*args, **kwargs) -> str: ...

# === Third-party dependency: psycopg2.extensions ===
# Used symbols: binary_types, string_types

# === Third-party dependency: sqlalchemy ===
# Used symbols: types

# === Third-party dependency: sqlalchemy.dialects.postgresql ===
# Used symbols: DOUBLE_PRECISION, ENUM, INTERVAL, JSON

# === Third-party dependency: sqlalchemy.engine.url ===
class URL(NamedTuple): ...

# === Third-party dependency: sqlalchemy.types ===
# Used symbols: Date, DateTime, String

# === Internal dependency: superset.constants ===
class TimeGrain(StrEnum): ...

# === Internal dependency: superset.db_engine_specs.aws_iam ===
class AWSIAMAuthMixin: ...

# === Internal dependency: superset.db_engine_specs.base ===
class DatabaseCategory: ...
class BaseEngineSpec:
    ...
class BasicParametersMixin:

# === Internal dependency: superset.errors ===
class SupersetErrorType(StrEnum): ...
class ErrorLevel(StrEnum): ...
class SupersetError: ...

# === Internal dependency: superset.exceptions ===
class SupersetException(Exception): ...
class SupersetSecurityException(SupersetErrorException): ...

# === Internal dependency: superset.sql.parse ===
def process_jinja_sql(sql: str, database: Database, template_params: Optional[dict[str, Any]] = ...) -> JinjaSQLResult: ...

# === Internal dependency: superset.utils.core ===
class GenericDataType(IntEnum): ...
def create_ssl_cert_file(certificate: str) -> str: ...

# === Internal dependency: superset.utils.json ===
def loads(obj: Union[bytes, bytearray, str], encoding: Union[str, None] = ..., allow_nan: bool = ..., object_hook: Union[Callable[[dict[Any, Any]], Any], None] = ...) -> Any: ...
# re-export: from simplejson import JSONDecodeError