from typing import Any

# === Third-party dependency: flask ===
# Used symbols: g, has_request_context, request

# === Third-party dependency: flask_appbuilder.const ===
API_URI_RIS_KEY: str

# === Third-party dependency: sqlalchemy.exc ===
class SQLAlchemyError(HasDescriptionCode, Exception): ...

# === Internal dependency: superset ===
# re-export: from superset.extensions import db

# === Internal dependency: superset.extensions ===
stats_logger_manager: BaseStatsLoggerManager

# === Internal dependency: superset.extensions.stats_logger ===
class BaseStatsLoggerManager: ...

# === Internal dependency: superset.models.core ===
class Log(Model):
    ...

# === Internal dependency: superset.utils.core ===
class LoggerLevel(StrEnum): ...
def get_user_id() -> int | None: ...
def to_int(v: Any, value_if_invalid: int = ...) -> int: ...

# === Internal dependency: superset.utils.json ===
def dumps(obj: Any, default: Optional[Callable[[Any], Any]] = ..., allow_nan: bool = ..., ignore_nan: bool = ..., sort_keys: bool = ..., indent: Union[str, int, None] = ..., separators: Union[tuple[str, str], None] = ..., cls: Union[type[simplejson.JSONEncoder], None] = ..., encoding: Optional[str] = ...) -> str: ...
def loads(obj: Union[bytes, bytearray, str], encoding: Union[str, None] = ..., allow_nan: bool = ..., object_hook: Union[Callable[[dict[Any, Any]], Any], None] = ...) -> Any: ...

# === Internal dependency: superset.views.core ===
# re-export: from superset.views.utils import get_form_data