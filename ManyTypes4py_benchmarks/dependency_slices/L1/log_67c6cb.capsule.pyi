# === Third-party dependency: flask ===
# Used symbols: g, has_request_context, request

# === Third-party dependency: flask_appbuilder.const ===
API_URI_RIS_KEY: str

# === Third-party dependency: sqlalchemy.exc ===
class SQLAlchemyError(HasDescriptionCode, Exception): ...

# === Internal dependency: superset ===
from superset.extensions import db

# === Internal dependency: superset.extensions ===
stats_logger_manager = BaseStatsLoggerManager(...)

# === Internal dependency: superset.extensions.stats_logger ===
class BaseStatsLoggerManager: ...

# === Internal dependency: superset.models.core ===
class Log(Model):
    ...

# === Internal dependency: superset.utils.core ===
class LoggerLevel(StrEnum): ...
def get_user_id(): ...
def to_int(v, value_if_invalid=...): ...

# === Internal dependency: superset.utils.json ===
def dumps(obj, default=..., allow_nan=..., ignore_nan=..., sort_keys=..., indent=..., separators=..., cls=..., encoding=...): ...
def loads(obj, encoding=..., allow_nan=..., object_hook=...): ...

# === Internal dependency: superset.views.core ===
from superset.views.utils import get_form_data