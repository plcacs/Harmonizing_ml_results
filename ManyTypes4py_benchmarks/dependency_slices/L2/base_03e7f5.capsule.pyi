from typing import Any

# === Internal dependency: alerta.app ===
alarm_model: AlarmModel

# === Unresolved dependency: alerta.database.backends.mongodb.utils ===
# Used unresolved symbols: Query

# === Internal dependency: alerta.database.base ===
class Database(Base):
    def name(self) -> Any: ...
    def version(self) -> Any: ...
    def is_alive(self) -> Any: ...

# === Internal dependency: alerta.exceptions ===
class NoCustomerMatch(BaseError): ...

# === Unresolved dependency: alerta.models.alarms ===
# Used unresolved symbols: AlarmModel

# === Internal dependency: alerta.models.enums ===
ADMIN_SCOPES: Any

# === Internal dependency: alerta.models.heartbeat ===
class HeartbeatStatus(StrEnum): ...

# === Internal dependency: alerta.utils.collections ===
def merge(dict1, dict2) -> Any: ...

# === Third-party dependency: flask ===
# Used symbols: current_app

# === Third-party dependency: pymongo ===
ASCENDING: int
TEXT: str
# re-export: from pymongo.collection import ReturnDocument
# re-export: from pymongo.mongo_client import MongoClient

# === Third-party dependency: pymongo.errors ===
class ConnectionFailure(PyMongoError): ...