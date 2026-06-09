# === Internal dependency: alerta.app ===
alarm_model = AlarmModel(...)

# === Unresolved dependency: alerta.database.backends.mongodb.utils ===
# Used unresolved symbols: Query

# === Internal dependency: alerta.database.base ===
class Database(Base):
    def name(self): ...
    def version(self): ...
    def is_alive(self): ...

# === Internal dependency: alerta.exceptions ===
class NoCustomerMatch(BaseError): ...

# === Unresolved dependency: alerta.models.alarms ===
# Used unresolved symbols: AlarmModel

# === Internal dependency: alerta.models.enums ===
class Scope(str):
    read = 'read'
    write = 'write'
    admin = 'admin'
ADMIN_SCOPES = [Scope.admin, Scope.read, Scope.write]

# === Internal dependency: alerta.models.heartbeat ===
class HeartbeatStatus(StrEnum): ...

# === Internal dependency: alerta.utils.collections ===
def merge(dict1, dict2): ...

# === Third-party dependency: flask ===
# Used symbols: current_app

# === Third-party dependency: pymongo ===
ASCENDING: int
TEXT: str
# re-export: from pymongo.collection import ReturnDocument
# re-export: from pymongo.mongo_client import MongoClient

# === Third-party dependency: pymongo.errors ===
class ConnectionFailure(PyMongoError): ...