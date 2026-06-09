from typing import Any

# === Internal dependency: alerta.app ===
def create_app(config_override: Dict[str, Any] = ..., environment: str = ...) -> Flask: ...
alarm_model: AlarmModel
db: Database
plugins: Plugins

# === Internal dependency: alerta.database.base ===
class Database(Base): ...

# === Unresolved dependency: alerta.models.alarms ===
# Used unresolved symbols: AlarmModel

# === Internal dependency: alerta.models.alert ===
class Alert:
    def __init__(self, resource: str, event: str, **kwargs) -> None: ...
    def get_body(self, history: bool = ...) -> Dict[str, Any]: ...

# === Internal dependency: alerta.plugins ===
class PluginBase:
    def pre_receive(self, alert: 'Alert', **kwargs) -> 'Alert': ...
    def post_receive(self, alert: 'Alert', **kwargs) -> Optional['Alert']: ...
    def status_change(self, alert: 'Alert', status: str, text: str, **kwargs) -> Any: ...

# === Internal dependency: alerta.utils.api ===
def process_alert(alert: Alert) -> Alert: ...

# === Internal dependency: alerta.utils.plugin ===
class Plugins: ...