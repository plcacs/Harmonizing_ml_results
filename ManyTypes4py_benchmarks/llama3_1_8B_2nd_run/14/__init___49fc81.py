from importlib import import_module
from typing import Any, Dict, Type

class Base:
    pass

def load_alarm_model(model: str) -> Type:
    try:
        return import_module(f'alerta.models.alarms.{model.lower()}')
    except Exception:
        raise ImportError(f'Failed to load {model} alarm model')

class AlarmModel(Base):
    name: str | None = None
    Severity: Dict[str, str] = {}
    Colors: Dict[str, str] = {}
    Status: Dict[str, str] = {}
    DEFAULT_STATUS: str | None = None
    DEFAULT_NORMAL_SEVERITY: str | None = None
    DEFAULT_PREVIOUS_SEVERITY: str | None = None
    NORMAL_SEVERITY_LEVEL: str | None = None

    def __init__(self, app: Any | None = None) -> None:
        self.app: Any | None = None
        if app is not None:
            self.register(app)

    def init_app(self, app: Any) -> None:
        cls = load_alarm_model(app.config['ALARM_MODEL'])
        self.__class__ = type('AlarmModelImpl', (cls.StateMachine, AlarmModel), {})
        self.register(app)

    def register(self, app: Any) -> None:
        raise NotImplementedError

    def trend(self, previous: Any, current: Any) -> Any:
        raise NotImplementedError

    def transition(self, alert: Any, current_status: str | None = None, previous_status: str | None = None, action: str | None = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_suppressed(alert: Any) -> bool:
        raise NotImplementedError
