from importlib import import_module
from typing import Any, Dict, Type, TypeVar

T = TypeVar('T')

class Base:
    pass

def load_alarm_model(model: str) -> T:
    try:
        return import_module(f'alerta.models.alarms.{model.lower()}')
    except Exception:
        raise ImportError(f'Failed to load {model} alarm model')

class AlarmModel(Base):
    name: str = None
    Severity: Dict[str, Any] = {}
    Colors: Dict[str, Any] = {}
    Status: Dict[str, Any] = {}
    DEFAULT_STATUS: str = None
    DEFAULT_NORMAL_SEVERITY: str = None
    DEFAULT_PREVIOUS_SEVERITY: str = None
    NORMAL_SEVERITY_LEVEL: str = None

    def __init__(self, app: Any = None) -> None:
        self.app = None
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

    def transition(self, alert: Any, current_status: Any = None, previous_status: Any = None, action: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_suppressed(alert: Any) -> bool:
        raise NotImplementedError
