from importlib import import_module
from typing import Any, Optional
from types import ModuleType

class Base:
    pass

def load_alarm_model(model: str) -> ModuleType:
    try:
        return import_module(f'alerta.models.alarms.{model.lower()}')
    except Exception:
        raise ImportError(f'Failed to load {model} alarm model')

class AlarmModel(Base):
    name: Optional[str] = None
    Severity: dict = {}
    Colors: dict = {}
    Status: dict = {}
    DEFAULT_STATUS: Optional[Any] = None
    DEFAULT_NORMAL_SEVERITY: Optional[Any] = None
    DEFAULT_PREVIOUS_SEVERITY: Optional[Any] = None
    NORMAL_SEVERITY_LEVEL: Optional[Any] = None

    def __init__(self, app: Optional[Any] = None) -> None:
        self.app: Optional[Any] = None
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

    def transition(
        self,
        alert: Any,
        current_status: Optional[Any] = None,
        previous_status: Optional[Any] = None,
        action: Optional[Any] = None,
        **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_suppressed(alert: Any) -> Any:
        raise NotImplementedError
