from importlib import import_module
from typing import Any, Optional, Dict, Type


class Base:
    pass


def load_alarm_model(model: str) -> Any:
    try:
        return import_module(f'alerta.models.alarms.{model.lower()}')
    except Exception:
        raise ImportError(f'Failed to load {model} alarm model')


class AlarmModel(Base):
    name: Optional[str] = None
    Severity: Dict[Any, Any] = {}
    Colors: Dict[Any, Any] = {}
    Status: Dict[Any, Any] = {}
    DEFAULT_STATUS: Any = None
    DEFAULT_NORMAL_SEVERITY: Any = None
    DEFAULT_PREVIOUS_SEVERITY: Any = None
    NORMAL_SEVERITY_LEVEL: Any = None

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
    def is_suppressed(alert: Any) -> bool:
        raise NotImplementedError