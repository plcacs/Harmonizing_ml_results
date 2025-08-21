from importlib import import_module
from types import ModuleType
from typing import Any, Dict, Optional, Protocol, Mapping


class Base:
    pass


class AppLike(Protocol):
    config: Mapping[str, Any]


def load_alarm_model(model: str) -> ModuleType:
    try:
        return import_module(f'alerta.models.alarms.{model.lower()}')
    except Exception:
        raise ImportError(f'Failed to load {model} alarm model')


class AlarmModel(Base):
    name: Optional[str] = None
    Severity: Dict[str, Any] = {}
    Colors: Dict[str, Any] = {}
    Status: Dict[str, Any] = {}
    DEFAULT_STATUS: Optional[str] = None
    DEFAULT_NORMAL_SEVERITY: Optional[str] = None
    DEFAULT_PREVIOUS_SEVERITY: Optional[str] = None
    NORMAL_SEVERITY_LEVEL: Optional[int] = None

    def __init__(self, app: Optional[AppLike] = None) -> None:
        self.app: Optional[AppLike] = None
        if app is not None:
            self.register(app)

    def init_app(self, app: AppLike) -> None:
        cls: ModuleType = load_alarm_model(app.config['ALARM_MODEL'])
        self.__class__ = type('AlarmModelImpl', (cls.StateMachine, AlarmModel), {})  # type: ignore[misc, assignment]
        self.register(app)

    def register(self, app: AppLike) -> None:
        raise NotImplementedError

    def trend(self, previous: Any, current: Any) -> str:
        raise NotImplementedError

    def transition(
        self,
        alert: Any,
        current_status: Optional[str] = None,
        previous_status: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_suppressed(alert: Any) -> bool:
        raise NotImplementedError