from importlib import import_module
from typing import Optional, Dict, Any, Type, ModuleType

class Base:
    pass

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
    DEFAULT_NORMAL_SEVERITY: Optional[int] = None
    DEFAULT_PREVIOUS_SEVERITY: Optional[int] = None
    NORMAL_SEVERITY_LEVEL: Optional[int] = None

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
        current_status: Optional[str] = None, 
        previous_status: Optional[str] = None, 
        action: Optional[str] = None, 
        **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_suppressed(alert: Any) -> bool:
        raise NotImplementedError
