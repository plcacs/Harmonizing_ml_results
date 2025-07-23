from importlib import import_module
from typing import Optional, Type

class Base:
    pass

def load_alarm_model(model: str) -> Type:
    try:
        return import_module(f'alerta.models.alarms.{model.lower()}')
    except Exception:
        raise ImportError(f'Failed to load {model} alarm model')

class AlarmModel(Base):
    name: Optional[str] = None
    Severity: dict = {}
    Colors: dict = {}
    Status: dict = {}
    DEFAULT_STATUS: Optional[str] = None
    DEFAULT_NORMAL_SEVERITY: Optional[str] = None
    DEFAULT_PREVIOUS_SEVERITY: Optional[str] = None
    NORMAL_SEVERITY_LEVEL: Optional[int] = None

    def __init__(self, app: Optional[object] = None) -> None:
        self.app: Optional[object] = None
        if app is not None:
            self.register(app)

    def init_app(self, app: object) -> None:
        cls = load_alarm_model(app.config['ALARM_MODEL'])
        self.__class__ = type('AlarmModelImpl', (cls.StateMachine, AlarmModel), {})
        self.register(app)

    def register(self, app: object) -> None:
        raise NotImplementedError

    def trend(self, previous: str, current: str) -> str:
        raise NotImplementedError

    def transition(self, alert: object, current_status: Optional[str] = None, previous_status: Optional[str] = None, action: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    @staticmethod
    def is_suppressed(alert: object) -> bool:
        raise NotImplementedError
