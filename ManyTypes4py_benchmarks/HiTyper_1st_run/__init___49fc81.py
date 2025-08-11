from importlib import import_module

class Base:
    pass

def load_alarm_model(model: str) -> Union[str, dict[str, typing.Any], typing.Callable[str, None]]:
    try:
        return import_module(f'alerta.models.alarms.{model.lower()}')
    except Exception:
        raise ImportError(f'Failed to load {model} alarm model')

class AlarmModel(Base):
    name = None
    Severity = {}
    Colors = {}
    Status = {}
    DEFAULT_STATUS = None
    DEFAULT_NORMAL_SEVERITY = None
    DEFAULT_PREVIOUS_SEVERITY = None
    NORMAL_SEVERITY_LEVEL = None

    def __init__(self, app: None=None) -> None:
        self.app = None
        if app is not None:
            self.register(app)

    def init_app(self, app: Any) -> None:
        cls = load_alarm_model(app.config['ALARM_MODEL'])
        self.__class__ = type('AlarmModelImpl', (cls.StateMachine, AlarmModel), {})
        self.register(app)

    def register(self, app: Any) -> None:
        raise NotImplementedError

    def trend(self, previous: Union[bool, typing.Callable[str, object]], current: Union[bool, typing.Callable[str, object]]) -> None:
        raise NotImplementedError

    def transition(self, alert: Union[int, str, None], current_status: Union[None, int, str]=None, previous_status: Union[None, int, str]=None, action: Union[None, int, str]=None, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def is_suppressed(alert: Union[list[tuple[str]], list[dict[str, typing.Any]]]) -> None:
        raise NotImplementedError