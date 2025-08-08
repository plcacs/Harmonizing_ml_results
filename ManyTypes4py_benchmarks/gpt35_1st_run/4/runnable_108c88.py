from typing import Any, Sequence, Callable, List
import structlog
from gevent import Greenlet, GreenletExit

log: structlog.BoundLogger = structlog.get_logger(__name__)

class Runnable:
    args: Sequence[Any] = ()
    kwargs: dict = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.greenlet: Greenlet = Greenlet(self._run, *self.args, **self.kwargs)
        self.greenlet.name = f'{self.__class__.__name__}|{self.greenlet.name}'
        self.greenlets: List[Greenlet] = []

    def start(self) -> None:
        ...

    def _run(self, *args: Any, **kwargs: Any) -> None:
        ...

    def stop(self) -> None:
        ...

    def on_error(self, subtask: Greenlet) -> None:
        ...

    def _schedule_new_greenlet(self, func: Callable, *args: Any, in_seconds_from_now: float = None, **kwargs: Any) -> Greenlet:
        ...

    def __bool__(self) -> bool:
        ...

    def is_running(self) -> bool:
        ...

    def rawlink(self, callback: Callable) -> None:
        ...
