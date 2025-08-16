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
        if self.greenlet:
            raise RuntimeError(f'Greenlet {self.greenlet!r} already started')
        pristine: bool = not self.greenlet.dead and tuple(self.greenlet.args) == tuple(self.args) and (self.greenlet.kwargs == self.kwargs)
        if not pristine:
            self.greenlet = Greenlet(self._run, *self.args, **self.kwargs)
            self.greenlet.name = f'{self.__class__.__name__}|{self.greenlet.name}'
        self.greenlet.start()

    def _run(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def on_error(self, subtask: Greenlet) -> None:
        log.error('Runnable subtask died!', this=self, running=bool(self), subtask=subtask, exc=subtask.exception)
        if not self.greenlet:
            return
        exception = subtask.exception or GreenletExit()
        self.greenlet.kill(exception)

    def _schedule_new_greenlet(self, func: Callable, *args: Any, in_seconds_from_now: float = None, **kwargs: Any) -> Greenlet:
        def on_success(greenlet: Greenlet) -> None:
            if greenlet in self.greenlets:
                self.greenlets.remove(greenlet)
        greenlet = Greenlet(func, *args, **kwargs)
        greenlet.name = f'Greenlet<fn:{func.__name__}>'
        greenlet.link_exception(self.on_error)
        greenlet.link_value(on_success)
        self.greenlets.append(greenlet)
        if in_seconds_from_now:
            greenlet.start_later(in_seconds_from_now)
        else:
            greenlet.start()
        return greenlet

    def __bool__(self) -> bool:
        return bool(self.greenlet)

    def is_running(self) -> bool:
        return bool(self.greenlet)

    def rawlink(self, callback: Callable) -> None:
        if not self.greenlet:
            return
        self.greenlet.rawlink(callback)
