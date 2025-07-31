from typing import Any, Sequence, Optional
import structlog
from gevent import Greenlet, GreenletExit
from raiden.utils.typing import Callable, List

log = structlog.get_logger(__name__)


class Runnable:
    """Greenlet-like class, __run() inside one, but can be stopped and restarted

    Allows subtasks to crash self, and bubble up the exception in the greenlet
    In the future, when proper restart is implemented, may be replaced by actual greenlet
    """
    args: Sequence[Any] = tuple()
    kwargs: dict[str, Any] = {}  # type: ignore

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.greenlet: Optional[Greenlet[Any]] = Greenlet(self._run, *self.args, **self.kwargs)
        if self.greenlet is not None:  # safety check, should always be True
            self.greenlet.name = f'{self.__class__.__name__}|{self.greenlet.name}'
        self.greenlets: List[Greenlet[Any]] = []

    def start(self) -> None:
        """Synchronously start task

        Reimplements in children a call super().start() at end to start _run()
        Start-time exceptions may be raised
        """
        if self.greenlet:
            raise RuntimeError(f'Greenlet {self.greenlet!r} already started')
        pristine = not self.greenlet.dead and tuple(self.greenlet.args) == tuple(self.args) and (self.greenlet.kwargs == self.kwargs)  # type: ignore
        if not pristine:
            self.greenlet = Greenlet(self._run, *self.args, **self.kwargs)
            self.greenlet.name = f'{self.__class__.__name__}|{self.greenlet.name}'
        self.greenlet.start()  # type: ignore

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Reimplements in children to busy wait here

        This busy wait should be finished gracefully after stop(),
        or be killed and re-raise on subtasks exception"""
        raise NotImplementedError

    def stop(self) -> None:
        """Synchronous stop, gracefully tells _run() to exit

        Should wait subtasks to finish.
        Stop-time exceptions may be raised, run exceptions should not (accessible via get())
        """
        raise NotImplementedError

    def on_error(self, subtask: Greenlet[Any]) -> None:
        """Default callback for substasks link_exception

        Default callback re-raises the exception inside _run()"""
        log.error('Runnable subtask died!',
                  this=self,
                  running=bool(self),
                  subtask=subtask,
                  exc=subtask.exception)
        if not self.greenlet:
            return
        exception: BaseException = subtask.exception or GreenletExit()  # type: ignore
        self.greenlet.kill(exception)

    def _schedule_new_greenlet(self, func: Callable[..., Any], *args: Any, in_seconds_from_now: Optional[float] = None, **kwargs: Any) -> Greenlet[Any]:
        """Spawn a sub-task and ensures an error on it crashes self/main greenlet"""

        def on_success(greenlet: Greenlet[Any]) -> None:
            if greenlet in self.greenlets:
                self.greenlets.remove(greenlet)

        greenlet: Greenlet[Any] = Greenlet(func, *args, **kwargs)
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

    def rawlink(self, callback: Callable[[Greenlet[Any]], Any]) -> None:
        if not self.greenlet:
            return
        self.greenlet.rawlink(callback)