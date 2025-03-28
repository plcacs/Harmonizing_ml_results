from typing import Any, Sequence, Dict, Callable, List, Optional
import structlog
from gevent import Greenlet, GreenletExit
log = structlog.get_logger(__name__)


class Runnable:
    """Greenlet-like class, __run() inside one, but can be stopped and restarted

    Allows subtasks to crash self, and bubble up the exception in the greenlet
    In the future, when proper restart is implemented, may be replaced by actual greenlet
    """
    greenlet: Greenlet
    args: Sequence[Any] = ()
    kwargs: Dict[str, Any] = {}
    greenlets: List[Greenlet]

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.greenlet = Greenlet(self._run, *self.args, **self.kwargs)
        self.greenlet.name = f'{self.__class__.__name__}|{self.greenlet.name}'
        self.greenlets = []

    def start(self):
        """Synchronously start task

        Reimplements in children a call super().start() at end to start _run()
        Start-time exceptions may be raised
        """
        if self.greenlet:
            raise RuntimeError(f'Greenlet {self.greenlet!r} already started')
        pristine = not self.greenlet.dead and tuple(self.greenlet.args
            ) == self.args and self.greenlet.kwargs == self.kwargs
        if not pristine:
            self.greenlet = Greenlet(self._run, *self.args, **self.kwargs)
            self.greenlet.name = (
                f'{self.__class__.__name__}|{self.greenlet.name}')
        self.greenlet.start()

    def _run(self, *args: Any, **kwargs: Any):
        """Reimplements in children to busy wait here

        This busy wait should be finished gracefully after stop(),
        or be killed and re-raise on subtasks exception"""
        raise NotImplementedError

    def stop(self):
        """Synchronous stop, gracefully tells _run() to exit

        Should wait subtasks to finish.
        Stop-time exceptions may be raised, run exceptions should not (accessible via get())
        """
        raise NotImplementedError

    def on_error(self, subtask):
        """Default callback for subtasks link_exception

        Default callback re-raises the exception inside _run()"""
        log.error('Runnable subtask died!', this=self, running=bool(self),
            subtask=subtask, exc=subtask.exception)
        if not self.greenlet:
            return
        exception = subtask.exception or GreenletExit()
        self.greenlet.kill(exception)

    def _schedule_new_greenlet(self, func, *args: Any, in_seconds_from_now:
        Optional[int]=None, **kwargs: Any):
        """Spawn a sub-task and ensures an error on it crashes self/main greenlet"""

        def on_success(greenlet):
            if greenlet in self.greenlets:
                self.greenlets.remove(greenlet)
        greenlet = Greenlet(func, *args, **kwargs)
        greenlet.name = f'Greenlet<fn:{func.__name__}>'
        greenlet.link_exception(self.on_error)
        greenlet.link_value(on_success)
        self.greenlets.append(greenlet)
        if in_seconds_from_now is not None:
            greenlet.start_later(in_seconds_from_now)
        else:
            greenlet.start()
        return greenlet

    def __bool__(self):
        return bool(self.greenlet)

    def is_running(self):
        return bool(self.greenlet)

    def rawlink(self, callback):
        if not self.greenlet:
            return
        self.greenlet.rawlink(callback)
