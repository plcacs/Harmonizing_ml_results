"""An I/O event loop for non-blocking sockets.

In Tornado 6.0, `.IOLoop` is a wrapper around the `asyncio` event loop, with a
slightly different interface. The `.IOLoop` interface is now provided primarily
for backwards compatibility; new code should generally use the `asyncio` event
loop interface directly. The `IOLoop.current` class method provides the
`IOLoop` instance corresponding to the running `asyncio` event loop.

"""
import asyncio
import concurrent.futures
import datetime
import functools
import numbers
import os
import sys
import time
import math
import random
import warnings
from inspect import isawaitable
from tornado.concurrent import Future, is_future, chain_future, future_set_exc_info, future_add_done_callback
from tornado.log import app_log
from tornado.util import Configurable, TimeoutError, import_object
import typing
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable, Dict, List, Set, cast
from typing_extensions import Protocol

class _Selectable(Protocol):
    def fileno(self) -> int: ...
    def close(self) -> None: ...

_T = TypeVar('_T')
_S = TypeVar('_S', bound=_Selectable)

class IOLoop(Configurable):
    """An I/O event loop."""
    NONE: int = 0
    READ: int = 1
    WRITE: int = 4
    ERROR: int = 24
    _ioloop_for_asyncio: Dict[asyncio.AbstractEventLoop, 'IOLoop'] = dict()
    _pending_tasks: Set[Future] = set()

    @classmethod
    def configure(cls, impl: Union[str, Type['IOLoop']], **kwargs: Any) -> None:
        from tornado.platform.asyncio import BaseAsyncIOLoop
        if isinstance(impl, str):
            impl = import_object(impl)
        if isinstance(impl, type) and (not issubclass(impl, BaseAsyncIOLoop)):
            raise RuntimeError('only AsyncIOLoop is allowed when asyncio is available')
        super().configure(impl, **kwargs)

    @staticmethod
    def instance() -> 'IOLoop':
        """Deprecated alias for `IOLoop.current()`."""
        return IOLoop.current()

    def install(self) -> None:
        """Deprecated alias for `make_current()`."""
        self.make_current()

    @staticmethod
    def clear_instance() -> None:
        """Deprecated alias for `clear_current()`."""
        IOLoop.clear_current()

    @staticmethod
    def current(instance: bool = True) -> Optional['IOLoop']:
        """Returns the current thread's `IOLoop`."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            if not instance:
                return None
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            return IOLoop._ioloop_for_asyncio[loop]
        except KeyError:
            if instance:
                from tornado.platform.asyncio import AsyncIOMainLoop
                current = AsyncIOMainLoop()
            else:
                current = None
        return current

    def make_current(self) -> None:
        """Makes this the `IOLoop` for the current thread."""
        warnings.warn('make_current is deprecated; start the event loop first', DeprecationWarning, stacklevel=2)
        self._make_current()

    def _make_current(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def clear_current() -> None:
        """Clears the `IOLoop` for the current thread."""
        warnings.warn('clear_current is deprecated', DeprecationWarning, stacklevel=2)
        IOLoop._clear_current()

    @staticmethod
    def _clear_current() -> None:
        old = IOLoop.current(instance=False)
        if old is not None:
            old._clear_current_hook()

    def _clear_current_hook(self) -> None:
        """Instance method called when an IOLoop ceases to be current."""
        pass

    @classmethod
    def configurable_base(cls) -> Type['IOLoop']:
        return IOLoop

    @classmethod
    def configurable_default(cls) -> Type['IOLoop']:
        from tornado.platform.asyncio import AsyncIOLoop
        return AsyncIOLoop

    def initialize(self, make_current: bool = True) -> None:
        if make_current:
            self._make_current()

    def close(self, all_fds: bool = False) -> None:
        """Closes the `IOLoop`, freeing any resources used."""
        raise NotImplementedError()

    def add_handler(self, fd: Union[int, _Selectable], handler: Callable[[Union[int, _Selectable], int], None], events: int) -> None:
        """Registers the given handler to receive the given events for ``fd``."""
        raise NotImplementedError()

    def update_handler(self, fd: Union[int, _Selectable], events: int) -> None:
        """Changes the events we listen for ``fd``."""
        raise NotImplementedError()

    def remove_handler(self, fd: Union[int, _Selectable]) -> None:
        """Stop listening for events on ``fd``."""
        raise NotImplementedError()

    def start(self) -> None:
        """Starts the I/O loop."""
        raise NotImplementedError()

    def stop(self) -> None:
        """Stop the I/O loop."""
        raise NotImplementedError()

    def run_sync(self, func: Callable[..., Union[None, Awaitable[_T]], timeout: Optional[float] = None) -> _T:
        """Starts the `IOLoop`, runs the given function, and stops the loop."""
        future_cell: List[Optional[Future]] = [None]

        def run() -> None:
            try:
                result = func()
                if result is not None:
                    from tornado.gen import convert_yielded
                    result = convert_yielded(result)
            except Exception:
                fut = Future()
                future_cell[0] = fut
                future_set_exc_info(fut, sys.exc_info())
            else:
                if is_future(result):
                    future_cell[0] = cast(Future, result)
                else:
                    fut = Future()
                    future_cell[0] = fut
                    fut.set_result(result)
            assert future_cell[0] is not None
            self.add_future(future_cell[0], lambda future: self.stop())
        
        self.add_callback(run)
        if timeout is not None:
            def timeout_callback() -> None:
                assert future_cell[0] is not None
                if not future_cell[0].cancel():
                    self.stop()
            timeout_handle = self.add_timeout(self.time() + timeout, timeout_callback)
        
        self.start()
        if timeout is not None:
            self.remove_timeout(timeout_handle)
        assert future_cell[0] is not None
        if future_cell[0].cancelled() or not future_cell[0].done():
            raise TimeoutError('Operation timed out after %s seconds' % timeout)
        return future_cell[0].result()

    def time(self) -> float:
        """Returns the current time according to the `IOLoop`'s clock."""
        return time.time()

    def add_timeout(self, deadline: Union[float, datetime.timedelta], callback: Callable[..., None], *args: Any, **kwargs: Any) -> object:
        """Runs the ``callback`` at the time ``deadline`` from the I/O loop."""
        if isinstance(deadline, numbers.Real):
            return self.call_at(deadline, callback, *args, **kwargs)
        elif isinstance(deadline, datetime.timedelta):
            return self.call_at(self.time() + deadline.total_seconds(), callback, *args, **kwargs)
        else:
            raise TypeError('Unsupported deadline %r' % deadline)

    def call_later(self, delay: float, callback: Callable[..., None], *args: Any, **kwargs: Any) -> object:
        """Runs the ``callback`` after ``delay`` seconds have passed."""
        return self.call_at(self.time() + delay, callback, *args, **kwargs)

    def call_at(self, when: float, callback: Callable[..., None], *args: Any, **kwargs: Any) -> object:
        """Runs the ``callback`` at the absolute time designated by ``when``."""
        return self.add_timeout(when, callback, *args, **kwargs)

    def remove_timeout(self, timeout: object) -> None:
        """Cancels a pending timeout."""
        raise NotImplementedError()

    def add_callback(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Calls the given callback on the next I/O loop iteration."""
        raise NotImplementedError()

    def add_callback_from_signal(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Calls the given callback on the next I/O loop iteration."""
        raise NotImplementedError()

    def spawn_callback(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Calls the given callback on the next IOLoop iteration."""
        self.add_callback(callback, *args, **kwargs)

    def add_future(self, future: Union[Future, concurrent.futures.Future], callback: Callable[[Future], None]) -> None:
        """Schedules a callback on the ``IOLoop`` when the given `.Future` is finished."""
        if isinstance(future, Future):
            future.add_done_callback(lambda f: self._run_callback(functools.partial(callback, f)))
        else:
            assert is_future(future)
            future_add_done_callback(future, lambda f: self.add_callback(callback, f))

    def run_in_executor(self, executor: Optional[concurrent.futures.Executor], func: Callable[..., _T], *args: Any) -> Future[_T]:
        """Runs a function in a ``concurrent.futures.Executor``."""
        if executor is None:
            if not hasattr(self, '_executor'):
                from tornado.process import cpu_count
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 5)
            executor = self._executor
        c_future = executor.submit(func, *args)
        t_future: Future[_T] = Future()
        self.add_future(c_future, lambda f: chain_future(f, t_future))
        return t_future

    def set_default_executor(self, executor: concurrent.futures.Executor) -> None:
        """Sets the default executor to use with :meth:`run_in_executor`."""
        self._executor = executor

    def _run_callback(self, callback: Callable[..., Any]) -> None:
        """Runs a callback with error handling."""
        try:
            ret = callback()
            if ret is not None:
                from tornado import gen
                try:
                    ret = gen.convert_yielded(ret)
                except gen.BadYieldError:
                    pass
                else:
                    self.add_future(ret, self._discard_future_result)
        except asyncio.CancelledError:
            pass
        except Exception:
            app_log.error('Exception in callback %r', callback, exc_info=True)

    def _discard_future_result(self, future: Future) -> None:
        """Avoid unhandled-exception warnings from spawned coroutines."""
        future.result()

    def split_fd(self, fd: Union[int, _Selectable]) -> Tuple[int, Union[int, _Selectable]]:
        if isinstance(fd, int):
            return (fd, fd)
        return (fd.fileno(), fd)

    def close_fd(self, fd: Union[int, _Selectable]) -> None:
        try:
            if isinstance(fd, int):
                os.close(fd)
            else:
                fd.close()
        except OSError:
            pass

    def _register_task(self, f: Future) -> None:
        self._pending_tasks.add(f)

    def _unregister_task(self, f: Future) -> None:
        self._pending_tasks.discard(f)

class _Timeout:
    """An IOLoop timeout, a UNIX timestamp and a callback"""
    __slots__ = ['deadline', 'callback', 'tdeadline']

    def __init__(self, deadline: float, callback: Callable[..., None], io_loop: IOLoop) -> None:
        if not isinstance(deadline, numbers.Real):
            raise TypeError('Unsupported deadline %r' % deadline)
        self.deadline = deadline
        self.callback = callback
        self.tdeadline = (deadline, next(io_loop._timeout_counter))

    def __lt__(self, other: '_Timeout') -> bool:
        return self.tdeadline < other.tdeadline

    def __le__(self, other: '_Timeout') -> bool:
        return self.tdeadline <= other.tdeadline

class PeriodicCallback:
    """Schedules the given callback to be called periodically."""

    def __init__(self, callback: Callable[..., Union[None, Awaitable[None]]], callback_time: Union[float, datetime.timedelta], jitter: float = 0) -> None:
        self.callback = callback
        if isinstance(callback_time, datetime.timedelta):
            self.callback_time = callback_time / datetime.timedelta(milliseconds=1)
        else:
            if callback_time <= 0:
                raise ValueError('Periodic callback must have a positive callback_time')
            self.callback_time = callback_time
        self.jitter = jitter
        self._running = False
        self._timeout: Optional[object] = None
        self.io_loop: Optional[IOLoop] = None
        self._next_timeout: float = 0.0

    def start(self) -> None:
        """Starts the timer."""
        self.io_loop = IOLoop.current()
        self._running = True
        self._next_timeout = self.io_loop.time()
        self._schedule_next()

    def stop(self) -> None:
        """Stops the timer."""
        self._running = False
        if self._timeout is not None and self.io_loop is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None

    def is_running(self) -> bool:
        """Returns ``True`` if this `.PeriodicCallback` has been started."""
        return self._running

    async def _run(self) -> None:
        if not self._running or self.io_loop is None:
            return
        try:
            val = self.callback()
            if val is not None and isawaitable(val):
                await val
        except Exception:
            app_log.error('Exception in callback %r', self.callback, exc_info=True)
        finally:
            self._schedule_next()

    def _schedule_next(self) -> None:
        if self._running and self.io_loop is not None:
            self._update_next(self.io_loop.time())
            self._timeout = self.io_loop.add_timeout(self._next_timeout, self._run)

    def _update_next(self, current_time: float) -> None:
        callback_time_sec = self.callback_time / 1000.0
        if self.jitter:
            callback_time_sec *= 1 + self.jitter * (random.random() - 0.5)
        if self._next_timeout <= current_time:
            self._next_timeout += (math.floor((current_time - self._next_timeout) / callback_time_sec) + 1) * callback_time_sec
        else:
            self._next_timeout += callback_time_sec
