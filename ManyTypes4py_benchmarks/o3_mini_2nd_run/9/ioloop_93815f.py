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
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable, cast

if typing.TYPE_CHECKING:
    from typing import Dict, List, Set
    from typing_extensions import Protocol
else:
    Protocol = object

class _Selectable(Protocol):
    def fileno(self) -> int: ...
    def close(self) -> None: ...

_T = TypeVar('_T')
_S = TypeVar('_S', bound=_Selectable)

class IOLoop(Configurable):
    NONE: int = 0
    READ: int = 1
    WRITE: int = 4
    ERROR: int = 24
    _ioloop_for_asyncio: dict = dict()
    _pending_tasks: set = set()

    @classmethod
    def configure(cls, impl: Union[str, Type[Any]], **kwargs: Any) -> None:
        from tornado.platform.asyncio import BaseAsyncIOLoop
        if isinstance(impl, str):
            impl = import_object(impl)
        if isinstance(impl, type) and (not issubclass(impl, BaseAsyncIOLoop)):
            raise RuntimeError('only AsyncIOLoop is allowed when asyncio is available')
        super().configure(impl, **kwargs)

    @staticmethod
    def instance() -> "IOLoop":
        return IOLoop.current()

    def install(self) -> None:
        self.make_current()

    @staticmethod
    def clear_instance() -> None:
        IOLoop.clear_current()

    @typing.overload
    @staticmethod
    def current() -> "IOLoop":
        ...
    @typing.overload
    @staticmethod
    def current(instance: bool) -> Optional["IOLoop"]:
        ...
    @staticmethod
    def current(instance: bool = True) -> Optional["IOLoop"]:
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
        warnings.warn('make_current is deprecated; start the event loop first', DeprecationWarning, stacklevel=2)
        self._make_current()

    def _make_current(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def clear_current() -> None:
        warnings.warn('clear_current is deprecated', DeprecationWarning, stacklevel=2)
        IOLoop._clear_current()

    @staticmethod
    def _clear_current() -> None:
        old = IOLoop.current(instance=False)
        if old is not None:
            old._clear_current_hook()

    def _clear_current_hook(self) -> None:
        pass

    @classmethod
    def configurable_base(cls) -> Type["IOLoop"]:
        return IOLoop

    @classmethod
    def configurable_default(cls) -> Type["IOLoop"]:
        from tornado.platform.asyncio import AsyncIOLoop
        return AsyncIOLoop

    def initialize(self, make_current: bool = True) -> None:
        if make_current:
            self._make_current()

    def close(self, all_fds: bool = False) -> None:
        raise NotImplementedError()

    @typing.overload
    def add_handler(self, fd: Union[int, _Selectable], handler: Callable[..., Any], events: int) -> None:
        ...
    @typing.overload
    def add_handler(self, fd: Union[int, _Selectable], handler: Callable[..., Any], events: int) -> None:
        ...
    def add_handler(self, fd: Union[int, _Selectable], handler: Callable[..., Any], events: int) -> None:
        raise NotImplementedError()

    def update_handler(self, fd: Union[int, _Selectable], events: int) -> None:
        raise NotImplementedError()

    def remove_handler(self, fd: Union[int, _Selectable]) -> None:
        raise NotImplementedError()

    def start(self) -> None:
        raise NotImplementedError()

    def stop(self) -> None:
        raise NotImplementedError()

    def run_sync(self, func: Callable[[], Union[Awaitable[Any], None]], timeout: Optional[Union[float, int]] = None) -> Any:
        future_cell: list[Optional[Future[Any]]] = [None]

        def run() -> None:
            try:
                result = func()
                if result is not None:
                    from tornado.gen import convert_yielded
                    result = convert_yielded(result)
            except Exception:
                fut: Future[Any] = Future()
                future_cell[0] = fut
                future_set_exc_info(fut, sys.exc_info())
            else:
                if is_future(result):
                    future_cell[0] = result
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
        return time.time()

    def add_timeout(self, deadline: Union[numbers.Real, datetime.timedelta], callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if isinstance(deadline, numbers.Real):
            return self.call_at(deadline, callback, *args, **kwargs)
        elif isinstance(deadline, datetime.timedelta):
            return self.call_at(self.time() + deadline.total_seconds(), callback, *args, **kwargs)
        else:
            raise TypeError('Unsupported deadline %r' % deadline)

    def call_later(self, delay: Union[float, int], callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.call_at(self.time() + delay, callback, *args, **kwargs)

    def call_at(self, when: float, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.add_timeout(when, callback, *args, **kwargs)

    def remove_timeout(self, timeout: Any) -> None:
        raise NotImplementedError()

    def add_callback(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def add_callback_from_signal(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def spawn_callback(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.add_callback(callback, *args, **kwargs)

    def add_future(self, future: Union[Future[Any], Any], callback: Callable[[Future[Any]], Any]) -> None:
        if isinstance(future, Future):
            future.add_done_callback(lambda f: self._run_callback(functools.partial(callback, f)))
        else:
            assert is_future(future)
            future_add_done_callback(future, lambda f: self.add_callback(callback, f))

    def run_in_executor(self, executor: Optional[concurrent.futures.Executor], func: Callable[..., Any], *args: Any) -> Future[Any]:
        if executor is None:
            if not hasattr(self, '_executor'):
                from tornado.process import cpu_count
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 5)
            executor = self._executor
        c_future: concurrent.futures.Future[Any] = executor.submit(func, *args)
        t_future: Future[Any] = Future()
        self.add_future(c_future, lambda f: chain_future(f, t_future))
        return t_future

    def set_default_executor(self, executor: concurrent.futures.Executor) -> None:
        self._executor = executor

    def _run_callback(self, callback: Callable[[], Any]) -> None:
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

    def _discard_future_result(self, future: Future[Any]) -> None:
        future.result()

    def split_fd(self, fd: Union[int, _Selectable]) -> Tuple[int, int]:
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

    def _register_task(self, f: Any) -> None:
        self._pending_tasks.add(f)

    def _unregister_task(self, f: Any) -> None:
        self._pending_tasks.discard(f)

class _Timeout:
    __slots__ = ['deadline', 'callback', 'tdeadline']

    def __init__(self, deadline: numbers.Real, callback: Callable[..., Any], io_loop: IOLoop) -> None:
        if not isinstance(deadline, numbers.Real):
            raise TypeError('Unsupported deadline %r' % deadline)
        self.deadline: numbers.Real = deadline
        self.callback: Callable[..., Any] = callback
        self.tdeadline: Tuple[Any, Any] = (deadline, next(io_loop._timeout_counter))

    def __lt__(self, other: "_Timeout") -> bool:
        return self.tdeadline < other.tdeadline

    def __le__(self, other: "_Timeout") -> bool:
        return self.tdeadline <= other.tdeadline

class PeriodicCallback:
    def __init__(self, callback: Callable[..., Any], callback_time: Union[float, datetime.timedelta], jitter: float = 0) -> None:
        self.callback: Callable[..., Any] = callback
        if isinstance(callback_time, datetime.timedelta):
            self.callback_time: float = callback_time / datetime.timedelta(milliseconds=1)
        else:
            if callback_time <= 0:
                raise ValueError('Periodic callback must have a positive callback_time')
            self.callback_time = callback_time
        self.jitter: float = jitter
        self._running: bool = False
        self._timeout: Optional[Any] = None

    def start(self) -> None:
        self.io_loop: IOLoop = cast(IOLoop, IOLoop.current())
        self._running = True
        self._next_timeout: float = self.io_loop.time()
        self._schedule_next()

    def stop(self) -> None:
        self._running = False
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None

    def is_running(self) -> bool:
        return self._running

    async def _run(self) -> None:
        if not self._running:
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
        if self._running:
            self._update_next(self.io_loop.time())
            self._timeout = self.io_loop.add_timeout(self._next_timeout, self._run)

    def _update_next(self, current_time: float) -> None:
        callback_time_sec: float = self.callback_time / 1000.0
        if self.jitter:
            callback_time_sec *= 1 + self.jitter * (random.random() - 0.5)
        if self._next_timeout <= current_time:
            self._next_timeout += (math.floor((current_time - self._next_timeout) / callback_time_sec) + 1) * callback_time_sec
        else:
            self._next_timeout += callback_time_sec