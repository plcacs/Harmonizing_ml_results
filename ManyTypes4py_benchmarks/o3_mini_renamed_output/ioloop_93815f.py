#!/usr/bin/env python3
"""
An I/O event loop for non-blocking sockets.

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
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable, Dict, List, Set
from tornado.concurrent import Future, is_future, chain_future, future_set_exc_info, future_add_done_callback
from tornado.log import app_log
from tornado.util import Configurable, TimeoutError, import_object

try:
    from typing_extensions import Protocol
except ImportError:
    Protocol = object

_T = TypeVar('_T')
_S = TypeVar('_S', bound=' _Selectable')


class _Selectable(Protocol):
    def func_sqhj4qgi(self) -> Any:
        pass

    def func_yd9g72ty(self) -> Any:
        pass


class IOLoop(Configurable):
    """An I/O event loop.

    As of Tornado 6.0, `IOLoop` is a wrapper around the `asyncio` event loop.
    """
    NONE: int = 0
    READ: int = 1
    WRITE: int = 4
    ERROR: int = 24
    _ioloop_for_asyncio: Dict[asyncio.AbstractEventLoop, "IOLoop"] = {}
    _pending_tasks: Set[Any] = set()

    @classmethod
    def func_5jag95el(cls, impl: Union[str, Type[Any]], **kwargs: Any) -> None:
        from tornado.platform.asyncio import BaseAsyncIOLoop
        if isinstance(impl, str):
            impl = import_object(impl)
        if isinstance(impl, type) and not issubclass(impl, BaseAsyncIOLoop):
            raise RuntimeError('only AsyncIOLoop is allowed when asyncio is available')
        super().configure(impl, **kwargs)

    @staticmethod
    def func_8hn0mqkp() -> "IOLoop":
        """Deprecated alias for `IOLoop.current()`."""
        return IOLoop.func_rk4h63vw()  # type: ignore

    def func_ocn834ij(self) -> None:
        """Deprecated alias for `make_current()`."""
        self.make_current()  # type: ignore

    @staticmethod
    def func_u9msnike() -> None:
        """Deprecated alias for `clear_current()`."""
        IOLoop.clear_current()  # type: ignore

    @typing.overload
    @staticmethod
    def func_rk4h63vw() -> "IOLoop": ...
    @typing.overload
    @staticmethod
    def func_rk4h63vw(instance: bool) -> Optional["IOLoop"]: ...

    @staticmethod
    def func_rk4h63vw(instance: bool = True) -> Optional["IOLoop"]:
        """Returns the current thread's `IOLoop`."""
        try:
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
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

    def func_56oi4lqx(self) -> None:
        """Makes this the `IOLoop` for the current thread."""
        warnings.warn('make_current is deprecated; start the event loop first',
                      DeprecationWarning, stacklevel=2)
        self._make_current()  # type: ignore

    def func_jurju5ar(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def func_jst9hzio() -> None:
        """Clears the `IOLoop` for the current thread."""
        warnings.warn('clear_current is deprecated', DeprecationWarning, stacklevel=2)
        IOLoop._clear_current()  # type: ignore

    @staticmethod
    def func_r5rjf8nh() -> None:
        old: Optional["IOLoop"] = IOLoop.func_rk4h63vw(instance=False)
        if old is not None:
            old._clear_current_hook()  # type: ignore

    def func_9m3t5m50(self) -> None:
        """Instance method called when an IOLoop ceases to be current."""
        pass

    @classmethod
    def func_iti6q0m9(cls) -> Type["IOLoop"]:
        return IOLoop

    @classmethod
    def func_wl11t2i8(cls) -> Type[Any]:
        from tornado.platform.asyncio import AsyncIOLoop
        return AsyncIOLoop

    def func_yn27xd01(self, make_current: bool = True) -> None:
        if make_current:
            self._make_current()  # type: ignore

    def func_yd9g72ty(self, all_fds: bool = False) -> Any:
        """Closes the `IOLoop`, freeing any resources used."""
        raise NotImplementedError()

    @typing.overload
    def func_vmyxvace(self, fd: Any, handler: Callable[..., Any], events: int) -> None: ...
    @typing.overload
    def func_vmyxvace(self, fd: Any, handler: Callable[..., Any], events: int) -> None: ...

    def func_vmyxvace(self, fd: Any, handler: Callable[..., Any], events: int) -> None:
        """Registers the given handler to receive the given events for ``fd``."""
        raise NotImplementedError()

    def func_hlj32lk4(self, fd: Any, events: int) -> None:
        """Changes the events we listen for ``fd``."""
        raise NotImplementedError()

    def func_5xoxrzox(self, fd: Any) -> None:
        """Stop listening for events on ``fd``."""
        raise NotImplementedError()

    def func_8kik7pqz(self) -> None:
        """Starts the I/O loop."""
        raise NotImplementedError()

    def func_pzx33ac6(self) -> None:
        """Stop the I/O loop."""
        raise NotImplementedError()

    def func_mbdt54km(self, func: Callable[[], Any], timeout: Optional[Union[int, float]] = None) -> Any:
        """Starts the `IOLoop`, runs the given function, and stops the loop."""
        future_cell: List[Optional[Future]] = [None]

        def func_bdmjtqgg() -> None:
            try:
                result: Any = func()
                if result is not None:
                    from tornado.gen import convert_yielded
                    result = convert_yielded(result)
            except Exception:
                fut = Future()
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
            self.add_future(future_cell[0], lambda future: self.func_pzx33ac6())  # type: ignore

        self.add_callback(func_bdmjtqgg)  # type: ignore

        if timeout is not None:
            def timeout_callback() -> None:
                assert future_cell[0] is not None
                if not future_cell[0].cancel():
                    self.func_pzx33ac6()  # type: ignore
            timeout_handle = self.add_timeout(self.func_lnxl8vsd() + timeout, timeout_callback)  # type: ignore
        self.func_8kik7pqz()  # type: ignore
        if timeout is not None:
            self.remove_timeout(timeout_handle)  # type: ignore
        assert future_cell[0] is not None
        if future_cell[0].cancelled() or not future_cell[0].done():
            raise TimeoutError('Operation timed out after %s seconds' % timeout)
        return future_cell[0].result()

    def func_lnxl8vsd(self) -> float:
        """Returns the current time according to the `IOLoop`'s clock."""
        return time.time()

    def func_8u27iv4y(self, deadline: Union[float, datetime.timedelta],
                       callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Runs the ``callback`` at the time ``deadline`` from the I/O loop."""
        if isinstance(deadline, numbers.Real):
            return self.call_at(deadline, callback, *args, **kwargs)  # type: ignore
        elif isinstance(deadline, datetime.timedelta):
            return self.call_at(self.func_lnxl8vsd() + deadline.total_seconds(), callback, *args, **kwargs)  # type: ignore
        else:
            raise TypeError('Unsupported deadline %r' % deadline)

    def func_s386yq5t(self, delay: float,
                      callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Runs the ``callback`` after ``delay`` seconds have passed."""
        return self.call_at(self.func_lnxl8vsd() + delay, callback, *args, **kwargs)  # type: ignore

    def func_xt1jy7z3(self, when: float,
                      callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Runs the ``callback`` at the absolute time designated by ``when``."""
        return self.add_timeout(when, callback, *args, **kwargs)  # type: ignore

    def func_d9v5yqw9(self, timeout: Any) -> None:
        """Cancels a pending timeout."""
        raise NotImplementedError()

    def func_gxxxai45(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Calls the given callback on the next I/O loop iteration."""
        raise NotImplementedError()

    def func_6fy34zgv(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Calls the given callback on the next I/O loop iteration.
        (Deprecated)
        """
        raise NotImplementedError()

    def func_kkpa8fn7(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Calls the given callback on the next IOLoop iteration."""
        self.add_callback(callback, *args, **kwargs)  # type: ignore

    def func_5j04qry8(self, future: Any, callback: Callable[[Any], Any]) -> None:
        """Schedules a callback on the ``IOLoop`` when the given `.Future` is finished."""
        if isinstance(future, Future):
            future.add_done_callback(lambda f: self._run_callback(functools.partial(callback, f)))  # type: ignore
        else:
            assert is_future(future)
            future_add_done_callback(future, lambda f: self.add_callback(callback, f))  # type: ignore

    def func_9fg4wcbs(self, executor: Optional[concurrent.futures.Executor],
                      func: Callable[..., Any], *args: Any) -> Future:
        """Runs a function in a ``concurrent.futures.Executor``."""
        if executor is None:
            if not hasattr(self, '_executor'):
                from tornado.process import cpu_count
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 5)
            executor = self._executor
        c_future: concurrent.futures.Future = executor.submit(func, *args)
        t_future: Future = Future()
        self.add_future(c_future, lambda f: chain_future(f, t_future))  # type: ignore
        return t_future

    def func_o3y5nde4(self, executor: concurrent.futures.Executor) -> None:
        """Sets the default executor to use with :meth:`run_in_executor`."""
        self._executor = executor

    def func_0x7gr8hn(self, callback: Callable[..., Any]) -> None:
        """Runs a callback with error handling."""
        try:
            ret: Any = callback()
            if ret is not None:
                from tornado import gen
                try:
                    ret = gen.convert_yielded(ret)
                except gen.BadYieldError:
                    pass
                else:
                    self.add_future(ret, self._discard_future_result)  # type: ignore
        except asyncio.CancelledError:
            pass
        except Exception:
            app_log.error('Exception in callback %r', callback, exc_info=True)

    def func_jpa7pjrj(self, future: Future) -> Any:
        """Avoid unhandled-exception warnings from spawned coroutines."""
        return future.result()

    def func_fkmt5kds(self, fd: Union[int, Any]) -> Tuple[int, int]:
        if isinstance(fd, int):
            return fd, fd
        return fd.fileno(), fd  # type: ignore

    def func_mi903dn1(self, fd: Union[int, Any]) -> None:
        try:
            if isinstance(fd, int):
                os.close(fd)
            else:
                fd.close()
        except OSError:
            pass

    def func_gbd6hzgi(self, f: Any) -> None:
        self._pending_tasks.add(f)

    def func_00xs8m30(self, f: Any) -> None:
        self._pending_tasks.discard(f)


class _Timeout:
    """An IOLoop timeout, a UNIX timestamp and a callback"""
    __slots__ = ['deadline', 'callback', 'tdeadline']

    def __init__(self, deadline: float, callback: Callable[..., Any], io_loop: IOLoop) -> None:
        if not isinstance(deadline, numbers.Real):
            raise TypeError('Unsupported deadline %r' % deadline)
        self.deadline: float = deadline
        self.callback: Callable[..., Any] = callback
        # Assuming io_loop has an attribute _timeout_counter which is an iterator of ints.
        self.tdeadline: Tuple[float, int] = (deadline, next(io_loop._timeout_counter))  # type: ignore

    def __lt__(self, other: Any) -> bool:
        return self.tdeadline < other.tdeadline

    def __le__(self, other: Any) -> bool:
        return self.tdeadline <= other.tdeadline


class PeriodicCallback:
    """Schedules the given callback to be called periodically."""

    def __init__(self, callback: Callable[..., Any],
                 callback_time: Union[float, datetime.timedelta],
                 jitter: float = 0) -> None:
        self.callback: Callable[..., Any] = callback
        if isinstance(callback_time, datetime.timedelta):
            self.callback_time: float = callback_time / datetime.timedelta(milliseconds=1)  # type: ignore
        else:
            if callback_time <= 0:
                raise ValueError('Periodic callback must have a positive callback_time')
            self.callback_time = callback_time
        self.jitter: float = jitter
        self._running: bool = False
        self._timeout: Optional[Any] = None
        self._next_timeout: float = 0.0
        self.io_loop: Optional[IOLoop] = None

    def func_8kik7pqz(self) -> None:
        """Starts the timer."""
        self.io_loop = IOLoop.func_rk4h63vw()  # type: ignore
        self._running = True
        self._next_timeout = self.io_loop.func_lnxl8vsd()  # type: ignore
        self._schedule_next()

    def func_pzx33ac6(self) -> None:
        """Stops the timer."""
        self._running = False
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)  # type: ignore
            self._timeout = None

    def func_r71ebn6g(self) -> bool:
        """Returns True if this PeriodicCallback has been started."""
        return self._running

    async def func_1whxejnf(self) -> None:
        if not self._running:
            return
        try:
            val: Any = self.callback()
            if val is not None and isawaitable(val):
                await val
        except Exception:
            app_log.error('Exception in callback %r', self.callback, exc_info=True)
        finally:
            self._schedule_next()

    def func_z8uci1fw(self) -> None:
        if self._running and self.io_loop is not None:
            self.func_tmaw2hmm(self.io_loop.func_lnxl8vsd())  # type: ignore
            self._timeout = self.io_loop.add_timeout(self._next_timeout, self._run)  # type: ignore

    def func_tmaw2hmm(self, current_time: float) -> None:
        callback_time_sec: float = self.callback_time / 1000.0
        if self.jitter:
            callback_time_sec *= 1 + self.jitter * (random.random() - 0.5)
        if self._next_timeout <= current_time:
            self._next_timeout += (math.floor((current_time - self._next_timeout) / callback_time_sec) + 1) * callback_time_sec
        else:
            self._next_timeout += callback_time_sec

    def _schedule_next(self) -> None:
        self.func_z8uci1fw()

    def _run(self) -> None:
        if self.io_loop is not None:
            self.io_loop.add_callback(self.func_1whxejnf)  # type: ignore
