#!/usr/bin/env python3
"""
This module implements generator‐based coroutines with type annotations.
"""

from __future__ import annotations
import asyncio
import builtins
import collections
from collections.abc import Generator, Awaitable
import concurrent.futures
import datetime
import functools
from functools import singledispatch, update_wrapper
import sys
import types
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, overload

from tornado.concurrent import (Future, is_future, chain_future, future_set_exc_info,
                                future_add_done_callback, future_set_result_unless_cancelled)
from tornado.ioloop import IOLoop
from tornado.log import app_log
from tornado.util import TimeoutError

try:
    import contextvars
except ImportError:
    contextvars = None

if typing.TYPE_CHECKING:
    from typing import Deque, Set, Iterable

_T = typing.TypeVar('_T')
_Yieldable = Union[None, Awaitable, List[Awaitable], Dict[Any, Awaitable], concurrent.futures.Future]


class KeyReuseError(Exception):
    pass


class UnknownKeyError(Exception):
    pass


class LeakedCallbackError(Exception):
    pass


class BadYieldError(Exception):
    pass


class ReturnValueIgnoredError(Exception):
    pass


def _value_from_stopiteration(e: BaseException) -> Any:
    try:
        return e.value
    except AttributeError:
        pass
    try:
        return e.args[0]
    except (AttributeError, IndexError):
        return None


def _create_future() -> Future:
    future = Future()
    source_traceback = getattr(future, '_source_traceback', ())
    while source_traceback:
        filename = source_traceback[-1][0]
        if filename == __file__:
            del source_traceback[-1]
        else:
            break
    return future


def _fake_ctx_run(f: Callable[..., _T], *args: Any, **kw: Any) -> _T:
    return f(*args, **kw)


@overload
def coroutine(func: Callable[..., Any]) -> Callable[..., Future]: ...


@overload
def coroutine(func: Callable[..., Any]) -> Callable[..., Future]: ...


def coroutine(func: Callable[..., Any]) -> Callable[..., Future]:
    """Decorator for asynchronous generators.

    Functions with this decorator return a `.Future`.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Future:
        future: Future = _create_future()
        if contextvars is not None:
            ctx_run: Callable[..., Any] = contextvars.copy_context().run  # type: ignore
        else:
            ctx_run = _fake_ctx_run
        try:
            result = ctx_run(func, *args, **kwargs)
        except (Return, StopIteration) as e:
            result = _value_from_stopiteration(e)
        except Exception:
            future_set_exc_info(future, sys.exc_info())
            try:
                return future
            finally:
                future = None  # type: ignore
        else:
            if isinstance(result, Generator):
                try:
                    yielded = ctx_run(next, result)
                except (StopIteration, Return) as e:
                    future_set_result_unless_cancelled(future, _value_from_stopiteration(e))
                except Exception:
                    future_set_exc_info(future, sys.exc_info())
                else:
                    runner = Runner(ctx_run, result, future, yielded)
                    future.add_done_callback(lambda _: runner)
                yielded = None
                try:
                    return future
                finally:
                    future = None  # type: ignore
        future_set_result_unless_cancelled(future, result)
        return future
    wrapper.__wrapped__ = func
    wrapper.__tornado_coroutine__ = True  # type: ignore[attr-defined]
    return wrapper


def is_coroutine_function(func: Callable[..., Any]) -> bool:
    """Return whether *func* is a coroutine function."""
    return getattr(func, '__tornado_coroutine__', False)


class Return(Exception):
    """Special exception to return a value from a `coroutine`."""

    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value
        self.args = (value,)


class WaitIterator:
    """Provides an iterator to yield the results of awaitables as they finish."""

    _unfinished: Dict[Future, Any]

    def __init__(self, *args: Future, **kwargs: Future) -> None:
        if args and kwargs:
            raise ValueError('You must provide args or kwargs, not both')
        if kwargs:
            self._unfinished = {f: k for k, f in kwargs.items()}
            futures: List[Future] = list(kwargs.values())
        else:
            self._unfinished = {f: i for i, f in enumerate(args)}
            futures = list(args)
        self._finished: collections.deque[Future] = collections.deque()
        self.current_index: Optional[Any] = None
        self.current_future: Optional[Future] = None
        self._running_future: Optional[Future] = None
        for future in futures:
            future_add_done_callback(future, self._done_callback)

    def done(self) -> bool:
        """Returns True if this iterator has no more results."""
        if self._finished or self._unfinished:
            return False
        self.current_index = None
        self.current_future = None
        return True

    def next(self) -> Future:
        """Returns a `.Future` that will yield the next available result."""
        self._running_future = _create_future()
        if self._finished:
            return self._return_result(self._finished.popleft())
        return self._running_future

    def _done_callback(self, done: Future) -> None:
        if self._running_future and (not self._running_future.done()):
            self._return_result(done)
        else:
            self._finished.append(done)

    def _return_result(self, done: Future) -> Future:
        """Set the returned future's state to that of the yielded future."""
        if self._running_future is None:
            raise Exception('no future is running')
        chain_future(done, self._running_future)
        res: Future = self._running_future
        self._running_future = None
        self.current_future = done
        self.current_index = self._unfinished.pop(done)
        return res

    def __aiter__(self) -> WaitIterator:
        return self

    async def __anext__(self) -> Any:
        if self.done():
            raise getattr(builtins, 'StopAsyncIteration')()
        return await self.next()


def multi(children: Union[List[_Yieldable], Dict[Any, _Yieldable]],
          quiet_exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ()) -> Future:
    """Runs multiple asynchronous operations in parallel.
    """
    return multi_future(children, quiet_exceptions=quiet_exceptions)


Multi = multi


def multi_future(children: Union[List[_Yieldable], Dict[Any, _Yieldable]],
                 quiet_exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ()) -> Future:
    """Wait for multiple asynchronous futures in parallel."""
    if isinstance(children, dict):
        keys: List[Any] = list(children.keys())
        children_seq = children.values()
    else:
        keys = None
        children_seq = children
    children_futs: List[Future] = list(map(convert_yielded, children_seq))
    assert all((is_future(i) or isinstance(i, _NullFuture) for i in children_futs))
    unfinished_children: set[Future] = set(children_futs)
    future: Future = _create_future()
    if not children_futs:
        future_set_result_unless_cancelled(future, {} if keys is not None else [])
    def callback(fut: Future) -> None:
        unfinished_children.remove(fut)
        if not unfinished_children:
            result_list: List[Any] = []
            for f in children_futs:
                try:
                    result_list.append(f.result())
                except Exception as e:
                    if future.done():
                        if not isinstance(e, quiet_exceptions):
                            app_log.error('Multiple exceptions in yield list', exc_info=True)
                    else:
                        future_set_exc_info(future, sys.exc_info())
            if not future.done():
                if keys is not None:
                    future_set_result_unless_cancelled(future, dict(zip(keys, result_list)))
                else:
                    future_set_result_unless_cancelled(future, result_list)
    listening: set[Future] = set()
    for f in children_futs:
        if f not in listening:
            listening.add(f)
            future_add_done_callback(f, callback)
    return future


def maybe_future(x: Any) -> Future:
    """Converts ``x`` into a `.Future`."""
    if is_future(x):
        return x
    else:
        fut = _create_future()
        fut.set_result(x)
        return fut


def with_timeout(timeout: Union[datetime.timedelta, float, int],
                 future: _Yieldable,
                 quiet_exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ()) -> Future:
    """Wraps a `.Future` (or other yieldable object) in a timeout."""
    future_converted: Future = convert_yielded(future)
    result: Future = _create_future()
    chain_future(future_converted, result)
    io_loop: IOLoop = IOLoop.current()

    def error_callback(future_inner: Future) -> None:
        try:
            future_inner.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not isinstance(e, quiet_exceptions):
                app_log.error('Exception in Future %r after timeout', future_inner, exc_info=True)

    def timeout_callback() -> None:
        if not result.done():
            result.set_exception(TimeoutError('Timeout'))
        future_add_done_callback(future_converted, error_callback)
    timeout_handle = io_loop.add_timeout(timeout, timeout_callback)
    if isinstance(future_converted, Future):
        future_add_done_callback(future_converted, lambda fut: io_loop.remove_timeout(timeout_handle))
    else:
        io_loop.add_future(future_converted, lambda fut: io_loop.remove_timeout(timeout_handle))
    return result


def sleep(duration: float) -> Future:
    """Return a `.Future` that resolves after the given number of seconds."""
    f: Future = _create_future()
    IOLoop.current().call_later(duration, lambda: future_set_result_unless_cancelled(f, None))
    return f


class _NullFuture:
    """_NullFuture resembles a Future that finished with a result of None."""

    def result(self) -> Any:
        return None

    def done(self) -> bool:
        return True


_null_future: Future = typing.cast(Future, _NullFuture())
moment: Future = typing.cast(Future, _NullFuture())
moment.__doc__ = (
    'A special object which may be yielded to allow the IOLoop to run for\n'
    'one iteration.\n\n'
    'This is not needed in normal use but it can be helpful in long-running\n'
    'coroutines that are likely to yield Futures that are ready instantly.\n\n'
    'Usage: ``yield gen.moment``\n\n'
    'In native coroutines, the equivalent of ``yield gen.moment`` is\n'
    '``await asyncio.sleep(0)``.\n\n'
    '.. versionadded:: 4.0\n\n'
    '.. deprecated:: 4.5\n'
    '   ``yield None`` (or ``yield`` with no argument) is now equivalent to\n'
    '   ``yield gen.moment``.\n'
)


class Runner:
    """Internal implementation of `tornado.gen.coroutine`.

    Maintains information about pending callbacks and their results.
    """

    def __init__(self, ctx_run: Callable[..., Any], gen: Generator, result_future: Future, first_yielded: Any) -> None:
        self.ctx_run: Callable[..., Any] = ctx_run
        self.gen: Generator = gen
        self.result_future: Future = result_future
        self.future: Future = _null_future  # type: ignore
        self.running: bool = False
        self.finished: bool = False
        self.io_loop: IOLoop = IOLoop.current()
        if self.ctx_run(self.handle_yield, first_yielded):
            # Remove references to allow GC
            gen = result_future = first_yielded = None  # type: ignore
            self.ctx_run(self.run)

    def run(self) -> None:
        """Starts or resumes the generator, running until it reaches a yield point that is not ready."""
        if self.running or self.finished:
            return
        try:
            self.running = True
            while True:
                future_local: Optional[Future] = self.future
                if future_local is None:
                    raise Exception('No pending future')
                if not future_local.done():
                    return
                self.future = None
                try:
                    try:
                        value = future_local.result()
                    except Exception as e:
                        exc: Optional[Exception] = e
                    else:
                        exc = None
                    finally:
                        future_local = None  # type: ignore
                    if exc is not None:
                        try:
                            yielded = self.gen.throw(exc)
                        finally:
                            del exc
                    else:
                        yielded = self.gen.send(value)
                except (StopIteration, Return) as e:
                    self.finished = True
                    self.future = _null_future  # type: ignore
                    future_set_result_unless_cancelled(self.result_future, _value_from_stopiteration(e))
                    self.result_future = None  # type: ignore
                    return
                except Exception:
                    self.finished = True
                    self.future = _null_future  # type: ignore
                    future_set_exc_info(self.result_future, sys.exc_info())
                    self.result_future = None  # type: ignore
                    return
                if not self.handle_yield(yielded):
                    return
                yielded = None
        finally:
            self.running = False

    def handle_yield(self, yielded: Any) -> bool:
        try:
            self.future = convert_yielded(yielded)
        except BadYieldError:
            self.future = Future()
            future_set_exc_info(self.future, sys.exc_info())
        if self.future is moment:
            self.io_loop.add_callback(self.ctx_run, self.run)
            return False
        elif self.future is None:
            raise Exception('no pending future')
        elif not self.future.done():
            def inner(f: Future) -> None:
                self.ctx_run(self.run)
            self.io_loop.add_future(self.future, inner)
            return False
        return True

    def handle_exception(self, typ: Type[BaseException], value: BaseException, tb: Optional[Any]) -> bool:
        if not self.running and (not self.finished):
            self.future = Future()
            future_set_exc_info(self.future, (typ, value, tb))
            self.ctx_run(self.run)
            return True
        else:
            return False


def _wrap_awaitable(awaitable: Awaitable) -> Future:
    fut: asyncio.Future = asyncio.ensure_future(awaitable)
    loop: IOLoop = IOLoop.current()
    loop._register_task(fut)
    fut.add_done_callback(lambda f: loop._unregister_task(f))
    return fut


def convert_yielded(yielded: Any) -> Future:
    """Convert a yielded object into a `.Future`."""
    if yielded is None or yielded is moment:
        return moment
    elif yielded is _null_future:
        return _null_future
    elif isinstance(yielded, (list, dict)):
        return multi(yielded)
    elif is_future(yielded):
        return typing.cast(Future, yielded)
    elif asyncio.iscoroutine(yielded) or isawaitable(yielded):
        return _wrap_awaitable(yielded)
    else:
        raise BadYieldError(f'yielded unknown object {yielded!r}')


convert_yielded = singledispatch(convert_yielded)
 
# End of annotated module.
