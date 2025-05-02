import asyncio
import builtins
import collections
from collections.abc import Generator
import concurrent.futures
import datetime
import functools
from functools import singledispatch
from inspect import isawaitable
import sys
import types
from tornado.concurrent import Future, is_future, chain_future, future_set_exc_info, future_add_done_callback, future_set_result_unless_cancelled
from tornado.ioloop import IOLoop
from tornado.log import app_log
from tornado.util import TimeoutError
try:
    import contextvars
except ImportError:
    contextvars = None
import typing
from typing import Union, Any, Callable, List, Type, Tuple, Awaitable, Dict, overload, Optional, Sequence, Deque, Set, Iterable

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

def _value_from_stopiteration(e: StopIteration) -> Any:
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

def _fake_ctx_run(f: Callable, *args: Any, **kw: Any) -> Any:
    return f(*args, **kw)

@overload
def coroutine(func: Callable[..., Generator]) -> Callable[..., Future]:
    ...

@overload
def coroutine(func: Callable[..., Generator]) -> Callable[..., Future]:
    ...

def coroutine(func: Callable[..., Generator]) -> Callable[..., Future]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Future:
        future = _create_future()
        if contextvars is not None:
            ctx_run = contextvars.copy_context().run
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
                future = None
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
                    future = None
        future_set_result_unless_cancelled(future, result)
        return future
    wrapper.__wrapped__ = func
    wrapper.__tornado_coroutine__ = True
    return wrapper

def is_coroutine_function(func: Callable) -> bool:
    return getattr(func, '__tornado_coroutine__', False)

class Return(Exception):
    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value
        self.args = (value,)

class WaitIterator:
    _unfinished: Dict[Future, Union[int, str]]

    def __init__(self, *args: Awaitable, **kwargs: Awaitable) -> None:
        if args and kwargs:
            raise ValueError('You must provide args or kwargs, not both')
        if kwargs:
            self._unfinished = {f: k for k, f in kwargs.items()}
            futures = list(kwargs.values())
        else:
            self._unfinished = {f: i for i, f in enumerate(args)}
            futures = args
        self._finished: Deque[Future] = collections.deque()
        self.current_index: Optional[Union[int, str]] = None
        self.current_future: Optional[Future] = None
        self._running_future: Optional[Future] = None
        for future in futures:
            future_add_done_callback(future, self._done_callback)

    def done(self) -> bool:
        if self._finished or self._unfinished:
            return False
        self.current_index = self.current_future = None
        return True

    def next(self) -> Future:
        self._running_future = Future()
        if self._finished:
            return self._return_result(self._finished.popleft())
        return self._running_future

    def _done_callback(self, done: Future) -> None:
        if self._running_future and (not self._running_future.done()):
            self._return_result(done)
        else:
            self._finished.append(done)

    def _return_result(self, done: Future) -> Future:
        if self._running_future is None:
            raise Exception('no future is running')
        chain_future(done, self._running_future)
        res = self._running_future
        self._running_future = None
        self.current_future = done
        self.current_index = self._unfinished.pop(done)
        return res

    def __aiter__(self) -> 'WaitIterator':
        return self

    def __anext__(self) -> Future:
        if self.done():
            raise getattr(builtins, 'StopAsyncIteration')()
        return self.next()

def multi(children: Union[List[Awaitable], Dict[Any, Awaitable]], quiet_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = ()) -> Future:
    return multi_future(children, quiet_exceptions=quiet_exceptions)
Multi = multi

def multi_future(children: Union[List[Awaitable], Dict[Any, Awaitable]], quiet_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = ()) -> Future:
    if isinstance(children, dict):
        keys = list(children.keys())
        children_seq = children.values()
    else:
        keys = None
        children_seq = children
    children_futs = list(map(convert_yielded, children_seq))
    assert all((is_future(i) or isinstance(i, _NullFuture) for i in children_futs))
    unfinished_children = set(children_futs)
    future = _create_future()
    if not children_futs:
        future_set_result_unless_cancelled(future, {} if keys is not None else [])

    def callback(fut: Future) -> None:
        unfinished_children.remove(fut)
        if not unfinished_children:
            result_list = []
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
    listening: Set[Future] = set()
    for f in children_futs:
        if f not in listening:
            listening.add(f)
            future_add_done_callback(f, callback)
    return future

def maybe_future(x: Any) -> Future:
    if is_future(x):
        return x
    else:
        fut = _create_future()
        fut.set_result(x)
        return fut

def with_timeout(timeout: Union[float, datetime.timedelta], future: Awaitable, quiet_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = ()) -> Future:
    future_converted = convert_yielded(future)
    result = _create_future()
    chain_future(future_converted, result)
    io_loop = IOLoop.current()

    def error_callback(future: Future) -> None:
        try:
            future.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not isinstance(e, quiet_exceptions):
                app_log.error('Exception in Future %r after timeout', future, exc_info=True)

    def timeout_callback() -> None:
        if not result.done():
            result.set_exception(TimeoutError('Timeout'))
        future_add_done_callback(future_converted, error_callback)
    timeout_handle = io_loop.add_timeout(timeout, timeout_callback)
    if isinstance(future_converted, Future):
        future_add_done_callback(future_converted, lambda future: io_loop.remove_timeout(timeout_handle))
    else:
        io_loop.add_future(future_converted, lambda future: io_loop.remove_timeout(timeout_handle))
    return result

def sleep(duration: float) -> Future:
    f = _create_future()
    IOLoop.current().call_later(duration, lambda: future_set_result_unless_cancelled(f, None))
    return f

class _NullFuture:
    def result(self) -> None:
        return None

    def done(self) -> bool:
        return True
_null_future = typing.cast(Future, _NullFuture())
moment = typing.cast(Future, _NullFuture())
moment.__doc__ = 'A special object which may be yielded to allow the IOLoop to run for\none iteration.\n\nThis is not needed in normal use but it can be helpful in long-running\ncoroutines that are likely to yield Futures that are ready instantly.\n\nUsage: ``yield gen.moment``\n\nIn native coroutines, the equivalent of ``yield gen.moment`` is\n``await asyncio.sleep(0)``.\n\n.. versionadded:: 4.0\n\n.. deprecated:: 4.5\n   ``yield None`` (or ``yield`` with no argument) is now equivalent to\n    ``yield gen.moment``.\n'

class Runner:
    def __init__(self, ctx_run: Callable[..., Any], gen: Generator, result_future: Future, first_yielded: Any) -> None:
        self.ctx_run = ctx_run
        self.gen = gen
        self.result_future = result_future
        self.future: Optional[Future] = _null_future
        self.running = False
        self.finished = False
        self.io_loop = IOLoop.current()
        if self.ctx_run(self.handle_yield, first_yielded):
            gen = result_future = first_yielded = None
            self.ctx_run(self.run)

    def run(self) -> None:
        if self.running or self.finished:
            return
        try:
            self.running = True
            while True:
                future = self.future
                if future is None:
                    raise Exception('No pending future')
                if not future.done():
                    return
                self.future = None
                try:
                    try:
                        value = future.result()
                    except Exception as e:
                        exc = e
                    else:
                        exc = None
                    finally:
                        future = None
                    if exc is not None:
                        try:
                            yielded = self.gen.throw(exc)
                        finally:
                            del exc
                    else:
                        yielded = self.gen.send(value)
                except (StopIteration, Return) as e:
                    self.finished = True
                    self.future = _null_future
                    future_set_result_unless_cancelled(self.result_future, _value_from_stopiteration(e))
                    self.result_future = None
                    return
                except Exception:
                    self.finished = True
                    self.future = _null_future
                    future_set_exc_info(self.result_future, sys.exc_info())
                    self.result_future = None
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
                f = None
                self.ctx_run(self.run)
            self.io_loop.add_future(self.future, inner)
            return False
        return True

    def handle_exception(self, typ: Type[BaseException], value: BaseException, tb: types.TracebackType) -> bool:
        if not self.running and (not self.finished):
            self.future = Future()
            future_set_exc_info(self.future, (typ, value, tb))
            self.ctx_run(self.run)
            return True
        else:
            return False

def _wrap_awaitable(awaitable: Awaitable) -> Future:
    fut = asyncio.ensure_future(awaitable)
    loop = IOLoop.current()
    loop._register_task(fut)
    fut.add_done_callback(lambda f: loop._unregister_task(f))
    return fut

def convert_yielded(yielded: Any) -> Future:
    if yielded is None or yielded is moment:
        return moment
    elif yielded is _null_future:
        return _null_future
    elif isinstance(yielded, (list, dict)):
        return multi(yielded)
    elif is_future(yielded):
        return typing.cast(Future, yielded)
    elif isawaitable(yielded):
        return _wrap_awaitable(yielded)
    else:
        raise BadYieldError(f'yielded unknown object {yielded!r}')
convert_yielded = singledispatch(convert_yielded)
