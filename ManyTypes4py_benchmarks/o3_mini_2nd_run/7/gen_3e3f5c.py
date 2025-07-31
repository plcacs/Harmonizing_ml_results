#!/usr/bin/env python3
"""
tornado.gen implements generator-based coroutines.

.. note::

   The "decorator and generator" approach in this module is a
   precursor to native coroutines (using ``async def`` and ``await``)
   which were introduced in Python 3.5. Applications that do not
   require compatibility with older versions of Python should use
   native coroutines instead. Some parts of this module are still
   useful with native coroutines, notably `multi`, `sleep`,
   `WaitIterator`, and `with_timeout`. Some of these functions have
   counterparts in the `asyncio` module which may be used as well,
   although the two may not necessarily be 100% compatible.

Coroutines provide an easier way to work in an asynchronous
environment than chaining callbacks. Code using coroutines is
technically asynchronous, but it is written as a single generator
instead of a collection of separate functions.

For example, here's a coroutine-based handler:

.. testcode::

    class GenAsyncHandler(RequestHandler):
        @gen.coroutine
        def get(self):
            http_client = AsyncHTTPClient()
            response = yield http_client.fetch("http://example.com")
            do_something_with_response(response)
            self.render("template.html")

Asynchronous functions in Tornado return an ``Awaitable`` or `.Future`;
yielding this object returns its result.

You can also yield a list or dict of other yieldable objects, which
will be started at the same time and run in parallel; a list or dict
of results will be returned when they are all finished:

.. testcode::

    @gen.coroutine
    def get(self):
        http_client = AsyncHTTPClient()
        response1, response2 = yield [http_client.fetch(url1),
                                      http_client.fetch(url2)]
        response_dict = yield dict(response3=http_client.fetch(url3),
                                   response4=http_client.fetch(url4))
        response3 = response_dict['response3']
        response4 = response_dict['response4']

If ``tornado.platform.twisted`` is imported, it is also possible to
yield Twisted's ``Deferred`` objects. See the `convert_yielded`
function to extend this mechanism.

.. versionchanged:: 3.2
   Dict support added.

.. versionchanged:: 4.1
   Support added for yielding ``asyncio`` Futures and Twisted Deferreds
   via ``singledispatch``.

"""
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
from typing import Any, Callable, Dict, List, Tuple, Union, overload, Type, Optional

from tornado.concurrent import Future, is_future, chain_future, future_set_exc_info, future_add_done_callback, future_set_result_unless_cancelled
from tornado.ioloop import IOLoop
from tornado.log import app_log
from tornado.util import TimeoutError

try:
    import contextvars
except ImportError:
    contextvars = None

import typing

if typing.TYPE_CHECKING:
    from typing import Sequence, Deque, Set, Iterable

_T = typing.TypeVar('_T')
_Yieldable = Union[None, asyncio.Future, List[asyncio.Future], Dict[Any, asyncio.Future], concurrent.futures.Future]

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

def _value_from_stopiteration(e: Exception) -> Any:
    try:
        return e.value
    except AttributeError:
        pass
    try:
        return e.args[0]
    except (AttributeError, IndexError):
        return None

def _create_future() -> Future[Any]:
    future: Future[Any] = Future()
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
def coroutine(func: Callable[..., Any]) -> Callable[..., Future[Any]]:
    ...

@overload
def coroutine(func: Any) -> Any:
    ...

def coroutine(func: Callable[..., Any]) -> Callable[..., Future[Any]]:
    """Decorator for asynchronous generators.

    For compatibility with older versions of Python, coroutines may
    also "return" by raising the special exception `Return(value)
    <Return>`.

    Functions with this decorator return a `.Future`.

    .. warning::

       When exceptions occur inside a coroutine, the exception
       information will be stored in the `.Future` object. You must
       examine the result of the `.Future` object, or the exception
       may go unnoticed by your code. This means yielding the function
       if called from another coroutine, using something like
       `.IOLoop.run_sync` for top-level calls, or passing the `.Future`
       to `.IOLoop.add_future`.

    .. versionchanged:: 6.0

       The ``callback`` argument was removed. Use the returned
       awaitable object instead.

    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = _create_future()
        if contextvars is not None:
            ctx_run: Callable[..., Any] = contextvars.copy_context().run
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
    wrapper.__tornado_coroutine__ = True
    return wrapper

def is_coroutine_function(func: Callable[..., Any]) -> bool:
    """Return whether *func* is a coroutine function, i.e. a function
    wrapped with `~.gen.coroutine`.

    .. versionadded:: 4.5
    """
    return getattr(func, '__tornado_coroutine__', False)

class Return(Exception):
    """Special exception to return a value from a `coroutine`.

    This exception exists for compatibility with older versions of
    Python (before 3.3). In newer code use the ``return`` statement
    instead.

    If this exception is raised, its value argument is used as the
    result of the coroutine::

        @gen.coroutine
        def fetch_json(url):
            response = yield AsyncHTTPClient().fetch(url)
            raise gen.Return(json_decode(response.body))

    By analogy with the return statement, the value argument is optional.
    """
    value: Any

    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value
        self.args = (value,)

class WaitIterator:
    """Provides an iterator to yield the results of awaitables as they finish.
    """
    _unfinished: Dict[Any, Any]

    def __init__(self, *args: Future[Any], **kwargs: Future[Any]) -> None:
        if args and kwargs:
            raise ValueError('You must provide args or kwargs, not both')
        if kwargs:
            self._unfinished = {f: k for k, f in kwargs.items()}
            futures: List[Future[Any]] = list(kwargs.values())
        else:
            self._unfinished = {f: i for i, f in enumerate(args)}
            futures = list(args)
        self._finished: collections.deque = collections.deque()
        self.current_index: Optional[Any] = None
        self.current_future: Optional[Future[Any]] = None
        self._running_future: Optional[Future[Any]] = None
        for future in futures:
            future_add_done_callback(future, self._done_callback)

    def done(self) -> bool:
        """Returns True if this iterator has no more results."""
        if self._finished or self._unfinished:
            return False
        self.current_index = None
        self.current_future = None
        return True

    def next(self) -> Future[Any]:
        """Returns a `.Future` that will yield the next available result.
        """
        self._running_future = _create_future()
        if self._finished:
            return self._return_result(self._finished.popleft())
        return self._running_future

    def _done_callback(self, done: Future[Any]) -> None:
        if self._running_future and (not self._running_future.done()):
            self._return_result(done)
        else:
            self._finished.append(done)

    def _return_result(self, done: Future[Any]) -> Future[Any]:
        """Sets the returned future's state to that of the future
        we yielded, and set the current future for the iterator.
        """
        if self._running_future is None:
            raise Exception('no future is running')
        chain_future(done, self._running_future)
        res: Future[Any] = self._running_future
        self._running_future = None
        self.current_future = done
        self.current_index = self._unfinished.pop(done)
        return res

    def __aiter__(self) -> "WaitIterator":
        return self

    async def __anext__(self) -> Any:
        if self.done():
            raise getattr(builtins, 'StopAsyncIteration')()
        return await self.next()

def multi(children: Union[List[Any], Dict[Any, Any]], quiet_exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ()) -> Future[Any]:
    """Runs multiple asynchronous operations in parallel.
    """
    return multi_future(children, quiet_exceptions=quiet_exceptions)
Multi = multi

def multi_future(children: Union[List[Any], Dict[Any, Any]], quiet_exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ()) -> Future[Any]:
    """Wait for multiple asynchronous futures in parallel.
    """
    if isinstance(children, dict):
        keys: List[Any] = list(children.keys())
        children_seq = children.values()
    else:
        keys = None
        children_seq = children
    children_futs: List[Future[Any]] = list(map(convert_yielded, children_seq))
    assert all((is_future(i) or isinstance(i, _NullFuture) for i in children_futs))
    unfinished_children = set(children_futs)
    future: Future[Any] = _create_future()
    if not children_futs:
        future_set_result_unless_cancelled(future, {} if keys is not None else [])
    def callback(fut: Future[Any]) -> None:
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
    listening: set = set()
    for f in children_futs:
        if f not in listening:
            listening.add(f)
            future_add_done_callback(f, callback)
    return future

def maybe_future(x: Any) -> Future[Any]:
    """Converts ``x`` into a `.Future`.
    """
    if is_future(x):
        return x
    else:
        fut: Future[Any] = _create_future()
        fut.set_result(x)
        return fut

def with_timeout(timeout: Union[datetime.timedelta, float, int],
                 future: Union[Future[Any], asyncio.Future, Any],
                 quiet_exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ()) -> Future[Any]:
    """Wraps a `.Future` (or other yieldable object) in a timeout.
    """
    future_converted: Future[Any] = convert_yielded(future)
    result: Future[Any] = _create_future()
    chain_future(future_converted, result)
    io_loop: IOLoop = IOLoop.current()
    def error_callback(fut: Future[Any]) -> None:
        try:
            fut.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not isinstance(e, quiet_exceptions):
                app_log.error('Exception in Future %r after timeout', fut, exc_info=True)
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

def sleep(duration: float) -> Future[None]:
    """Return a `.Future` that resolves after the given number of seconds.
    """
    f: Future[None] = _create_future()
    IOLoop.current().call_later(duration, lambda: future_set_result_unless_cancelled(f, None))
    return f

class _NullFuture:
    """_NullFuture resembles a Future that finished with a result of None.
    """
    def result(self) -> None:
        return None
    def done(self) -> bool:
        return True

_null_future: Future[Any] = typing.cast(Future[Any], _NullFuture())
moment: Future[Any] = typing.cast(Future[Any], _NullFuture())
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
    '    ``yield gen.moment``.\n'
)

class Runner:
    """Internal implementation of `tornado.gen.coroutine`.
    """
    def __init__(self,
                 ctx_run: Callable[..., Any],
                 gen: Generator[Any, Any, Any],
                 result_future: Future[Any],
                 first_yielded: Any) -> None:
        self.ctx_run: Callable[..., Any] = ctx_run
        self.gen: Generator[Any, Any, Any] = gen
        self.result_future: Future[Any] = result_future
        self.future: Union[Future[Any], _NullFuture] = _null_future
        self.running: bool = False
        self.finished: bool = False
        self.io_loop: IOLoop = IOLoop.current()
        if self.ctx_run(self.handle_yield, first_yielded):
            gen = result_future = first_yielded = None  # type: ignore
            self.ctx_run(self.run)

    def run(self) -> None:
        """Starts or resumes the generator, running until it reaches a
        yield point that is not ready.
        """
        if self.running or self.finished:
            return
        try:
            self.running = True
            while True:
                future_current: Union[Future[Any], None] = self.future
                if future_current is None:
                    raise Exception('No pending future')
                if not future_current.done():
                    return
                self.future = None  # type: ignore
                try:
                    try:
                        value = future_current.result()
                    except Exception as e:
                        exc = e
                    else:
                        exc = None
                    finally:
                        future_current = None
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
                    self.result_future = None  # type: ignore
                    return
                except Exception:
                    self.finished = True
                    self.future = _null_future
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
            def inner(f: Future[Any]) -> None:
                self.ctx_run(self.run)
            self.io_loop.add_future(self.future, inner)
            return False
        return True

    def handle_exception(self, typ: Any, value: Any, tb: Any) -> bool:
        if not self.running and (not self.finished):
            self.future = Future()
            future_set_exc_info(self.future, (typ, value, tb))
            self.ctx_run(self.run)
            return True
        else:
            return False

def _wrap_awaitable(awaitable: Any) -> Future[Any]:
    fut: asyncio.Future = asyncio.ensure_future(awaitable)
    loop: IOLoop = IOLoop.current()
    loop._register_task(fut)
    fut.add_done_callback(lambda f: loop._unregister_task(f))
    return fut

def convert_yielded(yielded: Any) -> Future[Any]:
    """Convert a yielded object into a `.Future`.
    """
    if yielded is None or yielded is moment:
        return moment
    elif yielded is _null_future:
        return _null_future
    elif isinstance(yielded, (list, dict)):
        return multi(yielded)
    elif is_future(yielded):
        return typing.cast(Future[Any], yielded)
    elif isawaitable(yielded):
        return _wrap_awaitable(yielded)
    else:
        raise BadYieldError(f'yielded unknown object {yielded!r}')

convert_yielded = singledispatch(convert_yielded)  # type: ignore
