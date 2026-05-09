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
from typing import Union, Any, Callable, List, Type, Tuple, Awaitable, Dict, overload
if typing.TYPE_CHECKING:
    from typing import Sequence, Deque, Optional, Set, Iterable
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

def _value_from_stopiteration(e):
    try:
        return e.value
    except AttributeError:
        pass
    try:
        return e.args[0]
    except (AttributeError, IndexError):
        return None

def _create_future():
    future = Future()
    source_traceback = getattr(future, '_source_traceback', ())
    while source_traceback:
        filename = source_traceback[-1][0]
        if filename == __file__:
            del source_traceback[-1]
        else:
            break
    return future

def _fake_ctx_run(f, *args, **kw):
    return f(*args, **kw)

@overload
def coroutine(func):
    ...

@overload
def coroutine(func):
    ...

def coroutine(func):
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
    def wrapper(*args, **kwargs):
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

def is_coroutine_function(func):
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

    def __init__(self, value=None):
        super().__init__()
        self.value = value
        self.args = (value,)

class WaitIterator:
    """Provides an iterator to yield the results of awaitables as they finish.

    Yielding a set of awaitables like this:

    ``results = yield [awaitable1, awaitable2]``

    pauses the coroutine until both ``awaitable1`` and ``awaitable2``
    return, and then restarts the coroutine with the results of both
    awaitables. If either awaitable raises an exception, the
    expression will raise that exception and all the results will be
    lost.

    If you need to get the result of each awaitable as soon as possible,
    or if you need the result of some awaitables even if others produce
    errors, you can use ``WaitIterator``::

      wait_iterator = gen.WaitIterator(awaitable1, awaitable2)
      while not wait_iterator.done():
          try:
              result = yield wait_iterator.next()
          except Exception as e:
              print("Error {} from {}".format(e, wait_iterator.current_future))
          else:
              print("Result {} received from {} at {}".format(
                  result, wait_iterator.current_future,
                  wait_iterator.current_index))

    Because results are returned as soon as they are available the
    output from the iterator *will not be in the same order as the
    input arguments*. If you need to know which future produced the
    current result, you can use the attributes
    ``WaitIterator.current_future``, or ``WaitIterator.current_index``
    to get the index of the awaitable from the input list. (if keyword
    arguments were used in the construction of the `WaitIterator`,
    ``current_index`` will use the corresponding keyword).

    `WaitIterator` implements the async iterator
    protocol, so it can be used with the ``async for`` statement (note
    that in this version the entire iteration is aborted if any value
    raises an exception, while the previous example can continue past
    individual errors)::

      async for result in gen.WaitIterator(future1, future2):
          print("Result {} received from {} at {}".format(
              result, wait_iterator.current_future,
              wait_iterator.current_index))

    .. versionadded:: 4.1

    .. versionchanged:: 4.3
       Added ``async for`` support in Python 3.5.

    """
    _unfinished = {}

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError('You must provide args or kwargs, not both')
        if kwargs:
            self._unfinished = {f: k for k, f in kwargs.items()}
            futures = list(kwargs.values())
        else:
            self._unfinished = {f: i for i, f in enumerate(args)}
            futures = args
        self._finished = collections.deque()
        self.current_index = None
        self.current_future = None
        self._running_future = None
        for future in futures:
            future_add_done_callback(future, self._done_callback)

    def done(self):
        """Returns True if this iterator has no more results."""
        if self._finished or self._unfinished:
            return False
        self.current_index = self.current_future = None
        return True

    def next(self):
        """Returns a `.Future` that will yield the next available result.

        Note that this `.Future` will not be the same object as any of
        the inputs.
        """
        self._running_future = Future()
        if self._finished:
            return self._return_result(self._finished.popleft())
        return self._running_future

    def _done_callback(self, done):
        if self._running_future and (not self._running_future.done()):
            self._return_result(done)
        else:
            self._finished.append(done)

    def _return_result(self, done):
        """Called set the returned future's state that of the future
        we yielded, and set the current future for the iterator.
        """
        if self._running_future is None:
            raise Exception('no future is running')
        chain_future(done, self._running_future)
        res = self._running_future
        self._running_future = None
        self.current_future = done
        self.current_index = self._unfinished.pop(done)
        return res

    def __aiter__(self):
        return self

    def __anext__(self):
        if self.done():
            raise getattr(builtins, 'StopAsyncIteration')()
        return self.next()

def multi(children, quiet_exceptions=()):
    """Runs multiple asynchronous operations in parallel.

    ``children`` may either be a list or a dict whose values are
    yieldable objects. ``multi()`` returns a new yieldable
    object that resolves to a parallel structure containing their
    results. If ``children`` is a list, the result is a list of
    results in the same order; if it is a dict, the result is a dict
    with the same keys.

    That is, ``results = yield multi(list_of_futures)`` is equivalent
    to::

        results = []
        for future in list_of_futures:
            results.append(yield future)

    If any children raise exceptions, ``multi()`` will raise the first
    one. All others will be logged, unless they are of types
    contained in the ``quiet_exceptions`` argument.

    In a ``yield``-based coroutine, it is not normally necessary to
    call this function directly, since the coroutine runner will
    do it automatically when a list or dict is yielded. However,
    it is necessary in ``await``-based coroutines, or to pass
    the ``quiet_exceptions`` argument.

    This function is available under the names ``multi()`` and ``Multi()``
    for historical reasons.

    Cancelling a `.Future` returned by ``multi()`` does not cancel its
    children. `asyncio.gather` is similar to ``multi()``, but it does
    cancel its children.

    .. versionchanged:: 4.2
       If multiple yieldables fail, any exceptions after the first
       (which is raised) will be logged. Added the ``quiet_exceptions``
       argument to suppress this logging for selected exception types.

    .. versionchanged:: 4.3
       Replaced the class ``Multi`` and the function ``multi_future``
       with a unified function ``multi``. Added support for yieldables
       other than ``YieldPoint`` and `.Future`.

    """
    return multi_future(children, quiet_exceptions=quiet_exceptions)
Multi = multi

def maybe_future(x):
    """Converts ``x`` into a `.Future`.

    If ``x`` is already a `.Future`, it is simply returned; otherwise
    it is wrapped in a new `.Future`.  This is suitable for use as
    ``result = yield gen.maybe_future(f())`` when you don't know whether
    ``f()`` returns a `.Future` or not.

    .. deprecated:: 4.3
       This function only handles ``Futures``, not other yieldable objects.
       Instead of `maybe_future`, check for the non-future result types
       you expect (often just ``None``), and ``yield`` anything unknown.
    """
    if is_future(x):
        return x
    else:
        fut = _create_future()
        fut.set_result(x)
        return fut

def with_timeout(timeout, future, quiet_exceptions=()):
    """Wraps a `.Future` (or other yieldable object) in a timeout.

    Raises `tornado.util.TimeoutError` if the input future does not
    complete before ``timeout``, which may be specified in any form
    allowed by `.IOLoop.add_timeout` (i.e. a `datetime.timedelta` or
    an absolute time relative to `.IOLoop.time`)

    If the wrapped `.Future