"""``tornado.gen`` implements generator-based coroutines.

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

from tornado.concurrent import (
    Future,
    is_future,
    chain_future,
    future_set_exc_info,
    future_add_done_callback,
    future_set_result_unless_cancelled,
)
from tornado.ioloop import IOLoop
from tornado.log import app_log
from tornado.util import TimeoutError

try:
    import contextvars
except ImportError:
    contextvars = None  # type: ignore

import typing
from typing import Union, Any, Callable, List, Type, Tuple, Awaitable, Dict, overload, Optional, Set, Iterable, Sequence, Deque, TypeVar, cast, AsyncIterator

if typing.TYPE_CHECKING:
    from typing import NoReturn

_T = TypeVar("_T")

_Yieldable = Union[
    None, Awaitable, List[Awaitable], Dict[Any, Awaitable], concurrent.futures.Future
]


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


def _value_from_stopiteration(e: Union[StopIteration, "Return"]) -> Any:
    try:
        # StopIteration has a value attribute beginning in py33.
        # So does our Return class.
        return e.value
    except AttributeError:
        pass
    try:
        # Cython backports coroutine functionality by putting the value in
        # e.args[0].
        return e.args[0]
    except (AttributeError, IndexError):
        return None


def _create_future() -> Future:
    future = Future()  # type: Future
    # Fixup asyncio debug info by removing extraneous stack entries
    source_traceback = getattr(future, "_source_traceback", ())
    while source_traceback:
        # Each traceback entry is equivalent to a
        # (filename, self.lineno, self.name, self.line) tuple
        filename = source_traceback[-1][0]
        if filename == __file__:
            del source_traceback[-1]
        else:
            break
    return future


def _fake_ctx_run(f: Callable[..., _T], *args: Any, **kw: Any) -> _T:
    return f(*args, **kw)


@overload
def coroutine(
    func: Callable[..., Generator[Any, Any, _T]]
) -> Callable[..., Future[_T]]: ...


@overload
def coroutine(func: Callable[..., _T]) -> Callable[..., Future[_T]]: ...


def coroutine(
    func: Union[Callable[..., Generator[Any, Any, _T]], Callable[..., _T]]
) -> Callable[..., Future[_T]]:
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
    def wrapper(*args: Any, **kwargs: Any) -> Future[_T]:
        future = _create_future()
        if contextvars is not None:
            ctx_run = contextvars.copy_context().run  # type: Callable
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
                # Avoid circular references
                future = None  # type: ignore
        else:
            if isinstance(result, Generator):
                # Inline the first iteration of Runner.run.  This lets us
                # avoid the cost of creating a Runner when the coroutine
                # never actually yields, which in turn allows us to
                # use "optional" coroutines in critical path code without
                # performance penalty for the synchronous case.
                try:
                    yielded = ctx_run(next, result)
                except (StopIteration, Return) as e:
                    future_set_result_unless_cancelled(
                        future, _value_from_stopiteration(e)
                    )
                except Exception:
                    future_set_exc_info(future, sys.exc_info())
                else:
                    # Provide strong references to Runner objects as long
                    # as their result future objects also have strong
                    # references (typically from the parent coroutine's
                    # Runner). This keeps the coroutine's Runner alive.
                    # We do this by exploiting the public API
                    # add_done_callback() instead of putting a private
                    # attribute on the Future.
                    # (GitHub issues #1769, #2229).
                    runner = Runner(ctx_run, result, future, yielded)
                    future.add_done_callback(lambda _: runner)
                yielded = None
                try:
                    return future
                finally:
                    # Subtle memory optimization: if next() raised an exception,
                    # the future's exc_info contains a traceback which
                    # includes this stack frame.  This creates a cycle,
                    # which will be collected at the next full GC but has
                    # been shown to greatly increase memory usage of
                    # benchmarks (relative to the refcount-based scheme
                    # used in the absence of cycles).  We can avoid the
                    # cycle by clearing the local variable after we return it.
                    future = None  # type: ignore
        future_set_result_unless_cancelled(future, result)
        return future

    wrapper.__wrapped__ = func  # type: ignore
    wrapper.__tornado_coroutine__ = True  # type: ignore
    return wrapper


def is_coroutine_function(func: Any) -> bool:
    """Return whether *func* is a coroutine function, i.e. a function
    wrapped with `~.gen.coroutine`.

    .. versionadded:: 4.5
    """
    return getattr(func, "__tornado_coroutine__", False)


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

    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value
        # Cython recognizes subclasses of StopIteration with a .args tuple.
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

    _unfinished: Dict[Future, Union[int, str]] = {}

    def __init__(self, *args: Future, **kwargs: Future) -> None:
        if args and kwargs:
            raise ValueError("You must provide args or kwargs, not both")

        if kwargs:
            self._unfinished = {f: k for (k, f) in kwargs.items()}
            futures = list(kwargs.values())  # type: Sequence[Future]
        else:
            self._unfinished = {f: i for (i, f) in enumerate(args)}
            futures = args

        self._finished: Deque[Future] = collections.deque()
        self.current_index: Optional[Union[str, int]] = None
        self.current_future: Optional[Future] = None
        self._running_future: Optional[Future] = None

        for future in futures:
            future_add_done_callback(future, self._done_callback)

    def done(self) -> bool:
        """Returns True if this iterator has no more results."""
        if self._finished or self._unfinished:
            return False
        # Clear the 'current' values when iteration is done.
        self.current_index = self.current_future = None
        return True

    def next(self) -> Future:
        """Returns a `.Future` that will yield the next available result.

        Note that this `.Future` will not be the same object as any of
        the inputs.
        """
        self._running_future = Future()

        if self._finished:
            return self._return_result(self._finished.popleft())

        return self._running_future

    def _done_callback(self, done: Future) -> None:
        if self._running_future and not self._running_future.done():
            self._return_result(done)
        else:
            self._finished.append(done)

    def _return_result(self, done: Future) -> Future:
        """Called set the returned future's state that of the future
        we yielded, and set the current future for the iterator.
        """
        if self._running_future is None:
            raise Exception("no future is running")
        chain_future(done, self._running_future)

        res = self._running_future
        self._running_future = None
        self.current_future = done
        self.current_index = self._unfinished.pop(done)

        return res

    def __aiter__(self) -> AsyncIterator:
        return self

    def __anext__(self) -> Future:
        if self.done():
            # Lookup by name to silence pyflakes on older versions.
            raise getattr(builtins, "StopAsyncIteration")()
        return self.next()


def multi(
    children: Union[List[_Yieldable], Dict[Any, _Yieldable]],
    quiet_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (),
) -> Union[Future[List], Future[Dict]]:
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


def multi_future(
    children: Union[List[_Yieldable], Dict[Any, _Yieldable]],
    quiet_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (),
) -> Union[Future[List], Future[Dict]]:
    """Wait for multiple asynchronous futures in parallel.

    Since Tornado 6.0, this function is exactly the same as `multi`.

    .. versionadded:: 4.0

    .. versionchanged:: 4.2
       If multiple ``Futures`` fail, any exceptions after the first (which is
       raised) will be logged. Added the ``quiet_exceptions``
       argument to suppress this logging for selected exception types.

    .. deprecated:: 4.3
       Use `multi` instead.
    """
    if isinstance(children, dict):
        keys = list(children.keys())  # type: Optional[List]
        children_seq = children.values()  # type: Iterable
    else:
        keys = None
        children_seq = children
    children_futs = list(map(convert_yielded, children_seq))
    assert all(is_future(i) or isinstance(i, _NullFuture) for i in children_futs)
    unfinished_