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

def _fake_ctx_run(f: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return f(*args, **kwargs)

@overload
def coroutine(func: Callable[..., Any]) -> Callable[..., Future]:
    ...

@overload
def coroutine(func: Callable[..., Any]) -> Callable[..., Future]:
    ...

def coroutine(func: Callable[..., Any]) -> Callable[..., Future]:
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

    def __init__(self, value: Any = None):
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
    _unfinished: Dict[Future, int] = {}

    def __init__(self, *args: Awaitable, **kwargs: Awaitable):
        if args and kwargs:
            raise ValueError('You must provide args or kwargs, not both')
        if kwargs:
            self._unfinished = {f: k for k, f in kwargs.items()}
            futures = list(kwargs.values())
        else:
            self._unfinished = {f: i for i, f in enumerate(args)}
            futures = args
        self._finished: Deque[Future] = collections.deque()
        self.current_index: Optional[int] = None
        self.current_future: Optional[Future] = None
        self._running_future: Optional[Future] = None
        for future in futures:
            future_add_done_callback(future, self._done_callback)

    def done(self) -> bool:
        """Returns True if this iterator has no more results."""
        if self._finished or self._unfinished:
            return False
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

    def _done_callback(self, done: Future):
        if self._running_future and (not self._running_future.done()):
            self._return_result(done)
        else:
            self._finished.append(done)

    def _return_result(self, done: Future) -> Future:
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

    def __aiter__(self) -> 'WaitIterator':
        return self

    def __anext__(self) -> Awaitable:
        if self.done():
            raise getattr(builtins, 'StopAsyncIteration')()
        return self.next()

def multi(children: Union[List[Awaitable], Dict[Any, Awaitable]], quiet_exceptions: Tuple[Type[Exception], ...] = ()) -> Future:
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

def multi_future(children: Union[List[Awaitable], Dict[Any, Awaitable]], quiet_exceptions: Tuple[Type[Exception], ...] = ()) -> Future:
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
    if isinstance(children, Dict[Any, Awaitable]):
        keys = list(children.keys())
        children_seq = children.values()
    else:
        keys = None
        children_seq = children
    children_futs = list(map(convert_yielded, children_seq))
    assert all((is_future(i) or isinstance(i, _NullFuture) for i in children_futs))
    unfinished_children: Set[Future] = set(children_futs)
    future = Future()
    if not children_futs:
        future_set_result_unless_cancelled(future, {} if keys is not None else [])

    def callback(fut: Future):
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
        fut = Future()
        fut.set_result(x)
        return fut

def with_timeout(timeout: Any, future: Awaitable, quiet_exceptions: Tuple[Type[Exception], ...] = ()) -> Future:
    """Wraps a `.Future` (or other yieldable object) in a timeout.

    Raises `tornado.util.TimeoutError` if the input future does not
    complete before ``timeout``, which may be specified in any form
    allowed by `.IOLoop.add_timeout` (i.e. a `datetime.timedelta` or
    an absolute time relative to `.IOLoop.time`)

    If the wrapped `.Future` fails after it has timed out, the exception
    will be logged unless it is either of a type contained in
    ``quiet_exceptions`` (which may be an exception type or a sequence of
    types), or an ``asyncio.CancelledError``.

    The wrapped `.Future` is not canceled when the timeout expires,
    permitting it to be reused. `asyncio.wait_for` is similar to this
    function but it does cancel the wrapped `.Future` on timeout.

    .. versionadded:: 4.0

    .. versionchanged:: 4.1
       Added the ``quiet_exceptions`` argument and the logging of unhandled
       exceptions.

    .. versionchanged:: 4.4
       Added support for yieldable objects other than `.Future`.

    .. versionchanged:: 6.0.3
       ``asyncio.CancelledError`` is now always considered "quiet".

    .. versionchanged:: 6.2
       ``tornado.util.TimeoutError`` is now an alias to ``asyncio.TimeoutError``.

    """
    future_converted = convert_yielded(future)
    result = Future()
    chain_future(future_converted, result)
    io_loop = IOLoop.current()

    def error_callback(future: Future):
        try:
            future.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not isinstance(e, quiet_exceptions):
                app_log.error('Exception in Future %r after timeout', future, exc_info=True)

    def timeout_callback():
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
    """Return a `.Future` that resolves after the given number of seconds.

    When used with ``yield`` in a coroutine, this is a non-blocking
    analogue to `time.sleep` (which should not be used in coroutines
    because it is blocking)::

        yield gen.sleep(0.5)

    Note that calling this function on its own does nothing; you must
    wait on the `.Future` it returns (usually by yielding it).

    .. versionadded:: 4.1
    """
    f = Future()
    IOLoop.current().call_later(duration, lambda: future_set_result_unless_cancelled(f, None))
    return f

class _NullFuture:
    """_NullFuture resembles a Future that finished with a result of None.

    It's not actually a `Future` to avoid depending on a particular event loop.
    Handled as a special case in the coroutine runner.

    We lie and tell the type checker that a _NullFuture is a Future so
    we don't have to leak _NullFuture into lots of public APIs. But
    this means that the type checker can't warn us when we're passing
    a _NullFuture into a code path that doesn't understand what to do
    with it.
    """

    def result(self) -> None:
        return None

    def done(self) -> bool:
        return True
_null_future = typing.cast(Future, _NullFuture())
moment = typing.cast(Future, _NullFuture())
moment.__doc__ = 'A special object which may be yielded to allow the IOLoop to run for\none iteration.\n\nThis is not needed in normal use but it can be helpful in long-running\ncoroutines that are likely to yield Futures that are ready instantly.\n\nUsage: ``yield gen.moment``\n\nIn native coroutines, the equivalent of ``yield gen.moment`` is\n``await asyncio.sleep(0)``.\n\n.. versionadded:: 4.0\n\n.. deprecated:: 4.5\n   ``yield None`` (or ``yield`` with no argument) is now equivalent to\n    ``yield gen.moment``.\n'

class Runner:
    """Internal implementation of `tornado.gen.coroutine`.

    Maintains information about pending callbacks and their results.

    The results of the generator are stored in ``result_future`` (a
    `.Future`)
    """

    def __init__(self, ctx_run: Callable[..., Any], gen: Generator, result_future: Future, first_yielded: Awaitable):
        self.ctx_run = ctx_run
        self.gen = gen
        self.result_future = result_future
        self.future: Future = _null_future
        self.running: bool = False
        self.finished: bool = False
        self.io_loop = IOLoop.current()

    def run(self) -> None:
        """Starts or resumes the generator, running until it reaches a
        yield point that is not ready.
        """
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

    def handle_yield(self, yielded: Awaitable) -> bool:
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

            def inner(f):
                f = None
                self.ctx_run(self.run)
            self.io_loop.add_future(self.future, inner)
            return False
        return True

    def handle_exception(self, typ: Type[Exception], value: Exception, tb: Tuple[type, Exception, str]) -> bool:
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

def convert_yielded(yielded: Awaitable) -> Future:
    """Convert a yielded object into a `.Future`.

    The default implementation accepts lists, dictionaries, and
    Futures. This has the side effect of starting any coroutines that
    did not start themselves, similar to `asyncio.ensure_future`.

    If the `~functools.singledispatch` library is available, this function
    may be extended to support additional types. For example::

        @convert_yielded.register(asyncio.Future)
        def _(asyncio_future):
            return tornado.platform.asyncio.to_tornado_future(asyncio_future)

    .. versionadded:: 4.1

    """
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
