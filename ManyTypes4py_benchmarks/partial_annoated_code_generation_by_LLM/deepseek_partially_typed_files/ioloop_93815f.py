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
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable, Dict, List, Set, overload
if typing.TYPE_CHECKING:
    from typing_extensions import Protocol
else:
    Protocol = object

class _Selectable(Protocol):

    def fileno(self) -> int:
        pass

    def close(self) -> None:
        pass
_T = TypeVar('_T')
_S = TypeVar('_S', bound=_Selectable)

class IOLoop(Configurable):
    """An I/O event loop.

    As of Tornado 6.0, `IOLoop` is a wrapper around the `asyncio` event loop.

    Example usage for a simple TCP server:

    .. testcode::

        import asyncio
        import errno
        import functools
        import socket

        import tornado
        from tornado.iostream import IOStream

        async def handle_connection(connection, address):
            stream = IOStream(connection)
            message = await stream.read_until_close()
            print("message from client:", message.decode().strip())

        def connection_ready(sock, fd, events):
            while True:
                try:
                    connection, address = sock.accept()
                except BlockingIOError:
                    return
                connection.setblocking(0)
                io_loop = tornado.ioloop.IOLoop.current()
                io_loop.spawn_callback(handle_connection, connection, address)

        async def main():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setblocking(0)
            sock.bind(("", 8888))
            sock.listen(128)

            io_loop = tornado.ioloop.IOLoop.current()
            callback = functools.partial(connection_ready, sock)
            io_loop.add_handler(sock.fileno(), callback, io_loop.READ)
            await asyncio.Event().wait()

        if __name__ == "__main__":
            asyncio.run(main())

    Most applications should not attempt to construct an `IOLoop` directly,
    and instead initialize the `asyncio` event loop and use `IOLoop.current()`.
    In some cases, such as in test frameworks when initializing an `IOLoop`
    to be run in a secondary thread, it may be appropriate to construct
    an `IOLoop` with ``IOLoop(make_current=False)``.

    In general, an `IOLoop` cannot survive a fork or be shared across processes
    in any way. When multiple processes are being used, each process should
    create its own `IOLoop`, which also implies that any objects which depend on
    the `IOLoop` (such as `.AsyncHTTPClient`) must also be created in the child
    processes. As a guideline, anything that starts processes (including the
    `tornado.process` and `multiprocessing` modules) should do so as early as
    possible, ideally the first thing the application does after loading its
    configuration, and *before* any calls to `.IOLoop.start` or `asyncio.run`.

    .. versionchanged:: 4.2
       Added the ``make_current`` keyword argument to the `IOLoop`
       constructor.

    .. versionchanged:: 5.0

       Uses the `asyncio` event loop by default. The ``IOLoop.configure`` method
       cannot be used on Python 3 except to redundantly specify the `asyncio`
       event loop.

    .. versionchanged:: 6.3
       ``make_current=True`` is now the default when creating an IOLoop -
       previously the default was to make the event loop current if there wasn't
       already a current one.
    """
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
        """Deprecated alias for `IOLoop.current()`.

        .. versionchanged:: 5.0

           Previously, this method returned a global singleton
           `IOLoop`, in contrast with the per-thread `IOLoop` returned
           by `current()`. In nearly all cases the two were the same
           (when they differed, it was generally used from non-Tornado
           threads to communicate back to the main thread's `IOLoop`).
           This distinction is not present in `asyncio`, so in order
           to facilitate integration with that package `instance()`
           was changed to be an alias to `current()`. Applications
           using the cross-thread communications aspect of
           `instance()` should instead set their own global variable
           to point to the `IOLoop` they want to use.

        .. deprecated:: 5.0
        """
        return IOLoop.current()

    def install(self) -> None:
        """Deprecated alias for `make_current()`.

        .. versionchanged:: 5.0

           Previously, this method would set this `IOLoop` as the
           global singleton used by `IOLoop.instance()`. Now that
           `instance()` is an alias for `current()`, `install()`
           is an alias for `make_current()`.

        .. deprecated:: 5.0
        """
        self.make_current()

    @staticmethod
    def clear_instance() -> None:
        """Deprecated alias for `clear_current()`.

        .. versionchanged:: 5.0

           Previously, this method would clear the `IOLoop` used as
           the global singleton by `IOLoop.instance()`. Now that
           `instance()` is an alias for `current()`,
           `clear_instance()` is an alias for `clear_current()`.

        .. deprecated:: 5.0

        """
        IOLoop.clear_current()

    @overload
    @staticmethod
    def current() -> 'IOLoop':
        pass

    @overload
    @staticmethod
    def current(instance: bool) -> Optional['IOLoop']:
        pass

    @staticmethod
    def current(instance: bool = True) -> Optional['IOLoop']:
        """Returns the current thread's `IOLoop`.

        If an `IOLoop` is currently running or has been marked as
        current by `make_current`, returns that instance.  If there is
        no current `IOLoop` and ``instance`` is true, creates one.

        .. versionchanged:: 4.1
           Added ``instance`` argument to control the fallback to
           `IOLoop.instance()`.
        .. versionchanged:: 5.0
           On Python 3, control of the current `IOLoop` is delegated
           to `asyncio`, with this and other methods as pass-through accessors.
           The ``instance`` argument now controls whether an `IOLoop`
           is created automatically when there is none, instead of
           whether we fall back to `IOLoop.instance()` (which is now
           an alias for this method). ``instance=False`` is deprecated,
           since even if we do not create an `IOLoop`, this method
           may initialize the asyncio loop.

        .. deprecated:: 6.2
           It is deprecated to call ``IOLoop.current()`` when no `asyncio`
           event loop is running.
        """
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
        """Makes this the `IOLoop` for the current thread.

        An `IOLoop` automatically becomes current for its thread
        when it is started, but it is sometimes useful to call
        `make_current` explicitly before starting the `IOLoop`,
        so that code run at startup time can find the right
        instance.

        .. versionchanged:: 4.1
           An `IOLoop` created while there is no current `IOLoop`
           will automatically become current.

        .. versionchanged:: 5.0
           This method also sets the current `asyncio` event loop.

        .. deprecated:: 6.2
           Setting and clearing the current event loop through Tornado is
           deprecated. Use ``asyncio.set_event_loop`` instead if you need this.
        """
        warnings.warn('make_current is deprecated; start the event loop first', DeprecationWarning, stacklevel=2)
        self._make_current()

    def _make_current(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def clear_current() -> None:
        """Clears the `IOLoop` for the current thread.

        Intended primarily for use by test frameworks in between tests.

        .. versionchanged:: 5.0
           This method also clears the current `asyncio` event loop.
        .. deprecated:: 6.2
        """
        warnings.warn('clear_current is deprecated', DeprecationWarning, stacklevel=2)
        IOLoop._clear_current()

    @staticmethod
    def _clear_current() -> None:
        old = IOLoop.current(instance=False)
        if old is not None:
            old._clear_current_hook()

    def _clear_current_hook(self) -> None:
        """Instance method called when an IOLoop ceases to be current.

        May be overridden by subclasses as a counterpart to make_current.
        """
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
        """Closes the `IOLoop`, freeing any resources used.

        If ``all_fds`` is true, all file descriptors registered on the
        IOLoop will be closed (not just the ones created by the
        `IOLoop` itself).

        Many applications will only use a single `IOLoop` that runs for the
        entire lifetime of the process.  In that case closing the `IOLoop`
        is not necessary since everything will be cleaned up when the
        process exits.  `IOLoop.close` is provided mainly for scenarios
        such as unit tests, which create and destroy a large number of
        ``IOLoops``.

        An `IOLoop` must be completely stopped before it can be closed.  This
        means that `IOLoop.stop()` must be called *and* `IOLoop.start()` must
        be allowed to return before attempting to call `IOLoop.close()`.
        Therefore the call to `close` will usually appear just after
        the call to `start` rather than near the call to `stop`.

        .. versionchanged:: 3.1
           If the `IOLoop` implementation supports non-integer objects
           for "file descriptors", those objects will have their
           ``close`` method when ``all_fds`` is true.
        """
        raise NotImplementedError()

    @overload
    def add_handler(self, fd: int, handler: Callable[[int, int], None], events: int) -> None:
        pass

    @overload
    def add_handler(self, fd: _S, handler: Callable[[_S, int], None], events: int) -> None:
        pass

    def add_handler(self, fd: Union[int, _Selectable], handler: Callable[..., None], events: int) -> None:
        """Registers the given handler to receive the given events for ``fd``.

        The ``fd`` argument may either be an integer file descriptor or
        a file-like object with a ``fileno()`` and ``close()`` method.

        The ``events`` argument is a bitwise or of the constants
        ``IOLoop.READ``, ``IOLoop.WRITE``, and ``IOLoop.ERROR``.

        When an event occurs, ``handler(fd, events)`` will be run.

        .. versionchanged:: 4.0
           Added the ability to pass file-like objects in addition to
           raw file descriptors.
        """
        raise NotImplementedError()

    def update_handler(self, fd: Union[int, _Selectable], events: int) -> None:
        """Changes the events we listen for ``fd``.

        .. versionchanged:: 4.0
           Added the ability to pass file-like objects in addition to
           raw file descriptors.
        """
        raise NotImplementedError()

    def remove_handler(self, fd: Union[int, _Selectable]) -> None:
        """Stop listening for events on ``fd``.

        .. versionchanged:: 4.0
           Added the ability to pass file-like objects in addition to
           raw file descriptors.
        """
        raise NotImplementedError()

    def start(self) -> None:
        """Starts the I/O loop.

        The loop will run until one of the callbacks calls `stop()`, which
        will make the loop stop after the current event iteration completes.
        """
        raise NotImplementedError()

    def stop(self) -> None:
        """Stop the I/O loop.

        If the event loop is not currently running, the next call to `start()`
        will return immediately.

        Note that even after `stop` has been called, the `IOLoop` is not
        completely stopped until `IOLoop.start` has also returned.
        Some work that was scheduled before the call to `stop` may still
        be run before the `IOLoop` shuts down.
        """
        raise NotImplementedError()

    def run_sync(self, func: Callable, timeout: Optional[float] = None) -> Any:
        """Starts the `IOLoop`, runs the given function, and stops the loop.

        The function must return either an awaitable object or
        ``None``. If the function returns an awaitable object, the
        `IOLoop` will run until the awaitable is resolved (and
        `run_sync()` will return the awaitable's result). If it raises
        an exception, the `IOLoop` will stop and the exception will be
        re-raised to the caller.

        The keyword-only argument ``timeout`` may be used to set
        a maximum duration for the function.  If the timeout expires,
        a `asyncio.TimeoutError` is raised.

        This method is useful to allow asynchronous calls in a
        ``main()`` function::

            async def main():
                # do stuff...

            if __name__ == '__main__':
                IOLoop.current().run_sync(main)

        .. versionchanged:: 4.3
           Returning a non-``None``, non-awaitable value is now an error.

        .. versionchanged:: 5.0
           If a timeout occurs, the ``func`` coroutine will be cancelled.

        .. versionchanged:: 6.2
           ``tornado.util.TimeoutError`` is now an alias to ``asyncio.TimeoutError``.
        """
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
            raise TimeoutError('Operation timed out after %s seconds'