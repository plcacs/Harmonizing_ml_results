"""Bridges between the `asyncio` module and Tornado IOLoop.

.. versionadded:: 3.2

This module integrates Tornado with the ``asyncio`` module introduced
in Python 3.4. This makes it possible to combine the two libraries on
the same event loop.

.. deprecated:: 5.0

   While the code in this module is still used, it is now enabled
   automatically when `asyncio` is available, so applications should
   no longer need to refer to this module directly.

.. note::

   Tornado is designed to use a selector-based event loop. On Windows,
   where a proactor-based event loop has been the default since Python 3.8,
   a selector event loop is emulated by running ``select`` on a separate thread.
   Configuring ``asyncio`` to use a selector event loop may improve performance
   of Tornado (but may reduce performance of other ``asyncio``-based libraries
   in the same process).
"""
import asyncio
import atexit
import concurrent.futures
import errno
import functools
import select
import socket
import sys
import threading
import typing
import warnings
from tornado.gen import convert_yielded
from tornado.ioloop import IOLoop, _Selectable
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, TypeVar, Union
if typing.TYPE_CHECKING:
    from typing_extensions import TypeVarTuple, Unpack

class _HasFileno(Protocol):
    def fileno(self) -> int:
        pass

_FileDescriptorLike = Union[int, _HasFileno]
_T = TypeVar('_T')
if typing.TYPE_CHECKING:
    _Ts = TypeVarTuple('_Ts')
_selector_loops: Set["SelectorThread"] = set()

def _atexit_callback() -> None:
    for loop in _selector_loops:
        with loop._select_cond:
            loop._closing_selector = True
            loop._select_cond.notify()
        try:
            loop._waker_w.send(b'a')
        except BlockingIOError:
            pass
        if loop._thread is not None:
            loop._thread.join()
    _selector_loops.clear()
atexit.register(_atexit_callback)

class BaseAsyncIOLoop(IOLoop):
    def initialize(self, asyncio_loop: asyncio.AbstractEventLoop, **kwargs: Any) -> None:
        self.asyncio_loop: asyncio.AbstractEventLoop = asyncio_loop
        self.selector_loop: Union[asyncio.AbstractEventLoop, AddThreadSelectorEventLoop] = asyncio_loop
        if hasattr(asyncio, 'ProactorEventLoop') and isinstance(asyncio_loop, asyncio.ProactorEventLoop):
            self.selector_loop = AddThreadSelectorEventLoop(asyncio_loop)
        self.handlers: Dict[int, Tuple[_Selectable, Callable[[_Selectable, int], None]]] = {}
        self.readers: Set[int] = set()
        self.writers: Set[int] = set()
        self.closing: bool = False
        for loop in IOLoop._ioloop_for_asyncio.copy():
            if loop.is_closed():
                try:
                    del IOLoop._ioloop_for_asyncio[loop]
                except KeyError:
                    pass
        existing_loop = IOLoop._ioloop_for_asyncio.setdefault(asyncio_loop, self)
        if existing_loop is not self:
            raise RuntimeError(f'IOLoop {existing_loop} already associated with asyncio loop {asyncio_loop}')
        super().initialize(**kwargs)

    def close(self, all_fds: bool = False) -> None:
        self.closing = True
        for fd in list(self.handlers):
            fileobj, handler_func = self.handlers[fd]
            self.remove_handler(fd)
            if all_fds:
                self.close_fd(fileobj)
        del IOLoop._ioloop_for_asyncio[self.asyncio_loop]
        if self.selector_loop is not self.asyncio_loop:
            self.selector_loop.close()
        self.asyncio_loop.close()

    def add_handler(self, fd: _FileDescriptorLike, handler: Callable[[_Selectable, int], None], events: int) -> None:
        fd, fileobj = self.split_fd(fd)
        if fd in self.handlers:
            raise ValueError('fd %s added twice' % fd)
        self.handlers[fd] = (fileobj, handler)
        if events & IOLoop.READ:
            self.selector_loop.add_reader(fd, self._handle_events, fd, IOLoop.READ)
            self.readers.add(fd)
        if events & IOLoop.WRITE:
            self.selector_loop.add_writer(fd, self._handle_events, fd, IOLoop.WRITE)
            self.writers.add(fd)

    def update_handler(self, fd: _FileDescriptorLike, events: int) -> None:
        fd, fileobj = self.split_fd(fd)
        if events & IOLoop.READ:
            if fd not in self.readers:
                self.selector_loop.add_reader(fd, self._handle_events, fd, IOLoop.READ)
                self.readers.add(fd)
        elif fd in self.readers:
            self.selector_loop.remove_reader(fd)
            self.readers.remove(fd)
        if events & IOLoop.WRITE:
            if fd not in self.writers:
                self.selector_loop.add_writer(fd, self._handle_events, fd, IOLoop.WRITE)
                self.writers.add(fd)
        elif fd in self.writers:
            self.selector_loop.remove_writer(fd)
            self.writers.remove(fd)

    def remove_handler(self, fd: _FileDescriptorLike) -> None:
        fd, fileobj = self.split_fd(fd)
        if fd not in self.handlers:
            return
        if fd in self.readers:
            self.selector_loop.remove_reader(fd)
            self.readers.remove(fd)
        if fd in self.writers:
            self.selector_loop.remove_writer(fd)
            self.writers.remove(fd)
        del self.handlers[fd]

    def _handle_events(self, fd: int, events: int) -> None:
        fileobj, handler_func = self.handlers[fd]
        handler_func(fileobj, events)

    def start(self) -> None:
        self.asyncio_loop.run_forever()

    def stop(self) -> None:
        self.asyncio_loop.stop()

    def call_at(self, when: float, callback: Callable[..., None], *args: Any, **kwargs: Any) -> asyncio.TimerHandle:
        return self.asyncio_loop.call_later(max(0, when - self.time()), self._run_callback, functools.partial(callback, *args, **kwargs))

    def remove_timeout(self, timeout: asyncio.TimerHandle) -> None:
        timeout.cancel()

    def add_callback(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        try:
            if asyncio.get_running_loop() is self.asyncio_loop:
                call_soon = self.asyncio_loop.call_soon
            else:
                call_soon = self.asyncio_loop.call_soon_threadsafe
        except RuntimeError:
            call_soon = self.asyncio_loop.call_soon_threadsafe
        try:
            call_soon(self._run_callback, functools.partial(callback, *args, **kwargs))
        except RuntimeError:
            pass
        except AttributeError:
            pass

    def add_callback_from_signal(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        warnings.warn('add_callback_from_signal is deprecated', DeprecationWarning)
        try:
            self.asyncio_loop.call_soon_threadsafe(self._run_callback, functools.partial(callback, *args, **kwargs))
        except RuntimeError:
            pass

    def run_in_executor(self, executor: Optional[concurrent.futures.Executor], func: Callable[..., _T], *args: Any) -> asyncio.Future[_T]:
        return self.asyncio_loop.run_in_executor(executor, func, *args)

    def set_default_executor(self, executor: concurrent.futures.Executor) -> None:
        return self.asyncio_loop.set_default_executor(executor)

class AsyncIOMainLoop(BaseAsyncIOLoop):
    """``AsyncIOMainLoop`` creates an `.IOLoop` that corresponds to the
    current ``asyncio`` event loop (i.e. the one returned by
    ``asyncio.get_event_loop()``).

    .. deprecated:: 5.0

       Now used automatically when appropriate; it is no longer necessary
       to refer to this class directly.

    .. versionchanged:: 5.0

       Closing an `AsyncIOMainLoop` now closes the underlying asyncio loop.
    """

    def initialize(self, **kwargs: Any) -> None:
        super().initialize(asyncio.get_event_loop(), **kwargs)

    def _make_current(self) -> None:
        pass

class AsyncIOLoop(BaseAsyncIOLoop):
    """``AsyncIOLoop`` is an `.IOLoop` that runs on an ``asyncio`` event loop.
    This class follows the usual Tornado semantics for creating new
    ``IOLoops``; these loops are not necessarily related to the
    ``asyncio`` default event loop.

    Each ``AsyncIOLoop`` creates a new ``asyncio.EventLoop``; this object
    can be accessed with the ``asyncio_loop`` attribute.

    .. versionchanged:: 6.2

       Support explicit ``asyncio_loop`` argument
       for specifying the asyncio loop to attach to,
       rather than always creating a new one with the default policy.

    .. versionchanged:: 5.0

       When an ``AsyncIOLoop`` becomes the current `.IOLoop`, it also sets
       the current `asyncio` event loop.

    .. deprecated:: 5.0

       Now used automatically when appropriate; it is no longer necessary
       to refer to this class directly.
    """

    def initialize(self, **kwargs: Any) -> None:
        self.is_current: bool = False
        loop: Optional[asyncio.AbstractEventLoop] = None
        if 'asyncio_loop' not in kwargs:
            kwargs['asyncio_loop'] = loop = asyncio.new_event_loop()
        try:
            super().initialize(**kwargs)
        except Exception:
            if loop is not None:
                loop.close()
            raise

    def close(self, all_fds: bool = False) -> None:
        if self.is_current:
            self._clear_current()
        super().close(all_fds=all_fds)

    def _make_current(self) -> None:
        if not self.is_current:
            try:
                self.old_asyncio = asyncio.get_event_loop()
            except (RuntimeError, AssertionError):
                self.old_asyncio = None
            self.is_current = True
        asyncio.set_event_loop(self.asyncio_loop)

    def _clear_current_hook(self) -> None:
        if self.is_current:
            asyncio.set_event_loop(self.old_asyncio)
            self.is_current = False

def to_tornado_future(asyncio_future: asyncio.Future[_T]) -> asyncio.Future[_T]:
    """Convert an `asyncio.Future` to a `tornado.concurrent.Future`.

    .. versionadded:: 4.1

    .. deprecated:: 5.0
       Tornado ``Futures`` have been merged with `asyncio.Future`,
       so this method is now a no-op.
    """
    return asyncio_future

def to_asyncio_future(tornado_future: Any) -> asyncio.Future[Any]:
    """Convert a Tornado yieldable object to an `asyncio.Future`.

    .. versionadded:: 4.1

    .. versionchanged:: 4.3
       Now accepts any yieldable object, not just
       `tornado.concurrent.Future`.

    .. deprecated:: 5.0
       Tornado ``Futures`` have been merged with `asyncio.Future`,
       so this method is now equivalent to `tornado.gen.convert_yielded`.
    """
    return convert_yielded(tornado_future)

if sys.platform == 'win32' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy
else:
    _BasePolicy = asyncio.DefaultEventLoopPolicy

class AnyThreadEventLoopPolicy(_BasePolicy):
    """Event loop policy that allows loop creation on any thread.

    The default `asyncio` event loop policy only automatically creates
    event loops in the main threads. Other threads must create event
    loops explicitly or `asyncio.get_event_loop` (and therefore
    `.IOLoop.current`) will fail. Installing this policy allows event
    loops to be created automatically on any thread, matching the
    behavior of Tornado versions prior to 5.0 (or 5.0 on Python 2).

    Usage::

        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())

    .. versionadded:: 5.0

    .. deprecated:: 6.2

        ``AnyThreadEventLoopPolicy`` affects the implicit creation
        of an event loop, which is deprecated in Python 3.10 and
        will be removed in a future version of Python. At that time
        ``AnyThreadEventLoopPolicy`` will no longer be useful.
        If you are relying on it, use `asyncio.new_event_loop`
        or `asyncio.run` explicitly in any non-main threads that
        need event loops.
    """

    def __init__(self) -> None:
        super().__init__()
        warnings.warn('AnyThreadEventLoopPolicy is deprecated, use asyncio.run or asyncio.new_event_loop instead', DeprecationWarning, stacklevel=2)

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return super().get_event_loop()
        except RuntimeError:
            loop = self.new_event_loop()
            self.set_event_loop(loop)
            return loop

class SelectorThread:
    """Define ``add_reader`` methods to be called in a background select thread.

    Instances of this class start a second thread to run a selector.
    This thread is completely hidden from the user;
    all callbacks are run on the wrapped event loop's thread.

    Typically used via ``AddThreadSelectorEventLoop``,
    but can be attached to a running asyncio loop.
    """
    _closed: bool = False

    def __init__(self, real_loop: asyncio.AbstractEventLoop) -> None:
        self._real_loop: asyncio.AbstractEventLoop = real_loop
        self._select_cond: threading.Condition = threading.Condition()
        self._select_args: Optional[Tuple[List[int], List[int]]] = None
        self._closing_selector: bool = False
        self._thread: Optional[threading.Thread] = None
        self._thread_manager_handle = self._thread_manager()

        async def thread_manager_anext() -> None:
            await self._thread_manager_handle.__anext__()
        self._real_loop.call_soon(lambda: self._real_loop.create_task(thread_manager_anext()))
        self._readers: Dict[int, Callable[[], None]] = {}
        self._writers: Dict[int, Callable[[], None]] = {}
        self._waker_r, self._waker_w = socket.socketpair()
        self._waker_r.setblocking(False)
        self._waker_w.setblocking(False)
        _selector_loops.add(self)
        self.add_reader(self._waker_r, self._consume_waker)

    def close(self) -> None:
        if self._closed:
            return
        with self._select_cond:
            self._closing_selector = True
            self._select_cond.notify()
        self._wake_selector()
        if self._thread is not None:
            self._thread.join()
        _selector_loops.discard(self)
        self.remove_reader(self._waker_r)
        self._waker_r.close()
        self._waker_w.close()
        self._closed = True

    async def _thread_manager(self) -> None:
        self._thread = threading.Thread(name='Tornado selector', daemon=True, target=self._run_select)
        self._thread.start()
        self._start_select()
        try:
            yield
        except GeneratorExit:
            self.close()
            raise

    def _wake_selector(self) -> None:
        if self._closed:
            return
        try:
            self._waker_w.send(b'a')
        except BlockingIOError:
            pass

    def _consume_waker(self) -> None:
        try:
            self._waker_r.recv(1024)
        except BlockingIOError:
            pass

    def _start_select(self) -> None:
        with self._select_cond:
            assert self._select_args is None
            self._select_args = (list(self._readers.keys()), list(self._writers.keys()))
            self._select_cond.notify()

    def _run_select(self) -> None:
        while True:
            with self._select_cond:
                while self._select_args is None and (not self._closing_selector):
                    self._select_cond.wait()
                if self._closing_selector:
                    return
                assert self._select_args is not None
                to_read, to_write = self._select_args
                self._select_args = None
            try:
                rs, ws, xs = select.select(to_read, to_write, to_write)
                ws = ws + xs
            except OSError as e:
                if e.errno == getattr(errno, 'WSAENOTSOCK', errno.EBADF):
                    rs, _, _ = select.select([self._waker_r.fileno()], [], [], 0)
                    if rs:
                        ws = []
                    else:
                        raise
                else:
                    raise
            try:
                self._real_loop.call_soon_threadsafe(self._handle_select, rs, ws)
            except RuntimeError:
                pass
            except AttributeError:
                pass

    def _handle_select(self, rs: List[int], ws: List[int]) -> None:
        for r in rs:
            self._handle_event(r, self._readers)
        for w in ws:
            self._handle_event(w, self._writers)
        self._start_select()

    def _handle_event(self, fd: int, cb_map: Dict[int, Callable[[], None]]) -> None:
        try:
            callback = cb_map[fd]
        except KeyError:
            return
        callback()

    def add_reader(self, fd: _FileDescriptorLike, callback: Callable[..., None], *args: Any) -> None:
        if isinstance(fd, int):
            fd_int = fd
        else:
            fd_int = fd.fileno()
        self._readers[fd_int] = functools.partial(callback, *args)
        self._wake_selector()

    def add_writer(self, fd: _FileDescriptorLike, callback: Callable[..., None], *args: Any) -> None:
        if isinstance(fd, int):
            fd_int = fd
        else:
            fd_int = fd.fileno()
        self._writers[fd_int] = functools.partial(callback, *args)
        self._wake_selector()

    def remove_reader(self, fd: _FileDescriptorLike) -> bool:
        if isinstance(fd, int):
            fd_int = fd
        else:
            fd_int = fd.fileno()
        try:
            del self._readers[fd_int]
        except KeyError:
            return False
        self._wake_selector()
        return True

    def remove_writer(self, fd: _FileDescriptorLike) -> bool:
        if isinstance(fd, int):
            fd_int = fd
        else:
            fd_int = fd.fileno()
        try:
            del self._writers[fd_int]
        except KeyError:
            return False
        self._wake_selector()
        return True

class AddThreadSelectorEventLoop(asyncio.AbstractEventLoop):
    """Wrap an event loop to add implementations of the ``add_reader`` method family.

    Instances of this class start a second thread to run a selector.
    This thread is completely hidden from the user; all callbacks are
    run on the wrapped event loop's thread.

    This class is used automatically by Tornado; applications should not need
    to refer to it directly.

    It is safe to wrap any event loop with this class, although it only makes sense
    for event loops that do not implement the ``add_reader`` family of methods
    themselves (i.e. ``WindowsProactorEventLoop``)

    Closing the ``AddThreadSelectorEventLoop`` also closes the wrapped event loop.

    """
    MY_ATTRIBUTES: Set[str] = {'_real_loop', '_selector', 'add_reader', 'add_writer', 'close', 'remove_reader', 'remove_writer'}

    def __getattribute__(self, name: str) -> Any:
        if name in AddThreadSelectorEventLoop.MY_ATTRIBUTES:
            return super().__getattribute__(name)
        return getattr(self._real_loop, name)

    def __init__(self, real_loop: asyncio.AbstractEventLoop) -> None:
        self._real_loop: asyncio.AbstractEventLoop = real_loop
        self._selector: SelectorThread = SelectorThread(real_loop)

    def close(self) -> None:
        self._selector.close()
        self._real_loop.close()

    def add_reader(self, fd: _FileDescriptorLike, callback: Callable[..., None], *args: Any) -> None:
        return self._selector.add_reader(fd, callback, *args)

    def add_writer(self, fd: _FileDescriptorLike, callback: Callable[..., None], *args: Any) -> None:
        return self._selector.add_writer(fd, callback, *args)

    def remove_reader(self, fd: _FileDescriptorLike) -> bool:
        return self._selector.remove_reader(fd)

    def remove_writer(self, fd: _FileDescriptorLike) -> bool:
        return self._selector.remove_writer(fd)
