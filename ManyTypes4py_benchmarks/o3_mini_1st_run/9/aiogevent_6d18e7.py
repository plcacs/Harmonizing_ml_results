#!/usr/bin/env python3
import asyncio
import selectors
import socket
import sys
from asyncio import AbstractEventLoopPolicy, Event as AIOEvent, Future, SelectorEventLoop, Handle
from typing import Any, Dict, cast, Optional, Callable, Tuple, List, Union
import gevent.core
import gevent.event
import gevent.hub
import greenlet
from gevent.event import Event as GEvent

socketpair = socket.socketpair
_PY3: bool = sys.version_info >= (3,)
_EVENT_READ: int = selectors.EVENT_READ
_EVENT_WRITE: int = selectors.EVENT_WRITE
_GEVENT10: bool = hasattr(gevent.hub.get_hub(), 'loop')

class _Selector(selectors._BaseSelectorImpl):
    _notified: Dict[int, int]
    _loop: "EventLoop"
    _event: Optional[gevent.event.Event]
    _gevent_events: Dict[int, Dict[int, Any]]
    _gevent_loop: Any

    def __init__(self, loop: "EventLoop") -> None:
        super(_Selector, self).__init__()
        self._notified = {}
        self._loop = loop
        self._event = None
        self._gevent_events = {}
        if _GEVENT10:
            self._gevent_loop = gevent.hub.get_hub().loop

    def close(self) -> None:
        keys = list(self.get_map().values())
        for key in keys:
            self.unregister(key.fd)
        super(_Selector, self).close()

    def _notify(self, fd: int, event: int) -> None:
        if fd in self._notified:
            self._notified[fd] |= event
        else:
            self._notified[fd] = event
        if self._event is not None:
            self._event.set()

    def _notify_read(self, event: Any, x: Any) -> None:
        self._notify(event.fd, _EVENT_READ)

    def _notify_write(self, event: Any, x: Any) -> None:
        self._notify(event.fd, _EVENT_WRITE)

    def _read_events(self) -> List[Tuple[selectors.SelectorKey, int]]:
        notified: Dict[int, int] = self._notified
        self._notified = {}
        ready: List[Tuple[selectors.SelectorKey, int]] = []
        for fd, events in notified.items():
            key = self.get_key(fd)
            ready.append((key, events & key.events))
            for event in (_EVENT_READ, _EVENT_WRITE):
                if key.events & event:
                    self._register(key.fd, event)
        return ready

    def _register(self, fd: int, event: int) -> None:
        if fd in self._gevent_events:
            event_dict = self._gevent_events[fd]
        else:
            event_dict = {}
            self._gevent_events[fd] = event_dict
        try:
            watcher = event_dict[event]
        except KeyError:
            pass
        else:
            if _GEVENT10:
                watcher.stop()
            else:
                watcher.cancel()
        if _GEVENT10:
            if event == _EVENT_READ:
                def func() -> None:
                    self._notify(fd, _EVENT_READ)
                watcher = self._gevent_loop.io(fd, 1)
                watcher.start(func)
            else:
                def func() -> None:
                    self._notify(fd, _EVENT_WRITE)
                watcher = self._gevent_loop.io(fd, 2)
                watcher.start(func)
            event_dict[event] = watcher
        else:
            if event == _EVENT_READ:
                gevent_event = gevent.core.read_event(fd, self._notify_read)
            else:
                gevent_event = gevent.core.write_event(fd, self._notify_write)
            event_dict[event] = gevent_event

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        key: selectors.SelectorKey = super(_Selector, self).register(fileobj, events, data)
        for event in (_EVENT_READ, _EVENT_WRITE):
            if events & event:
                self._register(key.fd, event)
        return key

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        key: selectors.SelectorKey = super(_Selector, self).unregister(fileobj)
        event_dict: Dict[int, Any] = self._gevent_events.pop(key.fd, {})
        for event in (_EVENT_READ, _EVENT_WRITE):
            try:
                watcher = event_dict[event]
            except KeyError:
                continue
            if _GEVENT10:
                watcher.stop()
            else:
                watcher.cancel()
        return key

    def select(self, timeout: Optional[float]) -> List[Tuple[selectors.SelectorKey, int]]:
        events: List[Tuple[selectors.SelectorKey, int]] = self._read_events()
        if events:
            return events
        self._event = gevent.event.Event()
        try:
            if timeout is not None:
                def timeout_cb(event: gevent.event.Event) -> None:
                    if event.ready():
                        return
                    event.set()
                gevent.spawn_later(timeout, timeout_cb, self._event)
                self._event.wait()
            else:
                self._event.wait()
            return self._read_events()
        finally:
            self._event = None

class EventLoop(asyncio.SelectorEventLoop):
    _greenlet: Optional[greenlet.greenlet]

    def __init__(self) -> None:
        self._greenlet = None
        selector: _Selector = _Selector(self)
        super(EventLoop, self).__init__(selector=selector)

    if _GEVENT10:
        def time(self) -> float:
            return gevent.core.time()

    def call_soon(self, callback: Callable[..., Any], *args: Any, context: Optional[Any] = None) -> Handle:
        handle: Handle = super(EventLoop, self).call_soon(callback, *args)
        if self._selector is not None and getattr(self._selector, '_event', None):
            self._write_to_self()
        return handle

    def call_at(self, when: float, callback: Callable[..., Any], *args: Any, context: Optional[Any] = None) -> Handle:
        handle: Handle = super(EventLoop, self).call_at(when, callback, *args)
        if self._selector is not None and getattr(self._selector, '_event', None):
            self._write_to_self()
        return handle

    def run_forever(self) -> None:
        self._greenlet = gevent.getcurrent()
        try:
            super(EventLoop, self).run_forever()
        finally:
            self._greenlet = None

def yield_future(future: Union[Future[Any], Any], loop: Optional[asyncio.AbstractEventLoop] = None) -> Any:
    """Wait for a future, a task, or a coroutine object from a greenlet.

    Yield control to other eligible greenlets until the future is done (finished
    successfully or failed with an exception).

    Return the result or raise the exception of the future.
    """
    fut: Future[Any] = asyncio.ensure_future(future, loop=loop)
    if fut._loop._greenlet == gevent.getcurrent():
        raise RuntimeError('yield_future() must not be called from the greenlet of the aiogreen event loop')
    event: gevent.event.Event = gevent.event.Event()

    def wakeup_event(fut: Future[Any]) -> None:
        event.set()

    fut.add_done_callback(wakeup_event)
    event.wait()
    return fut.result()

def yield_aio_event(aio_event: asyncio.Event) -> GEvent:
    """
    Converts an asyncio.Event into a gevent.event.Event

    Will set the returned GEvent whenever the underlying
    aio_event is set. Used to wait for an asyncio.Event
    inside of a greenlet.

    params:
        aio_event: Asyncio.Event to wait on
    """
    task: Future[Any] = asyncio.ensure_future(aio_event.wait())
    g_event: GEvent = GEvent()

    def wakeup_event(fut: Future[Any]) -> None:
        g_event.set()

    task.add_done_callback(wakeup_event)
    return g_event

def wrap_greenlet(gt: greenlet.greenlet, loop: Optional[asyncio.AbstractEventLoop] = None) -> Future[Any]:
    """Wrap a greenlet into a Future object.

    The Future object waits for the completion of a greenlet. The result or the
    exception of the greenlet will be stored in the Future object.

    Greenlet of greenlet and gevent modules are supported: gevent.greenlet
    and greenlet.greenlet.

    The greenlet must be wrapped before its execution starts. If the greenlet
    is running or already finished, an exception is raised.
    """
    fut: Future[Any] = Future(loop=loop)
    if not isinstance(gt, greenlet.greenlet):
        raise TypeError('greenlet.greenlet or gevent.greenlet request, not %s' % type(gt))
    if gt.dead:
        raise RuntimeError('wrap_greenlet: the greenlet already finished')
    if isinstance(gt, gevent.Greenlet):
        if _PY3:
            is_running: Callable[[greenlet.greenlet], bool] = greenlet.greenlet.__bool__
        else:
            is_running = greenlet.greenlet.__nonzero__
        if is_running(gt):
            raise RuntimeError('wrap_greenlet: the greenlet is running')
        try:
            orig_func: Callable[..., Any] = gt._run  # type: ignore
        except AttributeError:
            raise RuntimeError('wrap_greenlet: the _run attribute of the greenlet is not set')

        def wrap_func(*args: Any, **kw: Any) -> None:
            try:
                result: Any = orig_func(*args, **kw)
            except Exception as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
        gt._run = wrap_func  # type: ignore
    else:
        if gt:
            raise RuntimeError('wrap_greenlet: the greenlet is running')
        try:
            orig_func = gt.run
        except AttributeError:
            raise RuntimeError('wrap_greenlet: the run attribute of the greenlet is not set')

        def wrap_func(*args: Any, **kw: Any) -> None:
            try:
                result: Any = orig_func(*args, **kw)
            except Exception as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
        gt.run = wrap_func  # type: ignore
    return fut

class EventLoopPolicy(AbstractEventLoopPolicy):
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _loop_factory: Callable[[], asyncio.AbstractEventLoop] = EventLoop

    def __init__(self) -> None:
        self._loop = None

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        loop: Optional[asyncio.AbstractEventLoop] = self._loop
        if loop is None:
            loop = self._loop = self.new_event_loop()
        return cast(asyncio.AbstractEventLoop, loop)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop_factory()

    def get_child_watcher(self) -> None:
        pass

    def set_child_watcher(self, watcher: Any) -> None:
        pass
