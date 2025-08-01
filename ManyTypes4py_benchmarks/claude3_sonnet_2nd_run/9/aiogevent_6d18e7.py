import asyncio
import selectors
import socket
import sys
from asyncio import AbstractEventLoopPolicy, Event as AIOEvent, Future, SelectorEventLoop
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import gevent.core
import gevent.event
import gevent.hub
import greenlet
from gevent.event import Event as GEvent

socketpair = socket.socketpair
_PY3 = sys.version_info >= (3,)
_EVENT_READ = selectors.EVENT_READ
_EVENT_WRITE = selectors.EVENT_WRITE
_GEVENT10 = hasattr(gevent.hub.get_hub(), 'loop')

class _Selector(selectors._BaseSelectorImpl):

    def __init__(self, loop: 'EventLoop') -> None:
        super(_Selector, self).__init__()
        self._notified: Dict[int, int] = {}
        self._loop = loop
        self._event: Optional[gevent.event.Event] = None
        self._gevent_events: Dict[int, Dict[int, Any]] = {}
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
        notified = self._notified
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
            event_dict: Dict[int, Any] = {}
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

    def register(self, fileobj: Union[int, socket.socket], events: int, data: Any = None) -> selectors.SelectorKey:
        key = super(_Selector, self).register(fileobj, events, data)
        for event in (_EVENT_READ, _EVENT_WRITE):
            if events & event:
                self._register(key.fd, event)
        return key

    def unregister(self, fileobj: Union[int, socket.socket]) -> selectors.SelectorKey:
        key = super(_Selector, self).unregister(fileobj)
        event_dict = self._gevent_events.pop(key.fd, {})
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
        events = self._read_events()
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

    def __init__(self) -> None:
        self._greenlet: Optional[greenlet.greenlet] = None
        selector = _Selector(self)
        super(EventLoop, self).__init__(selector=selector)
    if _GEVENT10:

        def time(self) -> float:
            return gevent.core.time()

    def call_soon(self, callback: Any, *args: Any, context: Optional[Any] = None) -> asyncio.Handle:
        handle = super(EventLoop, self).call_soon(callback, *args)
        if self._selector is not None and self._selector._event:
            self._write_to_self()
        return handle

    def call_at(self, when: float, callback: Any, *args: Any, context: Optional[Any] = None) -> asyncio.TimerHandle:
        handle = super(EventLoop, self).call_at(when, callback, *args)
        if self._selector is not None and self._selector._event:
            self._write_to_self()
        return handle

    def run_forever(self) -> None:
        self._greenlet = gevent.getcurrent()
        try:
            super(EventLoop, self).run_forever()
        finally:
            self._greenlet = None

def yield_future(future: Any, loop: Optional[asyncio.AbstractEventLoop] = None) -> Any:
    """Wait for a future, a task, or a coroutine object from a greenlet.

    Yield control other eligible greenlet until the future is done (finished
    successfully or failed with an exception).

    Return the result or raise the exception of the future.

    The function must not be called from the greenlet running the aiogreen
    event loop.
    """
    future = asyncio.ensure_future(future, loop=loop)
    if future._loop._greenlet == gevent.getcurrent():
        raise RuntimeError('yield_future() must not be called from the greenlet of the aiogreen event loop')
    event = gevent.event.Event()

    def wakeup_event(fut: Future) -> None:
        event.set()
    future.add_done_callback(wakeup_event)
    event.wait()
    return future.result()

def yield_aio_event(aio_event: AIOEvent) -> GEvent:
    """
    Converts an asyncio.Event into a gevent.event.Event

    Will set the returned GEvent whenever the underlying
    aio_event is set. Used to wait for an asyncio.Event
    inside of a greenlet

    params:
        aio_event: Asyncio.Event to wait on
    """
    task = asyncio.ensure_future(aio_event.wait())
    g_event = GEvent()

    def wakeup_event(fut: Future) -> None:
        g_event.set()
    task.add_done_callback(wakeup_event)
    return g_event

def wrap_greenlet(gt: greenlet.greenlet, loop: Optional[asyncio.AbstractEventLoop] = None) -> Future:
    """Wrap a greenlet into a Future object.

    The Future object waits for the completion of a greenlet. The result or the
    exception of the greenlet will be stored in the Future object.

    Greenlet of greenlet and gevent modules are supported: gevent.greenlet
    and greenlet.greenlet.

    The greenlet must be wrapped before its execution starts. If the greenlet
    is running or already finished, an exception is raised.

    For gevent.Greenlet, the _run attribute must be set. For greenlet.greenlet,
    the run attribute must be set.
    """
    fut = Future(loop=loop)
    if not isinstance(gt, greenlet.greenlet):
        raise TypeError('greenlet.greenlet or gevent.greenlet request, not %s' % type(gt))
    if gt.dead:
        raise RuntimeError('wrap_greenlet: the greenlet already finished')
    if isinstance(gt, gevent.Greenlet):
        if _PY3:
            is_running = greenlet.greenlet.__bool__
        else:
            is_running = greenlet.greenlet.__nonzero__
        if is_running(gt):
            raise RuntimeError('wrap_greenlet: the greenlet is running')
        try:
            orig_func = gt._run
        except AttributeError:
            raise RuntimeError('wrap_greenlet: the _run attribute of the greenlet is not set')

        def wrap_func(*args: Any, **kw: Any) -> Any:
            try:
                result = orig_func(*args, **kw)
            except Exception as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
        gt._run = wrap_func
    else:
        if gt:
            raise RuntimeError('wrap_greenlet: the greenlet is running')
        try:
            orig_func = gt.run
        except AttributeError:
            raise RuntimeError('wrap_greenlet: the run attribute of the greenlet is not set')

        def wrap_func(*args: Any, **kw: Any) -> Any:
            try:
                result = orig_func(*args, **kw)
            except Exception as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
        gt.run = wrap_func
    return fut

class EventLoopPolicy(AbstractEventLoopPolicy):
    _loop_factory = EventLoop

    def __init__(self) -> None:
        self._loop: Optional[EventLoop] = None

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None:
            loop = self._loop = self.new_event_loop()
        return cast(asyncio.AbstractEventLoop, loop)

    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._loop = cast(Optional[EventLoop], loop)

    def new_event_loop(self) -> EventLoop:
        return self._loop_factory()

    def get_child_watcher(self) -> None:
        pass

    def set_child_watcher(self, watcher: Any) -> None:
        pass
