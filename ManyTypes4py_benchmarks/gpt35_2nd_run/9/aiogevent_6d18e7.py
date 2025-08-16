from typing import Any, Dict

class _Selector(selectors._BaseSelectorImpl):
    def __init__(self, loop: SelectorEventLoop) -> None:
    def _notify(self, fd: int, event: int) -> None:
    def _notify_read(self, event: AIOEvent, x: Any) -> None:
    def _notify_write(self, event: AIOEvent, x: Any) -> None:
    def _read_events(self) -> List[Tuple]:
    def _register(self, fd: int, event: int) -> None:
    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
    def select(self, timeout: float) -> List[Tuple]:

class EventLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
    def call_soon(self, callback: Any, *args: Any, context: Any = None) -> Any:
    def call_at(self, when: float, callback: Any, *args: Any, context: Any = None) -> Any:
    def run_forever(self) -> None

def yield_future(future: Any, loop: Any = None) -> Any:
def yield_aio_event(aio_event: AIOEvent) -> GEvent:
def wrap_greenlet(gt: greenlet.greenlet, loop: Any = None) -> Future:

class EventLoopPolicy(AbstractEventLoopPolicy):
    _loop_factory = EventLoop
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
    def new_event_loop(self) -> asyncio.AbstractEventLoop:
