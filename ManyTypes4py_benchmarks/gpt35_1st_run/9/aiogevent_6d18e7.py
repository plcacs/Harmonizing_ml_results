from typing import Any, Dict

class _Selector(selectors._BaseSelectorImpl):
    def __init__(self, loop: SelectorEventLoop):
    def _notify(self, fd: int, event: int):
    def _notify_read(self, event: AIOEvent, x: Any):
    def _notify_write(self, event: AIOEvent, x: Any):
    def _read_events(self) -> List[Tuple[selectors.SelectorKey, int]]:
    def _register(self, fd: int, event: int):
    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
    def select(self, timeout: float) -> List[Tuple[selectors.SelectorKey, int]]:

class EventLoop(asyncio.SelectorEventLoop):
    def __init__(self):
    def call_soon(self, callback, *args, context=None) -> asyncio.Handle:
    def call_at(self, when: float, callback, *args, context=None) -> asyncio.Handle:
    def run_forever(self):
