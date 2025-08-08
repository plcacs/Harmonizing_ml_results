from typing import Generic, TypeVar, cast

VT = TypeVar('VT')

class BaseSignal(Generic[VT]):
    def __init__(self, name: str = '', case: '_Case' = None, index: int = -1) -> None:
    async def send(self, value: Any = None, *, key: Any = None, force: bool = False) -> None:
    async def wait(self, *, key: Any = None, timeout: Seconds = None) -> VT:
    async def resolve(self, key: Any, event: SignalEvent) -> None
    def __set_name__(self, owner: Any, name: str) -> None:
    def _wakeup_resolvers(self) -> None:
    async def _wait_for_resolved(self, *, timeout: Seconds = None) -> None:
    def _get_current_value(self, key: Any) -> SignalEvent:
    def _index_key(self, key: Any) -> Tuple[str, str, Any]:
    def _set_current_value(self, key: Any, event: SignalEvent) -> None:
    def clone(self, **kwargs) -> 'BaseSignal':
    def _asdict(self, **kwargs) -> Dict[str, Any]:
    def __repr__(self) -> str:

class Signal(BaseSignal[VT]):
    async def send(self, value: Any = None, *, key: Any = None, force: bool = False) -> None:
    async def wait(self, *, key: Any = None, timeout: Seconds = None) -> VT:
    def _verify_event(self, ev: SignalEvent, key: Any, name: str, case: str) -> None:
    async def _wait_for_message_by_key(self, key: Any, *, timeout: Seconds = None, max_interval: float = 2.0) -> SignalEvent:
