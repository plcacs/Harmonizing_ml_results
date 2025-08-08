from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable

_T = TypeVar('_T')
_S = TypeVar('_S', bound=_Selectable)

class IOLoop(Configurable):
    NONE: int = 0
    READ: int = 1
    WRITE: int = 4
    ERROR: int = 24
    _ioloop_for_asyncio: Dict[asyncio.AbstractEventLoop, 'IOLoop'] = {}
    _pending_tasks: Set[Future] = set()

    @classmethod
    def configure(cls, impl: Union[str, Type[BaseAsyncIOLoop]], **kwargs: Any) -> None:
        ...

    @staticmethod
    def instance() -> 'IOLoop':
        ...

    def install(self) -> None:
        ...

    @staticmethod
    def clear_instance() -> None:
        ...

    @typing.overload
    @staticmethod
    def current() -> Optional['IOLoop']:
        ...

    @typing.overload
    @staticmethod
    def current(instance: bool = True) -> Optional['IOLoop']:
        ...

    @staticmethod
    def current(instance: bool = True) -> Optional['IOLoop']:
        ...

    def make_current(self) -> None:
        ...

    @staticmethod
    def clear_current() -> None:
        ...

    @classmethod
    def configurable_base(cls) -> Type['IOLoop']:
        ...

    @classmethod
    def configurable_default(cls) -> Type[AsyncIOLoop]:
        ...

    def initialize(self, make_current: bool = True) -> None:
        ...

    def close(self, all_fds: bool = False) -> None:
        ...

    @typing.overload
    def add_handler(self, fd: Union[int, _Selectable], handler: Callable, events: int) -> None:
        ...

    @typing.overload
    def add_handler(self, fd: Union[int, _Selectable], handler: Callable, events: int) -> None:
        ...

    def add_handler(self, fd: Union[int, _Selectable], handler: Callable, events: int) -> None:
        ...

    def update_handler(self, fd: Union[int, _Selectable], events: int) -> None:
        ...

    def remove_handler(self, fd: Union[int, _Selectable]) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def run_sync(self, func: Callable, timeout: Optional[float] = None) -> Any:
        ...

    def time(self) -> float:
        ...

    def add_timeout(self, deadline: Union[float, datetime.timedelta], callback: Callable, *args: Any, **kwargs: Any) -> Any:
        ...

    def call_later(self, delay: float, callback: Callable, *args: Any, **kwargs: Any) -> Any:
        ...

    def call_at(self, when: float, callback: Callable, *args: Any, **kwargs: Any) -> Any:
        ...

    def remove_timeout(self, timeout: Any) -> None:
        ...

    def add_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        ...

    def add_callback_from_signal(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        ...

    def spawn_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        ...

    def add_future(self, future: Future, callback: Callable) -> None:
        ...

    def run_in_executor(self, executor: Optional[concurrent.futures.Executor], func: Callable, *args: Any) -> Future:
        ...

    def set_default_executor(self, executor: concurrent.futures.Executor) -> None:
        ...

    def _run_callback(self, callback: Callable) -> None:
        ...

    def _discard_future_result(self, future: Future) -> None:
        ...

    def split_fd(self, fd: Union[int, _Selectable]) -> Tuple[int, int]:
        ...

    def close_fd(self, fd: Union[int, _Selectable]) -> None:
        ...

    def _register_task(self, f: Future) -> None:
        ...

    def _unregister_task(self, f: Future) -> None:
        ...

class _Timeout:
    def __init__(self, deadline: float, callback: Callable, io_loop: IOLoop) -> None:
        ...

    def __lt__(self, other: '_Timeout') -> bool:
        ...

    def __le__(self, other: '_Timeout') -> bool:
        ...

class PeriodicCallback:
    def __init__(self, callback: Callable, callback_time: Union[float, datetime.timedelta], jitter: float = 0) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def is_running(self) -> bool:
        ...

    async def _run(self) -> None:
        ...

    def _schedule_next(self) -> None:
        ...

    def _update_next(self, current_time: float) -> None:
        ...
