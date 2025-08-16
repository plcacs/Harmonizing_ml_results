from typing import Union, Any, Callable, List, Type, Tuple, Awaitable, Dict, overload

_T = typing.TypeVar('_T')
_Yieldable = Union[None, Awaitable, List[Awaitable], Dict[Any, Awaitable], concurrent.futures.Future]

def coroutine(func: Callable[..., Any]) -> Callable[..., Future]:
    ...

def is_coroutine_function(func: Callable[..., Any]) -> bool:
    ...

class Return(Exception):
    ...

class WaitIterator:
    ...

def multi(children: _Yieldable, quiet_exceptions: Tuple[Type[Exception], ...] = ()) -> Future:
    ...

def maybe_future(x: Any) -> Future:
    ...

def with_timeout(timeout: Union[float, datetime.timedelta], future: _Yieldable, quiet_exceptions: Tuple[Type[Exception], ...] = ()) -> Future:
    ...

def sleep(duration: float) -> Future:
    ...

class _NullFuture:
    ...
