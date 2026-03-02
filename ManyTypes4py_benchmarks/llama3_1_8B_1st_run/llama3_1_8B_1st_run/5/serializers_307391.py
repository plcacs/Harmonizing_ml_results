import abc
import typing
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar
from .codecs import CodecArg
from .core import K, OpenHeadersArg, V
if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .models import ModelArg as _ModelArg
    from .tuples import Message as _Message
else:

    class _AppT:
        ...

    class _ModelArg:
        ...

    class _Message:
        ...
__all__ = ['RegistryT', 'SchemaT']
KT = TypeVar('KT')
VT = TypeVar('VT')

class RegistryT(abc.ABC):
    def __init__(self, key_serializer: Optional[Callable[[KT], str]] = None, value_serializer: str = 'json'):
        ...

    def loads_key(self, typ: Any, key: KT, *, serializer: Optional[Callable[[KT], str]] = None) -> KT:
        ...

    def loads_value(self, typ: Any, value: VT, *, serializer: Optional[Callable[[VT], str]] = None) -> VT:
        ...

    def dumps_key(self, typ: Any, key: KT, *, serializer: Optional[Callable[[KT], str]] = None) -> str:
        ...

    def dumps_value(self, typ: Any, value: VT, *, serializer: Optional[Callable[[VT], str]] = None) -> str:
        ...

class SchemaT(Generic[KT, VT]):
    key_type: Optional[type] = None
    value_type: Optional[type] = None
    key_serializer: Optional[Callable[[KT], str]] = None
    value_serializer: Optional[Callable[[VT], str]] = None
    allow_empty: bool = False

    def __init__(self, *, key_type: Optional[type] = None, value_type: Optional[type] = None, key_serializer: Optional[Callable[[KT], str]] = None, value_serializer: Optional[Callable[[VT], str]] = None, allow_empty: Optional[bool] = None):
        ...

    def update(self, *, key_type: Optional[type] = None, value_type: Optional[type] = None, key_serializer: Optional[Callable[[KT], str]] = None, value_serializer: Optional[Callable[[VT], str]] = None, allow_empty: Optional[bool] = None) -> None:
        ...

    def loads_key(self, app: _AppT, message: _Message, *, loads: Optional[Callable[[KT], KT]] = None, serializer: Optional[Callable[[KT], str]] = None) -> KT:
        ...

    def loads_value(self, app: _AppT, message: _Message, *, loads: Optional[Callable[[VT], VT]] = None, serializer: Optional[Callable[[VT], str]] = None) -> VT:
        ...

    def dumps_key(self, app: _AppT, key: KT, *, serializer: Optional[Callable[[KT], str]] = None, headers: OpenHeadersArg) -> str:
        ...

    def dumps_value(self, app: _AppT, value: VT, *, serializer: Optional[Callable[[VT], str]] = None, headers: OpenHeadersArg) -> str:
        ...

    def on_dumps_key_prepare_headers(self, key: KT, headers: OpenHeadersArg) -> OpenHeadersArg:
        ...

    def on_dumps_value_prepare_headers(self, value: VT, headers: OpenHeadersArg) -> OpenHeadersArg:
        ...
