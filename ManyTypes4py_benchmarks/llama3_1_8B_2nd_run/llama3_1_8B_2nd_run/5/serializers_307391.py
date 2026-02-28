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
    def __init__(self, key_serializer: Optional[Callable[[KT], Any]] = None, value_serializer: str = 'json'):
        ...

    def loads_key(self, typ: typing.Type[K], key: KT, *, serializer: Optional[Callable[[KT], Any]] = None) -> K:
        ...

    def loads_value(self, typ: typing.Type[VT], value: VT, *, serializer: Optional[Callable[[VT], Any]] = None) -> VT:
        ...

    def dumps_key(self, typ: typing.Type[K], key: K, *, serializer: Optional[Callable[[K], Any]] = None) -> str:
        ...

    def dumps_value(self, typ: typing.Type[VT], value: VT, *, serializer: Optional[Callable[[VT], Any]] = None) -> str:
        ...

class SchemaT(Generic[KT, VT]):
    key_type: typing.Type[K] = None
    value_type: typing.Type[VT] = None
    key_serializer: Optional[Callable[[KT], Any]] = None
    value_serializer: Optional[Callable[[VT], Any]] = None
    allow_empty: bool = False

    def __init__(self, *, key_type: Optional[typing.Type[K]] = None, value_type: Optional[typing.Type[VT]] = None, key_serializer: Optional[Callable[[KT], Any]] = None, value_serializer: Optional[Callable[[VT], Any]] = None, allow_empty: Optional[bool] = None):
        ...

    @abc.abstractmethod
    def update(self, *, key_type: Optional[typing.Type[K]] = None, value_type: Optional[typing.Type[VT]] = None, key_serializer: Optional[Callable[[KT], Any]] = None, value_serializer: Optional[Callable[[VT], Any]] = None, allow_empty: Optional[bool] = None) -> 'SchemaT[KT, VT]':
        ...

    @abc.abstractmethod
    def loads_key(self, app: _AppT, message: _Message, *, loads: Optional[Callable[[KT], K]] = None, serializer: Optional[Callable[[KT], Any]] = None) -> K:
        ...

    @abc.abstractmethod
    def loads_value(self, app: _AppT, message: _Message, *, loads: Optional[Callable[[VT], VT]] = None, serializer: Optional[Callable[[VT], Any]] = None) -> VT:
        ...

    @abc.abstractmethod
    def dumps_key(self, app: _AppT, key: K, *, serializer: Optional[Callable[[K], Any]] = None, headers: OpenHeadersArg) -> str:
        ...

    @abc.abstractmethod
    def dumps_value(self, app: _AppT, value: VT, *, serializer: Optional[Callable[[VT], Any]] = None, headers: OpenHeadersArg) -> str:
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(self, key: K, headers: OpenHeadersArg) -> OpenHeadersArg:
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(self, value: VT, headers: OpenHeadersArg) -> OpenHeadersArg:
        ...
