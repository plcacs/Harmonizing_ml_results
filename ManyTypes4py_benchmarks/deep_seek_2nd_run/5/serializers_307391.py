import abc
import typing
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, Dict, Union
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

    @abc.abstractmethod
    def __init__(
        self,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: CodecArg = 'json'
    ) -> None:
        ...

    @abc.abstractmethod
    def loads_key(
        self,
        typ: Type[KT],
        key: bytes,
        *,
        serializer: Optional[CodecArg] = None
    ) -> KT:
        ...

    @abc.abstractmethod
    def loads_value(
        self,
        typ: Type[VT],
        value: bytes,
        *,
        serializer: Optional[CodecArg] = None
    ) -> VT:
        ...

    @abc.abstractmethod
    def dumps_key(
        self,
        typ: Type[KT],
        key: KT,
        *,
        serializer: Optional[CodecArg] = None
    ) -> bytes:
        ...

    @abc.abstractmethod
    def dumps_value(
        self,
        typ: Type[VT],
        value: VT,
        *,
        serializer: Optional[CodecArg] = None
    ) -> bytes:
        ...

class SchemaT(Generic[KT, VT]):
    key_type: Optional[Type[KT]] = None
    value_type: Optional[Type[VT]] = None
    key_serializer: Optional[CodecArg] = None
    value_serializer: Optional[CodecArg] = None
    allow_empty: bool = False

    def __init__(
        self,
        *,
        key_type: Optional[Type[KT]] = None,
        value_type: Optional[Type[VT]] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None
    ) -> None:
        ...

    @abc.abstractmethod
    def update(
        self,
        *,
        key_type: Optional[Type[KT]] = None,
        value_type: Optional[Type[VT]] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None
    ) -> None:
        ...

    @abc.abstractmethod
    def loads_key(
        self,
        app: _AppT,
        message: _Message,
        *,
        loads: Optional[Callable[[bytes], Any]] = None,
        serializer: Optional[CodecArg] = None
    ) -> KT:
        ...

    @abc.abstractmethod
    def loads_value(
        self,
        app: _AppT,
        message: _Message,
        *,
        loads: Optional[Callable[[bytes], Any]] = None,
        serializer: Optional[CodecArg] = None
    ) -> VT:
        ...

    @abc.abstractmethod
    def dumps_key(
        self,
        app: _AppT,
        key: KT,
        *,
        serializer: Optional[CodecArg] = None,
        headers: OpenHeadersArg
    ) -> bytes:
        ...

    @abc.abstractmethod
    def dumps_value(
        self,
        app: _AppT,
        value: VT,
        *,
        serializer: Optional[CodecArg] = None,
        headers: OpenHeadersArg
    ) -> bytes:
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(
        self,
        key: KT,
        headers: OpenHeadersArg
    ) -> None:
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(
        self,
        value: VT,
        headers: OpenHeadersArg
    ) -> None:
        ...
