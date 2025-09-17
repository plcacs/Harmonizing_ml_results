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
    @abc.abstractmethod
    def __init__(self, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = 'json') -> None:
        ...

    @abc.abstractmethod
    def loads_key(self, typ: Any, key: Any, *, serializer: Optional[Any] = None) -> Any:
        ...

    @abc.abstractmethod
    def loads_value(self, typ: Any, value: Any, *, serializer: Optional[Any] = None) -> Any:
        ...

    @abc.abstractmethod
    def dumps_key(self, typ: Any, key: Any, *, serializer: Optional[Any] = None) -> Any:
        ...

    @abc.abstractmethod
    def dumps_value(self, typ: Any, value: Any, *, serializer: Optional[Any] = None) -> Any:
        ...


class SchemaT(Generic[KT, VT]):
    key_type: Any = None
    value_type: Any = None
    key_serializer: Any = None
    value_serializer: Any = None
    allow_empty: bool = False

    def __init__(
        self,
        *,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
        allow_empty: Optional[bool] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def update(
        self,
        *,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
        allow_empty: Optional[bool] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def loads_key(
        self,
        app: "_AppT",
        message: "_Message",
        *,
        loads: Optional[Callable[..., Any]] = None,
        serializer: Optional[Any] = None,
    ) -> KT:
        ...

    @abc.abstractmethod
    def loads_value(
        self,
        app: "_AppT",
        message: "_Message",
        *,
        loads: Optional[Callable[..., Any]] = None,
        serializer: Optional[Any] = None,
    ) -> VT:
        ...

    @abc.abstractmethod
    def dumps_key(
        self,
        app: "_AppT",
        key: KT,
        *,
        serializer: Optional[Any] = None,
        headers: OpenHeadersArg,
    ) -> Any:
        ...

    @abc.abstractmethod
    def dumps_value(
        self,
        app: "_AppT",
        value: VT,
        *,
        serializer: Optional[Any] = None,
        headers: OpenHeadersArg,
    ) -> Any:
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(self, key: KT, headers: OpenHeadersArg) -> None:
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(self, value: VT, headers: OpenHeadersArg) -> None:
        ...