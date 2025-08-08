import abc
from typing import Any, Generic, Optional, TypeVar

from .codecs import CodecArg
from .core import K, OpenHeadersArg, V

KT = TypeVar('KT')
VT = TypeVar('VT')

class RegistryT(abc.ABC):

    @abc.abstractmethod
    def __init__(self, key_serializer: Any = None, value_serializer: str = 'json') -> None:
        ...

    @abc.abstractmethod
    def loads_key(self, typ: Any, key: Any, *, serializer: Any = None) -> None:
        ...

    @abc.abstractmethod
    def loads_value(self, typ: Any, value: Any, *, serializer: Any = None) -> None:
        ...

    @abc.abstractmethod
    def dumps_key(self, typ: Any, key: Any, *, serializer: Any = None) -> None:
        ...

    @abc.abstractmethod
    def dumps_value(self, typ: Any, value: Any, *, serializer: Any = None) -> None:
        ...

class SchemaT(Generic[KT, VT]):

    key_type: Optional[type] = None
    value_type: Optional[type] = None
    key_serializer: Optional[Any] = None
    value_serializer: Optional[Any] = None
    allow_empty: bool = False

    def __init__(self, *, key_type: Optional[type] = None, value_type: Optional[type] = None, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None, allow_empty: Optional[bool] = None) -> None:
        ...

    @abc.abstractmethod
    def update(self, *, key_type: Optional[type] = None, value_type: Optional[type] = None, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None, allow_empty: Optional[bool] = None) -> None:
        ...

    @abc.abstractmethod
    def loads_key(self, app: '_AppT', message: '_Message', *, loads: Any = None, serializer: Any = None) -> None:
        ...

    @abc.abstractmethod
    def loads_value(self, app: '_AppT', message: '_Message', *, loads: Any = None, serializer: Any = None) -> None:
        ...

    @abc.abstractmethod
    def dumps_key(self, app: '_AppT', key: Any, *, serializer: Any = None, headers: 'OpenHeadersArg') -> None:
        ...

    @abc.abstractmethod
    def dumps_value(self, app: '_AppT', value: Any, *, serializer: Any = None, headers: 'OpenHeadersArg') -> None:
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(self, key: Any, headers: 'OpenHeadersArg') -> None:
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(self, value: Any, headers: 'OpenHeadersArg') -> None:
        ...
