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
    class _AppT: ...      # noqa
    class _ModelArg: ...  # noqa
    class _Message: ...   # noqa

__all__ = ['RegistryT', 'SchemaT']

KT = TypeVar('KT')
VT = TypeVar('VT')


class RegistryT(abc.ABC):

    key_serializer: CodecArg
    value_serializer: CodecArg

    @abc.abstractmethod
    def __init__(self,
                 key_serializer: Optional[CodecArg] = None,
                 value_serializer: CodecArg = 'json') -> None:
        ...

    @abc.abstractmethod
    def loads_key(self,
                  typ: Optional['_ModelArg'],
                  key: Optional[bytes],
                  *,
                  serializer: Optional[CodecArg] = None) -> K:
        ...

    @abc.abstractmethod
    def loads_value(self,
                    typ: Optional['_ModelArg'],
                    value: Optional[bytes],
                    *,
                    serializer: Optional[CodecArg] = None) -> Any:
        ...

    @abc.abstractmethod
    def dumps_key(self,
                  typ: Optional['_ModelArg'],
                  key: K,
                  *,
                  serializer: Optional[CodecArg] = None) -> Optional[bytes]:
        ...

    @abc.abstractmethod
    def dumps_value(self,
                    typ: Optional['_ModelArg'],
                    value: V,
                    *,
                    serializer: Optional[CodecArg] = None) -> Optional[bytes]:
        ...


class SchemaT(Generic[KT, VT]):

    key_type: Optional['_ModelArg'] = None
    value_type: Optional['_ModelArg'] = None

    key_serializer: Optional[CodecArg] = None
    value_serializer: Optional[CodecArg] = None

    allow_empty: bool = False

    def __init__(self, *,
                 key_type: Optional['_ModelArg'] = None,
                 value_type: Optional['_ModelArg'] = None,
                 key_serializer: Optional[CodecArg] = None,
                 value_serializer: Optional[CodecArg] = None,
                 allow_empty: Optional[bool] = None) -> None:
        ...

    @abc.abstractmethod
    def update(self, *,
               key_type: Optional['_ModelArg'] = None,
               value_type: Optional['_ModelArg'] = None,
               key_serializer: Optional[CodecArg] = None,
               value_serializer: Optional[CodecArg] = None,
               allow_empty: Optional[bool] = None) -> None:
        ...

    @abc.abstractmethod
    def loads_key(self, app: '_AppT', message: '_Message', *,
                  loads: Optional[Callable[..., KT]] = None,
                  serializer: Optional[CodecArg] = None) -> KT:
        ...

    @abc.abstractmethod
    def loads_value(self, app: '_AppT', message: '_Message', *,
                    loads: Optional[Callable[..., VT]] = None,
                    serializer: Optional[CodecArg] = None) -> VT:
        ...

    @abc.abstractmethod
    def dumps_key(self, app: '_AppT', key: K, *,
                  serializer: Optional[CodecArg] = None,
                  headers: OpenHeadersArg) -> Tuple[Any, OpenHeadersArg]:
        ...

    @abc.abstractmethod
    def dumps_value(self, app: '_AppT', value: V, *,
                    serializer: Optional[CodecArg] = None,
                    headers: OpenHeadersArg) -> Tuple[Any, OpenHeadersArg]:
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(
            self, key: KT, headers: OpenHeadersArg) -> OpenHeadersArg:
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(
            self, value: VT, headers: OpenHeadersArg) -> OpenHeadersArg:
        ...
