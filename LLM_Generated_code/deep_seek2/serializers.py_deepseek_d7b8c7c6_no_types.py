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
    key_serializer: CodecArg
    value_serializer: CodecArg

    @abc.abstractmethod
    def __init__(self, key_serializer=None, value_serializer='json'):
        ...

    @abc.abstractmethod
    def loads_key(self, typ, key, *, serializer: CodecArg=None):
        ...

    @abc.abstractmethod
    def loads_value(self, typ, value, *, serializer: CodecArg=None):
        ...

    @abc.abstractmethod
    def dumps_key(self, typ, key, *, serializer: CodecArg=None):
        ...

    @abc.abstractmethod
    def dumps_value(self, typ, value, *, serializer: CodecArg=None):
        ...

class SchemaT(Generic[KT, VT]):
    key_type: Optional[_ModelArg] = None
    value_type: Optional[_ModelArg] = None
    key_serializer: Optional[CodecArg] = None
    value_serializer: Optional[CodecArg] = None
    allow_empty: bool = False

    def __init__(self, *, key_type: _ModelArg=None, value_type: _ModelArg=None, key_serializer: CodecArg=None, value_serializer: CodecArg=None, allow_empty: bool=None):
        ...

    @abc.abstractmethod
    def update(self, *, key_type: _ModelArg=None, value_type: _ModelArg=None, key_serializer: CodecArg=None, value_serializer: CodecArg=None, allow_empty: bool=None):
        ...

    @abc.abstractmethod
    def loads_key(self, app, message, *, loads: Callable[[Optional[bytes], Optional[CodecArg]], K]=None, serializer: CodecArg=None):
        ...

    @abc.abstractmethod
    def loads_value(self, app, message, *, loads: Callable[[Optional[bytes], Optional[CodecArg]], Any]=None, serializer: CodecArg=None):
        ...

    @abc.abstractmethod
    def dumps_key(self, app, key, *, serializer: CodecArg=None, headers: OpenHeadersArg):
        ...

    @abc.abstractmethod
    def dumps_value(self, app, value, *, serializer: CodecArg=None, headers: OpenHeadersArg):
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(self, key, headers):
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(self, value, headers):
        ...