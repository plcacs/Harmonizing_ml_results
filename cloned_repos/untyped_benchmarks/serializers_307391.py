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
    def __init__(self, key_serializer=None, value_serializer='json'):
        ...

    @abc.abstractmethod
    def loads_key(self, typ, key, *, serializer=None):
        ...

    @abc.abstractmethod
    def loads_value(self, typ, value, *, serializer=None):
        ...

    @abc.abstractmethod
    def dumps_key(self, typ, key, *, serializer=None):
        ...

    @abc.abstractmethod
    def dumps_value(self, typ, value, *, serializer=None):
        ...

class SchemaT(Generic[KT, VT]):
    key_type = None
    value_type = None
    key_serializer = None
    value_serializer = None
    allow_empty = False

    def __init__(self, *, key_type=None, value_type=None, key_serializer=None, value_serializer=None, allow_empty=None):
        ...

    @abc.abstractmethod
    def update(self, *, key_type=None, value_type=None, key_serializer=None, value_serializer=None, allow_empty=None):
        ...

    @abc.abstractmethod
    def loads_key(self, app, message, *, loads=None, serializer=None):
        ...

    @abc.abstractmethod
    def loads_value(self, app, message, *, loads=None, serializer=None):
        ...

    @abc.abstractmethod
    def dumps_key(self, app, key, *, serializer=None, headers):
        ...

    @abc.abstractmethod
    def dumps_value(self, app, value, *, serializer=None, headers):
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(self, key, headers):
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(self, value, headers):
        ...