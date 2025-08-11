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
    def __init__(self, key_serializer: Union[None, models.ModelArg, bool]=None, value_serializer: Union[None, models.ModelArg, bool]='json') -> None:
        ...

    @abc.abstractmethod
    def loads_key(self, typ, key, *, serializer: Union[None, bool, typing.Callable, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser]=None) -> None:
        ...

    @abc.abstractmethod
    def loads_value(self, typ, value, *, serializer: Union[None, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser, typing.Callable, bool]=None) -> None:
        ...

    @abc.abstractmethod
    def dumps_key(self, typ, key: Union[typing.Iterable[eth.abc.BlockHeaderAPI], bytes], *, serializer: Union[None, typing.Iterable[eth.abc.BlockHeaderAPI], bytes]=None) -> None:
        ...

    @abc.abstractmethod
    def dumps_value(self, typ, value: Union[int, Exception, collections.abc.AsyncGenerator], *, serializer: Union[None, int, Exception, collections.abc.AsyncGenerator]=None) -> None:
        ...

class SchemaT(Generic[KT, VT]):
    key_type = None
    value_type = None
    key_serializer = None
    value_serializer = None
    allow_empty = False

    def __init__(self, *, key_type: Union[None, models.ModelArg, bool]=None, value_type: Union[None, models.ModelArg, bool]=None, key_serializer: Union[None, models.ModelArg, bool]=None, value_serializer: Union[None, models.ModelArg, bool]=None, allow_empty: Union[None, models.ModelArg, bool]=None) -> None:
        ...

    @abc.abstractmethod
    def update(self, *, key_type: Union[None, codecs.CodecArg, bool, float]=None, value_type: Union[None, codecs.CodecArg, bool, float]=None, key_serializer: Union[None, codecs.CodecArg, bool, float]=None, value_serializer: Union[None, codecs.CodecArg, bool, float]=None, allow_empty: Union[None, codecs.CodecArg, bool, float]=None) -> None:
        ...

    @abc.abstractmethod
    def loads_key(self, app: Union[bool, typing.Callable, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser], message: Union[bool, typing.Callable, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser], *, loads: Union[None, bool, typing.Callable, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser]=None, serializer: Union[None, bool, typing.Callable, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser]=None) -> None:
        ...

    @abc.abstractmethod
    def loads_value(self, app: Union[zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser, typing.Callable, bool], message: Union[zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser, typing.Callable, bool], *, loads: Union[None, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser, typing.Callable, bool]=None, serializer: Union[None, zerver.models.UserProfile, django.contrib.auth.models.AnonymousUser, typing.Callable, bool]=None) -> None:
        ...

    @abc.abstractmethod
    def dumps_key(self, app: Union[typing.Iterable[eth.abc.BlockHeaderAPI], bytes], key: Union[typing.Iterable[eth.abc.BlockHeaderAPI], bytes], *, serializer: Union[None, typing.Iterable[eth.abc.BlockHeaderAPI], bytes]=None, headers: Union[typing.Iterable[eth.abc.BlockHeaderAPI], bytes]) -> None:
        ...

    @abc.abstractmethod
    def dumps_value(self, app: Union[int, Exception, collections.abc.AsyncGenerator], value: Union[int, Exception, collections.abc.AsyncGenerator], *, serializer: Union[None, int, Exception, collections.abc.AsyncGenerator]=None, headers: Union[int, Exception, collections.abc.AsyncGenerator]) -> None:
        ...

    @abc.abstractmethod
    def on_dumps_key_prepare_headers(self, key: Union[faustypes.core.V, faustypes.core.OpenHeadersArg, str], headers: Union[faustypes.core.V, faustypes.core.OpenHeadersArg, str]) -> None:
        ...

    @abc.abstractmethod
    def on_dumps_value_prepare_headers(self, value: Union[faustypes.core.V, faustypes.core.OpenHeadersArg, bytes], headers: Union[faustypes.core.V, faustypes.core.OpenHeadersArg, bytes]) -> None:
        ...