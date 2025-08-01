import typing
from contextlib import suppress
from typing import Any, Awaitable, Callable, Optional, Tuple, cast
from faust.exceptions import KeyDecodeError, ValueDecodeError
from faust.types.app import AppT
from faust.types.core import K, OpenHeadersArg, V
from faust.types.codecs import CodecArg
from faust.types.events import EventT
from faust.types.models import ModelArg
from faust.types.serializers import KT, SchemaT, VT
from faust.types.tuples import Message
__all__ = ['Schema']
if typing.TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg
    DecodeFunction = Callable[[Message, DefaultNamedArg(bool, 'propagate')], Awaitable[EventT]]
else:
    DecodeFunction = Callable[..., Awaitable[EventT]]
OnKeyDecodeErrorFun = Callable[[Exception, Message], Awaitable[None]]
OnValueDecodeErrorFun = Callable[[Exception, Message], Awaitable[None]]

async def _noop_decode_error(exc: Exception, message: Message) -> None:
    ...

class Schema(SchemaT):

    def __init__(self, *, key_type: Any = None, value_type: Any = None, 
                 key_serializer: Optional[CodecArg] = None, 
                 value_serializer: Optional[CodecArg] = None, 
                 allow_empty: Optional[bool] = None) -> None:
        self.update(key_type=key_type, value_type=value_type, 
                   key_serializer=key_serializer, value_serializer=value_serializer, 
                   allow_empty=allow_empty)

    def update(self, *, key_type: Any = None, value_type: Any = None, 
               key_serializer: Optional[CodecArg] = None, 
               value_serializer: Optional[CodecArg] = None, 
               allow_empty: Optional[bool] = None) -> None:
        if key_type is not None:
            self.key_type = key_type
        if value_type is not None:
            self.value_type = value_type
        if key_serializer is not None:
            self.key_serializer = key_serializer
        if value_serializer is not None:
            self.value_serializer = value_serializer
        if self.key_serializer is None and key_type:
            self.key_serializer = _model_serializer(key_type)
        if self.value_serializer is None and value_type:
            self.value_serializer = _model_serializer(value_type)
        if allow_empty is not None:
            self.allow_empty = allow_empty

    def loads_key(self, app: AppT, message: Message, *, 
                 loads: Optional[Callable] = None, 
                 serializer: Optional[CodecArg] = None) -> KT:
        if loads is None:
            loads = app.serializers.loads_key
        return cast(KT, loads(self.key_type, message.key, serializer=serializer or self.key_serializer))

    def loads_value(self, app: AppT, message: Message, *, 
                   loads: Optional[Callable] = None, 
                   serializer: Optional[CodecArg] = None) -> VT:
        if loads is None:
            loads = app.serializers.loads_value
        return loads(self.value_type, message.value, serializer=serializer or self.value_serializer)

    def dumps_key(self, app: AppT, key: K, *, 
                 serializer: Optional[CodecArg] = None, 
                 headers: OpenHeadersArg) -> Tuple[Any, OpenHeadersArg]:
        payload = app.serializers.dumps_key(self.key_type, key, serializer=serializer or self.key_serializer)
        return (payload, self.on_dumps_key_prepare_headers(key, headers))

    def dumps_value(self, app: AppT, value: V, *, 
                   serializer: Optional[CodecArg] = None, 
                   headers: OpenHeadersArg) -> Tuple[Any, OpenHeadersArg]:
        payload = app.serializers.dumps_value(self.value_type, value, serializer=serializer or self.value_serializer)
        return (payload, self.on_dumps_value_prepare_headers(value, headers))

    def on_dumps_key_prepare_headers(self, key: K, headers: OpenHeadersArg) -> OpenHeadersArg:
        return headers

    def on_dumps_value_prepare_headers(self, value: V, headers: OpenHeadersArg) -> OpenHeadersArg:
        return headers

    async def decode(self, app: AppT, message: Message, *, propagate: bool = False) -> Optional[EventT]:
        """Decode message from topic (compiled function not cached)."""
        decode = self.compile(app)
        return await decode(message, propagate=propagate)

    def compile(self, app: AppT, *, 
               on_key_decode_error: OnKeyDecodeErrorFun = _noop_decode_error, 
               on_value_decode_error: OnValueDecodeErrorFun = _noop_decode_error, 
               default_propagate: bool = False) -> DecodeFunction:
        """Compile function used to decode event."""
        allow_empty = self.allow_empty
        loads_key = app.serializers.loads_key
        loads_value = app.serializers.loads_value
        create_event = app.create_event
        schema_loads_key = self.loads_key
        schema_loads_value = self.loads_value

        async def decode(message: Message, *, propagate: bool = default_propagate) -> Optional[EventT]:
            try:
                k = schema_loads_key(app, message, loads=loads_key)
            except KeyDecodeError as exc:
                if propagate:
                    raise
                await on_key_decode_error(exc, message)
                return None
            else:
                try:
                    if message.value is None and allow_empty:
                        return create_event(k, None, message.headers, message)
                    v = schema_loads_value(app, message, loads=loads_value)
                except ValueDecodeError as exc:
                    if propagate:
                        raise
                    await on_value_decode_error(exc, message)
                    return None
                else:
                    return create_event(k, v, message.headers, message)
        return decode

    def __repr__(self) -> str:
        KT = self.key_type if self.key_type else '*default*'
        VT = self.key_type if self.value_type else '*default*'
        ks = self.key_serializer if self.key_serializer else '*default*'
        vs = self.value_serializer if self.value_serializer else '*default*'
        return f'<{type(self).__name__}: KT={KT} ({ks}) VT={VT} ({vs})>'

def _model_serializer(typ: Any) -> Optional[CodecArg]:
    with suppress(AttributeError):
        return typ._options.serializer
    return None
