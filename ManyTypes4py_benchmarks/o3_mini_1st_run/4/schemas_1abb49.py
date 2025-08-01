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
    key_type: Optional[ModelArg] = None
    value_type: Optional[ModelArg] = None
    key_serializer: Optional[CodecArg] = None
    value_serializer: Optional[CodecArg] = None
    allow_empty: Optional[bool] = None

    def __init__(
        self,
        *,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
    ) -> None:
        self.update(
            key_type=key_type,
            value_type=value_type,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            allow_empty=allow_empty,
        )

    def update(
        self,
        *,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
    ) -> None:
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

    def loads_key(
        self,
        app: AppT,
        message: Message,
        *,
        loads: Optional[Callable[..., Any]] = None,
        serializer: Optional[CodecArg] = None,
    ) -> KT:
        if loads is None:
            loads = app.serializers.loads_key
        return cast(KT, loads(self.key_type, message.key, serializer=serializer or self.key_serializer))

    def loads_value(
        self,
        app: AppT,
        message: Message,
        *,
        loads: Optional[Callable[..., Any]] = None,
        serializer: Optional[CodecArg] = None,
    ) -> VT:
        if loads is None:
            loads = app.serializers.loads_value
        return loads(self.value_type, message.value, serializer=serializer or self.value_serializer)

    def dumps_key(
        self,
        app: AppT,
        key: Any,
        *,
        serializer: Optional[CodecArg] = None,
        headers: OpenHeadersArg,
    ) -> Tuple[bytes, OpenHeadersArg]:
        payload: bytes = app.serializers.dumps_key(self.key_type, key, serializer=serializer or self.key_serializer)
        return (payload, self.on_dumps_key_prepare_headers(key, headers))

    def dumps_value(
        self,
        app: AppT,
        value: Any,
        *,
        serializer: Optional[CodecArg] = None,
        headers: OpenHeadersArg,
    ) -> Tuple[bytes, OpenHeadersArg]:
        payload: bytes = app.serializers.dumps_value(self.value_type, value, serializer=serializer or self.value_serializer)
        return (payload, self.on_dumps_value_prepare_headers(value, headers))

    def on_dumps_key_prepare_headers(self, key: Any, headers: OpenHeadersArg) -> OpenHeadersArg:
        return headers

    def on_dumps_value_prepare_headers(self, value: Any, headers: OpenHeadersArg) -> OpenHeadersArg:
        return headers

    async def decode(self, app: AppT, message: Message, *, propagate: bool = False) -> Optional[EventT]:
        """Decode message from topic (compiled function not cached)."""
        decode_func: Callable[[Message, bool], Awaitable[Optional[EventT]]] = self.compile(app)
        return await decode_func(message, propagate=propagate)

    def compile(
        self,
        app: AppT,
        *,
        on_key_decode_error: OnKeyDecodeErrorFun = _noop_decode_error,
        on_value_decode_error: OnValueDecodeErrorFun = _noop_decode_error,
        default_propagate: bool = False,
    ) -> Callable[[Message, bool], Awaitable[Optional[EventT]]]:
        allow_empty: Optional[bool] = self.allow_empty
        loads_key: Callable[..., Any] = app.serializers.loads_key
        loads_value: Callable[..., Any] = app.serializers.loads_value
        create_event: Callable[[Any, Any, OpenHeadersArg, Message], EventT] = app.create_event
        schema_loads_key: Callable[[AppT, Message, Any, Optional[CodecArg]], KT] = self.loads_key  # type: ignore
        schema_loads_value: Callable[[AppT, Message, Any, Optional[CodecArg]], VT] = self.loads_value  # type: ignore

        async def decode(message: Message, *, propagate: bool = default_propagate) -> Optional[EventT]:
            try:
                k: KT = schema_loads_key(app, message, loads=loads_key)
            except KeyDecodeError as exc:
                if propagate:
                    raise
                await on_key_decode_error(exc, message)
                return None
            else:
                try:
                    if message.value is None and allow_empty:
                        return create_event(k, None, message.headers, message)
                    v: VT = schema_loads_value(app, message, loads=loads_value)
                except ValueDecodeError as exc:
                    if propagate:
                        raise
                    await on_value_decode_error(exc, message)
                    return None
                else:
                    return create_event(k, v, message.headers, message)

        return decode

    def __repr__(self) -> str:
        kt_repr = self.key_type if self.key_type else '*default*'
        vt_repr = self.value_type if self.value_type else '*default*'
        ks_repr = self.key_serializer if self.key_serializer else '*default*'
        vs_repr = self.value_serializer if self.value_serializer else '*default*'
        return f'<{type(self).__name__}: KT={kt_repr} ({ks_repr}) VT={vt_repr} ({vs_repr})>'


def _model_serializer(typ: Any) -> Optional[CodecArg]:
    with suppress(AttributeError):
        return getattr(typ, '_options', {}).get('serializer')  # type: ignore
    return None