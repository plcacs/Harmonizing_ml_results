from faust.types.models import ModelArg
from faust.types.serializers import KT, VT
from faust.types.tuples import Message
from mypy_extensions import DefaultNamedArg
from typing import Any, Awaitable, Callable, Optional, Tuple

DecodeFunction = Callable[[Message, DefaultNamedArg(bool, 'propagate')], Awaitable[EventT]]
OnKeyDecodeErrorFun = Callable[[Exception, Message], Awaitable[None]]
OnValueDecodeErrorFun = Callable[[Exception, Message], Awaitable[None]]

async def _noop_decode_error(exc: Exception, message: Message) -> None:
    ...

class Schema(SchemaT):

    def __init__(self, *, key_type: Optional[ModelArg] = None, value_type: Optional[ModelArg] = None, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None, allow_empty: Optional[bool] = None) -> None:
        self.update(key_type=key_type, value_type=value_type, key_serializer=key_serializer, value_serializer=value_serializer, allow_empty=allow_empty)

    def update(self, *, key_type: Optional[ModelArg] = None, value_type: Optional[ModelArg] = None, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None, allow_empty: Optional[bool] = None) -> None:
        ...

    def loads_key(self, app: AppT, message: Message, *, loads: Optional[Any] = None, serializer: Optional[Any] = None) -> KT:
        ...

    def loads_value(self, app: AppT, message: Message, *, loads: Optional[Any] = None, serializer: Optional[Any] = None) -> VT:
        ...

    def dumps_key(self, app: AppT, key: KT, *, serializer: Optional[Any] = None, headers: Any) -> Tuple[Any, Any]:
        ...

    def dumps_value(self, app: AppT, value: VT, *, serializer: Optional[Any] = None, headers: Any) -> Tuple[Any, Any]:
        ...

    def on_dumps_key_prepare_headers(self, key: Any, headers: Any) -> Any:
        ...

    def on_dumps_value_prepare_headers(self, value: Any, headers: Any) -> Any:
        ...

    async def decode(self, app: AppT, message: Message, *, propagate: bool = False) -> EventT:
        ...

    def compile(self, app: AppT, *, on_key_decode_error: Callable[[Exception, Message], Awaitable[None]] = _noop_decode_error, on_value_decode_error: Callable[[Exception, Message], Awaitable[None]] = _noop_decode_error, default_propagate: bool = False) -> DecodeFunction:
        ...

    def __repr__(self) -> str:
        ...
