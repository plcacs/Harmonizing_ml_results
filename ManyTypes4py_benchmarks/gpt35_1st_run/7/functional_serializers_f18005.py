from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, TypeVar, overload
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler, WhenUsed
from typing_extensions import TypeAlias

@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class PlainSerializer:
    return_type: PydanticUndefined
    when_used: Literal['always'] = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> dict:
        ...

@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class WrapSerializer:
    return_type: PydanticUndefined
    when_used: Literal['always'] = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> dict:
        ...

if TYPE_CHECKING:
    _Partial: str = 'partial[Any] | partialmethod[Any]'
    FieldPlainSerializer: str = 'core_schema.SerializerFunction | _Partial'
    FieldWrapSerializer: str = 'core_schema.WrapSerializerFunction | _Partial'
    FieldSerializer: str = 'FieldPlainSerializer | FieldWrapSerializer'
    _FieldPlainSerializerT = TypeVar('_FieldPlainSerializerT', bound=FieldPlainSerializer)
    _FieldWrapSerializerT = TypeVar('_FieldWrapSerializerT', bound=FieldWrapSerializer)

@overload
def field_serializer(field, /, *fields: str, mode: Literal['plain', 'wrap'], return_type: PydanticUndefined = ..., when_used: Literal['always'] = ..., check_fields: Any = ...):
    ...

def field_serializer(*fields: str, mode: Literal['plain', 'wrap'] = 'plain', return_type: PydanticUndefined = PydanticUndefined, when_used: Literal['always'] = 'always', check_fields: Any = None) -> Callable:
    ...

if TYPE_CHECKING:
    ModelPlainSerializerWithInfo: Callable[[Any, SerializationInfo], Any]
    ModelPlainSerializerWithoutInfo: Callable[[Any], Any]
    ModelPlainSerializer: str = 'ModelPlainSerializerWithInfo | ModelPlainSerializerWithoutInfo'
    ModelWrapSerializerWithInfo: Callable[[Any, SerializerFunctionWrapHandler, SerializationInfo], Any]
    ModelWrapSerializerWithoutInfo: Callable[[Any, SerializerFunctionWrapHandler], Any]
    ModelWrapSerializer: str = 'ModelWrapSerializerWithInfo | ModelWrapSerializerWithoutInfo'
    ModelSerializer: str = 'ModelPlainSerializer | ModelWrapSerializer'
    _ModelPlainSerializerT = TypeVar('_ModelPlainSerializerT', bound=ModelPlainSerializer)
    _ModelWrapSerializerT = TypeVar('_ModelWrapSerializerT', bound=ModelWrapSerializer)

@overload
def model_serializer(f, /) -> Callable:
    ...

@overload
def model_serializer(*, mode: Literal['plain', 'wrap'], when_used: Literal['always'] = ..., return_type: PydanticUndefined = ...) -> Callable:
    ...

def model_serializer(f: Any = None, /, *, mode: Literal['plain', 'wrap'] = 'plain', when_used: Literal['always'] = 'always', return_type: PydanticUndefined = PydanticUndefined) -> Callable:
    ...

if TYPE_CHECKING:
    SerializeAsAny: Annotated[AnyType, ...] = Annotated[AnyType, ...]
else:
    class SerializeAsAny:
        def __class_getitem__(cls, item: Any) -> Annotated:
            ...

        def __get_pydantic_core_schema__(self, source_type: Any, handler: Any) -> dict:
            ...
        __hash__ = object.__hash__
