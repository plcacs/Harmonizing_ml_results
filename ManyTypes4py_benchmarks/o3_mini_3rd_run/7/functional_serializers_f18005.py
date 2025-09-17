from __future__ import annotations
import dataclasses
from functools import partial, partialmethod
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, TypeVar, overload, Optional
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler, WhenUsed
from typing_extensions import TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler

F = TypeVar("F", bound=Callable[..., Any])

@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class PlainSerializer:
    func: Callable[..., Any]
    return_type: Any = PydanticUndefined
    when_used: str = 'always'

    def __get_pydantic_core_schema__(self, source_type: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        schema: core_schema.CoreSchema = handler(source_type)
        try:
            rt: Any = _decorators.get_function_return_type(self.func, self.return_type, localns=handler._get_types_namespace().locals)
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema = None if rt is PydanticUndefined else handler.generate_schema(rt)
        schema['serialization'] = core_schema.plain_serializer_function_ser_schema(
            function=self.func,
            info_arg=_decorators.inspect_annotated_serializer(self.func, 'plain'),
            return_schema=return_schema,
            when_used=self.when_used
        )
        return schema

@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class WrapSerializer:
    func: Callable[..., Any]
    return_type: Any = PydanticUndefined
    when_used: str = 'always'

    def __get_pydantic_core_schema__(self, source_type: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        schema: core_schema.CoreSchema = handler(source_type)
        globalns, localns = handler._get_types_namespace()
        try:
            rt: Any = _decorators.get_function_return_type(self.func, self.return_type, localns=localns.locals)
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema = None if rt is PydanticUndefined else handler.generate_schema(rt)
        schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
            function=self.func,
            info_arg=_decorators.inspect_annotated_serializer(self.func, 'wrap'),
            return_schema=return_schema,
            when_used=self.when_used
        )
        return schema

if TYPE_CHECKING:
    _Partial = 'partial[Any] | partialmethod[Any]'
    FieldPlainSerializer: TypeAlias = 'core_schema.SerializerFunction | _Partial'
    FieldWrapSerializer: TypeAlias = 'core_schema.WrapSerializerFunction | _Partial'
    FieldSerializer: TypeAlias = 'FieldPlainSerializer | FieldWrapSerializer'
    _FieldPlainSerializerT = TypeVar('_FieldPlainSerializerT', bound=FieldPlainSerializer)
    _FieldWrapSerializerT = TypeVar('_FieldWrapSerializerT', bound=FieldWrapSerializer)

@overload
def field_serializer(field: Any, *fields: Any, mode: Any, return_type: Any = ..., when_used: Any = ..., check_fields: Any = ...) -> Callable[[F], F]:
    ...

@overload
def field_serializer(field: Any, *fields: Any, mode: Any = ..., return_type: Any = ..., when_used: Any = ..., check_fields: Any = ...) -> Callable[[F], F]:
    ...

def field_serializer(*fields: Any, mode: Literal['plain', 'wrap'] = 'plain', return_type: Any = PydanticUndefined, when_used: str = 'always', check_fields: Optional[bool] = None) -> Callable[[F], F]:
    def dec(f: F) -> F:
        dec_info = _decorators.FieldSerializerDecoratorInfo(
            fields=fields, mode=mode, return_type=return_type, when_used=when_used, check_fields=check_fields
        )
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    return dec

if TYPE_CHECKING:
    ModelPlainSerializerWithInfo = Callable[[Any, SerializationInfo], Any]
    ModelPlainSerializerWithoutInfo = Callable[[Any], Any]
    ModelPlainSerializer: TypeAlias = 'ModelPlainSerializerWithInfo | ModelPlainSerializerWithoutInfo'
    ModelWrapSerializerWithInfo = Callable[[Any, SerializerFunctionWrapHandler, SerializationInfo], Any]
    ModelWrapSerializerWithoutInfo = Callable[[Any, SerializerFunctionWrapHandler], Any]
    ModelWrapSerializer: TypeAlias = 'ModelWrapSerializerWithInfo | ModelWrapSerializerWithoutInfo'
    ModelSerializer: TypeAlias = 'ModelPlainSerializer | ModelWrapSerializer'
    _ModelPlainSerializerT = TypeVar('_ModelPlainSerializerT', bound=ModelPlainSerializer)
    _ModelWrapSerializerT = TypeVar('_ModelWrapSerializerT', bound=ModelWrapSerializer)

@overload
def model_serializer(f: F, /) -> F:
    ...

@overload
def model_serializer(*, mode: Literal['plain', 'wrap'], when_used: str = 'always', return_type: Any = ...) -> Callable[[F], F]:
    ...

@overload
def model_serializer(*, mode: Literal['plain', 'wrap'] = 'plain', when_used: str = 'always', return_type: Any = ...) -> Callable[[F], F]:
    ...

def model_serializer(f: Optional[F] = None, *, mode: Literal['plain', 'wrap'] = 'plain', when_used: str = 'always', return_type: Any = PydanticUndefined) -> Callable[[F], F] | F:
    def dec(func: F) -> F:
        dec_info = _decorators.ModelSerializerDecoratorInfo(
            mode=mode, return_type=return_type, when_used=when_used
        )
        return _decorators.PydanticDescriptorProxy(func, dec_info)
    if f is None:
        return dec
    else:
        return dec(f)

AnyType = TypeVar('AnyType')
if TYPE_CHECKING:
    SerializeAsAny: TypeAlias = Annotated[AnyType, ...]
else:

    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SerializeAsAny:
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, SerializeAsAny()]

        def __get_pydantic_core_schema__(self, source_type: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            schema: core_schema.CoreSchema = handler(source_type)
            schema_to_update = schema
            while schema_to_update['type'] == 'definitions':
                schema_to_update = schema_to_update.copy()
                schema_to_update = schema_to_update['schema']
            schema_to_update['serialization'] = core_schema.wrap_serializer_function_ser_schema(
                lambda x, h: h(x), schema=core_schema.any_schema()
            )
            return schema

        __hash__ = object.__hash__