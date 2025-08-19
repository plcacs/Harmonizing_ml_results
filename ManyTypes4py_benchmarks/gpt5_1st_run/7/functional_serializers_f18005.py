"""This module contains related classes and functions for serialization."""
from __future__ import annotations

import dataclasses
from functools import partial, partialmethod
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, TypeVar, overload

from pydantic_core import PydanticUndefined, core_schema
from pydantic_core.core_schema import CoreSchema, SerializationInfo, SerializerFunctionWrapHandler, WhenUsed
from typing_extensions import TypeAlias

from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler


Mode = Literal['plain', 'wrap']


@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class PlainSerializer:
    """Plain serializers use a function to modify the output of serialization.

    This is particularly helpful when you want to customize the serialization for annotated types.
    Consider an input of `list`, which will be serialized into a space-delimited string.

    ```python
    from typing import Annotated

    from pydantic import BaseModel, PlainSerializer

    CustomStr = Annotated[
        list, PlainSerializer(lambda x: ' '.join(x), return_type=str)
    ]

    class StudentModel(BaseModel):
        courses: CustomStr

    student = StudentModel(courses=['Math', 'Chemistry', 'English'])
    print(student.model_dump())
    #> {'courses': 'Math Chemistry English'}
    ```

    Attributes:
        func: The serializer function.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
        when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
            `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.
    """
    func: Callable[..., Any]
    return_type: Any = PydanticUndefined
    when_used: WhenUsed = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gets the Pydantic core schema.

        Args:
            source_type: The source type.
            handler: The `GetCoreSchemaHandler` instance.

        Returns:
            The Pydantic core schema.
        """
        schema: CoreSchema = handler(source_type)
        try:
            return_type: Any = _decorators.get_function_return_type(
                self.func, self.return_type, localns=handler._get_types_namespace().locals  # type: ignore[attr-defined]
            )
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema: CoreSchema | None = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.plain_serializer_function_ser_schema(
            function=self.func,
            info_arg=_decorators.inspect_annotated_serializer(self.func, 'plain'),
            return_schema=return_schema,
            when_used=self.when_used,
        )
        return schema


@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class WrapSerializer:
    """Wrap serializers receive the raw inputs along with a handler function that applies the standard serialization
    logic, and can modify the resulting value before returning it as the final output of serialization.

    For example, here's a scenario in which a wrap serializer transforms timezones to UTC **and** utilizes the existing `datetime` serialization logic.

    ```python
    from datetime import datetime, timezone
    from typing import Annotated, Any

    from pydantic import BaseModel, WrapSerializer

    class EventDatetime(BaseModel):
        start: datetime
        end: datetime

    def convert_to_utc(value: Any, handler, info) -> dict[str, datetime]:
        # Note that `handler` can actually help serialize the `value` for
        # further custom serialization in case it's a subclass.
        partial_result = handler(value, info)
        if info.mode == 'json':
            return {
                k: datetime.fromisoformat(v).astimezone(timezone.utc)
                for k, v in partial_result.items()
            }
        return {k: v.astimezone(timezone.utc) for k, v in partial_result.items()}

    UTCEventDatetime = Annotated[EventDatetime, WrapSerializer(convert_to_utc)]

    class EventModel(BaseModel):
        event_datetime: UTCEventDatetime

    dt = EventDatetime(
        start='2024-01-01T07:00:00-08:00', end='2024-01-03T20:00:00+06:00'
    )
    event = EventModel(event_datetime=dt)
    print(event.model_dump())
    '''
    {
        'event_datetime': {
            'start': datetime.datetime(
                2024, 1, 1, 15, 0, tzinfo=datetime.timezone.utc
            ),
            'end': datetime.datetime(
                2024, 1, 3, 14, 0, tzinfo=datetime.timezone.utc
            ),
        }
    }
    '''

    print(event.model_dump_json())
    '''
    {"event_datetime":{"start":"2024-01-01T15:00:00Z","end":"2024-01-03T14:00:00Z"}}
    '''
    ```

    Attributes:
        func: The serializer function to be wrapped.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
        when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
            `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.
    """
    func: Callable[..., Any]
    return_type: Any = PydanticUndefined
    when_used: WhenUsed = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """This method is used to get the Pydantic core schema of the class.

        Args:
            source_type: Source type.
            handler: Core schema handler.

        Returns:
            The generated core schema of the class.
        """
        schema: CoreSchema = handler(source_type)
        _globalns, _localns = handler._get_types_namespace()  # type: ignore[assignment]
        try:
            return_type: Any = _decorators.get_function_return_type(
                self.func, self.return_type, localns=handler._get_types_namespace().locals  # type: ignore[attr-defined]
            )
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema: CoreSchema | None = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
            function=self.func,
            info_arg=_decorators.inspect_annotated_serializer(self.func, 'wrap'),
            return_schema=return_schema,
            when_used=self.when_used,
        )
        return schema


if TYPE_CHECKING:
    _Partial: TypeAlias = 'partial[Any] | partialmethod[Any]'
    FieldPlainSerializer: TypeAlias = 'core_schema.SerializerFunction | _Partial'
    'A field serializer method or function in `plain` mode.'
    FieldWrapSerializer: TypeAlias = 'core_schema.WrapSerializerFunction | _Partial'
    'A field serializer method or function in `wrap` mode.'
    FieldSerializer: TypeAlias = 'FieldPlainSerializer | FieldWrapSerializer'
    'A field serializer method or function.'
    _FieldPlainSerializerT = TypeVar('_FieldPlainSerializerT', bound='FieldPlainSerializer')
    _FieldWrapSerializerT = TypeVar('_FieldWrapSerializerT', bound='FieldWrapSerializer')


@overload
def field_serializer(
    field: str,
    /,
    *fields: str,
    mode: Mode,
    return_type: Any = ...,
    when_used: WhenUsed = ...,
    check_fields: bool | None = ...,
) -> Callable[[Callable[..., Any]], Any]:
    ...


@overload
def field_serializer(
    field: str,
    /,
    *fields: str,
    mode: Mode = ...,
    return_type: Any = ...,
    when_used: WhenUsed = ...,
    check_fields: bool | None = ...,
) -> Callable[[Callable[..., Any]], Any]:
    ...


def field_serializer(
    *fields: str,
    mode: Mode = 'plain',
    return_type: Any = PydanticUndefined,
    when_used: WhenUsed = 'always',
    check_fields: bool | None = None,
) -> Callable[[Callable[..., Any]], Any]:
    """Decorator that enables custom field serialization.

    In the below example, a field of type `set` is used to mitigate duplication. A `field_serializer` is used to serialize the data as a sorted list.

    ```python
    from typing import Set

    from pydantic import BaseModel, field_serializer

    class StudentModel(BaseModel):
        name: str = 'Jane'
        courses: Set[str]

        @field_serializer('courses', when_used='json')
        def serialize_courses_in_order(self, courses: Set[str]):
            return sorted(courses)

    student = StudentModel(courses={'Math', 'Chemistry', 'English'})
    print(student.model_dump_json())
    #> {"name":"Jane","courses":["Chemistry","English","Math"]}
    ```

    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

    Four signatures are supported:

    - `(self, value: Any, info: FieldSerializationInfo)`
    - `(self, value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo)`
    - `(value: Any, info: SerializationInfo)`
    - `(value: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo)`

    Args:
        fields: Which field(s) the method should be called on.
        mode: The serialization mode.

            - `plain` means the function will be called instead of the default serialization logic,
            - `wrap` means the function will be called with an argument to optionally call the
               default serialization logic.
        return_type: Optional return type for the function, if omitted it will be inferred from the type annotation.
        when_used: Determines the serializer will be used for serialization.
        check_fields: Whether to check that the fields actually exist on the model.

    Returns:
        The decorator function.
    """

    def dec(f: Callable[..., Any]) -> Any:
        dec_info = _decorators.FieldSerializerDecoratorInfo(
            fields=fields, mode=mode, return_type=return_type, when_used=when_used, check_fields=check_fields
        )
        return _decorators.PydanticDescriptorProxy(f, dec_info)

    return dec


if TYPE_CHECKING:
    ModelPlainSerializerWithInfo = Callable[[Any, SerializationInfo], Any]
    'A model serializer method with the `info` argument, in `plain` mode.'
    ModelPlainSerializerWithoutInfo = Callable[[Any], Any]
    'A model serializer method without the `info` argument, in `plain` mode.'
    ModelPlainSerializer: TypeAlias = 'ModelPlainSerializerWithInfo | ModelPlainSerializerWithoutInfo'
    'A model serializer method in `plain` mode.'
    ModelWrapSerializerWithInfo = Callable[[Any, SerializerFunctionWrapHandler, SerializationInfo], Any]
    'A model serializer method with the `info` argument, in `wrap` mode.'
    ModelWrapSerializerWithoutInfo = Callable[[Any, SerializerFunctionWrapHandler], Any]
    'A model serializer method without the `info` argument, in `wrap` mode.'
    ModelWrapSerializer: TypeAlias = 'ModelWrapSerializerWithInfo | ModelWrapSerializerWithoutInfo'
    'A model serializer method in `wrap` mode.'
    ModelSerializer: TypeAlias = 'ModelPlainSerializer | ModelWrapSerializer'
    _ModelPlainSerializerT = TypeVar('_ModelPlainSerializerT', bound='ModelPlainSerializer')
    _ModelWrapSerializerT = TypeVar('_ModelWrapSerializerT', bound='ModelWrapSerializer')


@overload
def model_serializer(f: Callable[..., Any], /) -> Any:
    ...


@overload
def model_serializer(*, mode: Mode, when_used: WhenUsed = 'always', return_type: Any = ...) -> Callable[[Callable[..., Any]], Any]:
    ...


@overload
def model_serializer(*, mode: Mode = ..., when_used: WhenUsed = 'always', return_type: Any = ...) -> Callable[[Callable[..., Any]], Any]:
    ...


def model_serializer(
    f: Callable[..., Any] | None = None,
    /,
    *,
    mode: Mode = 'plain',
    when_used: WhenUsed = 'always',
    return_type: Any = PydanticUndefined,
) -> Any:
    """Decorator that enables custom model serialization.

    This is useful when a model need to be serialized in a customized manner, allowing for flexibility beyond just specific fields.

    An example would be to serialize temperature to the same temperature scale, such as degrees Celsius.

    ```python
    from typing import Literal

    from pydantic import BaseModel, model_serializer

    class TemperatureModel(BaseModel):
        unit: Literal['C', 'F']
        value: int

        @model_serializer()
        def serialize_model(self):
            if self.unit == 'F':
                return {'unit': 'C', 'value': int((self.value - 32) / 1.8)}
            return {'unit': self.unit, 'value': self.value}

    temperature = TemperatureModel(unit='F', value=212)
    print(temperature.model_dump())
    #> {'unit': 'C', 'value': 100}
    ```

    Two signatures are supported for `mode='plain'`, which is the default:

    - `(self)`
    - `(self, info: SerializationInfo)`

    And two other signatures for `mode='wrap'`:

    - `(self, nxt: SerializerFunctionWrapHandler)`
    - `(self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo)`

        See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

    Args:
        f: The function to be decorated.
        mode: The serialization mode.

            - `'plain'` means the function will be called instead of the default serialization logic
            - `'wrap'` means the function will be called with an argument to optionally call the default
                serialization logic.
        when_used: Determines when this serializer should be used.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.

    Returns:
        The decorator function.
    """

    def dec(f: Callable[..., Any]) -> Any:
        dec_info = _decorators.ModelSerializerDecoratorInfo(mode=mode, return_type=return_type, when_used=when_used)
        return _decorators.PydanticDescriptorProxy(f, dec_info)

    if f is None:
        return dec
    else:
        return dec(f)


AnyType = TypeVar('AnyType')

if TYPE_CHECKING:
    SerializeAsAny = Annotated[AnyType, ...]
    'Force serialization to ignore whatever is defined in the schema and instead ask the object\n    itself how it should be serialized.\n    In particular, this means that when model subclasses are serialized, fields present in the subclass\n    but not in the original schema will be included.\n    '
else:

    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SerializeAsAny:
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, SerializeAsAny()]

        def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            schema_to_update: CoreSchema = schema
            while schema_to_update['type'] == 'definitions':
                schema_to_update = schema_to_update.copy()
                schema_to_update = schema_to_update['schema']  # type: ignore[assignment]
            schema_to_update['serialization'] = core_schema.wrap_serializer_function_ser_schema(
                lambda x, h: h(x), schema=core_schema.any_schema()
            )
            return schema

        __hash__ = object.__hash__