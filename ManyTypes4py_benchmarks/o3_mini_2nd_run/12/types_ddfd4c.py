from __future__ import annotations
import base64
import dataclasses as _dataclasses
import re
from collections.abc import Hashable, Iterator
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from re import Pattern
from types import ModuleType
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Generic, Literal, Optional, TypeVar, Union, cast
import annotated_types
from annotated_types import BaseMetadata, MaxLen, MinLen
from pydantic_core import CoreSchema, PydanticCustomError, SchemaSerializer, core_schema
from typing_extensions import Protocol, TypeAlias, TypeAliasType, deprecated, get_args, get_origin
from ._internal import _fields, _internal_dataclass, _typing_extra, _utils, _validators
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .errors import PydanticUserError
from .json_schema import JsonSchemaValue
from .warnings import PydanticDeprecatedSince20

if TYPE_CHECKING:
    from ._internal._core_metadata import CoreMetadata

__all__ = (
    'Strict', 'StrictStr', 'SocketPath', 'conbytes', 'conlist', 'conset', 'confrozenset', 'constr',
    'ImportString', 'conint', 'PositiveInt', 'NegativeInt', 'NonNegativeInt', 'NonPositiveInt',
    'confloat', 'PositiveFloat', 'NegativeFloat', 'NonNegativeFloat', 'NonPositiveFloat', 'FiniteFloat',
    'condecimal', 'UUID1', 'UUID3', 'UUID4', 'UUID5', 'FilePath', 'DirectoryPath', 'NewPath', 'Json',
    'Secret', 'SecretStr', 'SecretBytes', 'StrictBool', 'StrictBytes', 'StrictInt', 'StrictFloat',
    'PaymentCardNumber', 'ByteSize', 'PastDate', 'FutureDate', 'PastDatetime', 'FutureDatetime', 'condate',
    'AwareDatetime', 'NaiveDatetime', 'AllowInfNan', 'EncoderProtocol', 'EncodedBytes', 'EncodedStr', 'Base64Encoder',
    'Base64Bytes', 'Base64Str', 'Base64UrlBytes', 'Base64UrlStr', 'GetPydanticSchema', 'StringConstraints',
    'Tag', 'Discriminator', 'JsonValue', 'OnErrorOmit', 'FailFast'
)

T = TypeVar('T')

@_dataclasses.dataclass
class Strict(_fields.PydanticMetadata, BaseMetadata):
    strict: bool = True

    def __hash__(self) -> int:
        return hash(self.strict)

StrictBool = Annotated[bool, Strict()]
'A boolean that must be either ``True`` or ``False``.'

def conint(*, strict: Optional[bool] = None, gt: Optional[int] = None, ge: Optional[int] = None, lt: Optional[int] = None, le: Optional[int] = None, multiple_of: Optional[int] = None) -> Any:
    return Annotated[int, (Strict(strict) if strict is not None else None), annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), (annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None)]

PositiveInt = Annotated[int, annotated_types.Gt(0)]
"An integer that must be greater than zero."
NegativeInt = Annotated[int, annotated_types.Lt(0)]
"An integer that must be less than zero."
NonPositiveInt = Annotated[int, annotated_types.Le(0)]
"An integer that must be less than or equal to zero."
NonNegativeInt = Annotated[int, annotated_types.Ge(0)]
"An integer that must be greater than or equal to zero."
StrictInt = Annotated[int, Strict()]
"An integer that must be validated in strict mode."

@_dataclasses.dataclass
class AllowInfNan(_fields.PydanticMetadata):
    allow_inf_nan: bool = True

    def __hash__(self) -> int:
        return hash(self.allow_inf_nan)

def confloat(*, strict: Optional[bool] = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, multiple_of: Optional[float] = None, allow_inf_nan: Optional[bool] = None) -> Any:
    return Annotated[float, (Strict(strict) if strict is not None else None), annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), (annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None), (AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None)]

PositiveFloat = Annotated[float, annotated_types.Gt(0)]
"An float that must be greater than zero."
NegativeFloat = Annotated[float, annotated_types.Lt(0)]
"A float that must be less than zero."
NonPositiveFloat = Annotated[float, annotated_types.Le(0)]
"A float that must be less than or equal to zero."
NonNegativeFloat = Annotated[float, annotated_types.Ge(0)]
"A float that must be greater than or equal to zero."
StrictFloat = Annotated[float, Strict(True)]
"A float that must be validated in strict mode."
FiniteFloat = Annotated[float, AllowInfNan(False)]
'A float that must be finite (not ``-inf``, ``inf``, or ``nan``).'

def conbytes(*, min_length: Optional[int] = None, max_length: Optional[int] = None, strict: Optional[bool] = None) -> Any:
    return Annotated[bytes, (Strict(strict) if strict is not None else None), annotated_types.Len(min_length or 0, max_length)]

StrictBytes = Annotated[bytes, Strict()]
'A bytes that must be validated in strict mode.'

@_dataclasses.dataclass(frozen=True)
class StringConstraints(annotated_types.GroupedMetadata):
    strip_whitespace: Optional[bool] = None
    to_upper: Optional[bool] = None
    to_lower: Optional[bool] = None
    strict: Optional[bool] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    def __iter__(self) -> Iterator[Any]:
        if self.min_length is not None:
            yield MinLen(self.min_length)
        if self.max_length is not None:
            yield MaxLen(self.max_length)
        if self.strict is not None:
            yield Strict(self.strict)
        if self.strip_whitespace is not None or self.pattern is not None or self.to_lower is not None or (self.to_upper is not None):
            yield _fields.pydantic_general_metadata(strip_whitespace=self.strip_whitespace, to_upper=self.to_upper, to_lower=self.to_lower, pattern=self.pattern)

def constr(*, strip_whitespace: Optional[bool] = None, to_upper: Optional[bool] = None, to_lower: Optional[bool] = None, strict: Optional[bool] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None) -> Any:
    return Annotated[str, StringConstraints(strip_whitespace=strip_whitespace, to_upper=to_upper, to_lower=to_lower, strict=strict, min_length=min_length, max_length=max_length, pattern=pattern)]

StrictStr = Annotated[str, Strict()]
'A string that must be validated in strict mode.'

HashableItemType = TypeVar('HashableItemType', bound=Hashable)

def conset(item_type: Any, *, min_length: Optional[int] = None, max_length: Optional[int] = None) -> Any:
    return Annotated[set[item_type], annotated_types.Len(min_length or 0, max_length)]

def confrozenset(item_type: Any, *, min_length: Optional[int] = None, max_length: Optional[int] = None) -> Any:
    return Annotated[frozenset[item_type], annotated_types.Len(min_length or 0, max_length)]

AnyItemType = TypeVar('AnyItemType')

def conlist(item_type: Any, *, min_length: Optional[int] = None, max_length: Optional[int] = None, unique_items: Optional[Any] = None) -> Any:
    if unique_items is not None:
        raise PydanticUserError('`unique_items` is removed, use `Set` instead(this feature is discussed in https://github.com/pydantic/pydantic-core/issues/296)', code='removed-kwargs')
    return Annotated[list[item_type], annotated_types.Len(min_length or 0, max_length)]

AnyType = TypeVar('AnyType')
if TYPE_CHECKING:
    ImportString = Annotated[AnyType, ...]
else:
    class ImportString:
        @classmethod
        def __class_getitem__(cls, item: AnyType) -> Any:
            return Annotated[item, cls()]

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            serializer = core_schema.plain_serializer_function_ser_schema(cls._serialize, when_used='json')
            if cls is source:
                return core_schema.no_info_plain_validator_function(function=_validators.import_string, serialization=serializer)
            else:
                return core_schema.no_info_before_validator_function(function=_validators.import_string, schema=handler(source), serialization=serializer)

        @classmethod
        def __get_pydantic_json_schema__(cls, cs: Any, handler: Callable[[Any], Any]) -> Any:
            return handler(core_schema.str_schema())

        @staticmethod
        def _serialize(v: Any) -> Any:
            if isinstance(v, ModuleType):
                return v.__name__
            elif hasattr(v, '__module__') and hasattr(v, '__name__'):
                return f'{v.__module__}.{v.__name__}'
            elif hasattr(v, 'name'):
                if v.name == '<stdout>':
                    return 'sys.stdout'
                elif v.name == '<stdin>':
                    return 'sys.stdin'
                elif v.name == '<stderr>':
                    return 'sys.stderr'
            else:
                return v

        def __repr__(self) -> str:
            return 'ImportString'

def condecimal(*, strict: Optional[bool] = None, gt: Optional[Decimal] = None, ge: Optional[Decimal] = None, lt: Optional[Decimal] = None, le: Optional[Decimal] = None, multiple_of: Optional[Decimal] = None, max_digits: Optional[int] = None, decimal_places: Optional[int] = None, allow_inf_nan: Optional[bool] = None) -> Any:
    return Annotated[Decimal, (Strict(strict) if strict is not None else None), annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), (annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None), _fields.pydantic_general_metadata(max_digits=max_digits, decimal_places=decimal_places), (AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None)]

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class UuidVersion:
    uuid_version: int

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema)
        field_schema.pop('anyOf', None)
        field_schema.update(type='string', format=f'uuid{self.uuid_version}')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: Callable[[Any], Any]) -> Any:
        if isinstance(self, source):
            return core_schema.uuid_schema(version=self.uuid_version)
        else:
            schema = handler(source)
            _check_annotated_type(schema['type'], 'uuid', self.__class__.__name__)
            schema['version'] = self.uuid_version
            return schema

    def __hash__(self) -> int:
        return hash(type(self.uuid_version))

UUID1 = Annotated[UUID, UuidVersion(1)]
'A [UUID](https://docs.python.org/3/library/uuid.html) that must be version 1.'
UUID3 = Annotated[UUID, UuidVersion(3)]
"A [UUID](https://docs.python.org/3/library/uuid.html) that must be version 3."
UUID4 = Annotated[UUID, UuidVersion(4)]
'A [UUID](https://docs.python.org/3/library/uuid.html) that must be version 4.'
UUID5 = Annotated[UUID, UuidVersion(5)]
"A [UUID](https://docs.python.org/3/library/uuid.html) that must be version 5."

@_dataclasses.dataclass
class PathType:
    path_type: str

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema)
        format_conversion = {'file': 'file-path', 'dir': 'directory-path'}
        field_schema.update(format=format_conversion.get(self.path_type, 'path'), type='string')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: Callable[[Any], Any]) -> Any:
        function_lookup: dict[str, Callable[[Any, Any], Any]] = {
            'file': cast(Callable[[Any, Any], Any], self.validate_file),
            'dir': cast(Callable[[Any, Any], Any], self.validate_directory),
            'new': cast(Callable[[Any, Any], Any], self.validate_new),
            'socket': cast(Callable[[Any, Any], Any], self.validate_socket)
        }
        return core_schema.with_info_after_validator_function(function_lookup[self.path_type], handler(source))

    @staticmethod
    def validate_file(path: Path, _: Any) -> Path:
        if path.is_file():
            return path
        else:
            raise PydanticCustomError('path_not_file', 'Path does not point to a file')

    @staticmethod
    def validate_socket(path: Path, _: Any) -> Path:
        if path.is_socket():
            return path
        else:
            raise PydanticCustomError('path_not_socket', 'Path does not point to a socket')

    @staticmethod
    def validate_directory(path: Path, _: Any) -> Path:
        if path.is_dir():
            return path
        else:
            raise PydanticCustomError('path_not_directory', 'Path does not point to a directory')

    @staticmethod
    def validate_new(path: Path, _: Any) -> Path:
        if path.exists():
            raise PydanticCustomError('path_exists', 'Path already exists')
        elif not path.parent.exists():
            raise PydanticCustomError('parent_does_not_exist', 'Parent directory does not exist')
        else:
            return path

    def __hash__(self) -> int:
        return hash(type(self.path_type))

FilePath = Annotated[Path, PathType('file')]
"A path that must point to a file."
DirectoryPath = Annotated[Path, PathType('dir')]
"A path that must point to a directory."
NewPath = Annotated[Path, PathType('new')]
'A path for a new file or directory that must not already exist. The parent directory must already exist.'
SocketPath = Annotated[Path, PathType('socket')]
'A path to an existing socket file'

if TYPE_CHECKING:
    Json = Annotated[Any, ...]
else:
    class Json:
        @classmethod
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, cls()]

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            if cls is source:
                return core_schema.json_schema(None)
            else:
                return core_schema.json_schema(handler(source))

        def __repr__(self) -> str:
            return 'Json'

        def __hash__(self) -> int:
            return hash(type(self))

        def __eq__(self, other: Any) -> bool:
            return type(other) is type(self)

def condate(*, strict: Optional[bool] = None, gt: Optional[date] = None, ge: Optional[date] = None, lt: Optional[date] = None, le: Optional[date] = None) -> Any:
    return Annotated[date, (Strict(strict) if strict is not None else None), annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le)]

if TYPE_CHECKING:
    AwareDatetime = Annotated[datetime, ...]
    NaiveDatetime = Annotated[datetime, ...]
    PastDatetime = Annotated[datetime, ...]
    FutureDatetime = Annotated[datetime, ...]
else:
    class AwareDatetime:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            if cls is source:
                return core_schema.datetime_schema(tz_constraint='aware')
            else:
                schema = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['tz_constraint'] = 'aware'
                return schema

        def __repr__(self) -> str:
            return 'AwareDatetime'

    class NaiveDatetime:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            if cls is source:
                return core_schema.datetime_schema(tz_constraint='naive')
            else:
                schema = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['tz_constraint'] = 'naive'
                return schema

        def __repr__(self) -> str:
            return 'NaiveDatetime'

    class PastDatetime:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            if cls is source:
                return core_schema.datetime_schema(now_op='past')
            else:
                schema = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['now_op'] = 'past'
                return schema

        def __repr__(self) -> str:
            return 'PastDatetime'

    class FutureDatetime:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            if cls is source:
                return core_schema.datetime_schema(now_op='future')
            else:
                schema = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['now_op'] = 'future'
                return schema

        def __repr__(self) -> str:
            return 'FutureDatetime'

class EncoderProtocol(Protocol):
    @classmethod
    def decode(cls, data: Any) -> Any:
        ...

    @classmethod
    def encode(cls, value: Any) -> Any:
        ...

    @classmethod
    def get_json_format(cls) -> str:
        ...

class Base64Encoder(EncoderProtocol):
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        try:
            return base64.b64decode(data)
        except ValueError as e:
            raise PydanticCustomError('base64_decode', "Base64 decoding error: '{error}'", {'error': str(e)})

    @classmethod
    def encode(cls, value: bytes) -> bytes:
        return base64.b64encode(value)

    @classmethod
    def get_json_format(cls) -> str:
        return 'base64'

class Base64UrlEncoder(EncoderProtocol):
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        try:
            return base64.urlsafe_b64decode(data)
        except ValueError as e:
            raise PydanticCustomError('base64_decode', "Base64 decoding error: '{error}'", {'error': str(e)})

    @classmethod
    def encode(cls, value: bytes) -> bytes:
        return base64.urlsafe_b64encode(value)

    @classmethod
    def get_json_format(cls) -> str:
        return 'base64url'

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedBytes:
    encoder: EncoderProtocol

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: Callable[[Any], Any]) -> Any:
        schema = handler(source)
        _check_annotated_type(schema['type'], 'bytes', self.__class__.__name__)
        return core_schema.with_info_after_validator_function(function=self.decode, schema=schema, serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode))

    def decode(self, data: Any, _: Any) -> Any:
        return self.encoder.decode(data)

    def encode(self, value: Any) -> Any:
        return self.encoder.encode(value)

    def __hash__(self) -> int:
        return hash(self.encoder)

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedStr:
    encoder: EncoderProtocol

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: Callable[[Any], Any]) -> Any:
        schema = handler(source)
        _check_annotated_type(schema['type'], 'str', self.__class__.__name__)
        return core_schema.with_info_after_validator_function(function=self.decode_str, schema=schema, serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode_str))

    def decode_str(self, data: Any, _: Any) -> Any:
        return self.encoder.decode(data.encode()).decode()

    def encode_str(self, value: Any) -> Any:
        return self.encoder.encode(value.encode()).decode()

    def __hash__(self) -> int:
        return hash(self.encoder)

Base64Bytes = Annotated[bytes, EncodedBytes(encoder=Base64Encoder)]
'A bytes type that is encoded and decoded using the standard (non-URL-safe) base64 encoder.'
Base64Str = Annotated[str, EncodedStr(encoder=Base64Encoder)]
"A str type that is encoded and decoded using the standard (non-URL-safe) base64 encoder."
Base64UrlBytes = Annotated[bytes, EncodedBytes(encoder=Base64UrlEncoder)]
'A bytes type that is encoded and decoded using the URL-safe base64 encoder.'
Base64UrlStr = Annotated[str, EncodedStr(encoder=Base64UrlEncoder)]
'A str type that is encoded and decoded using the URL-safe base64 encoder.'

__getattr__ = getattr_migration(__name__)

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class GetPydanticSchema:
    get_pydantic_core_schema: Optional[Callable[[Any, Callable[[Any], Any]], Any]] = None
    get_pydantic_json_schema: Optional[Callable[[Any, Callable[[Any], Any]], Any]] = None
    if not TYPE_CHECKING:
        def __getattr__(self, item: str) -> Any:
            if item == '__get_pydantic_core_schema__' and self.get_pydantic_core_schema:
                return self.get_pydantic_core_schema
            elif item == '__get_pydantic_json_schema__' and self.get_pydantic_json_schema:
                return self.get_pydantic_json_schema
            else:
                return object.__getattribute__(self, item)
    __hash__ = object.__hash__

@_dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class Tag:
    tag: str

    def __get_pydantic_core_schema__(self, source_type: Any, handler: Callable[[Any], Any]) -> Any:
        schema = handler(source_type)
        metadata = cast('CoreMetadata', schema.setdefault('metadata', {}))
        metadata['pydantic_internal_union_tag_key'] = self.tag
        return schema

@_dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class Discriminator:
    discriminator: Union[str, Callable[[Any], str]]
    custom_error_type: Optional[str] = None
    custom_error_message: Optional[str] = None
    custom_error_context: Optional[Any] = None

    def __get_pydantic_core_schema__(self, source_type: Any, handler: Callable[[Any], Any]) -> Any:
        origin = _typing_extra.get_origin(source_type)
        if not origin or not _typing_extra.origin_is_union(origin):
            raise TypeError(f'{type(self).__name__} must be used with a Union type, not {source_type}')
        if isinstance(self.discriminator, str):
            from pydantic import Field
            return handler(Annotated[source_type, Field(discriminator=self.discriminator)])
        else:
            original_schema = handler(source_type)
            return self._convert_schema(original_schema)

    def _convert_schema(self, original_schema: Any) -> Any:
        if original_schema['type'] != 'union':
            original_schema = core_schema.union_schema([original_schema])
        tagged_union_choices: dict[str, Any] = {}
        for choice in original_schema['choices']:
            tag: Optional[str] = None
            if isinstance(choice, tuple):
                choice, tag = choice
            metadata = cast('CoreMetadata | None', choice.get('metadata'))
            if metadata is not None:
                tag = metadata.get('pydantic_internal_union_tag_key') or tag
            if tag is None:
                raise PydanticUserError(f'`Tag` not provided for choice {choice} used with `Discriminator`', code='callable-discriminator-no-tag')
            tagged_union_choices[tag] = choice
        custom_error_type = self.custom_error_type
        if custom_error_type is None:
            custom_error_type = original_schema.get('custom_error_type')
        custom_error_message = self.custom_error_message
        if custom_error_message is None:
            custom_error_message = original_schema.get('custom_error_message')
        custom_error_context = self.custom_error_context
        if custom_error_context is None:
            custom_error_context = original_schema.get('custom_error_context')
        custom_error_type = original_schema.get('custom_error_type') if custom_error_type is None else custom_error_type
        return core_schema.tagged_union_schema(
            tagged_union_choices,
            self.discriminator,
            custom_error_type=custom_error_type,
            custom_error_message=custom_error_message,
            custom_error_context=custom_error_context,
            strict=original_schema.get('strict'),
            ref=original_schema.get('ref'),
            metadata=original_schema.get('metadata'),
            serialization=original_schema.get('serialization')
        )

def _get_type_name(x: Any) -> str:
    type_ = type(x)
    _JSON_TYPES = {int, float, str, bool, list, dict, type(None)}
    if type_ in _JSON_TYPES:
        return type_.__name__
    if isinstance(x, int):
        return 'int'
    if isinstance(x, float):
        return 'float'
    if isinstance(x, str):
        return 'str'
    if isinstance(x, list):
        return 'list'
    if isinstance(x, dict):
        return 'dict'
    return getattr(type_, '__name__', '<no type name>')

class _AllowAnyJson:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], Any]) -> Any:
        python_schema = handler(source_type)
        return core_schema.json_or_python_schema(json_schema=core_schema.any_schema(), python_schema=python_schema)

if TYPE_CHECKING:
    JsonValue = Union[list['JsonValue'], dict[str, 'JsonValue'], str, bool, int, float, None]
else:
    JsonValue = Annotated[
        Union[
            Annotated[list['JsonValue'], Tag('list')],
            Annotated[dict[str, 'JsonValue'], Tag('dict')],
            Annotated[str, Tag('str')],
            Annotated[bool, Tag('bool')],
            Annotated[int, Tag('int')],
            Annotated[float, Tag('float')],
            Annotated[None, Tag('NoneType')]
        ],
        Discriminator(_get_type_name, custom_error_type='invalid-json-value', custom_error_message='input was not a valid JSON value'),
        _AllowAnyJson
    ]

class _OnErrorOmit:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], Any]) -> Any:
        return core_schema.with_default_schema(schema=handler(source_type), on_error='omit')

OnErrorOmit = Annotated[T, _OnErrorOmit]
'''
When used as an item in a list, the key type in a dict, optional values of a TypedDict, etc.
this annotation omits the item from the iteration if there is any error validating it.
That is, instead of a ValidationError being propagated up and the entire iterable being discarded,
any invalid items are discarded and the valid ones are returned.
'''

@_dataclasses.dataclass
class FailFast(_fields.PydanticMetadata, BaseMetadata):
    fail_fast: bool = True
    """A `FailFast` annotation can be used to specify that validation should stop at the first error."""
