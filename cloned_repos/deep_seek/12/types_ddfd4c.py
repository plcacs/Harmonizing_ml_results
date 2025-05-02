from __future__ import annotations as _annotations
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
from typing import (
    TYPE_CHECKING, 
    Annotated, 
    Any, 
    Callable, 
    ClassVar, 
    Generic, 
    Literal, 
    TypeVar, 
    Union, 
    cast,
    Optional,
    Dict,
    List,
    Set,
    FrozenSet,
    Tuple,
    Type,
    TypeAlias,
    overload
)
from uuid import UUID
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
    'Strict', 'StrictStr', 'SocketPath', 'conbytes', 'conlist', 'conset', 
    'confrozenset', 'constr', 'ImportString', 'conint', 'PositiveInt', 
    'NegativeInt', 'NonNegativeInt', 'NonPositiveInt', 'confloat', 
    'PositiveFloat', 'NegativeFloat', 'NonNegativeFloat', 'NonPositiveFloat', 
    'FiniteFloat', 'condecimal', 'UUID1', 'UUID3', 'UUID4', 'UUID5', 
    'FilePath', 'DirectoryPath', 'NewPath', 'Json', 'Secret', 'SecretStr', 
    'SecretBytes', 'StrictBool', 'StrictBytes', 'StrictInt', 'StrictFloat', 
    'PaymentCardNumber', 'ByteSize', 'PastDate', 'FutureDate', 'PastDatetime', 
    'FutureDatetime', 'condate', 'AwareDatetime', 'NaiveDatetime', 
    'AllowInfNan', 'EncoderProtocol', 'EncodedBytes', 'EncodedStr', 
    'Base64Encoder', 'Base64Bytes', 'Base64Str', 'Base64UrlBytes', 
    'Base64UrlStr', 'GetPydanticSchema', 'StringConstraints', 'Tag', 
    'Discriminator', 'JsonValue', 'OnErrorOmit', 'FailFast'
)

T = TypeVar('T')
SecretType = TypeVar('SecretType')
HashableItemType = TypeVar('HashableItemType', bound=Hashable)
AnyItemType = TypeVar('AnyItemType')
AnyType = TypeVar('AnyType')

@_dataclasses.dataclass
class Strict(_fields.PydanticMetadata, BaseMetadata):
    strict: bool = True

    def __hash__(self) -> int:
        return hash(self.strict)

StrictBool: TypeAlias = Annotated[bool, Strict()]

def conint(
    *, 
    strict: Optional[bool] = None, 
    gt: Optional[int] = None, 
    ge: Optional[int] = None, 
    lt: Optional[int] = None, 
    le: Optional[int] = None, 
    multiple_of: Optional[int] = None
) -> Type[int]:
    return Annotated[
        int, 
        Strict(strict) if strict is not None else None, 
        annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), 
        annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None
    ]

PositiveInt: TypeAlias = Annotated[int, annotated_types.Gt(0)]
NegativeInt: TypeAlias = Annotated[int, annotated_types.Lt(0)]
NonPositiveInt: TypeAlias = Annotated[int, annotated_types.Le(0)]
NonNegativeInt: TypeAlias = Annotated[int, annotated_types.Ge(0)]
StrictInt: TypeAlias = Annotated[int, Strict()]

@_dataclasses.dataclass
class AllowInfNan(_fields.PydanticMetadata):
    allow_inf_nan: bool = True

    def __hash__(self) -> int:
        return hash(self.allow_inf_nan)

def confloat(
    *, 
    strict: Optional[bool] = None, 
    gt: Optional[float] = None, 
    ge: Optional[float] = None, 
    lt: Optional[float] = None, 
    le: Optional[float] = None, 
    multiple_of: Optional[float] = None, 
    allow_inf_nan: Optional[bool] = None
) -> Type[float]:
    return Annotated[
        float, 
        Strict(strict) if strict is not None else None, 
        annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), 
        annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None, 
        AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None
    ]

PositiveFloat: TypeAlias = Annotated[float, annotated_types.Gt(0)]
NegativeFloat: TypeAlias = Annotated[float, annotated_types.Lt(0)]
NonPositiveFloat: TypeAlias = Annotated[float, annotated_types.Le(0)]
NonNegativeFloat: TypeAlias = Annotated[float, annotated_types.Ge(0)]
StrictFloat: TypeAlias = Annotated[float, Strict(True)]
FiniteFloat: TypeAlias = Annotated[float, AllowInfNan(False)]

def conbytes(
    *, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None, 
    strict: Optional[bool] = None
) -> Type[bytes]:
    return Annotated[
        bytes, 
        Strict(strict) if strict is not None else None, 
        annotated_types.Len(min_length or 0, max_length)
    ]

StrictBytes: TypeAlias = Annotated[bytes, Strict()]

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
        if (self.strip_whitespace is not None or 
            self.pattern is not None or 
            self.to_lower is not None or 
            self.to_upper is not None):
            yield _fields.pydantic_general_metadata(
                strip_whitespace=self.strip_whitespace,
                to_upper=self.to_upper,
                to_lower=self.to_lower,
                pattern=self.pattern
            )

def constr(
    *, 
    strip_whitespace: Optional[bool] = None, 
    to_upper: Optional[bool] = None, 
    to_lower: Optional[bool] = None, 
    strict: Optional[bool] = None, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None, 
    pattern: Optional[str] = None
) -> Type[str]:
    return Annotated[
        str, 
        StringConstraints(
            strip_whitespace=strip_whitespace,
            to_upper=to_upper,
            to_lower=to_lower,
            strict=strict,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern
        )
    ]

StrictStr: TypeAlias = Annotated[str, Strict()]

def conset(
    item_type: Type[HashableItemType], 
    *, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None
) -> Type[Set[HashableItemType]]:
    return Annotated[
        set[item_type], 
        annotated_types.Len(min_length or 0, max_length)
    ]

def confrozenset(
    item_type: Type[HashableItemType], 
    *, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None
) -> Type[FrozenSet[HashableItemType]]:
    return Annotated[
        frozenset[item_type], 
        annotated_types.Len(min_length or 0, max_length)
    ]

def conlist(
    item_type: Type[AnyItemType], 
    *, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None, 
    unique_items: Optional[bool] = None
) -> Type[List[AnyItemType]]:
    if unique_items is not None:
        raise PydanticUserError(
            '`unique_items` is removed, use `Set` instead(this feature is discussed in https://github.com/pydantic/pydantic-core/issues/296)', 
            code='removed-kwargs'
        )
    return Annotated[
        list[item_type], 
        annotated_types.Len(min_length or 0, max_length)
    ]

if TYPE_CHECKING:
    ImportString: TypeAlias = Annotated[AnyType, ...]
else:
    class ImportString:
        @classmethod
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, cls()]

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            serializer = core_schema.plain_serializer_function_ser_schema(
                cls._serialize, 
                when_used='json'
            )
            if cls is source:
                return core_schema.no_info_plain_validator_function(
                    function=_validators.import_string, 
                    serialization=serializer
                )
            else:
                return core_schema.no_info_before_validator_function(
                    function=_validators.import_string, 
                    schema=handler(source), 
                    serialization=serializer
                )

        @classmethod
        def __get_pydantic_json_schema__(cls, cs: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
            return handler(core_schema.str_schema())

        @staticmethod
        def _serialize(v: Any) -> str:
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
                return str(v)

        def __repr__(self) -> str:
            return 'ImportString'

def condecimal(
    *, 
    strict: Optional[bool] = None, 
    gt: Optional[Decimal] = None, 
    ge: Optional[Decimal] = None, 
    lt: Optional[Decimal] = None, 
    le: Optional[Decimal] = None, 
    multiple_of: Optional[Decimal] = None, 
    max_digits: Optional[int] = None, 
    decimal_places: Optional[int] = None, 
    allow_inf_nan: Optional[bool] = None
) -> Type[Decimal]:
    return Annotated[
        Decimal, 
        Strict(strict) if strict is not None else None, 
        annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), 
        annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None, 
        _fields.pydantic_general_metadata(max_digits=max_digits, decimal_places=decimal_places), 
        AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None
    ]

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class UuidVersion:
    uuid_version: int

    def __get_pydantic_json_schema__(self, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        field_schema.pop('anyOf', None)
        field_schema.update(type='string', format=f'uuid{self.uuid_version}')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        if isinstance(self, source):
            return core_schema.uuid_schema(version=self.uuid_version)
        else:
            schema = handler(source)
            _check_annotated_type(schema['type'], 'uuid', self.__class__.__name__)
            schema['version'] = self.uuid_version
            return schema

    def __hash__(self) -> int:
        return hash(type(self.uuid_version))

UUID1: TypeAlias = Annotated[UUID, UuidVersion(1)]
UUID3: TypeAlias = Annotated[UUID, UuidVersion(3)]
UUID4: TypeAlias = Annotated[UUID, UuidVersion(4)]
UUID5: TypeAlias = Annotated[UUID, UuidVersion(5)]

@_dataclasses.dataclass
class PathType:
    path_type: str

    def __get_pydantic_json_schema__(self, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        format_conversion = {'file': 'file-path', 'dir': 'directory-path'}
        field_schema.update(format=format_conversion.get(self.path_type, 'path'), type='string')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        function_lookup = {
            'file': cast(core_schema.WithInfoValidatorFunction, self.validate_file),
            'dir': cast(core_schema.WithInfoValidatorFunction, self.validate_directory),
            'new': cast(core_schema.WithInfoValidatorFunction, self.validate_new),
            'socket': cast(core_schema.WithInfoValidatorFunction, self.validate_socket)
        }
        return core_schema.with_info_after_validator_function(
            function_lookup[self.path_type], 
            handler(source)
        )

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

FilePath: TypeAlias = Annotated[Path, PathType('file')]
DirectoryPath: TypeAlias = Annotated[Path, PathType('dir')]
NewPath: TypeAlias = Annotated[Path, PathType('new')]
SocketPath: TypeAlias = Annotated[Path, PathType('socket')]

if TYPE_CHECKING:
    Json: TypeAlias = Annotated[AnyType, ...]
else:
    class Json:
        @classmethod
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, cls()]

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
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

class _SecretBase(Generic[SecretType]):
    def __init__(self, secret_value: SecretType) -> None:
        self._secret_value = secret_value

    def get_secret_value(self) -> SecretType:
        return self._secret_value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.get_secret_value() == other.get_secret_value()

    def __hash__(self) -> int:
        return hash(self.get_secret_value())

    def __str__(self) -> str:
        return str(self._display())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._display()!r})'

    def _display(self) -> Any:
        raise NotImplementedError

def _serialize_secret(value: