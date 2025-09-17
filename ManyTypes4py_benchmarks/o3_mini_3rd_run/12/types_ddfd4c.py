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
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Generic, Literal, Optional, TypeVar, Union, cast
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
    'Strict', 'StrictStr', 'SocketPath', 'conbytes', 'conlist', 'conset', 'confrozenset', 'constr', 'ImportString', 'conint',
    'PositiveInt', 'NegativeInt', 'NonNegativeInt', 'NonPositiveInt', 'confloat', 'PositiveFloat', 'NegativeFloat',
    'NonNegativeFloat', 'NonPositiveFloat', 'FiniteFloat', 'condecimal', 'UUID1', 'UUID3', 'UUID4', 'UUID5', 'FilePath',
    'DirectoryPath', 'NewPath', 'Json', 'Secret', 'SecretStr', 'SecretBytes', 'StrictBool', 'StrictBytes', 'StrictInt',
    'StrictFloat', 'PaymentCardNumber', 'ByteSize', 'PastDate', 'FutureDate', 'PastDatetime', 'FutureDatetime', 'condate',
    'AwareDatetime', 'NaiveDatetime', 'AllowInfNan', 'EncoderProtocol', 'EncodedBytes', 'EncodedStr', 'Base64Encoder',
    'Base64Bytes', 'Base64Str', 'Base64UrlBytes', 'Base64UrlStr', 'GetPydanticSchema', 'StringConstraints', 'Tag', 'Discriminator',
    'JsonValue', 'OnErrorOmit', 'FailFast'
)
T = TypeVar('T')

@_dataclasses.dataclass
class Strict(_fields.PydanticMetadata, BaseMetadata):
    strict: bool = True

    def __hash__(self) -> int:
        return hash(self.strict)

StrictBool: type = Annotated[bool, Strict()]
'A boolean that must be either ``True`` or ``False``.'

def conint(*, strict: Optional[bool] = None, gt: Optional[int] = None, ge: Optional[int] = None, lt: Optional[int] = None, le: Optional[int] = None, multiple_of: Optional[int] = None) -> Any:
    return Annotated[int, Strict(strict) if strict is not None else None, annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None]

PositiveInt: type = Annotated[int, annotated_types.Gt(0)]
NegativeInt: type = Annotated[int, annotated_types.Lt(0)]
NonPositiveInt: type = Annotated[int, annotated_types.Le(0)]
NonNegativeInt: type = Annotated[int, annotated_types.Ge(0)]
StrictInt: type = Annotated[int, Strict()]
"An integer that must be validated in strict mode."

@_dataclasses.dataclass
class AllowInfNan(_fields.PydanticMetadata):
    allow_inf_nan: bool = True

    def __hash__(self) -> int:
        return hash(self.allow_inf_nan)

def confloat(*, strict: Optional[bool] = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, multiple_of: Optional[float] = None, allow_inf_nan: Optional[bool] = None) -> Any:
    return Annotated[float, Strict(strict) if strict is not None else None, annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None, AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None]

PositiveFloat: type = Annotated[float, annotated_types.Gt(0)]
NegativeFloat: type = Annotated[float, annotated_types.Lt(0)]
NonPositiveFloat: type = Annotated[float, annotated_types.Le(0)]
NonNegativeFloat: type = Annotated[float, annotated_types.Ge(0)]
StrictFloat: type = Annotated[float, Strict(True)]
FiniteFloat: type = Annotated[float, AllowInfNan(False)]
'A float that must be finite (not ``-inf``, ``inf``, or ``nan``).'

def conbytes(*, min_length: Optional[int] = None, max_length: Optional[int] = None, strict: Optional[bool] = None) -> Any:
    return Annotated[bytes, Strict(strict) if strict is not None else None, annotated_types.Len(min_length or 0, max_length)]

StrictBytes: type = Annotated[bytes, Strict()]
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

StrictStr: type = Annotated[str, Strict()]
'A string that must be validated in strict mode.'

HashableItemType = TypeVar('HashableItemType', bound=Hashable)

def conset(item_type: Any, *, min_length: Optional[int] = None, max_length: Optional[int] = None) -> Any:
    return Annotated[set[item_type], annotated_types.Len(min_length or 0, max_length)]

def confrozenset(item_type: Any, *, min_length: Optional[int] = None, max_length: Optional[int] = None) -> Any:
    return Annotated[frozenset[item_type], annotated_types.Len(min_length or 0, max_length)]

AnyItemType = TypeVar('AnyItemType')

def conlist(item_type: Any, *, min_length: Optional[int] = None, max_length: Optional[int] = None, unique_items: Optional[bool] = None) -> Any:
    if unique_items is not None:
        raise PydanticUserError('`unique_items` is removed, use `Set` instead(this feature is discussed in https://github.com/pydantic/pydantic-core/issues/296)', code='removed-kwargs')
    return Annotated[list[item_type], annotated_types.Len(min_length or 0, max_length)]

AnyType = TypeVar('AnyType')
if TYPE_CHECKING:
    ImportString: type = Annotated[AnyType, ...]
else:
    class ImportString:
        @classmethod
        def __class_getitem__(cls, item: Any) -> Any:
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
    return Annotated[Decimal, Strict(strict) if strict is not None else None, annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le), annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None, _fields.pydantic_general_metadata(max_digits=max_digits, decimal_places=decimal_places), AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None]

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class UuidVersion:
    uuid_version: int

    def __get_pydantic_json_schema__(self, core_schema_: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema_)
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

UUID1: type = Annotated[UUID, UuidVersion(1)]
UUID3: type = Annotated[UUID, UuidVersion(3)]
UUID4: type = Annotated[UUID, UuidVersion(4)]
UUID5: type = Annotated[UUID, UuidVersion(5)]

@_dataclasses.dataclass
class PathType:
    path_type: str

    def __get_pydantic_json_schema__(self, core_schema_: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema_)
        format_conversion: dict[str, str] = {'file': 'file-path', 'dir': 'directory-path'}
        field_schema.update(format=format_conversion.get(self.path_type, 'path'), type='string')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: Callable[[Any], Any]) -> Any:
        function_lookup: dict[str, Callable[[Any, Any], Any]] = {
            'file': cast(core_schema.WithInfoValidatorFunction, self.validate_file),
            'dir': cast(core_schema.WithInfoValidatorFunction, self.validate_directory),
            'new': cast(core_schema.WithInfoValidatorFunction, self.validate_new),
            'socket': cast(core_schema.WithInfoValidatorFunction, self.validate_socket)
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

FilePath: type = Annotated[Path, PathType('file')]
DirectoryPath: type = Annotated[Path, PathType('dir')]
NewPath: type = Annotated[Path, PathType('new')]
SocketPath: type = Annotated[Path, PathType('socket')]

if TYPE_CHECKING:
    Json: type = Annotated[AnyType, ...]
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

SecretType = TypeVar('SecretType')

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

def _serialize_secret(value: Any, info: Any) -> Any:
    if info.mode == 'json':
        return str(value)
    else:
        return value

class Secret(_SecretBase[SecretType]):
    def _display(self) -> str:
        return '**********' if self.get_secret_value() else ''

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        inner_type: Optional[Any] = None
        origin_type = get_origin(source)
        if origin_type is not None:
            inner_type = get_args(source)[0]
        else:
            bases = getattr(cls, '__orig_bases__', getattr(cls, '__bases__', []))
            for base in bases:
                if get_origin(base) is Secret:
                    inner_type = get_args(base)[0]
            if bases == [] or inner_type is None:
                raise TypeError(f"Can't get secret type from {cls.__name__}. Please use Secret[<type>], or subclass from Secret[<type>] instead.")
        inner_schema = handler.generate_schema(inner_type)
        def validate_secret_value(value: Any, handler: Any) -> Secret:
            if isinstance(value, Secret):
                value = value.get_secret_value()
            validated_inner = handler(value)
            return cls(validated_inner)
        return core_schema.json_or_python_schema(
            python_schema=core_schema.no_info_wrap_validator_function(validate_secret_value, inner_schema),
            json_schema=core_schema.no_info_after_validator_function(lambda x: cls(x), inner_schema),
            serialization=core_schema.plain_serializer_function_ser_schema(_serialize_secret, info_arg=True, when_used='always')
        )
    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.plain_serializer_function_ser_schema(_serialize_secret, info_arg=True, when_used='always')))

def _secret_display(value: Any) -> str:
    return '**********' if value else ''

def _serialize_secret_field(value: Any, info: Any) -> Any:
    if info.mode == 'json':
        return _secret_display(value.get_secret_value())
    else:
        return value

class _SecretField(_SecretBase[SecretType]):
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def get_json_schema(_core_schema: Any, handler: Any) -> Any:
            json_schema = handler(cls._inner_schema)
            _utils.update_not_none(json_schema, type='string', writeOnly=True, format='password')
            return json_schema
        json_schema = core_schema.no_info_after_validator_function(source, cls._inner_schema)
        def get_secret_schema(strict: bool) -> Any:
            return core_schema.json_or_python_schema(
                python_schema=core_schema.union_schema([core_schema.is_instance_schema(source), json_schema], custom_error_type=cls._error_kind, strict=strict),
                json_schema=json_schema,
                serialization=core_schema.plain_serializer_function_ser_schema(_serialize_secret_field, info_arg=True, when_used='always')
            )
        return core_schema.lax_or_strict_schema(
            lax_schema=get_secret_schema(strict=False),
            strict_schema=get_secret_schema(strict=True),
            metadata={'pydantic_js_functions': [get_json_schema]}
        )
    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.plain_serializer_function_ser_schema(_serialize_secret_field, info_arg=True, when_used='always')))

class SecretStr(_SecretField[str]):
    _inner_schema = core_schema.str_schema()
    _error_kind: str = 'string_type'

    def __len__(self) -> int:
        return len(self._secret_value)

    def _display(self) -> str:
        return _secret_display(self._secret_value)

class SecretBytes(_SecretField[bytes]):
    _inner_schema = core_schema.bytes_schema()
    _error_kind: str = 'bytes_type'

    def __len__(self) -> int:
        return len(self._secret_value)

    def _display(self) -> bytes:
        return _secret_display(self._secret_value).encode()

class PaymentCardBrand(str, Enum):
    amex = 'American Express'
    mastercard = 'Mastercard'
    visa = 'Visa'
    other = 'other'

    def __str__(self) -> str:
        return self.value

@deprecated('The `PaymentCardNumber` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_payment/#pydantic_extra_types.payment.PaymentCardNumber.', category=PydanticDeprecatedSince20)
class PaymentCardNumber(str):
    strip_whitespace: bool = True
    min_length: int = 12
    max_length: int = 19

    def __init__(self, card_number: str) -> None:
        self.validate_digits(card_number)
        card_number = self.validate_luhn_check_digit(card_number)
        self.bin = card_number[:6]
        self.last4 = card_number[-4:]
        self.brand = self.validate_brand(card_number)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.with_info_after_validator_function(
            cls.validate, core_schema.str_schema(min_length=cls.min_length, max_length=cls.max_length, strip_whitespace=cls.strip_whitespace)
        )

    @classmethod
    def validate(cls, input_value: str, /, _: Any) -> PaymentCardNumber:
        return cls(input_value)

    @property
    def masked(self) -> str:
        num_masked = len(self) - 10
        return f'{self.bin}{"*" * num_masked}{self.last4}'

    @classmethod
    def validate_digits(cls, card_number: str) -> None:
        if not card_number.isdigit():
            raise PydanticCustomError('payment_card_number_digits', 'Card number is not all digits')

    @classmethod
    def validate_luhn_check_digit(cls, card_number: str) -> str:
        sum_ = int(card_number[-1])
        length = len(card_number)
        parity = length % 2
        for i in range(length - 1):
            digit = int(card_number[i])
            if i % 2 == parity:
                digit *= 2
            if digit > 9:
                digit -= 9
            sum_ += digit
        valid = sum_ % 10 == 0
        if not valid:
            raise PydanticCustomError('payment_card_number_luhn', 'Card number is not luhn valid')
        return card_number

    @staticmethod
    def validate_brand(card_number: str) -> PaymentCardBrand:
        if card_number[0] == '4':
            brand: PaymentCardBrand = PaymentCardBrand.visa
        elif 51 <= int(card_number[:2]) <= 55:
            brand = PaymentCardBrand.mastercard
        elif card_number[:2] in {'34', '37'}:
            brand = PaymentCardBrand.amex
        else:
            brand = PaymentCardBrand.other
        required_length: Optional[Union[int, str]] = None
        if brand in PaymentCardBrand.mastercard:
            required_length = 16
            valid = len(card_number) == required_length
        elif brand == PaymentCardBrand.visa:
            required_length = '13, 16 or 19'
            valid = len(card_number) in {13, 16, 19}
        elif brand == PaymentCardBrand.amex:
            required_length = 15
            valid = len(card_number) == required_length
        else:
            valid = True
        if not valid:
            raise PydanticCustomError('payment_card_number_brand', 'Length for a {brand} card must be {required_length}', {'brand': brand, 'required_length': required_length})
        return brand

class ByteSize(int):
    byte_sizes: ClassVar[dict[str, float]] = {'b': 1, 'kb': 10 ** 3, 'mb': 10 ** 6, 'gb': 10 ** 9, 'tb': 10 ** 12, 'pb': 10 ** 15, 'eb': 10 ** 18, 'kib': 2 ** 10, 'mib': 2 ** 20, 'gib': 2 ** 30, 'tib': 2 ** 40, 'pib': 2 ** 50, 'eib': 2 ** 60, 'bit': 1 / 8, 'kbit': 10 ** 3 / 8, 'mbit': 10 ** 6 / 8, 'gbit': 10 ** 9 / 8, 'tbit': 10 ** 12 / 8, 'pbit': 10 ** 15 / 8, 'ebit': 10 ** 18 / 8, 'kibit': 2 ** 10 / 8, 'mibit': 2 ** 20 / 8, 'gibit': 2 ** 30 / 8, 'tibit': 2 ** 40 / 8, 'pibit': 2 ** 50 / 8, 'eibit': 2 ** 60 / 8}
    byte_sizes.update({k.lower()[0]: v for k, v in byte_sizes.items() if 'i' not in k})
    byte_string_pattern: ClassVar[str] = '^\\s*(\\d*\\.?\\d+)\\s*(\\w+)?'
    byte_string_re: ClassVar[Pattern[str]] = re.compile(byte_string_pattern, re.IGNORECASE)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
        return core_schema.with_info_after_validator_function(
            function=cls._validate,
            schema=core_schema.union_schema(
                [core_schema.str_schema(pattern=cls.byte_string_pattern), core_schema.int_schema(ge=0)],
                custom_error_type='byte_size',
                custom_error_message='could not parse value and unit from byte string'
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(int, return_schema=core_schema.int_schema(ge=0))
        )

    @classmethod
    def _validate(cls, input_value: Any, _: Any) -> ByteSize:
        try:
            return cls(int(input_value))
        except ValueError:
            pass
        str_match = cls.byte_string_re.match(str(input_value))
        if str_match is None:
            raise PydanticCustomError('byte_size', 'could not parse value and unit from byte string')
        scalar, unit = str_match.groups()
        if unit is None:
            unit = 'b'
        try:
            unit_mult = cls.byte_sizes[unit.lower()]
        except KeyError:
            raise PydanticCustomError('byte_size_unit', 'could not interpret byte unit: {unit}', {'unit': unit})
        return cls(int(float(scalar) * unit_mult))

    def human_readable(self, decimal: bool = False, separator: str = '') -> str:
        if decimal:
            divisor = 1000
            units = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
            final_unit = 'EB'
        else:
            divisor = 1024
            units = ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB')
            final_unit = 'EiB'
        num: float = float(self)
        for unit in units:
            if abs(num) < divisor:
                if unit == 'B':
                    return f'{num:0.0f}{separator}{unit}'
                else:
                    return f'{num:0.1f}{separator}{unit}'
            num /= divisor
        return f'{num:0.1f}{separator}{final_unit}'

    def to(self, unit: str) -> float:
        try:
            unit_div = self.byte_sizes[unit.lower()]
        except KeyError:
            raise PydanticCustomError('byte_size_unit', 'Could not interpret byte unit: {unit}', {'unit': unit})
        return self / unit_div

def _check_annotated_type(annotated_type: Any, expected_type: str, annotation: str) -> None:
    if annotated_type != expected_type:
        raise PydanticUserError(f"'{annotation}' cannot annotate '{annotated_type}'.", code='invalid-annotated-type')

if TYPE_CHECKING:
    PastDate: type = Annotated[date, ...]
    FutureDate: type = Annotated[date, ...]
else:
    class PastDate:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            if cls is source:
                return core_schema.date_schema(now_op='past')
            else:
                schema = handler(source)
                _check_annotated_type(schema['type'], 'date', cls.__name__)
                schema['now_op'] = 'past'
                return schema

        def __repr__(self) -> str:
            return 'PastDate'

    class FutureDate:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any]) -> Any:
            if cls is source:
                return core_schema.date_schema(now_op='future')
            else:
                schema = handler(source)
                _check_annotated_type(schema['type'], 'date', cls.__name__)
                schema['now_op'] = 'future'
                return schema

        def __repr__(self) -> str:
            return 'FutureDate'

def condate(*, strict: Optional[bool] = None, gt: Optional[date] = None, ge: Optional[date] = None, lt: Optional[date] = None, le: Optional[date] = None) -> Any:
    return Annotated[date, Strict(strict) if strict is not None else None, annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le)]

if TYPE_CHECKING:
    AwareDatetime: type = Annotated[datetime, ...]
    NaiveDatetime: type = Annotated[datetime, ...]
    PastDatetime: type = Annotated[datetime, ...]
    FutureDatetime: type = Annotated[datetime, ...]
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
    def get_json_format(cls) -> Any:
        ...

class Base64Encoder(EncoderProtocol):
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        try:
            return base64.b64decode(data)
        except ValueError as e:
            raise PydanticCustomError('base64_decode', f"Base64 decoding error: '{e}'", {'error': str(e)})

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
            raise PydanticCustomError('base64_decode', f"Base64 decoding error: '{e}'", {'error': str(e)})

    @classmethod
    def encode(cls, value: bytes) -> bytes:
        return base64.urlsafe_b64encode(value)

    @classmethod
    def get_json_format(cls) -> str:
        return 'base64url'

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedBytes:
    encoder: EncoderProtocol

    def __get_pydantic_json_schema__(self, core_schema_: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema_)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: Callable[[Any], Any]) -> Any:
        schema = handler(source)
        _check_annotated_type(schema['type'], 'bytes', self.__class__.__name__)
        return core_schema.with_info_after_validator_function(
            function=self.decode,
            schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode)
        )

    def decode(self, data: bytes, _: Any) -> Any:
        return self.encoder.decode(data)

    def encode(self, value: bytes) -> Any:
        return self.encoder.encode(value)

    def __hash__(self) -> int:
        return hash(self.encoder)

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedStr:
    encoder: EncoderProtocol

    def __get_pydantic_json_schema__(self, core_schema_: Any, handler: Callable[[Any], Any]) -> Any:
        field_schema = handler(core_schema_)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: Callable[[Any], Any]) -> Any:
        schema = handler(source)
        _check_annotated_type(schema['type'], 'str', self.__class__.__name__)
        return core_schema.with_info_after_validator_function(
            function=self.decode_str,
            schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode_str)
        )

    def decode_str(self, data: str, _: Any) -> str:
        return self.encoder.decode(data.encode()).decode()

    def encode_str(self, value: str) -> str:
        return self.encoder.encode(value.encode()).decode()

    def __hash__(self) -> int:
        return hash(self.encoder)

Base64Bytes: type = Annotated[bytes, EncodedBytes(encoder=Base64Encoder)]
Base64Str: type = Annotated[str, EncodedStr(encoder=Base64Encoder)]
Base64UrlBytes: type = Annotated[bytes, EncodedBytes(encoder=Base64UrlEncoder)]
Base64UrlStr: type = Annotated[str, EncodedStr(encoder=Base64UrlEncoder)]
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
    custom_error_context: Optional[dict[str, Any]] = None

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
        custom_error_type = self.custom_error_type if self.custom_error_type is not None else original_schema.get('custom_error_type')
        custom_error_message = self.custom_error_message if self.custom_error_message is not None else original_schema.get('custom_error_message')
        custom_error_context = self.custom_error_context if self.custom_error_context is not None else original_schema.get('custom_error_context')
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

_JSON_TYPES = {int, float, str, bool, list, dict, type(None)}

def _get_type_name(x: Any) -> str:
    type_ = type(x)
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
    JsonValue: Any = Union[list['JsonValue'], dict[str, 'JsonValue'], str, bool, int, float, None]
else:
    JsonValue: Any = Annotated[
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

OnErrorOmit: type = Annotated[T, _OnErrorOmit]
'''
When used as an item in a list, the key type in a dict, optional values of a TypedDict, etc.
this annotation omits the item from the iteration if there is any error validating it.
'''

@_dataclasses.dataclass
class FailFast(_fields.PydanticMetadata, BaseMetadata):
    fail_fast: bool = True
    """A `FailFast` annotation can be used to specify that validation should stop at the first error."""
