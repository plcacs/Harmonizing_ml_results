#!/usr/bin/env python3
"""The types module contains custom types used by pydantic."""

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
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
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

T = TypeVar('T')
HashableItemType = TypeVar('HashableItemType', bound=Hashable)
AnyItemType = TypeVar('AnyItemType')
SecretType = TypeVar('SecretType')

@_dataclasses.dataclass
class Strict(_fields.PydanticMetadata, BaseMetadata):
    """!!! abstract "Usage Documentation"
        [Strict Mode with `Annotated` `Strict`](../concepts/strict_mode.md#strict-mode-with-annotated-strict)

    A field metadata class to indicate that a field should be validated in strict mode.
    Use this class as an annotation via [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated), as seen below.

    Attributes:
        strict: Whether to validate the field in strict mode.
    """
    strict: bool = True

    def __hash__(self) -> int:
        return hash(self.strict)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BOOLEAN TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
StrictBool: type = Annotated[bool, Strict()]
"""A boolean that must be either ``True`` or ``False``."""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTEGER TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def conint(
    *,
    strict: bool | None = None,
    gt: int | None = None,
    ge: int | None = None,
    lt: int | None = None,
    le: int | None = None,
    multiple_of: int | None = None,
) -> type[int]:
    """
    A wrapper around `int` that allows for additional constraints.
    """
    return Annotated[  # type: ignore[return-type]
        int,
        Strict(strict) if strict is not None else None,
        annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le),
        annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None,
    ]


PositiveInt: type = Annotated[int, annotated_types.Gt(0)]
NegativeInt: type = Annotated[int, annotated_types.Lt(0)]
NonPositiveInt: type = Annotated[int, annotated_types.Le(0)]
NonNegativeInt: type = Annotated[int, annotated_types.Ge(0)]
StrictInt: type = Annotated[int, Strict()]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FLOAT TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@_dataclasses.dataclass
class AllowInfNan(_fields.PydanticMetadata):
    """A field metadata class to indicate that a field should allow `-inf`, `inf`, and `nan`."""
    allow_inf_nan: bool = True

    def __hash__(self) -> int:
        return hash(self.allow_inf_nan)


def confloat(
    *,
    strict: bool | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    multiple_of: float | None = None,
    allow_inf_nan: bool | None = None,
) -> type[float]:
    """
    A wrapper around `float` that allows for additional constraints.
    """
    return Annotated[  # type: ignore[return-type]
        float,
        Strict(strict) if strict is not None else None,
        annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le),
        annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None,
        AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None,
    ]


PositiveFloat: type = Annotated[float, annotated_types.Gt(0)]
NegativeFloat: type = Annotated[float, annotated_types.Lt(0)]
NonPositiveFloat: type = Annotated[float, annotated_types.Le(0)]
NonNegativeFloat: type = Annotated[float, annotated_types.Ge(0)]
StrictFloat: type = Annotated[float, Strict(True)]
FiniteFloat: type = Annotated[float, AllowInfNan(False)]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BYTES TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def conbytes(
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    strict: bool | None = None,
) -> type[bytes]:
    """A wrapper around `bytes` that allows for additional constraints."""
    return Annotated[  # type: ignore[return-type]
        bytes,
        Strict(strict) if strict is not None else None,
        annotated_types.Len(min_length or 0, max_length),
    ]


StrictBytes: type = Annotated[bytes, Strict()]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ STRING TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@_dataclasses.dataclass(frozen=True)
class StringConstraints(annotated_types.GroupedMetadata):
    """A field metadata class to apply constraints to `str` types."""
    strip_whitespace: bool | None = None
    to_upper: bool | None = None
    to_lower: bool | None = None
    strict: bool | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | Pattern[str] | None = None

    def __iter__(self) -> Iterator[BaseMetadata]:
        if self.min_length is not None:
            yield MinLen(self.min_length)
        if self.max_length is not None:
            yield MaxLen(self.max_length)
        if self.strict is not None:
            yield Strict(self.strict)
        if (
            self.strip_whitespace is not None
            or self.pattern is not None
            or self.to_lower is not None
            or self.to_upper is not None
        ):
            yield _fields.pydantic_general_metadata(
                strip_whitespace=self.strip_whitespace,
                to_upper=self.to_upper,
                to_lower=self.to_lower,
                pattern=self.pattern,
            )


def constr(
    *,
    strip_whitespace: bool | None = None,
    to_upper: bool | None = None,
    to_lower: bool | None = None,
    strict: bool | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | Pattern[str] | None = None,
) -> type[str]:
    """
    A wrapper around `str` that allows for additional constraints.
    """
    return Annotated[  # type: ignore[return-type]
        str,
        StringConstraints(
            strip_whitespace=strip_whitespace,
            to_upper=to_upper,
            to_lower=to_lower,
            strict=strict,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
        ),
    ]


StrictStr: type = Annotated[str, Strict()]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ COLLECTION TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def conset(
    item_type: type[HashableItemType], *, min_length: int | None = None, max_length: int | None = None
) -> type[set[HashableItemType]]:
    """A wrapper around `set` that allows for additional constraints."""
    return Annotated[set[item_type], annotated_types.Len(min_length or 0, max_length)]  # type: ignore[return-type]


def confrozenset(
    item_type: type[HashableItemType], *, min_length: int | None = None, max_length: int | None = None
) -> type[frozenset[HashableItemType]]:
    """A wrapper around `frozenset` that allows for additional constraints."""
    return Annotated[frozenset[item_type], annotated_types.Len(min_length or 0, max_length)]  # type: ignore[return-type]


def conlist(
    item_type: type[AnyItemType],
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    unique_items: bool | None = None,
) -> type[list[AnyItemType]]:
    """A wrapper around `list` that adds validation."""
    if unique_items is not None:
        raise PydanticUserError(
            (
                '`unique_items` is removed, use `Set` instead'
                '(this feature is discussed in https://github.com/pydantic/pydantic-core/issues/296)'
            ),
            code='removed-kwargs',
        )
    return Annotated[list[item_type], annotated_types.Len(min_length or 0, max_length)]  # type: ignore[return-type]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORT STRING TYPE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if False:  # TYPE_CHECKING placeholder
    ImportString = Annotated[Any, ...]
else:
    class ImportString:
        """A type that can be used to import a Python object from a string."""
        @classmethod
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, cls()]

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: type[Any], handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            serializer = core_schema.plain_serializer_function_ser_schema(cls._serialize, when_used='json')
            if cls is source:
                return core_schema.no_info_plain_validator_function(
                    function=_validators.import_string, serialization=serializer
                )
            else:
                return core_schema.no_info_before_validator_function(
                    function=_validators.import_string, schema=handler(source), serialization=serializer
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
                return v

        def __repr__(self) -> str:
            return 'ImportString'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DECIMAL TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def condecimal(
    *,
    strict: bool | None = None,
    gt: int | Decimal | None = None,
    ge: int | Decimal | None = None,
    lt: int | Decimal | None = None,
    le: int | Decimal | None = None,
    multiple_of: int | Decimal | None = None,
    max_digits: int | None = None,
    decimal_places: int | None = None,
    allow_inf_nan: bool | None = None,
) -> type[Decimal]:
    """
    A wrapper around Decimal that adds validation.
    """
    return Annotated[  # type: ignore[return-type]
        Decimal,
        Strict(strict) if strict is not None else None,
        annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le),
        annotated_types.MultipleOf(multiple_of) if multiple_of is not None else None,
        _fields.pydantic_general_metadata(max_digits=max_digits, decimal_places=decimal_places),
        AllowInfNan(allow_inf_nan) if allow_inf_nan is not None else None,
    ]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UUID TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class UuidVersion:
    """A field metadata class to indicate a UUID version."""
    uuid_version: Literal[1, 3, 4, 5]

    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        field_schema: dict[str, Any] = handler(core_schema)
        field_schema.pop('anyOf', None)
        field_schema.update(type='string', format=f'uuid{self.uuid_version}')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        if isinstance(self, source):
            return core_schema.uuid_schema(version=self.uuid_version)
        else:
            schema = handler(source)
            _check_annotated_type(schema['type'], 'uuid', self.__class__.__name__)
            schema['version'] = self.uuid_version  # type: ignore
            return schema

    def __hash__(self) -> int:
        return hash(type(self.uuid_version))


UUID1: type = Annotated[UUID, UuidVersion(1)]
UUID3: type = Annotated[UUID, UuidVersion(3)]
UUID4: type = Annotated[UUID, UuidVersion(4)]
UUID5: type = Annotated[UUID, UuidVersion(5)]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PATH TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@_dataclasses.dataclass
class PathType:
    path_type: Literal['file', 'dir', 'new', 'socket']

    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        field_schema: dict[str, Any] = handler(core_schema)
        format_conversion: dict[str, str] = {'file': 'file-path', 'dir': 'directory-path'}
        field_schema.update(format=format_conversion.get(self.path_type, 'path'), type='string')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        function_lookup: dict[str, Callable[[Path, CoreSchema.ValidationInfo], Path]] = {
            'file': cast(Callable[[Path, CoreSchema.ValidationInfo], Path], self.validate_file),
            'dir': cast(Callable[[Path, CoreSchema.ValidationInfo], Path], self.validate_directory),
            'new': cast(Callable[[Path, CoreSchema.ValidationInfo], Path], self.validate_new),
            'socket': cast(Callable[[Path, CoreSchema.ValidationInfo], Path], self.validate_socket),
        }
        return core_schema.with_info_after_validator_function(
            function_lookup[self.path_type],
            handler(source),
        )

    @staticmethod
    def validate_file(path: Path, _: CoreSchema.ValidationInfo) -> Path:
        if path.is_file():
            return path
        else:
            raise PydanticCustomError('path_not_file', 'Path does not point to a file')

    @staticmethod
    def validate_socket(path: Path, _: CoreSchema.ValidationInfo) -> Path:
        if path.is_socket():
            return path
        else:
            raise PydanticCustomError('path_not_socket', 'Path does not point to a socket')

    @staticmethod
    def validate_directory(path: Path, _: CoreSchema.ValidationInfo) -> Path:
        if path.is_dir():
            return path
        else:
            raise PydanticCustomError('path_not_directory', 'Path does not point to a directory')

    @staticmethod
    def validate_new(path: Path, _: CoreSchema.ValidationInfo) -> Path:
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ JSON TYPE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if False:  # TYPE_CHECKING placeholder
    Json = Annotated[Any, ...]
else:
    class Json:
        """A special type wrapper which loads JSON before parsing."""
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SECRET TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class _SecretBase(Generic[SecretType]):
    def __init__(self, secret_value: SecretType) -> None:
        self._secret_value: SecretType = secret_value

    def get_secret_value(self) -> SecretType:
        """Get the secret value."""
        return self._secret_value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.get_secret_value() == other.get_secret_value()

    def __hash__(self) -> int:
        return hash(self.get_secret_value())

    def __str__(self) -> str:
        return str(self._display())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._display()!r})'

    def _display(self) -> str | bytes:
        raise NotImplementedError


def _serialize_secret(value: Secret[SecretType], info: CoreSchema.SerializationInfo) -> str | Secret[SecretType]:
    if info.mode == 'json':
        return str(value)
    else:
        return value


class Secret(_SecretBase[SecretType]):
    """A generic base class used for defining a field with sensitive information."""
    def _display(self) -> str | bytes:
        return '**********' if self.get_secret_value() else ''

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> CoreSchema:
        inner_type: Any = None
        origin_type = get_origin(source)
        if origin_type is not None:
            inner_type = get_args(source)[0]
        else:
            bases = getattr(cls, '__orig_bases__', getattr(cls, '__bases__', []))
            for base in bases:
                if get_origin(base) is Secret:
                    inner_type = get_args(base)[0]
            if bases == [] or inner_type is None:
                raise TypeError(
                    f"Can't get secret type from {cls.__name__}. "
                    'Please use Secret[<type>], or subclass from Secret[<type>] instead.'
                )
        inner_schema = handler.generate_schema(inner_type)  # type: ignore
        def validate_secret_value(value: Any, handler: Callable[[Any], Any]) -> Secret[SecretType]:
            if isinstance(value, Secret):
                value = value.get_secret_value()
            validated_inner = handler(value)
            return cls(validated_inner)
        return core_schema.json_or_python_schema(
            python_schema=core_schema.no_info_wrap_validator_function(
                validate_secret_value,
                inner_schema,
            ),
            json_schema=core_schema.no_info_after_validator_function(lambda x: cls(x), inner_schema),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_secret,
                info_arg=True,
                when_used='always',
            ),
        )
    __pydantic_serializer__ = SchemaSerializer(
        core_schema.any_schema(
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_secret,
                info_arg=True,
                when_used='always',
            )
        )
    )


def _secret_display(value: SecretType) -> str:
    return '**********' if value else ''


def _serialize_secret_field(
    value: _SecretField[SecretType], info: CoreSchema.SerializationInfo
) -> str | _SecretField[SecretType]:
    if info.mode == 'json':
        return _secret_display(value.get_secret_value())
    else:
        return value


class _SecretField(_SecretBase[SecretType]):
    _inner_schema: ClassVar[CoreSchema]
    _error_kind: ClassVar[str]

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> CoreSchema:
        def get_json_schema(_core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
            json_schema: dict[str, Any] = handler(cls._inner_schema)
            _utils.update_not_none(
                json_schema,
                type='string',
                writeOnly=True,
                format='password',
            )
            return json_schema
        json_schema = core_schema.no_info_after_validator_function(
            source,  # construct the type
            cls._inner_schema,
        )
        def get_secret_schema(strict: bool) -> CoreSchema:
            return core_schema.json_or_python_schema(
                python_schema=core_schema.union_schema(
                    [
                        core_schema.is_instance_schema(source),
                        json_schema,
                    ],
                    custom_error_type=cls._error_kind,
                    strict=strict,
                ),
                json_schema=json_schema,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    _serialize_secret_field,
                    info_arg=True,
                    when_used='always',
                ),
            )
        return core_schema.lax_or_strict_schema(
            lax_schema=get_secret_schema(strict=False),
            strict_schema=get_secret_schema(strict=True),
            metadata={'pydantic_js_functions': [get_json_schema]},
        )
    __pydantic_serializer__ = SchemaSerializer(
        core_schema.any_schema(
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_secret_field,
                info_arg=True,
                when_used='always',
            )
        )
    )


class SecretStr(_SecretField[str]):
    _inner_schema: ClassVar[CoreSchema] = core_schema.str_schema()
    _error_kind: ClassVar[str] = 'string_type'

    def __len__(self) -> int:
        return len(self._secret_value)

    def _display(self) -> str:
        return _secret_display(self._secret_value)


class SecretBytes(_SecretField[bytes]):
    _inner_schema: ClassVar[CoreSchema] = core_schema.bytes_schema()
    _error_kind: ClassVar[str] = 'bytes_type'

    def __len__(self) -> int:
        return len(self._secret_value)

    def _display(self) -> bytes:
        return _secret_display(self._secret_value).encode()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PAYMENT CARD TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class PaymentCardBrand(str, Enum):
    amex = 'American Express'
    mastercard = 'Mastercard'
    visa = 'Visa'
    other = 'other'

    def __str__(self) -> str:
        return self.value


@deprecated(
    'The `PaymentCardNumber` class is deprecated, use `pydantic_extra_types` instead. '
    'See https://docs.pydantic.dev/latest/api/pydantic_extra_types_payment/#pydantic_extra_types.payment.PaymentCardNumber.',
    category=PydanticDeprecatedSince20,
)
class PaymentCardNumber(str):
    strip_whitespace: ClassVar[bool] = True
    min_length: ClassVar[int] = 12
    max_length: ClassVar[int] = 19
    bin: str
    last4: str
    brand: PaymentCardBrand

    def __init__(self, card_number: str) -> None:
        self.validate_digits(card_number)
        card_number = self.validate_luhn_check_digit(card_number)
        self.bin = card_number[:6]
        self.last4 = card_number[-4:]
        self.brand = self.validate_brand(card_number)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.with_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(
                min_length=cls.min_length, max_length=cls.max_length, strip_whitespace=cls.strip_whitespace
            ),
        )

    @classmethod
    def validate(cls, input_value: str, /, _: CoreSchema.ValidationInfo) -> PaymentCardNumber:
        return cls(input_value)

    @property
    def masked(self) -> str:
        num_masked: int = len(self) - 10
        return f'{self.bin}{"*" * num_masked}{self.last4}'

    @classmethod
    def validate_digits(cls, card_number: str) -> None:
        if not card_number.isdigit():
            raise PydanticCustomError('payment_card_number_digits', 'Card number is not all digits')

    @classmethod
    def validate_luhn_check_digit(cls, card_number: str) -> str:
        sum_: int = int(card_number[-1])
        length: int = len(card_number)
        parity: int = length % 2
        for i in range(length - 1):
            digit: int = int(card_number[i])
            if i % 2 == parity:
                digit *= 2
            if digit > 9:
                digit -= 9
            sum_ += digit
        valid: bool = (sum_ % 10 == 0)
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

        required_length: int | str | None = None
        if brand == PaymentCardBrand.mastercard:
            required_length = 16
            valid: bool = (len(card_number) == required_length)
        elif brand == PaymentCardBrand.visa:
            required_length = '13, 16 or 19'
            valid = len(card_number) in {13, 16, 19}
        elif brand == PaymentCardBrand.amex:
            required_length = 15
            valid = (len(card_number) == required_length)
        else:
            valid = True

        if not valid:
            raise PydanticCustomError(
                'payment_card_number_brand',
                'Length for a {brand} card must be {required_length}',
                {'brand': brand, 'required_length': required_length},
            )
        return brand


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BYTE SIZE TYPE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ByteSize(int):
    byte_sizes: ClassVar[dict[str, float]] = {
        'b': 1,
        'kb': 10**3,
        'mb': 10**6,
        'gb': 10**9,
        'tb': 10**12,
        'pb': 10**15,
        'eb': 10**18,
        'kib': 2**10,
        'mib': 2**20,
        'gib': 2**30,
        'tib': 2**40,
        'pib': 2**50,
        'eib': 2**60,
        'bit': 1 / 8,
        'kbit': 10**3 / 8,
        'mbit': 10**6 / 8,
        'gbit': 10**9 / 8,
        'tbit': 10**12 / 8,
        'pbit': 10**15 / 8,
        'ebit': 10**18 / 8,
        'kibit': 2**10 / 8,
        'mibit': 2**20 / 8,
        'gibit': 2**30 / 8,
        'tibit': 2**40 / 8,
        'pibit': 2**50 / 8,
        'eibit': 2**60 / 8,
    }
    byte_sizes.update({k.lower()[0]: v for k, v in byte_sizes.items() if 'i' not in k})
    byte_string_pattern: ClassVar[str] = r'^\s*(\d*\.?\d+)\s*(\w+)?'
    byte_string_re: ClassVar[re.Pattern] = re.compile(byte_string_pattern, re.IGNORECASE)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.with_info_after_validator_function(
            function=cls._validate,
            schema=core_schema.union_schema(
                [
                    core_schema.str_schema(pattern=cls.byte_string_pattern),
                    core_schema.int_schema(ge=0),
                ],
                custom_error_type='byte_size',
                custom_error_message='could not parse value and unit from byte string',
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                int, return_schema=core_schema.int_schema(ge=0)
            ),
        )

    @classmethod
    def _validate(cls, input_value: Any, _: CoreSchema.ValidationInfo) -> ByteSize:
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
            unit_mult: float = cls.byte_sizes[unit.lower()]
        except KeyError:
            raise PydanticCustomError('byte_size_unit', 'could not interpret byte unit: {unit}', {'unit': unit})
        return cls(int(float(scalar) * unit_mult))

    def human_readable(self, decimal: bool = False, separator: str = '') -> str:
        if decimal:
            divisor: float = 1000
            units: tuple[str, ...] = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
            final_unit: str = 'EB'
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
            unit_div: float = self.byte_sizes[unit.lower()]
        except KeyError:
            raise PydanticCustomError('byte_size_unit', 'Could not interpret byte unit: {unit}', {'unit': unit})
        return self / unit_div


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATE TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _check_annotated_type(annotated_type: str, expected_type: str, annotation: str) -> None:
    if annotated_type != expected_type:
        raise PydanticUserError(f"'{annotation}' cannot annotate '{annotated_type}'.", code='invalid-annotated-type')


if False:  # TYPE_CHECKING placeholder
    PastDate = Annotated[date, ...]
    FutureDate = Annotated[date, ...]
else:
    class PastDate:
        """A date in the past."""
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: type[Any], handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            if cls is source:
                return core_schema.date_schema(now_op='past')
            else:
                schema: dict[str, Any] = handler(source)
                _check_annotated_type(schema['type'], 'date', cls.__name__)
                schema['now_op'] = 'past'
                return schema

        def __repr__(self) -> str:
            return 'PastDate'

    class FutureDate:
        """A date in the future."""
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: type[Any], handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            if cls is source:
                return core_schema.date_schema(now_op='future')
            else:
                schema: dict[str, Any] = handler(source)
                _check_annotated_type(schema['type'], 'date', cls.__name__)
                schema['now_op'] = 'future'
                return schema

        def __repr__(self) -> str:
            return 'FutureDate'


def condate(
    *,
    strict: bool | None = None,
    gt: date | None = None,
    ge: date | None = None,
    lt: date | None = None,
    le: date | None = None,
) -> type[date]:
    """A wrapper for date that adds constraints."""
    return Annotated[  # type: ignore[return-type]
        date,
        Strict(strict) if strict is not None else None,
        annotated_types.Interval(gt=gt, ge=ge, lt=lt, le=le),
    ]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATETIME TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if False:  # TYPE_CHECKING placeholder
    AwareDatetime = Annotated[datetime, ...]
    NaiveDatetime = Annotated[datetime, ...]
    PastDatetime = Annotated[datetime, ...]
    FutureDatetime = Annotated[datetime, ...]
else:
    class AwareDatetime:
        """A datetime that requires timezone info."""
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: type[Any], handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            if cls is source:
                return core_schema.datetime_schema(tz_constraint='aware')
            else:
                schema: dict[str, Any] = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['tz_constraint'] = 'aware'
                return schema

        def __repr__(self) -> str:
            return 'AwareDatetime'

    class NaiveDatetime:
        """A datetime that doesn't require timezone info."""
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: type[Any], handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            if cls is source:
                return core_schema.datetime_schema(tz_constraint='naive')
            else:
                schema: dict[str, Any] = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['tz_constraint'] = 'naive'
                return schema

        def __repr__(self) -> str:
            return 'NaiveDatetime'

    class PastDatetime:
        """A datetime that must be in the past."""
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: type[Any], handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            if cls is source:
                return core_schema.datetime_schema(now_op='past')
            else:
                schema: dict[str, Any] = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['now_op'] = 'past'
                return schema

        def __repr__(self) -> str:
            return 'PastDatetime'

    class FutureDatetime:
        """A datetime that must be in the future."""
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: type[Any], handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            if cls is source:
                return core_schema.datetime_schema(now_op='future')
            else:
                schema: dict[str, Any] = handler(source)
                _check_annotated_type(schema['type'], 'datetime', cls.__name__)
                schema['now_op'] = 'future'
                return schema

        def __repr__(self) -> str:
            return 'FutureDatetime'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Encoded TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class EncoderProtocol(Protocol):
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        ...
    @classmethod
    def encode(cls, value: bytes) -> bytes:
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
    def get_json_format(cls) -> Literal['base64']:
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
    def get_json_format(cls) -> Literal['base64url']:
        return 'base64url'


@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedBytes:
    encoder: type[EncoderProtocol]

    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        field_schema: dict[str, Any] = handler(core_schema)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: type[Any], handler: GetCoreSchemaHandler) -> CoreSchema:
        schema: dict[str, Any] = handler(source)
        _check_annotated_type(schema['type'], 'bytes', self.__class__.__name__)
        return core_schema.with_info_after_validator_function(
            function=self.decode,
            schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode),
        )

    def decode(self, data: bytes, _: CoreSchema.ValidationInfo) -> bytes:
        return self.encoder.decode(data)

    def encode(self, value: bytes) -> bytes:
        return self.encoder.encode(value)

    def __hash__(self) -> int:
        return hash(self.encoder)


@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedStr:
    encoder: type[EncoderProtocol]

    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        field_schema: dict[str, Any] = handler(core_schema)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: type[Any], handler: GetCoreSchemaHandler) -> CoreSchema:
        schema: dict[str, Any] = handler(source)
        _check_annotated_type(schema['type'], 'str', self.__class__.__name__)
        return core_schema.with_info_after_validator_function(
            function=self.decode_str,
            schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode_str),
        )

    def decode_str(self, data: str, _: CoreSchema.ValidationInfo) -> str:
        return self.encoder.decode(data.encode()).decode()

    def encode_str(self, value: str) -> str:
        return self.encoder.encode(value.encode()).decode()

    def __hash__(self) -> int:
        return hash(self.encoder)


Base64Bytes: type = Annotated[bytes, EncodedBytes(encoder=Base64Encoder)]
Base64Str: type = Annotated[str, EncodedStr(encoder=Base64Encoder)]
Base64UrlBytes: type = Annotated[bytes, EncodedBytes(encoder=Base64UrlEncoder)]
Base64UrlStr: type = Annotated[str, EncodedStr(encoder=Base64UrlEncoder)]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ JsonValue TYPE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _get_type_name(x: Any) -> str:
    _JSON_TYPES: set[type] = {int, float, str, bool, list, dict, type(None)}
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
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        python_schema: CoreSchema = handler(source_type)
        return core_schema.json_or_python_schema(json_schema=core_schema.any_schema(), python_schema=python_schema)


if False:  # TYPE_CHECKING placeholder
    JsonValue: TypeAlias = Union[
        list['JsonValue'],
        dict[str, 'JsonValue'],
        str,
        bool,
        int,
        float,
        None,
    ]
else:
    JsonValue: TypeAlias = Annotated[
        Union[
            Annotated[list['JsonValue'], Tag('list')],
            Annotated[dict[str, 'JsonValue'], Tag('dict')],
            Annotated[str, Tag('str')],
            Annotated[bool, Tag('bool')],
            Annotated[int, Tag('int')],
            Annotated[float, Tag('float')],
            Annotated[None, Tag('NoneType')],
        ],
        Discriminator(
            _get_type_name,
            custom_error_type='invalid-json-value',
            custom_error_message='input was not a valid JSON value',
        ),
        _AllowAnyJson,
    ]


class _OnErrorOmit:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.with_default_schema(schema=handler(source_type), on_error='omit')


OnErrorOmit: type = Annotated[T, _OnErrorOmit]
"""
When used in an iterable this annotation omits items that fail validation.
"""


@_dataclasses.dataclass
class FailFast(_fields.PydanticMetadata, BaseMetadata):
    fail_fast: bool = True
    """An annotation indicating that validation should stop at the first error."""
    
# End of module.
