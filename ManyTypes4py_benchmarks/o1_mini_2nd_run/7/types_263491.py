import abc
import math
import re
import warnings
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import new_class
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from uuid import UUID
from weakref import WeakSet
from pydantic.v1 import errors
from pydantic.v1.datetime_parse import parse_date
from pydantic.v1.utils import import_string, update_not_none
from pydantic.v1.validators import (
    bytes_validator,
    constr_length_validator,
    constr_lower,
    constr_strip_whitespace,
    constr_upper,
    decimal_validator,
    float_finite_validator,
    float_validator,
    frozenset_validator,
    int_validator,
    list_validator,
    number_multiple_validator,
    number_size_validator,
    path_exists_validator,
    path_validator,
    set_validator,
    str_validator,
    strict_bytes_validator,
    strict_float_validator,
    strict_int_validator,
    strict_str_validator,
)

__all__ = [
    'NoneStr',
    'NoneBytes',
    'StrBytes',
    'NoneStrBytes',
    'StrictStr',
    'ConstrainedBytes',
    'conbytes',
    'ConstrainedList',
    'conlist',
    'ConstrainedSet',
    'conset',
    'ConstrainedFrozenSet',
    'confrozenset',
    'ConstrainedStr',
    'constr',
    'PyObject',
    'ConstrainedInt',
    'conint',
    'PositiveInt',
    'NegativeInt',
    'NonNegativeInt',
    'NonPositiveInt',
    'ConstrainedFloat',
    'confloat',
    'PositiveFloat',
    'NegativeFloat',
    'NonNegativeFloat',
    'NonPositiveFloat',
    'FiniteFloat',
    'ConstrainedDecimal',
    'condecimal',
    'UUID1',
    'UUID3',
    'UUID4',
    'UUID5',
    'FilePath',
    'DirectoryPath',
    'Json',
    'JsonWrapper',
    'SecretField',
    'SecretStr',
    'SecretBytes',
    'StrictBool',
    'StrictBytes',
    'StrictInt',
    'StrictFloat',
    'PaymentCardNumber',
    'ByteSize',
    'PastDate',
    'FutureDate',
    'ConstrainedDate',
    'condate',
]

NoneStr: Optional[str]
NoneBytes: Optional[bytes]
StrBytes: Union[str, bytes]
NoneStrBytes: Optional[StrBytes]
OptionalInt: Optional[int]
OptionalIntFloat: Union[OptionalInt, float]
OptionalIntFloatDecimal: Union[OptionalIntFloat, Decimal]
OptionalDate: Optional[date]
StrIntFloat: Union[str, int, float]

if TYPE_CHECKING:
    from typing_extensions import Annotated
    from pydantic.v1.dataclasses import Dataclass
    from pydantic.v1.main import BaseModel
    from pydantic.v1.typing import CallableGenerator

    ModelOrDc = Type[Union['BaseModel', 'Dataclass']]

T = TypeVar('T')
_DEFINED_TYPES: WeakSet = WeakSet()


@overload
def _registered(typ: Type[Any]) -> Type[Any]:
    ...


@overload
def _registered(typ: Type[Any]) -> Type[Any]:
    ...


def _registered(typ: Type[Any]) -> Type[Any]:
    _DEFINED_TYPES.add(typ)
    return typ


class ConstrainedNumberMeta(type):
    def __new__(cls, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> 'ConstrainedInt':
        new_cls = cast('ConstrainedInt', type.__new__(cls, name, bases, dct))
        if new_cls.gt is not None and new_cls.ge is not None:
            raise errors.ConfigError('bounds gt and ge cannot be specified at the same time')
        if new_cls.lt is not None and new_cls.le is not None:
            raise errors.ConfigError('bounds lt and le cannot be specified at the same time')
        return _registered(new_cls)


if TYPE_CHECKING:
    StrictBool: Type[bool]
else:

    class StrictBool(int):
        """
        StrictBool to allow for bools which are not type-coerced.
        """

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(type='boolean')

        @classmethod
        def __get_validators__(cls) -> CallableGenerator:
            yield cls.validate

        @classmethod
        def validate(cls, value: Any) -> bool:
            """
            Ensure that we only allow bools.
            """
            if isinstance(value, bool):
                return value
            raise errors.StrictBoolError()


class ConstrainedInt(int, metaclass=ConstrainedNumberMeta):
    strict: ClassVar[bool] = False
    gt: Optional[float] = None
    ge: Optional[float] = None
    lt: Optional[float] = None
    le: Optional[float] = None
    multiple_of: Optional[float] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            exclusiveMinimum=cls.gt,
            exclusiveMaximum=cls.lt,
            minimum=cls.ge,
            maximum=cls.le,
            multipleOf=cls.multiple_of,
        )

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield strict_int_validator if cls.strict else int_validator
        yield number_size_validator
        yield number_multiple_validator


def conint(*, strict: bool = False, gt: Optional[float] = None, ge: Optional[float] = None,
           lt: Optional[float] = None, le: Optional[float] = None,
           multiple_of: Optional[float] = None) -> Type[ConstrainedInt]:
    namespace: Dict[str, Any] = dict(strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of)
    return type('ConstrainedIntValue', (ConstrainedInt,), namespace)


if TYPE_CHECKING:
    PositiveInt: Type[int]
    NegativeInt: Type[int]
    NonPositiveInt: Type[int]
    NonNegativeInt: Type[int]
    StrictInt: Type[int]
else:

    class PositiveInt(ConstrainedInt):
        gt: Optional[float] = 0

    class NegativeInt(ConstrainedInt):
        lt: Optional[float] = 0

    class NonPositiveInt(ConstrainedInt):
        le: Optional[float] = 0

    class NonNegativeInt(ConstrainedInt):
        ge: Optional[float] = 0

    class StrictInt(ConstrainedInt):
        strict: ClassVar[bool] = True


class ConstrainedFloat(float, metaclass=ConstrainedNumberMeta):
    strict: ClassVar[bool] = False
    gt: Optional[float] = None
    ge: Optional[float] = None
    lt: Optional[float] = None
    le: Optional[float] = None
    multiple_of: Optional[float] = None
    allow_inf_nan: Optional[bool] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            exclusiveMinimum=cls.gt,
            exclusiveMaximum=cls.lt,
            minimum=cls.ge,
            maximum=cls.le,
            multipleOf=cls.multiple_of,
        )
        if field_schema.get('exclusiveMinimum') == -math.inf:
            del field_schema['exclusiveMinimum']
        if field_schema.get('minimum') == -math.inf:
            del field_schema['minimum']
        if field_schema.get('exclusiveMaximum') == math.inf:
            del field_schema['exclusiveMaximum']
        if field_schema.get('maximum') == math.inf:
            del field_schema['maximum']

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield strict_float_validator if cls.strict else float_validator
        yield number_size_validator
        yield number_multiple_validator
        yield float_finite_validator


def confloat(*, strict: bool = False, gt: Optional[float] = None, ge: Optional[float] = None,
            lt: Optional[float] = None, le: Optional[float] = None,
            multiple_of: Optional[float] = None, allow_inf_nan: Optional[bool] = None) -> Type[ConstrainedFloat]:
    namespace: Dict[str, Any] = dict(
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
    )
    return type('ConstrainedFloatValue', (ConstrainedFloat,), namespace)


if TYPE_CHECKING:
    PositiveFloat: Type[float]
    NegativeFloat: Type[float]
    NonPositiveFloat: Type[float]
    NonNegativeFloat: Type[float]
    StrictFloat: Type[float]
    FiniteFloat: Type[float]
else:

    class PositiveFloat(ConstrainedFloat):
        gt: Optional[float] = 0

    class NegativeFloat(ConstrainedFloat):
        lt: Optional[float] = 0

    class NonPositiveFloat(ConstrainedFloat):
        le: Optional[float] = 0

    class NonNegativeFloat(ConstrainedFloat):
        ge: Optional[float] = 0

    class StrictFloat(ConstrainedFloat):
        strict: ClassVar[bool] = True

    class FiniteFloat(ConstrainedFloat):
        allow_inf_nan: ClassVar[bool] = False


class ConstrainedBytes(bytes):
    strip_whitespace: ClassVar[bool] = False
    to_upper: ClassVar[bool] = False
    to_lower: ClassVar[bool] = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    strict: ClassVar[bool] = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length)

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield strict_bytes_validator if cls.strict else bytes_validator
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator


def conbytes(*, strip_whitespace: bool = False, to_upper: bool = False, to_lower: bool = False,
            min_length: Optional[int] = None, max_length: Optional[int] = None,
            strict: bool = False) -> Type[ConstrainedBytes]:
    namespace: Dict[str, Any] = dict(
        strip_whitespace=strip_whitespace,
        to_upper=to_upper,
        to_lower=to_lower,
        min_length=min_length,
        max_length=max_length,
        strict=strict,
    )
    return _registered(type('ConstrainedBytesValue', (ConstrainedBytes,), namespace))


if TYPE_CHECKING:
    StrictBytes: Type[bytes]
else:

    class StrictBytes(ConstrainedBytes):
        strict: ClassVar[bool] = True


class ConstrainedStr(str):
    strip_whitespace: ClassVar[bool] = False
    to_upper: ClassVar[bool] = False
    to_lower: ClassVar[bool] = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    curtail_length: Optional[int] = None
    regex: Optional[Union[str, Pattern]] = None
    strict: ClassVar[bool] = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            minLength=cls.min_length,
            maxLength=cls.max_length,
            pattern=cls.regex and cls._get_pattern(cls.regex),
        )

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield strict_str_validator if cls.strict else str_validator
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator
        yield cls.validate

    @classmethod
    def validate(cls, value: str) -> str:
        if cls.curtail_length and len(value) > cls.curtail_length:
            value = value[:cls.curtail_length]
        if cls.regex:
            if not re.match(cls.regex, value):
                raise errors.StrRegexError(pattern=cls._get_pattern(cls.regex))
        return value

    @staticmethod
    def _get_pattern(regex: Union[str, Pattern]) -> str:
        return regex if isinstance(regex, str) else regex.pattern


def constr(*, strip_whitespace: bool = False, to_upper: bool = False, to_lower: bool = False,
           strict: bool = False, min_length: Optional[int] = None, max_length: Optional[int] = None,
           curtail_length: Optional[int] = None, regex: Optional[str] = None) -> Type[ConstrainedStr]:
    compiled_regex = re.compile(regex) if regex else None
    namespace: Dict[str, Any] = dict(
        strip_whitespace=strip_whitespace,
        to_upper=to_upper,
        to_lower=to_lower,
        strict=strict,
        min_length=min_length,
        max_length=max_length,
        curtail_length=curtail_length,
        regex=compiled_regex,
    )
    return _registered(type('ConstrainedStrValue', (ConstrainedStr,), namespace))


if TYPE_CHECKING:
    StrictStr: Type[str]
else:

    class StrictStr(ConstrainedStr):
        strict: ClassVar[bool] = True


class ConstrainedSet(set):
    __origin__: Type[Set[Any]] = set
    min_items: Optional[int] = None
    max_items: Optional[int] = None

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.set_length_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items)

    @classmethod
    def set_length_validator(cls, v: Any) -> Set[Any]:
        if v is None:
            return set()
        v = set_validator(v)
        v_len = len(v)
        if cls.min_items is not None and v_len < cls.min_items:
            raise errors.SetMinLengthError(limit_value=cls.min_items)
        if cls.max_items is not None and v_len > cls.max_items:
            raise errors.SetMaxLengthError(limit_value=cls.max_items)
        return v


def conset(item_type: Type[Any], *, min_items: Optional[int] = None, max_items: Optional[int] = None) -> Type[ConstrainedSet]:
    namespace: Dict[str, Any] = {
        'min_items': min_items,
        'max_items': max_items,
        'item_type': item_type,
        '__args__': [item_type],
    }
    return new_class('ConstrainedSetValue', (ConstrainedSet,), {}, lambda ns: ns.update(namespace))


class ConstrainedFrozenSet(FrozenSet[Any]):
    __origin__: Type[FrozenSet[Any]] = frozenset
    min_items: Optional[int] = None
    max_items: Optional[int] = None

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.frozenset_length_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items)

    @classmethod
    def frozenset_length_validator(cls, v: Any) -> FrozenSet[Any]:
        if v is None:
            return frozenset()
        v = frozenset_validator(v)
        v_len = len(v)
        if cls.min_items is not None and v_len < cls.min_items:
            raise errors.FrozenSetMinLengthError(limit_value=cls.min_items)
        if cls.max_items is not None and v_len > cls.max_items:
            raise errors.FrozenSetMaxLengthError(limit_value=cls.max_items)
        return v


def confrozenset(item_type: Type[Any], *, min_items: Optional[int] = None, max_items: Optional[int] = None) -> Type[ConstrainedFrozenSet]:
    namespace: Dict[str, Any] = {
        'min_items': min_items,
        'max_items': max_items,
        'item_type': item_type,
        '__args__': [item_type],
    }
    return new_class('ConstrainedFrozenSetValue', (ConstrainedFrozenSet,), {}, lambda ns: ns.update(namespace))


class ConstrainedList(list):
    __origin__: Type[List[Any]] = list
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: Optional[bool] = None

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.list_length_validator
        if cls.unique_items:
            yield cls.unique_items_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items, uniqueItems=cls.unique_items)

    @classmethod
    def list_length_validator(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        v = list_validator(v)
        v_len = len(v)
        if cls.min_items is not None and v_len < cls.min_items:
            raise errors.ListMinLengthError(limit_value=cls.min_items)
        if cls.max_items is not None and v_len > cls.max_items:
            raise errors.ListMaxLengthError(limit_value=cls.max_items)
        return v

    @classmethod
    def unique_items_validator(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        for i, value in enumerate(v, start=1):
            if value in v[i:]:
                raise errors.ListUniqueItemsError()
        return v


def conlist(item_type: Type[Any], *, min_items: Optional[int] = None, max_items: Optional[int] = None,
            unique_items: Optional[bool] = None) -> Type[ConstrainedList]:
    namespace: Dict[str, Any] = {
        'min_items': min_items,
        'max_items': max_items,
        'unique_items': unique_items,
        'item_type': item_type,
        '__args__': (item_type,),
    }
    return new_class('ConstrainedListValue', (ConstrainedList,), {}, lambda ns: ns.update(namespace))


if TYPE_CHECKING:
    PyObject: Type[Callable[..., Any]]
else:

    class PyObject:
        validate_always: ClassVar[bool] = True

        @classmethod
        def __get_validators__(cls) -> CallableGenerator:
            yield cls.validate

        @classmethod
        def validate(cls, value: Any) -> Callable[..., Any]:
            if isinstance(value, Callable):
                return value
            try:
                value = str_validator(value)
            except errors.StrError:
                raise errors.PyObjectError(error_message='value is neither a valid import path not a valid callable')
            try:
                return import_string(value)
            except ImportError as e:
                raise errors.PyObjectError(error_message=str(e))


class ConstrainedDecimal(Decimal, metaclass=ConstrainedNumberMeta):
    gt: Optional[float] = None
    ge: Optional[float] = None
    lt: Optional[float] = None
    le: Optional[float] = None
    max_digits: Optional[int] = None
    decimal_places: Optional[int] = None
    multiple_of: Optional[float] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            exclusiveMinimum=cls.gt,
            exclusiveMaximum=cls.lt,
            minimum=cls.ge,
            maximum=cls.le,
            multipleOf=cls.multiple_of,
        )

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield decimal_validator
        yield number_size_validator
        yield number_multiple_validator
        yield cls.validate

    @classmethod
    def validate(cls, value: Decimal) -> Decimal:
        try:
            normalized_value = value.normalize()
        except InvalidOperation:
            normalized_value = value
        digit_tuple = normalized_value.as_tuple().digits
        exponent = normalized_value.as_tuple().exponent
        if exponent in {'F', 'n', 'N'}:
            raise errors.DecimalIsNotFiniteError()
        if exponent >= 0:
            digits = len(digit_tuple) + exponent
            decimals = 0
        elif abs(exponent) > len(digit_tuple):
            digits = decimals = abs(exponent)
        else:
            digits = len(digit_tuple)
            decimals = abs(exponent)
        whole_digits = digits - decimals
        if cls.max_digits is not None and digits > cls.max_digits:
            raise errors.DecimalMaxDigitsError(max_digits=cls.max_digits)
        if cls.decimal_places is not None and decimals > cls.decimal_places:
            raise errors.DecimalMaxPlacesError(decimal_places=cls.decimal_places)
        if cls.max_digits is not None and cls.decimal_places is not None:
            expected = cls.max_digits - cls.decimal_places
            if whole_digits > expected:
                raise errors.DecimalWholeDigitsError(whole_digits=expected)
        return value


def condecimal(*, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None,
               le: Optional[float] = None, max_digits: Optional[int] = None,
               decimal_places: Optional[int] = None, multiple_of: Optional[float] = None) -> Type[ConstrainedDecimal]:
    namespace: Dict[str, Any] = dict(
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        max_digits=max_digits,
        decimal_places=decimal_places,
        multiple_of=multiple_of,
    )
    return type('ConstrainedDecimalValue', (ConstrainedDecimal,), namespace)


if TYPE_CHECKING:
    UUID1: Type[UUID]
    UUID3: Type[UUID]
    UUID4: Type[UUID]
    UUID5: Type[UUID]
else:

    class UUID1(UUID):
        _required_version: ClassVar[int] = 1

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(type='string', format=f'uuid{cls._required_version}')

    class UUID3(UUID1):
        _required_version: ClassVar[int] = 3

    class UUID4(UUID1):
        _required_version: ClassVar[int] = 4

    class UUID5(UUID1):
        _required_version: ClassVar[int] = 5


if TYPE_CHECKING:
    FilePath: Type[Path]
    DirectoryPath: Type[Path]
else:

    class FilePath(Path):

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(format='file-path')

        @classmethod
        def __get_validators__(cls) -> CallableGenerator:
            yield path_validator
            yield path_exists_validator
            yield cls.validate

        @classmethod
        def validate(cls, value: Path) -> Path:
            if not value.is_file():
                raise errors.PathNotAFileError(path=value)
            return value


    class DirectoryPath(Path):

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(format='directory-path')

        @classmethod
        def __get_validators__(cls) -> CallableGenerator:
            yield path_validator
            yield path_exists_validator
            yield cls.validate

        @classmethod
        def validate(cls, value: Path) -> Path:
            if not value.is_dir():
                raise errors.PathNotADirectoryError(path=value)
            return value


class JsonWrapper:
    inner_type: Type[Any]


class JsonMeta(type):
    def __getitem__(self, t: Type[Any]) -> Type[JsonWrapper]:
        if t is Any:
            return Json
        return _registered(type('JsonWrapperValue', (JsonWrapper,), {'inner_type': t}))


if TYPE_CHECKING:
    Json: Type['Annotated[T, ...]']
else:

    class Json(metaclass=JsonMeta):

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(type='string', format='json-string')


class SecretField(abc.ABC):
    """
    Note: this should be implemented as a generic like `SecretField(ABC, Generic[T])`,
          the `__init__()` should be part of the abstract class and the
          `get_secret_value()` method should use the generic `T` type.

          However Cython doesn't support very well generics at the moment and
          the generated code fails to be imported (see
          https://github.com/cython/cython/issues/2753).
    """

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.get_secret_value() == other.get_secret_value()

    def __str__(self) -> str:
        return '**********' if self.get_secret_value() else ''

    def __hash__(self) -> int:
        return hash(self.get_secret_value())

    @abc.abstractmethod
    def get_secret_value(self) -> Any:
        ...


class SecretStr(SecretField):
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            type='string',
            writeOnly=True,
            format='password',
            minLength=cls.min_length,
            maxLength=cls.max_length,
        )

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.validate
        yield constr_length_validator

    @classmethod
    def validate(cls, value: Any) -> 'SecretStr':
        if isinstance(value, cls):
            return value
        value = str_validator(value)
        return cls(value)

    def __init__(self, value: str) -> None:
        self._secret_value: str = value

    def __repr__(self) -> str:
        return f"SecretStr('{self}')"

    def __len__(self) -> int:
        return len(self._secret_value)

    def display(self) -> str:
        warnings.warn('`secret_str.display()` is deprecated, use `str(secret_str)` instead', DeprecationWarning)
        return str(self)

    def get_secret_value(self) -> str:
        return self._secret_value


class SecretBytes(SecretField):
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            type='string',
            writeOnly=True,
            format='password',
            minLength=cls.min_length,
            maxLength=cls.max_length,
        )

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.validate
        yield constr_length_validator

    @classmethod
    def validate(cls, value: Any) -> 'SecretBytes':
        if isinstance(value, cls):
            return value
        value = bytes_validator(value)
        return cls(value)

    def __init__(self, value: bytes) -> None:
        self._secret_value: bytes = value

    def __repr__(self) -> str:
        return f"SecretBytes(b'{self}')"

    def __len__(self) -> int:
        return len(self._secret_value)

    def display(self) -> str:
        warnings.warn('`secret_bytes.display()` is deprecated, use `str(secret_bytes)` instead', DeprecationWarning)
        return str(self)

    def get_secret_value(self) -> bytes:
        return self._secret_value


class PaymentCardBrand(str, Enum):
    amex: str = 'American Express'
    mastercard: str = 'Mastercard'
    visa: str = 'Visa'
    other: str = 'other'

    def __str__(self) -> str:
        return self.value


class PaymentCardNumber(str):
    """
    Based on: https://en.wikipedia.org/wiki/Payment_card_number
    """
    strip_whitespace: ClassVar[bool] = True
    min_length: ClassVar[int] = 12
    max_length: ClassVar[int] = 19

    bin: str
    last4: str
    brand: PaymentCardBrand

    def __init__(self, card_number: str) -> None:
        super().__init__()
        self.bin = card_number[:6]
        self.last4 = card_number[-4:]
        self.brand = self._get_brand(card_number)

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield str_validator
        yield constr_strip_whitespace
        yield constr_length_validator
        yield cls.validate_digits
        yield cls.validate_luhn_check_digit
        yield cls
        yield cls.validate_length_for_brand

    @property
    def masked(self) -> str:
        num_masked = len(self) - 10
        return f'{self.bin}{"*" * num_masked}{self.last4}'

    @classmethod
    def validate_digits(cls, card_number: str) -> str:
        if not card_number.isdigit():
            raise errors.NotDigitError
        return card_number

    @classmethod
    def validate_luhn_check_digit(cls, card_number: str) -> str:
        """
        Based on: https://en.wikipedia.org/wiki/Luhn_algorithm
        """
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
        valid: bool = sum_ % 10 == 0
        if not valid:
            raise errors.LuhnValidationError
        return card_number

    @classmethod
    def validate_length_for_brand(cls, card_number: str) -> str:
        """
        Validate length based on BIN for major brands:
        https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_(IIN)
        """
        required_length: Optional[Union[int, str]] = None
        if card_number.brand in PaymentCardBrand.mastercard:
            required_length = 16
            valid: bool = len(card_number) == required_length
        elif card_number.brand == PaymentCardBrand.visa:
            required_length = '13, 16 or 19'
            valid = len(card_number) in {13, 16, 19}
        elif card_number.brand == PaymentCardBrand.amex:
            required_length = 15
            valid = len(card_number) == required_length
        else:
            valid = True
        if not valid:
            raise errors.InvalidLengthForBrand(brand=card_number.brand, required_length=required_length)
        return card_number

    @staticmethod
    def _get_brand(card_number: str) -> PaymentCardBrand:
        if card_number[0] == '4':
            brand = PaymentCardBrand.visa
        elif 51 <= int(card_number[:2]) <= 55:
            brand = PaymentCardBrand.mastercard
        elif card_number[:2] in {'34', '37'}:
            brand = PaymentCardBrand.amex
        else:
            brand = PaymentCardBrand.other
        return brand


BYTE_SIZES: Dict[str, int] = {
    'b': 1,
    'kb': 10 ** 3,
    'mb': 10 ** 6,
    'gb': 10 ** 9,
    'tb': 10 ** 12,
    'pb': 10 ** 15,
    'eb': 10 ** 18,
    'kib': 2 ** 10,
    'mib': 2 ** 20,
    'gib': 2 ** 30,
    'tib': 2 ** 40,
    'pib': 2 ** 50,
    'eib': 2 ** 60,
}
BYTE_SIZES.update({k.lower()[0]: v for k, v in BYTE_SIZES.items() if 'i' not in k})
byte_string_re: Pattern = re.compile(r'^\s*(\d*\.?\d+)\s*(\w+)?', re.IGNORECASE)


class ByteSize(int):

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> 'ByteSize':
        try:
            return cls(int(v))
        except (ValueError, TypeError):
            pass
        str_match = byte_string_re.match(str(v))
        if str_match is None:
            raise errors.InvalidByteSize()
        scalar_str, unit = str_match.groups()
        if unit is None:
            unit = 'b'
        try:
            unit_mult = BYTE_SIZES[unit.lower()]
        except KeyError:
            raise errors.InvalidByteSizeUnit(unit=unit)
        return cls(int(float(scalar_str) * unit_mult))

    def human_readable(self, decimal: bool = False) -> str:
        if decimal:
            divisor: float = 1000
            units: List[str] = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
            final_unit: str = 'EB'
        else:
            divisor: float = 1024
            units: List[str] = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
            final_unit: str = 'EiB'
        num: float = float(self)
        for unit in units:
            if abs(num) < divisor:
                return f'{num:0.1f}{unit}'
            num /= divisor
        return f'{num:0.1f}{final_unit}'

    def to(self, unit: str) -> float:
        try:
            unit_div = BYTE_SIZES[unit.lower()]
        except KeyError:
            raise errors.InvalidByteSizeUnit(unit=unit)
        return self / unit_div


if TYPE_CHECKING:
    PastDate: Type[date]
    FutureDate: Type[date]
else:

    class PastDate(date):

        @classmethod
        def __get_validators__(cls) -> CallableGenerator:
            yield parse_date
            yield cls.validate

        @classmethod
        def validate(cls, value: date) -> date:
            if value >= date.today():
                raise errors.DateNotInThePastError()
            return value


    class FutureDate(date):

        @classmethod
        def __get_validators__(cls) -> CallableGenerator:
            yield parse_date
            yield cls.validate

        @classmethod
        def validate(cls, value: date) -> date:
            if value <= date.today():
                raise errors.DateNotInTheFutureError()
            return value


class ConstrainedDate(date, metaclass=ConstrainedNumberMeta):
    gt: Optional[date] = None
    ge: Optional[date] = None
    lt: Optional[date] = None
    le: Optional[date] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            exclusiveMinimum=cls.gt,
            exclusiveMaximum=cls.lt,
            minimum=cls.ge,
            maximum=cls.le,
        )

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield parse_date
        yield number_size_validator


def condate(*, gt: Optional[date] = None, ge: Optional[date] = None,
           lt: Optional[date] = None, le: Optional[date] = None) -> Type[ConstrainedDate]:
    namespace: Dict[str, Any] = dict(gt=gt, ge=ge, lt=lt, le=le)
    return type('ConstrainedDateValue', (ConstrainedDate,), namespace)
