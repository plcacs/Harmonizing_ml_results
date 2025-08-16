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

NoneStr = Optional[str]
NoneBytes = Optional[bytes]
StrBytes = Union[str, bytes]
NoneStrBytes = Optional[StrBytes]
OptionalInt = Optional[int]
OptionalIntFloat = Union[OptionalInt, float]
OptionalIntFloatDecimal = Union[OptionalIntFloat, Decimal]
OptionalDate = Optional[date]
StrIntFloat = Union[str, int, float]

if TYPE_CHECKING:
    from typing_extensions import Annotated

    from pydantic.v1.dataclasses import Dataclass
    from pydantic.v1.main import BaseModel
    from pydantic.v1.typing import CallableGenerator

    ModelOrDc = Type[Union[BaseModel, Dataclass]]

T = TypeVar('T')
_DEFINED_TYPES: 'WeakSet[Type[Any]]' = WeakSet()


@overload
def _registered(typ: Type[T]) -> Type[T]:
    ...


@overload
def _registered(typ: 'ConstrainedNumberMeta') -> 'ConstrainedNumberMeta':
    ...


def _registered(typ: Union[Type[T], 'ConstrainedNumberMeta']) -> Union[Type[T], 'ConstrainedNumberMeta']:
    _DEFINED_TYPES.add(typ)
    return typ


class ConstrainedNumberMeta(type):
    def __new__(cls, name: str, bases: Any, dct: Dict[str, Any]) -> 'ConstrainedInt':
        new_cls = cast('ConstrainedInt', type.__new__(cls, name, bases, dct))

        if new_cls.gt is not None and new_cls.ge is not None:
            raise errors.ConfigError('bounds gt and ge cannot be specified at the same time')
        if new_cls.lt is not None and new_cls.le is not None:
            raise errors.ConfigError('bounds lt and le cannot be specified at the same time')

        return _registered(new_cls)


class StrictBool(int):
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='boolean')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        raise errors.StrictBoolError()


class ConstrainedInt(int, metaclass=ConstrainedNumberMeta):
    strict: bool = False
    gt: OptionalInt = None
    ge: OptionalInt = None
    lt: OptionalInt = None
    le: OptionalInt = None
    multiple_of: OptionalInt = None

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
    def __get_validators__(cls) -> 'CallableGenerator':
        yield strict_int_validator if cls.strict else int_validator
        yield number_size_validator
        yield number_multiple_validator


def conint(
    *,
    strict: bool = False,
    gt: Optional[int] = None,
    ge: Optional[int] = None,
    lt: Optional[int] = None,
    le: Optional[int] = None,
    multiple_of: Optional[int] = None,
) -> Type[int]:
    namespace = dict(strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of)
    return type('ConstrainedIntValue', (ConstrainedInt,), namespace)


class PositiveInt(ConstrainedInt):
    gt: int = 0


class NegativeInt(ConstrainedInt):
    lt: int = 0


class NonPositiveInt(ConstrainedInt):
    le: int = 0


class NonNegativeInt(ConstrainedInt):
    ge: int = 0


class StrictInt(ConstrainedInt):
    strict: bool = True


class ConstrainedFloat(float, metaclass=ConstrainedNumberMeta):
    strict: bool = False
    gt: OptionalIntFloat = None
    ge: OptionalIntFloat = None
    lt: OptionalIntFloat = None
    le: OptionalIntFloat = None
    multiple_of: OptionalIntFloat = None
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

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield strict_float_validator if cls.strict else float_validator
        yield number_size_validator
        yield number_multiple_validator
        yield float_finite_validator


def confloat(
    *,
    strict: bool = False,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None,
    multiple_of: Optional[float] = None,
    allow_inf_nan: Optional[bool] = None,
) -> Type[float]:
    namespace = dict(strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan)
    return type('ConstrainedFloatValue', (ConstrainedFloat,), namespace)


class PositiveFloat(ConstrainedFloat):
    gt: float = 0


class NegativeFloat(ConstrainedFloat):
    lt: float = 0


class NonPositiveFloat(ConstrainedFloat):
    le: float = 0


class NonNegativeFloat(ConstrainedFloat):
    ge: float = 0


class StrictFloat(ConstrainedFloat):
    strict: bool = True


class FiniteFloat(ConstrainedFloat):
    allow_inf_nan: bool = False


class ConstrainedBytes(bytes):
    strip_whitespace: bool = False
    to_upper: bool = False
    to_lower: bool = False
    min_length: OptionalInt = None
    max_length: OptionalInt = None
    strict: bool = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield strict_bytes_validator if cls.strict else bytes_validator
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator


def conbytes(
    *,
    strip_whitespace: bool = False,
    to_upper: bool = False,
    to_lower: bool = False,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    strict: bool = False,
) -> Type[bytes]:
    namespace = dict(
        strip_whitespace=strip_whitespace,
        to_upper=to_upper,
        to_lower=to_lower,
        min_length=min_length,
        max_length=max_length,
        strict=strict,
    )
    return _registered(type('ConstrainedBytesValue', (ConstrainedBytes,), namespace))


class StrictBytes(ConstrainedBytes):
    strict: bool = True


class ConstrainedStr(str):
    strip_whitespace: bool = False
    to_upper: bool = False
    to_lower: bool = False
    min_length: OptionalInt = None
    max_length: OptionalInt = None
    curtail_length: OptionalInt = None
    regex: Optional[Union[str, Pattern[str]]] = None
    strict: bool = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            minLength=cls.min_length,
            maxLength=cls.max_length,
            pattern=cls.regex and cls._get_pattern(cls.regex),
        )

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield strict_str_validator if cls.strict else str_validator
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator
        yield cls.validate

    @classmethod
    def validate(cls, value: str) -> str:
        if cls.curtail_length and len(value) > cls.curtail_length:
            value = value[: cls.curtail_length]

        if cls.regex:
            if not re.match(cls.regex, value):
                raise errors.StrRegexError(pattern=cls._get_pattern(cls.regex))

        return value

    @staticmethod
    def _get_pattern(regex: Union[str, Pattern[str]]) -> str:
        return regex if isinstance(regex, str) else regex.pattern


def constr(
    *,
    strip_whitespace: bool = False,
    to_upper: bool = False,
    to_lower: bool = False,
    strict: bool = False,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    curtail_length: Optional[int] = None,
    regex: Optional[str] = None,
) -> Type[str]:
    namespace = dict(
        strip_whitespace=strip_whitespace,
        to_upper=to_upper,
        to_lower=to_lower,
        strict=strict,
        min_length=min_length,
        max_length=max_length,
        curtail_length=curtail_length,
        regex=regex and re.compile(regex),
    )
    return _registered(type('ConstrainedStrValue', (ConstrainedStr,), namespace))


class StrictStr(ConstrainedStr):
    strict: bool = True


class ConstrainedSet(set):
    __origin__ = set
    __args__: Set[Type[T]]
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    item_type: Type[T]

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.set_length_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items)

    @classmethod
    def set_length_validator(cls, v: Optional[Set[T]]) -> Optional[Set[T]]:
        if v is None:
            return None

        v = set_validator(v)
        v_len = len(v)

        if cls.min_items is not None and v_len < cls.min_items:
            raise errors.SetMinLengthError(limit_value=cls.min_items)

        if cls.max_items is not None and v_len > cls.max_items:
            raise errors.SetMaxLengthError(limit_value=cls.max_items)

        return v


def conset(item_type: Type[T], *, min_items: Optional[int] = None, max_items: Optional[int] = None) -> Type[Set[T]]:
    namespace = {'min_items': min_items, 'max_items': max_items, 'item_type': item_type, '__args__': [item_type]}
    return new_class('ConstrainedSetValue', (ConstrainedSet,), {}, lambda ns: ns.update(namespace))


class ConstrainedFrozenSet(frozenset):
    __origin__ = frozenset
    __args__: FrozenSet[Type[T]]
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    item_type: Type[T]

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.frozenset_length_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items)

    @classmethod
    def frozenset_length_validator(cls, v: Optional[FrozenSet[T]]) -> Optional[FrozenSet[T]]:
        if v is None:
            return None

        v = frozenset_validator(v)
        v_len = len(v)

        if cls.min_items is not None and v_len < cls.min_items:
            raise errors.FrozenSetMinLengthError(limit_value=cls.min_items)

        if cls.max_items is not None and v_len > cls.max_items:
            raise errors.FrozenSetMaxLengthError(limit_value=cls.max_items)

        return v


def confrozenset(
    item_type: Type[T], *, min_items: Optional[int] = None, max_items: Optional[int] = None
) -> Type[FrozenSet[T]]:
    namespace = {'min_items': min_items, 'max_items': max_items, 'item_type': item_type, '__args__': [item_type]}
    return new_class('ConstrainedFrozenSetValue', (ConstrainedFrozenSet,), {}, lambda ns: ns.update(namespace))


class ConstrainedList(list):
    __origin__ = list
    __args__: Tuple[Type[T], ...]
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: Optional[bool] = None
    item_type: Type[T]

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.list_length_validator
        if cls.unique_items:
            yield cls.unique_items_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items, uniqueItems=cls.unique_items)

    @classmethod
    def list_length_validator(cls, v: Optional[List[T]]) -> Optional[List[T]]:
        if v is None:
            return None

        v = list_validator(v)
        v_len = len(v)

        if cls.min_items is not None and v_len < cls.min_items:
            raise errors.ListMinLengthError(limit_value=cls.min_items)

        if cls.max_items is not None and v_len > cls.max_items:
            raise errors.ListMaxLengthError(limit_value=cls.max_items)

        return v

    @classmethod
    def unique_items_validator(cls, v: Optional[List[T]]) -> Optional[List[T]]:
        if v is None:
            return None

        for i, value in enumerate(v, start=1):
            if value in v[i:]:
                raise errors.ListUniqueItemsError()

        return v


def conlist(
    item_type: Type[T], *, min_items: Optional[int] = None, max_items: Optional[int] = None, unique_items: bool = None
) -> Type[List[T]]:
    namespace = dict(
        min_items=min_items, max_items=max_items, unique_items=unique_items, item_type=item_type, __args__=(item_type,)
    )
    return new_class('ConstrainedListValue', (ConstrainedList,), {}, lambda ns: ns.update(namespace))


class PyObject:
    validate_always: bool = True

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod