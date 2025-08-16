from typing import Optional, Union, TypeVar, cast, Callable, Type, Annotated
from pydantic.v1.dataclasses import Dataclass
from pydantic.v1.main import BaseModel

NoneStr: Optional[str]
NoneBytes: Optional[bytes]
StrBytes: Union[str, bytes]
NoneStrBytes: Optional[StrBytes]
OptionalInt: Optional[int]
OptionalIntFloat: Union[OptionalInt, float]
OptionalIntFloatDecimal: Union[OptionalIntFloat, Decimal]
OptionalDate: Optional[date]
StrIntFloat: Union[str, int, float]
ModelOrDc: Type[Union[BaseModel, Dataclass]]
T = TypeVar('T')

def _registered(typ: T) -> T:
    ...

class ConstrainedNumberMeta(type):
    ...

StrictBool: Union[bool, int]

class ConstrainedInt(int, metaclass=ConstrainedNumberMeta):
    ...

def conint(*, strict: bool = False, gt: int = None, ge: int = None, lt: int = None, le: int = None, multiple_of: int = None) -> Type[ConstrainedInt]:
    ...

PositiveInt: int
NegativeInt: int
NonPositiveInt: int
NonNegativeInt: int
StrictInt: int

class ConstrainedFloat(float, metaclass=ConstrainedNumberMeta):
    ...

def confloat(*, strict: bool = False, gt: float = None, ge: float = None, lt: float = None, le: float = None, multiple_of: float = None, allow_inf_nan: bool = None) -> Type[ConstrainedFloat]:
    ...

PositiveFloat: float
NegativeFloat: float
NonPositiveFloat: float
NonNegativeFloat: float
StrictFloat: float
FiniteFloat: float

class ConstrainedBytes(bytes):
    ...

def conbytes(*, strip_whitespace: bool = False, to_upper: bool = False, to_lower: bool = False, min_length: int = None, max_length: int = None, strict: bool = False) -> Type[ConstrainedBytes]:
    ...

StrictBytes: bytes

class ConstrainedStr(str):
    ...

def constr(*, strip_whitespace: bool = False, to_upper: bool = False, to_lower: bool = False, strict: bool = False, min_length: int = None, max_length: int = None, curtail_length: int = None, regex: Optional[Pattern[str]] = None) -> Type[ConstrainedStr]:
    ...

StrictStr: str

class ConstrainedSet(set):
    ...

def conset(item_type: Type[T], *, min_items: int = None, max_items: int = None) -> Type[ConstrainedSet]:
    ...

class ConstrainedFrozenSet(frozenset):
    ...

def confrozenset(item_type: Type[T], *, min_items: int = None, max_items: int = None) -> Type[ConstrainedFrozenSet]:
    ...

class ConstrainedList(list):
    ...

def conlist(item_type: Type[T], *, min_items: int = None, max_items: int = None, unique_items: bool = None) -> Type[ConstrainedList]:
    ...

PyObject: Callable[..., Any]

class ConstrainedDecimal(Decimal, metaclass=ConstrainedNumberMeta):
    ...

def condecimal(*, gt: Decimal = None, ge: Decimal = None, lt: Decimal = None, le: Decimal = None, max_digits: int = None, decimal_places: int = None, multiple_of: Decimal = None) -> Type[ConstrainedDecimal]:
    ...

UUID1: UUID
UUID3: UUID
UUID4: UUID
UUID5: UUID

FilePath: Path
DirectoryPath: Path

class JsonWrapper:
    ...

class JsonMeta(type):
    ...

Json: Annotated[T, ...]

class SecretField(abc.ABC):
    ...

class SecretStr(SecretField):
    ...

class SecretBytes(SecretField):
    ...

class PaymentCardBrand(str, Enum):
    ...

class PaymentCardNumber(str):
    ...

class ByteSize(int):
    ...

class PastDate(date):
    ...

class FutureDate(date):
    ...

class ConstrainedDate(date, metaclass=ConstrainedNumberMeta):
    ...

def condate(*, gt: date = None, ge: date = None, lt: date = None, le: date = None) -> Type[ConstrainedDate]:
    ...
