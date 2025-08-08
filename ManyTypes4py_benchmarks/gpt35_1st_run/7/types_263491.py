from typing import Optional, Union, TypeVar, Type, Callable, Any, Annotated

NoneStr = Optional[str]
NoneBytes = Optional[bytes]
StrBytes = Union[str, bytes]
NoneStrBytes = Optional[StrBytes]
OptionalInt = Optional[int]
OptionalIntFloat = Union[OptionalInt, float]
OptionalIntFloatDecimal = Union[OptionalIntFloat, Decimal]
OptionalDate = Optional[date]
StrIntFloat = Union[str, int, float]
ModelOrDc = Type[Union[BaseModel, Dataclass]]
T = TypeVar('T')

def _registered(typ: T) -> T:
    ...

class ConstrainedNumberMeta(type):
    ...

StrictBool = bool

class ConstrainedInt(int, metaclass=ConstrainedNumberMeta):
    ...

def conint(*, strict: bool = False, gt: Optional[int] = None, ge: Optional[int] = None, lt: Optional[int] = None, le: Optional[int] = None, multiple_of: Optional[int] = None) -> Type[ConstrainedInt]:
    ...

PositiveInt = int
NegativeInt = int
NonPositiveInt = int
NonNegativeInt = int
StrictInt = int

class ConstrainedFloat(float, metaclass=ConstrainedNumberMeta):
    ...

def confloat(*, strict: bool = False, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, multiple_of: Optional[float] = None, allow_inf_nan: Optional[bool] = None) -> Type[ConstrainedFloat]:
    ...

PositiveFloat = float
NegativeFloat = float
NonPositiveFloat = float
NonNegativeFloat = float
StrictFloat = float
FiniteFloat = float

class ConstrainedBytes(bytes):
    ...

def conbytes(*, strip_whitespace: bool = False, to_upper: bool = False, to_lower: bool = False, min_length: Optional[int] = None, max_length: Optional[int] = None, strict: bool = False) -> Type[ConstrainedBytes]:
    ...

StrictBytes = bytes

class ConstrainedStr(str):
    ...

def constr(*, strip_whitespace: bool = False, to_upper: bool = False, to_lower: bool = False, strict: bool = False, min_length: Optional[int] = None, max_length: Optional[int] = None, curtail_length: Optional[int] = None, regex: Optional[Union[str, Pattern[str]]] = None) -> Type[ConstrainedStr]:
    ...

StrictStr = str

class ConstrainedSet(set):
    ...

def conset(item_type: Type[T], *, min_items: Optional[int] = None, max_items: Optional[int] = None) -> Type[ConstrainedSet]:
    ...

class ConstrainedFrozenSet(frozenset):
    ...

def confrozenset(item_type: Type[T], *, min_items: Optional[int] = None, max_items: Optional[int] = None) -> Type[ConstrainedFrozenSet]:
    ...

class ConstrainedList(list):
    ...

def conlist(item_type: Type[T], *, min_items: Optional[int] = None, max_items: Optional[int] = None, unique_items: Optional[bool] = None) -> Type[ConstrainedList]:
    ...

PyObject = Callable[..., Any]

class ConstrainedDecimal(Decimal, metaclass=ConstrainedNumberMeta):
    ...

def condecimal(*, gt: Optional[Decimal] = None, ge: Optional[Decimal] = None, lt: Optional[Decimal] = None, le: Optional[Decimal] = None, max_digits: Optional[int] = None, decimal_places: Optional[int] = None, multiple_of: Optional[Decimal] = None) -> Type[ConstrainedDecimal]:
    ...

UUID1 = UUID
UUID3 = UUID
UUID4 = UUID
UUID5 = UUID

FilePath = Path
DirectoryPath = Path

class JsonMeta(type):
    ...

class Json(metaclass=JsonMeta):
    ...

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

def condate(*, gt: Optional[date] = None, ge: Optional[date] = None, lt: Optional[date] = None, le: Optional[date] = None) -> Type[ConstrainedDate]:
    ...
