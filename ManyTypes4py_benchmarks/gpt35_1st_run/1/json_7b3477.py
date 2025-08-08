from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union
import datetime
import enum
import uuid
from collections import Counter, deque
from decimal import Decimal

__all__: List[str] = ['JSONEncoder', 'dumps', 'loads', 'str_to_decimal']
orjson: Any = None
DEFAULT_TEXTUAL_TYPES: List[Type[Union[Decimal, uuid.UUID, bytes]]] = [Decimal, uuid.UUID, bytes]
T = TypeVar('T')
TypeTuple = Tuple[Type[T], ...]
IsInstanceArg = Union[Type[Any], TypeTuple[Any]]
DJANGO_TEXTUAL_TYPES: List[Type[Any]] = []
TEXTUAL_TYPES: Tuple[Type[Union[Decimal, uuid.UUID, bytes]]] = tuple(DEFAULT_TEXTUAL_TYPES + DJANGO_TEXTUAL_TYPES)
_JSON_DEFAULT_KWARGS: Mapping[str, Any] = {}
DECIMAL_MAXLEN: int = 1000
SEQUENCE_TYPES: Tuple[Type[Any], ...] = (set, deque)
DateTypeTuple: Tuple[Type[Union[datetime.date, datetime.time]], ...] = (datetime.date, datetime.time)
DatetimeTypeTuple: Tuple[Type[Union[datetime.time, datetime.datetime]], ...] = (datetime.time, datetime.datetime)
MAPPING_TYPES: Tuple[Type[Any], ...] = (Counter,)
DATE_TYPES: Tuple[Type[Union[datetime.date, datetime.time]], ...] = (datetime.date, datetime.time)
VALUE_DELEGATE_TYPES: Tuple[Type[enum.Enum], ...] = (enum.Enum,)
HAS_TIME: Tuple[Type[Union[datetime.datetime, datetime.time]], ...] = (datetime.datetime, datetime.time)

def str_to_decimal(s: str, maxlen: int = DECIMAL_MAXLEN) -> Decimal:
    ...

def on_default(o: Any, *, sequences: Tuple[Type[Any], ...] = SEQUENCE_TYPES, maps: Tuple[Type[Any], ...] = MAPPING_TYPES, dates: Tuple[Type[Any], ...] = DATE_TYPES, value_delegate: Tuple[Type[Any], ...] = VALUE_DELEGATE_TYPES, has_time: Tuple[Type[Any], ...] = HAS_TIME, _isinstance: Callable = isinstance, _dict: Callable = dict, _str: Callable = str, _list: Callable = list, textual: Tuple[Type[Any], ...] = TEXTUAL_TYPES) -> Any:
    ...

class JSONEncoder(json.JSONEncoder):
    ...

def dumps(obj: Any, json_dumps: Callable = json.dumps, cls: Type[JSONEncoder] = JSONEncoder, **kwargs: Any) -> str:
    ...

def loads(s: str, json_loads: Callable = json.loads, **kwargs: Any) -> Any:
    ...
