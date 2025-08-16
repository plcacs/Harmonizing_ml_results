from decimal import Decimal
import datetime
import enum
import typing
import uuid
from collections import Counter, deque
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union

__all__: List[str] = ['JSONEncoder', 'dumps', 'loads', 'str_to_decimal']
if typing.TYPE_CHECKING:
    import orjson
else:
    orjson: Any = None
DEFAULT_TEXTUAL_TYPES: List[Type[Any]] = [Decimal, uuid.UUID, bytes]
T = TypeVar('T')
TypeTuple = Tuple[Type[T], ...]
IsInstanceArg = Union[Type[Any], TypeTuple[Any]]
try:
    from django.utils.functional import Promise
    DJANGO_TEXTUAL_TYPES: List[Type[Any]] = [Promise]
except ImportError:
    DJANGO_TEXTUAL_TYPES: List[Type[Any]] = []
TEXTUAL_TYPES: Tuple[Type[Any], ...] = tuple(DEFAULT_TEXTUAL_TYPES + DJANGO_TEXTUAL_TYPES)
try:
    import simplejson as json
    _JSON_DEFAULT_KWARGS: Mapping[str, Union[bool, str]] = {'use_decimal': False, 'namedtuple_as_object': False}
except ImportError:
    import json
    _JSON_DEFAULT_KWARGS: Mapping[str, Union[bool, str]] = {}
DECIMAL_MAXLEN: int = 1000
SEQUENCE_TYPES: Tuple[Type[Any], ...] = (set, deque)
DateTypeTuple: Tuple[Union[Type[datetime.date], Type[datetime.time]], ...] = (datetime.date, datetime.time)
DatetimeTypeTuple: Tuple[Union[Type[datetime.time], Type[datetime.datetime]], ...] = (datetime.time, datetime.datetime)
MAPPING_TYPES: Tuple[Type[Any], ...] = (Counter,)
DATE_TYPES: Tuple[Type[Any], ...] = (datetime.date, datetime.time)
VALUE_DELEGATE_TYPES: Tuple[Type[Any], ...] = (enum.Enum,)
HAS_TIME: Tuple[Type[Any], ...] = (datetime.datetime, datetime.time)

def str_to_decimal(s: str, maxlen: int = DECIMAL_MAXLEN) -> Decimal:
    ...

def on_default(o: Any, *, sequences: Tuple[Type[Any], ...] = SEQUENCE_TYPES, maps: Tuple[Type[Any], ...] = MAPPING_TYPES, dates: Tuple[Type[Any], ...] = DATE_TYPES, value_delegate: Tuple[Type[Any], ...] = VALUE_DELEGATE_TYPES, has_time: Tuple[Type[Any], ...] = HAS_TIME, _isinstance: Callable[[Any, IsInstanceArg], bool] = isinstance, _dict: Callable[[], dict] = dict, _str: Callable[[Any], str] = str, _list: Callable[[Any], list] = list, textual: Tuple[Type[Any], ...] = TEXTUAL_TYPES) -> Union[str, dict, str]:
    ...

class JSONEncoder(json.JSONEncoder):
    ...

if orjson is not None:
    def dumps(obj: Any, json_dumps: Callable[[Any], bytes] = orjson.dumps, cls: Type[JSONEncoder] = JSONEncoder, **kwargs: Any) -> bytes:
        ...

    def loads(s: bytes, json_loads: Callable[[bytes], Any] = orjson.loads, **kwargs: Any) -> Any:
        ...
else:
    def dumps(obj: Any, json_dumps: Callable[[Any], str] = json.dumps, cls: Type[JSONEncoder] = JSONEncoder, **kwargs: Any) -> str:
        ...

    def loads(s: str, json_loads: Callable[[str], Any] = json.loads, **kwargs: Any) -> Any:
        ...
