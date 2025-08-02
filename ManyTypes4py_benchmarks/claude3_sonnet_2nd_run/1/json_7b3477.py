"""JSON utilities."""
import datetime
import enum
import typing
import uuid
from collections import Counter, deque
from decimal import Decimal
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union, Dict, Set, cast
__all__ = ['JSONEncoder', 'dumps', 'loads', 'str_to_decimal']
if typing.TYPE_CHECKING:
    import orjson
else:
    orjson = None
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
    _JSON_DEFAULT_KWARGS: Dict[str, bool] = {'use_decimal': False, 'namedtuple_as_object': False}
except ImportError:
    import json
    _JSON_DEFAULT_KWARGS: Dict[str, bool] = {}
DECIMAL_MAXLEN: int = 1000
SEQUENCE_TYPES: Tuple[Type[Any], ...] = (set, deque)
DateTypeTuple = Tuple[Union[Type[datetime.date], Type[datetime.time]], ...]
DatetimeTypeTuple = Tuple[Union[Type[datetime.time], Type[datetime.datetime]], ...]
MAPPING_TYPES: Tuple[Type[Any], ...] = (Counter,)
DATE_TYPES: Tuple[Type[Any], ...] = (datetime.date, datetime.time)
VALUE_DELEGATE_TYPES: Tuple[Type[Any], ...] = (enum.Enum,)
HAS_TIME: Tuple[Type[Any], ...] = (datetime.datetime, datetime.time)

def str_to_decimal(s: Optional[str], maxlen: int = DECIMAL_MAXLEN) -> Optional[Decimal]:
    """Convert string to :class:`~decimal.Decimal`.

    Args:
        s (str): Number to convert.
        maxlen (int): Max length of string.  Default is 100.

    Raises:
        ValueError: if length exceeds maximum length, or if value is not
            a valid number (e.g. Inf, NaN or sNaN).

    Returns:
        Decimal: Converted number.
    """
    if s is None:
        return None
    if len(s) > maxlen:
        raise ValueError(f'string of length {len(s)} is longer than limit ({maxlen})')
    v = Decimal(s)
    if not v.is_finite():
        raise ValueError(f'Illegal value in decimal: {s!r}')
    return v

def on_default(o: Any, *, sequences: Tuple[Type[Any], ...] = SEQUENCE_TYPES, maps: Tuple[Type[Any], ...] = MAPPING_TYPES, dates: Tuple[Type[Any], ...] = DATE_TYPES, value_delegate: Tuple[Type[Any], ...] = VALUE_DELEGATE_TYPES, has_time: Tuple[Type[Any], ...] = HAS_TIME, _isinstance: Callable = isinstance, _dict: Callable = dict, _str: Callable = str, _list: Callable = list, textual: Tuple[Type[Any], ...] = TEXTUAL_TYPES) -> Any:
    if _isinstance(o, textual):
        return _str(o)
    elif _isinstance(o, maps):
        return _dict(o)
    elif _isinstance(o, dates):
        if not _isinstance(o, has_time):
            o = datetime.datetime(o.year, o.month, o.day, 0, 0, 0, 0)
        r = o.isoformat()
        if r.endswith('+00:00'):
            r = r[:-6] + 'Z'
        return r
    elif isinstance(o, value_delegate):
        return o.value
    elif isinstance(o, sequences):
        return _list(o)
    else:
        to_json = getattr(o, '__json__', None)
        if to_json is not None:
            return to_json()
        raise TypeError(f'JSON cannot serialize {type(o).__name__!r}: {o!r}')

class JSONEncoder(json.JSONEncoder):
    """Faust customized :class:`json.JSONEncoder`.

    Our version supports additional types like :class:`~uuid.UUID`, and
    importantly includes microsecond information in datetimes.
    """

    def default(self, o: Any, *, callback: Callable = on_default) -> Any:
        """Try to convert non-built-in json type to json."""
        return callback(o)
if orjson is not None:

    def dumps(obj: Any, json_dumps: Callable = cast(Any, orjson).dumps, cls: Type[JSONEncoder] = JSONEncoder, **kwargs: Any) -> bytes:
        """Serialize to json."""
        return json_dumps(obj, default=on_default, option=cast(Any, orjson).OPT_NAIVE_UTC)

    def loads(s: Union[str, bytes], json_loads: Callable = cast(Any, orjson).loads, **kwargs: Any) -> Any:
        """Deserialize json string."""
        return json_loads(s)
else:

    def dumps(obj: Any, json_dumps: Callable = json.dumps, cls: Type[JSONEncoder] = JSONEncoder, **kwargs: Any) -> str:
        """Serialize to json.  See :func:`json.dumps`."""
        return json_dumps(obj, cls=cls, **dict(_JSON_DEFAULT_KWARGS, **kwargs))

    def loads(s: Union[str, bytes], json_loads: Callable = json.loads, **kwargs: Any) -> Any:
        """Deserialize json string.  See :func:`json.loads`."""
        return json_loads(s, **kwargs)
