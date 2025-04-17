"""JSON utilities."""
import datetime
import enum
import typing
import uuid

from collections import Counter, deque
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

__all__ = [
    'JSONEncoder',
    'dumps',
    'loads',
    'str_to_decimal',
]

if typing.TYPE_CHECKING:
    import orjson
else:  # pragma: no cover
    orjson = None  # noqa

DEFAULT_TEXTUAL_TYPES: List[Type] = [Decimal, uuid.UUID, bytes]

T = TypeVar('T')

TypeTuple = Tuple[Type[T], ...]
IsInstanceArg = Union[Type[Any], TypeTuple[Any]]

try:  # pragma: no cover
    from django.utils.functional import Promise
    DJANGO_TEXTUAL_TYPES: List[Type] = [Promise]
except ImportError:  # pragma: no cover
    DJANGO_TEXTUAL_TYPES = []

TEXTUAL_TYPES: TypeTuple[Any] = tuple(
    DEFAULT_TEXTUAL_TYPES + DJANGO_TEXTUAL_TYPES)

try:  # pragma: no cover
    import simplejson as json

    _JSON_DEFAULT_KWARGS: dict = {
        'use_decimal': False,
        'namedtuple_as_object': False,
    }

except ImportError:  # pragma: no cover
    import json  # type: ignore
    _JSON_DEFAULT_KWARGS: dict = {}

DECIMAL_MAXLEN: int = 1000

SEQUENCE_TYPES: TypeTuple[Iterable] = (set, deque)

DateTypeTuple = Tuple[Union[Type[datetime.date], Type[datetime.time]], ...]
DatetimeTypeTuple = Tuple[
    Union[
        Type[datetime.time],
        Type[datetime.datetime],
    ],
    ...,
]

MAPPING_TYPES: TypeTuple[Mapping] = (Counter,)

DATE_TYPES: DateTypeTuple = (datetime.date, datetime.time)

VALUE_DELEGATE_TYPES: TypeTuple[enum.Enum] = (enum.Enum,)

HAS_TIME: DatetimeTypeTuple = (datetime.datetime, datetime.time)


def str_to_decimal(s: str, maxlen: int = DECIMAL_MAXLEN) -> Optional[Decimal]:
    if s is None:
        return None
    if len(s) > maxlen:
        raise ValueError(
            f'string of length {len(s)} is longer than limit ({maxlen})')
    v = Decimal(s)
    if not v.is_finite():
        raise ValueError(f'Illegal value in decimal: {s!r}')
    return v


def on_default(o: Any,
               *,
               sequences: TypeTuple[Iterable] = SEQUENCE_TYPES,
               maps: TypeTuple[Mapping] = MAPPING_TYPES,
               dates: DateTypeTuple = DATE_TYPES,
               value_delegate: TypeTuple[enum.Enum] = VALUE_DELEGATE_TYPES,
               has_time: DatetimeTypeTuple = HAS_TIME,
               _isinstance: Callable[[Any, IsInstanceArg], bool] = isinstance,
               _dict: Callable = dict,
               _str: Callable[[Any], str] = str,
               _list: Callable = list,
               textual: TypeTuple[Any] = TEXTUAL_TYPES) -> Any:
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
        raise TypeError(
            f'JSON cannot serialize {type(o).__name__!r}: {o!r}')


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any, *,
                callback: Callable[[Any], Any] = on_default) -> Any:
        return callback(o)


if orjson is not None:  # pragma: no cover

    def dumps(obj: Any,
              json_dumps: Callable = orjson.dumps,
              cls: Type[JSONEncoder] = JSONEncoder,
              **kwargs: Any) -> str:
        return json_dumps(
            obj,
            default=on_default,
            option=orjson.OPT_NAIVE_UTC,
        )

    def loads(s: str,
              json_loads: Callable = orjson.loads,
              **kwargs: Any) -> Any:
        return json_loads(s)
else:

    def dumps(obj: Any,
              json_dumps: Callable = json.dumps,
              cls: Type[JSONEncoder] = JSONEncoder,
              **kwargs: Any) -> str:
        return json_dumps(obj, cls=cls, **dict(_JSON_DEFAULT_KWARGS, **kwargs))

    def loads(s: str,
              json_loads: Callable = json.loads,
              **kwargs: Any) -> Any:
        return json_loads(s, **kwargs)
