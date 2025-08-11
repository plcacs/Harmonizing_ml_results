"""JSON utilities."""
import datetime
import enum
import typing
import uuid
from collections import Counter, deque
from decimal import Decimal
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union
__all__ = ['JSONEncoder', 'dumps', 'loads', 'str_to_decimal']
if typing.TYPE_CHECKING:
    import orjson
else:
    orjson = None
DEFAULT_TEXTUAL_TYPES = [Decimal, uuid.UUID, bytes]
T = TypeVar('T')
TypeTuple = Tuple[Type[T], ...]
IsInstanceArg = Union[Type[Any], TypeTuple[Any]]
try:
    from django.utils.functional import Promise
    DJANGO_TEXTUAL_TYPES = [Promise]
except ImportError:
    DJANGO_TEXTUAL_TYPES = []
TEXTUAL_TYPES = tuple(DEFAULT_TEXTUAL_TYPES + DJANGO_TEXTUAL_TYPES)
try:
    import simplejson as json
    _JSON_DEFAULT_KWARGS = {'use_decimal': False, 'namedtuple_as_object': False}
except ImportError:
    import json
    _JSON_DEFAULT_KWARGS = {}
DECIMAL_MAXLEN = 1000
SEQUENCE_TYPES = (set, deque)
DateTypeTuple = Tuple[Union[Type[datetime.date], Type[datetime.time]], ...]
DatetimeTypeTuple = Tuple[Union[Type[datetime.time], Type[datetime.datetime]], ...]
MAPPING_TYPES = (Counter,)
DATE_TYPES = (datetime.date, datetime.time)
VALUE_DELEGATE_TYPES = (enum.Enum,)
HAS_TIME = (datetime.datetime, datetime.time)

def str_to_decimal(s: Union[str, list[str]], maxlen: int=DECIMAL_MAXLEN) -> Union[None, Decimal]:
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

def on_default(o: Union[datetime.datetime, None, datetime.date], *, sequences: Union[tuple[typing.Type], set[str], list[str]]=SEQUENCE_TYPES, maps: Union[dict[str, typing.Any], typing.Type]=MAPPING_TYPES, dates: Union[str, tuple[str], int, None]=DATE_TYPES, value_delegate: Union[tuple[tuple[typing.Any]], tuple[typing.Type], typing.Type]=VALUE_DELEGATE_TYPES, has_time: Union[bool, str]=HAS_TIME, _isinstance: Any=isinstance, _dict: dict=dict, _str: str=str, _list: list=list, textual: Union[list[str], str]=TEXTUAL_TYPES) -> str:
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

    def default(self, o: Union[typing.Callable, bool], *, callback: Any=on_default) -> Union[str, typing.Callable, hashstore.utils.args.CommandArgs]:
        """Try to convert non-built-in json type to json."""
        return callback(o)
if orjson is not None:

    def dumps(obj: Union[typing.Callable, typing.Type, T], json_dumps: Any=orjson.dumps, cls: JSONEncoder=JSONEncoder, **kwargs):
        """Serialize to json."""
        return json_dumps(obj, default=on_default, option=orjson.OPT_NAIVE_UTC)

    def loads(s: Union[bool, str, bytes], json_loads: Any=orjson.loads, **kwargs) -> Union[dict, dict[str, typing.Any], list[typing.Any]]:
        """Deserialize json string."""
        return json_loads(s)
else:

    def dumps(obj: Union[typing.Callable, typing.Type, T], json_dumps: Any=json.dumps, cls: JSONEncoder=JSONEncoder, **kwargs):
        """Serialize to json.  See :func:`json.dumps`."""
        return json_dumps(obj, cls=cls, **dict(_JSON_DEFAULT_KWARGS, **kwargs))

    def loads(s: Union[bool, str, bytes], json_loads: Any=json.loads, **kwargs) -> Union[dict, dict[str, typing.Any], list[typing.Any]]:
        """Deserialize json string.  See :func:`json.loads`."""
        return json_loads(s, **kwargs)